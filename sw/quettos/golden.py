"""Bit-exact integer golden model of the Quettos Core decode and prefill programs.

Runs a :class:`quettos.quantize.QuantModel` through the integer operators the
hardware executes, every one a :mod:`quettos.numerics` primitive, with the
GEMVs evaluated as exact integer products through float64 BLAS, and produces
the logits, argmax tokens and KV cache the RTL must reproduce bit for bit.
Entry points: :func:`forward_tokens` (teacher-forced, every position at once),
:func:`step` (one descriptor program at one position), :func:`generate` (the
prefill/decode loop) and :func:`expected_tokens`.  Requant constants come from
:mod:`quettos.program`.  Dataflow: ``docs/ARCHITECTURE.md``; formats and shift
rules: ``docs/NUMERICS.md``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quettos import numerics, program
from quettos.calibrate import MODELS_OUT_DIR
from quettos.model import REPO_ROOT, ModelSpec
from quettos.numerics import SFLOAT_ONE, Stats
from quettos.program import ProgramConstants
from quettos.quantize import QuantLinear, QuantModel
from quettos.tokenizer_io import detokenize, prompt_tokens, token_bytes

HEAD_DIM = 64
EXPECTED_PROMPT_FILES: tuple[str, ...] = (
    "prompts/chat_short.json",
    "prompts/tool_call_weather.json",
)
EXACT_BITS = 53  # float64 integer exactness bound for every partial sum
W_ABS_MAX = 128  # int8 magnitude bound used in the exactness check
GEMV_CHUNK_ELEMS = 1 << 22  # accumulator elements per requant block (memory bound)
ACC_LIMIT = 1 << (program.ACC_W - 1)  # the hardware accumulator never wraps

TraceFn = Callable[[str, "int | None", np.ndarray], None]

# Every op output the trace hook receives, in program order.  ``layer`` is the
# layer index for the per-layer ops and ``None`` for embed and the LM head.
# ``<op>.scale`` carries the sfloat scales of a quantizing op as ``[..., 2]``
# ``(m, e)`` arrays; ``softmax.sreg`` the per-(token, head) ``SREG_out``.
TRACE_OPS_EMBED: tuple[str, ...] = ("embed",)
TRACE_OPS_LAYER: tuple[str, ...] = (
    "rmsnorm_in",
    "quant_in",
    "quant_in.scale",
    "gemv_qkv",
    "rope_q",
    "rope_k",
    "quant_q",
    "quant_q.scale",
    "subc_k",
    "quant_k",
    "quant_k.scale",
    "quant_v",
    "quant_v.scale",
    "gemv_scores",
    "softmax",
    "softmax.sreg",
    "gemv_pv",
    "quant_ctx",
    "quant_ctx.scale",
    "gemv_o",
    "rmsnorm_post",
    "quant_post",
    "quant_post.scale",
    "gemv_gu",
    "silu_mul",
    "quant_h",
    "quant_h.scale",
    "gemv_down",
)
TRACE_OPS_HEAD: tuple[str, ...] = (
    "rmsnorm_final",
    "quant_final",
    "quant_final.scale",
    "gemv_lm_head",
    "argmax",
)


# --------------------------------------------------------------------------- containers


@dataclass
class KVCache:
    """Per-layer, per-KV-head int8 K/V rows with their sfloat scales (``m = 0``: unwritten)."""

    max_ctx: int
    k: np.ndarray  # int8 [L, KV, max_ctx, D]
    v: np.ndarray  # int8 [L, KV, max_ctx, D]
    k_m: np.ndarray  # int64 [L, KV, max_ctx]
    k_e: np.ndarray
    v_m: np.ndarray
    v_e: np.ndarray
    length: int = 0  # positions 0 .. length-1 hold written K/V


@dataclass
class GoldenOutput:
    """Teacher-forced result: int32 logits ``[T, V]`` (FRAC 16), argmax ``[T]``, the KV cache."""

    logits: np.ndarray
    argmax: np.ndarray
    cache: KVCache


def new_cache(model: QuantModel, max_ctx: int = program.MAX_CTX) -> KVCache:
    """An empty cache for ``max_ctx`` positions (bounded by the checked-in RoPE table)."""
    if max_ctx < 1:
        raise ValueError("new_cache: max_ctx must be positive")
    shape = (model.n_layers, model.kv_heads, max_ctx)
    return KVCache(
        max_ctx=max_ctx,
        k=np.zeros((*shape, model.head_dim), dtype=np.int8),
        v=np.zeros((*shape, model.head_dim), dtype=np.int8),
        k_m=np.zeros(shape, dtype=np.int64),
        k_e=np.zeros(shape, dtype=np.int64),
        v_m=np.zeros(shape, dtype=np.int64),
        v_e=np.zeros(shape, dtype=np.int64),
    )


# --------------------------------------------------------------------------- run context


@dataclass
class _Run:
    model: QuantModel
    prog: ProgramConstants
    tables: numerics.Tables
    rope: np.ndarray
    a_bits: int
    stats: Stats | None
    trace: TraceFn | None

    def emit(self, name: str, layer: int | None, value: np.ndarray) -> None:
        if self.trace is not None:
            self.trace(name, layer, value)


def _prepare(
    model: QuantModel,
    a_bits: int,
    max_ctx: int,
    stats: Stats | None,
    trace: TraceFn | None,
    prog: ProgramConstants | None,
) -> _Run:
    """The run context: the program constants are those of the compiled program
    (``program.MAX_CTX``), whatever the cache size; ``max_ctx`` only bounds the
    cache and the RoPE rows."""
    if model.head_dim != HEAD_DIM:
        raise ValueError(f"golden: head_dim {model.head_dim} is not {HEAD_DIM}")
    tables = numerics.load_tables()
    if prog is None:
        prog = program.build(model, a_bits=a_bits, tables=tables)
    elif prog.a_bits != a_bits or prog.model != model.name or prog.layers != model.n_layers:
        raise ValueError("golden: program constants were built for another model, depth or width")
    if max_ctx > prog.max_ctx:
        raise ValueError(f"golden: cache of {max_ctx} positions exceeds MAX_CTX {prog.max_ctx}")
    rope = numerics.load_rope_table(model.rope_theta)
    if max_ctx > rope.shape[0]:
        raise ValueError(f"golden: max_ctx {max_ctx} exceeds the RoPE table ({rope.shape[0]})")
    return _Run(model, prog, tables, rope, a_bits, stats, trace)


# --------------------------------------------------------------------------- exact GEMV


def exact_matmul(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``a @ w.T`` as exact int64 for integer ``a[T, K]`` and int8 ``w[N, K]``.

    Evaluated in float64 BLAS; exact because every partial sum is an integer
    bounded by ``K * max|a| * 128 < 2**53`` (asserted), so no rounding can
    occur in any summation order.
    """
    k = a.shape[1]
    bound = k * numerics.absmax(a) * W_ABS_MAX
    if bound >= 1 << EXACT_BITS:
        raise ValueError(f"exact_matmul: partial-sum bound 2^{bound.bit_length()} not exact")
    acc = a.astype(np.float64) @ w.astype(np.float64).T
    return acc.astype(np.int64)


def _gemv_requant(
    run: _Run,
    a: np.ndarray,
    sx_m: np.ndarray,
    sx_e: np.ndarray,
    lin: QuantLinear,
    g: program.GemvConstants,
    *,
    old: np.ndarray | None = None,
    out_dtype=np.int64,
) -> np.ndarray:
    """One weight GEMV over ``a[T, K]``: exact products, then :func:`numerics.requant_rows`.

    Output channels are processed in blocks of at most ``GEMV_CHUNK_ELEMS``
    accumulator elements and ``GEMV_CHUNK_ELEMS`` weight elements, so neither
    the float64 weight copy nor the int64 accumulators exceed 32 MB; every
    block is exact, so the result is identical to one block.
    """
    n, k = lin.q.shape
    t = a.shape[0]
    if a.shape[1] != k:
        raise ValueError(f"{g.name}: activation length {a.shape[1]} != K {k}")
    out = np.empty((t, n), dtype=out_dtype)
    cols = max(1, min(GEMV_CHUNK_ELEMS // max(t, 1), GEMV_CHUNK_ELEMS // k))
    for n0 in range(0, n, cols):
        n1 = min(n, n0 + cols)
        acc = exact_matmul(a, lin.q[n0:n1])
        if numerics.absmax(acc) >= ACC_LIMIT:
            raise ValueError(f"{g.name}: accumulator exceeds {program.ACC_W} bits")
        out[:, n0:n1] = numerics.requant_rows(
            acc,
            lin.scale_m[n0:n1],
            lin.scale_e[n0:n1],
            sx_m,
            sx_e,
            g.s1,
            g.sbias,
            bias_q=lin.bias_q[n0:n1],
            old=None if old is None else old[:, n0:n1],
            stats=run.stats,
        )
    return out


# --------------------------------------------------------------------------- per-token ops


def _rmsnorm_rows(run: _Run, x: np.ndarray, gamma_q: np.ndarray, gamma_e: int, eps_c: int):
    m = run.model
    gq = gamma_q.astype(np.int64)
    return np.stack(
        [
            numerics.rmsnorm(x[t], gq, gamma_e, eps_c, m.sqrt_d, m.frac["X"], run.tables, run.stats)
            for t in range(x.shape[0])
        ]
    )


def _quant_rows(run: _Run, x: np.ndarray, width: int, frac_in: int):
    """Per-token VQUANT: int values ``[T, D]`` and the scale arrays ``m[T]``, ``e[T]``."""
    q = np.empty_like(x)
    sm = np.empty(x.shape[0], dtype=np.int64)
    se = np.empty(x.shape[0], dtype=np.int64)
    for t in range(x.shape[0]):
        q[t], s = numerics.quant(x[t], width, frac_in, run.tables, stats=run.stats)
        sm[t], se[t] = s.m, s.e
    return q, sm, se


def _quant_groups_rows(run: _Run, x: np.ndarray, width: int, frac_in: int, scale_mul=None):
    """Per-token, per-head VQUANT: values ``[T, G*64]`` and scales ``m[T, G]``, ``e[T, G]``."""
    t_rows, d = x.shape
    groups = d // HEAD_DIM
    q = np.empty_like(x)
    sm = np.empty((t_rows, groups), dtype=np.int64)
    se = np.empty((t_rows, groups), dtype=np.int64)
    for t in range(t_rows):
        q[t], scales = numerics.quant_groups(
            x[t], HEAD_DIM, width, frac_in, run.tables, scale_mul=scale_mul, stats=run.stats
        )
        sm[t] = [s.m for s in scales]
        se[t] = [s.e for s in scales]
    return q, sm, se


def _rope_rows(run: _Run, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            numerics.rope(x[t], run.rope[p, 0], run.rope[p, 1], HEAD_DIM, run.stats)
            for t, p in enumerate(positions)
        ]
    )


def _scales(m: np.ndarray, e: np.ndarray) -> np.ndarray:
    return np.stack([m, e], axis=-1)


# --------------------------------------------------------------------------- the program


def _attention(run: _Run, cache: KVCache, layer: int, positions: np.ndarray, q_i16, sq_m, sq_e):
    """Scores GEMV, softmax and PV GEMV for every query head; returns ``ctx[T, H*64]``."""
    m = run.model
    heads, n_rep = m.heads, m.heads // m.kv_heads
    t_rows = q_i16.shape[0]
    n_keys = int(positions[-1]) + 1
    keys = np.arange(n_keys, dtype=np.int64)
    valid = keys[None, :] <= positions[:, None]
    lengths = positions + 1
    g_sc, g_pv = run.prog["scores"], run.prog["pv"]
    one_m = np.full(HEAD_DIM, SFLOAT_ONE.m, dtype=np.int64)
    one_e = np.full(HEAD_DIM, SFLOAT_ONE.e, dtype=np.int64)
    ctx = np.empty((t_rows, heads * HEAD_DIM), dtype=np.int64)
    tracing = run.trace is not None
    if tracing:  # per-head score and weight blocks exist only for the trace hook
        scores_all = np.empty((t_rows, heads, n_keys), dtype=np.int64)
        w_all = np.empty((t_rows, heads, n_keys), dtype=np.int64)
        sreg_all = np.empty((t_rows, heads, 2), dtype=np.int64)
    for h in range(heads):
        g = h // n_rep
        sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        acc = exact_matmul(q_i16[:, sl], cache.k[layer, g, :n_keys])
        scores = numerics.requant_rows(
            acc,
            cache.k_m[layer, g, :n_keys],
            cache.k_e[layer, g, :n_keys],
            sq_m[:, h],
            sq_e[:, h],
            g_sc.s1,
            g_sc.sbias,
            valid=valid,
            stats=run.stats,
        )
        w, sreg_m, sreg_e = numerics.softmax_rows(
            scores,
            lengths,
            m.frac["S"],
            cache.v_m[layer, g, :n_keys],
            cache.v_e[layer, g, :n_keys],
            run.tables,
            stats=run.stats,
        )
        acc = exact_matmul(w, cache.v[layer, g, :n_keys].T.copy())
        if numerics.absmax(acc) >= ACC_LIMIT:
            raise ValueError("pv: accumulator exceeds the hardware width")
        ctx[:, sl] = numerics.requant_rows(
            acc, one_m, one_e, sreg_m, sreg_e, g_pv.s1, g_pv.sbias, stats=run.stats
        )
        if tracing:
            scores_all[:, h] = scores
            w_all[:, h] = w
            sreg_all[:, h, 0], sreg_all[:, h, 1] = sreg_m, sreg_e
    if tracing:
        run.emit("gemv_scores", layer, scores_all)
        run.emit("softmax", layer, w_all)
        run.emit("softmax.sreg", layer, sreg_all)
    run.emit("gemv_pv", layer, ctx)
    return ctx


def _layer(run: _Run, cache: KVCache, layer: int, positions: np.ndarray, x: np.ndarray):
    m, prog, frac = run.model, run.prog, run.model.frac
    lay = m.layers[layer]
    hd, kvd = m.heads * HEAD_DIM, m.kv_heads * HEAD_DIM

    xn = _rmsnorm_rows(run, x, lay.norm_in.gamma_q, lay.norm_in.gamma_e, m.eps_c["input"])
    run.emit("rmsnorm_in", layer, xn)
    a, sm, se = _quant_rows(run, xn, run.a_bits, frac["X"])
    run.emit("quant_in", layer, a)
    run.emit("quant_in.scale", layer, _scales(sm, se))
    qkv = _gemv_requant(run, a, sm, se, lay.wqkv, prog["qkv"])
    run.emit("gemv_qkv", layer, qkv)

    q = _rope_rows(run, qkv[:, :hd], positions)
    k = _rope_rows(run, qkv[:, hd : hd + kvd], positions)
    v = qkv[:, hd + kvd :]
    run.emit("rope_q", layer, q)
    run.emit("rope_k", layer, k)
    q_i16, sq_m, sq_e = _quant_groups_rows(run, q, 16, frac["QKV"], scale_mul=m.log2e_over_8)
    run.emit("quant_q", layer, q_i16)
    run.emit("quant_q.scale", layer, _scales(sq_m, sq_e))
    center = m.k_center[layer].astype(np.int64).reshape(kvd)
    k_c = numerics.subc(k, center[None, :], run.stats)
    run.emit("subc_k", layer, k_c)
    k_i8, sk_m, sk_e = _quant_groups_rows(run, k_c, 8, frac["QKV"])
    run.emit("quant_k", layer, k_i8)
    run.emit("quant_k.scale", layer, _scales(sk_m, sk_e))
    v_i8, sv_m, sv_e = _quant_groups_rows(run, v, 8, frac["QKV"])
    run.emit("quant_v", layer, v_i8)
    run.emit("quant_v.scale", layer, _scales(sv_m, sv_e))

    # KVWRITE: rows and meta at each token's position
    kv_view = (m.kv_heads, HEAD_DIM)
    for t, p in enumerate(positions):
        cache.k[layer, :, p] = k_i8[t].reshape(kv_view).astype(np.int8)
        cache.v[layer, :, p] = v_i8[t].reshape(kv_view).astype(np.int8)
        cache.k_m[layer, :, p], cache.k_e[layer, :, p] = sk_m[t], sk_e[t]
        cache.v_m[layer, :, p], cache.v_e[layer, :, p] = sv_m[t], sv_e[t]

    ctx = _attention(run, cache, layer, positions, q_i16, sq_m, sq_e)
    c, sm, se = _quant_rows(run, ctx, run.a_bits, frac["CTX"])
    run.emit("quant_ctx", layer, c)
    run.emit("quant_ctx.scale", layer, _scales(sm, se))
    x = _gemv_requant(run, c, sm, se, lay.wo, prog["o"], old=x)
    run.emit("gemv_o", layer, x)

    xn = _rmsnorm_rows(run, x, lay.norm_post.gamma_q, lay.norm_post.gamma_e, m.eps_c["post"])
    run.emit("rmsnorm_post", layer, xn)
    a, sm, se = _quant_rows(run, xn, run.a_bits, frac["X"])
    run.emit("quant_post", layer, a)
    run.emit("quant_post.scale", layer, _scales(sm, se))
    gu = _gemv_requant(run, a, sm, se, lay.wgu, prog["gu"])
    run.emit("gemv_gu", layer, gu)
    inter = m.intermediate
    hh = numerics.silu_mul(
        gu[:, :inter], gu[:, inter:], frac["GU"], frac["H"], run.tables, run.stats
    )
    run.emit("silu_mul", layer, hh)
    hq, sm, se = _quant_rows(run, hh, run.a_bits, frac["H"])
    run.emit("quant_h", layer, hq)
    run.emit("quant_h.scale", layer, _scales(sm, se))
    x = _gemv_requant(run, hq, sm, se, lay.wdown, prog["down"], old=x)
    run.emit("gemv_down", layer, x)
    return x


def _run_block(
    run: _Run, cache: KVCache, toks: np.ndarray, positions: np.ndarray, lm_head: bool
) -> np.ndarray | None:
    """Execute the program for consecutive positions; returns int32 logits or ``None``."""
    m, prog, frac = run.model, run.prog, run.model.frac
    g_embed = prog["embed"]
    x = np.stack(
        [
            numerics.embed_dequant(
                m.embed.q[tok],
                numerics.SFloat(int(m.embed.scale_m[tok]), int(m.embed.scale_e[tok])),
                frac["X"],
                s1=g_embed.s1,
                stats=run.stats,
            )
            for tok in toks
        ]
    )
    run.emit("embed", None, x)
    for layer in range(m.n_layers):
        x = _layer(run, cache, layer, positions, x)
    if not lm_head:
        return None
    xn = _rmsnorm_rows(run, x, m.norm_final.gamma_q, m.norm_final.gamma_e, m.eps_c["final"])
    run.emit("rmsnorm_final", None, xn)
    a, sm, se = _quant_rows(run, xn, run.a_bits, frac["X"])
    run.emit("quant_final", None, a)
    run.emit("quant_final.scale", None, _scales(sm, se))
    logits = _gemv_requant(run, a, sm, se, m.embed, prog["lm_head"], out_dtype=np.int32)
    run.emit("gemv_lm_head", None, logits)
    return logits


def _argmax_rows(logits: np.ndarray) -> np.ndarray:
    return np.array([numerics.argmax(row) for row in logits], dtype=np.int64)


def _check_ids(model: QuantModel, ids: np.ndarray) -> None:
    if ids.size and (np.any(ids < 0) or np.any(ids >= model.vocab)):
        raise ValueError("golden: token id out of range")


# --------------------------------------------------------------------------- public API


def forward_tokens(
    model: QuantModel,
    ids: Sequence[int],
    *,
    a_bits: int = 16,
    stats: Stats | None = None,
    trace: TraceFn | None = None,
    max_ctx: int = program.MAX_CTX,
    prog: ProgramConstants | None = None,
) -> GoldenOutput:
    """Teacher-forced forward over all ``T`` positions of ``ids`` with the LM head.

    Bit-identical to running :func:`step` at positions ``0 .. T-1`` with
    ``lm_head=True`` at every position: the per-token ops are independent and
    attention at position ``t`` sees the same causal window either way.
    ``a_bits`` (16 or 8) is the width of the activations fed to the weight
    GEMVs; q stays int16 and the KV cache int8.  ``max_ctx`` sizes the cache
    (at most ``program.MAX_CTX``); the requant constants are those of the
    compiled program at ``program.MAX_CTX`` for every cache size, so ``prog``
    (if given) must come from :func:`program.build` with its default ``max_ctx``.
    """
    toks = np.asarray(list(ids), dtype=np.int64)
    if toks.size < 1 or toks.size > max_ctx:
        raise ValueError(f"forward_tokens: sequence length {toks.size} not in [1, {max_ctx}]")
    _check_ids(model, toks)
    run = _prepare(model, a_bits, max_ctx, stats, trace, prog)
    cache = new_cache(model, max_ctx)
    positions = np.arange(toks.size, dtype=np.int64)
    logits = _run_block(run, cache, toks, positions, lm_head=True)
    cache.length = int(toks.size)
    assert logits is not None
    am = _argmax_rows(logits)
    run.emit("argmax", None, am)
    return GoldenOutput(logits=logits, argmax=am, cache=cache)


def step(
    model: QuantModel,
    cache: KVCache,
    tok: int,
    pos: int,
    *,
    lm_head: bool = True,
    a_bits: int = 16,
    stats: Stats | None = None,
    trace: TraceFn | None = None,
    prog: ProgramConstants | None = None,
) -> int | None:
    """Execute one program (``TOK = tok``, ``POS = pos``) and append K/V at ``pos``.

    ``lm_head=False`` runs ``prefill.prog`` (no final norm, no LM head, returns
    ``None``); ``lm_head=True`` runs ``decode.prog`` and returns the argmax
    token.  ``pos`` may not skip positions (``pos <= cache.length``); a step at
    an earlier position overwrites that K/V row and truncates the cache to it.
    """
    if pos < 0 or pos >= cache.max_ctx:
        raise ValueError(f"step: pos {pos} outside [0, {cache.max_ctx})")
    if pos > cache.length:
        raise ValueError(f"step: pos {pos} skips positions (cache holds {cache.length})")
    toks = np.asarray([tok], dtype=np.int64)
    _check_ids(model, toks)
    run = _prepare(model, a_bits, cache.max_ctx, stats, trace, prog)
    logits = _run_block(run, cache, toks, np.asarray([pos], dtype=np.int64), lm_head)
    cache.length = pos + 1
    if logits is None:
        return None
    am = _argmax_rows(logits)
    run.emit("argmax", None, am)
    return int(am[0])


def generate(
    model: QuantModel,
    prompt_ids: Sequence[int],
    max_new: int,
    *,
    eos_ids: Sequence[int] = (),
    a_bits: int = 16,
    stats: Stats | None = None,
    max_ctx: int = program.MAX_CTX,
    on_token: Callable[[int, int, int], None] | None = None,
) -> list[int]:
    """Greedy continuation of ``prompt_ids`` with the shared prefill/decode loop.

    ::

        for i in 0..P-2:  prefill.prog(TOK=prompt[i], POS=i)
        decode.prog(TOK=prompt[P-1], POS=P-1)            -> gen[0]
        for j >= 1:       decode.prog(TOK=gen[j-1], POS=P-1+j) -> gen[j]

    stopping after ``max_new`` tokens or once a generated id is in ``eos_ids``
    (that id is the last element).  ``on_token(j, pos, id)`` is called after
    every decode step.
    """
    prompt = [int(t) for t in prompt_ids]
    p_len = len(prompt)
    if p_len < 1:
        raise ValueError("generate: empty prompt")
    if max_new < 0 or p_len + max_new - 1 > max_ctx:
        raise ValueError("generate: prompt plus max_new exceeds max_ctx")
    prog = program.build(model, a_bits=a_bits)
    cache = new_cache(model, max_ctx)
    kw = {"a_bits": a_bits, "stats": stats, "prog": prog}
    for i in range(p_len - 1):
        step(model, cache, prompt[i], i, lm_head=False, **kw)
    gen: list[int] = []
    tok = prompt[p_len - 1]
    stop = set(int(e) for e in eos_ids)
    for j in range(max_new):
        pos = p_len - 1 + j
        nxt = step(model, cache, tok, pos, lm_head=True, **kw)
        assert nxt is not None
        gen.append(nxt)
        if on_token is not None:
            on_token(j, pos, nxt)
        if nxt in stop:
            break
        tok = nxt
    return gen


# --------------------------------------------------------------------------- expected tokens


def ids_sha256(ids: Sequence[int]) -> str:
    """SHA-256 of the compact JSON list of ids (``[1,2,3]``, no spaces)."""
    return hashlib.sha256(
        json.dumps([int(i) for i in ids], separators=(",", ":")).encode()
    ).hexdigest()


def prompt_key(path: Path | str) -> str:
    """The repo-relative POSIX path of a prompt file (the key of ``expected_tokens.json``)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def expected_tokens(
    model: QuantModel,
    spec: ModelSpec,
    prompt_files: Sequence[Path | str],
    *,
    max_new: int = 20,
    a_bits: int = 16,
    stats: Stats | None = None,
    on_token: Callable[[str, int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Greedy golden continuation of each prompt file, with text and id hashes.

    ``{"prompts": {file: {"prompt_tokens", "generated_ids", "sha256", "text"}}}``
    plus the model identity, ``a_bits``, ``max_new`` and the calibration hash the
    model was quantized with.  ``on_token(file, j, pos, id)`` reports progress.
    """
    table = token_bytes(spec)
    prompts: dict[str, Any] = {}
    for f in prompt_files:
        ids = prompt_tokens(spec, f)
        key = prompt_key(f)

        def report(j: int, pos: int, tok: int, key: str = key) -> None:
            if on_token is not None:
                on_token(key, j, pos, tok)

        gen = generate(
            model, ids, max_new, eos_ids=spec.eos_ids, a_bits=a_bits, stats=stats, on_token=report
        )
        prompts[key] = {
            "prompt_tokens": len(ids),
            "generated_ids": gen,
            "sha256": ids_sha256(gen),
            "text": detokenize(table, gen),
        }
    return {
        "format": "quettos-expected-tokens",
        "model": {"repo_id": model.repo_id, "name": model.name, "layers": model.n_layers},
        "numerics": model.numerics_version,
        "calib_tokens_sha256": model.calib_tokens_sha256,
        "a_bits": a_bits,
        "max_new": max_new,
        "eos_ids": [int(e) for e in spec.eos_ids],
        "sha256_of": "json list of generated_ids without spaces",
        "prompts": prompts,
    }


def expected_tokens_path(model: QuantModel) -> Path:
    return MODELS_OUT_DIR / model.name / "expected_tokens.json"


def expected_tokens_text(report: dict[str, Any]) -> str:
    """Canonical text of a report (sorted keys, UTF-8, trailing newline)."""
    return json.dumps(report, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def write_expected_tokens(report: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected_tokens_text(report), encoding="utf-8")
    return path
