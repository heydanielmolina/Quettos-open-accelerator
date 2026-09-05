"""Float32 numpy reference forward for Llama and Qwen2 decoders.

The floating-point oracle the integer numerics are measured against: it
reproduces the Hugging Face ``transformers`` fp32 forward without torch
(RMSNorm, rotate_half RoPE with float32 ``inv_freq``, grouped-query causal
attention, SwiGLU, tied LM head), reading weights one layer at a time through
:mod:`quettos.safetensors_np`.  Intermediates are exposed through a recorder
passed as ``hooks``; see :func:`forward` for the names.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from quettos import safetensors_np
from quettos.model import ModelSpec

MAX_TOKENS = 1024
LOG2E = math.log2(math.e)

HOOK_NAMES: tuple[str, ...] = (
    "embed",
    "x_norm_attn",
    "q",
    "k",
    "v",
    "v_raw",
    "q_rope",
    "k_rope",
    "scores",
    "ctx",
    "x_attn",
    "x_norm_mlp",
    "gate",
    "up",
    "h",
    "x",
    "x_norm_final",
    "logits",
)


class Recorder(Protocol):
    """Anything with ``record(name, layer, value)`` can observe the forward pass."""

    def record(self, name: str, layer: int | None, value: np.ndarray) -> None: ...


class Capture:
    """Recorder that keeps copies of selected intermediates in ``self.values[(name, layer)]``."""

    def __init__(self, names: Sequence[str] | None = None) -> None:
        self.names = None if names is None else frozenset(names)
        self.values: dict[tuple[str, int | None], np.ndarray] = {}

    def record(self, name: str, layer: int | None, value: np.ndarray) -> None:
        if self.names is None or name in self.names:
            self.values[(name, layer)] = np.array(value, copy=True)

    def get(self, name: str, layer: int | None = None) -> np.ndarray:
        return self.values[(name, layer)]


# --------------------------------------------------------------------------- weights


@dataclass(frozen=True)
class LayerWeights:
    """Float32 tensors of one decoder layer (HF names, ``[out, in]`` layout)."""

    norm_in: np.ndarray
    norm_post: np.ndarray
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    bq: np.ndarray | None
    bk: np.ndarray | None
    bv: np.ndarray | None
    w_gate: np.ndarray
    w_up: np.ndarray
    w_down: np.ndarray


def load_layer(spec: ModelSpec, i: int) -> LayerWeights:
    """Read every tensor of layer ``i`` as float32."""
    path = spec.path("model.safetensors")
    p = f"model.layers.{i}."

    def t(name: str) -> np.ndarray:
        return safetensors_np.load_tensor(path, p + name)

    bias = spec.has_qkv_bias
    return LayerWeights(
        norm_in=t("input_layernorm.weight"),
        norm_post=t("post_attention_layernorm.weight"),
        wq=t("self_attn.q_proj.weight"),
        wk=t("self_attn.k_proj.weight"),
        wv=t("self_attn.v_proj.weight"),
        wo=t("self_attn.o_proj.weight"),
        bq=t("self_attn.q_proj.bias") if bias else None,
        bk=t("self_attn.k_proj.bias") if bias else None,
        bv=t("self_attn.v_proj.bias") if bias else None,
        w_gate=t("mlp.gate_proj.weight"),
        w_up=t("mlp.up_proj.weight"),
        w_down=t("mlp.down_proj.weight"),
    )


def load_embedding(spec: ModelSpec) -> np.ndarray:
    """The tied embedding / LM-head table, ``[vocab, hidden]`` float32."""
    return safetensors_np.load_tensor(spec.path("model.safetensors"), "model.embed_tokens.weight")


def load_final_norm(spec: ModelSpec) -> np.ndarray:
    return safetensors_np.load_tensor(spec.path("model.safetensors"), "model.norm.weight")


# --------------------------------------------------------------------------- operators


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """``weight * x * rsqrt(mean(x^2) + eps)``, all in float32 (HF ``LlamaRMSNorm``)."""
    x = x.astype(np.float32, copy=False)
    var = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (weight * (x / np.sqrt(var + np.float32(eps)))).astype(np.float32)


def rope_inv_freq(theta: float, head_dim: int) -> np.ndarray:
    """``1 / theta ** (arange(0, D, 2) / D)`` evaluated in float32 like HF's rotary embedding."""
    idx = np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32)
    return (np.float32(1.0) / (np.float32(theta) ** (idx / np.float32(head_dim)))).astype(
        np.float32
    )


def rope_cos_sin(
    positions: np.ndarray, theta: float, head_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """``cos`` and ``sin`` tables ``[T, D]``, each half duplicated (HF ``cat(freqs, freqs)``)."""
    inv_freq = rope_inv_freq(theta, head_dim)
    freqs = positions.astype(np.float32)[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """``x * cos + rotate_half(x) * sin`` on ``[T, heads, D]`` with ``cos``/``sin`` ``[T, D]``."""
    c = cos[:, None, :]
    s = sin[:, None, :]
    return (x * c + rotate_half(x) * s).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return (x / (np.float32(1.0) + np.exp(-x))).astype(np.float32)


def softmax_rows(x: np.ndarray) -> np.ndarray:
    """Softmax over the last axis in float32; ``-inf`` entries become exactly 0."""
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return (e / np.sum(e, axis=-1, keepdims=True, dtype=np.float32)).astype(np.float32)


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """``x @ w.T + b`` in float32 (``w`` is ``[out, in]`` as stored by HF)."""
    y = x @ w.T
    if b is not None:
        y = y + b
    return y.astype(np.float32, copy=False)


def causal_mask(t: int) -> np.ndarray:
    """Boolean ``[T, T]`` mask, ``True`` where key ``j <= query i``."""
    return np.tril(np.ones((t, t), dtype=bool))


# --------------------------------------------------------------------------- forward


def _record(hooks: Recorder | None, name: str, layer: int | None, value: np.ndarray) -> None:
    if hooks is not None:
        hooks.record(name, layer, value)


def attention(
    spec: ModelSpec,
    x_norm: np.ndarray,
    w: LayerWeights,
    cos: np.ndarray,
    sin: np.ndarray,
    layer: int,
    hooks: Recorder | None,
) -> np.ndarray:
    """One attention block up to and including ``o_proj`` (no residual add)."""
    t = x_norm.shape[0]
    h, kv, d = spec.heads, spec.kv_heads, spec.head_dim
    n_rep = h // kv

    q = linear(x_norm, w.wq, w.bq)
    k = linear(x_norm, w.wk, w.bk)
    v_raw = linear(x_norm, w.wv)
    v = v_raw if w.bv is None else (v_raw + w.bv).astype(np.float32)
    _record(hooks, "q", layer, q)
    _record(hooks, "k", layer, k)
    _record(hooks, "v", layer, v)
    _record(hooks, "v_raw", layer, v_raw)

    q_r = apply_rope(q.reshape(t, h, d), cos, sin)
    k_r = apply_rope(k.reshape(t, kv, d), cos, sin)
    _record(hooks, "q_rope", layer, q_r.reshape(t, h * d))
    _record(hooks, "k_rope", layer, k_r.reshape(t, kv * d))

    # [H, T, D] with each KV head repeated n_rep times (HF repeat_kv).
    qh = np.transpose(q_r, (1, 0, 2))
    kh = np.repeat(np.transpose(k_r, (1, 0, 2)), n_rep, axis=0)
    vh = np.repeat(np.transpose(v.reshape(t, kv, d), (1, 0, 2)), n_rep, axis=0)

    scaling = np.float32(1.0 / math.sqrt(d))
    scores = (qh @ np.transpose(kh, (0, 2, 1))) * scaling  # [H, T, T]
    mask = causal_mask(t)
    if hooks is not None:
        hooks.record("scores", layer, np.where(mask, scores * np.float32(LOG2E), np.float32(0.0)))
    probs = softmax_rows(np.where(mask, scores, np.float32(-np.inf)))
    ctx = np.transpose(probs @ vh, (1, 0, 2)).reshape(t, h * d)
    _record(hooks, "ctx", layer, ctx)
    return linear(ctx, w.wo)


def mlp(x_norm: np.ndarray, w: LayerWeights, layer: int, hooks: Recorder | None) -> np.ndarray:
    gate = linear(x_norm, w.w_gate)
    up = linear(x_norm, w.w_up)
    hidden = (silu(gate) * up).astype(np.float32)
    _record(hooks, "gate", layer, gate)
    _record(hooks, "up", layer, up)
    _record(hooks, "h", layer, hidden)
    return linear(hidden, w.w_down)


def forward(
    spec: ModelSpec,
    ids: Sequence[int],
    hooks: Recorder | None = None,
    *,
    layers: int | None = None,
) -> np.ndarray:
    """Logits ``[T, vocab]`` (float32) for the token ids ``ids`` (``1 <= T <= 1024``).

    ``layers`` truncates the stack to its first ``layers`` decoder layers (the
    final norm and LM head still run); ``None`` runs the whole model.

    Hook names (``layer`` is the layer index, or ``None`` outside the stack):
    ``embed``, ``x_norm_attn``, ``q``/``k``/``v``, ``v_raw``, ``q_rope``, ``k_rope``,
    ``scores`` (log2 domain, causal window), ``ctx``, ``x_attn``, ``x_norm_mlp``,
    ``gate``/``up``, ``h``, ``x``, ``x_norm_final``, ``logits``.
    """
    ids_arr = np.asarray(list(ids), dtype=np.int64)
    t = ids_arr.shape[0]
    if t < 1 or t > MAX_TOKENS:
        raise ValueError(f"forward: sequence length {t} not in [1, {MAX_TOKENS}]")
    if np.any(ids_arr < 0) or np.any(ids_arr >= spec.vocab):
        raise ValueError("forward: token id out of range")
    n_layers = spec.layers if layers is None else min(layers, spec.layers)

    embed = load_embedding(spec)
    x = embed[ids_arr].astype(np.float32)
    _record(hooks, "embed", None, x)

    cos, sin = rope_cos_sin(np.arange(t), spec.rope_theta, spec.head_dim)
    eps = spec.rms_norm_eps

    for i in range(n_layers):
        w = load_layer(spec, i)
        xn = rms_norm(x, w.norm_in, eps)
        _record(hooks, "x_norm_attn", i, xn)
        x = (x + attention(spec, xn, w, cos, sin, i, hooks)).astype(np.float32)
        _record(hooks, "x_attn", i, x)
        xn = rms_norm(x, w.norm_post, eps)
        _record(hooks, "x_norm_mlp", i, xn)
        x = (x + mlp(xn, w, i, hooks)).astype(np.float32)
        _record(hooks, "x", i, x)
        del w

    xn = rms_norm(x, load_final_norm(spec), eps)
    _record(hooks, "x_norm_final", None, xn)
    logits = linear(xn, embed)
    _record(hooks, "logits", None, logits)
    return logits


# --------------------------------------------------------------------------- HF oracle


def hf_logits(spec: ModelSpec, ids: Sequence[int]) -> np.ndarray:
    """Logits ``[T, vocab]`` from ``transformers`` in fp32 with eager attention.

    Requires the optional ``ref`` dependency group (``uv sync --group ref``);
    raises :class:`ImportError` otherwise.  Loads from the local model
    directory, so no network access is needed.
    """
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = {"attn_implementation": "eager"}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(spec.model_dir), dtype=torch.float32, **kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            str(spec.model_dir), torch_dtype=torch.float32, **kwargs
        )
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor([list(ids)], dtype=torch.long)).logits[0]
    return out.float().numpy()


def compare_logits(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Max absolute difference and argmax agreement (fraction of positions) of two logit sets."""
    if a.shape != b.shape:
        raise ValueError(f"compare_logits: shapes {a.shape} vs {b.shape}")
    diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    agree = float(np.mean(np.argmax(a, axis=-1) == np.argmax(b, axis=-1)))
    return {"max_abs_diff": diff, "argmax_agreement": agree, "positions": float(a.shape[0])}
