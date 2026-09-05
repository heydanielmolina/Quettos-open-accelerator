"""Golden model: sequential steps == teacher-forced forward, the prefill/decode loop, counters,
argmax agreement with the fp32 reference, the trace hook and the per-GEMV constants.

The fast fixtures are the complete SmolLM2 and a 2-layer Qwen; tests marked
``slow`` run the complete Qwen from ``build/quant/`` and the checked-in
``expected_tokens.json`` files.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from quettos import calibrate, golden, numerics, program, quantize, reference_np
from quettos.model import REPO_ROOT, ModelSpec
from quettos.numerics import Stats
from quettos.tokenizer_io import prompt_tokens

PROMPTS = ("chat_short.json", "tool_call_weather.json")
SEQ_TOKENS = 40
# Top-1 agreement with fp32 for complete models at a_bits = 16, per prompt.  SmolLM2 is the
# gated model (docs/VERIFICATION.md); Qwen is reported, its floor only catches gross failures.
AGREEMENT_GATE = {"smollm2-135m-instruct": 0.90, "qwen2.5-0.5b-instruct": 0.80}


class Trace:
    """Records every ``(op, layer)`` the hook receives, with a copy of the array."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []
        self.values: dict[tuple[str, int | None], np.ndarray] = {}

    def __call__(self, name: str, layer: int | None, value: np.ndarray) -> None:
        self.calls.append((name, layer))
        self.values[(name, layer)] = np.array(value, copy=True)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def calib(spec: ModelSpec) -> dict:
    path = calibrate.calib_path(spec)
    if not path.is_file():
        pytest.skip(f"{path} not present")
    return calibrate.load_calib(path)


@pytest.fixture(scope="session")
def qmodel(spec: ModelSpec, calib: dict) -> quantize.QuantModel:
    """Complete SmolLM2; Qwen truncated to two layers."""
    layers = 2 if spec.arch == "qwen2" else None
    return quantize.build_quant_model(spec, calib, layers=layers)


@pytest.fixture(scope="session")
def qmodel_full(spec: ModelSpec) -> quantize.QuantModel:
    path = quantize.default_path(spec.name)
    if not path.is_file():
        pytest.skip(f"{path} not present (run: uv run quettos quantize)")
    model = quantize.load(path)
    if model.n_layers != spec.layers:
        pytest.skip(f"{path} is truncated to {model.n_layers} layers")
    return model


def _ids(spec: ModelSpec, prompt: str) -> list[int]:
    return prompt_tokens(spec, REPO_ROOT / "prompts" / prompt)


@pytest.fixture(scope="session")
def ids40(spec: ModelSpec) -> list[int]:
    ids = (_ids(spec, PROMPTS[0]) + _ids(spec, PROMPTS[1]))[:SEQ_TOKENS]
    assert len(ids) == SEQ_TOKENS
    return ids


def _caches_equal(a: golden.KVCache, b: golden.KVCache) -> bool:
    return a.length == b.length and all(
        np.array_equal(getattr(a, f), getattr(b, f)) for f in ("k", "v", "k_m", "k_e", "v_m", "v_e")
    )


def _agreement(golden_argmax: np.ndarray, ref_logits: np.ndarray) -> float:
    return float(np.mean(np.argmax(ref_logits, axis=-1) == golden_argmax))


# --------------------------------------------------------------------------- constants


@pytest.mark.parametrize("a_bits", [16, 8])
def test_program_constants_window_and_rules(qmodel: quantize.QuantModel, a_bits: int) -> None:
    pc = program.build(qmodel, a_bits=a_bits)
    assert set(pc.gemvs) == set(program.GEMV_NAMES)
    for g in pc.gemvs.values():
        lo, hi = g.shift_range
        assert 0 <= lo <= hi <= program.SHIFT_MAX, g
        # S is smallest at the largest exponents and largest at the smallest
        assert lo == numerics.requant_shift(
            numerics.SFloat(1 << 15, g.sw_e[1]), numerics.SFloat(1 << 15, g.sx_e[1]), g.sbias
        )
        assert hi == numerics.requant_shift(
            numerics.SFloat(1 << 15, g.sw_e[0]), numerics.SFloat(1 << 15, g.sx_e[0]), g.sbias
        )
        assert g.sw_e[0] <= g.sw_e[1] and g.sx_e[0] <= g.sx_e[1]
        if g.name == "embed":
            assert g.s1 == program.EMBED_S1
            assert g.sbias == numerics.sbias_for(g.frac_out, g.s1, program.EMBED_PRE_SHIFT)
            assert g.sx_e == (numerics.SFLOAT_ONE.e, numerics.SFLOAT_ONE.e)
            continue
        assert g.sbias == numerics.sbias_for(g.frac_out, g.s1)
        assert g.acc_bits == program.acc_bits_for(g.k, g.a_bits)
        expected = numerics.choose_s1(g.acc_bits, g.frac_out, g.sw_e[1], g.sx_e[1] + 1)
        # one octave of activation margin; lowered only when the hard bound needs it
        assert g.s1 == expected or (g.s1 < expected and g.hard_shift_min == 0)
        assert g.hard_shift_min >= 0
        # t = acc * Sw_m >> s1 fits the 40-bit stage-1 register
        assert g.acc_bits + 16 - g.s1 <= 40
    assert pc["scores"].k == 64 and pc["pv"].k == pc.max_ctx == program.MAX_CTX
    assert pc["pv"].sw_e == (numerics.SFLOAT_ONE.e, numerics.SFLOAT_ONE.e)
    assert pc["qkv"].a_bits == a_bits and pc["scores"].a_bits == 16
    assert pc["lm_head"].frac_out == 16 and pc["lm_head"].sw_e == pc["embed"].sw_e
    json.dumps(pc.as_dict())  # layout.json material


def test_acc_bits_rule() -> None:
    for k, a in ((896, 16), (4864, 16), (64, 16), (2048, 16), (576, 8), (1536, 8)):
        bound = k * ((1 << (a - 1)) - 1) * 127
        assert program.acc_bits_for(k, a) == min(40, bound.bit_length() + 1)
    assert program.acc_bits_for(4864, 16) == 36
    assert program.acc_bits_for(64, 16) == 29
    assert program.acc_bits_for(1 << 20, 16) == 40


def test_program_rejects_foreign_calibration(qmodel: quantize.QuantModel, calib: dict) -> None:
    other = dict(calib, tokens=dict(calib["tokens"], sha256="0" * 64))
    with pytest.raises(ValueError):
        program.build(qmodel, other)
    with pytest.raises(ValueError):
        program.build(qmodel, calib, a_bits=4)


def test_program_from_model_absmax_equals_program_from_calib(
    qmodel: quantize.QuantModel, calib: dict
) -> None:
    """The maxima the quantizer stores in ``extra`` give the same constants as calib.json."""
    assert qmodel.extra["absmax"] == calib["absmax"]
    from_model = program.build(qmodel)
    from_calib = program.build(qmodel, calib)
    assert from_model == from_calib
    bare = quantize.QuantModel(**{**qmodel.__dict__, "extra": {}})
    assert program.build(bare) == from_calib  # falls back to models/<name>/calib.json


# --------------------------------------------------------------------------- exactness


def test_exact_matmul_matches_integer_product_and_checks_bound() -> None:
    rng = np.random.default_rng(7)
    a = rng.integers(-32767, 32768, (5, 4864)).astype(np.int64)
    w = rng.integers(-127, 128, (33, 4864)).astype(np.int8)
    assert np.array_equal(golden.exact_matmul(a, w), a @ w.astype(np.int64).T)
    big = np.full((1, 1 << 16), 1 << 30, dtype=np.int64)  # 2^16 * 2^30 * 128 = 2^53
    with pytest.raises(ValueError):
        golden.exact_matmul(big, np.ones((1, 1 << 16), dtype=np.int8))


# --------------------------------------------------------------------------- equivalences


def test_sequential_steps_match_forward_tokens(
    spec: ModelSpec, qmodel: quantize.QuantModel, ids40: list[int]
) -> None:
    """step() at every position reproduces forward_tokens: logits, argmax, cache and counters."""
    st_all, tr_all = Stats(), Trace()
    out = golden.forward_tokens(qmodel, ids40, stats=st_all, trace=tr_all)
    assert out.logits.shape == (SEQ_TOKENS, spec.vocab) and out.logits.dtype == np.int32
    assert out.argmax.shape == (SEQ_TOKENS,) and out.cache.length == SEQ_TOKENS
    assert np.array_equal(tr_all.values[("gemv_lm_head", None)], out.logits)

    cache = golden.new_cache(qmodel)
    st_seq = Stats()
    for pos, tok in enumerate(ids40):
        tr = Trace()
        got = golden.step(qmodel, cache, tok, pos, lm_head=True, stats=st_seq, trace=tr)
        logits = tr.values[("gemv_lm_head", None)]
        assert logits.shape == (1, spec.vocab)
        assert np.array_equal(logits[0], out.logits[pos]), pos
        assert got == int(out.argmax[pos]) == numerics.argmax(out.logits[pos]), pos
        assert cache.length == pos + 1
    assert _caches_equal(cache, out.cache)
    assert st_seq == st_all
    assert st_all.err_shift == 0 and st_all.sat == 0


def test_prefill_step_writes_kv_without_lm_head(qmodel: quantize.QuantModel, ids40) -> None:
    cache = golden.new_cache(qmodel)
    tr = Trace()
    assert golden.step(qmodel, cache, ids40[0], 0, lm_head=False, trace=tr) is None
    names = {n for n, _ in tr.calls}
    assert not names & set(golden.TRACE_OPS_HEAD)
    assert cache.length == 1 and np.all(cache.k_m[:, :, 0] != 0)
    ref = golden.forward_tokens(qmodel, ids40[:1])
    assert _caches_equal(cache, ref.cache)


def test_generate_reproduces_forward_tokens(spec: ModelSpec, qmodel: quantize.QuantModel) -> None:
    """The prefill/decode loop yields exactly the greedy argmax of the teacher-forced forward."""
    prompt = _ids(spec, PROMPTS[0])
    p_len = len(prompt)
    max_new = 6
    st = Stats()
    seen: list[tuple[int, int, int]] = []
    gen = golden.generate(
        qmodel, prompt, max_new, eos_ids=spec.eos_ids, stats=st, on_token=lambda *a: seen.append(a)
    )
    assert 1 <= len(gen) <= max_new
    assert seen == [(j, p_len - 1 + j, tok) for j, tok in enumerate(gen)]
    if len(gen) < max_new:
        assert gen[-1] in spec.eos_ids
    full = golden.forward_tokens(qmodel, prompt + gen[:-1])
    assert full.argmax[p_len - 1 :].tolist() == gen
    assert st.err_shift == 0 and st.sat == 0


def test_forward_is_deterministic(qmodel: quantize.QuantModel, ids40: list[int]) -> None:
    a = golden.forward_tokens(qmodel, ids40)
    b = golden.forward_tokens(qmodel, ids40)
    assert np.array_equal(a.logits, b.logits) and np.array_equal(a.argmax, b.argmax)
    assert _caches_equal(a.cache, b.cache)


def test_cache_size_does_not_change_the_program(qmodel: quantize.QuantModel, ids40) -> None:
    """A small cache runs the same constants as the compiled program at MAX_CTX."""
    full = golden.forward_tokens(qmodel, ids40)
    small = golden.forward_tokens(qmodel, ids40, max_ctx=SEQ_TOKENS)
    assert np.array_equal(small.logits, full.logits)
    assert small.cache.max_ctx == SEQ_TOKENS and small.cache.length == SEQ_TOKENS
    prog = program.build(qmodel)
    assert prog.max_ctx == program.MAX_CTX
    again = golden.forward_tokens(qmodel, ids40, max_ctx=SEQ_TOKENS, prog=prog)
    assert np.array_equal(again.logits, full.logits)
    with pytest.raises(ValueError):  # constants of a smaller program cannot serve a larger cache
        golden.forward_tokens(qmodel, ids40, prog=program.build(qmodel, max_ctx=SEQ_TOKENS))


def test_gemv_blocks_are_exact_and_bounded(qmodel: quantize.QuantModel, monkeypatch) -> None:
    """Block size never changes a GEMV result; the float64 weight slice stays under the bound."""
    prog = program.build(qmodel)
    run = golden._prepare(qmodel, 16, 8, None, None, prog)
    rng = np.random.default_rng(5)
    lin = qmodel.layers[0].wgu
    n, k = lin.q.shape
    a = rng.integers(-32767, 32768, (3, k)).astype(np.int64)
    sx_m = np.full(3, 40000, dtype=np.int64)
    sx_e = np.full(3, -31, dtype=np.int64)
    one_block = golden._gemv_requant(run, a, sx_m, sx_e, lin, prog["gu"])
    seen: list[tuple[int, int]] = []
    orig = golden.exact_matmul
    chunk = golden.GEMV_CHUNK_ELEMS

    def spy(a_, w_):
        seen.append(w_.shape)
        return orig(a_, w_)

    monkeypatch.setattr(golden, "exact_matmul", spy)
    monkeypatch.setattr(golden, "GEMV_CHUNK_ELEMS", 21 * k)  # 21 weight rows per block
    blocks = golden._gemv_requant(run, a, sx_m, sx_e, lin, prog["gu"])
    assert np.array_equal(blocks, one_block)
    assert len(seen) == -(-n // 21) and all(rows <= 21 and cols == k for rows, cols in seen)
    # at T = 1 the LM-head weight slice is bounded by GEMV_CHUNK_ELEMS elements as well
    monkeypatch.setattr(golden, "GEMV_CHUNK_ELEMS", chunk)
    seen.clear()
    golden._gemv_requant(run, a[:1], sx_m[:1], sx_e[:1], qmodel.embed, prog["lm_head"])
    assert all(rows * cols <= chunk for rows, cols in seen) and len(seen) >= 2


@pytest.mark.parametrize("prompt", PROMPTS)
def test_counters_on_prompts(spec: ModelSpec, qmodel: quantize.QuantModel, prompt: str) -> None:
    """No shift error and no saturation; the VQUANT clips are counted and reported."""
    ids = _ids(spec, prompt)
    st = Stats()
    golden.forward_tokens(qmodel, ids, stats=st)
    print(f"{spec.name} layers={qmodel.n_layers} {prompt} T={len(ids)}: {st}")
    assert st.err_shift == 0
    assert st.sat == 0
    assert st.clip > 0


@pytest.mark.parametrize("prompt", PROMPTS)
def test_argmax_agreement_vs_fp32(
    spec: ModelSpec, qmodel: quantize.QuantModel, prompt: str
) -> None:
    """Top-1 agreement with reference_np over the same (possibly truncated) layer stack."""
    ids = _ids(spec, prompt)
    out = golden.forward_tokens(qmodel, ids)
    ref = reference_np.forward(spec, ids, layers=qmodel.n_layers)
    agree = _agreement(out.argmax, ref)
    print(f"{spec.name} layers={qmodel.n_layers} {prompt} T={len(ids)}: agreement {agree:.4f}")
    if qmodel.n_layers == spec.layers:
        assert agree >= AGREEMENT_GATE[spec.name]
    else:
        assert agree >= 0.5


def test_a_bits_8_runs_and_differs(spec: ModelSpec, qmodel: quantize.QuantModel) -> None:
    ids = _ids(spec, PROMPTS[0])
    st16, st8 = Stats(), Stats()
    out16 = golden.forward_tokens(qmodel, ids, a_bits=16, stats=st16)
    out8 = golden.forward_tokens(qmodel, ids, a_bits=8, stats=st8)
    assert not np.array_equal(out16.logits, out8.logits)
    assert st8.err_shift == 0 and st8.sat == 0
    ref = reference_np.forward(spec, ids, layers=qmodel.n_layers)
    a8, a16 = _agreement(out8.argmax, ref), _agreement(out16.argmax, ref)
    print(f"{spec.name} layers={qmodel.n_layers} agreement a8 {a8:.4f} a16 {a16:.4f}; {st8}")
    # the two widths never share program constants
    assert program.build(qmodel, a_bits=8)["qkv"].s1 != program.build(qmodel, a_bits=16)["qkv"].s1
    with pytest.raises(ValueError):
        golden.forward_tokens(qmodel, ids, a_bits=8, prog=program.build(qmodel, a_bits=16))


def test_trace_hook_receives_every_op(spec: ModelSpec, qmodel: quantize.QuantModel) -> None:
    ids = _ids(spec, PROMPTS[0])[:5]
    t = len(ids)
    tr = Trace()
    out = golden.forward_tokens(qmodel, ids, trace=tr)
    want = [(n, None) for n in golden.TRACE_OPS_EMBED]
    for layer in range(qmodel.n_layers):
        want += [(n, layer) for n in golden.TRACE_OPS_LAYER]
    want += [(n, None) for n in golden.TRACE_OPS_HEAD]
    assert tr.calls == want
    h, kv, d, hid = spec.heads, spec.kv_heads, spec.head_dim, spec.hidden
    shapes = {
        ("embed", None): (t, hid),
        ("rmsnorm_in", 0): (t, hid),
        ("quant_in", 0): (t, hid),
        ("quant_in.scale", 0): (t, 2),
        ("gemv_qkv", 0): (t, (h + 2 * kv) * d),
        ("rope_q", 0): (t, h * d),
        ("rope_k", 0): (t, kv * d),
        ("quant_q.scale", 0): (t, h, 2),
        ("subc_k", 0): (t, kv * d),
        ("quant_k", 0): (t, kv * d),
        ("quant_v.scale", 0): (t, kv, 2),
        ("gemv_scores", 0): (t, h, t),
        ("softmax", 0): (t, h, t),
        ("softmax.sreg", 0): (t, h, 2),
        ("gemv_pv", 0): (t, h * d),
        ("gemv_o", 0): (t, hid),
        ("gemv_gu", 0): (t, 2 * spec.intermediate),
        ("silu_mul", 0): (t, spec.intermediate),
        ("quant_h", 0): (t, spec.intermediate),
        ("gemv_down", qmodel.n_layers - 1): (t, hid),
        ("rmsnorm_final", None): (t, hid),
        ("gemv_lm_head", None): (t, spec.vocab),
        ("argmax", None): (t,),
    }
    for key, shape in shapes.items():
        assert tr.values[key].shape == shape, key
    assert np.array_equal(tr.values[("argmax", None)], out.argmax)
    # int8 K rows and their scales, as written to the cache
    k_q = tr.values[("quant_k", 0)]
    assert int(np.max(np.abs(k_q))) <= 127
    assert np.array_equal(
        k_q.reshape(t, kv, d).astype(np.int8), out.cache.k[0, :, :t].transpose(1, 0, 2)
    )
    # scores beyond the causal window are zero, softmax weights are int16 and zero there too
    scores, w = tr.values[("gemv_scores", 0)], tr.values[("softmax", 0)]
    upper = ~reference_np.causal_mask(t)
    assert np.all(scores.transpose(1, 0, 2)[:, upper] == 0) and np.all(
        w.transpose(1, 0, 2)[:, upper] == 0
    )
    assert int(np.max(w)) <= numerics.I16_MAX and int(np.min(w)) >= 0


def test_step_rejects_bad_positions_and_ids(qmodel: quantize.QuantModel) -> None:
    cache = golden.new_cache(qmodel, max_ctx=8)
    with pytest.raises(ValueError):
        golden.step(qmodel, cache, 1, 1)  # skips position 0
    with pytest.raises(ValueError):
        golden.step(qmodel, cache, 1, 8)
    with pytest.raises(ValueError):
        golden.step(qmodel, cache, qmodel.vocab, 0)
    with pytest.raises(ValueError):
        golden.forward_tokens(qmodel, [])
    with pytest.raises(ValueError):
        golden.forward_tokens(qmodel, [1] * 9, max_ctx=8)
    with pytest.raises(ValueError):
        golden.generate(qmodel, [1] * 5, 5, max_ctx=8)  # last decode position would be 8
    assert len(golden.generate(qmodel, [1] * 5, 4, max_ctx=8)) == 4  # position 7 still fits


def test_ids_sha256_is_the_compact_json_hash() -> None:
    import hashlib

    assert golden.ids_sha256([1, 2, 3]) == hashlib.sha256(b"[1,2,3]").hexdigest()
    assert golden.ids_sha256([]) == hashlib.sha256(b"[]").hexdigest()


# --------------------------------------------------------------------------- complete models


@pytest.mark.slow
@pytest.mark.parametrize("prompt", PROMPTS)
def test_full_model_argmax_agreement_vs_fp32(
    spec: ModelSpec, qmodel_full: quantize.QuantModel, prompt: str
) -> None:
    ids = _ids(spec, prompt)
    st = Stats()
    out = golden.forward_tokens(qmodel_full, ids, stats=st)
    ref = reference_np.forward(spec, ids)
    agree = _agreement(out.argmax, ref)
    print(f"{spec.name} {prompt} T={len(ids)}: agreement {agree:.4f} {st}")
    assert st.err_shift == 0 and st.sat == 0
    assert agree >= AGREEMENT_GATE[spec.name]


@pytest.mark.slow
def test_full_model_counters_over_calibration_set(
    spec: ModelSpec, qmodel_full: quantize.QuantModel
) -> None:
    total = Stats()
    for ids in calibrate.calibration_sequences(spec):
        st = Stats()
        golden.forward_tokens(qmodel_full, ids, stats=st)
        print(f"{spec.name} T={len(ids)}: {st}")
        total = total + st
    print(f"{spec.name} calibration set: {total}")
    assert total.err_shift == 0 and total.sat == 0


@pytest.mark.slow
def test_expected_tokens_file_reproduces(spec: ModelSpec, qmodel_full: quantize.QuantModel) -> None:
    """The checked-in greedy continuations are exactly what the golden model produces."""
    path = golden.expected_tokens_path(qmodel_full)
    if not path.is_file():
        pytest.skip(f"{path} not present")
    stored = json.loads(path.read_text(encoding="utf-8"))
    files = [REPO_ROOT / k for k in stored["prompts"]]
    st = Stats()
    fresh = golden.expected_tokens(
        qmodel_full, spec, files, max_new=stored["max_new"], a_bits=stored["a_bits"], stats=st
    )
    for key, r in fresh["prompts"].items():
        print(f"{spec.name} {key}: {r['prompt_tokens']} -> {len(r['generated_ids'])} {r['text']!r}")
        assert r["sha256"] == golden.ids_sha256(r["generated_ids"])
    assert fresh == stored
    assert golden.expected_tokens_text(fresh) == path.read_text(encoding="utf-8")
    assert st.err_shift == 0 and st.sat == 0
