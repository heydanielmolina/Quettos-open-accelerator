"""Quantizer and calibration file: exactness of every integer against the numerics primitives,
and the pairwise Q/K smoothing (factor rule, fold exactness, K-centering consistency)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from quettos import calibrate, numerics, quantize, reference_np
from quettos.model import ModelSpec
from quettos.reference_np import LayerWeights, load_embedding, load_layer

TEST_LAYERS = 2
CALIB_MAX_BYTES = 300_000
LOG_GEOMEAN_TOL = 1e-5  # six-digit rounding of the factors moves mean(log s) by < 1e-6


@pytest.fixture(scope="session")
def calib(spec: ModelSpec) -> dict:
    path = calibrate.calib_path(spec)
    if not path.is_file():
        pytest.skip(f"{path} not present")
    return calibrate.load_calib(path)


@pytest.fixture(scope="session")
def factors(calib: dict) -> np.ndarray:
    return np.asarray(calib["qk_smoothing"]["factors"], dtype=np.float64)


@pytest.fixture(scope="session")
def qmodel(spec: ModelSpec, calib: dict) -> quantize.QuantModel:
    return quantize.build_quant_model(spec, calib, layers=TEST_LAYERS)


@pytest.fixture(scope="session")
def layer0(spec: ModelSpec) -> LayerWeights:
    return load_layer(spec, 0)


@pytest.fixture(scope="session")
def layer0_smoothed(spec: ModelSpec, layer0: LayerWeights, factors: np.ndarray) -> LayerWeights:
    return quantize.smooth_qk(layer0, factors[0], spec.heads, spec.kv_heads, spec.head_dim)


def _check_int8_rows(lin: quantize.QuantLinear, w: np.ndarray) -> None:
    w = np.asarray(w, dtype=np.float64)
    assert lin.q.dtype == np.int8
    assert lin.q.shape == w.shape
    assert int(np.max(np.abs(lin.q.astype(np.int64)))) <= 127
    absmax = np.max(np.abs(w), axis=1)
    for n in range(w.shape[0]):
        want = numerics.sfloat_from_float(absmax[n] / 127.0) if absmax[n] else numerics.SFLOAT_ZERO
        assert (int(lin.scale_m[n]), int(lin.scale_e[n])) == (want.m, want.e), n
    step = lin.scale_m.astype(np.float64) * np.exp2(lin.scale_e.astype(np.float64))
    err = np.abs(lin.dequant() - w)
    assert np.all(err <= step[:, None] / 2 + 1e-12)
    zero_rows = absmax == 0
    assert np.all(lin.q[zero_rows] == 0)


def test_linear_rows_layer0(
    qmodel: quantize.QuantModel, layer0: LayerWeights, layer0_smoothed: LayerWeights
) -> None:
    """The QKV rows are quantized from the smoothed W_q / W_k; V, o, gate|up and down as loaded."""
    lay = qmodel.layers[0]
    ws = layer0_smoothed
    _check_int8_rows(lay.wqkv, np.concatenate([ws.wq, ws.wk, layer0.wv], axis=0))
    _check_int8_rows(lay.wo, layer0.wo)
    _check_int8_rows(lay.wgu, np.concatenate([layer0.w_gate, layer0.w_up], axis=0))
    _check_int8_rows(lay.wdown, layer0.w_down)


def test_embedding_rows(spec: ModelSpec, qmodel: quantize.QuantModel) -> None:
    table = load_embedding(spec)
    assert qmodel.embed.q.shape == (spec.vocab, spec.hidden)
    rows = np.arange(0, spec.vocab, max(1, spec.vocab // 512))
    sub = quantize.QuantLinear(
        qmodel.embed.q[rows],
        qmodel.embed.scale_m[rows],
        qmodel.embed.scale_e[rows],
        qmodel.embed.bias_q[rows],
    )
    _check_int8_rows(sub, table[rows])
    assert np.all(qmodel.embed.bias_q == 0)


def test_bias_q(
    spec: ModelSpec,
    qmodel: quantize.QuantModel,
    layer0: LayerWeights,
    layer0_smoothed: LayerWeights,
) -> None:
    lay = qmodel.layers[0]
    h, kv, d = spec.heads, spec.kv_heads, spec.head_dim
    frac_qkv, frac_x = qmodel.frac["QKV"], qmodel.frac["X"]
    assert lay.wqkv.bias_q.dtype == np.int32
    assert lay.wo.bias_q.dtype == np.int32
    # V rows never carry a bias in the QKV GEMV: it is folded into o_proj.
    assert np.all(lay.wqkv.bias_q[h * d :][kv * d :] == 0)
    if spec.has_qkv_bias:
        ws = layer0_smoothed
        assert np.array_equal(lay.wqkv.bias_q[: h * d], numerics.to_fixed(ws.bq, frac_qkv))
        assert np.array_equal(
            lay.wqkv.bias_q[h * d : h * d + kv * d], numerics.to_fixed(ws.bk, frac_qkv)
        )
        fold = quantize.fold_v_bias(layer0.wo, layer0.bv, h, kv, d)
        assert np.array_equal(lay.wo.bias_q, numerics.to_fixed(fold, frac_x))
        assert np.any(lay.wo.bias_q != 0)
    else:
        assert np.all(lay.wqkv.bias_q == 0)
        assert np.all(lay.wo.bias_q == 0)


@pytest.mark.parametrize("heads,kv_heads", [(14, 2), (9, 3), (4, 4)])
def test_v_bias_fold_is_exact(heads: int, kv_heads: int) -> None:
    """o_proj(sum p (v + b_v)) == o_proj(sum p v) + W_o @ tile(b_v) because the p sum to one."""
    rng = np.random.default_rng(1234)
    d, hidden, t = 64, 128, 17
    n_rep = heads // kv_heads
    w_o = rng.standard_normal((hidden, heads * d))
    v = rng.standard_normal((t, kv_heads, d))
    b_v = rng.standard_normal((kv_heads, d)) * 20
    p = rng.random((heads, t))
    p /= p.sum(axis=1, keepdims=True)
    ctx = np.zeros((heads, d))
    ctx_b = np.zeros((heads, d))
    for hh in range(heads):
        g = hh // n_rep
        ctx[hh] = p[hh] @ v[:, g]
        ctx_b[hh] = p[hh] @ (v[:, g] + b_v[g])
    fold = quantize.fold_v_bias(w_o, b_v.reshape(-1), heads, kv_heads, d)
    lhs = w_o @ ctx_b.reshape(-1)
    rhs = w_o @ ctx.reshape(-1) + fold
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-9)
    tile = quantize.tile_v_bias(b_v.reshape(-1), heads, kv_heads, d).reshape(heads, d)
    for hh in range(heads):
        assert np.array_equal(tile[hh], b_v[hh // n_rep])


# --------------------------------------------------------------------------- Q/K smoothing


def _rope64(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """rotate_half RoPE on ``[T, heads, D]`` in float64 (``cos``/``sin`` are ``[T, D]``)."""
    return x * cos[:, None, :] + reference_np.rotate_half(x) * sin[:, None, :]


def _random_layer(rng: np.random.Generator, heads: int, kv_heads: int, d: int, hidden: int):
    z = np.zeros(1)
    return LayerWeights(
        norm_in=z,
        norm_post=z,
        wq=rng.standard_normal((heads * d, hidden)),
        wk=rng.standard_normal((kv_heads * d, hidden)),
        wv=rng.standard_normal((kv_heads * d, hidden)),
        wo=z,
        bq=rng.standard_normal(heads * d) * 3,
        bk=rng.standard_normal(kv_heads * d) * 30,
        bv=rng.standard_normal(kv_heads * d),
        w_gate=z,
        w_up=z,
        w_down=z,
    )


@pytest.mark.parametrize("heads,kv_heads", [(14, 2), (9, 3), (4, 4)])
def test_qk_smoothing_fold_preserves_scores(heads: int, kv_heads: int) -> None:
    """Folded W_q/b_q, W_k/b_k give the same post-RoPE q . k for every (query head, key)."""
    rng = np.random.default_rng(99)
    d, half, hidden, t = 64, 32, 40, 11
    n_rep = heads // kv_heads
    w = _random_layer(rng, heads, kv_heads, d, hidden)
    s = np.clip(np.exp(rng.normal(size=(kv_heads, half)) * 1.5), 1 / 16, 16)
    ws = quantize.smooth_qk(w, s, heads, kv_heads, d)
    x = rng.standard_normal((t, hidden))
    cos, sin = (a.astype(np.float64) for a in reference_np.rope_cos_sin(np.arange(t), 1e4, d))

    def scores(lw: LayerWeights) -> np.ndarray:
        q = _rope64((x @ lw.wq.T + lw.bq).reshape(t, heads, d), cos, sin)
        k = _rope64((x @ lw.wk.T + lw.bk).reshape(t, kv_heads, d), cos, sin)
        return np.stack([q[:, hh] @ k[:, hh // n_rep].T for hh in range(heads)])

    ref = scores(w)
    assert np.allclose(scores(ws), ref, rtol=1e-12, atol=1e-9 * np.max(np.abs(ref)))
    # the factor is constant over a RoPE pair, so RoPE commutes with the fold
    s_k, s_q = calibrate.tile_factors(s, heads, kv_heads, d)
    q = (x @ w.wq.T + w.bq).reshape(t, heads, d)
    assert np.allclose(
        _rope64(q * s_q.reshape(heads, d), cos, sin), _rope64(q, cos, sin) * s_q.reshape(heads, d)
    )
    # the channel tiling: pair (d, d + 32) shares one factor, query heads repeat their KV head
    s_kd = s_k.reshape(kv_heads, d)
    assert np.array_equal(s_kd[:, :half], s) and np.array_equal(s_kd[:, half:], s)
    for hh in range(heads):
        assert np.array_equal(s_q.reshape(heads, d)[hh], s_kd[hh // n_rep])
    # the centering row follows K: (k / s) - (c / s) == (k - c) / s
    c = rng.standard_normal((kv_heads, d)) * 10
    k = (x @ w.wk.T + w.bk).reshape(t, kv_heads, d)
    ks = (x @ ws.wk.T + ws.bk).reshape(t, kv_heads, d)
    assert np.allclose(ks - c / s_kd, (k - c) / s_kd, rtol=1e-12, atol=1e-12)
    # V, o and the MLP are untouched; a model without QKV biases keeps them absent
    assert ws.wv is w.wv and ws.wo is w.wo and ws.w_down is w.w_down
    w0 = dataclasses.replace(w, bq=None, bk=None)
    ws0 = quantize.smooth_qk(w0, s, heads, kv_heads, d)
    assert ws0.bq is None and ws0.bk is None
    assert np.array_equal(ws0.wq, ws.wq) and np.array_equal(ws0.wk, ws.wk)
    with pytest.raises(ValueError):
        calibrate.tile_factors(s[:, : half - 1], heads, kv_heads, d)


def _tiny_spec(layers: int, heads: int, kv_heads: int) -> ModelSpec:
    return ModelSpec(
        name="tiny",
        repo_id="test/tiny",
        arch="llama",
        layers=layers,
        hidden=64,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=64,
        intermediate=128,
        vocab=256,
        has_qkv_bias=False,
        rope_theta=1e4,
        rms_norm_eps=1e-5,
        tied_embeddings=True,
        max_position_embeddings=64,
    )


def test_qk_smoothing_factor_rule_synthetic() -> None:
    """The factor rule on a recorder with known post-RoPE Q/K: formula, normalization, cap."""
    spec = _tiny_spec(layers=2, heads=4, kv_heads=2)
    h, kv, d, half = spec.heads, spec.kv_heads, spec.head_dim, 32
    rng = np.random.default_rng(5)
    rec = calibrate.CalibrationRecorder(spec)
    blocks = []
    for t in (7, 9):
        rec.begin_sequence()
        for layer in range(spec.layers):
            q = rng.standard_normal((t, h * d)).astype(np.float32)
            k = (rng.standard_normal((t, kv * d)) * 4 + 20).astype(np.float32)
            if layer == 1:  # plant a huge K range on pair 3 and a huge Q range on pair 7
                k[:, 3] *= 1e4
                q[:, 7] *= 1e4
            rec.record("q_rope", layer, q)
            rec.record("k_rope", layer, k)
            blocks.append((layer, q.astype(np.float64), k.astype(np.float64)))
    centers = calibrate.k_center_rows(rec)
    cap = 16.0
    f = calibrate.qk_smoothing_factors(rec, centers, alpha=0.5, cap=cap)
    assert f.shape == (spec.layers, kv, half)
    assert np.all(f >= 1 / cap) and np.all(f <= cap)
    assert np.array_equal(f, calibrate.qk_smoothing_factors(rec, centers, alpha=0.5, cap=cap))
    for layer in range(spec.layers):
        qs = np.concatenate([q for lay, q, _ in blocks if lay == layer])
        ks = np.concatenate([k for lay, _, k in blocks if lay == layer])
        q_max = np.abs(qs).reshape(-1, kv, h // kv, d).max(axis=(0, 2))
        kc_max = np.abs(ks.reshape(-1, kv, d) - centers[layer][None]).max(axis=0)
        pq = np.maximum(q_max[:, :half], q_max[:, half:])
        pk = np.maximum(kc_max[:, :half], kc_max[:, half:])
        s = np.sqrt(pk / pq)
        s = s / np.exp(np.mean(np.log(s), axis=1, keepdims=True))
        s = np.clip(s, 1 / cap, cap)
        want = np.vectorize(calibrate.round_sig)(s)
        assert np.array_equal(f[layer], want), layer
    assert np.all(np.abs(np.log(f[0]).mean(axis=1)) < LOG_GEOMEAN_TOL)  # unclipped rows
    assert f[1, 0, 3] == cap and f[1, 0, 7] == 1 / cap  # the planted pairs hit the cap


def test_qk_smoothing_factors_in_calib(spec: ModelSpec, calib: dict, factors: np.ndarray) -> None:
    """Shape, range, six-digit storage, geometric-mean normalization and the recorded rule."""
    qk = calib["qk_smoothing"]
    half = spec.head_dim // 2
    cap, alpha = qk["cap"], qk["alpha"]
    assert (alpha, cap) == (calibrate.QK_SMOOTH_ALPHA, calibrate.QK_SMOOTH_CAP)
    assert qk["digits"] == calibrate.FLOAT_DIGITS
    assert factors.shape == (spec.layers, spec.kv_heads, half)
    assert np.all(np.isfinite(factors)) and np.all(factors >= 1 / cap) and np.all(factors <= cap)
    assert qk["factor_min"] == factors.min() and qk["factor_max"] == factors.max()
    assert all(calibrate.round_sig(v) == v for v in factors.ravel())
    unclipped = (factors.min(axis=2) > 1 / cap) & (factors.max(axis=2) < cap)
    assert np.all(np.abs(np.log(factors).mean(axis=2)[unclipped]) < LOG_GEOMEAN_TOL)
    before = qk["absmax_unsmoothed"]
    assert set(before) == {"QKV", "K_centered"}
    assert before["K_centered"] > 0 and before["QKV"] >= before["K_centered"]
    assert calib["absmax"]["QKV"] >= calib["absmax"]["K_centered"] > 0
    assert quantize.smoothing_factors(calib, spec).shape == factors.shape


def test_smoothing_factors_validation(spec: ModelSpec, calib: dict) -> None:
    bad = dict(calib)
    del bad["qk_smoothing"]
    with pytest.raises(ValueError):
        quantize.smoothing_factors(bad, spec)
    qk = calib["qk_smoothing"]
    wrong_shape = dict(calib, qk_smoothing=dict(qk, factors=qk["factors"][:-1]))
    with pytest.raises(ValueError):
        quantize.smoothing_factors(wrong_shape, spec)
    f = np.asarray(qk["factors"], dtype=np.float64)
    f[0, 0, 0] = qk["cap"] * 2
    with pytest.raises(ValueError):
        quantize.smoothing_factors(dict(calib, qk_smoothing=dict(qk, factors=f.tolist())), spec)


def test_k_center(
    spec: ModelSpec, qmodel: quantize.QuantModel, calib: dict, factors: np.ndarray
) -> None:
    """The K-centering rows are the calibration means divided by the per-channel factors."""
    assert qmodel.k_center.shape == (TEST_LAYERS, spec.kv_heads, spec.head_dim)
    assert qmodel.k_center.dtype == np.int32
    rows = np.asarray(calib["k_center"], dtype=np.float64)
    assert rows.shape == (spec.layers, spec.kv_heads, spec.head_dim)
    per_channel = calibrate.pair_to_channels(factors[:TEST_LAYERS])
    assert per_channel.shape == (TEST_LAYERS, spec.kv_heads, spec.head_dim)
    want = numerics.to_fixed(rows[:TEST_LAYERS] / per_channel, qmodel.frac["QKV"])
    assert np.array_equal(qmodel.k_center, want)
    assert qmodel.extra["qk_smoothing"] == {
        "alpha": calib["qk_smoothing"]["alpha"],
        "cap": calib["qk_smoothing"]["cap"],
        "factor_min": float(factors.min()),
        "factor_max": float(factors.max()),
    }


# --------------------------------------------------------------------------- the rest


def _check_gamma(norm: quantize.QuantNorm, gamma: np.ndarray) -> None:
    assert norm.gamma_q.dtype == np.int16
    assert norm.gamma_e <= 0
    q, e = numerics.quantize_gamma(gamma)
    assert e == norm.gamma_e
    assert np.array_equal(q, norm.gamma_q.astype(np.int64))
    assert np.max(np.abs(norm.dequant() - gamma.astype(np.float64))) <= 2.0 ** (norm.gamma_e - 1)


def test_gamma_roundtrip(
    spec: ModelSpec, qmodel: quantize.QuantModel, layer0: LayerWeights
) -> None:
    _check_gamma(qmodel.layers[0].norm_in, layer0.norm_in)
    _check_gamma(qmodel.layers[0].norm_post, layer0.norm_post)
    from quettos.reference_np import load_final_norm

    _check_gamma(qmodel.norm_final, load_final_norm(spec))


def test_constants(spec: ModelSpec, qmodel: quantize.QuantModel) -> None:
    frac_x = qmodel.frac["X"]
    eps_c = numerics.eps_const(spec.rms_norm_eps, spec.hidden, frac_x)
    assert qmodel.eps_c == {"input": eps_c, "post": eps_c, "final": eps_c}
    assert qmodel.sqrt_d == numerics.sfloat_from_float(math.sqrt(spec.hidden))
    assert qmodel.log2e_over_8 == numerics.sfloat_from_float(math.log2(math.e) / 8)
    assert qmodel.frac["LOGITS"] == 16
    assert qmodel.frac["S"] >= 16
    assert qmodel.frac["GU"] >= 13
    assert qmodel.frac["H"] <= 2 * qmodel.frac["GU"]


def test_frac_headroom(calib: dict) -> None:
    """Every class keeps at least two bits of headroom: 2**(31 - FRAC) >= 4 * absmax."""
    frac = calib["frac"]
    absmax = calib["absmax"]
    owner = {
        "X": "X",
        "XN": "X",
        "QKV": "QKV",
        "S": "S",
        "GU": "GU",
        "H": "H",
        "CTX": "CTX",
        "S_centered": "S",
        "K_centered": "QKV",
    }
    owner["LOGITS"] = "LOGITS"
    for cls, a in absmax.items():
        f = frac[owner[cls]]
        assert 2.0 ** (31 - f) >= 4 * a, (cls, f, a)
    for cls in ("X", "QKV", "S", "GU", "H", "CTX"):
        assert frac[cls] <= 16


def test_frac_rule() -> None:
    assert calibrate.frac_for_absmax(0.0) == 16
    assert calibrate.frac_for_absmax(1.0) == 16
    assert calibrate.frac_for_absmax(25982.0) == 14
    assert calibrate.frac_for_absmax(8192.0) == 16
    assert calibrate.frac_for_absmax(8192.5) == 15
    for a in np.logspace(-3, 8, 200):
        f = calibrate.frac_for_absmax(float(a))
        assert 2.0 ** (31 - f) >= 4 * a
        assert 0 <= f <= 16


def test_save_load_roundtrip(qmodel: quantize.QuantModel, tmp_path: Path) -> None:
    path = quantize.save(qmodel, tmp_path / "q.npz")
    back = quantize.load(path)
    assert quantize.models_equal(qmodel, back)
    assert back.n_layers == TEST_LAYERS
    assert back.embed.q.dtype == np.int8
    assert back.norm_final.gamma_q.dtype == np.int16
    assert back.k_center.dtype == np.int32
    assert back.extra["qk_smoothing"] == qmodel.extra["qk_smoothing"]
    with np.load(path, allow_pickle=False) as z:
        man = json.loads(str(z["manifest"]))
    assert man["format"] == "quettos-quant"
    assert man["numerics"] == calibrate.NUMERICS_VERSION
    assert man["layers"] == TEST_LAYERS


def test_determinism(spec: ModelSpec, calib: dict, qmodel: quantize.QuantModel) -> None:
    again = quantize.build_quant_model(spec, calib, layers=TEST_LAYERS)
    assert quantize.models_equal(qmodel, again)


def test_rejects_foreign_calib(spec: ModelSpec, calib: dict) -> None:
    other = dict(calib)
    other["model"] = dict(calib["model"], repo_id="someone/else")
    with pytest.raises(ValueError):
        quantize.build_quant_model(spec, other, layers=1)


# --------------------------------------------------------------------------- calib.json


def test_calib_json_is_canonical(spec: ModelSpec, calib: dict) -> None:
    """The checked-in file is exactly what the writer emits for its own content, and small."""
    path = calibrate.calib_path(spec)
    text = path.read_text(encoding="utf-8")
    assert calibrate.calib_json_text(calib) == text
    assert path.stat().st_size < CALIB_MAX_BYTES
    assert calib["numerics"] == calibrate.NUMERICS_VERSION
    assert calib["model"]["repo_id"] == spec.repo_id
    assert calib["model"]["layers"] == spec.layers
    assert len(calib["per_layer_absmax"]["X"]) == spec.layers
    assert len(calib["per_layer_absmax"]["H"]) == spec.layers
    assert set(calib["absmax"]) == set(calibrate.CLASSES) | {"S_centered", "K_centered"}
    assert set(calib["frac"]) == set(calibrate.FRAC_CLASSES)
    assert set(calib["qk_smoothing"]) == {
        "alpha",
        "cap",
        "digits",
        "pair",
        "rule",
        "fold",
        "factors",
        "factor_min",
        "factor_max",
        "absmax_unsmoothed",
    }
    assert "timestamp" not in text.lower()


def test_calib_tokens_hash(spec: ModelSpec, calib: dict) -> None:
    seqs = calibrate.calibration_sequences(spec)
    assert calib["tokens"]["sequences"] == [len(s) for s in seqs]
    assert calib["tokens"]["count"] == sum(len(s) for s in seqs)
    assert calib["tokens"]["sha256"] == calibrate.token_ids_sha256(seqs)
    assert 900 <= calib["tokens"]["count"] <= 1500


def test_calib_gates_present(calib: dict) -> None:
    gate = calib["k_centering_gate"]
    assert 0 < gate["rel_rms_error_centered"] < gate["rel_rms_error_raw"] < 0.05
    assert 0 < gate["rms_error_log2_centered"] < gate["rms_error_log2_raw"]
    assert 0 < gate["rel_rms_error_smoothed"] < 0.05
    for v in calibrate.GATE_VARIANTS:
        assert 0 < gate[f"rms_error_log2_{v}"] <= gate[f"max_error_log2_{v}"]
    layers = calib["model"]["layers"]
    for v in ("centered", "smoothed"):
        per_layer = gate[f"layer_rms_error_log2_{v}"]
        assert len(per_layer) == layers and all(x > 0 for x in per_layer)
        assert max(per_layer) >= gate[f"rms_error_log2_{v}"] >= min(per_layer)
    spread = calib["v_scale_spread"]
    assert spread["p50"] <= spread["p99"] <= spread["max"]
    assert sum(spread["histogram"].values()) == spread["count"]


def _close(a, b, rel: float = 1e-4) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], rel) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, rel) for x, y in zip(a, b, strict=True))
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(a - b) <= rel * max(1.0, abs(a), abs(b))
    return a == b


def test_recalibration_matches_checked_in_file(spec: ModelSpec) -> None:
    """A fresh calibration pass reproduces models/<name>/calib.json.

    Integers, formats, token hashes and histograms must match exactly; floats
    (the smoothing factors included) may differ by BLAS rounding across
    machines (relative 1e-4).
    """
    fresh = json.loads(calibrate.calib_json_text(calibrate.calibrate(spec)))
    stored = calibrate.load_calib(calibrate.calib_path(spec))
    assert fresh["frac"] == stored["frac"]
    assert fresh["tokens"] == stored["tokens"]
    assert fresh["v_scale_spread"] == stored["v_scale_spread"]
    assert _close(fresh, stored), "calibration drifted from the checked-in file"
