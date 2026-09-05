"""Quantizer and calibration file: exactness of every integer against the numerics primitives."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from quettos import calibrate, numerics, quantize
from quettos.model import ModelSpec
from quettos.reference_np import LayerWeights, load_embedding, load_layer

TEST_LAYERS = 2
CALIB_MAX_BYTES = 300_000


@pytest.fixture(scope="session")
def calib(spec: ModelSpec) -> dict:
    path = calibrate.calib_path(spec)
    if not path.is_file():
        pytest.skip(f"{path} not present")
    return calibrate.load_calib(path)


@pytest.fixture(scope="session")
def qmodel(spec: ModelSpec, calib: dict) -> quantize.QuantModel:
    return quantize.build_quant_model(spec, calib, layers=TEST_LAYERS)


@pytest.fixture(scope="session")
def layer0(spec: ModelSpec) -> LayerWeights:
    return load_layer(spec, 0)


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
    spec: ModelSpec, qmodel: quantize.QuantModel, layer0: LayerWeights
) -> None:
    lay = qmodel.layers[0]
    _check_int8_rows(lay.wqkv, np.concatenate([layer0.wq, layer0.wk, layer0.wv], axis=0))
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


def test_bias_q(spec: ModelSpec, qmodel: quantize.QuantModel, layer0: LayerWeights) -> None:
    lay = qmodel.layers[0]
    h, kv, d = spec.heads, spec.kv_heads, spec.head_dim
    frac_qkv, frac_x = qmodel.frac["QKV"], qmodel.frac["X"]
    assert lay.wqkv.bias_q.dtype == np.int32
    assert lay.wo.bias_q.dtype == np.int32
    # V rows never carry a bias in the QKV GEMV: it is folded into o_proj.
    assert np.all(lay.wqkv.bias_q[h * d :][kv * d :] == 0)
    if spec.has_qkv_bias:
        assert np.array_equal(lay.wqkv.bias_q[: h * d], numerics.to_fixed(layer0.bq, frac_qkv))
        assert np.array_equal(
            lay.wqkv.bias_q[h * d : h * d + kv * d], numerics.to_fixed(layer0.bk, frac_qkv)
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


def test_k_center(spec: ModelSpec, qmodel: quantize.QuantModel, calib: dict) -> None:
    assert qmodel.k_center.shape == (TEST_LAYERS, spec.kv_heads, spec.head_dim)
    assert qmodel.k_center.dtype == np.int32
    rows = np.asarray(calib["k_center"], dtype=np.float64)
    assert rows.shape == (spec.layers, spec.kv_heads, spec.head_dim)
    assert np.array_equal(
        qmodel.k_center, numerics.to_fixed(rows[:TEST_LAYERS], qmodel.frac["QKV"])
    )


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
    assert gate["rms_error_log2_centered"] <= gate["max_error_log2_centered"]
    assert gate["rms_error_log2_raw"] <= gate["max_error_log2_raw"]
    per_layer = gate["layer_rms_error_log2_centered"]
    assert len(per_layer) == calib["model"]["layers"] and all(v > 0 for v in per_layer)
    assert max(per_layer) >= gate["rms_error_log2_centered"] >= min(per_layer)
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
    may differ by BLAS rounding across machines (relative 1e-4).
    """
    fresh = json.loads(calibrate.calib_json_text(calibrate.calibrate(spec)))
    stored = calibrate.load_calib(calibrate.calib_path(spec))
    assert fresh["frac"] == stored["frac"]
    assert fresh["tokens"] == stored["tokens"]
    assert fresh["v_scale_spread"] == stored["v_scale_spread"]
    assert _close(fresh, stored), "calibration drifted from the checked-in file"
