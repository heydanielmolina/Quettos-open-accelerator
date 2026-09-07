"""Quality measurement: the metric definitions on synthetic logits, the SmolLM2 W8A16 gate on
the calibration set, the row naming and merging that keep the shipped and ``-nosmooth`` rows in
one file, and agreement of the checked-in ``quality.json`` files with a fresh evaluation
(``slow`` for the complete models from ``build/quant/``)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from quettos import calibrate, numerics, quality, quantize
from quettos.model import ModelSpec

GATE_KL_NATS = 0.02
GATE_TOP1_PERCENT = 93.0
REL = 1e-4  # float tolerance between a stored report and a fresh evaluation


# --------------------------------------------------------------------------- metric definitions


def _softmax64(z: np.ndarray) -> np.ndarray:
    e = np.exp(np.asarray(z, dtype=np.float64) - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def test_log_softmax_normalizes_and_is_shift_invariant() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(size=(5, 100)) * 10.0
    lp = quality.log_softmax(z.astype(np.float32))
    assert lp.dtype == np.float64 and lp.shape == z.shape
    assert np.allclose(np.sum(np.exp(lp), axis=-1), 1.0, atol=1e-12)
    assert np.allclose(quality.log_softmax(z + 1000.0), quality.log_softmax(z), atol=1e-9)
    assert np.allclose(np.exp(lp), _softmax64(z.astype(np.float32)), atol=1e-15)


def test_score_sequence_on_identical_distributions() -> None:
    """Integer logits whose real values the float32 reference holds exactly: zero KL, equal NLL."""
    rng = np.random.default_rng(2)
    t, v = 9, 50
    ints = rng.integers(-(1 << 20), 1 << 20, (t, v)).astype(np.int32)
    ref = (ints.astype(np.float64) * 2.0**-16).astype(np.float32)  # exact: |ints| < 2**24
    ids = rng.integers(0, v, t).tolist()
    tm = quality.score_sequence(ref, ints, ids)
    assert tm.kl.shape == (t - 1,) and tm.match.all()
    assert np.max(np.abs(tm.kl)) < 1e-12
    assert np.array_equal(tm.nll_fp32, tm.nll_int)
    s = tm.summary()
    assert s["tokens"] == t - 1 and s["top1_matches"] == t - 1 and s["top1_percent"] == 100.0
    assert s["delta_nll"] == 0.0 and s["delta_nll_se"] == 0.0
    assert s["ppl_fp32"] == pytest.approx(math.exp(s["nll_fp32"]))
    assert s["ppl_int"] == s["ppl_fp32"]


def test_score_sequence_matches_direct_formulas_across_row_chunks() -> None:
    rng = np.random.default_rng(3)
    t, v = quality.ROW_CHUNK + 6, 30  # the scored rows span two chunks
    ref = (rng.normal(size=(t, v)) * 4.0).astype(np.float32)
    ints = rng.integers(-(30 << 16), 30 << 16, (t, v)).astype(np.int32)
    ids = rng.integers(0, v, t).tolist()
    tm = quality.score_sequence(ref, ints, ids)
    p = _softmax64(ref[:-1])
    q = _softmax64(ints[:-1].astype(np.float64) / 65536.0)
    rows = np.arange(t - 1)
    tgt = np.asarray(ids[1:])
    assert np.allclose(tm.kl, np.sum(p * (np.log(p) - np.log(q)), axis=-1), atol=1e-12)
    assert np.allclose(tm.nll_fp32, -np.log(p[rows, tgt]), atol=1e-12)
    assert np.allclose(tm.nll_int, -np.log(q[rows, tgt]), atol=1e-12)
    assert np.array_equal(tm.match, np.argmax(ref[:-1], -1) == np.argmax(ints[:-1], -1))
    assert np.all(tm.kl >= 0.0)
    # the golden's argmax rule (first maximal index) is the one used on the integer side
    tie = np.zeros((2, v), dtype=np.int32)
    tie[:, [3, 7]] = 5
    ref_tie = np.zeros((2, v), dtype=np.float32)
    ref_tie[:, 3] = 1.0
    tm_tie = quality.score_sequence(ref_tie, tie, [0, 1])
    assert numerics.argmax(tie[0]) == 3 and tm_tie.match[0]
    ref_tie[:, 7] = 2.0
    assert not quality.score_sequence(ref_tie, tie, [0, 1]).match[0]


def test_summary_statistics_by_hand() -> None:
    tm = quality.TokenMetrics(
        match=np.array([True, False, True, True]),
        kl=np.array([0.1, 0.2, 0.3, 0.4]),
        nll_fp32=np.array([1.0, 2.0, 3.0, 4.0]),
        nll_int=np.array([1.5, 2.0, 3.5, 4.0]),
    )
    s = tm.summary()
    assert s["tokens"] == 4 and s["top1_matches"] == 3 and s["top1_percent"] == 75.0
    assert s["kl_mean"] == pytest.approx(0.25)
    assert s["nll_fp32"] == pytest.approx(2.5) and s["nll_int"] == pytest.approx(2.75)
    assert s["delta_nll"] == pytest.approx(0.25)
    assert s["delta_nll_se"] == pytest.approx(np.std([0.5, 0.0, 0.5, 0.0], ddof=1) / 2.0)
    assert s["ppl_fp32"] == pytest.approx(math.exp(2.5))
    assert s["ppl_int"] == pytest.approx(math.exp(2.75))
    both = quality.TokenMetrics.concat([tm, tm]).summary()
    assert both["tokens"] == 8 and both["kl_mean"] == pytest.approx(0.25)
    assert both["delta_nll_se"] == pytest.approx(np.std([0.5, 0.0] * 4, ddof=1) / math.sqrt(8))
    one = quality.TokenMetrics(np.array([True]), np.array([0.0]), np.array([1.0]), np.array([1.0]))
    assert one.summary()["delta_nll_se"] == 0.0


def test_score_sequence_rejects_bad_input() -> None:
    ref = np.zeros((3, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        quality.score_sequence(ref, np.zeros((3, 6), dtype=np.int32), [0, 1, 2])
    with pytest.raises(ValueError):
        quality.score_sequence(ref, np.zeros((3, 5), dtype=np.int32), [0, 1])
    with pytest.raises(ValueError):
        quality.score_sequence(ref[:1], np.zeros((1, 5), dtype=np.int32), [0])
    with pytest.raises(ValueError):
        quality.TokenMetrics(np.zeros(0, bool), np.zeros(0), np.zeros(0), np.zeros(0)).summary()


def test_close_compares_floats_relatively_and_the_rest_exactly() -> None:
    assert quality.close({"a": 1.0, "b": [1, "x"]}, {"a": 1.00005, "b": [1, "x"]}, REL)
    assert not quality.close({"a": 1.0}, {"a": 1.001}, REL)
    assert quality.close(0.0, 0.0, REL) and not quality.close(0.0, 1e-6, REL)
    assert not quality.close({"a": 1}, {"a": 2}, REL)
    assert not quality.close({"a": 1}, {"b": 1}, REL)
    assert not quality.close([1.0], [1.0, 2.0], REL)
    assert not quality.close(True, 1, REL) and quality.close(True, True, REL)
    assert quality.close(3, 3.0, REL)  # an int stored where a float was computed


def test_quality_json_text_is_canonical() -> None:
    rep = {"z": {"kl": 0.123456789, "n": 3}, "a": [1.0000004, "s", True], "m": 1e-7 / 3}
    text = quality.quality_json_text(rep)
    assert text.endswith("\n") and text.index('"a"') < text.index('"m"') < text.index('"z"')
    back = json.loads(text)
    assert back["z"]["kl"] == 0.123457 and back["z"]["n"] == 3 and back["a"][0] == 1.0
    assert back["m"] == 3.33333e-08 and back["a"][2] is True
    assert quality.quality_json_text(back) == text


# --------------------------------------------------------------------------- rows


def _stub_model(*, smoothing: bool) -> quantize.QuantModel:
    """A model carrying only what the row naming reads: the smoothing flag of its build."""
    z = np.zeros(1)
    lin = quantize.QuantLinear(z, z, z, z)
    return quantize.QuantModel(
        name="stub",
        repo_id="test/stub",
        arch="llama",
        hidden=64,
        heads=1,
        kv_heads=1,
        head_dim=64,
        intermediate=64,
        vocab=8,
        has_qkv_bias=False,
        rms_norm_eps=1e-5,
        rope_theta=1e4,
        frac={"LOGITS": 16},
        eps_c={},
        sqrt_d=numerics.SFLOAT_ONE,
        log2e_over_8=numerics.SFLOAT_ONE,
        calib_tokens_sha256="0" * 64,
        layers=[],
        norm_final=quantize.QuantNorm(z, 0),
        embed=lin,
        k_center=z,
        extra={"qk_smoothing": {"enabled": smoothing}},
    )


def test_config_name_labels_the_build_the_row_came_from() -> None:
    smoothed, plain = _stub_model(smoothing=True), _stub_model(smoothing=False)
    assert [quality.config_name(smoothed, b) for b in (16, 8)] == ["W8A16", "W8A8"]
    assert [quality.config_name(plain, b) for b in (16, 8)] == ["W8A16-nosmooth", "W8A8-nosmooth"]
    assert set(quality.ROW_NAMES) == {"W8A16", "W8A8", "W8A16-nosmooth", "W8A8-nosmooth"}
    assert len(set(quality.ROW_NAMES)) == len(quality.ROW_NAMES)
    with pytest.raises(ValueError):
        quality.config_name(smoothed, 4)


def _stub_report(rows: dict) -> dict:
    return {
        "format": "quettos-quality",
        "numerics": 1,
        "model": {"repo_id": "test/stub", "name": "stub", "layers": 2},
        "calib_tokens_sha256": "ab",
        "protocol": {"ids_sha256": "ab", "tokens": 4, "logits_frac": 16},
        "rows": rows,
    }


def test_merge_quality_keeps_the_rows_of_earlier_runs(tmp_path) -> None:
    """A run of one build adds its rows and leaves every other row of the file as stored."""
    path = tmp_path / "quality.json"
    stored = _stub_report({"W8A16": {"kl_mean": 1.0}, "W8A16-nosmooth": {"kl_mean": 2.0}})
    path.write_text(quality.quality_json_text(stored), encoding="utf-8")
    fresh = _stub_report({"W8A16": {"kl_mean": 3.0}})
    merged = quality.merge_quality(fresh, path)
    assert merged["rows"] == {"W8A16": {"kl_mean": 3.0}, "W8A16-nosmooth": {"kl_mean": 2.0}}
    assert {k: v for k, v in merged.items() if k != "rows"} == {
        k: v for k, v in fresh.items() if k != "rows"
    }
    # the kept row is the stored one, so rewriting the file reproduces its text exactly
    kept = quality.quality_json_text(
        _stub_report({"W8A16-nosmooth": merged["rows"]["W8A16-nosmooth"]})
    )
    assert kept == quality.quality_json_text(
        _stub_report({"W8A16-nosmooth": stored["rows"]["W8A16-nosmooth"]})
    )
    # a report of another model, numerics, calibration or protocol replaces the file
    for key, value in (
        ("numerics", 2),
        ("calib_tokens_sha256", "cd"),
        ("model", {"repo_id": "test/other", "name": "other", "layers": 2}),
        ("protocol", {"ids_sha256": "cd", "tokens": 4, "logits_frac": 16}),
    ):
        assert quality.merge_quality({**fresh, key: value}, path)["rows"] == fresh["rows"], key
    assert quality.merge_quality(fresh, tmp_path / "absent.json") is fresh


# --------------------------------------------------------------------------- SmolLM2 gate


@pytest.fixture(scope="session")
def smollm2_quality(smollm2: ModelSpec) -> tuple[quantize.QuantModel, list[list[int]], dict]:
    """Complete SmolLM2 from ``calib.json`` and its W8A16 row over the calibration set."""
    path = calibrate.calib_path(smollm2)
    if not path.is_file():
        pytest.skip(f"{path} not present")
    qmodel = quantize.build_quant_model(smollm2, path)
    seqs = calibrate.calibration_sequences(smollm2)
    return qmodel, seqs, quality.evaluate(smollm2, qmodel, seqs, a_bits=16)


def test_smollm2_w8a16_gate(smollm2_quality) -> None:
    """KL <= 0.02 nats and top-1 >= 93% against fp32 on the calibration set, no sat/err_shift."""
    qmodel, seqs, row = smollm2_quality
    print(
        f"{qmodel.name} {row['config']}: tokens {row['tokens']} top-1 {row['top1_percent']:.2f}% "
        f"KL {row['kl_mean']:.5f} delta-NLL {row['delta_nll']:+.5f} +/- {row['delta_nll_se']:.5f} "
        f"PPL {row['ppl_fp32']:.3f} -> {row['ppl_int']:.3f} {row['stats']}"
    )
    assert (
        row["tokens"] == sum(len(s) - 1 for s in seqs) == sum(q["tokens"] for q in row["sequences"])
    )
    assert row["stats"]["sat"] == 0 and row["stats"]["err_shift"] == 0
    assert row["kl_mean"] <= GATE_KL_NATS
    assert row["top1_percent"] >= GATE_TOP1_PERCENT
    assert row["ppl_int"] == pytest.approx(math.exp(row["nll_int"]))
    assert row["delta_nll"] == pytest.approx(row["nll_int"] - row["nll_fp32"])


@pytest.fixture(scope="session")
def smollm2_nosmooth(smollm2: ModelSpec, smollm2_quality) -> dict:
    """W8A16 row of the same model built with every Q/K smoothing factor forced to 1."""
    _, seqs, _ = smollm2_quality
    plain = quantize.build_quant_model(smollm2, calibrate.calib_path(smollm2), smoothing=False)
    assert not quantize.smoothing_enabled(plain)
    return quality.evaluate(smollm2, plain, seqs, a_bits=16)


def test_smollm2_nosmooth_row(smollm2_quality, smollm2_nosmooth) -> None:
    """The ablation is the same protocol on the same tokens against the same fp32 reference."""
    qmodel, seqs, shipped = smollm2_quality
    row = smollm2_nosmooth
    print(
        f"{qmodel.name} {row['config']}: tokens {row['tokens']} top-1 {row['top1_percent']:.2f}% "
        f"KL {row['kl_mean']:.5f} delta-NLL {row['delta_nll']:+.5f} +/- {row['delta_nll_se']:.5f} "
        f"PPL {row['ppl_fp32']:.3f} -> {row['ppl_int']:.3f} {row['stats']}"
    )
    assert row["config"] == "W8A16" + quantize.NOSMOOTH_SUFFIX
    assert row["a_bits"] == shipped["a_bits"] == 16
    assert row["tokens"] == shipped["tokens"] == sum(len(s) - 1 for s in seqs)
    assert row["nll_fp32"] == pytest.approx(shipped["nll_fp32"])  # one reference for both rows
    assert row["stats"]["sat"] == 0 and row["stats"]["err_shift"] == 0
    path = quality.quality_path(qmodel.name)
    if not path.is_file():
        pytest.skip(f"{path} not present (run: uv run quettos check smollm2 --no-qk-smoothing)")
    _assert_row_close(quality.load_quality(path)["rows"][row["config"]], row)


def test_smollm2_quality_json_matches_fresh_w8a16(smollm2_quality) -> None:
    qmodel, seqs, row = smollm2_quality
    path = quality.quality_path(qmodel.name)
    if not path.is_file():
        pytest.skip(f"{path} not present (run: uv run quettos check smollm2)")
    stored = quality.load_quality(path)
    assert stored["format"] == "quettos-quality"
    assert stored["numerics"] == qmodel.numerics_version
    assert stored["calib_tokens_sha256"] == qmodel.calib_tokens_sha256
    assert stored["model"] == {
        "repo_id": qmodel.repo_id,
        "name": qmodel.name,
        "layers": qmodel.n_layers,
    }
    fresh = quality.report(qmodel, seqs, [row])
    assert stored["protocol"] == fresh["protocol"]
    assert set(stored["rows"]) == set(quality.ROW_NAMES)
    _assert_row_close(stored["rows"]["W8A16"], row)
    assert quality.quality_json_text(stored) == path.read_text(encoding="utf-8")


def _assert_row_close(stored: dict, fresh: dict) -> None:
    """Integers and strings exact; floats within 2e-3 absolute + 1e-3 relative.

    The fp32 reference is evaluated by whatever BLAS the machine has, which
    moves the shared-side metrics by a few 1e-4; the golden side is integer.
    """
    assert stored.keys() == fresh.keys()
    for key, a in stored.items():
        b = fresh[key]
        if isinstance(a, dict):
            _assert_row_close(a, b)
        elif isinstance(a, list):
            assert len(a) == len(b), key
            for x, y in zip(a, b, strict=True):
                if isinstance(x, dict):
                    _assert_row_close(x, y)
                elif isinstance(x, float) or isinstance(y, float):
                    assert abs(x - y) <= 2e-3 + 1e-3 * abs(y), (key, x, y)
                else:
                    assert x == y, key
        elif isinstance(a, float) or isinstance(b, float):
            assert abs(a - b) <= 2e-3 + 1e-3 * abs(b), (key, a, b)
        else:
            assert a == b, key


# --------------------------------------------------------------------------- complete models


@pytest.fixture(scope="session")
def qmodel_full(spec: ModelSpec) -> quantize.QuantModel:
    path = quantize.default_path(spec.name)
    if not path.is_file():
        pytest.skip(f"{path} not present (run: uv run quettos quantize)")
    model = quantize.load(path)
    if model.n_layers != spec.layers:
        pytest.skip(f"{path} is truncated to {model.n_layers} layers")
    return model


@pytest.fixture(scope="session")
def qmodel_nosmooth(spec: ModelSpec) -> quantize.QuantModel:
    """The K-centering-only build: ``build/quant/<name>-nosmooth.npz``, else built from calib."""
    path = quantize.default_path(spec.name, smoothing=False)
    if path.is_file():
        model = quantize.load(path)
        if model.n_layers == spec.layers:
            return model
    calib = calibrate.calib_path(spec)
    if not calib.is_file():
        pytest.skip(f"{calib} not present")
    return quantize.build_quant_model(spec, calib, smoothing=False)


@pytest.mark.slow
def test_quality_json_matches_fresh_evaluation(spec: ModelSpec, qmodel_full, qmodel_nosmooth):
    """Every row of the checked-in file agrees with a fresh evaluation of the build it names."""
    path = quality.quality_path(qmodel_full.name)
    if not path.is_file():
        pytest.skip(f"{path} not present (run: uv run quettos check)")
    stored = quality.load_quality(path)
    seqs = calibrate.calibration_sequences(spec)
    ref = quality.reference_logits(spec, seqs)
    rows = [
        quality.evaluate(spec, model, seqs, a_bits=b, ref=ref)
        for model in (qmodel_full, qmodel_nosmooth)
        for b in quality.CONFIGS
    ]
    fresh = quality.report(qmodel_full, seqs, rows)
    assert set(fresh["rows"]) == set(quality.ROW_NAMES)
    for row in rows:
        print(
            f"{spec.name} {row['config']}: top-1 {row['top1_percent']:.2f}% "
            f"KL {row['kl_mean']:.5f} delta-NLL {row['delta_nll']:+.5f} {row['stats']}"
        )
        assert row["stats"]["sat"] == 0 and row["stats"]["err_shift"] == 0
    assert quality.close(stored, fresh, REL)
    print(f"{spec.name}: text identical {quality.quality_json_text(fresh) == path.read_text()}")
