"""Quality measurement: the metric definitions on synthetic logits, the SmolLM2 W8A16 gates on
the calibration set and on the held-out WikiText-2 windows, the corpus fetch and its hashes, the
row naming and merging that keep the shipped and ``-nosmooth`` rows and the two sets in one file,
agreement of the checked-in ``quality.json`` files with a fresh evaluation (``slow`` for the
complete models from ``build/quant/``), and the provenance pass that reads those files back and
says which text each published row was scored on without scoring anything."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from quettos import calibrate, cli, corpus, numerics, quality, quantize, synthetic
from quettos.model import ModelSpec
from quettos.tokenizer_io import encode

GATE_KL_NATS = 0.02
GATE_TOP1_PERCENT = 93.0
REL = 1e-4  # float tolerance between a stored report and a fresh evaluation
HELDOUT_WINDOWS_FAST = 2  # windows the SmolLM2 held-out gate scores fresh
HELDOUT_WINDOWS_SLOW = 4  # windows every row of the held-out block is re-evaluated over


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
    # the calibration half of the document; the held-out half has its own test
    calibration_half = {k: v for k, v in stored.items() if k != quality.HELDOUT_KEY}
    assert quality.HELDOUT_KEY in stored
    assert quality.close(calibration_half, fresh, REL)
    text = quality.quality_json_text({**fresh, quality.HELDOUT_KEY: stored[quality.HELDOUT_KEY]})
    print(f"{spec.name}: text identical {text == path.read_text()}")


# --------------------------------------------------------------------------- held-out corpus


@pytest.fixture(scope="session")
def heldout_archive() -> Path:
    """The verified WikiText-2 archive, or a skip: the tests never download it."""
    try:
        return corpus.fetch_archive(download=False)
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"held-out archive unavailable: {exc}")


def test_cut_windows_slices_the_stream_without_overlap() -> None:
    ids = list(range(1000))
    w = corpus.cut_windows(ids, length=100, count=3)
    assert [len(x) for x in w] == [100, 100, 100]
    assert w[0] == list(range(100)) and w[2] == list(range(200, 300))
    assert [i for win in w for i in win] == list(range(300))  # contiguous, in order
    assert corpus.cut_windows(ids, length=100, count=3) == w  # a pure function of the stream
    for length, count in ((100, 11), (1, 1), (100, 0), (0, 1)):
        with pytest.raises(ValueError):
            corpus.cut_windows(ids, length=length, count=count)


def test_fetch_archive_checks_the_hash_of_what_it_reads(tmp_path) -> None:
    wrong = tmp_path / "wikitext-2-raw-v1.zip"
    wrong.write_bytes(b"not the archive")
    with pytest.raises(ValueError, match="sha256"):
        corpus.fetch_archive(path=wrong, download=False)
    with pytest.raises(FileNotFoundError):
        corpus.fetch_archive(path=tmp_path / "absent.zip", download=False)
    assert corpus.sha256_file(wrong) == hashlib.sha256(b"not the archive").hexdigest()


def test_heldout_windows_are_the_text_the_record_names(smollm2, heldout_archive) -> None:
    """The provenance block rebuilds the windows: archive hash, member hash, then the cut."""
    src = corpus.source_record(count=4)
    assert src["archive_sha256"] == corpus.WIKITEXT2_SHA256
    assert heldout_archive.stat().st_size == corpus.WIKITEXT2_BYTES == src["archive_bytes"]
    raw = corpus.split_bytes(src["split"])
    assert hashlib.sha256(raw).hexdigest() == src["member_sha256"]
    seqs = corpus.heldout_sequences(smollm2, count=4)
    assert [len(s) for s in seqs] == [corpus.WINDOW_TOKENS] * 4 == [src["window_tokens"]] * 4
    stream = encode(smollm2, raw.decode("utf-8"))
    assert [i for w in seqs for i in w] == list(stream[: 4 * corpus.WINDOW_TOKENS])
    with pytest.raises(ValueError):
        corpus.source_record(split="dev")


def test_heldout_text_is_none_of_the_text_the_quantizer_saw(qwen, heldout_archive) -> None:
    """No calibration passage or prompt is in the held-out split: the sets are disjoint text."""
    text = corpus.split_text()
    for passage in calibrate.CALIB_TEXTS:
        assert passage not in text
        assert passage[:120] not in text
    for name in calibrate.CALIB_PROMPT_FILES:
        prompt = json.loads((calibrate.PROMPTS_DIR / name).read_text(encoding="utf-8"))
        for message in prompt["messages"]:
            assert message["content"][:120] not in text
    calib_ids = calibrate.calibration_sequences(qwen)
    assert calibrate.token_ids_sha256(calib_ids) != calibrate.token_ids_sha256(
        corpus.heldout_sequences(qwen, count=2)
    )


# --------------------------------------------------------------------------- streamed rows


@pytest.fixture(scope="session")
def synthetic_model(tmp_path_factory) -> synthetic.SyntheticModel:
    return synthetic.build(synthetic.SHAPES[3], 7, out_dir=tmp_path_factory.mktemp("syn-quality"))


def test_evaluate_multi_is_evaluate_run_one_sequence_at_a_time(synthetic_model) -> None:
    """Streaming the reference gives the same rows as holding every sequence's logits at once."""
    spec, qmodel, seqs = synthetic_model.spec, synthetic_model.quant, synthetic_model.sequences
    builds = [(qmodel, 16), (qmodel, 8)]
    seen: list[tuple[int, int]] = []
    rows = quality.evaluate_multi(
        spec, builds, seqs, progress=lambda done, total, _s: seen.append((done, total))
    )
    assert seen == [(i, len(seqs)) for i in range(1, len(seqs) + 1)]
    one_at_a_time = [quality.evaluate(spec, qmodel, seqs, a_bits=b) for _, b in builds]
    assert [r["config"] for r in rows] == ["W8A16", "W8A8"]
    assert rows == one_at_a_time
    assert [len(r["sequences"]) for r in rows] == [len(seqs), len(seqs)]
    assert all(set(s) == set(quality.SEQ_KEYS) for r in rows for s in r["sequences"])


def test_evaluate_multi_rejects_builds_it_cannot_share_a_reference_with(synthetic_model) -> None:
    spec, qmodel, seqs = synthetic_model.spec, synthetic_model.quant, synthetic_model.sequences
    assert qmodel.n_layers == 2
    truncated = dataclasses.replace(qmodel, layers=qmodel.layers[:1])
    for builds, bad_seqs in (
        ([], seqs),
        ([(qmodel, 16)], []),
        ([(qmodel, 16), (qmodel, 16)], seqs),  # one row name twice
        ([(qmodel, 16), (truncated, 8)], seqs),  # different stacks, one reference
    ):
        with pytest.raises(ValueError):
            quality.evaluate_multi(spec, builds, bad_seqs)


# --------------------------------------------------------------------------- the two sets


_SOURCE = {
    "name": "wikitext-2-raw-v1",
    "url": "https://example.invalid/wikitext-2-raw-v1.zip",
    "archive_bytes": 4721645,
    "archive_sha256": "aa" * 32,
    "member": "wikitext-2-raw/wiki.test.raw",
    "member_sha256": "bb" * 32,
    "split": "test",
    "windows": 2,
    "window_tokens": 4,
}


def test_heldout_block_records_the_corpus_it_scored() -> None:
    model = _stub_model(smoothing=True)
    seqs = [[1, 2, 3, 4], [5, 6, 7, 8]]
    row = {"config": "W8A16", "kl_mean": 0.5}
    block = quality.heldout_block(model, seqs, [row], _SOURCE)
    assert block["rows"] == {"W8A16": row}
    assert block["protocol"]["source"] == _SOURCE
    assert block["protocol"]["tokens"] == 6
    assert block["protocol"]["ids_sha256"] == calibrate.token_ids_sha256(seqs)
    assert block["protocol"]["logits_frac"] == 16
    assert block["protocol"]["scoring"] == quality.PROTOCOL["scoring"]
    text = block["protocol"]["text"]
    assert "wiki.test.raw" in text and "2 non-overlapping windows of 4 tokens" in text
    doc = quality.report(model, [[1, 2], [3, 4]], [], heldout=block)
    assert doc[quality.HELDOUT_KEY] == block and doc["rows"] == {}
    assert doc["protocol"]["text"] == quality.PROTOCOL["text"]  # the sets keep their own protocol
    assert quality.HELDOUT_KEY not in quality.report(model, [[1, 2]], [])


def _stub_heldout(rows: dict, *, source: dict | None = None) -> dict:
    return {
        "protocol": {"text": "held out", "source": source or _SOURCE, "tokens": 6},
        "rows": rows,
    }


def test_merge_quality_keeps_the_two_sets_apart(tmp_path) -> None:
    """Each block merges on its own protocol; a run of one set never touches the other's rows."""
    path = tmp_path / "quality.json"
    stored = _stub_report({"W8A16": {"kl_mean": 1.0}})
    stored[quality.HELDOUT_KEY] = _stub_heldout(
        {"W8A16": {"kl_mean": 5.0}, "W8A8": {"kl_mean": 6.0}}
    )
    path.write_text(quality.quality_json_text(stored), encoding="utf-8")

    calib_run = _stub_report({"W8A16": {"kl_mean": 3.0}})
    merged = quality.merge_quality(calib_run, path)
    assert merged["rows"] == {"W8A16": {"kl_mean": 3.0}}
    assert merged[quality.HELDOUT_KEY] == stored[quality.HELDOUT_KEY]

    heldout_run = {**_stub_report({}), quality.HELDOUT_KEY: _stub_heldout({"W8A16": {"kl": 7.0}})}
    merged = quality.merge_quality(heldout_run, path)
    assert merged["rows"] == stored["rows"]
    assert merged[quality.HELDOUT_KEY]["rows"] == {"W8A16": {"kl": 7.0}, "W8A8": {"kl_mean": 6.0}}

    other_windows = {
        **_stub_report({}),
        quality.HELDOUT_KEY: _stub_heldout(
            {"W8A16": {"kl": 9.0}}, source={**_SOURCE, "windows": 3}
        ),
    }
    merged = quality.merge_quality(other_windows, path)
    assert merged["rows"] == stored["rows"]
    assert merged[quality.HELDOUT_KEY]["rows"] == {"W8A16": {"kl": 9.0}}

    # a report of another model replaces the file, held-out block and all
    replaced = quality.merge_quality({**calib_run, "numerics": 2}, path)
    assert replaced["rows"] == calib_run["rows"] and quality.HELDOUT_KEY not in replaced


# --------------------------------------------------------------------------- SmolLM2 held-out


def _heldout_rows(spec: ModelSpec, builds, count: int) -> list[dict]:
    return quality.evaluate_multi(spec, builds, corpus.heldout_sequences(spec, count=count))


def _stored_heldout(name: str) -> dict:
    path = quality.quality_path(name)
    if not path.is_file():
        pytest.skip(f"{path} not present (run: uv run quettos check)")
    stored = quality.load_quality(path)
    if quality.HELDOUT_KEY not in stored:
        pytest.skip(f"{path} carries no held-out block (run: uv run quettos check --heldout)")
    return stored[quality.HELDOUT_KEY]


def test_heldout_protocol_names_the_windows_it_scored(spec: ModelSpec, heldout_archive) -> None:
    """The stored provenance is the corpus this clone rebuilds, ids and all."""
    protocol = _stored_heldout(spec.name)["protocol"]
    source = protocol["source"]
    assert source == corpus.source_record(split=source["split"], count=source["windows"])
    seqs = corpus.heldout_sequences(spec, split=source["split"], count=source["windows"])
    assert protocol["ids_sha256"] == calibrate.token_ids_sha256(seqs)
    assert protocol["tokens"] == sum(len(s) - 1 for s in seqs)
    assert set(_stored_heldout(spec.name)["rows"]) == set(quality.ROW_NAMES)


@pytest.fixture(scope="session")
def smollm2_heldout(smollm2: ModelSpec, heldout_archive, smollm2_quality) -> dict:
    """W8A16 over the first :data:`HELDOUT_WINDOWS_FAST` held-out windows of SmolLM2."""
    qmodel, _, _ = smollm2_quality
    return _heldout_rows(smollm2, [(qmodel, 16)], HELDOUT_WINDOWS_FAST)[0]


def test_smollm2_w8a16_heldout_gate(smollm2_heldout) -> None:
    """The calibration-set gate, met on text the quantizer never saw."""
    row = smollm2_heldout
    print(
        f"smollm2 held-out {row['config']}: tokens {row['tokens']} "
        f"top-1 {row['top1_percent']:.2f}% KL {row['kl_mean']:.5f} "
        f"delta-NLL {row['delta_nll']:+.5f} +/- {row['delta_nll_se']:.5f} "
        f"PPL {row['ppl_fp32']:.3f} -> {row['ppl_int']:.3f} {row['stats']}"
    )
    assert row["tokens"] == HELDOUT_WINDOWS_FAST * (corpus.WINDOW_TOKENS - 1)
    assert row["stats"]["sat"] == 0 and row["stats"]["err_shift"] == 0
    assert row["kl_mean"] <= GATE_KL_NATS
    assert row["top1_percent"] >= GATE_TOP1_PERCENT
    assert row["delta_nll"] == pytest.approx(row["nll_int"] - row["nll_fp32"])


def test_smollm2_heldout_json_matches_fresh_windows(smollm2, smollm2_heldout) -> None:
    """The stored per-window values are what a fresh run of those windows produces."""
    stored = _stored_heldout(smollm2.name)["rows"]["W8A16"]
    _assert_row_close(
        {"sequences": stored["sequences"][:HELDOUT_WINDOWS_FAST]},
        {"sequences": smollm2_heldout["sequences"]},
    )


@pytest.mark.slow
def test_heldout_json_matches_fresh_windows(
    spec: ModelSpec, heldout_archive, qmodel_full, qmodel_nosmooth
):
    """Every held-out row of the checked-in file, over a prefix of the windows it names."""
    stored = _stored_heldout(spec.name)["rows"]
    builds = [(m, b) for m in (qmodel_full, qmodel_nosmooth) for b in quality.CONFIGS]
    rows = _heldout_rows(spec, builds, HELDOUT_WINDOWS_SLOW)
    assert {r["config"] for r in rows} == set(quality.ROW_NAMES)
    for row in rows:
        print(
            f"{spec.name} held-out {row['config']}: top-1 {row['top1_percent']:.2f}% "
            f"KL {row['kl_mean']:.5f} delta-NLL {row['delta_nll']:+.5f} {row['stats']}"
        )
        assert row["stats"]["sat"] == 0 and row["stats"]["err_shift"] == 0
        _assert_row_close(
            {"sequences": stored[row["config"]]["sequences"][:HELDOUT_WINDOWS_SLOW]},
            {"sequences": row["sequences"]},
        )


# --------------------------------------------------------------------------- provenance


def test_scored_sets_puts_the_held_out_block_first() -> None:
    rep = _stub_report({"W8A16": {"kl_mean": 1.0}})
    assert [name for name, _ in quality.scored_sets(rep)] == [quality.CALIB_SET]
    rep[quality.HELDOUT_KEY] = _stub_heldout({"W8A8": {"kl_mean": 2.0}})
    blocks = quality.scored_sets(rep)
    assert [name for name, _ in blocks] == [quality.HELDOUT_SET, quality.CALIB_SET]
    assert blocks[0][1] is rep[quality.HELDOUT_KEY]
    assert blocks[1][1] == {"protocol": rep["protocol"], "rows": rep["rows"]}


def _stored_report(spec: ModelSpec) -> dict:
    path = quality.quality_path(spec.name)
    if not path.is_file():
        pytest.skip(f"{path} not present (run: uv run quettos check)")
    stored = quality.load_quality(path)
    if quality.HELDOUT_KEY not in stored:
        pytest.skip(f"{path} carries no held-out block (run: uv run quettos check --heldout)")
    return stored


def test_provenance_rebuilds_the_text_every_published_row_was_scored_on(
    spec: ModelSpec, heldout_archive
) -> None:
    """Both sets of the checked-in file, ids and all, without running a model."""
    stored = _stored_report(spec)
    sets = quality.provenance(spec, stored, download=False)
    assert [s.name for s in sets] == [quality.HELDOUT_SET, quality.CALIB_SET]
    heldout, calib = sets
    for block in sets:
        assert set(block.rows) == set(quality.ROW_NAMES)
        assert block.ok, [c for c in block.checks if not c.ok]
        assert (
            block.tokens
            == stored[quality.HELDOUT_KEY if block.name == quality.HELDOUT_SET else "protocol"].get(
                "protocol", stored["protocol"]
            )["tokens"]
        )
    labels = [c.label for c in heldout.checks]
    assert labels == [
        "archive",
        "member",
        "the cut",
        "scored ids",
        "scored positions",
        "not the calibration ids",
    ]
    assert (
        corpus.WIKITEXT2_SHA256[:16] in dict((c.label, c.detail) for c in heldout.checks)["archive"]
    )
    assert "wiki.test.raw" in heldout.text and str(corpus.WINDOW_TOKENS) in heldout.text
    assert heldout.tokens == corpus.WINDOWS * (corpus.WINDOW_TOKENS - 1)
    assert [c.label for c in calib.checks] == [
        "scored ids",
        "scored positions",
        "sequence lengths",
        "the ids calibration saw",
    ]
    assert calib.tokens == sum(len(x) - 1 for x in calibrate.calibration_sequences(spec))


@pytest.mark.parametrize(
    ("doctor", "failing"),
    [
        (("heldout", "ids_sha256"), {"scored ids"}),
        (("heldout", "tokens"), {"scored positions"}),
        (("calib", "ids_sha256"), {"scored ids", "the ids calibration saw"}),
        (("calib", "tokens"), {"scored positions"}),
        (("calib", "sequences"), {"sequence lengths"}),
        (("report", "calib_tokens_sha256"), {"the ids calibration saw"}),
        (("source", "member_sha256"), {"member", "the cut"}),
        (("source", "archive_sha256"), {"archive", "the cut"}),
        # a record that lies about the cut still rebuilds an archive record of
        # its own: the ids it names are what catches it
        (("source", "windows"), {"scored ids", "scored positions"}),
    ],
)
def test_provenance_names_the_record_that_does_not_rebuild(
    smollm2: ModelSpec, heldout_archive, doctor, failing
) -> None:
    """A field changed in the file is one named failing check, and never a silent pass."""
    stored = json.loads(json.dumps(_stored_report(smollm2)))  # a copy to doctor
    where, field = doctor
    block = {
        "heldout": stored[quality.HELDOUT_KEY]["protocol"],
        "calib": stored["protocol"],
        "report": stored,
        "source": stored[quality.HELDOUT_KEY]["protocol"]["source"],
    }[where]
    block[field] = (
        2 if isinstance(block[field], int) else ["f" * 8] if field == "sequences" else "f" * 64
    )
    sets = quality.provenance(smollm2, stored, download=False)
    assert {c.label for s in sets for c in s.checks if not c.ok} == failing
    assert not all(s.ok for s in sets)


def test_provenance_catches_held_out_rows_that_name_the_calibration_ids(
    smollm2: ModelSpec, heldout_archive
) -> None:
    """The suspicion the command exists to settle, made true in the file and caught."""
    stored = json.loads(json.dumps(_stored_report(smollm2)))
    stored[quality.HELDOUT_KEY]["protocol"]["ids_sha256"] = stored["protocol"]["ids_sha256"]
    heldout = quality.provenance(smollm2, stored, download=False)[0]
    failed = {c.label for c in heldout.checks if not c.ok}
    assert failed == {"scored ids", "not the calibration ids"}


def test_provenance_command_ends_in_a_verdict(
    smollm2: ModelSpec, heldout_archive, capsys, tmp_path
) -> None:
    """``uv run quettos provenance`` over a good file and over a doctored one."""
    stored = _stored_report(smollm2)
    good = tmp_path / "quality.json"
    good.write_text(quality.quality_json_text(stored), encoding="utf-8")
    assert cli.main(["provenance", smollm2.repo_id, "--file", str(good), "--no-download"]) == 0
    out = capsys.readouterr().out
    assert "provenance: OK" in out and f"{len(quality.ROW_NAMES) * 2} published rows" in out
    assert stored[quality.HELDOUT_KEY]["protocol"]["ids_sha256"][:16] in out
    assert stored["protocol"]["ids_sha256"][:16] in out
    assert "FAIL" not in out

    doctored = json.loads(json.dumps(stored))
    doctored[quality.HELDOUT_KEY]["protocol"]["ids_sha256"] = "f" * 64
    bad = tmp_path / "doctored.json"
    bad.write_text(quality.quality_json_text(doctored), encoding="utf-8")
    assert cli.main(["provenance", smollm2.repo_id, "--file", str(bad), "--no-download"]) == 1
    out = capsys.readouterr().out
    assert "FAIL scored ids" in out and "provenance: FAILED -- 1 check(s)" in out

    assert cli.main(["provenance", smollm2.repo_id, "--file", str(tmp_path / "absent.json")]) == 1
