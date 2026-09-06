"""Synthetic tiny models: the checkpoint round-trips through the safetensors reader, the
calibration report feeds the quantizer, the golden model runs bit-for-bit deterministically on
every fixed shape and on random shapes, and one build is a pure function of (shape, seed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from quettos import golden, program, quantize, reference_np, safetensors_np, synthetic
from quettos.calibrate import canonical_json_text
from quettos.model import expected_tensor_shapes
from quettos.numerics import Stats
from quettos.synthetic import SHAPES, Shape

T = 10  # tokens per golden run


@pytest.fixture(scope="module")
def out_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("synthetic")


def _ids(shape: Shape, seed: int, n: int = T) -> list[int]:
    return np.random.default_rng(seed + 100).integers(0, shape.vocab, n).tolist()


@pytest.mark.parametrize("index", range(len(SHAPES)))
def test_fixed_shape_builds_and_golden_runs(index: int, out_root: Path) -> None:
    shape = SHAPES[index]
    m = synthetic.build(shape, seed=index, out_dir=out_root / f"fixed{index}")
    spec, q = m.spec, m.quant
    assert (spec.layers, spec.hidden, spec.heads, spec.kv_heads) == (
        shape.layers,
        shape.hidden,
        shape.heads,
        shape.kv_heads,
    )
    assert spec.head_dim == 64 and spec.has_qkv_bias == shape.has_qkv_bias
    assert q.n_layers == shape.layers and q.vocab == shape.vocab and q.hidden == shape.hidden
    assert q.embed.q.shape == (shape.vocab, shape.hidden)
    assert q.layers[0].wqkv.q.shape == ((shape.heads + 2 * shape.kv_heads) * 64, shape.hidden)
    assert q.k_center.shape == (shape.layers, shape.kv_heads, 64)
    assert q.calib_tokens_sha256 == m.calib["tokens"]["sha256"]
    assert q.extra["absmax"] == m.calib["absmax"] and q.frac == m.calib["frac"]
    assert q.frac["S"] >= 16 and q.frac["GU"] >= 13
    if shape.has_qkv_bias:
        assert np.any(q.layers[0].wqkv.bias_q != 0) and np.any(q.layers[0].wo.bias_q != 0)
    else:
        assert np.all(q.layers[0].wqkv.bias_q == 0) and np.all(q.layers[0].wo.bias_q == 0)

    # the written checkpoint is exactly the float32 arrays handed out
    path = spec.path("model.safetensors")
    for name, arr in synthetic.tensors_for(spec, m.layers, m.embedding, m.final_norm).items():
        back = safetensors_np.load_tensor(path, name)
        assert back.dtype == np.float32 and np.array_equal(back, arr), name
    assert safetensors_np.tensor_shapes(path) == expected_tensor_shapes(spec)

    # the golden model runs with clean counters and the program constants are in the window
    pc = program.build(q)
    for g in pc.gemvs.values():
        g.check_window()
    ids = _ids(shape, index)
    st = Stats()
    out = golden.forward_tokens(q, ids, stats=st, max_ctx=synthetic.MAX_CTX)
    assert out.logits.shape == (T, shape.vocab) and out.cache.length == T
    assert st.err_shift == 0 and st.sat == 0
    ref = reference_np.forward(spec, ids)
    assert ref.shape == (T, shape.vocab)
    # sequential steps reproduce the teacher-forced forward on this shape
    cache = golden.new_cache(q, max_ctx=synthetic.MAX_CTX)
    for pos, tok in enumerate(ids):
        assert golden.step(q, cache, tok, pos, prog=pc) == int(out.argmax[pos]), pos
    assert np.array_equal(cache.k, out.cache.k) and np.array_equal(cache.v, out.cache.v)


def test_build_is_a_function_of_shape_and_seed(out_root: Path) -> None:
    shape = SHAPES[1]
    a = synthetic.build(shape, seed=3, out_dir=out_root / "a")
    b = synthetic.build(shape, seed=3, out_dir=out_root / "b")
    assert a.spec.name == b.spec.name and a.sequences == b.sequences
    assert a.calib == b.calib
    assert all(np.array_equal(x.wq, y.wq) for x, y in zip(a.layers, b.layers, strict=True))
    assert np.array_equal(a.embedding, b.embedding)
    assert quantize.models_equal(a.quant, b.quant)
    ids = _ids(shape, 3)
    la = golden.forward_tokens(a.quant, ids, max_ctx=synthetic.MAX_CTX).logits
    lb = golden.forward_tokens(b.quant, ids, max_ctx=synthetic.MAX_CTX).logits
    assert np.array_equal(la, lb)
    c = synthetic.build(shape, seed=4, out_dir=out_root / "c")
    assert c.spec.name != a.spec.name and not np.array_equal(c.embedding, a.embedding)
    assert not quantize.models_equal(a.quant, c.quant)


def test_default_output_directory() -> None:
    m = synthetic.build(SHAPES[0], seed=0)
    assert m.spec.model_dir == synthetic.model_dir_for(SHAPES[0], 0)
    assert m.spec.model_dir.is_relative_to(synthetic.SYNTHETIC_DIR)
    assert m.spec.path("model.safetensors").is_file()
    assert m.spec.name.startswith("synthetic-l1-h64-a1-kv1-i128-v128-n-t1e5-s0")


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_random_shapes(seed: int, out_root: Path) -> None:
    rng = np.random.default_rng(seed)
    shape = synthetic.random_shape(rng)
    assert shape.hidden in synthetic.HIDDEN_CHOICES and shape.vocab in synthetic.VOCAB_CHOICES
    assert shape.kv_heads in synthetic.KV_HEAD_CHOICES and shape.layers in (1, 2)
    assert shape.heads % shape.kv_heads == 0 and shape.rope_theta in synthetic.ROPE_THETAS
    m = synthetic.build(shape, seed=seed, out_dir=out_root / f"random{seed}")
    st = Stats()
    out = golden.forward_tokens(m.quant, _ids(shape, seed), stats=st, max_ctx=synthetic.MAX_CTX)
    assert out.logits.shape == (T, shape.vocab) and st.err_shift == 0 and st.sat == 0


def test_shape_validation() -> None:
    with pytest.raises(ValueError):
        Shape(heads=1, kv_heads=2)
    with pytest.raises(ValueError):
        Shape(heads=3, kv_heads=2)
    with pytest.raises(ValueError):
        Shape(hidden=60)
    with pytest.raises(ValueError):
        Shape(rope_theta=1e4)
    with pytest.raises(ValueError):
        Shape(layers=0)
    with pytest.raises(ValueError):
        Shape(vocab=1)
    assert Shape().name(5).endswith("-s5")


def test_calibration_report_carries_what_the_quantizer_reads(out_root: Path) -> None:
    m = synthetic.build(SHAPES[2], seed=9, out_dir=out_root / "calib")
    calib = m.calib
    assert calib["numerics"] == 1 and calib["model"]["repo_id"] == m.spec.repo_id
    assert set(calib["frac"]) == {"X", "QKV", "S", "GU", "H", "CTX", "LOGITS"}
    assert set(calib["absmax"]) >= {"X", "XN", "QKV", "S", "GU", "H", "CTX", "LOGITS"}
    assert "S_centered" in calib["absmax"] and "K_centered" in calib["absmax"]
    factors = np.asarray(calib["qk_smoothing"]["factors"])
    assert factors.shape == (m.spec.layers, m.spec.kv_heads, 32)
    assert np.all(factors >= 1 / 16) and np.all(factors <= 16)
    assert np.asarray(calib["k_center"]).shape == (m.spec.layers, m.spec.kv_heads, 64)
    assert calib["tokens"]["sequences"] == [len(s) for s in m.sequences]
    # the report is already rounded the way calib.json is written
    assert json.loads(canonical_json_text(calib)) == calib
