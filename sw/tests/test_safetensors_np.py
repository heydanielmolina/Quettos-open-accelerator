"""Numpy bf16 safetensors reader: exact decode, shapes, parameter counts."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from quettos.model import ModelSpec, expected_tensor_shapes
from quettos.safetensors_np import (
    bf16_bits_to_f32,
    iter_tensors,
    load_tensor,
    parameter_count,
    read_header,
    tensor_shapes,
)

EXPECTED_PARAMS = {"qwen2": 494_032_768, "llama": 134_515_008}


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bf16 bit pattern (reference for the test)."""
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def test_bf16_decode_exhaustive() -> None:
    """All 65,536 bf16 bit patterns decode to the float32 with the same top 16 bits."""
    bits = np.arange(2**16, dtype=np.uint16)
    f = bf16_bits_to_f32(bits)
    assert f.dtype == np.float32
    assert np.array_equal(f.view(np.uint32) >> 16, bits.astype(np.uint32))
    assert np.array_equal(f.view(np.uint32) & 0xFFFF, np.zeros(2**16, dtype=np.uint32))
    finite = np.isfinite(f)
    assert np.array_equal(_bf16_bits(f[finite]), bits[finite])


def test_synthetic_file_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 5)).astype(np.float32)
    a_bits = _bf16_bits(a)
    b = np.arange(6, dtype=np.float32).reshape(2, 3)
    header = {
        "__metadata__": {"format": "pt"},
        "a": {"dtype": "BF16", "shape": [3, 5], "data_offsets": [0, a_bits.nbytes]},
        "b": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [a_bits.nbytes, a_bits.nbytes + b.nbytes],
        },
    }
    hjson = json.dumps(header).encode()
    path = tmp_path / "t.safetensors"
    path.write_bytes(struct.pack("<Q", len(hjson)) + hjson + a_bits.tobytes() + b.tobytes())

    hdr, start = read_header(path)
    assert set(hdr) == {"a", "b"} and start == 8 + len(hjson)
    assert tensor_shapes(path) == {"a": (3, 5), "b": (2, 3)}
    assert parameter_count(path) == 21
    got_a = load_tensor(path, "a")
    assert got_a.shape == (3, 5) and np.array_equal(_bf16_bits(got_a), a_bits)
    assert np.array_equal(load_tensor(path, "b"), b)
    assert [n for n, _ in iter_tensors(path)] == ["a", "b"]
    with pytest.raises(KeyError):
        load_tensor(path, "missing")


def test_real_checkpoint_shapes_and_params(spec: ModelSpec) -> None:
    path = spec.path("model.safetensors")
    header, _ = read_header(path)
    assert {info["dtype"] for info in header.values()} == {"BF16"}
    assert tensor_shapes(path) == expected_tensor_shapes(spec)
    assert parameter_count(path) == EXPECTED_PARAMS[spec.arch]
    total = 0
    for name, arr in iter_tensors(path):
        assert arr.dtype == np.float32
        assert arr.shape == expected_tensor_shapes(spec)[name]
        total += arr.size
    assert total == EXPECTED_PARAMS[spec.arch]


def test_real_checkpoint_matches_torch(spec: ModelSpec) -> None:
    """Spot-check the bf16 decode against safetensors.torch (ref group only)."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    path = spec.path("model.safetensors")
    ref = load_file(str(path))
    names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
        f"model.layers.{spec.layers - 1}.self_attn.o_proj.weight",
    ]
    if spec.has_qkv_bias:
        names += ["model.layers.0.self_attn.k_proj.bias", "model.layers.5.self_attn.v_proj.bias"]
    for name in names:
        ours = load_tensor(path, name)
        theirs = ref[name].to(torch.float32).numpy()
        assert ours.shape == theirs.shape
        assert np.array_equal(ours, theirs), name
