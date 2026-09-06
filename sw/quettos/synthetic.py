"""Random tiny models for tests: float weights, calibration report and int8 :class:`QuantModel`.

:func:`build` draws a Llama-shaped decoder of a :class:`Shape` from a seed,
writes it as a float32 ``model.safetensors`` under ``build/synthetic/`` so
:mod:`quettos.reference_np`, :mod:`quettos.calibrate` and :mod:`quettos.quantize`
read it exactly like a downloaded checkpoint, calibrates it on random token
ids with the calibration rules and quantizes it; :func:`quettos.golden.forward_tokens`
runs on the result.  :data:`SHAPES` are the fixed test shapes and
:func:`random_shape` draws one (``docs/VERIFICATION.md``, random tiny shapes).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quettos import calibrate
from quettos.model import BUILD_DIR, ModelSpec, expected_tensor_shapes
from quettos.quantize import QuantModel, build_quant_model
from quettos.reference_np import LayerWeights

SYNTHETIC_DIR = BUILD_DIR / "synthetic"
HEAD_DIM = 64
ROPE_THETAS: tuple[float, ...] = (1e5, 1e6)  # the checked-in RoPE tables
HIDDEN_CHOICES: tuple[int, ...] = (64, 128, 192)
VOCAB_CHOICES: tuple[int, ...] = (128, 256)
KV_HEAD_CHOICES: tuple[int, ...] = (1, 2, 3)
N_REP_CHOICES: tuple[int, ...] = (1, 2, 3)
INTERMEDIATE_CHOICES: tuple[int, ...] = (128, 192, 256)
CALIB_LENGTHS: tuple[int, ...] = (16, 24, 32)  # random calibration sequences per model
MAX_CTX = 64  # positions a synthetic model is calibrated and tested at


@dataclass(frozen=True)
class Shape:
    """Architecture of a synthetic model; ``head_dim`` is fixed at 64 and embeddings are tied."""

    layers: int = 1
    hidden: int = 64
    heads: int = 1
    kv_heads: int = 1
    intermediate: int = 128
    vocab: int = 128
    has_qkv_bias: bool = False
    rope_theta: float = 1e5
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.layers < 1 or self.hidden < 8 or self.hidden % 8:
            raise ValueError("Shape: layers >= 1 and hidden a positive multiple of 8")
        if self.kv_heads < 1 or self.heads < self.kv_heads or self.heads % self.kv_heads:
            raise ValueError("Shape: heads must be a positive multiple of kv_heads")
        if self.intermediate < 8 or self.intermediate % 8 or self.vocab < 2:
            raise ValueError("Shape: intermediate a positive multiple of 8, vocab >= 2")
        if self.rope_theta not in ROPE_THETAS:
            raise ValueError(f"Shape: rope_theta must be one of {ROPE_THETAS}")
        if not self.eps > 0:
            raise ValueError("Shape: eps must be positive")

    def name(self, seed: int) -> str:
        bias = "b" if self.has_qkv_bias else "n"
        return (
            f"synthetic-l{self.layers}-h{self.hidden}-a{self.heads}-kv{self.kv_heads}"
            f"-i{self.intermediate}-v{self.vocab}-{bias}-t{self.rope_theta:.0e}-s{seed}"
        ).replace("e+0", "e")


SHAPES: tuple[Shape, ...] = (
    Shape(1, 64, 1, 1, 128, 128, False, 1e5, 1e-5),
    Shape(2, 128, 4, 2, 256, 256, True, 1e6, 1e-6),
    Shape(1, 192, 3, 3, 192, 128, True, 1e5, 1e-5),
    Shape(2, 64, 2, 1, 128, 256, False, 1e6, 1e-5),
    Shape(1, 128, 6, 3, 256, 128, False, 1e5, 1e-6),
)


def random_shape(rng: np.random.Generator) -> Shape:
    """A shape drawn from the tiny-shape sets: hidden 64-192, vocab 128-256, kv_heads 1-3."""
    kv = int(rng.choice(KV_HEAD_CHOICES))
    return Shape(
        layers=int(rng.integers(1, 3)),
        hidden=int(rng.choice(HIDDEN_CHOICES)),
        heads=kv * int(rng.choice(N_REP_CHOICES)),
        kv_heads=kv,
        intermediate=int(rng.choice(INTERMEDIATE_CHOICES)),
        vocab=int(rng.choice(VOCAB_CHOICES)),
        has_qkv_bias=bool(rng.integers(0, 2)),
        rope_theta=float(rng.choice(ROPE_THETAS)),
        eps=float(rng.choice([1e-5, 1e-6])),
    )


@dataclass
class SyntheticModel:
    """Everything :func:`build` produces: spec, float weights, calibration and the QuantModel."""

    shape: Shape
    seed: int
    spec: ModelSpec
    layers: list[LayerWeights]
    embedding: np.ndarray
    final_norm: np.ndarray
    sequences: list[list[int]]
    calib: dict
    quant: QuantModel


# --------------------------------------------------------------------------- weights


def spec_for(shape: Shape, seed: int, model_dir: Path) -> ModelSpec:
    return ModelSpec(
        name=shape.name(seed),
        repo_id=f"synthetic/{shape.name(seed)}",
        arch="llama",
        layers=shape.layers,
        hidden=shape.hidden,
        heads=shape.heads,
        kv_heads=shape.kv_heads,
        head_dim=HEAD_DIM,
        intermediate=shape.intermediate,
        vocab=shape.vocab,
        has_qkv_bias=shape.has_qkv_bias,
        rope_theta=shape.rope_theta,
        rms_norm_eps=shape.eps,
        tied_embeddings=True,
        max_position_embeddings=MAX_CTX,
        eos_ids=[],
        bos_id=None,
        model_dir=model_dir,
    )


def _normal(rng: np.random.Generator, shape: tuple[int, ...], std: float) -> np.ndarray:
    return (rng.standard_normal(shape) * std).astype(np.float32)


def draw_weights(shape: Shape, rng: np.random.Generator):
    """Float32 ``(layers, embedding, final_norm)`` with ``1/sqrt(fan_in)`` projection scales.

    Norm weights are ``1 + N(0, 0.1)``; with ``has_qkv_bias`` the K bias is
    drawn wide (``N(0, 2)``) so K-centering and the Q/K smoothing fold act.
    """
    h, d, inter = shape.hidden, HEAD_DIM, shape.intermediate
    q_dim, kv_dim = shape.heads * d, shape.kv_heads * d
    layers = []
    for _ in range(shape.layers):
        bias = shape.has_qkv_bias
        layers.append(
            LayerWeights(
                norm_in=(1.0 + _normal(rng, (h,), 0.1)).astype(np.float32),
                norm_post=(1.0 + _normal(rng, (h,), 0.1)).astype(np.float32),
                wq=_normal(rng, (q_dim, h), h**-0.5),
                wk=_normal(rng, (kv_dim, h), h**-0.5),
                wv=_normal(rng, (kv_dim, h), h**-0.5),
                wo=_normal(rng, (h, q_dim), q_dim**-0.5),
                bq=_normal(rng, (q_dim,), 0.5) if bias else None,
                bk=_normal(rng, (kv_dim,), 2.0) if bias else None,
                bv=_normal(rng, (kv_dim,), 0.3) if bias else None,
                w_gate=_normal(rng, (inter, h), h**-0.5),
                w_up=_normal(rng, (inter, h), h**-0.5),
                w_down=_normal(rng, (h, inter), inter**-0.5),
            )
        )
    embedding = _normal(rng, (shape.vocab, h), 0.5)
    final_norm = (1.0 + _normal(rng, (h,), 0.1)).astype(np.float32)
    return layers, embedding, final_norm


def tensors_for(spec: ModelSpec, layers, embedding, final_norm) -> dict[str, np.ndarray]:
    """The checkpoint as ``{HF tensor name: float32 array}`` (:func:`expected_tensor_shapes`)."""
    out = {"model.embed_tokens.weight": embedding, "model.norm.weight": final_norm}
    for i, w in enumerate(layers):
        p = f"model.layers.{i}."
        out[p + "input_layernorm.weight"] = w.norm_in
        out[p + "post_attention_layernorm.weight"] = w.norm_post
        out[p + "self_attn.q_proj.weight"] = w.wq
        out[p + "self_attn.k_proj.weight"] = w.wk
        out[p + "self_attn.v_proj.weight"] = w.wv
        out[p + "self_attn.o_proj.weight"] = w.wo
        if spec.has_qkv_bias:
            out[p + "self_attn.q_proj.bias"] = w.bq
            out[p + "self_attn.k_proj.bias"] = w.bk
            out[p + "self_attn.v_proj.bias"] = w.bv
        out[p + "mlp.gate_proj.weight"] = w.w_gate
        out[p + "mlp.up_proj.weight"] = w.w_up
        out[p + "mlp.down_proj.weight"] = w.w_down
    want = expected_tensor_shapes(spec)
    got = {k: tuple(v.shape) for k, v in out.items()}
    if got != want:
        raise AssertionError("tensors_for: shapes disagree with expected_tensor_shapes")
    return out


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write float32 tensors in the safetensors format (8-byte header length, JSON, payload)."""
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name], dtype="<f4")
        header[name] = {
            "dtype": "F32",
            "shape": list(a.shape),
            "data_offsets": [offset, offset + a.nbytes],
        }
        offset += a.nbytes
        blobs.append(a.tobytes())
    text = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    text += b" " * (-len(text) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        f.writelines(blobs)


# --------------------------------------------------------------------------- calibration


def calibration_report(spec: ModelSpec, seqs: list[list[int]]) -> dict:
    """``calib.json`` of ``spec`` calibrated on the given token ids, rounded like the file.

    :func:`calibrate.calibrate` with ``seqs`` in place of the tokenized prompt
    files, passed through :func:`calibrate.canonical_json_text`.
    """
    return json.loads(calibrate.canonical_json_text(calibrate.calibrate(spec, seqs)))


def calibration_ids(shape: Shape, rng: np.random.Generator) -> list[list[int]]:
    """Random token id sequences of lengths :data:`CALIB_LENGTHS` over the vocabulary."""
    return [rng.integers(0, shape.vocab, n).tolist() for n in CALIB_LENGTHS]


# --------------------------------------------------------------------------- build


def model_dir_for(shape: Shape, seed: int) -> Path:
    return SYNTHETIC_DIR / shape.name(seed)


def build(shape: Shape, seed: int = 0, *, out_dir: Path | None = None) -> SyntheticModel:
    """Draw, write, calibrate and quantize one synthetic model.

    The weights, the calibration token ids and therefore every integer of the
    QuantModel are a function of ``(shape, seed)``; the checkpoint lands in
    ``out_dir`` (default ``build/synthetic/<name>/``) as ``model.safetensors``.
    """
    rng = np.random.default_rng([seed, shape.layers, shape.hidden, shape.heads, shape.vocab])
    model_dir = model_dir_for(shape, seed) if out_dir is None else Path(out_dir)
    spec = spec_for(shape, seed, model_dir)
    layers, embedding, final_norm = draw_weights(shape, rng)
    write_safetensors(
        spec.path("model.safetensors"), tensors_for(spec, layers, embedding, final_norm)
    )
    seqs = calibration_ids(shape, rng)
    calib = calibration_report(spec, seqs)
    quant = build_quant_model(spec, calib)
    return SyntheticModel(shape, seed, spec, layers, embedding, final_norm, seqs, calib, quant)
