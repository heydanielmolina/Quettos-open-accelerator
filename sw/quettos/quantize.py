"""Offline int8 quantizer: float checkpoint + ``calib.json`` -> :class:`QuantModel`.

Every integer is produced by a :mod:`quettos.numerics` primitive, so the
dequantized values are exactly what the hardware reconstructs: per-channel int8
weights with sfloat scales grouped as the programs stream them, QKV biases in
``FRAC_QKV``, the V bias folded into the ``o_proj`` bias, int16 gammas,
K-centering rows and the descriptor constants.  The pairwise Q/K smoothing
factors of ``calib.json`` are folded into ``W_q``/``b_q``, ``W_k``/``b_k`` and
the K-centering rows before quantization (:func:`smooth_qk`); ``smoothing=False``
forces every factor to 1 and builds the K-centering-only ablation from the same
``calib.json``.  :func:`save` and :func:`load` round-trip the model through one
``.npz``.  Formats and the fold: ``docs/NUMERICS.md``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from quettos import numerics
from quettos.calibrate import NUMERICS_VERSION, load_calib, pair_to_channels, tile_factors
from quettos.model import BUILD_DIR, ModelSpec
from quettos.numerics import SFloat
from quettos.reference_np import LayerWeights, load_embedding, load_final_norm, load_layer

QUANT_DIR = BUILD_DIR / "quant"
NOSMOOTH_SUFFIX = "-nosmooth"  # file and quality-row label of the smoothing-free build
ROW_CHUNK = 4096  # rows quantized per numerics call (bounds the float64 temporaries)
SIGMOID_MIN_FRAC = 13  # numerics.sigmoid_q15 requirement on FRAC_GU
XHAT_BITS = 32  # qcore_vpu_lane carries the VRMSNORM intermediate as an int32


# --------------------------------------------------------------------------- containers


@dataclass
class QuantLinear:
    """int8 rows ``q[N, K]`` with per-row sfloat scales ``(m, e)`` and an int32 ``bias_q[N]``."""

    q: np.ndarray
    scale_m: np.ndarray
    scale_e: np.ndarray
    bias_q: np.ndarray

    def scales(self) -> list[SFloat]:
        return [SFloat(int(m), int(e)) for m, e in zip(self.scale_m, self.scale_e, strict=True)]

    def dequant(self) -> np.ndarray:
        """``q * scale`` per row in float64 (tests and reporting)."""
        s = self.scale_m.astype(np.float64) * np.exp2(self.scale_e.astype(np.float64))
        return self.q.astype(np.float64) * s[:, None]


@dataclass
class QuantNorm:
    gamma_q: np.ndarray  # int16
    gamma_e: int

    def dequant(self) -> np.ndarray:
        return self.gamma_q.astype(np.float64) * 2.0**self.gamma_e


@dataclass
class QuantLayer:
    wqkv: QuantLinear
    wo: QuantLinear
    wgu: QuantLinear
    wdown: QuantLinear
    norm_in: QuantNorm
    norm_post: QuantNorm


@dataclass
class QuantModel:
    """Everything the compiler lays out into ``image.bin`` for one model."""

    name: str
    repo_id: str
    arch: str
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    has_qkv_bias: bool
    rms_norm_eps: float
    rope_theta: float
    frac: dict[str, int]
    eps_c: dict[str, int]
    sqrt_d: SFloat
    log2e_over_8: SFloat
    calib_tokens_sha256: str
    layers: list[QuantLayer]
    norm_final: QuantNorm
    embed: QuantLinear
    k_center: np.ndarray  # int32 [layers, kv_heads, head_dim] in FRAC_QKV
    numerics_version: int = NUMERICS_VERSION
    # "absmax": calibration maxima per class; "qk_smoothing": alpha, cap and factor range
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return len(self.layers)


# --------------------------------------------------------------------------- building blocks


def quantize_linear(w: np.ndarray, bias_q: np.ndarray | None = None) -> QuantLinear:
    """Per-row int8 quantization of ``w[N, K]`` through :func:`numerics.quantize_rows_int8`."""
    n = w.shape[0]
    q = np.empty(w.shape, dtype=np.int8)
    m = np.empty(n, dtype=np.int32)
    e = np.empty(n, dtype=np.int32)
    for start in range(0, n, ROW_CHUNK):
        stop = min(start + ROW_CHUNK, n)
        q64, scales = numerics.quantize_rows_int8(w[start:stop])
        q[start:stop] = q64.astype(np.int8)
        m[start:stop] = [s.m for s in scales]
        e[start:stop] = [s.e for s in scales]
    if bias_q is None:
        bias_q = np.zeros(n, dtype=np.int32)
    return QuantLinear(q, m, e, np.asarray(bias_q, dtype=np.int32))


def quantize_norm(gamma: np.ndarray) -> QuantNorm:
    q, e = numerics.quantize_gamma(gamma)
    return QuantNorm(q.astype(np.int16), int(e))


def tile_v_bias(b_v: np.ndarray, heads: int, kv_heads: int, head_dim: int) -> np.ndarray:
    """``[heads * head_dim]`` vector repeating each KV head's V bias over its query heads."""
    n_rep = heads // kv_heads
    per_kv = np.asarray(b_v, dtype=np.float64).reshape(kv_heads, head_dim)
    return np.repeat(per_kv, n_rep, axis=0).reshape(heads * head_dim)


def fold_v_bias(w_o: np.ndarray, b_v: np.ndarray, heads: int, kv_heads: int, head_dim: int):
    """Real-valued ``W_o @ tile(b_v)`` (float64), the exact V-bias fold into ``o_proj``."""
    return np.asarray(w_o, dtype=np.float64) @ tile_v_bias(b_v, heads, kv_heads, head_dim)


def smooth_qk(
    w: LayerWeights, factors: np.ndarray, heads: int, kv_heads: int, head_dim: int
) -> LayerWeights:
    """Fold one layer's pairwise Q/K smoothing factors ``[kv_heads, head_dim/2]`` into its weights.

    Rows ``d`` and ``d + head_dim/2`` of ``W_k`` and ``b_k`` (KV head ``g``) are
    divided by ``factors[g, d]``; the same rows of ``W_q`` and ``b_q`` of every
    query head ``g`` serves are multiplied by it.  ``q . k`` is unchanged in
    exact arithmetic, and because a factor is constant over a RoPE pair the fold
    commutes with RoPE.  The four arrays come back as float64; everything else
    is passed through.
    """
    s_k, s_q = tile_factors(factors, heads, kv_heads, head_dim)
    return dataclasses.replace(
        w,
        wq=np.asarray(w.wq, dtype=np.float64) * s_q[:, None],
        bq=None if w.bq is None else np.asarray(w.bq, dtype=np.float64) * s_q,
        wk=np.asarray(w.wk, dtype=np.float64) / s_k[:, None],
        bk=None if w.bk is None else np.asarray(w.bk, dtype=np.float64) / s_k,
    )


def smoothing_factors(calib: dict[str, Any], spec: ModelSpec) -> np.ndarray:
    """The ``qk_smoothing`` factors of ``calib`` as float64 ``[layers, kv_heads, head_dim/2]``.

    Rejects a report without the block, with the wrong shape, or with a factor
    outside ``[1/cap, cap]`` or not finite.
    """
    block = calib.get("qk_smoothing")
    if block is None:
        raise ValueError("calib.json has no qk_smoothing block; rerun `quettos calibrate`")
    f = np.asarray(block["factors"], dtype=np.float64)
    want = (spec.layers, spec.kv_heads, spec.head_dim // 2)
    if f.shape != want:
        raise ValueError(f"calib.json qk_smoothing factors shape {f.shape} is not {want}")
    cap = float(block["cap"])
    if not np.all(np.isfinite(f)) or np.any(f < 1.0 / cap) or np.any(f > cap):
        raise ValueError(f"calib.json qk_smoothing factors leave [1/{cap:g}, {cap:g}]")
    return f


def quantize_layer(spec: ModelSpec, w: LayerWeights, frac: dict[str, int]) -> QuantLayer:
    h, kv, d = spec.heads, spec.kv_heads, spec.head_dim
    wqkv = np.concatenate([w.wq, w.wk, w.wv], axis=0).astype(np.float64)
    bias = np.zeros(wqkv.shape[0], dtype=np.float64)
    fold = np.zeros(spec.hidden, dtype=np.float64)
    if spec.has_qkv_bias:
        if w.bq is None or w.bk is None or w.bv is None:
            raise ValueError("quantize_layer: model declares QKV biases but a bias is missing")
        bias[: h * d] = w.bq
        bias[h * d : h * d + kv * d] = w.bk
        fold = fold_v_bias(w.wo, w.bv, h, kv, d)
    return QuantLayer(
        wqkv=quantize_linear(wqkv, numerics.to_fixed(bias, frac["QKV"])),
        wo=quantize_linear(w.wo.astype(np.float64), numerics.to_fixed(fold, frac["X"])),
        wgu=quantize_linear(np.concatenate([w.w_gate, w.w_up], axis=0).astype(np.float64)),
        wdown=quantize_linear(w.w_down.astype(np.float64)),
        norm_in=quantize_norm(w.norm_in),
        norm_post=quantize_norm(w.norm_post),
    )


def xhat_bits(hidden: int, frac_x: int) -> int:
    """Signed bits the VRMSNORM intermediate needs for ``sqrt(d) * 2**FRAC_X * (1 + 2**-13)``.

    ``xhat = round_shift(x * Rc_m, S1)`` is bounded by ``sqrt(d)`` times the
    scale of its class because ``|x| <= sqrt(sum(x**2))``; the ``2**-13`` covers
    the rsqrt table, the ``x >> sh`` truncation and the rounding.
    """
    bound = math.sqrt(hidden) * 2.0**frac_x * (1.0 + 2.0**-13) + 1.0
    return int(math.ceil(bound)).bit_length() + 1


def check_rmsnorm_domain(hidden: int, frac_x: int, eps_c: int, sqrt_d: SFloat) -> None:
    """The VRMSNORM intermediate has to fit the int32 the hardware carries it in.

    ``numerics.rmsnorm`` holds ``xhat`` exactly and saturates only ``y``;
    ``qcore_vpu_lane`` produces it as a lane result, so it is an int32 there and
    a saturation of it is a ``SAT_VPU`` event.  Two conditions make the two
    models identical for every vector a program can present: ``xhat`` fits
    ``XHAT_BITS`` bits, and ``S1`` is non-negative, which holds when
    ``eps_c >= 2**(2 * (FRAC_X + e_d))`` with ``e_d`` the ``sqrt(d)`` exponent
    (``docs/NUMERICS.md``, RMSNorm).
    """
    bits = xhat_bits(hidden, frac_x)
    if bits > XHAT_BITS:
        raise ValueError(
            f"VRMSNORM xhat needs {bits} signed bits at hidden {hidden} with "
            f"FRAC_X = {frac_x}; the hardware carries it in {XHAT_BITS}"
        )
    floor_bits = 2 * (frac_x + sqrt_d.e)
    need = 1 << floor_bits if floor_bits > 0 else 1
    if eps_c < need:
        raise ValueError(
            f"VRMSNORM eps_c = {eps_c} is below {need} at FRAC_X = {frac_x} with a "
            f"sqrt(d) exponent of {sqrt_d.e}; S1 would clamp and ERR_SHIFT would count"
        )


def check_fracs(frac: dict[str, int]) -> None:
    """Constraints the integer operators place on the per-class formats."""
    for cls in ("X", "QKV", "S", "GU", "H", "CTX", "LOGITS"):
        if cls not in frac:
            raise ValueError(f"calib.json: missing FRAC for class {cls}")
        if not 0 <= frac[cls] <= 30:
            raise ValueError(f"FRAC_{cls} = {frac[cls]} out of range")
    if frac["GU"] < SIGMOID_MIN_FRAC:
        raise ValueError(f"FRAC_GU = {frac['GU']} < {SIGMOID_MIN_FRAC} (sigmoid_q15)")
    if frac["H"] > 2 * frac["GU"]:
        raise ValueError("FRAC_H must be <= 2 * FRAC_GU (silu_mul)")
    if frac["S"] < 16:
        raise ValueError("FRAC_S must be >= 16 (softmax fraction)")


# --------------------------------------------------------------------------- build


def build_quant_model(
    spec: ModelSpec,
    calib: dict[str, Any] | Path | str,
    *,
    layers: int | None = None,
    smoothing: bool = True,
) -> QuantModel:
    """Quantize ``spec`` with the formats, K-centering rows and smoothing factors of ``calib``.

    Every layer's ``W_q``/``b_q`` and ``W_k``/``b_k`` are folded with
    :func:`smooth_qk` and its K-centering row divided by the same per-channel
    factors before quantization.  ``smoothing=False`` replaces the validated
    factors by 1 and changes nothing else -- same ``calib.json``, formats,
    K-centering rows, class maxima and program constants -- which is the
    K-centering-only ablation.  ``layers`` keeps only the first ``layers``
    decoder layers (for fast tests); the norm, embedding and constants are
    always produced.  The calibration maxima per class travel with the model in
    ``extra["absmax"]`` so the program constants (:mod:`quettos.program`)
    derive from the ``.npz`` alone; ``extra["qk_smoothing"]`` records whether
    the fold was applied, the rule's ``alpha`` and ``cap`` and the factor range.
    """
    if not isinstance(calib, dict):
        calib = load_calib(calib)
    if calib.get("numerics") != NUMERICS_VERSION:
        raise ValueError(
            f"calib.json numerics version {calib.get('numerics')} != {NUMERICS_VERSION}"
        )
    if calib["model"]["repo_id"] != spec.repo_id:
        raise ValueError(f"calib.json is for {calib['model']['repo_id']}, not {spec.repo_id}")
    frac = {k: int(v) for k, v in calib["frac"].items()}
    check_fracs(frac)
    eps_c = numerics.eps_const(spec.rms_norm_eps, spec.hidden, frac["X"])
    sqrt_d = numerics.sfloat_from_float(math.sqrt(spec.hidden))
    check_rmsnorm_domain(spec.hidden, frac["X"], eps_c, sqrt_d)
    n_layers = spec.layers if layers is None else min(layers, spec.layers)
    factors = smoothing_factors(calib, spec)
    if not smoothing:
        factors = np.ones_like(factors)
    h, kv, d = spec.heads, spec.kv_heads, spec.head_dim

    q_layers = [
        quantize_layer(spec, smooth_qk(load_layer(spec, i), factors[i], h, kv, d), frac)
        for i in range(n_layers)
    ]
    k_center = np.asarray(calib["k_center"], dtype=np.float64)[:n_layers]
    if k_center.shape != (n_layers, kv, d):
        raise ValueError(f"calib.json k_center shape {k_center.shape} does not match the model")
    k_center = k_center / pair_to_channels(factors[:n_layers])
    absmax = {k: float(v) for k, v in calib["absmax"].items()}
    return QuantModel(
        name=spec.name,
        repo_id=spec.repo_id,
        arch=spec.arch,
        hidden=spec.hidden,
        heads=spec.heads,
        kv_heads=spec.kv_heads,
        head_dim=spec.head_dim,
        intermediate=spec.intermediate,
        vocab=spec.vocab,
        has_qkv_bias=spec.has_qkv_bias,
        rms_norm_eps=spec.rms_norm_eps,
        rope_theta=spec.rope_theta,
        frac=frac,
        eps_c={"input": eps_c, "post": eps_c, "final": eps_c},
        sqrt_d=sqrt_d,
        log2e_over_8=numerics.sfloat_from_float(math.log2(math.e) / 8),
        calib_tokens_sha256=str(calib["tokens"]["sha256"]),
        layers=q_layers,
        norm_final=quantize_norm(load_final_norm(spec)),
        embed=quantize_linear(load_embedding(spec).astype(np.float64)),
        k_center=numerics.to_fixed(k_center, frac["QKV"]).astype(np.int32),
        extra={
            "absmax": absmax,
            "qk_smoothing": {
                "enabled": bool(smoothing),
                "alpha": float(calib["qk_smoothing"]["alpha"]),
                "cap": float(calib["qk_smoothing"]["cap"]),
                "factor_min": float(factors.min()),
                "factor_max": float(factors.max()),
            },
        },
    )


# --------------------------------------------------------------------------- save / load

_LINEAR_FIELDS = ("q", "scale_m", "scale_e", "bias_q")
_LAYER_LINEARS = ("wqkv", "wo", "wgu", "wdown")
_LAYER_NORMS = ("norm_in", "norm_post")


def _sfloat_json(s: SFloat) -> dict[str, int]:
    return {"m": s.m, "e": s.e}


def _sfloat_from_json(d: dict[str, int]) -> SFloat:
    return SFloat(int(d["m"]), int(d["e"]))


def manifest(model: QuantModel) -> dict[str, Any]:
    """Scalar side of the model (everything that is not an array)."""
    return {
        "format": "quettos-quant",
        "numerics": model.numerics_version,
        "name": model.name,
        "repo_id": model.repo_id,
        "arch": model.arch,
        "hidden": model.hidden,
        "heads": model.heads,
        "kv_heads": model.kv_heads,
        "head_dim": model.head_dim,
        "intermediate": model.intermediate,
        "vocab": model.vocab,
        "layers": model.n_layers,
        "has_qkv_bias": model.has_qkv_bias,
        "rms_norm_eps": model.rms_norm_eps,
        "rope_theta": model.rope_theta,
        "frac": dict(sorted(model.frac.items())),
        "eps_c": dict(sorted(model.eps_c.items())),
        "sqrt_d": _sfloat_json(model.sqrt_d),
        "log2e_over_8": _sfloat_json(model.log2e_over_8),
        "calib_tokens_sha256": model.calib_tokens_sha256,
        "gamma_e": {
            "final": model.norm_final.gamma_e,
            "layers": [
                {"norm_in": lay.norm_in.gamma_e, "norm_post": lay.norm_post.gamma_e}
                for lay in model.layers
            ],
        },
        "extra": model.extra,
    }


def arrays(model: QuantModel) -> dict[str, np.ndarray]:
    """Every array of the model keyed the way :func:`save` stores it."""
    out: dict[str, np.ndarray] = {}
    for i, lay in enumerate(model.layers):
        for lin_name in _LAYER_LINEARS:
            lin: QuantLinear = getattr(lay, lin_name)
            for f in _LINEAR_FIELDS:
                out[f"layers.{i}.{lin_name}.{f}"] = getattr(lin, f)
        for norm_name in _LAYER_NORMS:
            out[f"layers.{i}.{norm_name}.gamma_q"] = getattr(lay, norm_name).gamma_q
    out["norm_final.gamma_q"] = model.norm_final.gamma_q
    for f in _LINEAR_FIELDS:
        out[f"embed.{f}"] = getattr(model.embed, f)
    out["k_center"] = model.k_center
    return out


def smoothing_enabled(model: QuantModel) -> bool:
    """Whether the Q/K smoothing fold was applied when ``model`` was built."""
    return bool(model.extra.get("qk_smoothing", {}).get("enabled", True))


def default_path(name: str, *, smoothing: bool = True) -> Path:
    """``build/quant/<name>.npz``, or ``<name>-nosmooth.npz`` for the ablation build."""
    return QUANT_DIR / f"{name}{'' if smoothing else NOSMOOTH_SUFFIX}.npz"


def save(model: QuantModel, path: Path | str | None = None) -> Path:
    """Write the model as an uncompressed ``.npz``; returns the path."""
    if path is None:
        path = default_path(model.name, smoothing=smoothing_enabled(model))
    else:
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = arrays(model)
    payload["manifest"] = np.array(json.dumps(manifest(model), sort_keys=True))
    np.savez(path, **payload)
    return path


def _linear_from(z: Any, prefix: str) -> QuantLinear:
    return QuantLinear(*(np.array(z[f"{prefix}.{f}"]) for f in _LINEAR_FIELDS))


def load(path: Path | str) -> QuantModel:
    """Read a model written by :func:`save`."""
    with np.load(path, allow_pickle=False) as z:
        man = json.loads(str(z["manifest"]))
        if man.get("format") != "quettos-quant":
            raise ValueError(f"{path}: not a quettos quant file")
        layers = []
        for i in range(int(man["layers"])):
            ge = man["gamma_e"]["layers"][i]
            layers.append(
                QuantLayer(
                    wqkv=_linear_from(z, f"layers.{i}.wqkv"),
                    wo=_linear_from(z, f"layers.{i}.wo"),
                    wgu=_linear_from(z, f"layers.{i}.wgu"),
                    wdown=_linear_from(z, f"layers.{i}.wdown"),
                    norm_in=QuantNorm(
                        np.array(z[f"layers.{i}.norm_in.gamma_q"]), int(ge["norm_in"])
                    ),
                    norm_post=QuantNorm(
                        np.array(z[f"layers.{i}.norm_post.gamma_q"]), int(ge["norm_post"])
                    ),
                )
            )
        return QuantModel(
            name=man["name"],
            repo_id=man["repo_id"],
            arch=man["arch"],
            hidden=int(man["hidden"]),
            heads=int(man["heads"]),
            kv_heads=int(man["kv_heads"]),
            head_dim=int(man["head_dim"]),
            intermediate=int(man["intermediate"]),
            vocab=int(man["vocab"]),
            has_qkv_bias=bool(man["has_qkv_bias"]),
            rms_norm_eps=float(man["rms_norm_eps"]),
            rope_theta=float(man["rope_theta"]),
            frac={k: int(v) for k, v in man["frac"].items()},
            eps_c={k: int(v) for k, v in man["eps_c"].items()},
            sqrt_d=_sfloat_from_json(man["sqrt_d"]),
            log2e_over_8=_sfloat_from_json(man["log2e_over_8"]),
            calib_tokens_sha256=man["calib_tokens_sha256"],
            layers=layers,
            norm_final=QuantNorm(np.array(z["norm_final.gamma_q"]), int(man["gamma_e"]["final"])),
            embed=_linear_from(z, "embed"),
            k_center=np.array(z["k_center"]),
            numerics_version=int(man["numerics"]),
            extra=dict(man.get("extra", {})),
        )


def models_equal(a: QuantModel, b: QuantModel) -> bool:
    """Exact equality: identical manifests and bit-identical arrays with the same dtypes."""
    if manifest(a) != manifest(b):
        return False
    aa, bb = arrays(a), arrays(b)
    if aa.keys() != bb.keys():
        return False
    return all(aa[k].dtype == bb[k].dtype and np.array_equal(aa[k], bb[k]) for k in aa)


def weight_bytes(model: QuantModel) -> int:
    """int8 weight bytes streamed per token (all GEMV rows plus the LM head)."""
    return sum(v.nbytes for k, v in arrays(model).items() if k.endswith(".q"))
