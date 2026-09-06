"""Calibration: measure float activation ranges and fix the integer formats.

``uv run quettos calibrate <alias>`` runs the float32 reference forward over
about a thousand tokens (the prompt files plus three prose passages, rendered
through the chat template) and writes ``models/<name>/calib.json``: per-class
maxima, the ``FRAC`` of each class, K-centering rows, the pairwise Q/K
smoothing factors the quantizer folds into the weights, and the K-centering and
V-scale gates.  Output is deterministic (sorted keys, six significant digits,
no timestamps).  The measured quantities, the FRAC rule and the smoothing rule
are documented in ``docs/NUMERICS.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from quettos import reference_np
from quettos.model import REPO_ROOT, ModelSpec
from quettos.reference_np import LOG2E
from quettos.tokenizer_io import encode, prompt_tokens, render_chat

NUMERICS_VERSION = 1
FRAC_MAX = 16
HEADROOM_BITS = 2
FRAC_S_MIN = 16
FRAC_LOGITS = 16
K_CACHE_BITS = 8
QK_SMOOTH_ALPHA = 0.5  # s_p = (max|k_c|_p / max|q|_p) ** alpha
QK_SMOOTH_CAP = 16.0  # factors are clipped to [1/cap, cap]
FLOAT_DIGITS = 6  # significant digits of every float written to calib.json

PROMPTS_DIR = REPO_ROOT / "prompts"
MODELS_OUT_DIR = REPO_ROOT / "models"

CALIB_PROMPT_FILES: tuple[str, ...] = (
    "chat_short.json",
    "tool_call_weather.json",
    "long_512.json",
)

# Plain prose rendered as user turns through the chat template.
CALIB_TEXTS: tuple[str, ...] = (
    "A river is never quite the same twice. In spring it runs high and brown "
    "with melted snow, dragging branches and gravel along the bed and spreading "
    "across the low meadows on either side. By late summer the same stretch is a "
    "clear ribbon between banks of dry stones, slow enough that leaves drift on "
    "the surface without turning. People who live beside a river learn its "
    "moods the way others learn a timetable: which pools hold fish after a "
    "storm, where the ford is safe to cross, and how many days of rain it takes "
    "before the path along the bank goes under. Canals were built to tame this "
    "variety. A canal keeps one level between its locks, carries the same depth "
    "in every season, and asks nothing of the weather except that it not freeze.",
    "Bread needs four things: flour, water, salt and time. Yeast can be added "
    "or it can be gathered from the air and the flour itself, as bakers did for "
    "thousands of years before it was sold in packets. Mixing wakes the gluten "
    "in the flour; kneading lines it up into a net that can hold the gas the "
    "yeast produces; resting lets the net relax so the dough can be shaped "
    "without tearing. A hot oven sets the crust first and traps steam inside, "
    "which is why the loaf keeps rising for the first few minutes of baking. "
    "The crackle you hear when a loaf cools is the crust shrinking as moisture "
    "moves outward from the crumb. Stale bread is not dry bread; it is bread "
    "whose starch has crystallised again, which is why a few minutes in the "
    "oven can make yesterday's loaf soft once more.",
    "A bicycle stays upright because a moving wheel resists being tipped and "
    "because the rider steers, almost without noticing, into every lean. The "
    "front fork is angled so that the contact patch of the tyre trails behind "
    "the steering axis, and this trail makes the wheel turn toward the side the "
    "bicycle is falling, correcting the fall before it grows. Gears let the "
    "rider trade force for speed: a small cog at the back and a large ring at "
    "the front for a fast road, the reverse for a steep climb. Pneumatic tyres, "
    "introduced in the 1880s, did more for comfort than any suspension since, "
    "because the air absorbs the small shocks of the road before they reach the "
    "frame. A well-kept chain, properly inflated tyres and brakes that bite "
    "evenly are most of what a bicycle asks in return for years of service.",
)

# Which calibration class each recorded intermediate belongs to.
CLASS_OF: dict[str, str] = {
    "embed": "X",
    "x_attn": "X",
    "x": "X",
    "x_norm_attn": "XN",
    "x_norm_mlp": "XN",
    "x_norm_final": "XN",
    "q": "QKV",
    "k": "QKV",
    "v": "QKV",
    "v_raw": "QKV",
    "q_rope": "QKV",
    "k_rope": "QKV",
    "scores": "S",
    "ctx": "CTX",
    "gate": "GU",
    "up": "GU",
    "h": "H",
    "logits": "LOGITS",
}
CLASSES: tuple[str, ...] = ("X", "XN", "QKV", "S", "GU", "H", "CTX", "LOGITS")
FRAC_CLASSES: tuple[str, ...] = ("X", "QKV", "S", "GU", "H", "CTX", "LOGITS")
GATE_VARIANTS: tuple[str, ...] = ("raw", "centered", "smoothed")


# --------------------------------------------------------------------------- calibration set


def calibration_sequences(spec: ModelSpec) -> list[list[int]]:
    """Token id lists of the calibration set, in a fixed order."""
    seqs = [prompt_tokens(spec, PROMPTS_DIR / f) for f in CALIB_PROMPT_FILES]
    for text in CALIB_TEXTS:
        rendered = render_chat(spec, [{"role": "user", "content": text}])
        seqs.append(encode(spec, rendered))
    return seqs


def token_ids_sha256(seqs: Sequence[Sequence[int]]) -> str:
    payload = json.dumps([list(map(int, s)) for s in seqs], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def round_sig(x: float, digits: int = FLOAT_DIGITS) -> float:
    """``x`` rounded to ``digits`` significant digits, the precision of ``calib.json``."""
    return float(f"{x:.{digits}g}")


# --------------------------------------------------------------------------- FRAC rule


def frac_for_absmax(absmax: float) -> int:
    """``min(16, 29 - ceil(log2(absmax)))``: at least two headroom bits in an int32."""
    if not math.isfinite(absmax) or absmax < 0:
        raise ValueError(f"frac_for_absmax: bad absmax {absmax!r}")
    if absmax == 0.0:
        return FRAC_MAX
    return min(FRAC_MAX, 31 - HEADROOM_BITS - math.ceil(math.log2(absmax)))


def choose_fracs(absmax: dict[str, float]) -> dict[str, int]:
    """Per-class fraction bits from the measured maxima (see the module docstring).

    ``S`` is sized from the centered scores ``q . (k - c)`` and ``QKV`` also
    covers the centered K vectors, because those are the values the hardware
    holds once K-centering is applied; ``QKV`` and ``K_centered`` are the
    maxima of the smoothed Q and K.  The raw ``q . k`` maximum is recorded for
    reference only.
    """
    s_absmax = absmax.get("S_centered", absmax["S"])
    qkv_absmax = max(absmax["QKV"], absmax.get("K_centered", 0.0))
    fracs = {
        "X": min(frac_for_absmax(absmax["X"]), frac_for_absmax(absmax["XN"])),
        "QKV": frac_for_absmax(qkv_absmax),
        "S": frac_for_absmax(s_absmax),
        "GU": frac_for_absmax(absmax["GU"]),
        "H": frac_for_absmax(absmax["H"]),
        "CTX": frac_for_absmax(absmax["CTX"]),
        "LOGITS": FRAC_LOGITS,
    }
    if fracs["S"] < FRAC_S_MIN:
        raise ValueError(
            f"FRAC_S = {fracs['S']} from centered score absmax {s_absmax:.6g}; "
            f"the softmax needs at least {FRAC_S_MIN} fraction bits"
        )
    return fracs


# --------------------------------------------------------------------------- recorder


class CalibrationRecorder:
    """Accumulates the statistics above across one or more forward passes."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.absmax: dict[str, float] = {c: 0.0 for c in CLASSES}
        self.layer_absmax: dict[str, list[float]] = {
            "X": [0.0] * spec.layers,
            "H": [0.0] * spec.layers,
        }
        self.embed_absmax = 0.0
        # Per-channel maxima of the pre-RoPE projections and the V maximum: the
        # QKV class maximum of the smoothed model is rebuilt from these.
        self.q_pre_absmax = np.zeros((spec.layers, spec.heads * spec.head_dim))
        self.k_pre_absmax = np.zeros((spec.layers, spec.kv_heads * spec.head_dim))
        self.v_absmax = 0.0
        # Per sequence, per layer: post-RoPE Q / K and raw V (for the gates).
        self.q_rope: list[list[np.ndarray]] = []
        self.k_rope: list[list[np.ndarray]] = []
        self.v_raw: list[list[np.ndarray]] = []

    def begin_sequence(self) -> None:
        self.q_rope.append([])
        self.k_rope.append([])
        self.v_raw.append([])

    def record(self, name: str, layer: int | None, value: np.ndarray) -> None:
        if not self.k_rope:
            self.begin_sequence()
        cls = CLASS_OF[name]
        a = float(np.max(np.abs(value))) if value.size else 0.0
        self.absmax[cls] = max(self.absmax[cls], a)
        if name == "embed":
            self.embed_absmax = max(self.embed_absmax, a)
        if layer is not None:
            if cls == "X":
                self.layer_absmax["X"][layer] = max(self.layer_absmax["X"][layer], a)
            elif cls == "H":
                self.layer_absmax["H"][layer] = max(self.layer_absmax["H"][layer], a)
        if name in ("v", "v_raw"):
            self.v_absmax = max(self.v_absmax, a)
        if name == "q":
            self.q_pre_absmax[layer] = np.maximum(self.q_pre_absmax[layer], _channel_absmax(value))
        elif name == "k":
            self.k_pre_absmax[layer] = np.maximum(self.k_pre_absmax[layer], _channel_absmax(value))
        elif name == "q_rope":
            self.q_rope[-1].append(np.array(value, dtype=np.float32, copy=True))
        elif name == "k_rope":
            self.k_rope[-1].append(np.array(value, dtype=np.float32, copy=True))
        elif name == "v_raw":
            self.v_raw[-1].append(np.array(value, dtype=np.float32, copy=True))


def _channel_absmax(value: np.ndarray) -> np.ndarray:
    """``max |value|`` over the token axis of a ``[T, C]`` block, as float64 ``[C]``."""
    return np.max(np.abs(np.asarray(value, dtype=np.float64)), axis=0)


# --------------------------------------------------------------------------- K-centering


def k_center_rows(rec: CalibrationRecorder) -> np.ndarray:
    """Mean post-RoPE K per (layer, KV head) over all calibration tokens: ``[L, KV, D]``."""
    spec = rec.spec
    out = np.zeros((spec.layers, spec.kv_heads, spec.head_dim), dtype=np.float64)
    for layer in range(spec.layers):
        ks = np.concatenate([seq[layer] for seq in rec.k_rope], axis=0).astype(np.float64)
        out[layer] = ks.reshape(-1, spec.kv_heads, spec.head_dim).mean(axis=0)
    return out


# --------------------------------------------------------------------------- Q/K smoothing


def pair_to_channels(factors: np.ndarray) -> np.ndarray:
    """``[..., head_dim/2]`` pair factors to ``[..., head_dim]`` channel factors.

    Channels ``d`` and ``d + head_dim/2`` (one RoPE pair) share ``factors[..., d]``.
    """
    f = np.asarray(factors, dtype=np.float64)
    return np.concatenate([f, f], axis=-1)


def tile_factors(
    factors: np.ndarray, heads: int, kv_heads: int, head_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel factors of one layer: ``s_k[kv_heads * head_dim]`` and ``s_q[heads * head_dim]``.

    ``factors`` is ``[kv_heads, head_dim/2]``: channels ``d`` and ``d + head_dim/2``
    of KV head ``g`` carry ``factors[g, d]``, and the query heads
    ``g * n_rep .. (g + 1) * n_rep - 1`` served by ``g`` repeat its row.
    """
    f = np.asarray(factors, dtype=np.float64)
    if f.shape != (kv_heads, head_dim // 2):
        raise ValueError(f"tile_factors: shape {f.shape} is not {(kv_heads, head_dim // 2)}")
    per_head = pair_to_channels(f)  # [KV, D]
    s_k = per_head.reshape(kv_heads * head_dim)
    s_q = np.repeat(per_head, heads // kv_heads, axis=0).reshape(heads * head_dim)
    return s_k, s_q


def qk_smoothing_factors(
    rec: CalibrationRecorder,
    centers: np.ndarray,
    *,
    alpha: float = QK_SMOOTH_ALPHA,
    cap: float = QK_SMOOTH_CAP,
) -> np.ndarray:
    """Pairwise Q/K smoothing factors ``[L, KV, head_dim/2]`` from the calibration maxima.

    For layer ``l``, KV head ``g`` and RoPE pair ``p = (d, d + head_dim/2)``::

        s_p = (max|k_c|_p / max|q|_p) ** alpha

    with the maxima over both dimensions of the pair and all calibration
    tokens, ``k_c`` the centered post-RoPE K of head ``g`` and ``q`` the
    post-RoPE Q of the query heads ``g`` serves (a pair whose K or Q maximum is
    zero starts at 1).  Each ``(l, g)`` row is divided by its geometric mean,
    clipped to ``[1/cap, cap]`` and rounded to ``FLOAT_DIGITS`` significant
    digits, the precision of ``calib.json``, so the stored factors are exactly
    the ones the maxima below and the quantizer use.
    """
    spec = rec.spec
    h, kv, d = spec.heads, spec.kv_heads, spec.head_dim
    half = d // 2
    n_rep = h // kv
    out = np.ones((spec.layers, kv, half), dtype=np.float64)
    for layer in range(spec.layers):
        q_max = np.maximum.reduce([_channel_absmax(seq[layer]) for seq in rec.q_rope])
        q_max = q_max.reshape(kv, n_rep, d).max(axis=1)  # [KV, D]
        k_c = [
            np.abs(seq[layer].astype(np.float64).reshape(-1, kv, d) - centers[layer][None])
            for seq in rec.k_rope
        ]
        kc_max = np.maximum.reduce([blk.max(axis=0) for blk in k_c])  # [KV, D]
        pq = np.maximum(q_max[:, :half], q_max[:, half:])
        pk = np.maximum(kc_max[:, :half], kc_max[:, half:])
        s = np.ones_like(pq)
        ok = (pq > 0) & (pk > 0)
        s[ok] = (pk[ok] / pq[ok]) ** alpha
        s = s / np.exp(np.mean(np.log(s), axis=1, keepdims=True))
        s = np.clip(s, 1.0 / cap, cap)
        out[layer] = np.vectorize(round_sig)(s)
    return out


def smoothed_qkv_absmax(rec: CalibrationRecorder, factors: np.ndarray) -> float:
    """``max |value|`` of class ``QKV`` after the fold.

    V is unchanged; every Q channel is multiplied and every K channel divided
    by its factor, before and after RoPE (a factor is constant over a RoPE
    pair, so the post-RoPE channels scale by the same amount).
    """
    spec = rec.spec
    h, kv, d = spec.heads, spec.kv_heads, spec.head_dim
    out = rec.v_absmax
    for layer in range(spec.layers):
        s_k, s_q = tile_factors(factors[layer], h, kv, d)
        q_rope = np.maximum.reduce([_channel_absmax(seq[layer]) for seq in rec.q_rope])
        k_rope = np.maximum.reduce([_channel_absmax(seq[layer]) for seq in rec.k_rope])
        for vals in (
            rec.q_pre_absmax[layer] * s_q,
            rec.k_pre_absmax[layer] / s_k,
            q_rope * s_q,
            k_rope / s_k,
        ):
            out = max(out, float(np.max(vals)))
    return out


def _smoothed_layer(
    rec: CalibrationRecorder, s_idx: int, layer: int, centers: np.ndarray, factors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Post-RoPE ``q * s_q`` ``[T, H*D]``, ``k / s_k`` ``[T, KV*D]`` and ``c / s_k`` ``[KV, D]``."""
    spec = rec.spec
    h, kv, d = spec.heads, spec.kv_heads, spec.head_dim
    s_k, s_q = tile_factors(factors[layer], h, kv, d)
    q = rec.q_rope[s_idx][layer].astype(np.float64) * s_q[None, :]
    k = rec.k_rope[s_idx][layer].astype(np.float64) / s_k[None, :]
    c = centers[layer].reshape(kv * d) / s_k
    return q, k, c.reshape(kv, d)


# --------------------------------------------------------------------------- gates


def int8_roundtrip(x: np.ndarray) -> np.ndarray:
    """Per-row symmetric int8 round-to-nearest with an ``absmax/127`` scale, dequantized."""
    lim = (1 << (K_CACHE_BITS - 1)) - 1
    a = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = np.where(a > 0, a / lim, 1.0)
    q = np.clip(np.floor(x / scale + 0.5), -lim, lim)
    return q * scale


def k_centering_gate(
    rec: CalibrationRecorder, centers: np.ndarray, factors: np.ndarray
) -> dict[str, Any]:
    """Score error of the int8 K cache over the causal window, three variants, two units.

    ``raw`` quantizes K as projected, ``centered`` quantizes ``k - c``, and
    ``smoothed`` quantizes ``(k - c) / s`` with the per-channel smoothing
    factors ``s``; every variant is dequantized back to the unsmoothed domain
    (``s * int8((k - c) / s) + c``) and scored against the same exact ``q . k``.
    ``rel_rms_error_<v>`` is ``rms(q.k_int8 - q.k) / rms(q.k)`` over all
    layers; ``rms_error_log2_<v>`` and ``max_error_log2_<v>`` are the RMS and
    largest absolute error of the log2-domain score ``q.k / sqrt(d) * log2(e)``,
    the unit the softmax consumes (one unit halves a token's weight);
    ``layer_rms_error_log2_<v>`` gives that RMS per layer for ``centered`` and
    ``smoothed``.
    """
    spec = rec.spec
    h, kv, d = spec.heads, spec.kv_heads, spec.head_dim
    n_rep = h // kv
    scale = float(LOG2E) / math.sqrt(d)
    err = dict.fromkeys(GATE_VARIANTS, 0.0)
    err_max = dict.fromkeys(GATE_VARIANTS, 0.0)
    per_layer = ("centered", "smoothed")
    layer_err = {v: np.zeros(spec.layers, dtype=np.float64) for v in per_layer}
    layer_count = np.zeros(spec.layers, dtype=np.int64)
    ref = 0.0
    count = 0
    for s_idx in range(len(rec.k_rope)):
        for layer in range(spec.layers):
            q = rec.q_rope[s_idx][layer].astype(np.float64)
            k = rec.k_rope[s_idx][layer].astype(np.float64)
            t = q.shape[0]
            qh = np.transpose(q.reshape(t, h, d), (1, 0, 2))  # [H, T, D]
            kk = k.reshape(t, kv, d)
            c = centers[layer][None, :, :]  # [1, KV, D]
            s_k = pair_to_channels(factors[layer])[None, :, :]  # [1, KV, D]
            variants = {
                "exact": kk,
                "raw": int8_roundtrip(kk),
                "centered": int8_roundtrip(kk - c) + c,
                "smoothed": s_k * int8_roundtrip((kk - c) / s_k) + c,
            }
            mask = reference_np.causal_mask(t)
            scores = {}
            for name, kvar in variants.items():
                kh = np.repeat(np.transpose(kvar, (1, 0, 2)), n_rep, axis=0)  # [H, T, D]
                scores[name] = (qh @ np.transpose(kh, (0, 2, 1)))[:, mask]
            ref += float(np.sum(scores["exact"] ** 2))
            count += scores["exact"].size
            for name in GATE_VARIANTS:
                diff = scores[name] - scores["exact"]
                err[name] += float(np.sum(diff**2))
                err_max[name] = max(err_max[name], float(np.max(np.abs(diff))))
                if name in per_layer:
                    layer_err[name][layer] += float(np.sum(diff**2))
            layer_count[layer] += scores["exact"].size
    out: dict[str, Any] = {}
    for name in GATE_VARIANTS:
        out[f"rel_rms_error_{name}"] = math.sqrt(err[name] / ref)
        out[f"rms_error_log2_{name}"] = math.sqrt(err[name] / count) * scale
        out[f"max_error_log2_{name}"] = err_max[name] * scale
    for name in per_layer:
        out[f"layer_rms_error_log2_{name}"] = (
            np.sqrt(layer_err[name] / layer_count) * scale
        ).tolist()
    return out


def centered_maxima(
    rec: CalibrationRecorder, centers: np.ndarray, factors: np.ndarray
) -> dict[str, float]:
    """Maxima of the values the hardware holds after smoothing and K-centering.

    ``K_centered`` is ``max |(k_rope - c) / s|`` and ``S_centered`` the largest
    log2-domain causal score ``((q s) . ((k - c) / s)) / sqrt(d) * log2(e)``,
    which the fold leaves unchanged.  Factors of 1 give the unsmoothed maxima.
    """
    spec = rec.spec
    h, kv, d = spec.heads, spec.kv_heads, spec.head_dim
    n_rep = h // kv
    scale = float(LOG2E) / math.sqrt(d)
    k_max = 0.0
    s_max = 0.0
    for s_idx in range(len(rec.k_rope)):
        for layer in range(spec.layers):
            q, k, c = _smoothed_layer(rec, s_idx, layer, centers, factors)
            t = q.shape[0]
            kc = k.reshape(t, kv, d) - c[None, :, :]
            k_max = max(k_max, float(np.max(np.abs(kc))))
            qh = np.transpose(q.reshape(t, h, d), (1, 0, 2))  # [H, T, D]
            kh = np.repeat(np.transpose(kc, (1, 0, 2)), n_rep, axis=0)  # [H, T, D]
            scores = (qh @ np.transpose(kh, (0, 2, 1))) * scale
            mask = reference_np.causal_mask(t)
            s_max = max(s_max, float(np.max(np.abs(scores[:, mask]))))
    return {"K_centered": k_max, "S_centered": s_max}


def v_scale_spread(rec: CalibrationRecorder) -> dict[str, Any]:
    """Histogram of ``e_max - e_t`` (V exponent gap to the sequence maximum) and its quantiles."""
    spec = rec.spec
    kv, d = spec.kv_heads, spec.head_dim
    hist: dict[int, int] = {}
    for seq in rec.v_raw:
        for layer in range(spec.layers):
            v = seq[layer].astype(np.float64).reshape(-1, kv, d)
            a = np.max(np.abs(v), axis=-1)  # [T, KV]
            for head in range(kv):
                col = a[:, head]
                col = col[col > 0]
                if col.size == 0:
                    continue
                e = np.floor(np.log2(col)).astype(np.int64)
                gaps = int(e.max()) - e
                for g, n in zip(*np.unique(gaps, return_counts=True), strict=True):
                    hist[int(g)] = hist.get(int(g), 0) + int(n)
    total = sum(hist.values())
    keys = sorted(hist)
    cumulative = 0
    quantile = {}
    for g in keys:
        cumulative += hist[g]
        for name, frac in (("p50", 0.5), ("p99", 0.99)):
            if name not in quantile and cumulative >= frac * total:
                quantile[name] = g
    return {
        "histogram": {str(g): hist[g] for g in keys},
        "count": total,
        "p50": quantile.get("p50", 0),
        "p99": quantile.get("p99", 0),
        "max": keys[-1] if keys else 0,
    }


# --------------------------------------------------------------------------- driver


def calibrate(spec: ModelSpec, seqs: Sequence[Sequence[int]] | None = None) -> dict[str, Any]:
    """Run the calibration set (default :func:`calibration_sequences`) and assemble the report."""
    seqs = calibration_sequences(spec) if seqs is None else [list(map(int, s)) for s in seqs]
    rec = CalibrationRecorder(spec)
    for ids in seqs:
        rec.begin_sequence()
        reference_np.forward(spec, ids, hooks=rec)

    centers = k_center_rows(rec)
    factors = qk_smoothing_factors(rec, centers)
    unsmoothed = centered_maxima(rec, centers, np.ones_like(factors))
    absmax = {c: rec.absmax[c] for c in CLASSES}
    absmax_unsmoothed = {"QKV": absmax["QKV"], "K_centered": unsmoothed["K_centered"]}
    absmax["QKV"] = smoothed_qkv_absmax(rec, factors)
    absmax.update(centered_maxima(rec, centers, factors))
    fracs = choose_fracs(absmax)
    return {
        "numerics": NUMERICS_VERSION,
        "model": {
            "repo_id": spec.repo_id,
            "name": spec.name,
            "arch": spec.arch,
            "layers": spec.layers,
            "hidden": spec.hidden,
            "heads": spec.heads,
            "kv_heads": spec.kv_heads,
            "head_dim": spec.head_dim,
            "intermediate": spec.intermediate,
            "vocab": spec.vocab,
            "has_qkv_bias": spec.has_qkv_bias,
            "rms_norm_eps": spec.rms_norm_eps,
            "rope_theta": spec.rope_theta,
        },
        "tokens": {
            "count": sum(len(s) for s in seqs),
            "sequences": [len(s) for s in seqs],
            "sha256": token_ids_sha256(seqs),
        },
        "absmax": absmax,
        "embed_absmax": rec.embed_absmax,
        "frac": fracs,
        "frac_rule": (
            f"min({FRAC_MAX}, {31 - HEADROOM_BITS} - ceil(log2(absmax))); "
            "X covers X and XN; S from S_centered; QKV covers K_centered; "
            "QKV and K_centered are the smoothed maxima; LOGITS fixed at 16"
        ),
        "per_layer_absmax": {
            "X": list(rec.layer_absmax["X"]),
            "H": list(rec.layer_absmax["H"]),
        },
        "k_center": centers.tolist(),
        "qk_smoothing": {
            "alpha": QK_SMOOTH_ALPHA,
            "cap": QK_SMOOTH_CAP,
            "digits": FLOAT_DIGITS,
            "pair": "(d, d + head_dim/2) of every head",
            "rule": (
                "s_p = (max|k_c|_p / max|q|_p)^alpha over the calibration tokens, both "
                "dimensions of the pair and the query heads the KV head serves (k_c: centered "
                "post-RoPE K; q: post-RoPE Q); per (layer, KV head) divided by the geometric "
                "mean; clipped to [1/cap, cap]; rounded to `digits` significant digits"
            ),
            "fold": (
                "rows d and d + head_dim/2 of W_k, b_k and the K-centering row divided by s_p; "
                "the same rows of W_q, b_q of every served query head multiplied by s_p"
            ),
            "factors": factors.tolist(),
            "factor_min": float(factors.min()),
            "factor_max": float(factors.max()),
            "absmax_unsmoothed": absmax_unsmoothed,
        },
        "k_centering_gate": {
            "k_bits": K_CACHE_BITS,
            "metric": (
                "rel: rms(q.k_int8 - q.k) / rms(q.k); log2: error of q.k / sqrt(d) * log2(e); "
                "both over the causal window; raw K, centered K, and centered K divided by the "
                "smoothing factors"
            ),
            **k_centering_gate(rec, centers, factors),
        },
        "v_scale_spread": v_scale_spread(rec),
    }


def _round_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        return round_sig(obj)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v) for v in obj]
    return obj


def canonical_json_text(obj: Any) -> str:
    """Canonical JSON of a report: sorted keys, floats at six significant digits, newline."""
    return json.dumps(_round_floats(obj), sort_keys=True, indent=1) + "\n"


def calib_json_text(report: dict[str, Any]) -> str:
    """Canonical text of a calibration report (:func:`canonical_json_text`)."""
    return canonical_json_text(report)


def calib_path(spec: ModelSpec) -> Path:
    return MODELS_OUT_DIR / spec.name / "calib.json"


def load_calib(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_calib(spec: ModelSpec, out: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Calibrate ``spec`` and write the report; returns the path and the report."""
    report = calibrate(spec)
    path = calib_path(spec) if out is None else Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(calib_json_text(report), encoding="utf-8")
    return path, report
