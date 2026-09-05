"""Per-GEMV requant constants shared by the golden model and the compiler.

Every GEMV of the decode program dequantizes its accumulator with the two
descriptor shifts ``s1`` (``sh0``) and ``sbias`` (``sh1``) of
:func:`quettos.numerics.requant`.  This module derives them once per model and
activation width: the accumulator width from ``K`` and the operand widths, the
reachable weight and activation scale exponents, ``choose_s1`` / ``sbias_for``,
and the proof that the stage-2 shift stays inside the hardware window
``[0, 63]`` at both exponent extremes.  Entry point: :func:`build`.
Rules: ``docs/NUMERICS.md`` (Requant, Accumulator).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quettos import numerics
from quettos.calibrate import MODELS_OUT_DIR, load_calib
from quettos.numerics import SFLOAT_ONE, SFloat
from quettos.quantize import QuantLinear, QuantModel

MAX_CTX = 2048
ACC_W = 40  # hardware accumulator width (signed bits)
W_BITS = 8  # streamed weights, K cache and V cache are int8
SHIFT_MAX = 63  # top of the requant stage-2 clamp window
MIN_ACT_ABSMAX = 1  # smallest non-zero activation vector the S window is proven for (1 LSB)
EMBED_S1 = 16  # embed_dequant default stage-1 shift
EMBED_PRE_SHIFT = 24  # EMBED feeds acc = q << 24

GEMV_NAMES: tuple[str, ...] = ("embed", "qkv", "o", "gu", "down", "lm_head", "scores", "pv")


@dataclass(frozen=True)
class GemvConstants:
    """Descriptor constants of one GEMV class and the exponent window they were proven on.

    ``sw_e`` / ``sx_e`` are ``(min, max)`` of the scale exponents reached on
    the calibration set; ``shift_range`` is the stage-2 shift at those two
    extremes (the precision window, which ``choose_s1`` keeps at or above 16
    where the 40-bit bound allows, with one octave of activation margin).
    ``hard_shift_min`` is the stage-2 shift when every operand sits at the
    int32 saturation bound of its class; it is the proof that the hardware
    clamp ``[0, 63]`` is never reached on any input, calibration or not.
    """

    name: str
    k: int
    a_bits: int
    frac_out: int
    acc_bits: int
    sw_e: tuple[int, int]
    sx_e: tuple[int, int]
    s1: int
    sbias: int
    shift_range: tuple[int, int]
    hard_shift_min: int

    def check_window(self) -> None:
        """Raise unless the stage-2 shift lies in ``[0, 63]`` at the hard bounds and the window."""
        lo, hi = self.shift_range
        if self.hard_shift_min < 0 or lo < 0 or hi > SHIFT_MAX:
            raise ValueError(
                f"{self.name}: requant shift window {self.shift_range}, hard minimum "
                f"{self.hard_shift_min}, leaves [0, {SHIFT_MAX}]"
            )


@dataclass(frozen=True)
class ProgramConstants:
    """The constants of one program: per-GEMV shifts plus the settings they derive from."""

    model: str
    a_bits: int
    max_ctx: int
    layers: int
    frac: dict[str, int]
    gemvs: dict[str, GemvConstants]

    def __getitem__(self, name: str) -> GemvConstants:
        return self.gemvs[name]

    def as_dict(self) -> dict[str, Any]:
        """Plain-JSON view of the constants."""
        return {
            "model": self.model,
            "a_bits": self.a_bits,
            "max_ctx": self.max_ctx,
            "layers": self.layers,
            "frac": dict(sorted(self.frac.items())),
            "gemvs": {
                n: {
                    "k": g.k,
                    "a_bits": g.a_bits,
                    "frac_out": g.frac_out,
                    "acc_bits": g.acc_bits,
                    "sw_e": list(g.sw_e),
                    "sx_e": list(g.sx_e),
                    "s1": g.s1,
                    "sbias": g.sbias,
                    "shift_range": list(g.shift_range),
                    "hard_shift_min": g.hard_shift_min,
                }
                for n, g in self.gemvs.items()
            },
        }


# --------------------------------------------------------------------------- rules


def acc_bits_for(k: int, a_bits: int, w_bits: int = W_BITS) -> int:
    """Signed width of the accumulator values a GEMV over ``k`` terms can produce.

    ``|acc| <= k * (2**(a_bits-1) - 1) * (2**(w_bits-1) - 1)``, so the width is
    ``bitlen(bound) + 1``, capped at the hardware accumulator (``ACC_W`` = 40).
    For the scores GEMV ``k = 64``; for PV ``k = max_ctx`` (the number of
    softmax weights the row can hold, a bound rather than the typical sum).
    """
    bound = k * ((1 << (a_bits - 1)) - 1) * ((1 << (w_bits - 1)) - 1)
    return min(ACC_W, numerics.bitlen(bound) + 1)


def weight_exponent_range(lins: Iterable[QuantLinear]) -> tuple[int, int]:
    """``(min, max)`` scale exponent over the non-zero rows of the given matrices."""
    lo, hi = None, None
    for lin in lins:
        e = np.asarray(lin.scale_e, dtype=np.int64)[np.asarray(lin.scale_m) != 0]
        if e.size == 0:
            continue
        lo = int(e.min()) if lo is None else min(lo, int(e.min()))
        hi = int(e.max()) if hi is None else max(hi, int(e.max()))
    if lo is None or hi is None:
        raise ValueError("weight_exponent_range: every row has the zero scale")
    return lo, hi


def quant_scale(absmax_int: int, width: int, frac_in: int, tables: numerics.Tables, *, scale_mul):
    """The VQUANT scale of a vector whose absmax (in class LSB) is ``absmax_int``."""
    x = np.array([int(absmax_int)], dtype=np.int64)
    return numerics.quant(x, width, frac_in, tables, scale_mul=scale_mul)[1]


def activation_exponent_range(
    absmax_real: float,
    frac_in: int,
    width: int,
    tables: numerics.Tables,
    *,
    scale_mul: SFloat | None = None,
) -> tuple[int, int]:
    """``(min, max)`` VQUANT scale exponent for one activation class.

    The maximum comes from the calibration absmax of the class converted to
    its fixed-point LSB (``to_fixed``) through the ``a_eff`` rule of
    :func:`numerics.quant`; the minimum from the smallest non-zero vector,
    ``absmax = MIN_ACT_ABSMAX`` LSB (a zero vector takes the zero-scale path and
    needs no shift).  ``scale_mul`` is applied exactly as VQUANT applies it.
    """
    a_max = int(numerics.to_fixed(np.array([absmax_real]), frac_in)[0])
    if a_max < MIN_ACT_ABSMAX:
        raise ValueError(f"activation absmax {absmax_real} is below one LSB of FRAC {frac_in}")
    hi = quant_scale(a_max, width, frac_in, tables, scale_mul=scale_mul).e
    lo = quant_scale(MIN_ACT_ABSMAX, width, frac_in, tables, scale_mul=scale_mul).e
    return lo, hi


def activation_exponent_hard_max(
    frac_in: int, width: int, tables: numerics.Tables, *, scale_mul: SFloat | None = None
) -> int:
    """The VQUANT scale exponent of a vector saturating its int32 class (the hard bound)."""
    return quant_scale(numerics.I32_MAX, width, frac_in, tables, scale_mul=scale_mul).e


def sreg_exponent_range(sv_e: tuple[int, int], frac_s: int, tables: numerics.Tables):
    """``SREG_out`` exponents of the softmax for the given V-scale exponent window.

    Evaluated through :func:`numerics.softmax` on a one-token row so the
    ``2**(1 + e_max)`` rule is never restated here.
    """
    out = []
    for e in sv_e:
        _, sreg = numerics.softmax(
            np.zeros(1, dtype=np.int64), 1, frac_s, [SFloat(1 << 15, e)], tables
        )
        out.append(sreg.e)
    return out[0], out[1]


def gemv_constants(
    name: str,
    k: int,
    a_bits: int,
    frac_out: int,
    sw_e: tuple[int, int],
    sx_e: tuple[int, int],
    *,
    acc_bits: int | None = None,
    s1: int | None = None,
    pre_shift: int = 0,
    sw_e_hard: int | None = None,
    sx_e_hard: int | None = None,
) -> GemvConstants:
    """Choose ``s1``/``sbias`` and evaluate the stage-2 shift at the window and hard extremes.

    ``choose_s1`` sees the calibration maximum plus one octave of margin on the
    activation side.  If the shift at the hard bounds (``sw_e_hard``,
    ``sx_e_hard``; default: the window maxima) would be negative, ``s1`` is
    lowered, within the 40-bit bound, until it is not.
    """
    acc = acc_bits_for(k, a_bits) if acc_bits is None else acc_bits
    sw_hard = sw_e[1] if sw_e_hard is None else sw_e_hard
    sx_hard = sx_e[1] if sx_e_hard is None else sx_e_hard
    if s1 is None:
        s1 = numerics.choose_s1(acc, frac_out, sw_e[1], sx_e[1] + 1)
    s1_min = max(0, acc + 16 - 40)

    def shift(a: int, b: int, sb: int) -> int:
        return numerics.requant_shift(SFloat(1 << 15, a), SFloat(1 << 15, b), sb)

    sbias = numerics.sbias_for(frac_out, s1, pre_shift)
    hard = shift(sw_hard, sx_hard, sbias)
    if hard < 0:
        s1 = max(s1_min, s1 + hard)
        sbias = numerics.sbias_for(frac_out, s1, pre_shift)
        hard = shift(sw_hard, sx_hard, sbias)
    at_max = shift(sw_e[1], sx_e[1], sbias)
    at_min = shift(sw_e[0], sx_e[0], sbias)
    g = GemvConstants(name, k, a_bits, frac_out, acc, sw_e, sx_e, s1, sbias, (at_max, at_min), hard)
    g.check_window()
    return g


# --------------------------------------------------------------------------- build


def calib_path_for(model: QuantModel) -> Path:
    return MODELS_OUT_DIR / model.name / "calib.json"


def build(
    model: QuantModel,
    calib: dict[str, Any] | Path | str | None = None,
    *,
    a_bits: int = 16,
    max_ctx: int = MAX_CTX,
    tables: numerics.Tables | None = None,
) -> ProgramConstants:
    """Constants of the decode/prefill programs of ``model`` at activation width ``a_bits``.

    The activation maxima come from ``model.extra["absmax"]`` (written by the
    quantizer) or, when ``calib`` is given or the model predates that field,
    from the calibration report the model was quantized from (dict or path;
    default ``models/<name>/calib.json``), whose token hash must match
    ``model.calib_tokens_sha256``.  Activation classes: ``XN`` feeds the QKV,
    gate|up and LM-head GEMVs, ``CTX`` the o_proj GEMV, ``H`` the down GEMV;
    ``QKV`` bounds q (with the ``log2(e)/8`` scale) and V, ``K_centered`` the K
    cache.  Weight exponents are taken over all layers of the model, so one
    constant per GEMV class serves every layer.
    """
    if a_bits not in (8, 16):
        raise ValueError("a_bits must be 8 or 16")
    if calib is None and "absmax" in model.extra:
        absmax = model.extra["absmax"]
    else:
        if calib is None:
            calib = calib_path_for(model)
        if not isinstance(calib, dict):
            calib = load_calib(calib)
        if calib["tokens"]["sha256"] != model.calib_tokens_sha256:
            raise ValueError(
                "calib.json does not match the calibration the model was quantized with"
            )
        absmax = calib["absmax"]
    tables = numerics.load_tables() if tables is None else tables
    frac = model.frac
    d = model.head_dim

    def act(cls: str, frac_in: int, width: int, scale_mul: SFloat | None = None):
        return activation_exponent_range(
            float(absmax[cls]), frac_in, width, tables, scale_mul=scale_mul
        )

    def hard(frac_in: int, width: int, scale_mul: SFloat | None = None) -> int:
        return activation_exponent_hard_max(frac_in, width, tables, scale_mul=scale_mul)

    xn = act("XN", frac["X"], a_bits)
    ctx = act("CTX", frac["CTX"], a_bits)
    hh = act("H", frac["H"], a_bits)
    sq = act("QKV", frac["QKV"], 16, model.log2e_over_8)
    sk = act("K_centered", frac["QKV"], W_BITS)
    sv = act("QKV", frac["QKV"], W_BITS)
    sreg = sreg_exponent_range(sv, frac["S"], tables)
    one = (SFLOAT_ONE.e, SFLOAT_ONE.e)
    xn_hard = hard(frac["X"], a_bits)
    ctx_hard = hard(frac["CTX"], a_bits)
    hh_hard = hard(frac["H"], a_bits)
    sq_hard = hard(frac["QKV"], 16, model.log2e_over_8)
    sk_hard = hard(frac["QKV"], W_BITS)
    sreg_hard = sreg_exponent_range((sv[0], hard(frac["QKV"], W_BITS)), frac["S"], tables)[1]

    layers = model.layers
    embed_e = weight_exponent_range([model.embed])
    gemvs = [
        gemv_constants(
            "embed",
            model.hidden,
            8,
            frac["X"],
            embed_e,
            one,
            acc_bits=W_BITS + EMBED_PRE_SHIFT,
            s1=EMBED_S1,
            pre_shift=EMBED_PRE_SHIFT,
        ),
        gemv_constants(
            "qkv",
            model.hidden,
            a_bits,
            frac["QKV"],
            weight_exponent_range(la.wqkv for la in layers),
            xn,
            sx_e_hard=xn_hard,
        ),
        gemv_constants(
            "o",
            model.heads * d,
            a_bits,
            frac["X"],
            weight_exponent_range(la.wo for la in layers),
            ctx,
            sx_e_hard=ctx_hard,
        ),
        gemv_constants(
            "gu",
            model.hidden,
            a_bits,
            frac["GU"],
            weight_exponent_range(la.wgu for la in layers),
            xn,
            sx_e_hard=xn_hard,
        ),
        gemv_constants(
            "down",
            model.intermediate,
            a_bits,
            frac["X"],
            weight_exponent_range(la.wdown for la in layers),
            hh,
            sx_e_hard=hh_hard,
        ),
        gemv_constants(
            "lm_head", model.hidden, a_bits, frac["LOGITS"], embed_e, xn, sx_e_hard=xn_hard
        ),
        gemv_constants("scores", d, 16, frac["S"], sk, sq, sw_e_hard=sk_hard, sx_e_hard=sq_hard),
        gemv_constants("pv", max_ctx, 16, frac["CTX"], one, sreg, sx_e_hard=sreg_hard),
    ]
    return ProgramConstants(
        model=model.name,
        a_bits=a_bits,
        max_ctx=max_ctx,
        layers=len(layers),
        frac=dict(frac),
        gemvs={g.name: g for g in gemvs},
    )
