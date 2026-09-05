"""Integer numerics for Quettos Core: the single source of truth.

Every integer operation the RTL performs is defined here once, in Python ints
and numpy ``int64`` arrays; the table generator, quantizer, golden model, ISA
simulator and hardware tests import these functions and the RTL mirrors them
bit for bit.  The RTL knows no fixed-point format: every fraction position,
exponent bias and constant is a compiler value carried in a descriptor field.

Conventions: ``round_shift`` (round half toward +inf, arithmetic shift) is the
only rounding operation; ``sat`` saturates and counts events in :class:`Stats`;
an sfloat ``{m, e}`` is ``m * 2**e`` with ``m`` in ``[2**15, 2**16)`` and the
canonical zero ``{0, 0}``; a ``FRAC_*`` class value ``v`` represents
``v * 2**-FRAC``.  Each function's docstring states its exact formula and widths.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TABLES_DIR = Path(__file__).resolve().parent / "tables"
LUTS_JSON = TABLES_DIR / "luts.json"

I16_MAX = (1 << 15) - 1
I32_MAX = (1 << 31) - 1
Q15_ONE = 1 << 15  # 1.0 in Q1.15

# --------------------------------------------------------------------------- primitives


def round_shift(x, s: int):
    """Round half toward +inf, then arithmetic shift right by ``s`` (``0 <= s <= 63``).

    Works on Python ints and numpy int64 arrays.  ``s == 0`` returns ``x``
    unchanged; ``s == 63`` is the largest shift the requant stage-2 clamp can
    produce.  Callers guarantee ``x + 2**(s-1)`` fits in 63 bits.
    """
    if s < 0:
        raise ValueError(f"round_shift: negative shift {s}")
    if s == 0:
        return x
    if s > 63:
        raise ValueError(f"round_shift: shift {s} too large")
    return (x + (1 << (s - 1))) >> s


def sat(x, bits: int, stats: Stats | None = None):
    """Saturate ``x`` to the signed ``bits``-bit range; count events in ``stats``."""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if isinstance(x, np.ndarray):
        if stats is not None:
            stats.sat += int(np.count_nonzero((x < lo) | (x > hi)))
        return np.clip(x, lo, hi)
    if x < lo or x > hi:
        if stats is not None:
            stats.sat += 1
        return lo if x < lo else hi
    return x


def bitlen(x: int) -> int:
    """Number of bits needed for the non-negative integer ``x`` (0 -> 0)."""
    return int(x).bit_length()


def absmax(x: np.ndarray) -> int:
    """Largest magnitude in ``x`` as a Python int (0 for an empty or zero vector)."""
    if x.size == 0:
        return 0
    return int(np.max(np.abs(x)))


@dataclass
class Stats:
    """Event counters mirrored by the hardware ``SAT_*`` / ``ERR_*`` CSRs."""

    sat: int = 0  # saturations at sat40/sat32 points
    err_shift: int = 0  # requant shift outside [0, 63] (must stay 0 on correct programs)
    clip: int = 0  # VQUANT clips at +-(2**(w-1)-1): expected, not a fault

    def __add__(self, other: Stats) -> Stats:
        return Stats(self.sat + other.sat, self.err_shift + other.err_shift, self.clip + other.clip)


# --------------------------------------------------------------------------- sfloat


@dataclass(frozen=True)
class SFloat:
    """Unsigned scale ``m * 2**e`` with ``m`` in ``[2**15, 2**16)`` or the zero ``{0, 0}``."""

    m: int
    e: int

    def __post_init__(self) -> None:
        if self.m == 0:
            if self.e != 0:
                raise ValueError("sfloat zero must be {0, 0}")
        elif not (1 << 15) <= self.m < (1 << 16):
            raise ValueError(f"sfloat mantissa {self.m} out of [2^15, 2^16)")

    @property
    def is_zero(self) -> bool:
        return self.m == 0

    def value(self) -> float:
        """Real value (for tests and reporting; never used in the datapath)."""
        return self.m * 2.0**self.e

    def shifted(self, k: int) -> SFloat:
        """Same mantissa, exponent moved by ``k`` (exact multiply by ``2**k``)."""
        return self if self.m == 0 else SFloat(self.m, self.e + k)


SFLOAT_ZERO = SFloat(0, 0)
SFLOAT_ONE = SFloat(1 << 15, -15)


def sfloat_from_int(a: int, e: int = 0) -> SFloat:
    """Encode the non-negative integer ``a * 2**e`` (``a`` up to 63 bits).

    Exact when ``a`` has at most 16 significant bits; otherwise the mantissa is
    ``round_shift(a, bitlen(a) - 16)`` with the ``2**16`` rounding overflow
    folded into the exponent.
    """
    if a < 0:
        raise ValueError("sfloat_from_int: negative")
    if a == 0:
        return SFLOAT_ZERO
    length = bitlen(a)
    if length <= 16:
        return SFloat(a << (16 - length), e + length - 16)
    sh = length - 16
    m = round_shift(a, sh)
    if m == 1 << 16:
        return SFloat(1 << 15, e + sh + 1)
    return SFloat(m, e + sh)


def sfloat_from_float(x: float) -> SFloat:
    """Encode a positive real (offline use: weight scales, constants).

    ``x = f * 2**k`` with ``f`` in ``[0.5, 1)`` (``math.frexp``); the mantissa is
    ``f * 2**16`` rounded half up, with the ``2**16`` overflow folded into the
    exponent.  Deterministic across platforms because ``frexp`` is exact.
    """
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"sfloat_from_float: bad value {x!r}")
    if x == 0.0:
        return SFLOAT_ZERO
    f, k = math.frexp(x)
    m = int(math.floor(f * (1 << 16) + 0.5))
    e = k - 16
    if m == 1 << 16:
        return SFloat(1 << 15, e + 1)
    return SFloat(m, e)


def sfloat_mul(a: SFloat, b: SFloat) -> SFloat:
    """Product of two sfloats, rounded once to a 16-bit mantissa.

    ``p = m_a * m_b`` lies in ``[2**30, 2**32)``.  If ``p >= 2**31`` the
    mantissa is ``round_shift(p, 16)`` with exponent ``e_a + e_b + 16``,
    otherwise ``round_shift(p, 15)`` with exponent ``e_a + e_b + 15``.  A
    rounding overflow to ``2**16`` becomes ``{2**15, e + 1}``.  Zero times
    anything is the canonical zero.  The result mantissa is always in range.
    """
    if a.is_zero or b.is_zero:
        return SFLOAT_ZERO
    p = a.m * b.m
    if p >= 1 << 31:
        m, e = round_shift(p, 16), a.e + b.e + 16
    else:
        m, e = round_shift(p, 15), a.e + b.e + 15
    if m == 1 << 16:
        return SFloat(1 << 15, e + 1)
    return SFloat(m, e)


# --------------------------------------------------------------------------- lookup tables

# Table definitions.  Each table stores, per entry i, the Q1.15 value v_i =
# round_half_up(f(x_i) * 2**15) and the forward difference dv_i = v_{i+1} - v_i
# (with v_N the value at the right end of the domain).  Interpolation is
# out = v_i + round_shift(dv_i * frac8, 8) with frac8 the next 8 bits below the index.
#
#   exp2    256 entries  x_i = i/256            in [0, 1)   f = 2**x       v in [32768, 65535]
#   sigmoid 512 entries  x_i = i/32             in [0, 16)  f = sigmoid    v in [16384, 32768]
#   rsqrt   512 entries  seg 0: x_i = 1 + i/256 in [1, 2);  seg 1: x_i = 2 + 2*(i-256)/256 in [2, 4)
#                                                            f = 1/sqrt(x)  v in (16384, 32768]
#   recip   256 entries  x_i = 1 + i/256        in [1, 2)   f = 1/x        v in (16384, 32768]
#
# Right-end values: exp2 -> 65536, sigmoid -> round(sigmoid(16)*2**15),
# rsqrt -> 16384, recip -> 16384.

TABLE_SPECS: dict[str, dict[str, int]] = {
    "exp2": {"entries": 256},
    "sigmoid": {"entries": 512},
    "rsqrt": {"entries": 512},
    "recip": {"entries": 256},
}


@dataclass(frozen=True)
class Lut:
    """One (value, delta) table.  ``v`` is uint16-range, ``dv`` is int16-range."""

    name: str
    v: np.ndarray
    dv: np.ndarray

    def __post_init__(self) -> None:
        n = TABLE_SPECS[self.name]["entries"]
        if self.v.shape != (n,) or self.dv.shape != (n,):
            raise ValueError(f"{self.name}: expected {n} entries")
        if int(self.v.min()) < 0 or int(self.v.max()) > 0xFFFF:
            raise ValueError(f"{self.name}: value out of uint16")
        if int(self.dv.min()) < -(1 << 15) or int(self.dv.max()) > I16_MAX:
            raise ValueError(f"{self.name}: delta out of int16")

    def interp(self, idx, frac8):
        """``v[idx] + round_shift(dv[idx] * frac8, 8)``; ``idx`` and ``frac8`` may be arrays."""
        return self.v[idx] + ((self.dv[idx] * frac8 + 128) >> 8)


@dataclass(frozen=True)
class Tables:
    exp2: Lut
    sigmoid: Lut
    rsqrt: Lut
    recip: Lut
    meta: dict = field(default_factory=dict, compare=False)


def load_tables(path: Path | str = LUTS_JSON) -> Tables:
    """Load the checked-in tables generated by ``quettos.lutgen``."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    luts = {
        name: Lut(
            name,
            np.asarray(raw[name]["v"], dtype=np.int64),
            np.asarray(raw[name]["dv"], dtype=np.int64),
        )
        for name in TABLE_SPECS
    }
    return Tables(
        luts["exp2"], luts["sigmoid"], luts["rsqrt"], luts["recip"], meta=raw.get("meta", {})
    )


def recip_q15(a_hi: int, tables: Tables) -> int:
    """``1/m`` in Q1.15 for ``m = a_hi / 2**15``, ``a_hi`` in ``[2**15, 2**16)``.

    Index = bits 14..7 of ``a_hi``, frac8 = bits 6..0 shifted left by one.
    Result in ``[16384, 32768]``.
    """
    if not (1 << 15) <= a_hi < (1 << 16):
        raise ValueError(f"recip_q15: {a_hi} not normalized")
    idx = (a_hi >> 7) & 0xFF
    frac8 = (a_hi & 0x7F) << 1
    return int(tables.recip.interp(idx, frac8))


def rsqrt_q15(m_q16: int, tables: Tables) -> int:
    """``1/sqrt(m)`` in Q1.15 for ``m = m_q16 / 2**16`` in ``[1, 4)``.

    ``m_q16`` lies in ``[2**16, 2**18)``.

    Segment 0 (``m < 2``): index = bits 15..8, frac8 = bits 7..0.
    Segment 1 (``m >= 2``): index = 256 + bits 16..9, frac8 = bits 8..1.
    """
    if not (1 << 16) <= m_q16 < (1 << 18):
        raise ValueError(f"rsqrt_q15: {m_q16} not in [2^16, 2^18)")
    if m_q16 < (1 << 17):
        idx = (m_q16 >> 8) & 0xFF
        frac8 = m_q16 & 0xFF
    else:
        idx = 256 + ((m_q16 >> 9) & 0xFF)
        frac8 = (m_q16 >> 1) & 0xFF
    return int(tables.rsqrt.interp(idx, frac8))


def exp2_q15(f16, tables: Tables):
    """``2**(f16 / 2**16)`` in Q1.15 for ``f16`` in ``[0, 2**16)``; result in ``[32768, 65535]``.

    ``f16`` may be an int64 array.  Index = bits 15..8, frac8 = bits 7..0.
    """
    idx = (f16 >> 8) & 0xFF
    frac8 = f16 & 0xFF
    return tables.exp2.interp(idx, frac8)


def sigmoid_q15(x_q: np.ndarray, frac: int, tables: Tables) -> np.ndarray:
    """``sigmoid(x)`` in Q1.15 for signed fixed-point ``x_q`` with ``frac`` fraction bits.

    Uses ``sigmoid(-x) = 1 - sigmoid(x)`` (``32768 - sig(|x|)``).  ``|x| >= 16``
    saturates to ``32768`` (1.0).  Table index = ``floor(|x| * 32)`` (bits
    ``frac-5`` and up), frac8 = the next 8 bits below the index.  Requires
    ``frac >= 13``.
    """
    if frac < 13:
        raise ValueError("sigmoid_q15 needs frac >= 13")
    ax = np.abs(x_q)
    idx = ax >> (frac - 5)
    frac8 = (ax >> (frac - 13)) & 0xFF
    in_range = idx < 512
    idx_c = np.where(in_range, idx, 0)
    pos = np.where(in_range, tables.sigmoid.interp(idx_c, frac8), Q15_ONE)
    return np.where(x_q >= 0, pos, Q15_ONE - pos)


# --------------------------------------------------------------------------- requant


def requant(
    acc,
    sw: SFloat,
    sx: SFloat,
    s1: int,
    sbias: int,
    bias_q: int = 0,
    old=None,
    stats: Stats | None = None,
):
    """Dequantize an accumulator into an int32 fixed-point output (one GEMV output).

    Real value of the accumulator is ``acc * Sw * Sx``.  The output class has
    ``FRAC_out`` fraction bits, and the compiler sets ``sbias = -(FRAC_out + s1)``
    (plus ``+24`` for EMBED, whose accumulator is pre-shifted left by 24), so::

        t = sat40(round_shift(acc * Sw_m, s1))
        S = sbias - (Sw_e + Sx_e)          # hardware clamps to [0, 63], counts ERR_SHIFT
        y = sat32(round_shift(t * Sx_m, S))
        y = sat32(y + bias_q)
        y = sat32(y + old)                 # only with the accumulate flag (fused residual add)

    If either scale is the canonical zero (padded channel or padded token) the
    dequantized term is exactly 0 with no shift and no ERR/SAT event; the bias
    and accumulate adds still apply.  Widths: ``acc`` is 40-bit, ``Sw_m``/``Sx_m``
    16-bit, so both products fit in 56 bits.  ``acc`` and ``old`` may be arrays.
    """
    if sw.is_zero or sx.is_zero:
        y = np.zeros_like(acc) if isinstance(acc, np.ndarray) else 0
    else:
        t = sat(round_shift(acc * sw.m, s1), 40, stats)
        shift = sbias - (sw.e + sx.e)
        if shift < 0 or shift > 63:
            if stats is not None:
                stats.err_shift += int(np.size(acc))
            shift = min(max(shift, 0), 63)
        y = sat(round_shift(t * sx.m, shift), 32, stats)
    if bias_q:
        y = sat(y + bias_q, 32, stats)
    if old is not None:
        y = sat(y + old, 32, stats)
    return y


def sbias_for(frac_out: int, s1: int, pre_shift: int = 0) -> int:
    """Descriptor ``sh1`` for an output class: ``-(frac_out + s1) + pre_shift``."""
    return -(frac_out + s1) + pre_shift


def choose_s1(acc_bits: int, frac_out: int, sw_e_max: int, sx_e_max: int) -> int:
    """Compiler rule for the stage-1 shift ``sh0`` of a GEMV.

    ``acc_bits`` is the signed width of the accumulator values the GEMV can
    produce (40 for the hardware accumulator; smaller when K and the operand
    widths bound it tighter).  ``t = acc * Sw_m >> s1`` must fit 40 bits, so
    ``s1 >= acc_bits + 16 - 40``.  Stage-1 rounding contributes
    ``0.5 * Sx_m * 2**-S`` output LSBs, so the smallest stage-2 shift over the
    reachable scales should stay at or above 16.  ``S = -(frac_out + s1) - Sw_e
    - Sx_e`` is smallest at the LARGEST reachable exponents, hence
    ``s1 <= -(frac_out + sw_e_max + sx_e_max) - 16``.  Returns the largest
    ``s1 <= 16`` that meets the 40-bit bound and, when feasible, the precision
    bound.  The compiler separately checks with ``requant_shift`` that ``S``
    stays within ``[0, 63]`` at the smallest reachable exponents.
    """
    s1_min = max(0, acc_bits + 16 - 40)
    s1_prec = -(frac_out + sw_e_max + sx_e_max) - 16
    return max(s1_min, min(16, s1_prec))


def requant_shift(sw: SFloat, sx: SFloat, sbias: int) -> int:
    """The stage-2 shift the hardware will compute (before clamping); compiler range check."""
    return sbias - (sw.e + sx.e)


def embed_dequant(
    q_row: np.ndarray, sw: SFloat, frac_x: int, s1: int = 16, stats: Stats | None = None
) -> np.ndarray:
    """EMBED: dequantize an int8 embedding row into the residual class.

    The hardware feeds ``acc = q << 24`` through the requant pipe with
    ``Sx = 1.0`` and ``sbias = -(FRAC_X + s1) + 24``.  With ``8 <= s1 <= 24``
    the 48-bit stage-1 product fits 40 bits after the shift and the dropped bits
    are all zero, so the result is a single rounding of ``q * Sw * 2**FRAC_X``.
    """
    if not 8 <= s1 <= 24:
        raise ValueError("embed_dequant: s1 must lie in [8, 24]")
    acc = q_row.astype(np.int64) << 24
    return requant(acc, sw, SFLOAT_ONE, s1, sbias_for(frac_x, s1, 24), stats=stats)


# --------------------------------------------------------------------------- VQUANT


def quant(
    x: np.ndarray,
    width: int,
    frac_in: int,
    tables: Tables,
    *,
    scale_mul: SFloat | None = None,
    amax: int | None = None,
    stats: Stats | None = None,
) -> tuple[np.ndarray, SFloat]:
    """Per-vector symmetric quantization to ``width`` bits (16 or 8) with an exact sfloat scale.

    With ``a = absmax(x)`` (or the tracked ``amax``) the scale basis is
    ``a_eff = a + (a >> (w-1)) + 1``, so the absmax element maps to
    ``2**(w-1) - 1`` instead of ``2**(w-1)`` (the same ``2**(w-1) - 1`` levels
    as an ``absmax / (2**(w-1) - 1)`` scale; the ``+1`` keeps the rounding of
    the absmax element below the half-way point for every ``a``).
    ``a_eff = a_hi * 2**e_a`` with ``a_hi`` its top 16 bits, and the scale is
    ``Sx = {a_hi, e_a - (w-1) - frac_in}``,
    exact in sfloat and within ``2**-15`` of ``a_eff / 2**(w-1)`` real units::

        inv   = recip_q15(a_hi)                 # (1/m) in Q1.15, m = a_hi / 2**15
        q     = round_shift(x * inv, 31 + e_a - w)
        q     = clip(q, -(2**(w-1)-1), 2**(w-1)-1)   # counted in stats.clip, not a fault

    since ``x * 2**(w-1) / a_eff = x * inv * 2**-(31 + e_a - w)``.  The clip is
    reachable only at the absmax element and only through the reciprocal
    table's error: never for ``w = 8``; for ``w = 16`` the table's ~1 LSB error
    spans the two-level margin, so roughly one vector in four clips its absmax
    element by one level (harmless, counted in ``stats.clip``).  A zero vector
    gives ``q = 0`` and the canonical zero scale.  ``scale_mul`` (a descriptor
    sfloat constant, e.g. ``log2(e)/8`` for q) multiplies the scale.  Widths:
    ``x`` is int32, ``inv`` 16-bit, product 47 bits.
    """
    if width not in (8, 16):
        raise ValueError("quant: width must be 8 or 16")
    a = absmax(x) if amax is None else int(amax)
    if a == 0:
        return np.zeros_like(x), SFLOAT_ZERO
    a_eff = a + (a >> (width - 1)) + 1
    e_a = bitlen(a_eff) - 16
    a_hi = a_eff >> e_a if e_a >= 0 else a_eff << (-e_a)
    inv = recip_q15(a_hi, tables)
    shift = 31 + e_a - width
    q = round_shift(x * inv, shift)
    lim = (1 << (width - 1)) - 1
    if stats is not None:
        stats.clip += int(np.count_nonzero((q > lim) | (q < -lim)))
    q = np.clip(q, -lim, lim)
    sx = SFloat(a_hi, e_a - (width - 1) - frac_in)
    if scale_mul is not None:
        sx = sfloat_mul(sx, scale_mul)
    return q, sx


def quant_groups(
    x: np.ndarray,
    group: int,
    width: int,
    frac_in: int,
    tables: Tables,
    *,
    scale_mul: SFloat | None = None,
    stats: Stats | None = None,
) -> tuple[np.ndarray, list[SFloat]]:
    """``quant`` applied independently to consecutive groups of ``group`` elements (per head)."""
    if x.size % group:
        raise ValueError("quant_groups: length not a multiple of group")
    out = np.empty_like(x)
    scales: list[SFloat] = []
    for g in range(x.size // group):
        sl = slice(g * group, (g + 1) * group)
        out[sl], s = quant(x[sl], width, frac_in, tables, scale_mul=scale_mul, stats=stats)
        scales.append(s)
    return out, scales


# --------------------------------------------------------------------------- RMSNorm


def eps_const(eps: float, d: int, frac_x: int) -> int:
    """``eps_c = round(eps * d * 2**(2*FRAC_X))``, the descriptor ``imm32`` for VRMSNORM."""
    return int(math.floor(eps * d * 2.0 ** (2 * frac_x) + 0.5))


def quantize_gamma(gamma: np.ndarray) -> tuple[np.ndarray, int]:
    """int16 gamma with one per-tensor exponent: ``gamma ~= q * 2**e``, ``|q| <= 32767``.

    ``e`` is the smallest exponent (``<= 0``) such that the largest magnitude
    fits; this is the offline encoding used by ``quantize.py``.  Requires
    ``max |gamma| <= 32767``; larger values have no encoding and raise.
    """
    g = np.asarray(gamma, dtype=np.float64)
    m = float(np.max(np.abs(g))) if g.size else 0.0
    if m == 0.0:
        return np.zeros(g.shape, dtype=np.int64), 0
    if m > I16_MAX:
        raise ValueError("quantize_gamma: magnitude exceeds the int16 encoding")
    e = int(math.floor(math.log2(m))) - 14
    q = np.floor(g / 2.0**e + 0.5).astype(np.int64)
    while int(np.max(np.abs(q))) > I16_MAX:
        e += 1
        q = np.floor(g / 2.0**e + 0.5).astype(np.int64)
    return q, min(e, 0)


def rmsnorm(
    x: np.ndarray,
    gamma_q: np.ndarray,
    gamma_e: int,
    eps_c: int,
    sqrt_d: SFloat,
    frac_x: int,
    tables: Tables,
    stats: Stats | None = None,
) -> np.ndarray:
    """RMSNorm on an int32 vector of class ``FRAC_X``, output in the same class.

    ::

        amax = absmax(x);  sh = max(0, bitlen(amax) - 15)
        ss   = sum((x >> sh)**2)                      # <= 48 bits
        ss'  = ss + (eps_c >> 2*sh)         # (mean(x^2) + eps) * d * 2**(2*FRAC_X - 2*sh)
        ss'  = m * 2**(2e) with m in [1, 4):  L = bitlen(ss'), 2e = L-1 if L odd else L-2
        R    = rsqrt_q15(m)                            # 1/sqrt(m) in Q1.15
        Rc   = sfloat_mul(sfloat_from_int(R, -15), sqrt_d)   # sqrt(d)/sqrt(m), R normalized
        xhat = round_shift(x * Rc_m, S1),  S1 = -(Rc_e + FRAC_X - sh - e)
        y    = sat32(round_shift(xhat * gamma_q, G)),  G = -gamma_e

    because ``rsqrt(mean + eps) = sqrt(d) * 2**(FRAC_X - sh - e) * R * 2**-15``.
    ``gamma_q`` is int16 with per-tensor exponent ``gamma_e`` (``<= 0``).

    ``S1`` is computed from data; it is non-negative whenever
    ``eps_c >= 2**(2 * (FRAC_X + e_d))`` with ``e_d = floor(log2(sqrt(d))) - 15``
    (``2**10`` for ``FRAC_X = 16`` and ``512 <= d < 1024``; both supported
    models use ``eps_c > 1.5e6``).  If a
    program still produces ``S1 < 0`` the shift is clamped to 0 and counted in
    ``err_shift``, mirroring the requant clamp.  ``|xhat|`` is bounded by
    ``sqrt(d) * 2**FRAC_X * (1 + 2**-13)`` (21 bits at ``d = 896``,
    ``FRAC_X = 16``); ``xhat * gamma_q`` fits 37 bits.
    """
    amax = absmax(x)
    sh = max(0, bitlen(amax) - 15)
    xs = x >> sh
    ss = int(np.sum(xs * xs))
    ss2 = ss + (eps_c >> (2 * sh))
    if ss2 <= 0:
        return np.zeros_like(x)
    length = bitlen(ss2)
    e2 = length - 1 if (length - 1) % 2 == 0 else length - 2
    e = e2 // 2
    m_q16 = ss2 >> (e2 - 16) if e2 >= 16 else ss2 << (16 - e2)
    r = rsqrt_q15(m_q16, tables)
    rc = sfloat_mul(sfloat_from_int(r, -15), sqrt_d)
    s1 = -(rc.e + frac_x - sh - e)
    if s1 < 0:
        if stats is not None:
            stats.err_shift += int(x.size)
        s1 = 0
    xhat = round_shift(x * rc.m, s1)
    g_shift = -gamma_e
    if g_shift < 0:
        raise ValueError("rmsnorm: gamma exponent must be <= 0")
    return sat(round_shift(xhat * gamma_q, g_shift), 32, stats)


# --------------------------------------------------------------------------- RoPE


def rope(
    x: np.ndarray,
    cos_row: np.ndarray,
    sin_row: np.ndarray,
    head_dim: int = 64,
    stats: Stats | None = None,
) -> np.ndarray:
    """Rotary embedding (rotate_half convention) on ``heads * head_dim`` int32 values.

    ``cos_row``/``sin_row`` are the ``head_dim/2`` int16 Q1.14 entries for the
    current position.  For each head and ``i < head_dim/2``::

        a' = sat32(round_shift(a * cos_i - b * sin_i, 14))
        b' = sat32(round_shift(b * cos_i + a * sin_i, 14))

    with ``a = x[i]`` and ``b = x[i + head_dim/2]``.
    """
    half = head_dim // 2
    if x.size % head_dim or cos_row.shape != (half,) or sin_row.shape != (half,):
        raise ValueError("rope: bad shapes")
    xh = x.reshape(-1, head_dim)
    a, b = xh[:, :half], xh[:, half:]
    out = np.empty_like(xh)
    out[:, :half] = sat(round_shift(a * cos_row - b * sin_row, 14), 32, stats)
    out[:, half:] = sat(round_shift(b * cos_row + a * sin_row, 14), 32, stats)
    return out.reshape(x.shape)


def rope_table_path(theta: float, max_pos: int = 2048) -> Path:
    """Checked-in table file for a RoPE base, e.g. ``rope_theta1e6_2048.npy``."""
    return TABLES_DIR / f"rope_theta{theta:.0e}_{max_pos}.npy".replace("e+0", "e")


def load_rope_table(theta: float, max_pos: int = 2048) -> np.ndarray:
    """int16 array ``[max_pos, 2, head_dim/2]``: ``[pos, 0]`` = cos, ``[pos, 1]`` = sin, Q1.14."""
    return np.load(rope_table_path(theta, max_pos)).astype(np.int64)


# --------------------------------------------------------------------------- K-centering


def subc(x: np.ndarray, c: np.ndarray, stats: Stats | None = None) -> np.ndarray:
    """``sat32(x - c)`` elementwise (VSUBC)."""
    return sat(x - c, 32, stats)


# --------------------------------------------------------------------------- softmax


SOFTMAX_EXT_BITS = 8  # extra fraction bits carried by e_t and p_t beyond Q1.15
SOFTMAX_CLAMP = 15 + SOFTMAX_EXT_BITS + 2  # distances beyond this many log2 units round to 0


def softmax(
    scores: np.ndarray,
    length: int,
    frac_s: int,
    v_scales: list[SFloat],
    tables: Tables,
    stats: Stats | None = None,
) -> tuple[np.ndarray, SFloat]:
    """Attention softmax over ``scores[:length]`` (log2 domain, class ``FRAC_S``).

    Returns int16 weights ``w`` (zeros beyond ``length``) and the sfloat
    ``SREG_out`` such that ``sum_t w_t * V_t * SREG_out`` reproduces
    ``sum_t p_t * Sv_t * V_t``.  The exponential and the probability carry
    ``E = 8`` extra fraction bits (Q1.23) so that tokens 13 to 23 log2 units
    below the maximum, which attention sinks make common, keep their relative
    precision; the multipliers are 24 x 16 unsigned (25 x 17 in the RTL's
    signed convention)::

        m   = max s_t
        d   = clamp(m - s_t, 0, 25 << FRAC_S)          # beyond 25 log2 units the weight rounds to 0
        n   = ceil(d / 2**FRAC_S);  g = n * 2**FRAC_S - d       # 2**-d = 2**-n * 2**(g / 2**FRAC_S)
        e_t = round_shift(exp2_q15(g as 16-bit fraction) << 8, n)   # Q1.23; max token gives 2**23
        sum = sum_t e_t                                  # <= 2**36 for 8192 tokens
        inv = recip_q15(sum_hi),  sum = sum_hi * 2**e_s
        p_t = round_shift(e_t * inv, 7 + e_s)            # Q1.23 probability, max exactly 2**23
        e_max = max Sv_e over the row
        w_t = round_shift(p_t * Sv_m[t], 24 + e_max - Sv_e[t])         in [0, 32768]
                                                 # shift amounts above 40 saturate to 40 (w_t = 0)
        SREG_out = 2**(1 + e_max)

    ``w_t`` reaches 32768 only when ``p_t = 1.0`` (every other token is at least
    25 log2 units below the maximum) and ``Sv_m[t] = 65535``; that value is
    clipped to 32767 and counted in ``stats.clip`` like the VQUANT clip.  The
    weights are 16-bit: a token rounds to zero weight when ``p_t * Sv_t`` is
    below ``2**e_max`` (half of ``SREG_out``), and every token's weight carries
    a rounding error of up to ``2**e_max`` in either direction, so the row's
    total mass is off by at most ``L * 2**e_max / Sv_max <= L * 2**-15`` (3.1%
    at 1024 tokens in the worst case; ``L * 2**-16`` when the largest V mantissa
    is 65535).
    Tokens whose V scale is the canonical zero get ``w_t = 0`` and do not take
    part in ``e_max``.  Widths: ``m - s_t`` needs a 33-bit signed subtraction
    before the clamp, ``n <= 25``, ``e_t * inv`` fits 39 bits, ``p_t * Sv_m``
    fits 39 bits, and the per-token ``w`` shift lies in ``[24, 54]`` for real V
    scales.
    """
    if length < 1 or length > scores.size or len(v_scales) < length:
        raise ValueError("softmax: bad length")
    ext = SOFTMAX_EXT_BITS
    s = scores[:length]
    m = int(np.max(s))
    d = np.clip(m - s, 0, SOFTMAX_CLAMP << frac_s)
    n = (d + (1 << frac_s) - 1) >> frac_s
    g = (n << frac_s) - d
    f16 = g >> (frac_s - 16) if frac_s >= 16 else g << (16 - frac_s)
    e2 = exp2_q15(f16, tables) << ext
    half = np.where(n > 0, np.left_shift(1, np.maximum(n - 1, 0)), 0)
    e_t = (e2 + half) >> n
    total = int(np.sum(e_t))
    e_s = bitlen(total) - 16
    sum_hi = total >> e_s if e_s >= 0 else total << (-e_s)
    inv = recip_q15(sum_hi, tables)
    p = round_shift(e_t * inv, 15 - ext + e_s)
    sv_m = np.array([sc.m for sc in v_scales[:length]], dtype=np.int64)
    sv_e = np.array([sc.e for sc in v_scales[:length]], dtype=np.int64)
    nonzero = sv_m != 0
    if not np.any(nonzero):
        return np.zeros_like(scores), SFLOAT_ZERO
    e_max = int(np.max(sv_e[nonzero]))
    # zero-scale tokens get a dummy shift; shifts above 40 give 0 since p * Sv_m < 2**39
    shifts = np.minimum(np.where(nonzero, 16 + ext + e_max - sv_e, 16), 40)
    w = np.zeros_like(scores)
    prod = p * sv_m
    w_len = np.where(nonzero, (prod + (1 << (shifts - 1))) >> shifts, 0)
    if stats is not None:
        stats.clip += int(np.count_nonzero(w_len > I16_MAX))
    w[:length] = np.minimum(w_len, I16_MAX)
    return w, SFloat(1 << 15, 1 + e_max - 15)


# --------------------------------------------------------------------------- SiLU


def silu_mul(
    g: np.ndarray,
    u: np.ndarray,
    frac_gu: int,
    frac_h: int,
    tables: Tables,
    stats: Stats | None = None,
) -> np.ndarray:
    """``h = silu(g) * u`` from class ``FRAC_GU`` (both inputs) into class ``FRAC_H``.

    ::

        sig  = sigmoid_q15(g)                      # Q1.15, symmetric, |g| >= 16 -> 1.0
        silu = round_shift(g * sig, 15)            # class FRAC_GU
        h    = sat32(round_shift(silu * u, 2*FRAC_GU - FRAC_H))

    Widths: ``g * sig`` is 47 bits; ``silu * u`` is at most 62 bits.
    """
    sh_h = 2 * frac_gu - frac_h
    if sh_h < 0:
        raise ValueError("silu_mul: FRAC_H must be <= 2*FRAC_GU")
    sig = sigmoid_q15(g, frac_gu, tables)
    silu = round_shift(g * sig, 15)
    return sat(round_shift(silu * u, sh_h), 32, stats)


# --------------------------------------------------------------------------- argmax / weights


def argmax(logits: np.ndarray) -> int:
    """Strict-greater argmax scanning ascending ids: ties resolve to the lowest id."""
    return int(np.argmax(logits))  # numpy returns the first maximal index


def quantize_rows_int8(w: np.ndarray) -> tuple[np.ndarray, list[SFloat]]:
    """Per-output-channel symmetric int8 RTN (offline, float64).

    For each row ``n``: ``Sw = sfloat(absmax / 127)`` (encoded first, so the
    dequantized scale is exactly what the hardware uses), then
    ``q = round_half_up(w / Sw.value())`` clipped to ``[-127, 127]``.  An
    all-zero row gets ``q = 0`` and the canonical zero scale.
    """
    w = np.asarray(w, dtype=np.float64)
    q = np.zeros(w.shape, dtype=np.int64)
    scales: list[SFloat] = []
    for n in range(w.shape[0]):
        a = float(np.max(np.abs(w[n]))) if w.shape[1] else 0.0
        if a == 0.0:
            scales.append(SFLOAT_ZERO)
            continue
        s = sfloat_from_float(a / 127.0)
        q[n] = np.clip(np.floor(w[n] / s.value() + 0.5), -127, 127).astype(np.int64)
        scales.append(s)
    return q, scales


def to_fixed(x: np.ndarray, frac: int) -> np.ndarray:
    """Offline: real values to int32 fixed point with ``frac`` fraction bits.

    Round half up, saturating to int32.
    """
    q = np.floor(np.asarray(x, dtype=np.float64) * 2.0**frac + 0.5)
    return np.clip(q, -(1 << 31), I32_MAX).astype(np.int64)


def from_fixed(x: np.ndarray, frac: int) -> np.ndarray:
    """Real values of a fixed-point vector (tests and reporting only)."""
    return np.asarray(x, dtype=np.float64) * 2.0**-frac
