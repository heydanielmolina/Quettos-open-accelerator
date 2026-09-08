"""Property tests for :mod:`quettos.numerics` against independent references.

Every check below rebuilds the expected result from the mathematical definition
in the function's docstring, using Python big integers, :class:`fractions.Fraction`,
mpmath or float64, never by calling back into the code path under test.  Error
bounds are derived from the rounding budget of each operation; the measured
maxima on the checked-in tables are quoted next to the bound they justify.
Units: "LSB" is one step of the output format (Q1.15 for the tables).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from fractions import Fraction

import mpmath
import numpy as np
import pytest
from quettos import numerics as N
from quettos.numerics import SFLOAT_ONE, SFLOAT_ZERO, Lut, SFloat, Stats

HALF = Fraction(1, 2)
I32_MIN = -(1 << 31)

# --------------------------------------------------------------------------- reference helpers


def frac_value(s: SFloat) -> Fraction:
    """Exact rational value of an sfloat."""
    return Fraction(s.m) * Fraction(2) ** s.e


def round_half_up(x: Fraction) -> int:
    """Round half toward +inf, exactly."""
    return math.floor(x + HALF)


def ref_sfloat(x: Fraction) -> SFloat:
    """Correctly rounded (half up) sfloat of a positive rational: the mathematical encoding.

    ``x`` lies in ``[2**(k-1), 2**k)``; the mantissa is ``round_half_up(x / 2**(k-16))``
    with a ``2**16`` overflow folded into the exponent.
    """
    assert x > 0
    k = x.numerator.bit_length() - x.denominator.bit_length()
    while Fraction(2) ** (k - 1) > x:
        k -= 1
    while Fraction(2) ** k <= x:
        k += 1
    m = round_half_up(x / Fraction(2) ** (k - 16))
    if m == 1 << 16:
        return SFloat(1 << 15, k - 15)
    return SFloat(m, k - 16)


def q15_error(got: np.ndarray, ref: list[mpmath.mpf]) -> float:
    """Largest |got_i - ref_i| where ``ref`` is the unrounded Q1.15 value (mpmath)."""
    return max(float(abs(mpmath.mpf(int(g)) - r)) for g, r in zip(got.tolist(), ref, strict=True))


def rand_sfloat(rng: np.random.Generator, e_lo: int, e_hi: int) -> SFloat:
    return SFloat(int(rng.integers(1 << 15, 1 << 16)), int(rng.integers(e_lo, e_hi + 1)))


@pytest.fixture(scope="module")
def tables() -> N.Tables:
    return N.load_tables()


@pytest.fixture(scope="module")
def mp128():
    """mpmath context at 128-bit precision (the precision the tables were generated with)."""
    old = mpmath.mp.prec
    mpmath.mp.prec = 128
    yield mpmath.mp
    mpmath.mp.prec = old


# =========================================================================== primitives


def test_round_shift_exhaustive_small_range() -> None:
    for s in range(0, 7):
        for x in range(-80, 81):
            expect = x if s == 0 else round_half_up(Fraction(x, 1 << s))
            assert N.round_shift(x, s) == expect, (x, s)
    xs = np.arange(-80, 81, dtype=np.int64)
    for s in range(0, 7):
        expect = np.array([x if s == 0 else round_half_up(Fraction(x, 1 << s)) for x in xs])
        got = N.round_shift(xs, s)
        assert got.dtype == np.int64
        np.testing.assert_array_equal(got, expect)


def test_round_shift_half_toward_plus_inf() -> None:
    # exact halves: -1.5 -> -1, -0.5 -> 0, 0.5 -> 1, 1.5 -> 2 (never away from zero)
    assert N.round_shift(-3, 1) == -1
    assert N.round_shift(-1, 1) == 0
    assert N.round_shift(1, 1) == 1
    assert N.round_shift(3, 1) == 2
    assert N.round_shift(-6, 2) == -1  # -1.5
    assert N.round_shift(-5, 2) == -1  # -1.25
    assert N.round_shift(-7, 2) == -2  # -1.75
    assert N.round_shift(6, 2) == 2  # 1.5


def test_round_shift_shift_range() -> None:
    with pytest.raises(ValueError):
        N.round_shift(1, -1)
    with pytest.raises(ValueError):
        N.round_shift(1, 64)
    assert N.round_shift(1, 63) == 0  # 63 is the top of the requant stage-2 clamp range
    assert N.round_shift((1 << 62) - 1, 63) == 0
    assert N.round_shift(-(1 << 62), 63) == 0
    assert N.round_shift(1 << 61, 62) == 1  # 0.5 rounds up
    assert N.round_shift((1 << 61) - 1, 62) == 0
    assert N.round_shift(np.array([5, -5], dtype=np.int64), 0).tolist() == [5, -5]


@pytest.mark.parametrize("bits", [8, 16, 32, 40])
def test_sat_counts_every_clipped_element(bits: int) -> None:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    vals = [lo - 3, lo - 1, lo, lo + 1, -1, 0, 1, hi - 1, hi, hi + 1, hi + 1000]
    x = np.array(vals, dtype=np.int64)
    st = Stats()
    got = N.sat(x, bits, st)
    expect = [min(max(v, lo), hi) for v in vals]
    assert got.tolist() == expect
    assert st.sat == sum(1 for v in vals if v < lo or v > hi) == 4
    # scalar path agrees elementwise and counts one event per clipped value
    st2 = Stats()
    assert [N.sat(v, bits, st2) for v in vals] == expect
    assert st2.sat == st.sat
    # no stats object: same values, no crash
    assert N.sat(x, bits).tolist() == expect
    assert N.sat(hi + 5, bits) == hi
    # in-range values are untouched and not counted
    st3 = Stats()
    inside = np.array([lo, 0, hi], dtype=np.int64)
    assert N.sat(inside, bits, st3).tolist() == [lo, 0, hi]
    assert st3.sat == 0


def test_bitlen_and_absmax() -> None:
    assert N.bitlen(0) == 0
    for k in range(0, 63):
        assert N.bitlen(1 << k) == k + 1
        assert N.bitlen((1 << k) - 1) == k
    assert N.absmax(np.zeros(0, dtype=np.int64)) == 0
    assert N.absmax(np.zeros(5, dtype=np.int64)) == 0
    a = N.absmax(np.array([3, -7, 5], dtype=np.int64))
    assert a == 7 and isinstance(a, int)
    assert N.absmax(np.array([I32_MIN, 5], dtype=np.int64)) == 1 << 31


def test_stats_add() -> None:
    s = Stats(1, 2, 3) + Stats(10, 20, 30)
    assert (s.sat, s.err_shift, s.clip) == (11, 22, 33)
    assert Stats() == Stats(0, 0, 0)


# =========================================================================== sfloat


def test_sfloat_constructor_range_checks() -> None:
    assert SFloat(1 << 15, -3).value() == (1 << 15) * 2.0**-3
    assert SFloat((1 << 16) - 1, 5).m == (1 << 16) - 1
    for bad in [(1 << 15) - 1, 1 << 16, -1, 1, -(1 << 15)]:
        with pytest.raises(ValueError):
            SFloat(bad, 0)
    with pytest.raises(ValueError):
        SFloat(0, 1)
    assert SFloat(0, 0).is_zero and SFLOAT_ZERO == SFloat(0, 0)
    assert not SFloat(1 << 15, 0).is_zero
    assert SFLOAT_ONE.value() == 1.0 and frac_value(SFLOAT_ONE) == 1
    assert SFloat(40000, -3).shifted(5) == SFloat(40000, 2)
    assert SFLOAT_ZERO.shifted(7) == SFLOAT_ZERO


def test_sfloat_from_int_exact_up_to_16_bits() -> None:
    rng = np.random.default_rng(11)
    for a in [1, 2, 3, (1 << 15) - 1, 1 << 15, (1 << 16) - 1] + rng.integers(
        1, 1 << 16, size=2000
    ).tolist():
        for e in (0, -20, 7):
            s = N.sfloat_from_int(int(a), e)
            assert (1 << 15) <= s.m < (1 << 16)
            assert frac_value(s) == Fraction(int(a)) * Fraction(2) ** e
            assert s == ref_sfloat(Fraction(int(a)) * Fraction(2) ** e)
    # 16 significant bits with trailing zeros are still exact
    for k in range(1, 40):
        a = ((1 << 16) - 1) << k
        assert frac_value(N.sfloat_from_int(a)) == a
    assert N.sfloat_from_int(0) == SFLOAT_ZERO
    assert N.sfloat_from_int(0, 9) == SFLOAT_ZERO
    with pytest.raises(ValueError):
        N.sfloat_from_int(-1)


def test_sfloat_from_int_rounds_half_up_for_wide_inputs() -> None:
    rng = np.random.default_rng(12)
    cases = [(1 << 16) + 1, (1 << 17) + 3, (1 << 62) + 12345, (1 << 63) - 1]
    for bits in range(17, 64):
        cases += rng.integers(1 << (bits - 1), 1 << bits, size=40, dtype=np.uint64).tolist()
    # rounding overflow: 2**k - 1 for k > 16 rounds to 2**16 and folds into the exponent
    cases += [(1 << k) - 1 for k in range(17, 64)]
    for a in cases:
        a = int(a)
        for e in (0, -30):
            s = N.sfloat_from_int(a, e)
            exact = Fraction(a) * Fraction(2) ** e
            assert (1 << 15) <= s.m < (1 << 16)
            assert s == ref_sfloat(exact), (a, e)
            assert abs(frac_value(s) - exact) <= exact * Fraction(1, 1 << 16)


def test_sfloat_from_float_matches_exact_rounding() -> None:
    rng = np.random.default_rng(13)
    xs = [1.0, 0.5, 3.0, 2.0**-40, 2.0**40, 5e-324, 1.7e308, math.pi, 1 / 127]
    xs += (10.0 ** rng.uniform(-30, 30, size=3000)).tolist()
    for x in xs:
        s = N.sfloat_from_float(x)
        exact = Fraction(x)
        assert (1 << 15) <= s.m < (1 << 16)
        assert s == ref_sfloat(exact), x
        assert abs(frac_value(s) - exact) <= exact * Fraction(1, 1 << 16)
    assert N.sfloat_from_float(1.0) == SFloat(1 << 15, -15)
    assert N.sfloat_from_float(0.5) == SFloat(1 << 15, -16)
    assert N.sfloat_from_float(3.0) == SFloat(49152, -14)
    assert N.sfloat_from_float(0.0) == SFLOAT_ZERO
    assert N.sfloat_from_float(-0.0) == SFLOAT_ZERO
    # an exact half rounds up: f * 2**16 = 32768.5
    assert N.sfloat_from_float(0.5 + 2.0**-17) == SFloat(32769, -16)
    for bad in (-1.0, math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError):
            N.sfloat_from_float(bad)


@pytest.mark.parametrize("k", range(-8, 9))
def test_sfloat_from_float_rounding_overflow(k: int) -> None:
    # mantissa would round to 2**16 -> {2**15, e + 1}, value exactly 2**k
    s = N.sfloat_from_float((1 - 2.0**-18) * 2.0**k)
    assert s == SFloat(1 << 15, k - 15)
    assert frac_value(s) == Fraction(2) ** k


def test_sfloat_mul_edge_mantissas_all_exponent_combinations() -> None:
    edges = [1 << 15, (1 << 15) + 1, (1 << 16) - 1]
    exps = [-64, -17, -1, 0, 1, 16, 63]
    for ma, mb, ea, eb in itertools.product(edges, edges, exps, exps):
        a, b = SFloat(ma, ea), SFloat(mb, eb)
        r = N.sfloat_mul(a, b)
        exact = frac_value(a) * frac_value(b)
        assert (1 << 15) <= r.m < (1 << 16)
        assert r == ref_sfloat(exact), (a, b, r)
        assert abs(frac_value(r) - exact) <= exact * Fraction(1, 1 << 16)
        # rounded once: error at most half an output ulp
        assert abs(frac_value(r) - exact) <= Fraction(2) ** (r.e - 1)


def test_sfloat_mul_random_200k_pairs() -> None:
    rng = np.random.default_rng(14)
    n = 200_000
    ma = rng.integers(1 << 15, 1 << 16, size=n)
    mb = rng.integers(1 << 15, 1 << 16, size=n)
    ea = rng.integers(-40, 40, size=n)
    eb = rng.integers(-40, 40, size=n)
    got_m = np.empty(n, dtype=np.int64)
    got_e = np.empty(n, dtype=np.int64)
    for i, (x, y, p, q) in enumerate(
        zip(ma.tolist(), mb.tolist(), ea.tolist(), eb.tolist(), strict=True)
    ):
        r = N.sfloat_mul(SFloat(x, p), SFloat(y, q))
        got_m[i], got_e[i] = r.m, r.e
    assert np.all((got_m >= (1 << 15)) & (got_m < (1 << 16)))
    # exact products fit in 32 bits; float64 evaluates the error exactly
    prod = (ma * mb).astype(np.float64)
    got = got_m.astype(np.float64) * 2.0 ** (got_e - ea - eb).astype(np.float64)
    rel = np.abs(got - prod) / prod
    assert rel.max() <= 2.0**-16
    # spot-check exactness of the rounding against the rational reference
    for i in range(0, n, 997):
        exact = Fraction(int(ma[i]) * int(mb[i])) * Fraction(2) ** int(ea[i] + eb[i])
        assert SFloat(int(got_m[i]), int(got_e[i])) == ref_sfloat(exact)


def test_sfloat_mul_exponent_gain_is_15_or_16() -> None:
    """The gain ``sfloat_mul`` adds to ``e_a + e_b`` is exhaustively 15 or 16, never 17.

    17 would need the ``p >= 2**31`` branch to round up to ``2**16``, so
    ``p >= 2**32 - 2**15``; the largest product of two canonical mantissas is
    ``(2**16 - 1)**2 = 2**32 - 2**17 + 1``, below that.  The gain is
    non-decreasing in ``m_b`` (``p`` grows, the branch and the rounding only
    move up), so ``m_b = 2**16 - 1`` is the maximum over all ``m_b`` and the
    scan below is a complete proof over both mantissas.
    """
    m_max = (1 << 16) - 1
    assert N.round_shift(m_max * m_max, 16) < 1 << 16
    gains = {N.sfloat_mul(SFloat(ma, 0), SFloat(m_max, 0)).e for ma in range(1 << 15, 1 << 16)}
    assert max(gains) == N.SFLOAT_MUL_E_MAX == 16
    assert min(N.sfloat_mul(SFloat(ma, 0), SFloat(1 << 15, 0)).e for ma in (1 << 15, m_max)) == (
        N.SFLOAT_MUL_E_MIN
    )
    assert N.SFLOAT_MUL_E_MIN == 15
    # both ways of reaching the top gain: the 2**31 branch, and the rounding
    # overflow of the branch below it
    assert N.sfloat_mul(SFloat(m_max, 0), SFloat(m_max, 0)).e == N.SFLOAT_MUL_E_MAX
    assert 32769 * 65534 < 1 << 31
    assert N.sfloat_mul(SFloat(32769, 0), SFloat(65534, 0)) == SFloat(1 << 15, N.SFLOAT_MUL_E_MAX)


def test_quant_scale_exponents_upper_bound_uses_the_reachable_gain() -> None:
    """``SCALE_MUL`` widens the bound by the gain ``sfloat_mul`` can actually reach.

    ``quant_scale_exponents`` composes the ``Sx`` window with one
    :func:`sfloat_mul`, so its widening is exactly ``[SFLOAT_MUL_E_MIN,
    SFLOAT_MUL_E_MAX]`` and not a branch the rounding rule makes unreachable.
    """
    m_max = (1 << 16) - 1
    gain_hi = max(N.sfloat_mul(SFloat(ma, 0), SFloat(m_max, 0)).e for ma in range(1 << 15, 1 << 16))
    gain_lo = N.sfloat_mul(SFloat(1 << 15, 0), SFloat(1 << 15, 0)).e
    for width in (8, 16):
        for frac_in in (0, 8, 16, 30):
            base_lo, base_hi = N.quant_scale_exponents(width, frac_in)
            for e in (-40, -1, 0, 7, 40):
                mul = SFloat(50000, e)
                lo, hi = N.quant_scale_exponents(width, frac_in, mul)
                assert (lo, hi) == (base_lo + e + gain_lo, base_hi + e + gain_hi)


def test_sfloat_mul_zero_and_one() -> None:
    s = SFloat(51234, -7)
    assert N.sfloat_mul(SFLOAT_ZERO, s) == SFLOAT_ZERO
    assert N.sfloat_mul(s, SFLOAT_ZERO) == SFLOAT_ZERO
    assert N.sfloat_mul(SFLOAT_ZERO, SFLOAT_ZERO) == SFLOAT_ZERO
    for m in (1 << 15, 40000, (1 << 16) - 1):
        assert N.sfloat_mul(SFLOAT_ONE, SFloat(m, 3)) == SFloat(m, 3)
        assert N.sfloat_mul(SFloat(m, 3), SFLOAT_ONE) == SFloat(m, 3)


# =========================================================================== lookup tables

RIGHT_END = {"exp2": 65536, "sigmoid": 32768, "rsqrt": 16384, "recip": 16384}


def _table_fn(name: str):
    one = mpmath.mpf(1)
    if name == "exp2":
        return lambda x: mpmath.power(2, x)
    if name == "sigmoid":
        return lambda x: one / (one + mpmath.exp(-x))
    if name == "rsqrt":
        return lambda x: one / mpmath.sqrt(x)
    return lambda x: one / x


def _table_x(name: str, i: int) -> mpmath.mpf:
    if name == "exp2":
        return mpmath.mpf(i) / 256
    if name == "sigmoid":
        return mpmath.mpf(i) / 32
    if name == "rsqrt":
        return 1 + mpmath.mpf(i) / 256 if i < 256 else 2 + mpmath.mpf(2 * (i - 256)) / 256
    return 1 + mpmath.mpf(i) / 256


def test_load_tables_shapes_ranges_and_checksum(tables: N.Tables) -> None:
    for name, spec in N.TABLE_SPECS.items():
        lut: Lut = getattr(tables, name)
        assert lut.name == name
        assert lut.v.shape == (spec["entries"],) and lut.dv.shape == (spec["entries"],)
        assert lut.v.dtype == np.int64 and lut.dv.dtype == np.int64
        assert 0 <= int(lut.v.min()) and int(lut.v.max()) <= 0xFFFF
        assert -(1 << 15) <= int(lut.dv.min()) and int(lut.dv.max()) <= N.I16_MAX
    assert tables.exp2.v[0] == 32768 and int(tables.exp2.v.max()) <= 65535
    assert np.all(np.diff(tables.exp2.v) > 0)
    assert tables.sigmoid.v[0] == 16384 and int(tables.sigmoid.v.max()) <= 32768
    assert np.all(np.diff(tables.sigmoid.v) >= 0)
    assert tables.rsqrt.v[0] == 32768 and tables.rsqrt.v[256] == 23170  # 2**15 / sqrt(2)
    assert int(tables.rsqrt.v.min()) > 16384 and np.all(np.diff(tables.rsqrt.v) < 0)
    assert tables.recip.v[0] == 32768 and int(tables.recip.v.min()) > 16384
    assert np.all(np.diff(tables.recip.v) < 0)
    # meta.sha256 is the hash of the compact sorted JSON of the four tables
    with open(N.LUTS_JSON, encoding="utf-8") as fh:
        raw = json.load(fh)
    payload = json.dumps(
        {k: raw[k] for k in N.TABLE_SPECS}, sort_keys=True, separators=(",", ":")
    ).encode()
    assert tables.meta["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("name", list(N.TABLE_SPECS))
def test_table_values_and_forward_differences(name: str, tables: N.Tables, mp128) -> None:
    lut: Lut = getattr(tables, name)
    f = _table_fn(name)
    n = N.TABLE_SPECS[name]["entries"]
    expect_v = [
        int(mpmath.floor(f(_table_x(name, i)) * 32768 + mpmath.mpf(1) / 2)) for i in range(n)
    ]
    assert lut.v.tolist() == expect_v
    right = {"exp2": 1, "sigmoid": 16, "rsqrt": 4, "recip": 2}[name]
    v_end = int(mpmath.floor(f(mpmath.mpf(right)) * 32768 + mpmath.mpf(1) / 2))
    assert v_end == RIGHT_END[name]
    expect_dv = [expect_v[i + 1] - expect_v[i] for i in range(n - 1)] + [v_end - expect_v[-1]]
    assert lut.dv.tolist() == expect_dv


def test_lut_interp_definition_and_validation(tables: N.Tables) -> None:
    lut = tables.recip
    for idx in (0, 17, 255):
        for frac8 in (0, 1, 127, 128, 255):
            expect = int(lut.v[idx]) + round_half_up(Fraction(int(lut.dv[idx]) * frac8, 256))
            assert int(lut.interp(idx, frac8)) == expect
    idx = np.array([0, 17, 255])
    frac8 = np.array([0, 128, 255])
    assert lut.interp(idx, frac8).tolist() == [
        int(lut.interp(int(i), int(f))) for i, f in zip(idx, frac8, strict=True)
    ]
    v = np.zeros(256, dtype=np.int64) + 40000
    dv = np.zeros(256, dtype=np.int64)
    Lut("recip", v, dv)  # valid
    with pytest.raises(ValueError):
        Lut("recip", v[:255], dv[:255])
    with pytest.raises(ValueError):
        Lut("recip", v + 0x10000, dv)
    with pytest.raises(ValueError):
        Lut("recip", v, dv + (1 << 15))
    with pytest.raises(ValueError):
        Lut("recip", v, dv - (1 << 15) - 1)
    with pytest.raises(FileNotFoundError):
        N.load_tables(N.TABLES_DIR / "does_not_exist.json")


def test_exp2_q15_dense_sweep(tables: N.Tables, mp128) -> None:
    f16 = np.arange(1 << 16, dtype=np.int64)
    got = np.asarray(N.exp2_q15(f16, tables), dtype=np.int64)
    ref = [mpmath.power(2, mpmath.mpf(i) / 65536) * 32768 for i in range(1 << 16)]
    err = q15_error(got, ref)
    assert err <= 2.0, err  # measured maximum over the domain: 0.981 LSB
    assert int(got.min()) == 32768 and int(got.max()) == 65535
    assert np.all(np.diff(got) >= 0)
    assert int(N.exp2_q15(0, tables)) == 32768
    assert int(N.exp2_q15(1 << 15, tables)) == 46341  # 2**0.5 * 32768 = 46340.95


def test_recip_q15_dense_sweep(tables: N.Tables, mp128) -> None:
    a_hi = list(range(1 << 15, 1 << 16))
    got = np.array([N.recip_q15(a, tables) for a in a_hi], dtype=np.int64)
    ref = [mpmath.mpf(32768 * 32768) / a for a in a_hi]
    err = q15_error(got, ref)
    assert err <= 2.0, err  # measured maximum: 1.032 LSB
    assert int(got.max()) == 32768 and got[0] == 32768
    assert int(got.min()) == 16384  # 1/(2 - 2**-15) rounds to 16384
    assert np.all(np.diff(got) <= 0)
    for bad in ((1 << 15) - 1, 1 << 16, 0):
        with pytest.raises(ValueError):
            N.recip_q15(bad, tables)


def test_rsqrt_q15_dense_sweep(tables: N.Tables, mp128) -> None:
    mq = list(range(1 << 16, 1 << 18))
    got = np.array([N.rsqrt_q15(m, tables) for m in mq], dtype=np.int64)
    ref = [mpmath.mpf(32768) / mpmath.sqrt(mpmath.mpf(m) / 65536) for m in mq]
    err = q15_error(got, ref)
    assert err <= 2.0, err  # measured maximum: 1.052 LSB (segment 1), 0.981 LSB (segment 0)
    assert got[0] == 32768 and int(got.min()) == 16384
    assert np.all(np.diff(got) <= 0)
    # segment boundary at m = 2: continuous, and the first segment-1 entry is v[256]
    assert N.rsqrt_q15(1 << 17, tables) == 23170
    assert abs(N.rsqrt_q15((1 << 17) - 1, tables) - 23170) <= 1
    for bad in ((1 << 16) - 1, 1 << 18):
        with pytest.raises(ValueError):
            N.rsqrt_q15(bad, tables)


def test_sigmoid_q15_dense_sweep_frac13(tables: N.Tables, mp128) -> None:
    frac = 13
    xq = np.arange(0, (16 << frac) + 1, dtype=np.int64)
    got = N.sigmoid_q15(xq, frac, tables)
    ref = [mpmath.mpf(32768) / (1 + mpmath.exp(-mpmath.mpf(x) / (1 << frac))) for x in xq.tolist()]
    err = q15_error(got, ref)
    assert err <= 2.0, err  # measured maximum: 1.248 LSB (at x = 1.518)
    assert np.all(np.diff(got) >= 0)
    assert got[0] == 16384 and got[-1] == 32768


def test_sigmoid_q15_symmetry_saturation_and_frac_rule(tables: N.Tables) -> None:
    rng = np.random.default_rng(15)
    for frac in (13, 14, 16, 20, 24):
        x = rng.integers(-(20 << frac), 20 << frac, size=20000, dtype=np.int64)
        x = np.concatenate([x, np.array([0, 1, -1, (16 << frac) - 1, 16 << frac, -(16 << frac)])])
        pos, neg = N.sigmoid_q15(x, frac, tables), N.sigmoid_q15(-x, frac, tables)
        assert np.all(pos + neg == 32768)
        assert np.all(pos[x >= (16 << frac)] == 32768)
        assert np.all(pos[x <= -(16 << frac)] == 0)
        assert np.all((pos >= 0) & (pos <= 32768))
        assert int(N.sigmoid_q15(np.array([0]), frac, tables)[0]) == 16384
    for frac in (12, 5, 0):
        with pytest.raises(ValueError):
            N.sigmoid_q15(np.array([1]), frac, tables)


def test_sigmoid_q15_higher_frac_truncates_input_bits(tables: N.Tables) -> None:
    # bits below frac-13 are dropped before interpolation: at most 2**-13 of input error,
    # slope <= 1/4, so <= 1 extra LSB on top of the 1.25 LSB table error.
    for frac, bound in ((16, 2.0), (20, 2.25)):
        lo, hi = -(16 << frac), 16 << frac
        rng = np.random.default_rng(frac)
        xq = rng.integers(lo, hi, size=1_000_000, dtype=np.int64)
        if frac == 16:
            xq = np.arange(lo, hi, dtype=np.int64)
        got = N.sigmoid_q15(xq, frac, tables).astype(np.float64)
        ref = 32768.0 / (1.0 + np.exp(-xq / 2.0**frac))
        err = float(np.max(np.abs(got - ref)))
        assert err <= bound, (frac, err)  # measured: 1.911 LSB at frac 16, 1.991 LSB at frac 20


# =========================================================================== requant


def requant_exact(acc: int, sw: SFloat, sx: SFloat, s1: int, sbias: int) -> Fraction:
    """Real value the requant pipe approximates, in output LSBs."""
    return Fraction(acc * sw.m * sx.m) * Fraction(2) ** (sw.e + sx.e - sbias - s1)


def test_requant_within_one_lsb_when_stage2_shift_is_wide() -> None:
    rng = np.random.default_rng(21)
    worst = Fraction(0)
    checked = 0
    while checked < 3000:
        frac_out = int(rng.integers(8, 20))
        sw, sx = rand_sfloat(rng, -30, -8), rand_sfloat(rng, -30, -8)
        sbias0 = N.sbias_for(frac_out, 0)
        s_max = N.requant_shift(sw, sx, sbias0)  # S at s1 = 0
        if s_max < 16:
            continue
        s1 = int(rng.integers(0, min(16, s_max - 16) + 1))
        sbias = N.sbias_for(frac_out, s1)
        shift = N.requant_shift(sw, sx, sbias)
        assert 16 <= shift <= 62
        acc_bits = int(rng.integers(2, 40))  # signed width of the accumulator
        acc = int(rng.integers(-(1 << (acc_bits - 1)), 1 << (acc_bits - 1)))
        if abs(acc * sw.m) >> s1 >= 1 << 39:
            continue
        exact = requant_exact(acc, sw, sx, s1, sbias)
        if abs(exact) >= (1 << 31) - 1:
            continue
        st = Stats()
        y = N.requant(acc, sw, sx, s1, sbias, stats=st)
        assert st == Stats()
        err = abs(Fraction(y) - exact)
        worst = max(worst, err)
        assert err < 1, (acc, sw, sx, s1, sbias)
        assert abs(y - round_half_up(exact)) <= 1
        checked += 1
    assert worst <= HALF + Fraction((1 << 16) - 1, 1 << 17)  # 0.5 + 0.5 * Sx_m * 2**-16


def test_requant_general_bound_with_narrow_stage2_shift() -> None:
    rng = np.random.default_rng(22)
    checked = 0
    while checked < 3000:
        s1 = int(rng.integers(0, 17))
        sw, sx = rand_sfloat(rng, -30, -5), rand_sfloat(rng, -30, -5)
        shift = int(rng.integers(0, 16))
        sbias = shift + sw.e + sx.e
        # keep |y| < 2**31 and |t| < 2**39: t < 2**(15 + S)
        t_max = min((1 << 39) - 1, 1 << (14 + shift))
        acc_max = (t_max << s1) // sw.m
        if acc_max < 1:
            continue
        acc = int(rng.integers(-acc_max, acc_max + 1))
        exact = requant_exact(acc, sw, sx, s1, sbias)
        if abs(exact) >= (1 << 31) - 1:
            continue
        st = Stats()
        y = N.requant(acc, sw, sx, s1, sbias, stats=st)
        assert st == Stats(), (acc, sw, sx, s1, sbias)
        stage1 = Fraction(sx.m, 1 << (shift + 1))  # 0.5 * Sx_m * 2**-S
        assert abs(Fraction(y) - exact) <= HALF + stage1
        assert abs(y - round_half_up(exact)) <= 1 + stage1
        checked += 1


def test_requant_arrays_match_scalars_and_stats_add_up() -> None:
    rng = np.random.default_rng(23)
    sw, sx = SFloat(51234, -20), SFloat(40001, -22)
    s1, sbias = 4, N.sbias_for(14, 4)
    acc = rng.integers(-(1 << 39), 1 << 39, size=64, dtype=np.int64)
    acc[:4] = [(1 << 39) - 1, -(1 << 39), 0, 1]
    old = rng.integers(-(1 << 30), 1 << 30, size=64, dtype=np.int64)
    st = Stats()
    y = N.requant(acc, sw, sx, s1, sbias, bias_q=1234, old=old, stats=st)
    st_scalar = Stats()
    ys = [
        N.requant(int(a), sw, sx, s1, sbias, bias_q=1234, old=int(o), stats=st_scalar)
        for a, o in zip(acc.tolist(), old.tolist(), strict=True)
    ]
    assert y.dtype == np.int64 and y.tolist() == ys
    assert st == st_scalar


def test_requant_zero_scale_rule() -> None:
    acc = np.array([123456789, -5, 0, 1 << 38], dtype=np.int64)
    old = np.array([10, -20, 30, -40], dtype=np.int64)
    for sw, sx in ((SFLOAT_ZERO, SFloat(40000, -20)), (SFloat(40000, -20), SFLOAT_ZERO)):
        st = Stats()
        # an absurd sbias would be an ERR_SHIFT event if the shift were evaluated
        y = N.requant(acc, sw, sx, 16, 10_000, bias_q=7, old=old, stats=st)
        assert y.tolist() == (old + 7).tolist()
        assert st == Stats()
        assert N.requant(5, sw, sx, 16, -10_000, stats=st) == 0
        assert N.requant(5, sw, sx, 16, -10_000, bias_q=-3, old=9, stats=st) == 6
        assert st == Stats()


def test_requant_shift_clamp_low_counts_err_shift() -> None:
    sw, sx = SFloat(40000, -3), SFloat(50000, -4)
    s1 = 2
    acc = np.array([1000, -777, 5, 0], dtype=np.int64)
    sbias = -20  # S = -13 -> clamped to 0
    assert N.requant_shift(sw, sx, sbias) == -13
    st = Stats()
    y = N.requant(acc, sw, sx, s1, sbias, stats=st)
    assert st.err_shift == acc.size
    expect = [
        min(max(round_half_up(Fraction(int(a) * sw.m, 1 << s1)) * sx.m, I32_MIN), N.I32_MAX)
        for a in acc.tolist()
    ]
    assert y.tolist() == expect
    assert st.sat == sum(
        1
        for a in acc.tolist()
        if abs(round_half_up(Fraction(int(a) * sw.m, 1 << s1)) * sx.m) > N.I32_MAX
    )
    st2 = Stats()
    assert N.requant(1000, sw, sx, s1, sbias, stats=st2) == expect[0]
    assert st2.err_shift == 1


@pytest.mark.parametrize(("shift", "err_per_element"), [(63, 0), (64, 1), (200, 1)])
def test_requant_shift_63_and_clamp_high(shift: int, err_per_element: int) -> None:
    # The documented hardware range is [0, 63]: S = 63 is valid and S > 63 clamps to 63 with an
    # ERR_SHIFT count; either way |t * Sx_m| < 2**56 so the stage-2 result is exactly 0.
    sw, sx = SFloat(40000, -40), SFloat(50000, -40)
    sbias = shift + sw.e + sx.e
    assert N.requant_shift(sw, sx, sbias) == shift
    acc = np.array([1000, -777, 5, 0], dtype=np.int64)
    st = Stats()
    y = N.requant(acc, sw, sx, 0, sbias, bias_q=3, stats=st)
    assert y.tolist() == [3, 3, 3, 3]
    assert st.err_shift == err_per_element * acc.size
    assert st.sat == 0


def test_requant_sat40_and_sat32_counts() -> None:
    sw, sx = SFloat((1 << 16) - 1, -20), SFloat(40000, -20)
    # stage 1 overflows 40 bits: acc = 2**39 - 1 times a full mantissa with s1 = 0
    acc = (1 << 39) - 1
    sbias = 30 + sw.e + sx.e  # S = 30, keeps stage 2 inside int32
    st = Stats()
    y = N.requant(acc, sw, sx, 0, sbias, stats=st)
    assert st.sat == 1 and st.err_shift == 0
    assert y == round_half_up(Fraction(((1 << 39) - 1) * sx.m, 1 << 30))
    # stage 2 overflows int32 for some elements only: count equals the clipped elements
    acc = np.array([1 << 20, -(1 << 20), 1 << 10, 0, 1 << 30], dtype=np.int64)
    s1 = 16
    sbias = sw.e + sx.e  # S = 0: y = t * Sx_m
    st = Stats()
    y = N.requant(acc, sw, sx, s1, sbias, stats=st)
    exact = [requant_exact(int(a), sw, sx, s1, sbias) for a in acc.tolist()]
    clipped = [e > N.I32_MAX or e < I32_MIN for e in exact]
    assert clipped == [True, True, False, False, True]
    assert st.sat == sum(clipped) and st.err_shift == 0
    for yy, e, c in zip(y.tolist(), exact, clipped, strict=True):
        if c:
            assert yy == (N.I32_MAX if e > 0 else I32_MIN)
        else:
            assert abs(Fraction(yy) - e) <= HALF + Fraction(sx.m, 2)  # S = 0: stage-1 term


def test_requant_bias_and_accumulate_saturate_in_order() -> None:
    sw, sx = SFloat(1 << 15, -15), SFloat(1 << 15, -15)  # both exactly 1.0
    s1 = 15  # t = acc exactly (fits 40 bits); S = 15 undoes the second mantissa: y = acc
    sbias = N.sbias_for(0, s1)
    assert N.requant_shift(sw, sx, sbias) == 15
    acc = N.I32_MAX - 10
    st = Stats()
    assert N.requant(acc, sw, sx, s1, sbias, stats=st) == acc and st == Stats()
    # bias pushes past the top: saturate, then the accumulate add pulls back down
    assert N.requant(acc, sw, sx, s1, sbias, bias_q=20, old=-5, stats=st) == N.I32_MAX - 5
    assert st.sat == 1
    # negative direction
    st = Stats()
    assert N.requant(I32_MIN + 2, sw, sx, s1, sbias, bias_q=-5, stats=st) == I32_MIN
    assert st.sat == 1


def test_sbias_for_and_requant_shift() -> None:
    assert N.sbias_for(14, 16) == -30
    assert N.sbias_for(16, 4, 24) == 4
    assert N.requant_shift(SFloat(40000, -20), SFloat(40000, -21), -30) == 11
    assert N.requant_shift(SFLOAT_ONE, SFLOAT_ONE, 0) == 30


def test_embed_dequant_is_a_single_rounding() -> None:
    rng = np.random.default_rng(24)
    for frac_x in (14, 16):
        for _ in range(50):
            q = rng.integers(-127, 128, size=96, dtype=np.int64)
            sw = rand_sfloat(rng, -30, -12)
            st = Stats()
            y = N.embed_dequant(q, sw, frac_x, stats=st)
            assert st == Stats()
            exact = [Fraction(int(v) * sw.m) * Fraction(2) ** (sw.e + frac_x) for v in q.tolist()]
            assert y.tolist() == [round_half_up(e) for e in exact]
            assert (
                max(abs(Fraction(int(g)) - e) for g, e in zip(y.tolist(), exact, strict=True))
                <= HALF
            )
    assert N.embed_dequant(np.zeros(8, dtype=np.int64), SFloat(40000, -20), 16).tolist() == [0] * 8
    assert N.embed_dequant(np.array([5, -5], dtype=np.int64), SFLOAT_ZERO, 16).tolist() == [0, 0]


def test_choose_s1_respects_both_documented_bounds() -> None:
    rng = np.random.default_rng(25)
    for _ in range(2000):
        acc_bits = int(rng.integers(16, 41))
        frac_out = int(rng.integers(8, 20))
        sw_e_max = int(rng.integers(-32, -4))
        sx_e_max = int(rng.integers(-32, -4))
        s1 = N.choose_s1(acc_bits, frac_out, sw_e_max, sx_e_max)
        s1_min = max(0, acc_bits + 16 - 40)
        s1_prec = -(frac_out + sw_e_max + sx_e_max) - 16
        assert s1 >= s1_min
        if s1_prec >= s1_min:
            assert s1 <= s1_prec
            # stage-2 shift at the largest reachable exponents stays >= 16
            assert -(frac_out + s1) - sw_e_max - sx_e_max >= 16
        else:
            assert s1 == s1_min
        # 40-bit bound: a full accumulator of this signed width times a full mantissa fits
        t = (((1 << (acc_bits - 1)) - 1) * ((1 << 16) - 1)) >> s1
        assert t < 1 << 39
    assert N.choose_s1(40, 14, -25, -25) == 16
    assert N.choose_s1(40, 14, -5, -3) == 16  # precision bound infeasible: 40-bit bound wins
    assert N.choose_s1(30, 14, -5, -3) == 6


# =========================================================================== VQUANT


@pytest.mark.parametrize("width", [8, 16])
def test_quant_scale_and_dequantization_error(width: int, tables: N.Tables) -> None:
    rng = np.random.default_rng(31 + width)
    lim = (1 << (width - 1)) - 1
    tol = Fraction(1, 1 << (width - 1)) + Fraction(2, 1 << 15)
    for it in range(300):
        n = int(rng.integers(1, 260))
        mag = int(rng.integers(0, 31))
        x = rng.integers(-(1 << mag), (1 << mag) + 1, size=n, dtype=np.int64)
        if it % 3 == 0:
            x[rng.integers(n)] = (1 << mag) if it % 2 else -(1 << mag)
        frac_in = int(rng.integers(8, 20))
        st = Stats()
        q, sx = N.quant(x, width, frac_in, tables, stats=st)
        a = N.absmax(x)
        if a == 0:
            assert sx == SFLOAT_ZERO and not q.any() and st.clip == 0
            continue
        assert q.dtype == np.int64 and int(np.max(np.abs(q))) <= lim
        a_eff = a + (a >> (width - 1)) + 1
        e_a = N.bitlen(a_eff) - 16
        a_hi = a_eff >> e_a if e_a >= 0 else a_eff << -e_a
        assert (1 << 15) <= a_hi < (1 << 16)
        # exact scale: (top 16 bits of a_eff) * 2**e_a / 2**(w-1) in real units
        assert sx == SFloat(a_hi, e_a - (width - 1) - frac_in)
        a_used = Fraction(a_hi) * Fraction(2) ** e_a
        assert frac_value(sx) == a_used / (1 << (width - 1)) / Fraction(2) ** frac_in
        if a_eff < 1 << 16:
            assert a_used == a_eff
        else:
            assert 0 <= a_eff - a_used < Fraction(2) ** e_a
        # dequantized error relative to absmax
        deq = [Fraction(int(v)) * frac_value(sx) for v in q.tolist()]
        worst = max(
            abs(d - Fraction(int(v)) / Fraction(2) ** frac_in)
            for d, v in zip(deq, x.tolist(), strict=True)
        )
        assert worst <= tol * Fraction(a) / Fraction(2) ** frac_in, (a, width)
        # for a >= 2**(w-1) the absmax element lands on the top level or one below
        # (two below at w = 16, where the reciprocal table's error is two levels wide);
        # no clip for w = 8
        if a >= 1 << (width - 1):
            assert abs(int(q[int(np.argmax(np.abs(x)))])) >= lim - (1 if width == 8 else 2)
        if width == 8:
            assert st.clip == 0


@pytest.mark.parametrize("width", [8, 16])
def test_quant_scale_exponents_bound_every_reachable_scale(width: int, tables: N.Tables) -> None:
    """The static bound holds for every magnitude the hardware can present, and is attained.

    ``a`` is a u32 -- an int32 absmax, or a tracked absmax an SREG word holds --
    so ``e_a = bitlen(a_eff) - 16`` runs over ``[-14, 17]`` and nothing else,
    with and without the ``SCALE_MUL`` constant.
    """
    rng = np.random.default_rng(90 + width)
    x = np.array([7, -3, 0, 1], dtype=np.int64)
    amaxes = (1, 2, 3, 12345, (1 << 30) - 1, 1 << 31, (1 << 32) - 1)
    for frac_in in (0, 8, 16, 30):
        lo, hi = N.quant_scale_exponents(width, frac_in)
        seen = [N.quant(x, width, frac_in, tables, amax=a)[1].e for a in amaxes]
        assert min(seen) == lo and max(seen) == hi
        assert (lo, hi) == (
            N.QUANT_E_A_MIN - (width - 1) - frac_in,
            N.QUANT_E_A_MAX - (width - 1) - frac_in,
        )
        for _ in range(50):
            mul = N.sfloat_from_float(float(rng.uniform(1e-6, 1e6)))
            mlo, mhi = N.quant_scale_exponents(width, frac_in, mul)
            got = [N.quant(x, width, frac_in, tables, amax=a, scale_mul=mul)[1].e for a in amaxes]
            assert mlo <= min(got) and max(got) <= mhi, (frac_in, mul, got, mlo, mhi)
    assert N.quant_scale_exponents(8, 0, SFLOAT_ZERO) == N.quant_scale_exponents(8, 0)
    with pytest.raises(ValueError):
        N.quant_scale_exponents(12, 0)


def test_quant_power_of_two_absmax_is_exact_rounding(tables: N.Tables) -> None:
    # absmax = 2**k: the scale basis a_eff = 2**k + 2**(k-w+1) keeps the absmax element on
    # +-(2**(w-1) - 1) and q equals round_half_up(x * 2**(w-1) / a_used) within the table error.
    rng = np.random.default_rng(33)
    for width in (8, 16):
        lim = (1 << (width - 1)) - 1
        for k in (0, 3, 7, 14, 15, 16, 20, 27, 30):
            x = rng.integers(-(1 << k), (1 << k) + 1, size=200, dtype=np.int64)
            x[0], x[1] = 1 << k, -(1 << k)
            st = Stats()
            q, sx = N.quant(x, width, 12, tables, stats=st)
            a_eff = (1 << k) + ((1 << k) >> (width - 1)) + 1
            e_a = N.bitlen(a_eff) - 16
            a_hi = a_eff >> e_a if e_a >= 0 else a_eff << -e_a
            a_used = Fraction(a_hi) * Fraction(2) ** e_a
            assert frac_value(sx) == a_used / (1 << (width - 1)) / Fraction(2) ** 12
            raw = [
                round_half_up(Fraction(int(v) * (1 << (width - 1))) / a_used) for v in x.tolist()
            ]
            assert all(
                abs(int(qq) - min(max(r, -lim), lim)) <= 1
                for qq, r in zip(q.tolist(), raw, strict=True)
            )
            if k >= width + 2:
                assert abs(int(q[0])) == lim and abs(int(q[1])) == lim
            elif k >= width - 1:
                assert abs(int(q[0])) >= lim - 1 and abs(int(q[1])) >= lim - 1
            if width == 8:
                assert st.clip == 0  # the +1 in a_eff keeps 127.5 out of reach
            else:
                assert st.clip <= 2  # int16: only the table error can push the maxima over


def test_quant_small_amax_negative_exponent_exact_scale(tables: N.Tables) -> None:
    for a in (1, 2, 3, 5, 100, 32767, 32768):
        x = np.array([a, -a // 2, 0], dtype=np.int64)
        for width in (8, 16):
            q, sx = N.quant(x, width, 16, tables)
            a_eff = a + (a >> (width - 1)) + 1
            assert frac_value(sx) == Fraction(a_eff, 1 << (width - 1)) / Fraction(2) ** 16
            expected = round_half_up(Fraction(a * (1 << (width - 1)), a_eff))
            assert abs(abs(int(q[0])) - expected) <= 1  # table error at most one step
            if a >= 1 << (width - 1):
                assert abs(int(q[0])) >= (1 << (width - 1)) - 2


def test_quant_groups_equals_per_group_quant(tables: N.Tables) -> None:
    rng = np.random.default_rng(34)
    x = rng.integers(-(1 << 20), 1 << 20, size=14 * 64, dtype=np.int64)
    x[64:128] = 0  # one all-zero head
    c = N.sfloat_from_float(0.18)
    st = Stats()
    q, scales = N.quant_groups(x, 64, 8, 14, tables, scale_mul=c, stats=st)
    assert q.shape == x.shape and len(scales) == 14
    st_ref = Stats()
    for h in range(14):
        qh, sh = N.quant(x[h * 64 : (h + 1) * 64], 8, 14, tables, scale_mul=c, stats=st_ref)
        assert q[h * 64 : (h + 1) * 64].tolist() == qh.tolist() and scales[h] == sh
    assert scales[1] == SFLOAT_ZERO and st == st_ref
    with pytest.raises(ValueError):
        N.quant_groups(x[:100], 64, 8, 14, tables)


# =========================================================================== RMSNorm


def _rmsnorm_ref(x: np.ndarray, gamma_q: np.ndarray, gamma_e: int, eps: float, frac_x: int):
    xr = N.from_fixed(x, frac_x)
    return xr / math.sqrt(float(np.mean(xr * xr)) + eps) * (gamma_q * 2.0**gamma_e)


def test_rmsnorm_vs_float64_reference(tables: N.Tables) -> None:
    rng = np.random.default_rng(41)
    worst = 0.0
    for it in range(240):
        d, eps = (896, 1e-6) if it % 2 else (576, 1e-5)
        frac_x = 14 if it % 4 < 2 else 16
        cap = (1 << 30) / 2**frac_x  # keep the fixed-point input inside int32
        scale = 10 ** rng.uniform(-2, math.log10(8000))
        xr = rng.standard_normal(d) * scale
        if it % 3 == 0:
            xr[rng.integers(d)] *= 30  # a single outlier channel
        xr = np.clip(xr, -cap, cap)
        x = N.to_fixed(xr, frac_x)
        gamma = rng.uniform(0.05, 4.0, size=d) * rng.choice([-1, 1], size=d)
        gq, ge = N.quantize_gamma(gamma)
        st = Stats()
        y = N.rmsnorm(
            x,
            gq,
            ge,
            N.eps_const(eps, d, frac_x),
            N.sfloat_from_float(math.sqrt(d)),
            frac_x,
            tables,
            st,
        )
        assert st == Stats() and y.dtype == np.int64
        ref = _rmsnorm_ref(x, gq, ge, eps, frac_x)
        err = float(np.max(np.abs(N.from_fixed(y, frac_x) - ref)))
        rel = err / float(np.max(np.abs(ref)))
        worst = max(worst, rel)
        assert rel <= 2.0**-12, (it, rel)
    assert worst <= 2.0**-12  # measured: 0.23 * 2**-12


def test_rmsnorm_scale_invariance(tables: N.Tables) -> None:
    rng = np.random.default_rng(42)
    d, frac_x = 896, 14
    xr = rng.standard_normal(d) * 3.0
    gq, ge = N.quantize_gamma(rng.uniform(0.5, 1.5, size=d))
    sqrt_d = N.sfloat_from_float(math.sqrt(d))
    eps_c = N.eps_const(1e-6, d, frac_x)
    y1 = N.rmsnorm(N.to_fixed(xr, frac_x), gq, ge, eps_c, sqrt_d, frac_x, tables)
    y2 = N.rmsnorm(N.to_fixed(xr * 64, frac_x), gq, ge, eps_c, sqrt_d, frac_x, tables)
    assert float(np.max(np.abs(y1 - y2))) <= 2.0**-11 * float(np.max(np.abs(y1))) + 1


def test_rmsnorm_eps_dominated_tiny_input_is_finite_and_accurate(tables: N.Tables) -> None:
    d, frac_x, eps = 896, 16, 1e-6
    x = np.array([1, -2, 3, 0] * (d // 4), dtype=np.int64)  # ~1e-5 real, rms**2 << eps
    gq, ge = N.quantize_gamma(np.ones(d))
    y = N.rmsnorm(
        x, gq, ge, N.eps_const(eps, d, frac_x), N.sfloat_from_float(math.sqrt(d)), frac_x, tables
    )
    assert np.all(np.isfinite(y)) and int(np.max(np.abs(y))) < 1 << 20
    ref = _rmsnorm_ref(x, gq, ge, eps, frac_x)
    assert (
        float(np.max(np.abs(N.from_fixed(y, frac_x) - ref)))
        <= 2.0**-12 * float(np.max(np.abs(ref))) + 2.0**-frac_x
    )
    # zero input with a zero eps constant: all zeros, no division by zero
    assert not N.rmsnorm(
        np.zeros(d, dtype=np.int64), gq, ge, 0, N.sfloat_from_float(math.sqrt(d)), frac_x, tables
    ).any()


@pytest.mark.parametrize(
    ("sqrt_e", "s1", "clamped"),
    [(10, -8, True), (0, 2, False), (-61, 63, False), (-62, 64, True), (-90, 92, True)],
)
def test_rmsnorm_shift_clamps_at_both_ends(
    tables: N.Tables, sqrt_e: int, s1: int, clamped: bool
) -> None:
    """``S1`` is a 6-bit shift: outside ``[0, 63]`` it clamps and counts, at either end.

    ``x`` has absmax 1 and ``ss' = n``, so ``sh = 0``, ``e = 2`` and
    ``Rc = {2**15, sqrt_e}``: the constant alone places ``S1 = 2 - sqrt_e``.
    63 is the last shift the field carries and 64 the first one past it, and
    both the requant and ``qcore_vpu_scalar`` clamp and count the same way.
    """
    n = 16
    x = np.array([1, -1] * (n // 2), dtype=np.int64)
    gq = np.ones(n, dtype=np.int64)
    st = Stats()
    y = N.rmsnorm(x, gq, 0, 0, SFloat(1 << 15, sqrt_e), 0, tables, st)
    assert st.err_shift == (n if clamped else 0)
    assert y.tolist() == N.round_shift(x * (1 << 15), min(max(s1, 0), N.SHIFT_MAX)).tolist()
    assert st.sat == 0


def test_rmsnorm_rejects_positive_gamma_exponent(tables: N.Tables) -> None:
    d = 64
    x = np.arange(d, dtype=np.int64) * 1000
    gq = np.ones(d, dtype=np.int64) * 16384
    with pytest.raises(ValueError):
        N.rmsnorm(x, gq, 1, N.eps_const(1e-6, d, 14), N.sfloat_from_float(8.0), 14, tables)
    N.rmsnorm(x, gq, 0, N.eps_const(1e-6, d, 14), N.sfloat_from_float(8.0), 14, tables)


def test_quantize_gamma_round_trip_and_minimal_exponent() -> None:
    rng = np.random.default_rng(43)
    cases = [rng.uniform(-3, 3, size=200) for _ in range(20)]
    cases += [np.array([1.0, -0.5, 0.25]), np.array([2 - 2.0**-16, 1.0]), np.array([1e-3, -2e-3])]
    cases += [np.array([32767.0, 1.0]), np.array([0.7])]
    for g in cases:
        q, e = N.quantize_gamma(g)
        assert q.dtype == np.int64 and int(np.max(np.abs(q))) <= N.I16_MAX
        assert float(np.max(np.abs(g - q * 2.0**e))) <= 2.0 ** (e - 1)
        # one exponent lower would not fit int16
        assert float(np.max(np.abs(np.floor(g / 2.0 ** (e - 1) + 0.5)))) > N.I16_MAX
        if float(np.max(np.abs(g))) < 1 << 15:
            assert e <= 0
    q, e = N.quantize_gamma(np.zeros(5))
    assert not q.any() and e == 0
    with pytest.raises(ValueError):
        N.quantize_gamma(np.array([40000.0, -3.0]))


def test_eps_const_matches_exact_rounding() -> None:
    for eps, d, frac_x in ((1e-6, 896, 16), (1e-6, 896, 14), (1e-5, 576, 14), (1e-5, 576, 16)):
        exact = Fraction(eps) * d * Fraction(2) ** (2 * frac_x)
        assert N.eps_const(eps, d, frac_x) == round_half_up(exact)
    assert N.eps_const(1e-6, 896, 16) == 3848291


# =========================================================================== RoPE


def test_rope_matches_big_integer_reference(tables: N.Tables) -> None:
    rng = np.random.default_rng(51)
    tab = N.load_rope_table(1e6)
    for pos in (0, 1, 2, 77, 1023, 2047):
        cos_row, sin_row = tab[pos, 0], tab[pos, 1]
        x = rng.integers(I32_MIN, N.I32_MAX + 1, size=64 * 5, dtype=np.int64)
        st = Stats()
        y = N.rope(x, cos_row, sin_row, stats=st)
        assert y.shape == x.shape and y.dtype == np.int64
        expect = np.empty_like(x)
        clipped = 0
        for h in range(5):
            for i in range(32):
                a, b = int(x[h * 64 + i]), int(x[h * 64 + 32 + i])
                c, s = int(cos_row[i]), int(sin_row[i])
                ra = round_half_up(Fraction(a * c - b * s, 1 << 14))
                rb = round_half_up(Fraction(b * c + a * s, 1 << 14))
                for j, r in ((h * 64 + i, ra), (h * 64 + 32 + i, rb)):
                    clipped += r < I32_MIN or r > N.I32_MAX
                    expect[j] = min(max(r, I32_MIN), N.I32_MAX)
        assert y.tolist() == expect.tolist()
        assert st.sat == clipped
        # float64 rotation with the table values: only the final rounding separates them
        xh = x.reshape(5, 64).astype(np.float64)
        a, b = xh[:, :32], xh[:, 32:]
        fa = (a * cos_row - b * sin_row) / 2**14
        fb = (b * cos_row + a * sin_row) / 2**14
        f = np.concatenate([fa, fb], axis=1).reshape(-1)
        inside = (f > I32_MIN) & (f < N.I32_MAX)
        assert float(np.max(np.abs(y[inside] - f[inside]))) <= 0.5


def test_rope_identity_at_position_zero_heads_independent_and_shapes() -> None:
    rng = np.random.default_rng(52)
    tab = N.load_rope_table(1e5)
    x = rng.integers(-(1 << 24), 1 << 24, size=64 * 3, dtype=np.int64)
    assert N.rope(x, tab[0, 0], tab[0, 1]).tolist() == x.tolist()
    y = N.rope(x, tab[500, 0], tab[500, 1])
    for h in range(3):
        yh = N.rope(x[h * 64 : (h + 1) * 64], tab[500, 0], tab[500, 1])
        assert y[h * 64 : (h + 1) * 64].tolist() == yh.tolist()
    with pytest.raises(ValueError):
        N.rope(x[:100], tab[1, 0], tab[1, 1])
    with pytest.raises(ValueError):
        N.rope(x, tab[1, 0][:31], tab[1, 1])
    with pytest.raises(ValueError):
        N.rope(x, tab[1, 0], tab[1, 1][:16])
    assert N.rope(x[:96], tab[1, 0][:24], tab[1, 1][:24], head_dim=48).shape == (96,)


def test_rope_saturation_is_counted() -> None:
    x = np.zeros(64, dtype=np.int64)
    x[0], x[32] = N.I32_MAX, I32_MIN  # a = max, b = min
    cos_row = np.full(32, 11585, dtype=np.int64)  # ~ 1/sqrt(2) in Q1.14
    sin_row = np.full(32, 11585, dtype=np.int64)
    st = Stats()
    y = N.rope(x, cos_row, sin_row, stats=st)
    assert y[0] == N.I32_MAX  # (a - b) * 0.707 > 2**31
    assert st.sat == 1
    assert y[32] == round_half_up(Fraction((I32_MIN + N.I32_MAX) * 11585, 1 << 14))


@pytest.mark.parametrize("theta", [1e6, 1e5])
def test_rope_tables_match_float64_and_naming(theta: float) -> None:
    path = N.rope_table_path(theta)
    assert path.name == {1e6: "rope_theta1e6_2048.npy", 1e5: "rope_theta1e5_2048.npy"}[theta]
    assert path.exists()
    raw = np.load(path)
    assert raw.dtype == np.int16 and raw.shape == (2048, 2, 32)
    tab = N.load_rope_table(theta)
    assert tab.dtype == np.int64 and tab.shape == (2048, 2, 32)
    assert int(tab.min()) >= -16384 and int(tab.max()) <= 16384
    assert tab[0, 0].tolist() == [16384] * 32 and not tab[0, 1].any()
    inv_freq = theta ** (-np.arange(32) * 2.0 / 64)
    ang = np.arange(2048)[:, None] * inv_freq[None, :]
    ref = np.stack(
        [np.floor(np.cos(ang) * 16384 + 0.5), np.floor(np.sin(ang) * 16384 + 0.5)], axis=1
    )
    assert float(np.max(np.abs(tab - ref))) <= 1.0  # measured: identical
    norm = tab[:, 0].astype(np.float64) ** 2 + tab[:, 1].astype(np.float64) ** 2
    assert float(np.max(np.abs(norm - 2.0**28))) <= 2 * 16384 + 1


# =========================================================================== softmax


def _softmax_ref(scores: np.ndarray, frac_s: int) -> np.ndarray:
    s = N.from_fixed(scores, frac_s)
    p = 2.0 ** (s - s.max())
    return p / p.sum()


def _softmax_check(scores, length, frac_s, vs, tables):
    """Run softmax and return (w, sreg, exact p, Sv values, per-token error, sum error).

    Errors are in units of ``2**-15 * max(Sv)``.
    """
    w, sreg = N.softmax(scores, length, frac_s, vs, tables)
    assert w.dtype == np.int64 and w.shape == scores.shape
    assert not w[length:].any() and int(w.min()) >= 0 and int(w.max()) <= 32767
    p = _softmax_ref(scores[:length], frac_s)
    sv = np.array([v.value() for v in vs[:length]])
    unit = sv.max() * 2.0**-15
    tok = np.abs(w[:length] * sreg.value() - p * sv) / unit
    tot = abs(float(np.sum(w[:length]) * sreg.value() - np.sum(p * sv))) / unit
    return w, sreg, p, sv, tok, tot


def test_softmax_random_rows_rigorous_and_statistical_bounds(tables: N.Tables) -> None:
    rng = np.random.default_rng(61)
    worst_tok = worst_sum = 0.0
    for it in range(300):
        frac_s = (16, 20, 24)[it % 3]
        length = int(rng.integers(1, 601))
        spread = rng.uniform(0.5, 20)
        s = N.to_fixed(rng.standard_normal(length) * spread, frac_s)
        scores = np.concatenate([s, rng.integers(-1000, 1000, size=5)]).astype(np.int64)
        e_max = int(rng.integers(-24, -18))
        vs = [
            SFloat(int(rng.integers(1 << 15, 1 << 16)), int(rng.integers(e_max - 5, e_max + 1)))
            for _ in range(length)
        ]
        w, sreg, p, sv, tok, tot = _softmax_check(scores, length, frac_s, vs, tables)
        assert sreg == SFloat(1 << 15, 1 + max(v.e for v in vs) - 15)
        # Rigorous per-token budget (units of 2**-15 * Sv_max): 1 for the rounding of w;
        # 3.6 for the rounding of p, the recip table (<= 1.05 LSB) and the truncation of the
        # sum's low bits; each token with d > 0 has up to 1.5 units of e_t error (table + rounding),
        # which shifts p_t by (2 + 2 * p_t * L_tr) / total with total >= 2**15 / kappa.
        p_max = float(p.max())
        l_tr = int(np.sum(s < s.max()))
        kappa = 1.0 / (1.0 / p_max - 2.0 * l_tr / 2**15)
        bound = 1.0 + (sv / sv.max()) * (3.6 + (2.0 + 2.0 * p * l_tr) * kappa)
        assert np.all(tok <= bound), (it, float(np.max(tok / bound)))
        worst_tok = max(worst_tok, float(tok.max()))
        worst_sum = max(worst_sum, tot)
        # statistical bound for this seed: rounding is random-signed, so it grows like sqrt(L)
        assert tot <= 6.0 + 2.5 * math.sqrt(length), (it, tot)
    assert worst_tok <= 64.0  # measured: 1.12 units
    assert worst_sum <= 80.0  # measured: 29.9 units


def test_softmax_exact_powers_of_two_distances(tables: N.Tables) -> None:
    # distances that are whole log2 units give e_t = 2**(15-k) exactly (no table, no floor);
    # only the reciprocal and the two roundings remain: at most ~4.6 units per token.
    rng = np.random.default_rng(62)
    worst = 0.0
    for it in range(200):
        frac_s = 20 if it % 2 else 16
        length = int(rng.integers(1, 601))
        k = rng.integers(0, 17, size=length)  # 16 -> e_t = 0
        k[rng.integers(length)] = 0
        scores = (1000 - k) << frac_s
        vs = [rand_sfloat(rng, -24, -19) for _ in range(length)]
        w, sreg, p, sv, tok, tot = _softmax_check(
            scores.astype(np.int64), length, frac_s, vs, tables
        )
        assert float(tok.max()) <= 5.0, (it, float(tok.max()))
        worst = max(worst, float(tok.max()))
        assert w[:length][k >= 16].sum() == 0  # 2**-16 and below vanish
    assert worst <= 5.0  # measured: 0.63 units


def test_softmax_length_one_and_all_equal(tables: N.Tables) -> None:
    scores = np.array([12345, -7, 99], dtype=np.int64)
    for sv in (SFloat(40000, -20), SFloat(40001, -20), SFloat(1 << 15, -3), SFloat(65534, 4)):
        st = Stats()
        w, sreg = N.softmax(scores, 1, 16, [sv], tables, stats=st)
        assert st == Stats() and w.tolist()[1:] == [0, 0]
        assert sreg == SFloat(1 << 15, 1 + sv.e - 15)
        got = Fraction(int(w[0])) * frac_value(sreg)
        if sv.m % 2 == 0:
            assert got == frac_value(sv)  # p = 1.0 exactly, w = Sv_m / 2
        else:
            assert abs(got - frac_value(sv)) == Fraction(2) ** sv.e  # half an ulp of Sv_m
    # all-equal scores share the weight equally
    for length in (2, 3, 7, 64, 333):
        scores = np.full(length + 3, -4444, dtype=np.int64)
        sv = SFloat(60000, -18)
        w, sreg = N.softmax(scores, length, 16, [sv] * length, tables)
        assert not w[length:].any() and len(set(w[:length].tolist())) == 1
        # each token: p = 1/L within 3.6 units, w rounding adds one unit of 2**(1+e)
        share = frac_value(sv) / length
        err = abs(Fraction(int(w[0])) * frac_value(sreg) - share)
        assert err <= Fraction(2) ** sv.e + Fraction(36, 10) * frac_value(sv) / (1 << 15)


def test_softmax_length_one_full_scale_mantissa_saturates_w() -> None:
    # p = 1.0 and Sv_m = 65535 give round_shift(2**23 * 65535, 24) = 32768, one above int16:
    # the value is clipped to 32767 and counted as a clip (like VQUANT), never as a saturation.
    tables = N.load_tables()
    st = Stats()
    w, sreg = N.softmax(
        np.array([5], dtype=np.int64), 1, 16, [SFloat(65535, -20)], tables, stats=st
    )
    assert w.tolist() == [32767] and sreg == SFloat(1 << 15, -34)
    assert st.sat == 0 and st.clip == 1


def test_softmax_zero_v_scales_extreme_scores_and_errors(tables: N.Tables) -> None:
    frac_s = 16
    m = 1000 << frac_s
    scores = np.array(
        [
            m,
            m - (1 << frac_s),
            m - (16 << frac_s),
            m - (15 << frac_s) - (1 << (frac_s - 1)),
            m - (100 << frac_s),
            5,
            6,
        ],
        dtype=np.int64,
    )
    vs = [
        SFloat(40000, -30),
        SFLOAT_ZERO,
        SFloat(50000, -28),
        SFloat(50000, -31),
        SFloat(50000, -29),
    ]
    st = Stats()
    w, sreg = N.softmax(scores, 5, frac_s, vs, tables, stats=st)
    assert st == Stats()
    assert w[1] == 0  # zero V scale
    # d = 16 log2 units keeps a weight of 2**-16 (a fraction of a unit); d = 100 rounds to zero
    assert w[2] <= 1 and w[4] == 0
    assert w[3] <= 1  # d = 15.5
    assert w[5:].tolist() == [0, 0] and w[0] > 0
    # e_max excludes the zero-scale token (whose canonical exponent 0 would dominate)
    assert sreg == SFloat(1 << 15, 1 + (-28) - 15)
    # the zero-scale token keeps its probability mass: token 0 gets p = 1 / (1 + 1/2)
    p0 = Fraction(int(w[0])) * frac_value(sreg) / frac_value(vs[0])
    assert abs(p0 - Fraction(2, 3)) <= Fraction(5, 1 << 15)
    # every V scale zero: zeros and the canonical zero SREG
    w0, s0 = N.softmax(scores, 3, frac_s, [SFLOAT_ZERO] * 3, tables)
    assert not w0.any() and s0 == SFLOAT_ZERO
    # zero-scale tokens with a very negative e_max still give zeros (shift below zero is masked)
    vs2 = [SFloat(40000, -60), SFLOAT_ZERO, SFloat(50000, -61)]
    w2, s2 = N.softmax(scores[:3], 3, frac_s, vs2, tables)
    assert w2[1] == 0 and w2[0] > 0 and s2 == SFloat(1 << 15, 1 - 60 - 15)
    for bad_len in (0, 8, -1):
        with pytest.raises(ValueError):
            N.softmax(scores, bad_len, frac_s, vs + vs, tables)
    with pytest.raises(ValueError):
        N.softmax(scores, 6, frac_s, vs, tables)  # fewer scales than length


def test_softmax_class_window_is_the_one_the_isa_names(tables: N.Tables) -> None:
    """``FRAC_S`` outside ``[16, 30]`` is refused by name, at both ends and to the byte's end."""
    assert (N.SOFTMAX_FRAC_MIN, N.SOFTMAX_FRAC_MAX) == (16, 30)
    scores = np.array([9 << 16, 4 << 16, 0, -5 << 16], dtype=np.int64)
    vs = [SFloat(40000, -20), SFloat(50000, -21), SFloat(60000, -19), SFloat(33000, -22)]
    sv_m = np.array([v.m for v in vs], dtype=np.int64)
    sv_e = np.array([v.e for v in vs], dtype=np.int64)
    for frac_s in (N.SOFTMAX_FRAC_MIN, 23, N.SOFTMAX_FRAC_MAX):
        w, sreg = N.softmax(scores, 4, frac_s, vs, tables)
        assert int(w[0]) > 0 and sreg == SFloat(1 << 15, 1 - 19 - 15)
        wr, m, e = N.softmax_rows(scores[None, :], np.array([4]), frac_s, sv_m, sv_e, tables)
        assert wr[0].tolist() == w.tolist() and (int(m[0]), int(e[0])) == (sreg.m, sreg.e)
    # one below the window, one above it, and the largest value the u8 field holds:
    # each is a named domain error rather than an overflow out of the clamp shift
    for frac_s in (0, N.SOFTMAX_FRAC_MIN - 1, N.SOFTMAX_FRAC_MAX + 1, 63, 255):
        with pytest.raises(ValueError, match="FRAC_S"):
            N.softmax(scores, 4, frac_s, vs, tables)
        with pytest.raises(ValueError, match="FRAC_S"):
            N.softmax_rows(scores[None, :], np.array([4]), frac_s, sv_m, sv_e, tables)


def test_softmax_refuses_a_scale_register_exponent_the_i8_cannot_hold(tables: N.Tables) -> None:
    """``SREG_out = {2**15, e_max - 14}``: ``e_max`` below ``E8_MIN + 14`` has no encoding."""
    assert (N.SOFTMAX_EMAX_MIN, N.SOFTMAX_EMAX_MAX) == (N.E8_MIN + 14, N.E8_MAX + 14)
    assert (N.SOFTMAX_EMAX_MIN, N.SOFTMAX_EMAX_MAX) == (-114, 141)
    scores = np.array([0, -(3 << 16)], dtype=np.int64)
    for e_max in (N.SOFTMAX_EMAX_MIN, -50, N.SOFTMAX_EMAX_MAX):
        vs = [SFloat(40000, e_max), SFloat(50000, e_max - 1)]
        w, sreg = N.softmax(scores, 2, 16, vs, tables)
        assert sreg == SFloat(1 << 15, e_max - 14) and N.E8_MIN <= sreg.e <= N.E8_MAX
        assert int(w[0]) > 0
    for e_max in (N.SOFTMAX_EMAX_MIN - 1, -200, N.SOFTMAX_EMAX_MAX + 1):
        vs = [SFloat(40000, e_max), SFloat(50000, e_max - 1)]
        sv_m = np.array([v.m for v in vs], dtype=np.int64)
        sv_e = np.array([v.e for v in vs], dtype=np.int64)
        with pytest.raises(ValueError, match="e_max"):
            N.softmax(scores, 2, 16, vs, tables)
        with pytest.raises(ValueError, match="e_max"):
            N.softmax_rows(scores[None, :], np.array([2]), 16, sv_m, sv_e, tables)
    # a token whose scale is the canonical zero takes no part in e_max, so it
    # cannot pull a row out of the window, and an all-zero row has no exponent
    vs = [SFloat(40000, N.SOFTMAX_EMAX_MIN), SFLOAT_ZERO]
    w, sreg = N.softmax(scores, 2, 16, vs, tables)
    assert int(w[1]) == 0 and sreg == SFloat(1 << 15, N.SOFTMAX_EMAX_MIN - 14)
    w0, s0 = N.softmax(scores, 2, 16, [SFLOAT_ZERO, SFLOAT_ZERO], tables)
    assert not w0.any() and s0 == SFLOAT_ZERO


# =========================================================================== SiLU


def test_silu_mul_vs_float64_reference(tables: N.Tables) -> None:
    rng = np.random.default_rng(71)
    worst_rel = 0.0
    for it in range(150):
        frac_gu = int(rng.integers(13, 21))
        frac_h = int(rng.integers(8, min(2 * frac_gu, 21) + 1))
        n = 400
        g = rng.uniform(-20, 20, size=n)
        u = rng.uniform(-20, 20, size=n)
        if it % 5 == 0:
            g[: n // 2] = rng.uniform(-0.5, 0.5, size=n // 2)  # the near-linear region
        gq, uq = N.to_fixed(g, frac_gu), N.to_fixed(u, frac_gu)
        st = Stats()
        h = N.silu_mul(gq, uq, frac_gu, frac_h, tables, st)
        assert st == Stats() and h.dtype == np.int64
        gr, ur = N.from_fixed(gq, frac_gu), N.from_fixed(uq, frac_gu)
        ref = gr / (1.0 + np.exp(-gr)) * ur
        err = np.abs(N.from_fixed(h, frac_h) - ref)
        # budget: sigmoid <= 2.25 LSB of Q1.15 times |g u|, silu rounding 0.5 LSB times |u|,
        # final rounding 0.5 LSB of the output class
        bound = (
            2.5 * 2.0**-15 * np.abs(gr * ur) + 0.5 * 2.0**-frac_gu * np.abs(ur) + 0.5 * 2.0**-frac_h
        )
        assert np.all(err <= bound + 1e-12), (it, float(np.max(err / bound)))
        rel = float(np.max(err) / np.max(np.abs(ref)))
        worst_rel = max(worst_rel, rel)
        assert rel <= 2.0**-13
    assert worst_rel <= 2.0**-13  # measured: 0.10 * 2**-13


def test_silu_mul_saturated_regions_and_errors(tables: N.Tables) -> None:
    frac_gu, frac_h = 16, 14
    g = np.array(
        [16 << frac_gu, 20 << frac_gu, -(16 << frac_gu), -(20 << frac_gu), 0], dtype=np.int64
    )
    u = np.array([12345, -777, 999, 5, 4242], dtype=np.int64)
    h = N.silu_mul(g, u, frac_gu, frac_h, tables)
    # g >= 16: sigmoid is exactly 1.0 so silu = g and h = round_shift(g * u, 2*frac_gu - frac_h)
    sh = 2 * frac_gu - frac_h
    assert h[0] == round_half_up(Fraction(int(g[0]) * int(u[0]), 1 << sh))
    assert h[1] == round_half_up(Fraction(int(g[1]) * int(u[1]), 1 << sh))
    assert h[2] == 0 and h[3] == 0 and h[4] == 0
    # output saturation is counted
    st = Stats()
    big = np.array([20 << frac_gu, 20 << frac_gu], dtype=np.int64)
    h2 = N.silu_mul(big, np.array([20 << frac_gu, 1], dtype=np.int64), frac_gu, 24, tables, st)
    assert h2[0] == N.I32_MAX and st.sat == 1
    with pytest.raises(ValueError):
        N.silu_mul(g, u, 14, 29, tables)
    N.silu_mul(g, u, 14, 28, tables)


# =========================================================================== subc, argmax, weights


def test_subc_exact_and_saturating() -> None:
    x = np.array([N.I32_MAX, I32_MIN, 100, -100], dtype=np.int64)
    c = np.array([-1, 1, 30, -30], dtype=np.int64)
    st = Stats()
    y = N.subc(x, c, st)
    assert y.tolist() == [N.I32_MAX, I32_MIN, 70, -70]
    assert st.sat == 2
    assert N.subc(np.array([5, 6], dtype=np.int64), np.array([1, 1], dtype=np.int64)).tolist() == [
        4,
        5,
    ]


def test_argmax_strict_greater_lowest_index_on_ties() -> None:
    assert N.argmax(np.array([1, 5, 5, 2], dtype=np.int64)) == 1
    assert N.argmax(np.array([-3, -3, -3], dtype=np.int64)) == 0
    assert N.argmax(np.array([7], dtype=np.int64)) == 0
    assert N.argmax(np.array([-9, -1, -1, -20], dtype=np.int64)) == 1
    assert N.argmax(np.array([0, 1, 2, 3, 3], dtype=np.int64)) == 3
    rng = np.random.default_rng(81)
    v = rng.integers(-1000, 1000, size=5000, dtype=np.int64)
    v[[10, 4000]] = 5000
    assert N.argmax(v) == 10 and isinstance(N.argmax(v), int)


def test_quantize_rows_int8_matches_fraction_rounding() -> None:
    rng = np.random.default_rng(82)
    w = rng.standard_normal((40, 96)) * 10 ** rng.uniform(-4, 1, size=(40, 1))
    w[3] = 0.0
    w[5, 17] = -w[5].max() * 3  # a negative absmax
    w[7] = np.round(w[7] * 4) / 4  # many exact ties among the values
    q, scales = N.quantize_rows_int8(w)
    assert q.shape == w.shape and q.dtype == np.int64 and len(scales) == 40
    assert int(np.max(np.abs(q))) <= 127
    for n in range(40):
        row = w[n]
        a = float(np.max(np.abs(row)))
        if a == 0.0:
            assert scales[n] == SFLOAT_ZERO and not q[n].any()
            continue
        s = scales[n]
        exact_scale = Fraction(a) / 127
        assert abs(frac_value(s) - exact_scale) <= exact_scale * Fraction(1, 1 << 16)
        expect = [
            min(max(round_half_up(Fraction(v) / frac_value(s)), -127), 127) for v in row.tolist()
        ]
        assert q[n].tolist() == expect
        assert abs(int(q[n][int(np.argmax(np.abs(row)))])) == 127
    q0, s0 = N.quantize_rows_int8(np.zeros((2, 4)))
    assert not q0.any() and s0 == [SFLOAT_ZERO, SFLOAT_ZERO]


def test_to_fixed_from_fixed() -> None:
    x = np.array([-1.5, -0.5, -0.25, 0.25, 0.5, 1.5, 2.5, 1e12, -1e12])
    assert N.to_fixed(x, 0).tolist() == [-1, 0, 0, 0, 1, 2, 3, N.I32_MAX, I32_MIN]
    assert N.to_fixed(np.array([0.3]), 16)[0] == round_half_up(Fraction(0.3) * (1 << 16))
    assert N.to_fixed(np.array([1.0]), 16).dtype == np.int64
    v = np.array([-(1 << 31), -3, 0, 5, (1 << 31) - 1], dtype=np.int64)
    for frac in (0, 14, 16):
        assert N.to_fixed(N.from_fixed(v, frac), frac).tolist() == v.tolist()
        assert N.from_fixed(v, frac).tolist() == [float(i) * 2.0**-frac for i in v.tolist()]
