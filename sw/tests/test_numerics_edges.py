"""Edge cases of ``quettos.numerics`` around clamps, preconditions and long softmax rows."""

from __future__ import annotations

import math

import numpy as np
import pytest
from quettos import numerics as N
from quettos.numerics import SFloat, Stats


@pytest.fixture(scope="module")
def tables() -> N.Tables:
    return N.load_tables()


def test_embed_dequant_s1_window() -> None:
    row = np.arange(-127, 128, dtype=np.int64)
    sw = N.sfloat_from_float(0.0123)
    for s1 in (8, 16, 24):
        y = N.embed_dequant(row, sw, 16, s1=s1)
        exact = row * sw.value() * 2**16
        assert np.max(np.abs(y - exact)) <= 0.5 + 1e-9
    for s1 in (7, 25, 0, 40):
        with pytest.raises(ValueError):
            N.embed_dequant(row, sw, 16, s1=s1)


def test_quantize_gamma_rejects_values_above_int16() -> None:
    with pytest.raises(ValueError):
        N.quantize_gamma(np.array([32768.0]))
    q, e = N.quantize_gamma(np.array([32767.0, -1.0]))
    assert q.tolist() == [32767, -1] and e == 0


def test_eps_const_rounds_half_up() -> None:
    assert N.eps_const(2.0**-33, 1, 16) == 1  # exact 0.5 rounds up
    assert N.eps_const(1e-6, 896, 16) == 3848291
    assert N.eps_const(1e-5, 576, 14) == 1546188


def test_rmsnorm_negative_s1_is_clamped_and_counted(tables: N.Tables) -> None:
    gq, ge = N.quantize_gamma(np.ones(896))
    sqrt_d = N.sfloat_from_float(math.sqrt(896))
    x = np.zeros(896, dtype=np.int64)
    x[0] = 1
    st = Stats()
    y = N.rmsnorm(x, gq, ge, 5, sqrt_d, 16, tables, stats=st)  # eps_c far below 2**10
    assert y.shape == x.shape and st.err_shift == 896
    st2 = Stats()
    N.rmsnorm(x, gq, ge, 1 << 10, sqrt_d, 16, tables, stats=st2)
    assert st2.err_shift == 0


def test_choose_s1_uses_the_largest_exponents() -> None:
    # Largest reachable exponents (-20, -20) at FRAC 16: the precision bound gives s1 <= 8,
    # and a 30-bit accumulator allows it (40-bit bound s1 >= 6).
    s1 = N.choose_s1(30, 16, -20, -20)
    assert s1 == 8 and -(16 + s1) - (-20) - (-20) >= 16
    # A 36-bit accumulator needs s1 >= 12, so the 40-bit bound wins over precision.
    assert N.choose_s1(36, 16, -20, -20) == 12
    # Infeasible precision bound: the default 16 stands.
    assert N.choose_s1(40, 16, -5, -5) == 16


def test_softmax_clip_is_counted_not_saturation(tables: N.Tables) -> None:
    # Single-token row: p = 1.0 with a full-scale V mantissa gives w = 32768 -> clipped.
    st = Stats()
    w, sreg = N.softmax(
        np.array([100 << 16], dtype=np.int64), 1, 16, [SFloat(65535, -20)], tables, stats=st
    )
    assert w[0] == 32767 and st.sat == 0 and st.clip == 1
    # Other tokens 100 log2 units below round to zero weight: p = 1.0 again, still a clip.
    scores = np.array([100 << 16, 0, 0], dtype=np.int64)
    vs = [SFloat(65535, -20), SFloat(40000, -20), SFloat(40000, -20)]
    st = Stats()
    w, sreg = N.softmax(scores, 3, 16, vs, tables, stats=st)
    assert w[0] == 32767 and st.sat == 0 and st.clip == 1
    # A token 10 log2 units below keeps a visible weight, so p < 1.0 and nothing clips.
    scores = np.array([100 << 16, 90 << 16], dtype=np.int64)
    st = Stats()
    w, sreg = N.softmax(scores, 2, 16, vs[:2], tables, stats=st)
    assert w[0] < 32767 and st.clip == 0


@pytest.mark.parametrize("sv_m", [50000, 32768, 65535])
@pytest.mark.parametrize("length", [2048, 8192])
def test_softmax_long_rows_with_dominant_token_are_unbiased(
    length: int, sv_m: int, tables: N.Tables
) -> None:
    """A sink token far above a Gaussian tail: the rounded exp keeps the normalizer honest."""
    rng = np.random.default_rng(length)
    frac_s = 16
    tail = rng.normal(0.0, 1.0, length - 1)
    sink = float(tail.max()) + 12.3  # log2 units above the tail
    real = np.concatenate([[sink], tail])
    scores = N.to_fixed(real, frac_s)
    vs = [SFloat(sv_m, -22)] * length
    w, sreg = N.softmax(scores, length, frac_s, vs, tables)
    p_ref = 2.0 ** (real - real.max())
    p_ref /= p_ref.sum()
    got = w * sreg.value() / vs[0].value()
    # the sink probability itself is accurate to a few units of 2**-15 ...
    err_units = np.abs(got - p_ref) * 2**15
    assert err_units[0] <= 80.0, err_units[0]
    # ... while 16-bit weights round every token by up to 2**e_max, so the row's total
    # mass is off by at most length * 2**e_max / Sv_max <= length * 2**-15 in either direction
    lost = 1.0 - float(np.sum(got))
    bound = length * (sreg.value() / 2.0) / vs[0].value()
    assert bound <= length * 2.0**-15 + 1e-12
    assert abs(lost) <= bound + 2.0**-12, (lost, bound)


def test_softmax_weight_shift_saturates_for_tiny_v_scales(tables: N.Tables) -> None:
    scores = np.array([1000 << 16, 999 << 16], dtype=np.int64)
    for e_small in (-49, -50, -51, -70):
        vs = [SFloat(50000, -10), SFloat(50000, e_small)]
        w, sreg = N.softmax(scores, 2, 16, vs, tables)
        assert w[1] == 0 and w[0] > 0 and sreg == SFloat(1 << 15, 1 - 10 - 15)
