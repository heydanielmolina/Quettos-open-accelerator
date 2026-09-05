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


# --------------------------------------------------------------------------- vectorized mirrors


def _rand_scales(rng: np.random.Generator, n: int, e_lo: int, e_hi: int, p_zero: float):
    m = rng.integers(1 << 15, 1 << 16, n).astype(np.int64)
    e = rng.integers(e_lo, e_hi + 1, n).astype(np.int64)
    zero = rng.random(n) < p_zero
    m[zero] = 0
    e[zero] = 0
    return m, e


def test_bitlen_array_matches_bitlen() -> None:
    vals = [0, 1, 2, 3, 4, 7, 8, (1 << 23), (1 << 23) + 1, (1 << 36) - 1, 1 << 36, (1 << 53) - 1]
    got = N.bitlen_array(np.array(vals, dtype=np.int64))
    assert got.tolist() == [N.bitlen(v) for v in vals]
    with pytest.raises(ValueError):
        N.bitlen_array(np.array([1 << 53], dtype=np.int64))
    with pytest.raises(ValueError):
        N.bitlen_array(np.array([-1], dtype=np.int64))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_requant_rows_is_the_scalar_requant_per_element(seed: int) -> None:
    """Every element and every counter equals the scalar ``requant`` with the (row, column) scales.

    The block mixes zero scales, unwritten (``valid == False``) elements, shifts
    that leave ``[0, 63]`` on both sides, sat40 / sat32 events, biases and the
    accumulate operand.
    """
    rng = np.random.default_rng(seed)
    t_rows, n_cols = 7, 23
    # accumulators up to 39 bits, some tiny, some huge
    mag = rng.integers(0, 40, (t_rows, n_cols))
    acc = (rng.integers(-(1 << 20), 1 << 20, (t_rows, n_cols)).astype(np.int64) << 19) >> (39 - mag)
    acc[0, 0] = (1 << 39) - 1
    acc[1, 1] = -(1 << 39)
    sw_m, sw_e = _rand_scales(rng, n_cols, -40, -5, 0.15)
    sx_m, sx_e = _rand_scales(rng, t_rows, -40, 10, 0.15)
    bias = rng.integers(-(1 << 30), 1 << 30, n_cols).astype(np.int64)
    bias[::3] = 0
    old = rng.integers(-(1 << 31), 1 << 31, (t_rows, n_cols)).astype(np.int64)
    valid = rng.random((t_rows, n_cols)) < 0.8
    for s1, sbias in ((16, -32), (10, -26), (0, -20), (13, -60), (16, 5)):
        for use_bias, use_old, use_valid in (
            (True, True, True),
            (False, False, False),
            (True, False, True),
        ):
            st_rows = Stats()
            got = N.requant_rows(
                acc,
                sw_m,
                sw_e,
                sx_m,
                sx_e,
                s1,
                sbias,
                bias_q=bias if use_bias else None,
                old=old if use_old else None,
                valid=valid if use_valid else None,
                stats=st_rows,
            )
            st_ref = Stats()
            want = np.zeros_like(acc)
            for t in range(t_rows):
                sx = SFloat(int(sx_m[t]), int(sx_e[t]))
                for n in range(n_cols):
                    sw = SFloat(int(sw_m[n]), int(sw_e[n]))
                    if use_valid and not valid[t, n]:
                        sw = N.SFLOAT_ZERO
                    want[t, n] = N.requant(
                        int(acc[t, n]),
                        sw,
                        sx,
                        s1,
                        sbias,
                        bias_q=int(bias[n]) if use_bias else 0,
                        old=int(old[t, n]) if use_old else None,
                        stats=st_ref,
                    )
            assert np.array_equal(got, want), (s1, sbias, use_bias, use_old, use_valid)
            assert st_rows == st_ref, (s1, sbias, st_rows, st_ref)
            assert got.dtype == np.int64


def test_requant_rows_exercises_every_counter() -> None:
    rng = np.random.default_rng(99)
    acc = rng.integers(-(1 << 38), 1 << 38, (4, 6)).astype(np.int64)
    sw_m, sw_e = _rand_scales(rng, 6, -30, -10, 0.0)
    sx_m, sx_e = _rand_scales(rng, 4, -30, -10, 0.0)
    st = Stats()
    N.requant_rows(acc, sw_m, sw_e, sx_m, sx_e, 0, -16, stats=st)  # s1 = 0 forces sat40
    assert st.sat > 0
    st = Stats()
    N.requant_rows(acc, sw_m, sw_e, sx_m, sx_e, 16, 100, stats=st)  # shifts above 63
    assert st.err_shift == acc.size
    st = Stats()
    N.requant_rows(acc, sw_m, sw_e, sx_m, sx_e, 16, -200, stats=st)  # shifts below 0
    assert st.err_shift == acc.size
    with pytest.raises(ValueError):
        N.requant_rows(acc[0], sw_m, sw_e, sx_m[:1], sx_e[:1], 16, -32)


@pytest.mark.parametrize("frac_s", [16, 18])
@pytest.mark.parametrize("seed", [11, 12, 13])
def test_softmax_rows_is_the_scalar_softmax_per_row(
    frac_s: int, seed: int, tables: N.Tables
) -> None:
    """Each row equals ``softmax(scores[t], lengths[t], ...)`` including zero V scales and clips."""
    rng = np.random.default_rng(seed)
    t_rows, width = 9, 37
    real = rng.normal(0.0, 6.0, (t_rows, width))
    real[2, :] = 0.0  # all-equal row
    real[3, 0] += 60.0  # one dominant token far above the rest (weights round to zero)
    real[4, :] = 0.0
    real[4, 5] = 100.0  # p = 1.0 exactly; with a full-scale V mantissa the weight clips at 32767
    scores = N.to_fixed(real, frac_s)
    scores[5, 10:] = rng.integers(-(1 << 31), 1 << 31, width - 10)  # garbage beyond the length
    lengths = rng.integers(1, width + 1, t_rows)
    lengths[2] = width
    lengths[3] = width
    lengths[4] = 6
    lengths[5] = 10
    lengths[6] = 1
    sv_m, sv_e = _rand_scales(rng, width, -30, -18, 0.2)
    sv_m[5], sv_e[5] = 65535, -10  # the row's largest V exponent
    st_rows = Stats()
    w, sreg_m, sreg_e = N.softmax_rows(scores, lengths, frac_s, sv_m, sv_e, tables, stats=st_rows)
    st_ref = Stats()
    v_scales = [SFloat(int(m), int(e)) for m, e in zip(sv_m, sv_e, strict=True)]
    for t in range(t_rows):
        want, sreg = N.softmax(scores[t], int(lengths[t]), frac_s, v_scales, tables, stats=st_ref)
        assert np.array_equal(w[t], want), t
        assert (int(sreg_m[t]), int(sreg_e[t])) == (sreg.m, sreg.e), t
    assert st_rows == st_ref
    assert st_ref.clip >= 1
    assert w.shape == scores.shape and w.dtype == np.int64


def test_softmax_rows_all_zero_scales_row_and_length_checks(tables: N.Tables) -> None:
    scores = np.zeros((2, 4), dtype=np.int64)
    sv_m = np.array([0, 0, 50000, 50000], dtype=np.int64)
    sv_e = np.array([0, 0, -20, -20], dtype=np.int64)
    w, m, e = N.softmax_rows(scores, np.array([2, 4]), 16, sv_m, sv_e, tables)
    assert np.all(w[0] == 0) and (int(m[0]), int(e[0])) == (0, 0)  # row 0 sees only zero scales
    assert np.any(w[1] > 0) and int(m[1]) == 1 << 15
    for bad in (np.array([0, 4]), np.array([5, 1]), np.array([1])):
        with pytest.raises(ValueError):
            N.softmax_rows(scores, bad, 16, sv_m, sv_e, tables)
