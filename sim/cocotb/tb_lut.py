"""cocotb tests of qcore_lut_rom and qcore_lut_interp against sw/quettos/numerics.py.

The ROM bench reads every entry of the table its build was elaborated with,
through each port on its own and through both ports in the same cycle, and
compares ``{v, dv}`` with the ``numerics.Lut`` the cocotb adapter loads; the
table is named by ``QC_LUT_TABLE``.  The interpolator bench drives the whole
input domain of all four tables -- every entry of every table crossed with
every one of the 256 fractions, which is every ``(index, frac8)`` pair the
vector unit can present -- and requires ``y`` to equal ``Lut.interp`` on each
one, then measures the interpolation error of the results against the real
functions.  Rounding boundaries, the ends of every table, the rsqrt segment
boundary and the extremes of the 25-bit product are driven directly.
"""

from __future__ import annotations

import os

import cocotb
import numpy as np
import qc_numerics as qn
import qc_stream as qs
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge

TABLE = os.environ.get("QC_LUT_TABLE", "exp2")
FRACS = 256
Q15_ONE = 1 << 15


# Entries whose fraction sweep is driven on its own, before the full domain:
# both ends of the table, the rsqrt segment boundary and a spread between.
def landmark_indices(entries: int) -> list[int]:
    marks = [0, 1, 2, entries // 4, entries // 2 - 1, entries // 2, entries // 2 + 1]
    marks += [255, 256, 257] if entries > 256 else []
    marks += [entries - 3, entries - 2, entries - 1]
    return sorted({i for i in marks if 0 <= i < entries})


def lut_of(name: str):
    """The checked-in (v, dv) table, through the cocotb numerics adapter."""
    return getattr(qn.tables(), name)


def expected(lut, idx: int, frac8: int) -> int:
    """``numerics.Lut.interp`` on one (entry, fraction) pair, as a Python int."""
    return int(lut.interp(idx, frac8))


async def start(dut, rst=None) -> None:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await FallingEdge(dut.clk)
    if rst is not None:
        await qs.reset(dut, rst)


# --------------------------------------------------------------------------- qcore_lut_rom


def rom_entries(dut) -> int:
    return 1 << len(dut.idx_a)


async def rom_idle(dut) -> None:
    dut.en_a.value = 0
    dut.en_b.value = 0
    dut.idx_a.value = 0
    dut.idx_b.value = 0
    await FallingEdge(dut.clk)


@cocotb.test()
async def test_rom_every_entry_port_a(dut):
    """Every entry of the table on port A, one per cycle, against numerics.Lut."""
    await start(dut)
    await rom_idle(dut)
    lut = lut_of(TABLE)
    n = rom_entries(dut)
    assert n == lut.v.shape[0], f"{TABLE}: built with {n} entries, table has {lut.v.shape[0]}"
    for i in range(n + 1):
        if i >= 1:
            got_v, got_dv = qs.value(dut.v_a), qs.signed_value(dut.dv_a, 16)
            assert got_v == int(lut.v[i - 1]), f"{TABLE}[{i - 1}] v {got_v} != {lut.v[i - 1]}"
            assert got_dv == int(lut.dv[i - 1]), f"{TABLE}[{i - 1}] dv {got_dv} != {lut.dv[i - 1]}"
        dut.en_a.value = 1 if i < n else 0
        if i < n:
            dut.idx_a.value = i
        await FallingEdge(dut.clk)
    dut.en_a.value = 0


@cocotb.test()
async def test_rom_every_entry_port_b(dut):
    """The same sweep on port B, descending, so the two ports are read independently."""
    await start(dut)
    await rom_idle(dut)
    lut = lut_of(TABLE)
    n = rom_entries(dut)
    order = list(range(n - 1, -1, -1))
    for i in range(n + 1):
        if i >= 1:
            j = order[i - 1]
            got_v, got_dv = qs.value(dut.v_b), qs.signed_value(dut.dv_b, 16)
            assert got_v == int(lut.v[j]), f"{TABLE}[{j}] v {got_v} != {lut.v[j]}"
            assert got_dv == int(lut.dv[j]), f"{TABLE}[{j}] dv {got_dv} != {lut.dv[j]}"
        dut.en_b.value = 1 if i < n else 0
        if i < n:
            dut.idx_b.value = order[i]
        await FallingEdge(dut.clk)
    dut.en_b.value = 0


@cocotb.test()
async def test_rom_both_ports_same_cycle(dut):
    """Two lanes read two different entries in one cycle; each port returns its own."""
    await start(dut)
    await rom_idle(dut)
    lut = lut_of(TABLE)
    n = rom_entries(dut)
    pairs = [(i, (n - 1 - i)) for i in range(n)]
    pairs += [(i, i) for i in (0, n // 2, n - 1)]  # both lanes on one entry
    for k in range(len(pairs) + 1):
        if k >= 1:
            a, b = pairs[k - 1]
            assert qs.value(dut.v_a) == int(lut.v[a]) and qs.value(dut.v_b) == int(lut.v[b])
            assert qs.signed_value(dut.dv_a, 16) == int(lut.dv[a])
            assert qs.signed_value(dut.dv_b, 16) == int(lut.dv[b])
        if k < len(pairs):
            dut.en_a.value = 1
            dut.en_b.value = 1
            dut.idx_a.value = pairs[k][0]
            dut.idx_b.value = pairs[k][1]
        else:
            dut.en_a.value = 0
            dut.en_b.value = 0
        await FallingEdge(dut.clk)


@cocotb.test()
async def test_rom_latency_and_hold(dut):
    """A read lands one cycle after the enabled cycle and holds while the enable is low."""
    await start(dut)
    await rom_idle(dut)
    lut = lut_of(TABLE)
    n = rom_entries(dut)
    for en, idx, v, dv in (
        (dut.en_a, dut.idx_a, dut.v_a, dut.dv_a),
        (dut.en_b, dut.idx_b, dut.v_b, dut.dv_b),
    ):
        first, second = 0, n - 1
        en.value = 1
        idx.value = first
        await FallingEdge(dut.clk)
        assert qs.value(v) == int(lut.v[first]) and qs.signed_value(dv, 16) == int(lut.dv[first])
        en.value = 0
        idx.value = second  # the address moves without an enable: the output must hold
        for _ in range(4):
            await FallingEdge(dut.clk)
            assert qs.value(v) == int(lut.v[first])
            assert qs.signed_value(dv, 16) == int(lut.dv[first])
        en.value = 1
        await FallingEdge(dut.clk)
        en.value = 0
        assert qs.value(v) == int(lut.v[second])
        assert qs.signed_value(dv, 16) == int(lut.dv[second])


@cocotb.test()
async def test_rom_ends_and_segment_boundary(dut):
    """The first and last entries, and for a 512-entry table the two sides of index 256."""
    await start(dut)
    await rom_idle(dut)
    lut = lut_of(TABLE)
    n = rom_entries(dut)
    marks = landmark_indices(n)
    for k in range(len(marks) + 1):
        if k >= 1:
            i = marks[k - 1]
            assert qs.value(dut.v_a) == int(lut.v[i]), f"{TABLE}[{i}] v"
            assert qs.signed_value(dut.dv_a, 16) == int(lut.dv[i]), f"{TABLE}[{i}] dv"
        if k < len(marks):
            dut.en_a.value = 1
            dut.idx_a.value = marks[k]
        else:
            dut.en_a.value = 0
        await FallingEdge(dut.clk)
    if n == 512:
        # rsqrt: entry 255 is the last of segment 0 and entry 256 the first of
        # segment 1, and dv[255] carries segment 0 into it.
        assert int(lut.v[255]) + int(lut.dv[255]) == int(lut.v[256])


# ----------------------------------------------------------------------- qcore_lut_interp


async def interp_drive(dut, vectors) -> list[int]:
    """Drive (v, dv, frac8) triples one per cycle; return y for each, in order.

    A triple presented in one cycle leaves on ``y`` in the next, so the result
    read at the top of iteration ``k`` belongs to vector ``k - 1``.
    """
    out: list[int] = []
    n = len(vectors)
    y, ov = dut.y, dut.out_valid
    dut.in_valid.value = 1
    for k in range(n + 1):
        if k >= 1:
            assert int(ov.value) == 1, f"out_valid low one cycle after vector {k - 1}"
            out.append(qs.value(y))
        if k < n:
            v, dv, frac8 = vectors[k]
            dut.v.value = v
            dut.dv.value = dv & 0xFFFF
            dut.frac8.value = frac8
        else:
            dut.in_valid.value = 0
        await FallingEdge(dut.clk)
    dut.in_valid.value = 0
    return out


@cocotb.test()
async def test_interp_valid_and_hold(dut):
    """out_valid follows in_valid by one cycle, reset clears it, and y holds between requests."""
    await start(dut, dut.rst)
    dut.in_valid.value = 0
    dut.v.value = 0
    dut.dv.value = 0
    dut.frac8.value = 0
    await FallingEdge(dut.clk)
    assert int(dut.out_valid.value) == 0
    dut.v.value = 1000
    dut.dv.value = 512
    dut.frac8.value = 128
    dut.in_valid.value = 1
    await FallingEdge(dut.clk)
    dut.in_valid.value = 0
    assert int(dut.out_valid.value) == 1, "out_valid one cycle after the request"
    y = qs.value(dut.y)
    assert y == 1000 + ((512 * 128 + 128) >> 8)
    for _ in range(4):
        await FallingEdge(dut.clk)
        assert int(dut.out_valid.value) == 0, "out_valid is one cycle wide"
        assert qs.value(dut.y) == y, "the result holds while no request is presented"
    dut.v.value = 0
    dut.dv.value = 0
    dut.frac8.value = 0
    dut.in_valid.value = 1
    await FallingEdge(dut.clk)
    assert int(dut.out_valid.value) == 1
    await qs.reset(dut, dut.rst)
    dut.in_valid.value = 0
    assert int(dut.out_valid.value) == 0, "reset clears out_valid"


@cocotb.test()
async def test_interp_rounding_boundaries(dut):
    """The +128 rounds half toward +inf and the shift is arithmetic, on both signs."""
    await start(dut, dut.rst)
    cases: list[tuple[int, int, int]] = []
    for dv in (1, -1, 2, -2, 3, -3, 255, -255, 256, -256, 257, -257):
        for frac8 in (0, 1, 42, 85, 127, 128, 129, 170, 254, 255):
            prod = dv * frac8
            # Place v so the result stays inside the [0, 65535] the tables keep.
            v = 32768
            cases.append((v, dv, frac8))
            assert 0 <= v + ((prod + 128) >> 8) <= 0xFFFF
    got = await interp_drive(dut, cases)
    for (v, dv, frac8), y in zip(cases, got, strict=True):
        want = v + ((dv * frac8 + 128) >> 8)
        assert y == want, f"v={v} dv={dv} frac8={frac8}: {y} != {want}"
    # A truncating shift would return the same value for these two.
    i_lo = cases.index((32768, -1, 128))
    i_hi = cases.index((32768, -1, 129))
    assert got[i_lo] == 32768 and got[i_hi] == 32767


@cocotb.test()
async def test_interp_product_extremes(dut):
    """The corners of the 25-bit product and the 16-bit inputs, results kept in range."""
    await start(dut, dut.rst)
    vs = (0, 1, 128, 32767, 32768, 40000, 65534, 65535)
    dvs = (0, 1, -1, 127, -128, 32767, -32768, 16384, -16384)
    fracs = (0, 1, 127, 128, 129, 254, 255)
    cases = []
    for v in vs:
        for dv in dvs:
            for frac8 in fracs:
                if 0 <= v + ((dv * frac8 + 128) >> 8) <= 0xFFFF:
                    cases.append((v, dv, frac8))
    assert len(cases) > 300
    got = await interp_drive(dut, cases)
    for (v, dv, frac8), y in zip(cases, got, strict=True):
        want = v + ((dv * frac8 + 128) >> 8)
        assert y == want, f"v={v} dv={dv} frac8={frac8}: {y} != {want}"


@cocotb.test()
async def test_interp_landmark_fraction_sweeps(dut):
    """Every fraction, at both ends of every table and at the rsqrt segment boundary."""
    await start(dut, dut.rst)
    for name in ("exp2", "sigmoid", "rsqrt", "recip"):
        lut = lut_of(name)
        n = int(lut.v.shape[0])
        cases = [
            (int(lut.v[i]), int(lut.dv[i]), f) for i in landmark_indices(n) for f in range(FRACS)
        ]
        got = await interp_drive(dut, cases)
        for (v, dv, frac8), y in zip(cases, got, strict=True):
            want = v + ((dv * frac8 + 128) >> 8)
            assert y == want, f"{name} v={v} dv={dv} frac8={frac8}: {y} != {want}"


@cocotb.test()
async def test_interp_every_entry_every_fraction(dut):
    """Every (entry, fraction) of all four tables -- the whole input domain -- bit for bit,
    and the interpolation error of the results against the real functions."""
    await start(dut, dut.rst)
    for name in ("exp2", "sigmoid", "rsqrt", "recip"):
        lut = lut_of(name)
        n = int(lut.v.shape[0])
        idx = np.repeat(np.arange(n, dtype=np.int64), FRACS)
        frac = np.tile(np.arange(FRACS, dtype=np.int64), n)
        want = np.asarray(lut.interp(idx, frac), dtype=np.int64)
        vectors = [
            (int(lut.v[i]), int(lut.dv[i]), int(f))
            for i, f in zip(idx.tolist(), frac.tolist(), strict=True)
        ]
        got = np.asarray(await interp_drive(dut, vectors), dtype=np.int64)
        bad = np.flatnonzero(got != want)
        if bad.size:
            k = int(bad[0])
            raise AssertionError(
                f"{name}: {bad.size} of {got.size} wrong; entry {idx[k]} frac8 {frac[k]}: "
                f"{got[k]} != {want[k]}"
            )
        err = np.abs(got.astype(np.float64) - exact_curve(name, idx, frac))
        worst = float(err.max())
        parts = ""
        if name == "rsqrt":  # the two segments are reported separately in docs/NUMERICS.md
            seg = idx >= 256
            parts = (
                f" (segment 0 {float(err[~seg].max()):.3f}, segment 1 {float(err[seg].max()):.3f})"
            )
        cocotb.log.info(
            f"{name}: {got.size} points, worst interpolation error {worst:.3f} LSB{parts}"
        )
        assert worst <= 2.0, f"{name}: interpolation error {worst:.3f} LSB exceeds the 2 LSB bound"
        assert np.all(got >= 0) and np.all(got <= 0xFFFF)


def exact_curve(name: str, idx: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """``f(x) * 2**15`` in float64 at the input each (entry, fraction) pair stands for.

    The pairs are exactly the index and fraction splits of ``numerics``: an exp2
    ``f16``, a sigmoid ``|x|`` at 13 fraction bits, an rsqrt ``m_q16`` in either
    segment and a normalized ``a_hi`` for the reciprocal.
    """
    i = idx.astype(np.float64)
    f = frac.astype(np.float64)
    if name == "exp2":
        return np.exp2((i * 256.0 + f) / 65536.0) * Q15_ONE
    if name == "sigmoid":
        x = (i * 256.0 + f) / float(1 << 13)
        return Q15_ONE / (1.0 + np.exp(-x))
    if name == "rsqrt":
        seg1 = idx >= 256
        m = np.where(seg1, 131072.0 + (i - 256.0) * 512.0 + f * 2.0, 65536.0 + i * 256.0 + f)
        return Q15_ONE / np.sqrt(m / 65536.0)
    if name == "recip":
        a_hi = 32768.0 + i * 128.0 + f / 2.0
        return Q15_ONE / (a_hi / Q15_ONE)
    raise ValueError(name)
