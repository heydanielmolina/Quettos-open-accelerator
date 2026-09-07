"""cocotb tests of qcore_vpu_lane: every op bit-exact against sw/quettos/numerics.py.

Millions of random elements across the six ops and the operand extremes, plus
directed cases at the boundaries the numerics document names: round half toward
+inf at every shift from zero to the maximum on both the 64-bit and the 49-bit
product, saturation at both signs and the largest non-saturating results, the
zero operands, the unsigned coefficient of L_MUL16 against the signed
coefficient of the rotation ops, the raw product on p64, and the two-cycle
latency with gaps, back-to-back elements and reset.
"""

from __future__ import annotations

import os

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge
from quettos import numerics

L_MUL32, L_MUL16, L_ROPE_A, L_ROPE_B, L_SUB, L_PASS = range(6)
OPS = (L_MUL32, L_MUL16, L_ROPE_A, L_ROPE_B, L_SUB, L_PASS)
OP_NAME = {
    L_MUL32: "L_MUL32",
    L_MUL16: "L_MUL16",
    L_ROPE_A: "L_ROPE_A",
    L_ROPE_B: "L_ROPE_B",
    L_SUB: "L_SUB",
    L_PASS: "L_PASS",
}
MUL_OPS = (L_MUL16, L_ROPE_A, L_ROPE_B)

M16 = (1 << 16) - 1
M32 = (1 << 32) - 1
I32_MAX = (1 << 31) - 1
I32_MIN = -(1 << 31)

# Elements driven by the two randomised sweeps; QCORE_VPU_LANE_VECTORS scales both.
SWEEP = int(os.environ.get("QCORE_VPU_LANE_VECTORS", "1500000"))


# --------------------------------------------------------------------------- reference


def wrap32(x: np.ndarray) -> np.ndarray:
    """Signed int32 interpretation of the low 32 bits of every element."""
    return ((x + (1 << 31)) & M32) - (1 << 31)


def signed16(c: np.ndarray) -> np.ndarray:
    """The int16 a rotation coefficient carries."""
    return ((c + (1 << 15)) & M16) - (1 << 15)


def round_shift_arr(x: np.ndarray, sh: np.ndarray) -> np.ndarray:
    """``numerics.round_shift`` per element, grouped by shift.

    ``numerics.round_shift`` adds ``2**(s-1)`` before the shift, so a group whose
    magnitudes would carry that sum past int64 is evaluated on Python ints, where
    the same function is exact.
    """
    out = np.zeros(x.shape, dtype=np.int64)
    for s in np.unique(sh):
        m = sh == s
        xs = x[m]
        s = int(s)
        headroom = (1 << 62) if s == 0 else (1 << 63) - (1 << (s - 1))
        if xs.size and (int(xs.max()) >= headroom or int(xs.min()) <= -headroom):
            out[m] = np.array([numerics.round_shift(int(v), s) for v in xs], dtype=np.int64)
        else:
            out[m] = numerics.round_shift(xs, s)
    return out


def lane_ref(
    op: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, c2: np.ndarray, sh: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y, sat, p64) of ``numerics`` for a batch of lane requests.

    ``a`` and ``b`` are signed int32, ``c`` and ``c2`` the raw 16-bit coefficient
    fields: L_MUL16 reads them as u16 and the rotation ops as int16.
    """
    cs, c2s = signed16(c), signed16(c2)
    p_ab = a * b
    raw = np.zeros(a.shape, dtype=np.int64)
    for code, prod in (
        (L_MUL32, p_ab),
        (L_MUL16, a * c),
        (L_ROPE_A, a * cs - b * c2s),
        (L_ROPE_B, b * cs + a * c2s),
    ):
        m = op == code
        if np.any(m):
            raw[m] = round_shift_arr(prod[m], sh[m])
    m = op == L_SUB
    raw[m] = a[m] - b[m]
    m = op >= L_PASS
    raw[m] = a[m]
    y = numerics.sat(raw, 32)
    return y, (y != raw).astype(np.int64), p_ab


# --------------------------------------------------------------------------- driver


async def setup(dut) -> None:
    """Start the clock, hold reset, leave every input at zero."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    for sig in (dut.in_valid, dut.op, dut.a, dut.b, dut.c, dut.c2, dut.sh):
        sig.value = 0
    for _ in range(3):
        await FallingEdge(dut.clk)
    dut.rst.value = 0
    await FallingEdge(dut.clk)


async def run_batch(dut, op, a, b, c, c2, sh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drive one element per cycle back to back; returns the (y, sat, p64) of each.

    ``y`` of element ``i`` is registered two clock edges after the edge that
    sampled it, so the falling edge closing loop step ``i + 1`` carries it.
    """
    n = len(op)
    op_l = np.asarray(op, dtype=np.int64).tolist()
    a_l = (np.asarray(a, dtype=np.int64) & M32).tolist()
    b_l = (np.asarray(b, dtype=np.int64) & M32).tolist()
    c_l = np.asarray(c, dtype=np.int64).tolist()
    c2_l = np.asarray(c2, dtype=np.int64).tolist()
    sh_l = np.asarray(sh, dtype=np.int64).tolist()
    s_v, s_op, s_a, s_b, s_c, s_c2, s_sh = (
        dut.in_valid,
        dut.op,
        dut.a,
        dut.b,
        dut.c,
        dut.c2,
        dut.sh,
    )
    o_v, o_y, o_p, o_s = dut.out_valid, dut.y, dut.p64, dut.sat
    ys: list[int] = []
    ps: list[int] = []
    ss: list[int] = []
    for i in range(n + 1):
        if i < n:
            s_v.value = 1
            s_op.value = op_l[i]
            s_a.value = a_l[i]
            s_b.value = b_l[i]
            s_c.value = c_l[i]
            s_c2.value = c2_l[i]
            s_sh.value = sh_l[i]
        else:
            s_v.value = 0
        await FallingEdge(dut.clk)
        if i:
            assert int(o_v.value) == 1, f"out_valid low at element {i - 1}"
            ys.append(o_y.value.to_unsigned())
            ps.append(o_p.value.to_unsigned())
            ss.append(int(o_s.value))
    s_v.value = 0
    await FallingEdge(dut.clk)
    assert int(o_v.value) == 0, "out_valid stayed high after the last element"
    return (
        wrap32(np.array(ys, dtype=np.int64)),
        np.array(ss, dtype=np.int64),
        np.array(ps, dtype=object),
    )


def to_signed64(p: np.ndarray) -> np.ndarray:
    """Signed value of the 64-bit p64 readback."""
    return np.array([int(v) - (1 << 64) if int(v) >> 63 else int(v) for v in p], dtype=np.int64)


async def check(dut, op, a, b, c, c2, sh, label: str) -> int:
    """Run a batch and compare every output against numerics; returns the element count."""
    op = np.asarray(op, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    c2 = np.asarray(c2, dtype=np.int64)
    sh = np.asarray(sh, dtype=np.int64)
    y_g, s_g, p_g = await run_batch(dut, op, a, b, c, c2, sh)
    y_r, s_r, p_r = lane_ref(op, a, b, c, c2, sh)
    p_g = to_signed64(p_g)
    for got, want, name in ((y_g, y_r, "y"), (s_g, s_r, "sat"), (p_g, p_r, "p64")):
        bad = np.flatnonzero(got != want)
        if bad.size:
            i = int(bad[0])
            raise AssertionError(
                f"{label}: {name} differs on {bad.size} of {len(op)} elements; first at {i}: "
                f"op={OP_NAME.get(int(op[i]), int(op[i]))} a={int(a[i])} b={int(b[i])} "
                f"c={int(c[i])} c2={int(c2[i])} sh={int(sh[i])} got={int(got[i])} "
                f"want={int(want[i])}"
            )
    return len(op)


# --------------------------------------------------------------------------- stimulus


SPECIAL32 = np.array(
    [
        0,
        1,
        -1,
        2,
        -2,
        3,
        -3,
        7,
        -7,
        I32_MAX,
        I32_MIN,
        I32_MAX - 1,
        I32_MIN + 1,
        1 << 30,
        -(1 << 30),
        (1 << 30) - 1,
        -((1 << 30) - 1),
        1 << 16,
        -(1 << 16),
        (1 << 16) - 1,
        -((1 << 16) - 1),
        1 << 15,
        -(1 << 15),
        (1 << 15) - 1,
        1 << 23,
        1 << 24,
        (1 << 24) - 1,
        0x5555_5555,
        -0x5555_5555,
        0x2AAA_AAAA,
        -0x2AAA_AAAB,
    ],
    dtype=np.int64,
)
SPECIAL16 = np.array(
    [0, 1, 2, 3, 0x3FFF, 0x4000, 0x4001, 0x7FFF, 0x8000, 0x8001, 0xC000, 0xFFFE, 0xFFFF],
    dtype=np.int64,
)
SPECIAL_SH = np.array([0, 1, 2, 13, 14, 15, 16, 17, 23, 24, 31, 32, 46, 47, 48, 62, 63])


def gen32(rng: np.random.Generator, n: int) -> np.ndarray:
    """int32 operands: uniform draws, the extremes and odd multiples of powers of two."""
    uni = wrap32(rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.int64))
    spec = SPECIAL32[rng.integers(0, SPECIAL32.size, n)]
    odd = rng.integers(0, 1 << 20, n, dtype=np.int64) * 2 + 1
    sign = 1 - 2 * rng.integers(0, 2, n, dtype=np.int64)
    pow2 = wrap32(sign * np.left_shift(odd, rng.integers(0, 32, n, dtype=np.int64)))
    near = wrap32(SPECIAL32[rng.integers(0, SPECIAL32.size, n)] + rng.integers(-4, 5, n))
    kind = rng.integers(0, 4, n)
    return np.select([kind == 0, kind == 1, kind == 2], [uni, spec, pow2], default=near).astype(
        np.int64
    )


def gen16(rng: np.random.Generator, n: int) -> np.ndarray:
    """Raw 16-bit coefficients: uniform draws, the boundary encodings and powers of two."""
    uni = rng.integers(0, 1 << 16, n, dtype=np.int64)
    spec = SPECIAL16[rng.integers(0, SPECIAL16.size, n)]
    pw = np.left_shift(np.int64(1), rng.integers(0, 16, n, dtype=np.int64))
    kind = rng.integers(0, 3, n)
    return np.select([kind == 0, kind == 1], [uni, spec], default=pw).astype(np.int64)


def gen_sh(rng: np.random.Generator, n: int) -> np.ndarray:
    """Shift amounts: uniform over [0, 63] half the time, the interesting ones otherwise."""
    uni = rng.integers(0, 64, n, dtype=np.int64)
    spec = SPECIAL_SH[rng.integers(0, SPECIAL_SH.size, n)]
    return np.where(rng.integers(0, 2, n) == 0, uni, spec).astype(np.int64)


# --------------------------------------------------------------------------- tests


@cocotb.test()
async def test_random_sweep(dut):
    """Every op over random operands, coefficients and shifts, back to back."""
    await setup(dut)
    rng = np.random.default_rng(0x5EED)
    done = 0
    chunk = 250000
    while done < SWEEP:
        n = min(chunk, SWEEP - done)
        op = np.array(OPS, dtype=np.int64)[rng.integers(0, len(OPS), n)]
        done += await check(
            dut,
            op,
            gen32(rng, n),
            gen32(rng, n),
            gen16(rng, n),
            gen16(rng, n),
            gen_sh(rng, n),
            "random sweep",
        )
    dut._log.info(f"random sweep: {done} elements")


@cocotb.test()
async def test_random_extremes(dut):
    """The same sweep restricted to the operand extremes, where saturation and the
    round-half boundary are dense."""
    await setup(dut)
    rng = np.random.default_rng(0xC0FFEE)
    done = 0
    chunk = 250000
    n_sat = 0
    while done < SWEEP:
        n = min(chunk, SWEEP - done)
        op = np.array(OPS, dtype=np.int64)[rng.integers(0, len(OPS), n)]
        a = SPECIAL32[rng.integers(0, SPECIAL32.size, n)]
        b = SPECIAL32[rng.integers(0, SPECIAL32.size, n)]
        c = SPECIAL16[rng.integers(0, SPECIAL16.size, n)]
        c2 = SPECIAL16[rng.integers(0, SPECIAL16.size, n)]
        sh = gen_sh(rng, n)
        _, s_r, _ = lane_ref(op, a, b, c, c2, sh)
        n_sat += int(s_r.sum())
        done += await check(dut, op, a, b, c, c2, sh, "extremes")
    assert n_sat > 0, "the extreme sweep never saturated"
    dut._log.info(f"extremes sweep: {done} elements, {n_sat} saturating")


def pow2_pair(h: int) -> tuple[int, int]:
    """int32 ``(a, b)`` whose product is exactly ``2**h`` (``0 <= h <= 62``)."""
    ja = min(h, 30)
    jb = h - ja
    if jb <= 30:
        return 1 << ja, 1 << jb
    if jb == 31:
        return -(1 << ja), I32_MIN
    return I32_MIN, I32_MIN


@cocotb.test()
async def test_round_half_mul32(dut):
    """L_MUL32 at every shift from 0 to 63, with products sitting exactly on the
    round-half boundary and one step either side of it."""
    await setup(dut)
    op: list[int] = []
    a: list[int] = []
    b: list[int] = []
    sh: list[int] = []
    halves: dict[int, int] = {}
    for s in range(64):
        cases: list[tuple[int, int]] = [(1, 1), (I32_MIN, I32_MIN), (I32_MAX, I32_MAX)]
        if s:
            pa, pb = pow2_pair(s - 1)
            cases.append((pa, pb))  # + 2**(s-1): exactly the half
            if pa != I32_MIN:
                cases.append((-pa, pb))  # - 2**(s-1)
            elif pb != I32_MIN:
                cases.append((pa, -pb))
            if s <= 32:
                half = 1 << (s - 1)
                for q in (0, 1, -1, 3, -5):
                    for off in (-1, 0, 1):
                        v = q * (1 << s) + half + off
                        if I32_MIN <= v <= I32_MAX:
                            cases.append((v, 1))
                            cases.append((v, -1))
        for pa, pb in cases:
            op.append(L_MUL32)
            a.append(pa)
            b.append(pb)
            sh.append(s)
            if s and ((pa * pb) & ((1 << s) - 1)) == (1 << (s - 1)):
                halves[s] = halves.get(s, 0) + 1
    missing = [s for s in range(1, 64) if s not in halves]
    assert not missing, f"no exact round-half case at shift(s) {missing}"
    n = len(op)
    await check(dut, op, a, b, [0] * n, [0] * n, sh, "round half (64-bit)")
    dut._log.info(f"round half, 64-bit product: {n} elements, halves at shifts 1..63")


@cocotb.test()
async def test_round_half_mul16(dut):
    """L_MUL16 and the rotation ops at every shift, with the 49-bit product on the
    round-half boundary wherever a 32-bit operand and a 16-bit coefficient reach it."""
    await setup(dut)
    op: list[int] = []
    a: list[int] = []
    b: list[int] = []
    c: list[int] = []
    c2: list[int] = []
    sh: list[int] = []
    halves: dict[int, int] = {}

    def add(o: int, va: int, vb: int, vc: int, vc2: int, s: int, prod: int) -> None:
        op.append(o)
        a.append(va)
        b.append(vb)
        c.append(vc)
        c2.append(vc2)
        sh.append(s)
        if s and (prod & ((1 << s) - 1)) == (1 << (s - 1)):
            halves[s] = halves.get(s, 0) + 1

    for s in range(64):
        add(L_MUL16, I32_MIN, 0, 0xFFFF, 0, s, I32_MIN * 0xFFFF)
        add(L_MUL16, I32_MAX, 0, 0xFFFF, 0, s, I32_MAX * 0xFFFF)
        add(L_ROPE_A, I32_MIN, I32_MIN, 0x8000, 0x8000, s, I32_MIN * -32768 - I32_MIN * -32768)
        if s == 0:
            continue
        h = s - 1
        ja, jc = min(h, 31), h - min(h, 31)
        if jc <= 15:
            va = -(1 << ja) if ja == 31 else (1 << ja)
            vc = 1 << jc
            add(L_MUL16, va, 0, vc, 0, s, va * vc)
            add(L_ROPE_A, va, 0, vc, 0, s, va * vc)
            add(L_ROPE_B, 0, va, vc, 0, s, va * vc)
            if va != I32_MIN:
                add(L_MUL16, -va, 0, vc, 0, s, -va * vc)
                add(L_ROPE_A, -va, 0, vc, 0, s, -va * vc)
        if s <= 32:
            half = 1 << (s - 1)
            for q in (0, 1, -1, 3):
                for off in (-1, 0, 1):
                    v = q * (1 << s) + half + off
                    if I32_MIN <= v <= I32_MAX:
                        add(L_MUL16, v, 0, 1, 0, s, v)
                        add(L_ROPE_A, v, 0, 1, 0, s, v)
    reach = [s for s in range(1, 48) if s not in halves]
    assert not reach, f"no exact round-half case at shift(s) {reach} on the 49-bit product"
    n = len(op)
    await check(dut, op, a, b, c, c2, sh, "round half (49-bit)")
    dut._log.info(f"round half, 49-bit product: {n} elements, halves at shifts 1..47")


@cocotb.test()
async def test_saturation_boundaries(dut):
    """Every saturating op driven to both limits and to the largest result that still
    fits, with the flag and the clamped value checked against numerics.sat."""
    await setup(dut)
    op: list[int] = []
    a: list[int] = []
    b: list[int] = []
    c: list[int] = []
    c2: list[int] = []
    sh: list[int] = []

    def add(o, va, vb, vc, vc2, s):
        op.append(o)
        a.append(va)
        b.append(vb)
        c.append(vc)
        c2.append(vc2)
        sh.append(s)

    # Exactly at each limit, then one unit past it, for every op that saturates.
    for lim in (I32_MAX, I32_MIN):
        step = 1 if lim > 0 else -1
        add(L_MUL32, lim, 1, 0, 0, 0)
        add(L_MUL32, lim, 2, 0, 0, 0)
        add(L_MUL16, lim, 0, 1, 0, 0)
        add(L_MUL16, lim, 0, 2, 0, 0)
        add(L_ROPE_A, lim, 0, 1, 0, 0)
        add(L_ROPE_A, lim, -step, 1, 1, 0)
        add(L_ROPE_B, 0, lim, 1, 0, 0)
        add(L_ROPE_B, step, lim, 1, 1, 0)
        add(L_SUB, lim, 0, 0, 0, 0)
        add(L_SUB, lim, -step, 0, 0, 0)
        add(L_PASS, lim, lim, 0xFFFF, 0xFFFF, 63)
    # The same limits reached through a shift: the product one unit past the limit
    # after rounding, over the shifts a compiled program uses.
    for s in range(31):
        lim = 1 << s
        add(L_MUL32, I32_MAX, lim, 0, 0, s)
        add(L_MUL32, I32_MIN, lim, 0, 0, s)
        if lim + 1 <= I32_MAX:
            add(L_MUL32, I32_MAX, lim + 1, 0, 0, s)
            add(L_MUL32, I32_MIN, lim + 1, 0, 0, s)
    # Full-scale coefficients against the extreme operands, over every shift.
    for s in range(64):
        add(L_MUL16, I32_MAX, 0, 0xFFFF, 0, s)
        add(L_MUL16, I32_MIN, 0, 0xFFFF, 0, s)
        add(L_ROPE_A, I32_MIN, I32_MAX, 0x8000, 0x7FFF, s)
        add(L_ROPE_B, I32_MIN, I32_MAX, 0x8000, 0x7FFF, s)

    y_g, s_g, _ = await run_batch(dut, op, a, b, c, c2, sh)
    op_a = np.array(op)
    y_r, s_r, _ = lane_ref(op_a, np.array(a), np.array(b), np.array(c), np.array(c2), np.array(sh))
    assert np.array_equal(y_g, y_r), "saturation values differ from numerics"
    assert np.array_equal(s_g, s_r), "saturation flags differ from numerics"
    assert not np.any(s_g[op_a == L_PASS]), "L_PASS raised sat"
    for code in (L_MUL32, L_MUL16, L_ROPE_A, L_ROPE_B, L_SUB):
        m = op_a == code
        for val, flag, what in (
            (I32_MAX, 1, "saturated high"),
            (I32_MIN, 1, "saturated low"),
            (I32_MAX, 0, "reached I32_MAX without saturating"),
            (I32_MIN, 0, "reached I32_MIN without saturating"),
        ):
            assert np.any(m & (y_g == val) & (s_g == flag)), f"{OP_NAME[code]} never {what}"
    dut._log.info(f"saturation: {len(op)} elements, {int(s_g.sum())} saturating")


@cocotb.test()
async def test_zero_rules(dut):
    """A zero operand or coefficient gives zero with no saturation, at every shift."""
    await setup(dut)
    op: list[int] = []
    a: list[int] = []
    b: list[int] = []
    c: list[int] = []
    c2: list[int] = []
    sh: list[int] = []
    for s in (0, 1, 15, 31, 62, 63):
        for v in (0, 1, -1, I32_MIN, I32_MAX):
            op += [L_MUL32, L_MUL32, L_MUL16, L_ROPE_A, L_ROPE_B, L_SUB, L_SUB, L_PASS]
            a += [0, v, 0, 0, 0, v, 0, 0]
            b += [v, 0, v, 0, 0, v, 0, v]
            c += [0, 0, 0, 0, 0, 0, 0, 0]
            c2 += [0, 0, 0, 0, 0, 0, 0, 0]
            sh += [s] * 8
            op += [L_MUL16, L_ROPE_A, L_ROPE_B]
            a += [v, v, v]
            b += [v, v, v]
            c += [0, 0, 0]
            c2 += [0xFFFF, 0, 0]
            sh += [s] * 3
    y_g, s_g, p_g = await run_batch(dut, op, a, b, c, c2, sh)
    assert not np.any(y_g), f"a zero operand produced {y_g[y_g != 0][:4]}"
    assert not np.any(s_g), "a zero operand raised sat"
    y_r, s_r, p_r = lane_ref(
        np.array(op), np.array(a), np.array(b), np.array(c), np.array(c2), np.array(sh)
    )
    assert np.array_equal(y_g, y_r) and np.array_equal(s_g, s_r)
    assert np.array_equal(to_signed64(p_g), p_r), "p64 differs on the zero cases"
    dut._log.info(f"zero rules: {len(op)} elements")


@cocotb.test()
async def test_coefficient_width(dut):
    """L_MUL16 reads its coefficient as u16 (Rc_m and Sv_m reach 65535, sigmoid 32768)
    and the rotation ops read theirs as int16; L_ROPE_A with b = 0 is the signed
    32 x 17 multiply the gamma stage of VRMSNORM needs."""
    await setup(dut)
    op: list[int] = []
    a: list[int] = []
    b: list[int] = []
    c: list[int] = []
    c2: list[int] = []
    sh: list[int] = []
    coeffs = [0x8000, 0x8001, 0xC000, 0xFFFF, 0x7FFF, 0x4000, 1]
    vals = [1, -1, 12345, -12345, I32_MAX, I32_MIN, 1 << 20, -(1 << 20)]
    for co in coeffs:
        for v in vals:
            for s in (0, 14, 15, 16):
                op += [L_MUL16, L_ROPE_A, L_ROPE_B]
                a += [v, v, 0]
                b += [0, 0, v]
                c += [co, co, co]
                c2 += [0, 0, 0]
                sh += [s, s, s]
    y_g, s_g, _ = await run_batch(dut, op, a, b, c, c2, sh)
    y_r, s_r, _ = lane_ref(
        np.array(op), np.array(a), np.array(b), np.array(c), np.array(c2), np.array(sh)
    )
    assert np.array_equal(y_g, y_r), "coefficient widening differs from numerics"
    assert np.array_equal(s_g, s_r)
    # The two readings really do differ on a coefficient with bit 15 set.
    i = [
        k
        for k in range(len(op))
        if op[k] == L_MUL16 and c[k] == 0xFFFF and a[k] == 12345 and sh[k] == 0
    ][0]
    j = [
        k
        for k in range(len(op))
        if op[k] == L_ROPE_A and c[k] == 0xFFFF and a[k] == 12345 and sh[k] == 0
    ][0]
    assert int(y_g[i]) == 12345 * 0xFFFF, "L_MUL16 did not read its coefficient as u16"
    assert int(y_g[j]) == -12345, "L_ROPE_A did not read its coefficient as int16"
    dut._log.info(f"coefficient width: {len(op)} elements")


@cocotb.test()
async def test_gamma_route(dut):
    """The VRMSNORM gamma stage, sat32(round_shift49(xhat * gamma, G)), reaches the
    lane two ways: L_ROPE_A with b = 0 and L_MUL32 with gamma sign-extended."""
    await setup(dut)
    rng = np.random.default_rng(0x9A33)
    n = 40000
    xhat = rng.integers(-(1 << 21), 1 << 21, n, dtype=np.int64)
    gamma = rng.integers(-32767, 32768, n, dtype=np.int64)
    g_shift = rng.integers(6, 31, n, dtype=np.int64)  # G = -gamma_e keeps xhat * gamma in int32
    want = numerics.sat(round_shift_arr(xhat * gamma, g_shift), 32)
    y_a, s_a, _ = await run_batch(
        dut,
        [L_ROPE_A] * n,
        xhat,
        np.zeros(n, dtype=np.int64),
        gamma & M16,
        np.zeros(n, dtype=np.int64),
        g_shift,
    )
    y_b, s_b, _ = await run_batch(
        dut,
        [L_MUL32] * n,
        xhat,
        gamma,
        np.zeros(n, dtype=np.int64),
        np.zeros(n, dtype=np.int64),
        g_shift,
    )
    assert np.array_equal(y_a, want), "L_ROPE_A with b = 0 is not the signed gamma multiply"
    assert np.array_equal(y_b, want), "L_MUL32 with a sign-extended gamma differs"
    assert not np.any(s_a) and not np.any(s_b), "the gamma stage saturated on in-range data"
    dut._log.info(f"gamma route: {2 * n} elements")


@cocotb.test()
async def test_timing(dut):
    """out_valid two cycles after in_valid, with gaps, single elements, back-to-back
    elements and a reset in the middle of the pipeline."""
    await setup(dut)
    # one element on its own: out_valid exactly two clock edges later
    dut.in_valid.value = 1
    dut.op.value = L_MUL32
    dut.a.value = 7
    dut.b.value = 6
    dut.sh.value = 0
    await FallingEdge(dut.clk)
    dut.in_valid.value = 0
    assert int(dut.out_valid.value) == 0, "out_valid rose one cycle after in_valid"
    await FallingEdge(dut.clk)
    assert int(dut.out_valid.value) == 1, "out_valid did not rise two cycles after in_valid"
    assert int(dut.y.value.to_unsigned()) == 42
    assert int(dut.p64.value.to_unsigned()) == 42
    await FallingEdge(dut.clk)
    assert int(dut.out_valid.value) == 0, "out_valid stayed high for a single element"

    # gaps: only the driven elements produce a result, and the values are unchanged
    rng = np.random.default_rng(0x71)
    seen: list[tuple[int, int]] = []
    pend: list[int] = []
    expect: list[int] = []
    for i in range(400):
        drive = bool(rng.integers(0, 2))
        va = int(rng.integers(-(1 << 20), 1 << 20))
        dut.in_valid.value = int(drive)
        dut.op.value = L_MUL32
        dut.a.value = va & M32
        dut.b.value = 3
        dut.sh.value = 1
        await FallingEdge(dut.clk)
        if drive:
            pend.append(numerics.round_shift(va * 3, 1))
        seen.append((int(dut.out_valid.value), dut.y.value.to_unsigned()))
        if i >= 2 and seen[-1][0]:
            expect.append(int(dut.y.value.to_unsigned()))
    dut.in_valid.value = 0
    for _ in range(3):
        await FallingEdge(dut.clk)
        if int(dut.out_valid.value):
            expect.append(int(dut.y.value.to_unsigned()))
    got = [wrap32(np.array([v], dtype=np.int64))[0] for v in expect]
    assert len(got) == len(pend), f"{len(got)} results for {len(pend)} elements"
    assert all(int(g) == p for g, p in zip(got, pend, strict=True)), "a gapped element changed"

    # reset clears out_valid while elements are in flight
    dut.in_valid.value = 1
    dut.a.value = 5
    dut.b.value = 5
    await FallingEdge(dut.clk)
    dut.rst.value = 1
    dut.in_valid.value = 0
    await FallingEdge(dut.clk)
    await FallingEdge(dut.clk)
    assert int(dut.out_valid.value) == 0, "reset did not clear out_valid"
    dut.rst.value = 0
    await FallingEdge(dut.clk)
    dut._log.info("timing: latency, gaps and reset")
