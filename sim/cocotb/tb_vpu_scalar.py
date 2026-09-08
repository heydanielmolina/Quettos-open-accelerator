"""cocotb tests of qcore_vpu_scalar: every response bit-exact against sw/quettos/numerics.py.

The two table domains are swept exhaustively -- all 196608 values of the rsqrt
argument m_q16 in [2^16, 2^18) and all 32768 values of the reciprocal argument in
[2^15, 2^16) -- and the three requests are then driven over random magnitudes,
classes and sfloat constants. Directed cases cover the leading-one search at
every bit length, values that are already normal, the mantissa range of
sfloat_mul over adversarial pairs including the rounding overflow that folds
into {2^15, e + 1}, the canonical zero, the shift clamp at both ends, and the
fixed latency with gaps, back-to-back requests and reset.

rsp_sx_m / rsp_sx_e are the VQUANT scale; the unit forms them from the same
fields for every request, so the tests check them on all three.
"""

from __future__ import annotations

import functools
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge
from quettos import numerics
from quettos.numerics import SFLOAT_ONE, SFLOAT_ZERO, SFloat

OP_RMS, OP_QUANT, OP_SOFT = 0, 1, 2
OP_NAME = {OP_RMS: "RMS_SCALE", OP_QUANT: "QUANT_SCALE", OP_SOFT: "SOFTMAX_NORM"}

M16 = (1 << 16) - 1
LAT = 5  # register stages between req_valid and rsp_valid

# Requests driven by each randomised sweep; QCORE_VPU_SCALAR_REQUESTS scales them.
SWEEP = int(os.environ.get("QCORE_VPU_SCALAR_REQUESTS", "1000000"))

TABLES = numerics.load_tables()


def i8(v: int) -> int:
    """The i8 an exponent is carried in (docs/RTL.md 4)."""
    return ((v + 128) & 0xFF) - 128


class Req:
    """One scalar request and the response numerics defines for it."""

    __slots__ = (
        "op",
        "x",
        "sh0",
        "sh",
        "w8",
        "mul_en",
        "aux",
        "m",
        "shift",
        "shift_err",
        "sx",
        "zero",
    )

    def __init__(
        self,
        op: int,
        x: int,
        *,
        sh0: int = 16,
        sh: int = 0,
        w8: bool = False,
        mul_en: bool = False,
        aux: SFloat = SFLOAT_ONE,
    ) -> None:
        self.op, self.x, self.sh0, self.sh = op, x, sh0, sh
        self.w8, self.mul_en, self.aux = w8, mul_en, aux
        self.zero = x == 0
        width = 8 if w8 else 16
        length = numerics.bitlen(x)

        # The VQUANT scale, formed for every request from req_x and the classes.
        if self.zero:
            self.sx = SFLOAT_ZERO
        else:
            e_a = length - 16
            a_hi = (x >> e_a) if e_a >= 0 else (x << -e_a)
            sx = SFloat(a_hi, i8(e_a - (width - 1) - sh0))
            self.sx = numerics.sfloat_mul(sx, aux) if mul_en else sx

        if self.zero:
            self.m, self.shift, self.shift_err = None, None, False
            return
        if op == OP_RMS:
            e2 = (length - 1) & ~1
            e = e2 >> 1
            m_q16 = (x >> (e2 - 16)) if e2 >= 16 else (x << (16 - e2))
            r = numerics.rsqrt_q15(m_q16, TABLES)
            rc = numerics.sfloat_mul(numerics.sfloat_from_int(r, -15), aux)
            assert i8(rc.e) == rc.e, "the test stimulus pushed Rc_e outside i8"
            self.m = rc.m
            raw = -(rc.e + sh0 - sh - e)
        else:
            e_a = length - 16
            hi = (x >> e_a) if e_a >= 0 else (x << -e_a)
            self.m = numerics.recip_q15(hi, TABLES)
            raw = (31 + e_a - width) if op == OP_QUANT else (7 + e_a)
        assert i8(self.sx.e) == self.sx.e, "the test stimulus pushed Sx_e outside i8"
        self.shift = min(max(raw, 0), 63)
        self.shift_err = raw < 0 or raw > 63

    def fields(self) -> tuple[int, ...]:
        return (
            self.op,
            self.x,
            self.sh0,
            self.sh,
            int(self.w8),
            int(self.mul_en),
            self.aux.m,
            self.aux.e & 0xFF,
        )

    def __repr__(self) -> str:
        return (
            f"{OP_NAME[self.op]}(x={self.x}, sh0={self.sh0}, sh={self.sh}, "
            f"w8={self.w8}, mul={self.mul_en}, aux={{{self.aux.m}, {self.aux.e}}})"
        )


# --------------------------------------------------------------------------- driver


async def setup(dut) -> None:
    """Start the clock, hold reset, leave every input at zero."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    for sig in (
        dut.req_valid,
        dut.req_op,
        dut.req_x,
        dut.req_sh0,
        dut.req_sh,
        dut.req_w8,
        dut.req_mul_en,
        dut.req_aux_m,
        dut.req_aux_e,
    ):
        sig.value = 0
    for _ in range(3):
        await FallingEdge(dut.clk)
    dut.rst.value = 0
    await FallingEdge(dut.clk)


async def run_batch(dut, reqs: list[Req]) -> list[tuple[int, ...]]:
    """Drive one request per cycle; returns (m, shift, err, sx_m, sx_e, zero) per request."""
    n = len(reqs)
    fields = [r.fields() for r in reqs]
    s_v, s_op, s_x, s_sh0, s_sh, s_w8, s_mul, s_am, s_ae = (
        dut.req_valid,
        dut.req_op,
        dut.req_x,
        dut.req_sh0,
        dut.req_sh,
        dut.req_w8,
        dut.req_mul_en,
        dut.req_aux_m,
        dut.req_aux_e,
    )
    o_v, o_m, o_sh, o_err, o_sxm, o_sxe, o_z = (
        dut.rsp_valid,
        dut.rsp_m,
        dut.rsp_shift,
        dut.rsp_shift_err,
        dut.rsp_sx_m,
        dut.rsp_sx_e,
        dut.rsp_zero,
    )
    out: list[tuple[int, ...]] = []
    for i in range(n + LAT - 1):
        if i < n:
            op, x, sh0, sh, w8, mul, am, ae = fields[i]
            s_v.value = 1
            s_op.value = op
            s_x.value = x
            s_sh0.value = sh0
            s_sh.value = sh
            s_w8.value = w8
            s_mul.value = mul
            s_am.value = am
            s_ae.value = ae
        else:
            s_v.value = 0
        await FallingEdge(dut.clk)
        if i >= LAT - 1:
            assert int(o_v.value) == 1, f"rsp_valid low at request {i - LAT + 1}"
            out.append(
                (
                    o_m.value.to_unsigned(),
                    o_sh.value.to_unsigned(),
                    int(o_err.value),
                    o_sxm.value.to_unsigned(),
                    o_sxe.value.to_unsigned(),
                    int(o_z.value),
                )
            )
    s_v.value = 0
    await FallingEdge(dut.clk)
    assert int(o_v.value) == 0, "rsp_valid stayed high after the last request"
    return out


def message(label: str, i: int, req: Req, what: str, g: int, w: int) -> str:
    """The text of one failed field comparison, naming the request that produced it."""
    return f"{label}: request {i} {req}: {what} got {g}, want {w}"


async def check(dut, reqs: list[Req], label: str) -> int:
    """Run a batch and compare every response against numerics; returns the count."""
    got = await run_batch(dut, reqs)
    for i, (r, (m, shift, err, sxm, sxe, zero)) in enumerate(zip(reqs, got, strict=True)):
        bad = functools.partial(message, label, i, r)
        assert zero == int(r.zero), bad("rsp_zero", zero, int(r.zero))
        assert sxm == r.sx.m, bad("rsp_sx_m", sxm, r.sx.m)
        assert sxe == (r.sx.e & 0xFF), bad("rsp_sx_e", sxe, r.sx.e & 0xFF)
        if r.zero:
            assert err == 0, bad("rsp_shift_err on a zero input", err, 0)
            continue
        assert m == r.m, bad("rsp_m", m, r.m)
        assert shift == r.shift, bad("rsp_shift", shift, r.shift)
        assert err == int(r.shift_err), bad("rsp_shift_err", err, int(r.shift_err))
    return len(reqs)


# --------------------------------------------------------------------------- stimulus


AUX_M = (1 << 15, (1 << 15) + 1, 0x9999, 0xBFFF, 0xC000, 0xD555, 0xFFFE, 0xFFFF, 0xAAAA)


def rand_aux(rng) -> SFloat:
    """A valid sfloat constant: the canonical zero or a normalised mantissa."""
    if rng.random() < 0.08:
        return SFLOAT_ZERO
    m = rng.choice(AUX_M) if rng.random() < 0.4 else rng.randrange(1 << 15, 1 << 16)
    return SFloat(m, rng.randint(-60, 60))


def rand_x(rng, bits: int) -> int:
    """A magnitude of up to ``bits`` bits, weighted onto the bit-length boundaries."""
    k = rng.randint(0, bits)
    if k == 0:
        return 0
    base = 1 << (k - 1)
    kind = rng.randrange(4)
    if kind == 0:
        return base
    if kind == 1:
        return (base << 1) - 1
    if kind == 2:
        return base + rng.randrange(base)
    return min((1 << bits) - 1, base + rng.randrange(base) + rng.randrange(3) - 1) or 1


# --------------------------------------------------------------------------- tests


@cocotb.test()
async def test_rsqrt_domain(dut):
    """Every value of the rsqrt argument: ss' swept over [2^16, 2^18), where the
    normalisation is the identity and m_q16 addresses each table entry and fraction."""
    await setup(dut)
    total = 0
    for lo in range(1 << 16, 1 << 18, 1 << 15):
        reqs = [Req(OP_RMS, x, sh0=16, sh=3) for x in range(lo, lo + (1 << 15))]
        total += await check(dut, reqs, "rsqrt domain")
    assert total == (1 << 18) - (1 << 16)
    dut._log.info(f"rsqrt domain: {total} requests (exhaustive over m_q16)")


@cocotb.test()
async def test_recip_domain(dut):
    """Every value of the reciprocal argument: a_eff swept over [2^15, 2^16), which is
    already normal, so a_hi is the input itself and every entry and fraction is hit."""
    await setup(dut)
    total = 0
    for op in (OP_QUANT, OP_SOFT):
        for lo in range(1 << 15, 1 << 16, 1 << 14):
            reqs = [Req(op, x, sh0=12, w8=bool(x & 1)) for x in range(lo, lo + (1 << 14))]
            total += await check(dut, reqs, "recip domain")
    assert total == 2 * (1 << 15)
    dut._log.info(f"recip domain: {total} requests (exhaustive over a_hi and sum_hi)")


@cocotb.test()
async def test_random_rms(dut):
    """RMS_SCALE over random ss' up to 49 bits with random FRAC_X, sh and sqrt(d)."""
    import random

    await setup(dut)
    rng = random.Random(0x5CA1)
    done = 0
    errs = 0
    while done < SWEEP:
        n = min(50000, SWEEP - done)
        reqs = [
            Req(
                OP_RMS,
                rand_x(rng, 49),
                sh0=rng.randint(0, 30),
                sh=rng.randint(0, 17),
                aux=rand_aux(rng),
            )
            for _ in range(n)
        ]
        errs += sum(r.shift_err for r in reqs if not r.zero)
        done += await check(dut, reqs, "random RMS_SCALE")
    assert errs > 0, "the RMS sweep never clamped S1"
    dut._log.info(f"random RMS_SCALE: {done} requests, {errs} shift clamps")


@cocotb.test()
async def test_random_quant_softmax(dut):
    """QUANT_SCALE and SOFTMAX_NORM over random magnitudes, both output widths and
    scale_mul on and off.

    ``a_eff`` is 33 bits; the softmax total is the sum of at most ``2**24``
    exponentials of at most ``2**23`` each, so 47 bits is the widest value the
    vector unit's accumulator presents on ``req_x``.
    """
    import random

    await setup(dut)
    rng = random.Random(0xC0DE)
    done = 0
    muls = 0
    while done < SWEEP:
        n = min(50000, SWEEP - done)
        reqs = []
        for _ in range(n):
            op = OP_QUANT if rng.random() < 0.6 else OP_SOFT
            mul = op == OP_QUANT and rng.random() < 0.5
            reqs.append(
                Req(
                    op,
                    rand_x(rng, 33 if op == OP_QUANT else 47),
                    sh0=rng.randint(0, 30),
                    w8=bool(rng.getrandbits(1)),
                    mul_en=mul,
                    aux=rand_aux(rng),
                )
            )
            muls += int(mul)
        done += await check(dut, reqs, "random QUANT / SOFTMAX")
    dut._log.info(f"random QUANT_SCALE / SOFTMAX_NORM: {done} requests, {muls} with SCALE_MUL")


@cocotb.test()
async def test_bit_lengths(dut):
    """The leading-one search at every bit length of every op, including the values
    that are already normal and need no shift."""
    await setup(dut)
    reqs: list[Req] = []
    for k in range(1, 50):
        base = 1 << (k - 1)
        for x in (base, base + 1, (base << 1) - 1, base | (base >> 1)):
            for op in (OP_RMS, OP_QUANT, OP_SOFT):
                reqs.append(Req(op, x, sh0=16, sh=k % 18, w8=bool(k & 1)))
    # Already normal: 16 significant bits for the reciprocal, [2^16, 2^18) for rsqrt.
    for x in (1 << 15, (1 << 16) - 1, 0xABCD | (1 << 15)):
        reqs.append(Req(OP_QUANT, x, sh0=16))
        reqs.append(Req(OP_SOFT, x, sh0=16))
    for x in (1 << 16, (1 << 17) - 1, 1 << 17, (1 << 18) - 1):
        reqs.append(Req(OP_RMS, x, sh0=16, sh=2))
    n = await check(dut, reqs, "bit lengths")
    dut._log.info(f"bit lengths: {n} requests")


@cocotb.test()
async def test_zero_input(dut):
    """A zero magnitude reports rsp_zero, leaves the canonical zero scale and raises
    no shift error, for every op and every class."""
    await setup(dut)
    reqs = [
        Req(op, 0, sh0=sh0, sh=sh, w8=bool(w8), mul_en=bool(mul), aux=aux)
        for op in (OP_RMS, OP_QUANT, OP_SOFT)
        for sh0 in (0, 16, 30)
        for sh in (0, 17)
        for w8 in (0, 1)
        for mul in (0, 1)
        for aux in (SFLOAT_ZERO, SFloat(1 << 15, -15), SFloat(0xFFFF, 7))
    ]
    # A non-zero request on either side, so the zero case is not a quiet pipeline.
    mixed: list[Req] = []
    for r in reqs:
        mixed.append(Req(OP_QUANT, 0x1234_5678, sh0=16))
        mixed.append(r)
    n = await check(dut, mixed, "zero input")
    dut._log.info(f"zero input: {n} requests")


@cocotb.test()
async def test_sfloat_mul_property(dut):
    """VQUANT scale_mul over adversarial mantissa pairs: the product mantissa stays in
    [2^15, 2^16), a zero operand gives the canonical zero, and the rounding overflow
    to 2^16 folds into {2^15, e + 1}."""
    await setup(dut)
    edge = [1 << 15, (1 << 15) + 1, 0xBFFF, 0xC000, 0xFFFE, 0xFFFF, 0xAAAA, 0xB504, 0xB505]
    reqs: list[Req] = []
    for am in edge:
        for xm in edge:
            for e in (-60, -15, 0, 7, 60):
                reqs.append(Req(OP_QUANT, xm, sh0=16, mul_en=True, aux=SFloat(am, e)))
    # Pairs whose 32-bit product rounds up to 2^16 and folds back to {2^15, e + 1}.
    folds = 0
    for xm in range(1 << 15, 1 << 16):
        am = ((1 << 31) - (1 << 14)) // xm + 1
        if not (1 << 15) <= am < (1 << 16):
            continue
        p = xm * am
        if p < (1 << 31) and numerics.round_shift(p, 15) == (1 << 16):
            reqs.append(Req(OP_QUANT, xm, sh0=16, mul_en=True, aux=SFloat(am, -15)))
            folds += 1
            if folds >= 64:
                break
    assert folds > 0, "no adversarial pair reached the mantissa rounding overflow"
    # A zero scale_mul, and a zero magnitude against a non-zero constant.
    reqs.append(Req(OP_QUANT, 0xDEAD, sh0=16, mul_en=True, aux=SFLOAT_ZERO))
    reqs.append(Req(OP_QUANT, 0, sh0=16, mul_en=True, aux=SFloat(0xFFFF, 3)))
    n = await check(dut, reqs, "sfloat_mul")
    for r in reqs:
        assert r.sx.m == 0 or (1 << 15) <= r.sx.m < (1 << 16), "numerics left the mantissa range"
    dut._log.info(f"sfloat_mul: {n} requests, {folds} rounding overflows")


def rms_boundary(x: int, target: int) -> Req:
    """A RMS_SCALE request whose raw shift is exactly ``target``.

    ``raw = -(Rc_e + FRAC_X - sh - e)``, so with ``FRAC_X`` and ``sh`` free the
    two fields place it anywhere: ``FRAC_X`` walks it down, ``sh`` walks it up.
    """
    length = numerics.bitlen(x)
    e2 = (length - 1) & ~1
    m_q16 = (x >> (e2 - 16)) if e2 >= 16 else (x << (16 - e2))
    r = numerics.rsqrt_q15(m_q16, TABLES)
    rc = numerics.sfloat_mul(numerics.sfloat_from_int(r, -15), SFLOAT_ONE)
    slack = (e2 >> 1) - rc.e - target  # FRAC_X - sh
    sh0, sh = (slack, 0) if slack >= 0 else (0, -slack)
    assert 0 <= sh0 <= 255 and 0 <= sh <= 63, f"no field pair reaches raw = {target} for x = {x}"
    req = Req(OP_RMS, x, sh0=sh0, sh=sh)
    assert req.shift == min(max(target, 0), 63) and req.shift_err == (not 0 <= target <= 63)
    return req


@cocotb.test()
async def test_shift_clamp(dut):
    """The shift clamp at both ends: a negative S1 and one past 63 both clamp and
    raise rsp_shift_err, and an in-range shift raises neither.

    The four boundary values are driven exactly: -1 and 64 clamp and raise the
    flag, 0 and 63 are the first and last shift the field carries and raise
    nothing.  ``numerics.rmsnorm`` clamps into the same range and counts the
    same event, so the two models meet at the edge as well as inside it."""
    await setup(dut)
    reqs: list[Req] = []
    for x in (1 << 16, (1 << 30) - 1, (1 << 48) + 12345):
        reqs += [rms_boundary(x, target) for target in (-1, 0, 63, 64)]
    # S1 = -(Rc_e + FRAC_X - sh - e): a large FRAC_X with a small e drives it negative.
    for frac in (0, 16, 40, 120, 200, 255):
        for sh in (0, 8, 17, 63):
            for x in (1 << 16, (1 << 30) - 1, (1 << 48) + 12345):
                reqs.append(Req(OP_RMS, x, sh0=frac, sh=sh, aux=SFloat(1 << 15, -15)))
    neg = sum(1 for r in reqs if not r.zero and r.shift == 0 and r.shift_err)
    big = sum(1 for r in reqs if not r.zero and r.shift == 63 and r.shift_err)
    ok = sum(1 for r in reqs if not r.zero and not r.shift_err)
    assert neg and big and ok, f"clamp coverage: {neg} negative, {big} past 63, {ok} in range"
    n = await check(dut, reqs, "shift clamp")
    dut._log.info(f"shift clamp: {n} requests, {neg} negative, {big} past 63")


@cocotb.test()
async def test_shift_clamp_matches_rmsnorm(dut):
    """The clamp the unit applies is the one ``numerics.rmsnorm`` applies.

    One vector, one gamma and a ``sqrt(d)`` constant swept across the top of the
    shift field: the reference clamps ``S1`` into ``[0, 63]`` and counts one
    ``err_shift`` per element at either end, and the mantissa and shift the unit
    answers with reproduce exactly that."""
    await setup(dut)
    import numpy as np

    n = 16
    x = np.array([1, -1] * (n // 2), dtype=np.int64)
    gamma = np.ones(n, dtype=np.int64)
    reqs, expected = [], []
    for sqrt_e in (-59, -61, -62, -100):
        sqrt_d = SFloat(1 << 15, sqrt_e)
        st = numerics.Stats()
        numerics.rmsnorm(x, gamma, 0, 0, sqrt_d, 0, TABLES, st)
        expected.append(st.err_shift)
        reqs.append(Req(OP_RMS, int(np.sum(x * x)), sh0=0, sh=0, aux=sqrt_d))
    assert [e // n for e in expected] == [r.shift_err for r in reqs], (
        "the reference and the request model disagree on which constants clamp"
    )
    assert expected == [0, 0, n, n], f"the sweep did not cross the ceiling: {expected}"
    n_req = await check(dut, reqs, "shift clamp vs rmsnorm")
    dut._log.info(f"shift clamp vs rmsnorm: {n_req} requests")


@cocotb.test()
async def test_timing(dut):
    """rsp_valid a fixed five cycles after req_valid, with a single request, gaps,
    back-to-back requests and a reset in the middle of the pipeline."""
    await setup(dut)
    r = Req(OP_QUANT, 0x1_0000, sh0=16)
    op, x, sh0, sh, w8, mul, am, ae = r.fields()
    dut.req_valid.value = 1
    dut.req_op.value = op
    dut.req_x.value = x
    dut.req_sh0.value = sh0
    dut.req_aux_m.value = am
    dut.req_aux_e.value = ae
    await FallingEdge(dut.clk)
    dut.req_valid.value = 0
    for k in range(LAT - 1):
        assert int(dut.rsp_valid.value) == 0, f"rsp_valid rose {k + 1} cycles after req_valid"
        await FallingEdge(dut.clk)
    assert int(dut.rsp_valid.value) == 1, f"rsp_valid did not rise {LAT} cycles after req_valid"
    assert dut.rsp_m.value.to_unsigned() == r.m
    assert dut.rsp_shift.value.to_unsigned() == r.shift
    await FallingEdge(dut.clk)
    assert int(dut.rsp_valid.value) == 0, "rsp_valid stayed high for a single request"

    # Gaps: only the driven requests answer, and the answers are unchanged.
    import random

    rng = random.Random(0x9)
    pend: list[Req] = []
    got: list[int] = []
    for _ in range(300):
        drive = bool(rng.getrandbits(1))
        req = Req(
            (OP_RMS, OP_QUANT, OP_SOFT)[rng.randrange(3)],
            rand_x(rng, 40) or 1,
            sh0=rng.randint(0, 30),
            sh=rng.randint(0, 17),
        )
        op, x, sh0, sh, w8, mul, am, ae = req.fields()
        dut.req_valid.value = int(drive)
        dut.req_op.value = op
        dut.req_x.value = x
        dut.req_sh0.value = sh0
        dut.req_sh.value = sh
        dut.req_w8.value = w8
        dut.req_mul_en.value = mul
        dut.req_aux_m.value = am
        dut.req_aux_e.value = ae
        await FallingEdge(dut.clk)
        if drive:
            pend.append(req)
        if int(dut.rsp_valid.value):
            got.append(dut.rsp_m.value.to_unsigned())
    dut.req_valid.value = 0
    for _ in range(LAT + 1):
        await FallingEdge(dut.clk)
        if int(dut.rsp_valid.value):
            got.append(dut.rsp_m.value.to_unsigned())
    assert len(got) == len(pend), f"{len(got)} responses for {len(pend)} requests"
    for req, m in zip(pend, got, strict=True):
        assert m == req.m, f"a gapped request changed: {req} gave {m}"

    # Reset clears rsp_valid while requests are in flight.
    dut.req_valid.value = 1
    await FallingEdge(dut.clk)
    dut.rst.value = 1
    dut.req_valid.value = 0
    for _ in range(LAT + 1):
        await FallingEdge(dut.clk)
        assert int(dut.rsp_valid.value) == 0, "reset did not clear rsp_valid"
    dut.rst.value = 0
    await FallingEdge(dut.clk)
    dut._log.info("timing: latency, gaps and reset")
