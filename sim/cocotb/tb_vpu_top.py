"""cocotb tests of qcore_vpu_top: the six V opcodes of ``docs/ISA.md``.

Every descriptor is run twice -- once on the DUT against a model of the VSRAM
banks, the SREG banks and the QMEM read port, once on ``sw/quettos/isa_sim.py``
over the same image and the same starting state -- and every element, every
scale register and every event count has to agree.  A mismatch reports the
element index it first appears at.

The randomised sweeps cover element counts of 1, a partial word, exactly one
word and several words, every source / destination / auxiliary offset within a
word, both activation rows, every flag combination of VQUANT and, in a quarter
of the cases, a destination written over its source; the directed coroutines
pin the boundary of each operation (the zero vector, an absmax of 2^31, the
sigmoid clamp and its odd symmetry, the saturating subtract, the reciprocal
bound of the quantizer, the epsilon-only RMS, the top of the RMS shift field,
the rotation's saturating edge, the softmax row of equal scores, the row far
enough below its maximum to round to zero, the token that reaches the weight
clip and the V scales that are the canonical zero), the in-place and cross-bank
halves of the range rule, the group index that saturates rather than wrapping
onto a register it already wrote, and the ranges that leave the VSRAM.  Memory
latencies 1, 32 and 200, a throttled arbiter and a sparse request grant are
swept over every operation that streams an operand.
"""

from __future__ import annotations

import dataclasses
import math
import os
import random
import struct

import cocotb
import numpy as np
import qc_numerics as qn
import qc_stream
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge
from quettos import compiler, isa, isa_sim, numerics
from quettos.isa import Descriptor, Opcode, VquantFlag
from quettos.isa_sim import softmax_len
from quettos.numerics import SFloat

WB = int(os.environ["QC_WB"])
B_MAX = int(os.environ["QC_B_MAX"])
VL = int(os.environ["QC_VL"])
VSRAM_WORDS = int(os.environ["QC_VSRAM_WORDS"])

NE = isa.VSRAM_WORD_ELEMS
ELEMS = VSRAM_WORDS * NE
MASK32 = (1 << 32) - 1
IMAGE_BYTES = 1 << 16
ROW_BASE = 4096  # where a streamed gamma / constant row starts
TABLES = numerics.load_tables()

# What every coroutine of this file has compared so far, printed at the end.
TOTALS = {"descriptors": 0, "elements": 0, "rows": 0}

# Descriptors driven by each randomised sweep; QCORE_VPU_TOP_CASES scales them.
CASES = int(os.environ.get("QCORE_VPU_TOP_CASES", "200"))


# --------------------------------------------------------------------------- the environment


class Env:
    """Models the VSRAM banks, the SREG banks and the QMEM read port of one core."""

    def __init__(self, dut, image: bytes, *, latency: int = 32, req_ready=None) -> None:
        self.dut = dut
        self.image = bytes(image)
        self.vsram = np.zeros((B_MAX, ELEMS), dtype=np.int64)
        self.sreg = [[0] * isa.SREG_COUNT for _ in range(B_MAX)]
        self.latency = latency
        self.req_ready = req_ready if req_ready is not None else (lambda c: 1)
        self.cycle = 0
        self.sat = 0
        self.err_shift = 0
        self.err_bounds = 0
        self.sreg_err = 0
        self.sreg_writes: list[tuple[int, int, int]] = []
        self.busy_cycles = 0
        self.done_cycles: list[int] = []
        self.requests: list[tuple[int, int]] = []
        self.beats_returned = 0
        self.writes: list[tuple[int, int]] = []  # (bank, word) of every accepted write
        self.src_row = 0
        self.dst_row = 0
        self.pos = 0
        # The bank the crossbar has to give port B for this descriptor's reads
        # and for its writes: 1 selects dst_row + row, 0 selects src_row + row.
        self.sel_read = 0
        self.sel_write = 1
        self._pending: list[tuple[int, int, int]] = []
        self._last_deliver = -1
        self._read_b_prev = False
        self._rd_a: tuple[int, int] | None = None
        self._rd_b: tuple[int, int] | None = None
        self._data_a = 0
        self._data_b = 0
        self._req_prev: tuple[int, int, int, int] | None = None
        self._held: tuple[int, int, int] | None = None

    # ---- memory image

    def beat(self, addr: int) -> int:
        end = min(addr + WB, len(self.image))
        raw = self.image[addr:end] + bytes(max(0, addr + WB - len(self.image)))
        return int.from_bytes(raw, "little")

    # ---- VSRAM words

    def word(self, bank: int, w: int) -> int:
        out = 0
        base = w * NE
        for j in range(NE):
            out |= int(self.vsram[bank, base + j] & MASK32) << (32 * j)
        return out

    def write_word(self, bank: int, w: int, strb: int, data: int) -> None:
        for j in range(NE):
            if (strb >> j) & 1:
                v = (data >> (32 * j)) & MASK32
                self.vsram[bank, w * NE + j] = qn.to_signed(v, 32)

    # ---- one cycle

    async def step(self) -> None:
        dut = self.dut
        v = qc_stream.value

        if int(dut.done.value):
            self.done_cycles.append(self.cycle)
        if int(dut.busy.value):
            self.busy_cycles += 1
        self.sat += v(dut.sat_inc)
        self.err_shift += v(dut.err_shift_inc)
        self.err_bounds += v(dut.err_bounds_inc)

        row = v(dut.cur_row)
        ena, addr_a = int(dut.vsa_en.value), v(dut.vsa_addr)
        enb, addr_b = int(dut.vsb_en.value), v(dut.vsb_addr)
        we, sel = v(dut.vsb_we), int(dut.vsb_sel_dst.value)
        bank_a = self.src_row + row
        bank_b = (self.dst_row if sel else self.src_row) + row
        assert not (enb and we), f"cycle {self.cycle}: port B read and write together"
        assert not (we and self._read_b_prev), (
            f"cycle {self.cycle}: a port B write in the cycle after a port B read, so the "
            "crossbar would change bank while the read returns"
        )
        self._read_b_prev = bool(enb)
        if ena:
            assert addr_a < VSRAM_WORDS and bank_a < B_MAX, f"cycle {self.cycle}: bad port A read"
        if we:
            assert addr_b < VSRAM_WORDS and bank_b < B_MAX, f"cycle {self.cycle}: bad port B write"
            assert sel == self.sel_write, (
                f"cycle {self.cycle}: vsb_sel_dst {sel} on a write, expected {self.sel_write}"
            )
            assert not (ena and bank_a == bank_b and addr_a == addr_b), (
                f"cycle {self.cycle}: word {addr_a} read on port A while port B writes it"
            )
            self.write_word(bank_b, addr_b, we, v(dut.vsb_wdata))
            self.writes.append((bank_b, addr_b))
        if enb:
            assert sel == self.sel_read, (
                f"cycle {self.cycle}: vsb_sel_dst {sel} on a read, expected {self.sel_read}"
            )

        if int(dut.sreg_wr_en.value):
            idx = v(dut.sreg_wr_idx)
            bank = self.dst_row + v(dut.sreg_wr_row)
            data = v(dut.sreg_wr_data)
            self.sreg_writes.append((bank, idx, data))
            if idx >= isa.SREG_COUNT:
                self.sreg_err += 1
            else:
                assert bank < B_MAX, f"cycle {self.cycle}: SREG write to bank {bank}"
                self.sreg[bank][idx] = data

        # QMEM read port: the transfer sampled last cycle, then this cycle's grant
        if self._req_prev is not None:
            pv, paddr, plen, prdy = self._req_prev
            if pv and prdy:
                self.requests.append((paddr, plen))
                for i in range(plen):
                    t = max(self.cycle - 1 + self.latency + i, self._last_deliver + 1)
                    self._last_deliver = t
                    self._pending.append((t, self.beat(paddr + i * WB), int(i == plen - 1)))
                self._pending.sort(key=lambda b: b[0])
        rv = int(dut.v_req_valid.value)
        payload = (v(dut.v_req_addr), v(dut.v_req_len), v(dut.v_req_tag)) if rv else None
        if rv:
            assert payload[2] == 3, f"cycle {self.cycle}: read tag {payload[2]} is not TAG_VPU"
            assert 1 <= payload[1] <= 255, f"cycle {self.cycle}: burst length {payload[1]}"
            if self._held is not None:
                assert payload == self._held, f"cycle {self.cycle}: the request changed while valid"
        else:
            assert self._held is None, f"cycle {self.cycle}: v_req_valid dropped before the grant"
        rdy = int(bool(self.req_ready(self.cycle)))
        dut.v_req_ready.value = rdy
        self._held = None if (not rv or rdy) else payload
        self._req_prev = (rv, payload[0] if rv else 0, payload[1] if rv else 0, rdy)

        if self._pending and self._pending[0][0] <= self.cycle:
            _, data, last = self._pending.pop(0)
            dut.rdv_valid.value = 1
            dut.rd_data.value = data
            dut.rd_data_last.value = last
            self.beats_returned += 1
        else:
            dut.rdv_valid.value = 0
            dut.rd_data_last.value = 0

        # VSRAM read data: the word of the read issued in the previous cycle
        if self._rd_a is not None:
            dut.vsa_rdata.value = self._data_a
        if self._rd_b is not None:
            dut.vsb_rdata.value = self._data_b
        self._rd_a = (bank_a, addr_a) if ena else None
        self._rd_b = (bank_b, addr_b) if enb else None
        self._data_a = self.word(bank_a, addr_a) if ena else 0
        self._data_b = self.word(bank_b, addr_b) if enb else 0

        await FallingEdge(dut.clk)
        self.cycle += 1

    async def issue(self, d: Descriptor, row_en: int, sreg_u32: list[int]) -> None:
        """Present the decoded bundle and pulse cmd_valid_vpu for one cycle."""
        dut = self.dut
        flags = VquantFlag(d.flags) if d.opcode == Opcode.VQUANT else VquantFlag(0)
        rows = 0
        for r in range(B_MAX):
            if (d.row_mask >> r) & 1 and (row_en >> r) & 1:
                rows |= 1 << r
        sqrt = SFloat(0, 0)
        if d.opcode == Opcode.VRMSNORM:
            sqrt = isa.sfloat_from_imm(d.addr_m & 0xFFFFFF)
        dut.cmd_op.value = int(d.opcode)
        dut.cmd_vq_w8.value = int(bool(flags & VquantFlag.W8))
        dut.cmd_vq_use_tracked.value = int(bool(flags & VquantFlag.USE_TRACKED))
        dut.cmd_vq_group.value = int(bool(flags & VquantFlag.GROUP))
        dut.cmd_vq_scale_mul.value = int(bool(flags & VquantFlag.SCALE_MUL))
        dut.cmd_track_absmax.value = int(d.track_absmax)
        dut.cmd_addr_a.value = d.addr_a
        dut.cmd_n.value = d.n
        dut.cmd_len.value = softmax_len(d, self.pos)[0] if d.opcode == Opcode.VSOFTMAX else 0
        dut.cmd_pos.value = self.pos
        dut.cmd_vs_src.value = d.vs_src
        dut.cmd_vs_dst.value = d.vs_dst
        dut.cmd_vs_aux.value = d.vs_aux
        dut.cmd_sreg_dst.value = d.sreg_dst
        dut.cmd_sh0.value = d.sh0
        dut.cmd_sh1.value = d.sh1 & 0xFF
        dut.cmd_imm32.value = d.imm32
        dut.cmd_sqrt_m.value = sqrt.m
        dut.cmd_sqrt_e.value = sqrt.e & 0xFF
        dut.cmd_rows.value = rows
        dut.cmd_sreg_u32.value = sum((w & MASK32) << (32 * r) for r, w in enumerate(sreg_u32))
        dut.cmd_valid_vpu.value = 1
        self.src_row = d.src_row
        self.dst_row = d.dst_row
        # VROPE rewrites its own source, so both its port-B reads and its
        # port-B writes address bank src_row + row; the softmax weight pass
        # reads and writes the destination bank.
        self.sel_read = 1 if d.opcode == Opcode.VSOFTMAX else 0
        self.sel_write = 0 if d.opcode == Opcode.VROPE else 1
        await self.step()
        dut.cmd_valid_vpu.value = 0

    async def run(self, d: Descriptor, row_en: int, sreg_u32: list[int], timeout: int) -> int:
        """Issue ``d`` and step until done; returns the descriptor's cycle count."""
        first = self.cycle
        seen = len(self.done_cycles)
        await self.issue(d, row_en, sreg_u32)
        for _ in range(timeout):
            await self.step()
            if len(self.done_cycles) > seen:
                break
        assert len(self.done_cycles) == seen + 1, (
            f"no done pulse within {timeout} cycles for {describe(d)}"
        )
        for _ in range(4):
            await self.step()
        return self.cycle - first


# --------------------------------------------------------------------------- the reference


def describe(d: Descriptor) -> str:
    return (
        f"{d.opcode.name}(n={d.n}, src={d.vs_src}, dst={d.vs_dst}, aux={d.vs_aux}, "
        f"flags={d.flags}, sh0={d.sh0}, sh1={d.sh1}, rows={d.row_mask}, "
        f"src_row={d.src_row}, dst_row={d.dst_row})"
    )


def participants(d: Descriptor, row_en: int) -> list[tuple[int, int, int]]:
    return [
        (r, d.src_row + r, d.dst_row + r)
        for r in range(B_MAX)
        if (d.row_mask >> r) & 1 and (row_en >> r) & 1
    ]


def sreg_index_errors(d: Descriptor, rows: list) -> tuple[int, int]:
    """``(writes, reads)`` of this descriptor that address an SREG index at or above 32."""
    writes = reads = 0
    flags = VquantFlag(d.flags) if d.opcode == Opcode.VQUANT else VquantFlag(0)
    for _ in rows:
        if d.opcode == Opcode.VSOFTMAX:
            # the output scale is written whether or not track_absmax is set
            writes += int(d.sreg_dst >= isa.SREG_COUNT)
        elif d.opcode == Opcode.VQUANT:
            if flags & VquantFlag.GROUP:
                groups = -(-d.n // d.vs_aux) if d.vs_aux else 1
                writes += sum(1 for g in range(groups) if d.sreg_dst + g >= isa.SREG_COUNT)
            else:
                if (flags & VquantFlag.USE_TRACKED) and d.sreg_src >= isa.SREG_COUNT:
                    reads += 1
                writes += int(d.sreg_dst >= isa.SREG_COUNT)
        elif d.track_absmax and d.opcode in (Opcode.VRMSNORM, Opcode.VSILUMUL):
            writes += int(d.sreg_dst >= isa.SREG_COUNT)
    return writes, reads


def reference(env: Env, d: Descriptor, row_en: int, sreg_pre: dict) -> isa_sim.Machine:
    """The same descriptor on isa_sim over the same image, VSRAM and SREG state."""
    m = isa_sim.Machine(env.image, wb=WB, b_max=B_MAX, vsram_words=VSRAM_WORDS, tables=qn.tables())
    m.csr["ROW_EN"] = row_en
    m.csr["POS"] = env.pos
    m.vsram[:] = env.vsram
    for (bank, idx), val in sreg_pre.items():
        m.sreg[bank][idx] = val
    isa_sim.execute(m, d)
    return m


def rmsnorm_xhat(x: np.ndarray, frac_x: int, sqrt_d: SFloat, eps_c: int) -> np.ndarray:
    """``xhat = round_shift(x * Rc_m, S1)``, the intermediate of :func:`numerics.rmsnorm`.

    Exact, the way the reference carries it: the hardware forms the same value
    in ``qcore_vpu_lane``, where it is an int32 and a saturation of it is a
    ``SAT_VPU`` event.
    """
    amax = numerics.absmax(x)
    sh = max(0, numerics.bitlen(amax) - 15)
    ss = int(np.sum((x >> sh) * (x >> sh))) + (eps_c >> (2 * sh))
    if ss <= 0:
        return np.zeros_like(x)
    length = numerics.bitlen(ss)
    e2 = length - 1 if (length - 1) % 2 == 0 else length - 2
    m_q16 = ss >> (e2 - 16) if e2 >= 16 else ss << (16 - e2)
    r = numerics.rsqrt_q15(m_q16, TABLES)
    rc = numerics.sfloat_mul(numerics.sfloat_from_int(r, -15), sqrt_d)
    s1 = -(rc.e + frac_x - sh - e2 // 2)
    return numerics.round_shift(x * rc.m, min(max(s1, 0), numerics.SHIFT_MAX))


def xhat_outside(xhat: np.ndarray) -> int:
    """Elements of an exact ``xhat`` that the lane's int32 result cannot hold."""
    return int(np.count_nonzero((xhat > (1 << 31) - 1) | (xhat < -(1 << 31))))


def check_xhat_domain(env: Env, d: Descriptor, rows: list) -> None:
    """VRMSNORM's intermediate is a lane output and so an int32: assert the stimulus stays there.

    ``numerics.rmsnorm`` carries ``xhat`` in a Python int and bounds it by
    ``sqrt(d) * 2**FRAC_X * (1 + 2**-13)``; ``quantize.check_rmsnorm_domain``
    refuses a model whose bound leaves int32, so the two models are identical
    for every program.  This asserts the same of the stimulus of the random
    sweeps rather than tolerating a mismatch; ``test_vrmsnorm_xhat_boundary``
    drives the edge itself from both sides.
    """
    for _, src, _ in rows:
        x = env.vsram[src, d.vs_src : d.vs_src + d.n]
        x = np.concatenate([x, np.zeros(d.n - x.size, dtype=np.int64)])
        xhat = rmsnorm_xhat(x, d.sh0, isa.sfloat_from_imm(d.addr_m & 0xFFFFFF), d.imm32)
        assert xhat_outside(xhat) == 0, (
            f"the stimulus left the documented xhat domain ({int(np.max(np.abs(xhat)))} "
            f"needs more than int32): {describe(d)}"
        )


def first_diff(exp: np.ndarray, got: np.ndarray) -> int | None:
    idx = np.flatnonzero(exp != got)
    return int(idx[0]) if idx.size else None


async def run_one(
    env: Env,
    d: Descriptor,
    row_en: int = 1,
    *,
    sreg_pre: dict | None = None,
    timeout: int = 400000,
) -> int:
    """Run one descriptor on the DUT and on isa_sim and compare everything they touch."""
    sreg_pre = dict(sreg_pre or {})
    env.sreg = [[0] * isa.SREG_COUNT for _ in range(B_MAX)]
    assert compiler.vector_overlap(d) is None, (
        f"the stimulus breaks the V-op range rule of docs/ISA.md: {compiler.vector_overlap(d)}"
    )
    rows = participants(d, row_en)
    if d.opcode == Opcode.VRMSNORM:
        check_xhat_domain(env, d, rows)
    m = reference(env, d, row_en, sreg_pre)
    for (bank, idx), val in sreg_pre.items():
        if isinstance(val, int) and idx < isa.SREG_COUNT:
            env.sreg[bank][idx] = val & MASK32

    before = env.vsram.copy()
    sat0, esh0, ebd0, serr0 = env.sat, env.err_shift, env.err_bounds, env.sreg_err
    env.sreg_writes.clear()
    env.writes.clear()
    idx = d.sreg_src % isa.SREG_COUNT
    u32 = [env.sreg[min(d.src_row + r, B_MAX - 1)][idx] for r in range(B_MAX)]
    if d.sreg_src >= isa.SREG_COUNT:
        u32 = [0] * B_MAX
    cycles = await env.run(d, row_en, u32, timeout)
    TOTALS["descriptors"] += 1
    TOTALS["rows"] += len(rows)
    TOTALS["elements"] += d.n * len(rows)

    # VSRAM, element by element
    for bank in range(B_MAX):
        bad = first_diff(m.vsram[bank], env.vsram[bank])
        assert bad is None, (
            f"{describe(d)}: VSRAM bank {bank} differs first at element {bad} "
            f"(expected {m.vsram[bank][bad]}, got {env.vsram[bank][bad]}, "
            f"was {before[bank][bad]})"
        )
    # SREG
    w_err, r_err = sreg_index_errors(d, rows)
    for bank in range(B_MAX):
        for idx in range(isa.SREG_COUNT):
            want = m.sreg[bank][idx]
            want_w = qn.sreg_pack(want) if isinstance(want, SFloat) else int(want)
            assert env.sreg[bank][idx] == want_w, (
                f"{describe(d)}: SREG[{bank}][{idx}] = {env.sreg[bank][idx]:#x}, "
                f"expected {want_w:#x}"
            )
    # counters
    assert env.sat - sat0 == m.stats_vpu.sat, (
        f"{describe(d)}: SAT_VPU {env.sat - sat0} != {m.stats_vpu.sat}"
    )
    assert env.err_shift - esh0 == m.stats_vpu.err_shift, (
        f"{describe(d)}: ERR_SHIFT {env.err_shift - esh0} != {m.stats_vpu.err_shift}"
    )
    # the length clamp of a VSOFTMAX is the dispatcher's ERR_BOUNDS event, not
    # the unit's: the unit is handed cmd_len already inside [1, n]
    clamped = 0
    if d.opcode == Opcode.VSOFTMAX:
        clamped = softmax_len(d, env.pos)[1] * len(rows)
    want_bounds = m.err_bounds - w_err - r_err - clamped
    assert env.err_bounds - ebd0 == want_bounds, (
        f"{describe(d)}: ERR_BOUNDS {env.err_bounds - ebd0} != {want_bounds}"
    )
    assert env.sreg_err - serr0 == w_err, (
        f"{describe(d)}: SREG index errors {env.sreg_err - serr0} != {w_err}"
    )
    if d.opcode in (Opcode.VQUANT, Opcode.VSOFTMAX):
        assert int(env.dut.clip_count.value) == m.stats_vpu.clip, (
            f"{describe(d)}: clips {int(env.dut.clip_count.value)} != {m.stats_vpu.clip}"
        )
    return cycles


# --------------------------------------------------------------------------- stimulus


def rand_vec(rng: random.Random, n: int, bits: int) -> np.ndarray:
    """``n`` int32 values whose magnitudes reach about ``bits`` bits."""
    hi = 1 << max(1, min(bits, 31))
    return np.array([rng.randrange(-hi, hi) for _ in range(n)], dtype=np.int64)


def fill(env: Env, rng: random.Random, bank: int, start: int, values: np.ndarray) -> None:
    n = min(values.size, ELEMS - start)
    if n > 0:
        env.vsram[bank, start : start + n] = values[:n]


def scatter(env: Env, rng: random.Random) -> None:
    """A distinctive background so a write the hardware should not make cannot hide."""
    for bank in range(B_MAX):
        env.vsram[bank] = np.array(
            [rng.randrange(-(1 << 31), 1 << 31) for _ in range(64)], dtype=np.int64
        ).repeat(ELEMS // 64)


def gamma_bytes(g: np.ndarray) -> bytes:
    return g.astype("<i2").tobytes()


def const_bytes(c: np.ndarray) -> bytes:
    return c.astype("<i4").tobytes()


def image_with(row: bytes, addr: int) -> bytes:
    buf = bytearray(bytes(range(256)) * (IMAGE_BYTES // 256))
    buf[addr : addr + len(row)] = row
    return bytes(buf)


def sqrt_d_of(n: int) -> SFloat:
    return numerics.sfloat_from_float(math.sqrt(max(n, 1)))


def eps_of(n: int, frac_x: int) -> int:
    return numerics.eps_const(1e-6, max(n, 1), frac_x)


def offsets(rng: random.Random, over: str = "") -> tuple[int, int, int]:
    """Three element ranges, half of them word-aligned and half not.

    ``over`` picks the half of the range rule the descriptor exercises: empty
    for three disjoint ranges, ``"src"`` or ``"aux"`` for the in-place form, a
    destination exactly over that source, which is what every compiled VQUANT
    and VSUBC does.
    """

    def one(base: int) -> int:
        return base + (rng.randrange(NE) if rng.randrange(2) else 0)

    src, dst, aux = one(64), one(4096), one(8192)
    if over == "src":
        dst = src
    elif over == "aux":
        dst = aux
    return src, dst, aux


LENGTHS = (1, 2, 3, 7, 8, 9, 15, 16, 17, 24, 33, 64, 65, 100, 257, 1000)


def in_place(case: int) -> str:
    """Every fourth randomised case writes its destination over its source."""
    return "src" if case % 4 == 1 else ""


async def setup(dut, seed: int, *, latency: int = 32, req_ready=None) -> tuple[Env, random.Random]:
    rng = random.Random(seed)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for sig in (
        dut.cmd_valid_vpu,
        dut.cmd_op,
        dut.cmd_vq_w8,
        dut.cmd_vq_use_tracked,
        dut.cmd_vq_group,
        dut.cmd_vq_scale_mul,
        dut.cmd_track_absmax,
        dut.cmd_addr_a,
        dut.cmd_n,
        dut.cmd_len,
        dut.cmd_pos,
        dut.cmd_vs_src,
        dut.cmd_vs_dst,
        dut.cmd_vs_aux,
        dut.cmd_sreg_dst,
        dut.cmd_sh0,
        dut.cmd_sh1,
        dut.cmd_imm32,
        dut.cmd_sqrt_m,
        dut.cmd_sqrt_e,
        dut.cmd_rows,
        dut.cmd_sreg_u32,
        dut.v_req_ready,
        dut.rdv_valid,
        dut.rd_data,
        dut.rd_data_last,
        dut.vsa_rdata,
        dut.vsb_rdata,
    ):
        sig.value = 0
    await qc_stream.reset(dut, dut.rst)
    env = Env(dut, bytes(IMAGE_BYTES), latency=latency, req_ready=req_ready)
    return env, rng


def fill_sources(
    env: Env,
    rng: random.Random,
    d: Descriptor,
    row_en: int,
    *,
    bits: int | None = None,
    values: np.ndarray | None = None,
    aux: np.ndarray | None = None,
) -> None:
    """Random (or given) source and auxiliary vectors in every participating bank."""
    for _, src, _ in participants(d, row_en):
        x = values if values is not None else rand_vec(rng, d.n, bits or rng.randrange(2, 32))
        fill(env, rng, src, d.vs_src, x)
        if d.opcode == Opcode.VSILUMUL:
            u = aux if aux is not None else rand_vec(rng, d.n, bits or rng.randrange(2, 32))
            fill(env, rng, src, d.vs_aux, u)


def rows_of(rng: random.Random) -> tuple[int, int, int, int]:
    """``(row_mask, ROW_EN, src_row, dst_row)`` inside the configuration's row count."""
    if B_MAX == 1:
        return 1, 1, 0, 0
    pick = rng.randrange(4)
    if pick == 0:
        return 3, 3, 0, 0
    if pick == 1:
        return 1, 1, 1, 1
    if pick == 2:
        return 1, 1, 0, 1
    return 2, 3, 0, 0


def subc_descriptor(
    rng: random.Random, n: int, *, bits: int | None = None, over: str = ""
) -> tuple[Descriptor, np.ndarray, int]:
    src, dst, _ = offsets(rng, over)
    mask, _, sr, dr = rows_of(rng)
    addr = ROW_BASE + 4 * rng.randrange(4)
    c = rand_vec(rng, n, bits if bits is not None else rng.randrange(2, 32))
    d = isa.vsubc(vs_src=src, vs_dst=dst, n=n, addr_a=addr, src_row=sr, dst_row=dr, row_mask=mask)
    return d, c, addr


# --------------------------------------------------------------------------- VSUBC


@cocotb.test()
async def test_vsubc(dut) -> None:
    """Random VSUBC descriptors over every length class, offset and row layout."""
    env, rng = await setup(dut, 11)
    saturating = 0
    for case in range(CASES):
        n = LENGTHS[case % len(LENGTHS)] if case < len(LENGTHS) else rng.choice(LENGTHS)
        big = case % 3 == 0
        d, c, addr = subc_descriptor(rng, n, bits=31 if big else None, over=in_place(case))
        row_en = 3 if B_MAX > 1 else 1
        env.image = image_with(const_bytes(c), addr)
        scatter(env, rng)
        fill_sources(env, rng, d, row_en, bits=31 if big else rng.randrange(2, 32))
        before = env.sat
        await run_one(env, d, row_en)
        if env.sat > before:
            saturating += 1
    assert saturating >= 3, f"only {saturating} of {CASES} VSUBC cases saturated"


# --------------------------------------------------------------------------- VQUANT


@cocotb.test()
async def test_vquant(dut) -> None:
    """Random VQUANT descriptors over both widths and every flag combination."""
    env, rng = await setup(dut, 22)
    seen_flags: set[int] = set()
    clipped = 0
    for case in range(CASES * 2):
        n = LENGTHS[case % len(LENGTHS)] if case < len(LENGTHS) else rng.choice(LENGTHS)
        src, dst, _ = offsets(rng, in_place(case))
        mask, _, sr, dr = rows_of(rng)
        row_en = 3 if B_MAX > 1 else 1
        width = 8 if rng.randrange(2) else 16
        frac_in = rng.randrange(0, 31)
        use_tracked = bool(rng.randrange(2))
        scale_mul = (
            numerics.sfloat_from_float(rng.uniform(0.01, 100.0)) if rng.randrange(2) else None
        )
        d = isa.vquant(
            vs_src=src,
            vs_dst=dst,
            n=n,
            width=width,
            frac_in=frac_in,
            sreg_dst=rng.randrange(0, 30),
            use_tracked=use_tracked,
            sreg_src=rng.randrange(0, 30),
            scale_mul=scale_mul,
            src_row=sr,
            dst_row=dr,
            row_mask=mask,
        )
        seen_flags.add(d.flags)
        scatter(env, rng)
        bits = 31 if case % 5 == 0 else rng.randrange(1, 32)
        fill_sources(env, rng, d, row_en, bits=bits)
        pre = {}
        if use_tracked:
            for _, srcb, _ in participants(d, row_en):
                x = env.vsram[srcb, d.vs_src : d.vs_src + d.n]
                amax = numerics.absmax(x)
                # the tracked absmax is what a previous descriptor left: usually
                # the exact absmax, sometimes a larger one
                pre[(srcb, d.sreg_src)] = amax if rng.randrange(4) else min(amax * 2 + 1, MASK32)
        before = int(dut.clip_count.value)
        await run_one(env, d, row_en, sreg_pre=pre)
        if int(dut.clip_count.value) > 0:
            clipped += 1
        assert before >= 0
    assert len(seen_flags) >= 8, f"only {len(seen_flags)} VQUANT flag combinations were driven"
    assert clipped >= 1, "no VQUANT descriptor reached the reciprocal clip"


@cocotb.test()
async def test_vquant_group(dut) -> None:
    """GROUP quantization: one scale per vs_aux elements into consecutive SREGs.

    The sweep has the same in-place share as the others, because in place is the
    form every compiled program uses: the compiler quantizes q, K and V per head
    over their own ranges.
    """
    env, rng = await setup(dut, 33)
    over_src = 0
    for case in range(CASES // 2):
        group = rng.choice((1, 2, 3, 4, 7, 8, 16))
        groups = rng.randrange(1, 6)
        n = group * groups
        src, dst, _ = offsets(rng, in_place(case))
        over_src += int(src == dst)
        width = 8 if rng.randrange(2) else 16
        scale_mul = numerics.sfloat_from_float(rng.uniform(0.1, 10.0)) if rng.randrange(2) else None
        d = isa.vquant(
            vs_src=src,
            vs_dst=dst,
            n=n,
            width=width,
            frac_in=rng.randrange(0, 31),
            sreg_dst=rng.randrange(0, isa.SREG_COUNT - groups),
            group=group,
            scale_mul=scale_mul,
        )
        scatter(env, rng)
        x = rand_vec(rng, n, rng.randrange(2, 32))
        if case % 3 == 0:  # a zero group among non-zero ones
            x[: min(group, n)] = 0
        fill_sources(env, rng, d, 1, values=x)
        await run_one(env, d, 1)

        # docs/RTL.md 4: USE_TRACKED together with GROUP ignores USE_TRACKED, so
        # the pair has to leave exactly what GROUP alone leaves. isa_sim rejects
        # the combination, so the reference here is the descriptor without it.
        if case % 4 == 0:
            grouped = env.vsram.copy()
            scales = [env.sreg[0][d.sreg_dst + g] for g in range(groups)]
            both = Descriptor(
                opcode=Opcode.VQUANT,
                row_mask=1,
                flags=d.flags | int(VquantFlag.USE_TRACKED),
                n=n,
                vs_src=d.vs_src,
                vs_dst=d.vs_dst,
                vs_aux=d.vs_aux,
                sreg_src=20,
                sreg_dst=d.sreg_dst,
                sh0=d.sh0,
                imm32=d.imm32,
            )
            env.vsram[:] = 0
            fill_sources(env, rng, both, 1, values=x)
            await env.run(both, 1, [1 << 30] * B_MAX, 400000)
            assert np.array_equal(
                env.vsram[0, both.vs_dst : both.vs_dst + n],
                grouped[0, d.vs_dst : d.vs_dst + n],
            ), "GROUP with USE_TRACKED did not ignore USE_TRACKED"
            for g in range(groups):
                assert env.sreg[0][d.sreg_dst + g] == scales[g], (
                    f"GROUP with USE_TRACKED wrote a different scale for group {g}"
                )

    assert over_src >= CASES // 16, f"only {over_src} of the GROUP cases ran in place"

    # The compiled form, directed: in place at a base that is not word-aligned,
    # with a group length that does not divide a VSRAM word, so a group starts
    # mid-word and the write of one group lands while the next group's operands
    # are being read out of the same words.
    for group, groups, base in ((3, 5, 65), (5, 4, 4097), (6, 3, 71)):
        assert base % NE and NE % group, f"({group}, {base}) is not the unaligned case"
        n = group * groups
        d = isa.vquant(
            vs_src=base,
            vs_dst=base,
            n=n,
            width=8,
            frac_in=16,
            sreg_dst=2,
            group=group,
        )
        scatter(env, rng)
        fill_sources(env, rng, d, 1, values=rand_vec(rng, n, 30))
        await run_one(env, d, 1)


@cocotb.test()
async def test_vquant_group_index_saturates(dut) -> None:
    """More groups than the register file holds: the group index saturates, it does not wrap.

    ``sreg_dst + g`` is an eight-bit SREG index, so group 256 would come back
    around to ``SREG[sreg_dst]`` and overwrite a scale the same descriptor
    already wrote.  The rule is that a group form writes at most
    ``isa.SREG_COUNT`` scales (``compiler.sreg_range``, and ``isa.vquant``
    refuses to build one, so this descriptor is built field by field); the
    hardware holds to it by saturating the index, which puts every group past
    the file at or above it -- dropped and counted, exactly as
    ``sw/quettos/isa_sim.py`` drops it.
    """
    env, rng = await setup(dut, 257)
    n = 257
    d = Descriptor(
        opcode=Opcode.VQUANT,
        row_mask=1,
        flags=int(VquantFlag.GROUP) | int(VquantFlag.W8),
        n=n,
        vs_src=64,
        vs_dst=64,
        vs_aux=1,
        sreg_dst=0,
        sh0=16,
    )
    assert compiler.sreg_range(d) is not None, "the case has to be one the compiler refuses"
    scatter(env, rng)
    fill_sources(env, rng, d, 1, values=rand_vec(rng, n, 28))
    await run_one(env, d, 1)
    assert [i for _, i, _ in env.sreg_writes if i >= isa.SREG_COUNT], (
        "no group addressed an SREG index above the file"
    )


@cocotb.test()
async def test_vquant_partial_group(dut) -> None:
    """A GROUP length that does not divide n leaves the last group short.

    ``sw/quettos/isa_sim.py`` rejects that descriptor, so the reference here is
    ``numerics.quant`` applied to the slices the hardware is specified to form:
    consecutive groups of ``vs_aux`` elements, the last one holding what is left.
    """
    env, rng = await setup(dut, 666)
    for group, n in ((8, 20), (4, 6), (16, 17), (3, 8)):
        width = 8 if rng.randrange(2) else 16
        frac_in = rng.randrange(0, 31)
        d = Descriptor(
            opcode=Opcode.VQUANT,
            row_mask=1,
            flags=int(VquantFlag.GROUP) | (int(VquantFlag.W8) if width == 8 else 0),
            n=n,
            vs_src=64,
            vs_dst=4096,
            vs_aux=group,
            sreg_dst=1,
            sh0=frac_in,
        )
        scatter(env, rng)
        x = rand_vec(rng, n, rng.randrange(4, 32))
        fill_sources(env, rng, d, 1, values=x)
        env.sreg = [[0] * isa.SREG_COUNT for _ in range(B_MAX)]
        await env.run(d, 1, [0] * B_MAX, 400000)
        for g, start in enumerate(range(0, n, group)):
            sl = x[start : min(start + group, n)]
            q, sx = numerics.quant(sl, width, frac_in, TABLES)
            got = env.vsram[0, 4096 + start : 4096 + start + sl.size]
            bad = first_diff(q, got)
            assert bad is None, (
                f"partial GROUP (n={n}, group={group}): group {g} differs first at "
                f"element {start + bad} (expected {q[bad]}, got {got[bad]})"
            )
            assert env.sreg[0][1 + g] == qn.sreg_pack(sx), (
                f"partial GROUP (n={n}, group={group}): SREG[{1 + g}] is not group {g}'s scale"
            )


# --------------------------------------------------------------------------- VRMSNORM


def rmsnorm_descriptor(rng: random.Random, n: int, *, frac_x: int | None = None, over: str = ""):
    src, dst, _ = offsets(rng, over)
    mask, _, sr, dr = rows_of(rng)
    addr = ROW_BASE + 2 * rng.randrange(4)
    frac_x = rng.randrange(12, 19) if frac_x is None else frac_x
    gq, ge = numerics.quantize_gamma(
        np.array([rng.uniform(-3.0, 3.0) for _ in range(n)], dtype=np.float64)
    )
    d = isa.vrmsnorm(
        vs_src=src,
        vs_dst=dst,
        n=n,
        addr_a=addr,
        eps_c=eps_of(n, frac_x),
        frac_x=frac_x,
        g=-ge,
        sqrt_d=sqrt_d_of(n),
        sreg_dst=rng.randrange(0, isa.SREG_COUNT),
        track_absmax=bool(rng.randrange(2)),
        src_row=sr,
        dst_row=dr,
        row_mask=mask,
    )
    return d, gq, addr


@cocotb.test()
async def test_vrmsnorm(dut) -> None:
    """Random VRMSNORM descriptors: the absmax pass, the sum of squares, the scale and gamma."""
    env, rng = await setup(dut, 44)
    row_en = 3 if B_MAX > 1 else 1
    tracked = 0
    for case in range(CASES):
        n = LENGTHS[case % len(LENGTHS)] if case < len(LENGTHS) else rng.choice(LENGTHS)
        d, gq, addr = rmsnorm_descriptor(rng, n, over=in_place(case))
        env.image = image_with(gamma_bytes(gq), addr)
        scatter(env, rng)
        bits = 31 if case % 4 == 0 else rng.randrange(4, 32)
        fill_sources(env, rng, d, row_en, bits=bits)
        await run_one(env, d, row_en)
        tracked += int(d.track_absmax)
    assert tracked >= 5, "track_absmax was hardly exercised"


def edge_sqrt_d(x: np.ndarray, frac_x: int, eps_c: int) -> SFloat:
    """The largest ``sqrt(d)`` constant whose intermediate still fits the lane's int32.

    ``xhat`` scales with the constant, so one scan of the mantissa around the
    estimate finds it, and the mantissa step bounds how far short of the edge it
    lands: ``2**31 / 2**16``.
    """
    lim = (1 << 31) - 1
    got = int(np.max(np.abs(rmsnorm_xhat(x, frac_x, SFloat(1 << 15, 0), eps_c))))
    assert got > 0, "the probe vector produced a zero intermediate"
    start = numerics.sfloat_from_float((1 << 15) * lim / got)
    best = None
    for m in range(max(1 << 15, start.m - 64), min(1 << 16, start.m + 65)):
        cand = SFloat(m, start.e)
        if xhat_outside(rmsnorm_xhat(x, frac_x, cand, eps_c)) == 0:
            best = cand
    assert best is not None, "the mantissa scan found no constant inside the domain"
    reached = int(np.max(np.abs(rmsnorm_xhat(x, frac_x, best, eps_c))))
    assert reached > lim - (1 << 16), f"the scan stopped at {reached}, short of the edge"
    return best


@cocotb.test()
async def test_vrmsnorm_xhat_boundary(dut) -> None:
    """VRMSNORM at the int32 edge of ``xhat``, from both sides of it.

    Inside the edge the unit reproduces ``numerics.rmsnorm`` element for element
    and counter for counter, with the intermediate within one mantissa step of
    the largest int32 holds.  Past it the lane saturates the intermediate, and
    every element that does so is a ``SAT_VPU`` event: the difference is a
    counter the run reports rather than a value that changes quietly.
    ``quantize.check_rmsnorm_domain`` keeps a compiled model on the inside.
    """
    env, rng = await setup(dut, 121)
    lim, neg = (1 << 31) - 1, -(1 << 31)
    n, frac_x, eps_c = 16, 16, 0
    probes = (
        np.array([lim if j % 2 == 0 else neg for j in range(n)], dtype=np.int64),
        np.array([lim] + [0] * (n - 1), dtype=np.int64),
        np.array([neg] + [0] * (n - 1), dtype=np.int64),
    )

    # inside: the intermediate sits at the top of the domain, and every value and
    # every counter still equals the reference
    gq = np.array([1 - 2 * (j % 2) for j in range(n)], dtype=np.int64)
    env.image = image_with(gamma_bytes(gq), ROW_BASE)
    for probe in probes:
        inside = edge_sqrt_d(probe, frac_x, eps_c)
        for g_shift in (0, 8):
            d = isa.vrmsnorm(
                vs_src=64,
                vs_dst=4096,
                n=n,
                addr_a=ROW_BASE,
                eps_c=eps_c,
                frac_x=frac_x,
                g=g_shift,
                sqrt_d=inside,
                sreg_dst=6,
                track_absmax=True,
            )
            scatter(env, rng)
            fill_sources(env, rng, d, 1, values=probe)
            await run_one(env, d, 1)

    # outside: twice that constant leaves every intermediate past int32. G = 8 keeps
    # the second product well inside it, which is what made the old difference
    # invisible -- the result changed and no counter moved.
    probe = probes[0]
    inside = edge_sqrt_d(probe, frac_x, eps_c)
    outside = SFloat(inside.m, inside.e + 1)
    gq = np.ones(n, dtype=np.int64)
    env.image = image_with(gamma_bytes(gq), ROW_BASE)
    d = isa.vrmsnorm(
        vs_src=64,
        vs_dst=4096,
        n=n,
        addr_a=ROW_BASE,
        eps_c=eps_c,
        frac_x=frac_x,
        g=8,
        sqrt_d=outside,
        sreg_dst=6,
        track_absmax=True,
    )
    exact = rmsnorm_xhat(probe, frac_x, outside, eps_c)
    assert xhat_outside(exact) == n, f"only {xhat_outside(exact)} of {n} elements left int32"
    want = numerics.round_shift(np.clip(exact, neg, lim) * gq, 8)
    assert xhat_outside(want) == 0, "the second product has to stay inside int32"
    assert not np.array_equal(want, numerics.round_shift(exact * gq, 8)), (
        "the outside case has to differ from the exact intermediate, or it proves nothing"
    )
    scatter(env, rng)
    fill_sources(env, rng, d, 1, values=probe)
    before = env.sat
    await env.run(d, 1, [0] * B_MAX, 400000)
    got = env.vsram[0, d.vs_dst : d.vs_dst + n]
    bad = first_diff(want, got)
    assert bad is None, (
        f"past the xhat edge the unit is the saturated model: element {bad} is "
        f"{got[bad] if bad is not None else 0}, expected {want[bad] if bad is not None else 0}"
    )
    assert env.sat - before == n, (
        f"{env.sat - before} of {n} saturated intermediates reached SAT_VPU"
    )


# --------------------------------------------------------------------------- VSILUMUL


@cocotb.test()
async def test_vsilumul(dut) -> None:
    """Random VSILUMUL descriptors: the sigmoid lookup, both shifts and the absmax."""
    env, rng = await setup(dut, 55)
    row_en = 3 if B_MAX > 1 else 1
    saturating = 0
    for case in range(CASES):
        n = LENGTHS[case % len(LENGTHS)] if case < len(LENGTHS) else rng.choice(LENGTHS)
        src, dst, aux = offsets(rng, ("", "src", "", "aux")[case % 4])
        mask, _, sr, dr = rows_of(rng)
        frac_gu = rng.randrange(13, 31)
        sh_h = rng.randrange(0, 5) if case % 3 == 0 else rng.randrange(0, 64)
        d = isa.vsilumul(
            vs_src=src,
            vs_aux=aux,
            vs_dst=dst,
            n=n,
            frac_gu=frac_gu,
            sh_h=sh_h,
            sreg_dst=rng.randrange(0, isa.SREG_COUNT),
            track_absmax=bool(rng.randrange(2)),
            src_row=sr,
            dst_row=dr,
            row_mask=mask,
        )
        scatter(env, rng)
        fill_sources(env, rng, d, row_en, bits=31 if case % 3 == 0 else rng.randrange(2, 32))
        before = env.sat
        await run_one(env, d, row_en)
        if env.sat > before:
            saturating += 1
    assert saturating >= 3, f"only {saturating} of {CASES} VSILUMUL cases saturated"


# --------------------------------------------------------------------------- VROPE

HEAD = isa.HEAD_DIM  # 64 elements, 32 pairs
ROPE_BYTES = isa.ROPE_ROW_BYTES  # 32 cos then 32 sin, int16 Q1.14
POSITIONS = (0, 1, 31, 32, 63, 64, 65, 255, 400)  # POS edges the table row is addressed from


def rope_row_bytes(rng: random.Random, *, full: bool = False) -> bytes:
    """One table row: ``HEAD/2`` cosines then ``HEAD/2`` sines.

    ``full`` draws over the whole int16 range instead of Q1.14, which is what
    puts ``a * cos - b * sin`` past the int32 the lane saturates to.
    """
    hi = (1 << 15) if full else ((1 << 14) + 1)
    vals = [rng.randrange(-hi, hi) for _ in range(HEAD)]
    return np.array(vals, dtype="<i2").tobytes()


def rope_descriptor(rng: random.Random, heads: int) -> tuple[Descriptor, int]:
    """A VROPE over ``heads`` heads with a random offset, row layout and table base."""
    src, _, _ = offsets(rng)
    mask, _, sr, _ = rows_of(rng)
    addr = ROW_BASE + ROPE_BYTES * rng.randrange(4)
    d = isa.vrope(vs_src=src, n=heads * HEAD, addr_a=addr, src_row=sr, row_mask=mask)
    return d, addr


@cocotb.test()
async def test_vrope(dut) -> None:
    """Random VROPE descriptors over every head count, offset, row layout and POS edge."""
    env, rng = await setup(dut, 66)
    row_en = 3 if B_MAX > 1 else 1
    saturating = 0
    for case in range(CASES):
        heads = (1, 2, 3, 4, 8)[case % 5] if case < 5 else rng.randrange(1, 6)
        d, addr = rope_descriptor(rng, heads)
        env.pos = POSITIONS[case % len(POSITIONS)]
        full = case % 3 == 0
        env.image = image_with(rope_row_bytes(rng, full=full), addr + env.pos * ROPE_BYTES)
        scatter(env, rng)
        fill_sources(env, rng, d, row_en, bits=31 if full else rng.randrange(2, 32))
        before = env.sat
        await run_one(env, d, row_en)
        if env.sat > before:
            saturating += 1
    assert saturating >= 3, f"only {saturating} of {CASES} VROPE cases saturated"


@cocotb.test()
async def test_vrope_pairs_and_saturation(dut) -> None:
    """The two halves of every pair, the rotation identity, and the int32 edge of both results."""
    env, rng = await setup(dut, 77)

    # cos = 1.0, sin = 0 in Q1.14 leaves the vector alone: any element that
    # moved would be a pair the unit crossed the wrong way.
    row = np.zeros(HEAD, dtype=np.int64)
    row[: HEAD // 2] = 1 << 14
    d = isa.vrope(vs_src=64, n=2 * HEAD, addr_a=ROW_BASE)
    env.pos = 7
    env.image = image_with(row.astype("<i2").tobytes(), ROW_BASE + env.pos * ROPE_BYTES)
    scatter(env, rng)
    x = rand_vec(rng, d.n, 16)
    fill_sources(env, rng, d, 1, values=x)
    await run_one(env, d, 1)
    got = env.vsram[0, d.vs_src : d.vs_src + d.n]
    bad = first_diff(x, got)
    assert bad is None, f"the identity rotation moved element {bad}: {x[bad]} -> {got[bad]}"

    # cos = 0, sin = 1.0 swaps the halves with a sign: a' = -b, b' = a. Every
    # element of both halves therefore has to change, which is what makes this
    # a check of the pairing rather than of the arithmetic.
    row = np.zeros(HEAD, dtype=np.int64)
    row[HEAD // 2 :] = 1 << 14
    env.image = image_with(row.astype("<i2").tobytes(), ROW_BASE + env.pos * ROPE_BYTES)
    scatter(env, rng)
    x = rand_vec(rng, d.n, 16)
    fill_sources(env, rng, d, 1, values=x)
    await run_one(env, d, 1)
    got = env.vsram[0, d.vs_src : d.vs_src + d.n]
    want = x.copy().reshape(-1, HEAD)
    a, b = x.reshape(-1, HEAD)[:, : HEAD // 2], x.reshape(-1, HEAD)[:, HEAD // 2 :]
    want[:, : HEAD // 2], want[:, HEAD // 2 :] = -b, a
    bad = first_diff(want.reshape(-1), got)
    assert bad is None, f"the quarter turn differs first at element {bad}"

    # the saturating edge: the most negative int32 against the most negative
    # cosine and a zero sine puts both a' and b' past int32, and each counts once
    row = np.zeros(HEAD, dtype=np.int64)
    row[: HEAD // 2] = -(1 << 15)
    env.image = image_with(row.astype("<i2").tobytes(), ROW_BASE + env.pos * ROPE_BYTES)
    d = isa.vrope(vs_src=64, n=HEAD, addr_a=ROW_BASE)
    scatter(env, rng)
    fill_sources(env, rng, d, 1, values=np.full(d.n, -(1 << 31), dtype=np.int64))
    before = env.sat
    await run_one(env, d, 1)
    assert env.sat - before == d.n, (
        f"{env.sat - before} of the {d.n} rotation results saturated, expected all of them"
    )


# --------------------------------------------------------------------------- VSOFTMAX


def meta_bytes(scales: list[SFloat]) -> bytes:
    """The 8-byte meta records of ``docs/MEMORY_MAP.md``: bias, mantissa, exponent, pad."""
    return b"".join(struct.pack("<iHbB", 0, s.m, s.e, 0) for s in scales)


def rand_scales(rng: random.Random, count: int, *, zeros: str = "mixed") -> list[SFloat]:
    """``count`` V scales: ``none``, ``mixed`` or ``all`` of them the canonical zero."""
    base = rng.randrange(-40, 40)
    spread = rng.choice((1, 2, 4, 20))
    out: list[SFloat] = []
    for _ in range(count):
        if zeros == "all" or (zeros == "mixed" and rng.randrange(4) == 0):
            out.append(SFloat(0, 0))
        else:
            e = max(-128, min(127, base + rng.randrange(-spread, spread + 1)))
            out.append(SFloat(rng.randrange(1 << 15, 1 << 16), e))
    return out


def rand_scores(rng: random.Random, count: int, frac_s: int, *, spread: int = 4) -> np.ndarray:
    """``count`` int32 scores in the log2 domain of class ``frac_s``, ``spread`` units wide."""
    step = min(spread << frac_s, (1 << 31) - 1)
    base = rng.randrange(-(1 << 20), 1 << 20)
    lo, hi = -(1 << 31), (1 << 31) - 1
    return np.array(
        [max(lo, min(hi, base + rng.randrange(-step, step + 1))) for _ in range(count)],
        dtype=np.int64,
    )


def softmax_descriptor(
    rng: random.Random,
    n: int,
    length: int | None,
    *,
    over: str = "",
    addr: int = ROW_BASE,
) -> Descriptor:
    src, dst, _ = offsets(rng, over)
    mask, _, sr, dr = rows_of(rng)
    return isa.vsoftmax(
        vs_src=src,
        vs_dst=dst,
        n=n,
        addr_a=addr,
        frac_s=rng.randrange(16, 31),
        sreg_dst=rng.randrange(0, isa.SREG_COUNT),
        length=length,
        src_row=sr,
        dst_row=dr,
        row_mask=mask,
    )


async def run_softmax(
    env: Env,
    rng: random.Random,
    d: Descriptor,
    length: int,
    row_en: int,
    *,
    scales: list[SFloat] | None = None,
    scores: np.ndarray | None = None,
    addr: int = ROW_BASE,
) -> None:
    """Lay the V-scale records and the scores down, then compare the descriptor."""
    sc = scales if scales is not None else rand_scales(rng, length)
    env.image = image_with(meta_bytes(sc), addr)
    scatter(env, rng)
    s = scores if scores is not None else rand_scores(rng, d.n, d.sh0)
    for _, src, _ in participants(d, row_en):
        fill(env, rng, src, d.vs_src, s)
    await run_one(env, d, row_en)


@cocotb.test()
async def test_vsoftmax(dut) -> None:
    """Random VSOFTMAX rows over every length class, offset, row layout and scale mix."""
    env, rng = await setup(dut, 88)
    row_en = 3 if B_MAX > 1 else 1
    zero_scale_rows = 0
    clipped = 0
    for case in range(CASES // 2):
        length = LENGTHS[case % len(LENGTHS)] if case < len(LENGTHS) else rng.choice(LENGTHS)
        n = length if case % 3 == 0 else length + rng.choice((1, 7, 8, 64))
        env.pos = length - 1
        from_pos = case % 2 == 0
        d = softmax_descriptor(rng, n, None if from_pos else length, over=in_place(case))
        mix = ("mixed", "none", "mixed", "all")[case % 4]
        before = int(dut.clip_count.value)
        await run_softmax(env, rng, d, length, row_en, scales=rand_scales(rng, length, zeros=mix))
        if mix == "all":
            zero_scale_rows += 1
        if int(dut.clip_count.value) > before:
            clipped += 1
    assert zero_scale_rows >= 3, "the all-zero V scale row was never driven"
    print(f"VSOFTMAX: {clipped} of {CASES // 2} rows reached the weight clip")


@cocotb.test()
async def test_vsoftmax_boundaries(dut) -> None:
    """The rows numerics.softmax singles out: equal scores, the clamp, the clip, the zero scale."""
    env, rng = await setup(dut, 99)
    frac_s = 20
    step = 1 << frac_s

    # every score equal: every distance is 0, every e_t is 2^23 and the weights
    # differ only by their V scales
    for length in (1, 2, 63, 64, 65):
        n = length + 8
        env.pos = length - 1
        d = isa.vsoftmax(
            vs_src=64, vs_dst=4096, n=n, addr_a=ROW_BASE, frac_s=frac_s, sreg_dst=2, length=length
        )
        scores = np.full(n, 12345, dtype=np.int64)
        scales = [SFloat(1 << 15, 3) for _ in range(length)]
        await run_softmax(env, rng, d, length, 1, scales=scales, scores=scores)

    # far below the maximum: 25 log2 units and beyond round to a zero weight,
    # and the token at the boundary is the one that decides it
    length, n = 32, 40
    env.pos = length - 1
    d = isa.vsoftmax(
        vs_src=64, vs_dst=4096, n=n, addr_a=ROW_BASE, frac_s=frac_s, sreg_dst=3, length=length
    )
    for below in (24, 25, 26, 40, 60):
        scores = np.zeros(n, dtype=np.int64)
        scores[1:length] = -below * step
        scores[2] = -(below * step) - (step // 2)
        scales = [SFloat(1 << 15, 0) for _ in range(length)]
        await run_softmax(env, rng, d, length, 1, scales=scales, scores=scores)

    # the token that reaches the clip: one score at the maximum with the largest
    # V mantissa and every other token 25 units below, so p = 1.0 and
    # w = round_shift(2^23 * 65535, 24) = 32768, clipped to 32767
    scores = np.full(n, -(30 * step), dtype=np.int64)
    scores[0] = 0
    scales = [SFloat((1 << 16) - 1, -5)] + [SFloat(1 << 15, -5) for _ in range(length - 1)]
    before = int(dut.clip_count.value)
    sat_before = env.sat
    await run_softmax(env, rng, d, length, 1, scales=scales, scores=scores)
    assert int(dut.clip_count.value) - before == 1, (
        f"{int(dut.clip_count.value) - before} clips, expected the one token at 32768"
    )
    assert env.vsram[0, d.vs_dst] == 32767, "the clipped weight is not the int16 maximum"
    assert env.sat == sat_before, "the weight clip raised SAT_VPU instead of counting as a clip"

    # the weight shift at its cap: one token holds the whole probability and its
    # own V scale sits `gap` exponents below e_max, so its shift is 24 + gap,
    # and 40 is the cap past which the product no longer reaches the weight
    length2, n2 = 4, 8
    env.pos = length2 - 1
    d2 = isa.vsoftmax(
        vs_src=64, vs_dst=4096, n=n2, addr_a=ROW_BASE, frac_s=frac_s, sreg_dst=8, length=length2
    )
    scores = np.full(n2, -(30 * step), dtype=np.int64)
    scores[0] = 0
    for gap in (14, 15, 16, 17):
        scales = [SFloat((1 << 16) - 1, 0), SFloat(1 << 15, gap)]
        scales += [SFloat(1 << 15, 0) for _ in range(length2 - 2)]
        await run_softmax(env, rng, d2, length2, 1, scales=scales, scores=scores)

    # a V scale far below the row's largest one: 24 + e_max - Sv_e runs past the
    # 40 the weight shift is capped at, and past the six bits the lane's shift
    # field holds, so the cap is what keeps those tokens at a zero weight
    scores = np.zeros(n, dtype=np.int64)
    scales = [SFloat(1 << 15, 60)] + [SFloat(1 << 15, -60) for _ in range(length - 1)]
    await run_softmax(env, rng, d, length, 1, scales=scales, scores=scores)
    assert env.vsram[0, d.vs_dst] > 0, "the token holding e_max lost its weight"
    assert not env.vsram[0, d.vs_dst + 1 : d.vs_dst + length].any(), (
        "a token 120 log2 units of scale below e_max kept a weight"
    )

    # a V scale that is the canonical zero takes no part in e_max and gets a
    # zero weight; a row of nothing but zero scales is the zero vector and the
    # zero scale register
    scales = [SFloat(1 << 15, 7)] + [SFloat(0, 0)] * (length - 1)
    await run_softmax(env, rng, d, length, 1, scales=scales)
    scales = [SFloat(0, 0)] * length
    await run_softmax(env, rng, d, length, 1, scales=scales)
    assert env.sreg[0][3] == 0, "an all-zero row has to leave the canonical zero scale"
    assert not env.vsram[0, d.vs_dst : d.vs_dst + n].any(), "an all-zero row has to be zeros"

    # extreme scores at both ends of int32, which is where the 33-bit difference
    # and the clamp are the only thing keeping the distance in range
    for frac in (16, 23, 30):
        d = isa.vsoftmax(
            vs_src=64, vs_dst=4096, n=n, addr_a=ROW_BASE, frac_s=frac, sreg_dst=4, length=length
        )
        scores = np.array(
            [(1 << 31) - 1 if i % 3 == 0 else -(1 << 31) for i in range(n)], dtype=np.int64
        )
        await run_softmax(env, rng, d, length, 1)
        await run_softmax(env, rng, d, length, 1, scores=scores)


@cocotb.test()
async def test_vsoftmax_class_window(dut) -> None:
    """Every class the ISA lets a VSOFTMAX carry: both edges of the window and one between them.

    The sequencer faults a descriptor outside ``isa.CLASS_WINDOW`` (docs/RTL.md
    3.5), so these three classes are the whole set this block is issued, and
    each one is compared against ``numerics.softmax`` on structured distances
    and on random rows.
    """
    env, rng = await setup(dut, 91)
    lo, hi = isa.CLASS_WINDOW[Opcode.VSOFTMAX]
    for frac_s in (lo, (lo + hi) // 2, hi):
        step = 1 << frac_s
        # whole log2 units past the 25-unit clamp, and half units between them,
        # as many as an int32 score holds at this class
        units = max(1, min(26, ((1 << 31) - 1) // step))
        for length, n in ((1, 8), (5, 5), (33, 40), (64, 72)):
            env.pos = length - 1
            d = isa.vsoftmax(
                vs_src=64,
                vs_dst=4096,
                n=n,
                addr_a=ROW_BASE,
                frac_s=frac_s,
                sreg_dst=4,
                length=length,
            )
            scores = np.zeros(n, dtype=np.int64)
            for i in range(1, length):
                scores[i] = -((i % (units + 1)) * step) - (step // 2 if i % 3 else 0)
            await run_softmax(
                env, rng, d, length, 1, scales=rand_scales(rng, length), scores=scores
            )
            await run_softmax(
                env,
                rng,
                d,
                length,
                1,
                scales=rand_scales(rng, length, zeros="none"),
                scores=rand_scores(rng, n, frac_s),
            )


@cocotb.test()
async def test_vsoftmax_length_and_fill(dut) -> None:
    """``len`` from POS and from the immediate, its clamp, and the zeros past it."""
    env, rng = await setup(dut, 121)
    n = 128
    for pos in (0, 1, 62, 63, 64, 127, 200):
        env.pos = pos
        length = min(pos + 1, n)
        d = isa.vsoftmax(
            vs_src=64, vs_dst=4096, n=n, addr_a=ROW_BASE, frac_s=18, sreg_dst=5, length=None
        )
        await run_softmax(env, rng, d, length, 1, scales=rand_scales(rng, n))
        tail = env.vsram[0, d.vs_dst + length : d.vs_dst + n]
        assert not tail.any(), f"POS {pos}: {int(np.count_nonzero(tail))} weights past len are set"

    # the immediate form, including a length of exactly n (no fill pass at all)
    for length in (1, n // 2, n):
        env.pos = 0
        d = isa.vsoftmax(
            vs_src=64, vs_dst=4096, n=n, addr_a=ROW_BASE, frac_s=18, sreg_dst=6, length=length
        )
        await run_softmax(env, rng, d, length, 1)

    # the longest row the bench can lay down: MAX_CTX tokens, which is the
    # capacity a compiled attention step gives the destination
    long_n = 2048
    env.pos = long_n - 1
    d = isa.vsoftmax(
        vs_src=64, vs_dst=4096, n=long_n, addr_a=ROW_BASE, frac_s=22, sreg_dst=7, length=None
    )
    await run_softmax(env, rng, d, long_n, 1)


@cocotb.test()
async def test_vsoftmax_memory(dut) -> None:
    """The V-scale stream at latencies 1, 32 and 200, and the two runs of it a row makes."""
    env, rng = await setup(dut, 131)
    grants = {
        "free": lambda c: 1,
        "every 4th": lambda c: (c % 4) == 0,
        "every 17th": lambda c: (c % 17) == 3,
    }
    for latency in (1, 32, 200):
        for name, pattern in grants.items():
            env.latency = latency
            env.req_ready = pattern
            length = rng.choice((1, 8, 33, 100))
            n = length + rng.choice((0, 9))
            env.pos = length - 1
            d = softmax_descriptor(rng, n, length)
            beats = env.beats_returned
            # every V scale is real, so the weight pass runs and streams the
            # records a second time
            await run_softmax(env, rng, d, length, 1, scales=rand_scales(rng, length, zeros="none"))
            per_pass = -(-length // (WB // 8))
            want = 2 * per_pass * len(participants(d, 1))
            assert env.beats_returned - beats == want, (
                f"VSOFTMAX len={length} at latency {latency} ({name}) returned "
                f"{env.beats_returned - beats} beats, expected {want}"
            )

            heads = rng.choice((1, 2))
            d, addr = rope_descriptor(rng, heads)
            env.pos = rng.choice(POSITIONS)
            env.image = image_with(rope_row_bytes(rng), addr + env.pos * ROPE_BYTES)
            scatter(env, rng)
            fill_sources(env, rng, d, 1, bits=20)
            beats = env.beats_returned
            await run_one(env, d, 1)
            want = -(-ROPE_BYTES // WB) * len(participants(d, 1))
            assert env.beats_returned - beats == want, (
                f"VROPE at latency {latency} ({name}) returned "
                f"{env.beats_returned - beats} beats, expected {want}"
            )


# --------------------------------------------------------------------------- boundary values


@cocotb.test()
async def test_boundaries(dut) -> None:
    """The boundary value of every operation, driven directly."""
    env, rng = await setup(dut, 66)
    lim = (1 << 31) - 1
    neg = -(1 << 31)

    # ---- VSUBC: both saturation directions, a zero constant row, the extremes
    n = 12
    x = np.array([lim, neg, lim, neg, 0, 1, -1, lim, neg, 0, 7, -7], dtype=np.int64)
    c = np.array([-1, 1, lim, neg, 0, lim, neg, 0, 0, neg, 3, -3], dtype=np.int64)
    d = isa.vsubc(vs_src=64, vs_dst=4096, n=n, addr_a=ROW_BASE)
    env.image = image_with(const_bytes(c), ROW_BASE)
    scatter(env, rng)
    fill_sources(env, rng, d, 1, values=x)
    before = env.sat
    await run_one(env, d, 1)
    assert env.sat - before == 3, f"the directed VSUBC saturated {env.sat - before} times, not 3"

    # ---- VQUANT: the zero vector, an absmax of 2^31, one element, both widths
    for width in (8, 16):
        for vec in (
            np.zeros(9, dtype=np.int64),
            np.array([neg, 0, lim, 1, -1, neg, lim, 0, 2], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([neg], dtype=np.int64),
            np.array([lim] * 8, dtype=np.int64),
        ):
            for frac_in in (0, 15, 30):
                d = isa.vquant(
                    vs_src=64,
                    vs_dst=4096,
                    n=int(vec.size),
                    width=width,
                    frac_in=frac_in,
                    sreg_dst=3,
                )
                scatter(env, rng)
                fill_sources(env, rng, d, 1, values=vec)
                await run_one(env, d, 1)

    # ---- VQUANT: a tracked absmax of 0 and of 2^32 - 1 over non-zero data
    vec = np.array([lim, neg, 5, -5, 0, 1], dtype=np.int64)
    for amax in (0, 1, 1 << 15, MASK32):
        d = isa.vquant(
            vs_src=64,
            vs_dst=4096,
            n=int(vec.size),
            width=16,
            frac_in=16,
            sreg_dst=4,
            use_tracked=True,
            sreg_src=9,
        )
        scatter(env, rng)
        fill_sources(env, rng, d, 1, values=vec)
        await run_one(env, d, 1, sreg_pre={(0, 9): amax})

    # ---- VQUANT: SCALE_MUL with the extreme sfloat mantissas
    for mul in (SFloat(1 << 15, -15), SFloat(0xFFFF, 40), SFloat(0x8000, -60)):
        d = isa.vquant(
            vs_src=64,
            vs_dst=4096,
            n=8,
            width=8,
            frac_in=8,
            sreg_dst=5,
            scale_mul=mul,
        )
        scatter(env, rng)
        fill_sources(env, rng, d, 1, values=np.array([3, -9, 100, lim, neg, 0, 7, -1]))
        await run_one(env, d, 1)

    # ---- VRMSNORM: the zero vector with and without an epsilon, and gamma extremes
    for eps_c in (0, 1 << 20):
        gq = np.array([32767, -32768, 0, 1, -1, 16384, -16384, 5], dtype=np.int64)
        d = isa.vrmsnorm(
            vs_src=64,
            vs_dst=4096,
            n=8,
            addr_a=ROW_BASE,
            eps_c=eps_c,
            frac_x=16,
            g=0,
            sqrt_d=sqrt_d_of(8),
            sreg_dst=6,
        )
        env.image = image_with(gamma_bytes(gq), ROW_BASE)
        scatter(env, rng)
        fill_sources(env, rng, d, 1, values=np.zeros(8, dtype=np.int64))
        await run_one(env, d, 1)

    # ---- VRMSNORM: an absmax of 2^31 and a G of 63
    gq = np.array([32767, -32768, 1, -1, 0, 32767, -32768, 3], dtype=np.int64)
    for g_shift in (0, 1, 63):
        d = isa.vrmsnorm(
            vs_src=64,
            vs_dst=4096,
            n=8,
            addr_a=ROW_BASE,
            eps_c=eps_of(8, 16),
            frac_x=16,
            g=g_shift,
            sqrt_d=sqrt_d_of(8),
            sreg_dst=6,
        )
        env.image = image_with(gamma_bytes(gq), ROW_BASE)
        scatter(env, rng)
        fill_sources(
            env,
            rng,
            d,
            1,
            values=np.array([neg, lim, neg, 0, 1, -1, lim, neg], dtype=np.int64),
        )
        await run_one(env, d, 1)

    # ---- VSILUMUL: the sigmoid clamp on both signs, a zero argument and the shift extremes
    frac_gu = 16
    big = 16 << frac_gu  # |g| >= 16 saturates the table
    vec = np.array([0, big, -big, big + 1, -big - 1, 1, -1, neg, lim, 1 << 20], dtype=np.int64)
    aux = np.array([lim, neg, 1, -1, 0, lim, neg, lim, neg, 3], dtype=np.int64)
    for sh_h in (0, 15, 32, 63):
        d = isa.vsilumul(
            vs_src=64,
            vs_aux=8192,
            vs_dst=4096,
            n=int(vec.size),
            frac_gu=frac_gu,
            sh_h=sh_h,
            sreg_dst=7,
        )
        scatter(env, rng)
        fill_sources(env, rng, d, 1, values=vec, aux=aux)
        await run_one(env, d, 1)

    # ---- VSILUMUL: every sigmoid index class at the smallest and largest FRAC_GU
    for frac_gu in (13, 30):
        step = max(1, (16 << frac_gu) // 40)
        vec = np.array(
            [min(max(v, neg), lim) for v in range(-20 * step, 20 * step, step)], dtype=np.int64
        )
        d = isa.vsilumul(
            vs_src=64,
            vs_aux=8192,
            vs_dst=4096,
            n=int(vec.size),
            frac_gu=frac_gu,
            sh_h=min(63, 2 * frac_gu - 16),
            sreg_dst=7,
        )
        scatter(env, rng)
        fill_sources(env, rng, d, 1, values=vec, aux=np.full(vec.size, 1 << 14, dtype=np.int64))
        await run_one(env, d, 1)


@cocotb.test()
async def test_shift_events(dut) -> None:
    """The ERR_SHIFT sources: a negative S1, and sh1 outside [0, 63] on both ops."""
    env, rng = await setup(dut, 77)

    # sh1 below zero and above 63, counted once per element per row
    for op, sh1 in (
        (Opcode.VRMSNORM, -3),
        (Opcode.VRMSNORM, 100),
        (Opcode.VSILUMUL, -1),
        (Opcode.VSILUMUL, 127),
    ):
        n = 20
        if op == Opcode.VRMSNORM:
            gq = np.array([rng.randrange(-32768, 32768) for _ in range(n)], dtype=np.int64)
            env.image = image_with(gamma_bytes(gq), ROW_BASE)
            d = Descriptor(
                opcode=op,
                row_mask=1,
                track_absmax=True,
                addr_a=ROW_BASE,
                addr_m=isa.sfloat_imm(sqrt_d_of(n)),
                n=n,
                vs_src=64,
                vs_dst=4096,
                sreg_dst=2,
                sh0=16,
                sh1=sh1,
                imm32=eps_of(n, 16),
            )
        else:
            d = Descriptor(
                opcode=op,
                row_mask=1,
                track_absmax=True,
                n=n,
                vs_src=64,
                vs_aux=8192,
                vs_dst=4096,
                sreg_dst=2,
                sh0=16,
                sh1=sh1,
            )
        scatter(env, rng)
        fill_sources(env, rng, d, 1, bits=20)
        before = env.err_shift
        await run_one(env, d, 1)
        assert env.err_shift - before >= n, (
            f"{op.name} with sh1 = {sh1} counted {env.err_shift - before} shift errors, not {n}"
        )

    # a negative S1: tiny values with no epsilon leave the scale exponent above FRAC_X
    n = 64
    gq = np.ones(n, dtype=np.int64)
    env.image = image_with(gamma_bytes(gq), ROW_BASE)
    d = isa.vrmsnorm(
        vs_src=64,
        vs_dst=4096,
        n=n,
        addr_a=ROW_BASE,
        eps_c=0,
        frac_x=16,
        g=0,
        sqrt_d=sqrt_d_of(n),
        sreg_dst=2,
    )
    scatter(env, rng)
    fill_sources(env, rng, d, 1, values=np.array([1, -1] * (n // 2), dtype=np.int64))
    before = env.err_shift
    await run_one(env, d, 1)
    assert env.err_shift - before == n, (
        f"the negative-S1 case counted {env.err_shift - before} shift errors, not {n}"
    )


@cocotb.test()
async def test_vrmsnorm_shift_ceiling(dut) -> None:
    """S1 at the top of the shift field and past it: one clamp model on both sides.

    ``S1 = -(Rc_e + FRAC_X - sh - e)``, so a ``sqrt(d)`` constant with a small
    enough exponent walks it up to 63 and beyond.  At 63 the unit and
    ``numerics.rmsnorm`` shift by 63 and count nothing; past it both clamp to 63
    and count one ``ERR_SHIFT`` per element, the rule the requant clamp and
    ``qcore_vpu_scalar`` already follow at the other end.
    """
    env, rng = await setup(dut, 1010)
    n = 16
    env.image = image_with(gamma_bytes(np.ones(n, dtype=np.int64)), ROW_BASE)
    x = np.array([1, -1] * (n // 2), dtype=np.int64)
    # x has absmax 1 and ss = n, so sh = 0, e = 2 and Rc = {2^15, sqrt_e}: S1 = 2 - sqrt_e.
    for sqrt_e, errors in ((-59, 0), (-61, 0), (-62, n), (-100, n)):
        d = isa.vrmsnorm(
            vs_src=64,
            vs_dst=4096,
            n=n,
            addr_a=ROW_BASE,
            eps_c=0,
            frac_x=0,
            g=0,
            sqrt_d=SFloat(1 << 15, sqrt_e),
            sreg_dst=2,
            track_absmax=True,
        )
        scatter(env, rng)
        fill_sources(env, rng, d, 1, values=x)
        before = env.err_shift
        await run_one(env, d, 1)
        assert env.err_shift - before == errors, (
            f"sqrt(d) exponent {sqrt_e} counted {env.err_shift - before} shift errors, not {errors}"
        )


# --------------------------------------------------------------------------- rows and ranges


@cocotb.test()
async def test_rows(dut) -> None:
    """Row masks, ROW_EN, the row bases, an empty participating set and n = 0."""
    env, rng = await setup(dut, 88)
    masks = [(1, 1, 0, 0)]
    if B_MAX > 1:
        masks += [
            (3, 3, 0, 0),
            (2, 3, 0, 0),
            (1, 3, 1, 1),
            (1, 1, 0, 1),
            (3, 1, 0, 0),
            (3, 2, 0, 0),
            (1, 2, 0, 0),
            (0, 3, 0, 0),
        ]
    for mask, row_en, sr, dr in masks:
        n = 21
        d = isa.vquant(
            vs_src=64,
            vs_dst=4096,
            n=n,
            width=8,
            frac_in=12,
            sreg_dst=1,
            src_row=sr,
            dst_row=dr,
            row_mask=mask,
        )
        scatter(env, rng)
        for r in range(B_MAX):
            fill(env, rng, r, d.vs_src, rand_vec(rng, n, 20 + r))
        await run_one(env, d, row_en)

    # n = 0 and an empty set both retire with no work
    for d in (
        isa.vquant(vs_src=64, vs_dst=4096, n=0, width=16, frac_in=8, sreg_dst=1),
        isa.vsubc(vs_src=64, vs_dst=4096, n=0, addr_a=ROW_BASE),
    ):
        scatter(env, rng)
        before = env.vsram.copy()
        await run_one(env, d, 1)
        assert np.array_equal(env.vsram, before), "a zero-length descriptor wrote VSRAM"
        assert env.sreg_writes == [], "a zero-length descriptor wrote an SREG"


@cocotb.test()
async def test_ranges(dut) -> None:
    """Operand ranges that leave the VSRAM, and SREG indices at or above 32."""
    env, rng = await setup(dut, 99)
    n = 20
    # read, write and auxiliary ranges past the end, one ERR_BOUNDS each
    cases = [
        isa.vquant(vs_src=ELEMS - 5, vs_dst=4096, n=n, width=16, frac_in=8, sreg_dst=1),
        isa.vquant(vs_src=64, vs_dst=ELEMS - 5, n=n, width=8, frac_in=8, sreg_dst=1),
        isa.vquant(vs_src=ELEMS - 5, vs_dst=ELEMS - 5, n=n, width=8, frac_in=8, sreg_dst=1),
        isa.vsilumul(
            vs_src=64, vs_aux=ELEMS - 3, vs_dst=4096, n=n, frac_gu=16, sh_h=16, sreg_dst=1
        ),
        isa.vsilumul(
            vs_src=ELEMS - 2, vs_aux=8192, vs_dst=ELEMS - 2, n=n, frac_gu=16, sh_h=16, sreg_dst=1
        ),
        # VROPE reads and writes one range and counts it twice, as isa_sim does
        isa.vrope(vs_src=ELEMS - 5, n=HEAD, addr_a=ROW_BASE),
        # VSOFTMAX counts the vs_src + len read and the vs_dst + n write
        isa.vsoftmax(
            vs_src=ELEMS - 5,
            vs_dst=4096,
            n=n,
            addr_a=ROW_BASE,
            frac_s=20,
            sreg_dst=1,
            length=n,
        ),
        isa.vsoftmax(
            vs_src=64,
            vs_dst=ELEMS - 5,
            n=n,
            addr_a=ROW_BASE,
            frac_s=20,
            sreg_dst=1,
            length=n,
        ),
    ]
    for d in cases:
        if d.opcode == Opcode.VROPE:
            env.pos = 2
            env.image = image_with(rope_row_bytes(rng), d.addr_a + env.pos * ROPE_BYTES)
        elif d.opcode == Opcode.VSOFTMAX:
            env.pos = n - 1
            env.image = image_with(meta_bytes(rand_scales(rng, n)), d.addr_a)
        scatter(env, rng)
        fill_sources(env, rng, d, 1, bits=20)
        before = env.err_bounds
        await run_one(env, d, 1)
        assert env.err_bounds > before, f"{describe(d)} raised no ERR_BOUNDS"

    # the read of a VSOFTMAX is vs_src + len, not vs_src + n: a source that
    # only its capacity would run past the end raises nothing
    env.pos = 4
    d = isa.vsoftmax(
        vs_src=ELEMS - n, vs_dst=4096, n=n, addr_a=ROW_BASE, frac_s=20, sreg_dst=1, length=5
    )
    env.image = image_with(meta_bytes(rand_scales(rng, n)), d.addr_a)
    scatter(env, rng)
    fill_sources(env, rng, d, 1, bits=20)
    before = env.err_bounds
    await run_one(env, d, 1)
    assert env.err_bounds == before, "a VSOFTMAX read inside its length raised ERR_BOUNDS"

    # a VSUBC whose ranges are both inside raises nothing
    d = isa.vsubc(vs_src=64, vs_dst=4096, n=n, addr_a=ROW_BASE)
    env.image = image_with(const_bytes(rand_vec(rng, n, 20)), ROW_BASE)
    scatter(env, rng)
    fill_sources(env, rng, d, 1, bits=20)
    before = env.err_bounds
    await run_one(env, d, 1)
    assert env.err_bounds == before, "an in-range VSUBC raised ERR_BOUNDS"

    # SREG indices at or above 32: the write is issued and the bank drops it
    for d in (
        Descriptor(
            opcode=Opcode.VQUANT, row_mask=1, n=16, vs_src=64, vs_dst=4096, sreg_dst=40, sh0=8
        ),
        Descriptor(
            opcode=Opcode.VQUANT,
            row_mask=1,
            flags=int(VquantFlag.GROUP),
            n=16,
            vs_src=64,
            vs_dst=4096,
            vs_aux=4,
            sreg_dst=30,
            sh0=8,
        ),
        Descriptor(
            opcode=Opcode.VQUANT,
            row_mask=1,
            flags=int(VquantFlag.GROUP),
            n=16,
            vs_src=64,
            vs_dst=4096,
            vs_aux=8,
            sreg_dst=252,
            sh0=8,
        ),
        Descriptor(
            opcode=Opcode.VRMSNORM,
            row_mask=1,
            track_absmax=True,
            addr_a=ROW_BASE,
            addr_m=isa.sfloat_imm(sqrt_d_of(16)),
            n=16,
            vs_src=64,
            vs_dst=4096,
            sreg_dst=99,
            sh0=16,
            sh1=0,
            imm32=eps_of(16, 16),
        ),
        # VSOFTMAX writes its output scale whether or not track_absmax is set
        Descriptor(
            opcode=Opcode.VSOFTMAX,
            row_mask=1,
            addr_a=ROW_BASE,
            n=16,
            vs_src=64,
            vs_dst=4096,
            sreg_dst=77,
            sh0=20,
            imm32=16,
        ),
    ):
        if d.opcode == Opcode.VRMSNORM:
            env.image = image_with(gamma_bytes(np.ones(16, dtype=np.int64)), ROW_BASE)
        elif d.opcode == Opcode.VSOFTMAX:
            env.pos = 15
            env.image = image_with(meta_bytes(rand_scales(rng, 16, zeros="none")), ROW_BASE)
        scatter(env, rng)
        fill_sources(env, rng, d, 1, bits=18)
        before = env.sreg_err
        await run_one(env, d, 1)
        assert env.sreg_err > before, f"{describe(d)} issued no out-of-range SREG write"


@cocotb.test()
async def test_in_place(dut) -> None:
    """The permitted half of the range rule: a destination exactly over a source.

    The compiler quantizes q, K, V and ``silu(gate) * up`` in place and centres
    K in place, so this half of the rule carries most of the vector work of a
    compiled program.  It is driven here at a word-aligned and at an unaligned
    element offset, on every bank the configuration has, and over the auxiliary
    range of the gated multiply as well as over the source.
    """
    env, rng = await setup(dut, 1111)
    n = 21
    ran = changed = 0
    gq, ge = numerics.quantize_gamma(
        np.array([rng.uniform(-3.0, 3.0) for _ in range(n)], dtype=np.float64)
    )
    c = rand_vec(rng, n, 22)
    for base in (64, 67):
        for row in range(B_MAX):
            rows = {"src_row": row, "dst_row": row, "row_mask": 1}
            aux = 8192 + base % NE
            for d in (
                isa.vquant(vs_src=base, vs_dst=base, n=n, width=16, frac_in=14, sreg_dst=1, **rows),
                isa.vquant(vs_src=base, vs_dst=base, n=n, width=8, frac_in=12, sreg_dst=2, **rows),
                isa.vsubc(vs_src=base, vs_dst=base, n=n, addr_a=ROW_BASE, **rows),
                isa.vrmsnorm(
                    vs_src=base,
                    vs_dst=base,
                    n=n,
                    addr_a=ROW_BASE,
                    eps_c=eps_of(n, 16),
                    frac_x=16,
                    g=-ge,
                    sqrt_d=sqrt_d_of(n),
                    sreg_dst=3,
                    **rows,
                ),
                isa.vsilumul(
                    vs_src=base,
                    vs_aux=aux,
                    vs_dst=base,
                    n=n,
                    frac_gu=16,
                    sh_h=16,
                    sreg_dst=4,
                    **rows,
                ),
                isa.vsilumul(
                    vs_src=base,
                    vs_aux=aux,
                    vs_dst=aux,
                    n=n,
                    frac_gu=16,
                    sh_h=16,
                    sreg_dst=5,
                    **rows,
                ),
            ):
                if d.opcode == Opcode.VSUBC:
                    env.image = image_with(const_bytes(c), ROW_BASE)
                elif d.opcode == Opcode.VRMSNORM:
                    env.image = image_with(gamma_bytes(gq), ROW_BASE)
                scatter(env, rng)
                fill_sources(env, rng, d, 1, bits=20)
                before = env.vsram.copy()
                await run_one(env, d, 1)
                ran += 1
                changed += int(not np.array_equal(before, env.vsram))
    assert changed == ran, f"only {changed} of {ran} in-place descriptors wrote a new value"


@cocotb.test()
async def test_cross_bank_ranges(dut) -> None:
    """Ranges that overlap in element index but not in bank are two memories, not one.

    Row ``r`` reads bank ``src_row + r`` and writes bank ``dst_row + r``, so a
    destination that partially overlaps a source is legal when the row bases
    differ.  ``docs/ISA.md``, ``compiler.vector_overlap`` and the simulation
    check in ``qcore_top`` -- the level that owns the bank crossbar -- are all
    qualified on the bases; the same descriptor with one base is the rejected
    form, and the unit still produces what isa_sim does.
    """
    env, rng = await setup(dut, 1212)
    if B_MAX < 2:
        dut._log.info("cross-bank ranges: this configuration has one bank")
        return
    n = 32
    c = rand_vec(rng, n, 22)
    env.image = image_with(const_bytes(c), ROW_BASE)
    banks = {"src_row": 0, "dst_row": 1, "row_mask": 1}
    for d in (
        isa.vquant(vs_src=64, vs_dst=68, n=n, width=8, frac_in=12, sreg_dst=1, **banks),
        isa.vquant(vs_src=68, vs_dst=64, n=n, width=16, frac_in=14, sreg_dst=2, **banks),
        isa.vsubc(vs_src=64, vs_dst=79, n=n, addr_a=ROW_BASE, **banks),
        isa.vsilumul(
            vs_src=64, vs_aux=100, vs_dst=104, n=n, frac_gu=16, sh_h=16, sreg_dst=3, **banks
        ),
    ):
        assert compiler.vector_overlap(d) is None, (
            f"the range rule rejected a legal cross-bank descriptor: {describe(d)}"
        )
        assert compiler.vector_overlap(dataclasses.replace(d, dst_row=0)) is not None, (
            f"the same ranges in one bank are the rejected form: {describe(d)}"
        )
        scatter(env, rng)
        fill_sources(env, rng, d, 1, bits=20)
        await run_one(env, d, 1)


# --------------------------------------------------------------------------- the memory port


@cocotb.test()
async def test_memory(dut) -> None:
    """The streamed operands at memory latencies 1, 32 and 200, and with a throttled arbiter."""
    env, rng = await setup(dut, 111)
    grants = {
        "free": lambda c: 1,
        "every 4th": lambda c: (c % 4) == 0,
        "every 17th": lambda c: (c % 17) == 3,
    }
    for latency in (1, 32, 200):
        for name, pattern in grants.items():
            env.latency = latency
            env.req_ready = pattern
            n = rng.choice((1, 8, 33, 100))
            d, c, addr = subc_descriptor(rng, n)
            env.image = image_with(const_bytes(c), addr)
            scatter(env, rng)
            fill_sources(env, rng, d, 1, bits=24)
            beats = env.beats_returned
            await run_one(env, d, 1)
            want = -(-n // (WB // 4)) * len(participants(d, 1))
            assert env.beats_returned - beats == want, (
                f"VSUBC n={n} at latency {latency} ({name}) returned "
                f"{env.beats_returned - beats} beats, expected {want}"
            )

            n = rng.choice((1, 8, 40))
            d, gq, addr = rmsnorm_descriptor(rng, n)
            env.image = image_with(gamma_bytes(gq), addr)
            scatter(env, rng)
            fill_sources(env, rng, d, 1, bits=24)
            beats = env.beats_returned
            await run_one(env, d, 1)
            want = -(-n // (WB // 2)) * len(participants(d, 1))
            assert env.beats_returned - beats == want, (
                f"VRMSNORM n={n} at latency {latency} ({name}) returned "
                f"{env.beats_returned - beats} beats, expected {want}"
            )
    for _, length in env.requests:
        assert length <= min(64, 16), f"a burst of {length} beats exceeds the FIFO reservation"


# --------------------------------------------------------------------------- the handshake


@cocotb.test()
async def test_protocol(dut) -> None:
    """busy and done around a descriptor, back-to-back issue, and the pass throughput."""
    env, rng = await setup(dut, 222)

    # busy covers exactly the descriptor and done is one cycle wide
    n = 64
    d = isa.vquant(vs_src=64, vs_dst=4096, n=n, width=16, frac_in=16, sreg_dst=1)
    scatter(env, rng)
    fill_sources(env, rng, d, 1, bits=24)
    assert int(dut.busy.value) == 0, "busy was high before the issue"
    busy0 = env.busy_cycles
    cycles = await run_one(env, d, 1)
    # busy rises the cycle after the issue pulse and covers the run to done; the
    # count excludes the issue cycle itself and the four settling steps of run()
    assert env.busy_cycles - busy0 == cycles - 5, (
        f"busy covered {env.busy_cycles - busy0} of the descriptor's {cycles - 5} cycles"
    )
    assert int(dut.busy.value) == 0, "busy stayed high after done"

    # back-to-back descriptors, issued the cycle after done
    for _ in range(6):
        d = isa.vquant(
            vs_src=64,
            vs_dst=4096,
            n=rng.choice((1, 8, 17, 64)),
            width=8,
            frac_in=rng.randrange(31),
            sreg_dst=rng.randrange(30),
        )
        scatter(env, rng)
        fill_sources(env, rng, d, 1, bits=rng.randrange(4, 32))
        await run_one(env, d, 1)

    # the one-product passes issue a chunk a cycle, the two-product passes every other
    n = 512
    d = isa.vquant(vs_src=64, vs_dst=4096, n=n, width=8, frac_in=16, sreg_dst=1)
    scatter(env, rng)
    fill_sources(env, rng, d, 1, bits=24)
    quant_cycles = await run_one(env, d, 1)
    d = isa.vsilumul(
        vs_src=64,
        vs_aux=8192,
        vs_dst=4096,
        n=n,
        frac_gu=16,
        sh_h=16,
        sreg_dst=1,
    )
    scatter(env, rng)
    fill_sources(env, rng, d, 1, bits=24)
    silu_cycles = await run_one(env, d, 1)
    per_quant = quant_cycles / (2 * n / VL)  # two passes of n elements
    per_silu = silu_cycles / (2 * n / VL)  # one pass at half rate
    print(
        f"VPU_TOP throughput: VQUANT {quant_cycles} cycles for two passes of {n} "
        f"({per_quant:.2f} x n/VL), VSILUMUL {silu_cycles} cycles for one pass of {n} "
        f"({per_silu:.2f} x n/VL)"
    )
    assert quant_cycles < 2 * n / VL + 80, "the one-product passes fell behind VL per cycle"
    assert silu_cycles < 2 * n / VL + 80, "the two-product pass fell behind VL per two cycles"

    # VROPE reads its table row once per row and is compute-bound after that:
    # one head is 32 pairs at VL a chunk, a chunk every two cycles.
    heads = n // HEAD
    d = isa.vrope(vs_src=64, n=n, addr_a=ROW_BASE)
    env.pos = 5
    env.image = image_with(rope_row_bytes(rng), ROW_BASE + env.pos * ROPE_BYTES)
    scatter(env, rng)
    fill_sources(env, rng, d, 1, bits=20)
    rope_cycles = await run_one(env, d, 1)
    assert rope_cycles < n / VL + 40 * heads, (
        f"VROPE took {rope_cycles} cycles for {heads} heads of {HEAD}"
    )

    # VSOFTMAX walks four passes and streams the V-scale records twice, so its
    # cost is reported rather than bounded: at the narrow beat it is the
    # operand stream, not the lanes, that sets the rate.
    env.pos = n - 1
    d = isa.vsoftmax(vs_src=64, vs_dst=4096, n=n, addr_a=ROW_BASE, frac_s=20, sreg_dst=1, length=n)
    env.image = image_with(meta_bytes(rand_scales(rng, n, zeros="none")), ROW_BASE)
    scatter(env, rng)
    fill(env, rng, 0, d.vs_src, rand_scores(rng, n, d.sh0))
    soft_cycles = await run_one(env, d, 1)
    print(
        f"VPU_TOP throughput: VROPE {rope_cycles} cycles for {heads} heads "
        f"({rope_cycles / (n / VL):.2f} x n/VL), VSOFTMAX {soft_cycles} cycles for "
        f"len = n = {n} ({soft_cycles / (n / VL):.2f} x n/VL)"
    )


# --------------------------------------------------------------------------- offsets and tables


@cocotb.test()
async def test_offsets(dut) -> None:
    """Every source, destination and auxiliary offset inside a word, and shared words."""
    env, rng = await setup(dut, 333)
    for so in range(NE):
        for do in range(NE):
            n = 17
            d = isa.vquant(
                vs_src=64 + so,
                vs_dst=4096 + do,
                n=n,
                width=16,
                frac_in=14,
                sreg_dst=1,
            )
            scatter(env, rng)
            fill_sources(env, rng, d, 1, bits=20)
            await run_one(env, d, 1)
    for ao in range(NE):
        for do in range(NE):
            n = 13
            d = isa.vsilumul(
                vs_src=65,
                vs_aux=8192 + ao,
                vs_dst=4096 + do,
                n=n,
                frac_gu=16,
                sh_h=16,
                sreg_dst=1,
            )
            scatter(env, rng)
            fill_sources(env, rng, d, 1, bits=20)
            await run_one(env, d, 1)

    # source and destination ranges that share a VSRAM word, in both directions
    for src, dst, n in ((64, 74, 10), (74, 64, 10), (67, 75, 8), (75, 67, 8), (64, 72, 8)):
        d = isa.vquant(vs_src=src, vs_dst=dst, n=n, width=8, frac_in=10, sreg_dst=2)
        scatter(env, rng)
        fill_sources(env, rng, d, 1, bits=22)
        await run_one(env, d, 1)


@cocotb.test()
async def test_sigmoid_domain(dut) -> None:
    """Every entry of the sigmoid table, on both signs, through VSILUMUL."""
    env, rng = await setup(dut, 444)
    frac_gu = 16
    step = 1 << (frac_gu - 5)  # one table index
    for base in range(0, 512, 128):
        vals = []
        for idx in range(base, base + 128):
            for frac in (0, 1, 127, 128, 255):
                vals.append(idx * step + (frac << (frac_gu - 13)))
        vec = np.array([v if i % 2 else -v for i, v in enumerate(vals)], dtype=np.int64)
        d = isa.vsilumul(
            vs_src=64,
            vs_aux=8192,
            vs_dst=4096,
            n=int(vec.size),
            frac_gu=frac_gu,
            sh_h=2 * frac_gu - 16,
            sreg_dst=1,
        )
        scatter(env, rng)
        fill_sources(
            env,
            rng,
            d,
            1,
            values=vec,
            aux=np.array([rng.randrange(-(1 << 20), 1 << 20) for _ in vec], dtype=np.int64),
        )
        await run_one(env, d, 1)


@cocotb.test()
async def test_tracked_absmax(dut) -> None:
    """A VQUANT reading a tracked absmax produces what the scanning VQUANT produces.

    ``docs/ISA.md`` calls the two paths interchangeable and bit-identical: this
    runs a VRMSNORM (and a VSILUMUL) that tracks its output absmax into an SREG,
    then quantizes that output twice -- once with ``USE_TRACKED`` reading the
    word the hardware wrote, once scanning -- and requires the same int8 vector
    and the same scale.
    """
    env, rng = await setup(dut, 555)
    for trial in range(8):
        n = rng.choice((8, 17, 64, 100))
        src, dst, aux = offsets(rng)
        if trial % 2 == 0:
            d1, gq, addr = rmsnorm_descriptor(rng, n)
            d1 = Descriptor(
                opcode=Opcode.VRMSNORM,
                row_mask=1,
                track_absmax=True,
                addr_a=addr,
                addr_m=d1.addr_m,
                n=n,
                vs_src=src,
                vs_dst=dst,
                sreg_dst=11,
                sh0=d1.sh0,
                sh1=d1.sh1,
                imm32=d1.imm32,
            )
            env.image = image_with(gamma_bytes(gq), addr)
        else:
            d1 = isa.vsilumul(
                vs_src=src,
                vs_aux=aux,
                vs_dst=dst,
                n=n,
                frac_gu=16,
                sh_h=rng.randrange(20, 40),
                sreg_dst=11,
                track_absmax=True,
            )
        scatter(env, rng)
        fill_sources(env, rng, d1, 1, bits=rng.randrange(16, 30))
        await run_one(env, d1, 1)
        tracked = env.sreg[0][11]
        assert tracked == numerics.absmax(env.vsram[0, d1.vs_dst : d1.vs_dst + n]), (
            "the tracked absmax is not the absmax of the output"
        )

        d2 = isa.vquant(
            vs_src=d1.vs_dst,
            vs_dst=8192,
            n=n,
            width=8,
            frac_in=16,
            sreg_dst=12,
            use_tracked=True,
            sreg_src=11,
        )
        state = env.vsram.copy()
        await run_one(env, d2, 1, sreg_pre={(0, 11): tracked})
        with_tracked = env.vsram[0, 8192 : 8192 + n].copy()
        scale_tracked = env.sreg[0][12]

        env.vsram[:] = state
        d3 = isa.vquant(
            vs_src=d1.vs_dst,
            vs_dst=8192,
            n=n,
            width=8,
            frac_in=16,
            sreg_dst=12,
        )
        await run_one(env, d3, 1)
        assert np.array_equal(with_tracked, env.vsram[0, 8192 : 8192 + n]), (
            "USE_TRACKED and the scanning VQUANT disagree on the quantized vector"
        )
        assert scale_tracked == env.sreg[0][12], (
            f"USE_TRACKED wrote scale {scale_tracked:#x}, scanning wrote {env.sreg[0][12]:#x}"
        )
    print(
        f"VPU_TOP compared {TOTALS['elements']} elements over {TOTALS['descriptors']} "
        f"descriptors and {TOTALS['rows']} row executions against isa_sim"
    )
