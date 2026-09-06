"""cocotb tests of qcore_kv_writer: the transposed K byte scatter, the row-major V beats, the
per-token meta record, the row bookkeeping and the capacity and range counters, every write
compared byte for byte against the same descriptor on sw/quettos/isa_sim.py."""

from __future__ import annotations

import os
import random

import cocotb
import numpy as np
import qc_numerics as qn
import qc_stream
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge
from quettos import isa, isa_sim
from quettos.isa import Descriptor, KvwriteFlag, Opcode
from quettos.numerics import SFloat

WB = int(os.environ["QC_WB"])
B_MAX = int(os.environ["QC_B_MAX"])
VSRAM_WORDS = int(os.environ["QC_VSRAM_WORDS"])

ELEMS = VSRAM_WORDS * isa.VSRAM_WORD_ELEMS
HEAD_DIM = isa.HEAD_DIM
META_BYTES = isa.META_BYTES
MAX_CTX = 2048
V_TILES = -(-HEAD_DIM // WB)
ADDR_A = 0
# The K^T region is 64 * MAX_CTX bytes; the V region is ceil(64/WB) * MAX_CTX * WB, which is
# larger once WB passes 64. The meta records follow whichever is bigger.
ADDR_M = max(HEAD_DIM * MAX_CTX, V_TILES * MAX_CTX * WB)
IMAGE_BYTES = ADDR_M + MAX_CTX * META_BYTES
POSITIONS = (0, 1, 63, 64, 2047)
MASK32 = (1 << 32) - 1


def _image(rng: random.Random) -> bytes:
    """A random background so a write the hardware should not make cannot hide in zeros."""
    return bytes(rng.randrange(256) for _ in range(256)) * (IMAGE_BYTES // 256)


def _scale(rng: random.Random) -> SFloat:
    return SFloat(rng.randrange(1 << 15, 1 << 16), rng.randrange(-60, 61))


class Env:
    """Drives the descriptor bundle, models VSRAM port B and the QMEM write port."""

    def __init__(self, dut, image: bytes) -> None:
        self.dut = dut
        self.image = image
        self.mem = bytearray(image)
        self.vsram = np.zeros((B_MAX, ELEMS), dtype=np.int64)
        self.beats: list[tuple[int, int, int]] = []  # (addr, strb, data) accepted, in order
        self.err = 0
        self.done_cycles: list[int] = []
        self.busy_cycles = 0
        self.cycle = 0
        self._rd_prev: tuple[int, int] | None = None
        self._held: tuple[int, int, int] | None = None

    def word(self, row: int, w: int) -> int:
        """VSRAM word ``w`` of row ``row``: element ``j`` at bits ``[32j +: 32]``."""
        out = 0
        for j in range(isa.VSRAM_WORD_ELEMS):
            out |= int(self.vsram[row, w * isa.VSRAM_WORD_ELEMS + j] & MASK32) << (32 * j)
        return out

    def apply(self, addr: int, strb: int, data: int) -> None:
        raw = data.to_bytes(WB, "little")
        for j in range(WB):
            if (strb >> j) & 1:
                assert 0 <= addr + j < IMAGE_BYTES, f"write to 0x{addr + j:x} leaves the image"
                self.mem[addr + j] = raw[j]

    async def step(self, ready: int) -> None:
        """One cycle: sample the ports, pair the beat with ``ready``, model the read latency."""
        dut = self.dut
        valid = int(dut.k_wr_valid.value)
        if valid:
            payload = (
                qc_stream.value(dut.k_wr_addr),
                qc_stream.value(dut.k_wr_strb),
                qc_stream.value(dut.k_wr_data),
            )
            if self._held is not None:
                assert payload == self._held, (
                    f"cycle {self.cycle}: the beat changed while k_wr_valid was high"
                )
            if ready:
                self.beats.append(payload)
                self.apply(*payload)
                self._held = None
            else:
                self._held = payload
        else:
            assert self._held is None, f"cycle {self.cycle}: k_wr_valid dropped before the accept"
        self.err += qc_stream.value(dut.err_bounds_inc)
        if int(dut.done.value):
            self.done_cycles.append(self.cycle)
        if int(dut.busy.value):
            self.busy_cycles += 1
        ven = int(dut.vsb_en.value)
        vaddr = qc_stream.value(dut.vsb_addr)
        vrow = qc_stream.value(dut.vsb_row)
        if ven:
            assert vaddr < VSRAM_WORDS, f"cycle {self.cycle}: read of word {vaddr}"
            assert vrow < B_MAX, f"cycle {self.cycle}: read of bank {vrow}"
        if self._rd_prev is not None:
            dut.vsb_rdata.value = self.word(*self._rd_prev)
        self._rd_prev = (vrow, vaddr) if ven else None
        dut.k_wr_ready.value = ready
        await FallingEdge(dut.clk)
        self.cycle += 1

    async def issue(
        self, d: Descriptor, pos: int, row_en: int, scales: list[SFloat], ready: int = 0
    ) -> None:
        """Present the decoded bundle and pulse ``cmd_valid_kv`` for one cycle."""
        dut = self.dut
        rows = 0
        sx_m = 0
        sx_e = 0
        for r in range(B_MAX):
            if (d.row_mask >> r) & 1 and (row_en >> r) & 1:
                rows |= 1 << r
            sx_m |= scales[r].m << (16 * r)
            sx_e |= (scales[r].e & 0xFF) << (8 * r)
        dut.cmd_kv_transposed.value = int(bool(d.flags & KvwriteFlag.TRANSPOSED))
        dut.cmd_addr_a.value = d.addr_a
        dut.cmd_addr_m.value = d.addr_m
        dut.cmd_k_stride.value = d.k
        dut.cmd_vs_src.value = d.vs_src
        dut.cmd_rows.value = rows
        dut.cmd_sx_m.value = sx_m
        dut.cmd_sx_e.value = sx_e
        dut.cmd_pos.value = pos
        dut.cmd_valid_kv.value = 1
        await self.step(ready)
        dut.cmd_valid_kv.value = 0

    async def run(
        self,
        d: Descriptor,
        pos: int,
        row_en: int,
        scales: list[SFloat],
        ready=lambda c: 1,
        timeout: int = 4000,
    ) -> int:
        """Issue ``d`` and step until ``done``; returns the cycle count of the descriptor."""
        first = self.cycle
        await self.issue(d, pos, row_en, scales, int(bool(ready(self.cycle))))
        seen = len(self.done_cycles)
        for _ in range(timeout):
            await self.step(int(bool(ready(self.cycle))))
            if len(self.done_cycles) > seen:
                break
        assert len(self.done_cycles) == seen + 1, f"no done pulse within {timeout} cycles"
        return self.cycle - first


def reference(env: Env, d: Descriptor, pos: int, row_en: int, scales) -> tuple[bytes, int]:
    """The same descriptor on isa_sim: the image it leaves and the ERR_BOUNDS events it counts."""
    m = isa_sim.Machine(env.image, wb=WB, b_max=B_MAX, vsram_words=VSRAM_WORDS, tables=qn.tables())
    m.csr["POS"] = pos
    m.csr["ROW_EN"] = row_en
    m.vsram[:] = env.vsram
    for r in range(B_MAX):
        m.sreg[r][d.sreg_src] = scales[r]
    isa_sim.execute(m, d)
    return bytes(m.mem), m.err_bounds


def expected_beats(d: Descriptor, pos: int) -> list[tuple[int, int]]:
    """``(address, strobed bytes)`` of every beat one row issues, in order."""
    writes, meta = isa_sim.kv_addresses(d, pos, WB)
    return list(writes) + [(meta, META_BYTES)]


def descriptor(
    rng: random.Random,
    *,
    transposed: bool,
    rows: int = 1,
    vs_src: int | None = None,
    k: int = MAX_CTX,
) -> Descriptor:
    off = rng.randrange(8)
    return Descriptor(
        opcode=Opcode.KVWRITE,
        flags=KvwriteFlag.TRANSPOSED if transposed else 0,
        row_mask=rows,
        addr_a=ADDR_A,
        addr_m=ADDR_M,
        n=HEAD_DIM,
        k=k,
        vs_src=(rng.randrange(0, ELEMS - HEAD_DIM - 8) & ~7) + off if vs_src is None else vs_src,
        sreg_src=rng.randrange(isa.SREG_COUNT),
    )


async def _setup(dut, seed: int) -> tuple[Env, random.Random]:
    rng = random.Random(seed)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for sig in (
        dut.cmd_valid_kv,
        dut.cmd_kv_transposed,
        dut.cmd_addr_a,
        dut.cmd_addr_m,
        dut.cmd_k_stride,
        dut.cmd_vs_src,
        dut.cmd_rows,
        dut.cmd_sx_m,
        dut.cmd_sx_e,
        dut.cmd_pos,
        dut.vsb_rdata,
        dut.k_wr_ready,
    ):
        sig.value = 0
    await qc_stream.reset(dut, dut.rst)
    return Env(dut, _image(rng)), rng


def _fill_window(env: Env, rng: random.Random, d: Descriptor) -> None:
    """Random int32 elements over the window a KVWRITE reads, zero elsewhere: fast and exact."""
    env.vsram[:] = 0
    lo = max(0, d.vs_src - 8)
    hi = min(ELEMS, d.vs_src + HEAD_DIM + 8)
    for r in range(B_MAX):
        env.vsram[r, lo:hi] = np.array(
            [rng.randrange(-(1 << 31), 1 << 31) for _ in range(hi - lo)], dtype=np.int64
        )


async def _one(env: Env, rng: random.Random, d: Descriptor, pos: int, row_en: int, **kw) -> int:
    """Run one descriptor, check the image, the beats and the counters; returns its cycles."""
    _fill_window(env, rng, d)
    scales = [_scale(rng) for _ in range(B_MAX)]
    want_mem, want_err = reference(env, d, pos, row_en, scales)
    env.mem = bytearray(env.image)
    env.beats.clear()
    env.err = 0
    busy_before = env.busy_cycles
    cycles = await env.run(d, pos, row_en, scales, **kw)
    active = [r for r in range(B_MAX) if (d.row_mask >> r) & 1 and (row_en >> r) & 1]
    if pos < d.k:
        want = []
        for _ in active:
            want += expected_beats(d, pos)
        got = [(a, bin(s).count("1")) for a, s, _ in env.beats]
        assert got == want, f"beat sequence {got[:4]}... != {want[:4]}..."
    else:
        assert env.beats == [], "a KVWRITE at or past its capacity wrote"
    assert bytes(env.mem) == want_mem, "the image differs from isa_sim's"
    assert env.err == want_err, f"ERR_BOUNDS {env.err} != {want_err}"
    assert env.busy_cycles - busy_before == cycles - 1, "busy did not cover the descriptor"
    return cycles


@cocotb.test()
async def test_transposed_matches_isa_sim(dut):
    """K^T: 64 single-byte-strobe beats per row at addr_a + (POS/WB)*64*WB + d*WB + POS%WB."""
    env, rng = await _setup(dut, 0x4B31)
    for pos in POSITIONS:
        d = descriptor(rng, transposed=True)
        cycles = await _one(env, rng, d, pos, 1)
        dut._log.info(f"transposed POS={pos} vs_src={d.vs_src}: {cycles} cycles, 65 beats")
        assert cycles <= HEAD_DIM + 20, f"{cycles} cycles for 65 beats"
        assert len(env.beats) == HEAD_DIM + 1, f"{len(env.beats)} beats, expected 65"
        assert all(s == 1 for _, s, _ in env.beats[:HEAD_DIM]), (
            "a K^T beat strobed more than one byte"
        )


@cocotb.test()
async def test_row_major_matches_isa_sim(dut):
    """V: ceil(64/WB) full-width beats per row at addr_a + (t*k + POS)*WB, zero-padded."""
    env, rng = await _setup(dut, 0x4B32)
    for pos in POSITIONS:
        d = descriptor(rng, transposed=False)
        cycles = await _one(env, rng, d, pos, 1)
        dut._log.info(
            f"row-major POS={pos} vs_src={d.vs_src}: {cycles} cycles, {V_TILES + 1} beats"
        )
        assert cycles <= V_TILES + 20, f"{cycles} cycles for {V_TILES + 1} beats"
        assert len(env.beats) == V_TILES + 1, f"{len(env.beats)} beats, expected {V_TILES + 1}"
        assert all(s == (1 << WB) - 1 for _, s, _ in env.beats[:V_TILES]), "a V beat left bytes out"


@cocotb.test()
async def test_meta_record(dut):
    """The last beat of a row is {0, m, e, 0} at addr_m + POS*8 with eight strobes.

    The scales cover both ends of the mantissa and the exponent and the zero scale, which the
    hardware carries through untouched.
    """
    env, rng = await _setup(dut, 0x4B33)
    edges = [SFloat(1 << 15, -128), SFloat(0xFFFF, 127), SFloat(0, 0), _scale(rng), _scale(rng)]
    for i, transposed in enumerate((True, False, True, False, True)):
        for pos in (0, 5, 2047):
            d = descriptor(rng, transposed=transposed)
            _fill_window(env, rng, d)
            scales = [edges[i]] * B_MAX
            env.mem = bytearray(env.image)
            env.beats.clear()
            await env.run(d, pos, 1, scales)
            addr, strb, data = env.beats[-1]
            assert addr == d.addr_m + pos * META_BYTES, "meta address"
            assert strb == (1 << META_BYTES) - 1, f"meta strobes {strb:#x}"
            record = (data & ((1 << 64) - 1)).to_bytes(META_BYTES, "little")
            assert record == qn.meta_bytes(0, scales[0]), f"meta record {record.hex()}"


@cocotb.test()
async def test_unaligned_source(dut):
    """Every element offset of vs_src: the 64 bytes come from the 8 or 9 words that hold them."""
    env, rng = await _setup(dut, 0x4B34)
    for off in range(8):
        for transposed in (True, False):
            d = descriptor(rng, transposed=transposed, vs_src=(1 << 8) + off)
            await _one(env, rng, d, 17, 1)


@cocotb.test()
async def test_two_rows(dut):
    """Participating rows run ascending, each with its own scale; the last row's meta stands."""
    env, rng = await _setup(dut, 0x4B35)
    if B_MAX < 2:
        return
    for transposed in (True, False):
        d = descriptor(rng, transposed=transposed, rows=0b11)
        await _one(env, rng, d, 100, 0b11)
        per_row = (HEAD_DIM if transposed else V_TILES) + 1
        assert len(env.beats) == 2 * per_row, f"{len(env.beats)} beats for two rows"
        assert env.beats[per_row - 1][0] == env.beats[2 * per_row - 1][0], "both rows wrote meta"


@cocotb.test()
async def test_row_enable_and_mask(dut):
    """A row outside row_mask & ROW_EN is skipped and costs no beat and no counter event."""
    env, rng = await _setup(dut, 0x4B36)
    if B_MAX < 2:
        return
    d = descriptor(rng, transposed=True, rows=0b11)
    await _one(env, rng, d, 7, 0b10)  # only row 1 participates
    assert len(env.beats) == HEAD_DIM + 1, "one row should have written"
    d = descriptor(rng, transposed=True, rows=0b10)
    await _one(env, rng, d, 8, 0b01)  # no row participates
    assert env.beats == [] and env.err == 0, "an empty participating set wrote or counted"


@cocotb.test()
async def test_capacity_error(dut):
    """POS at or past the token capacity writes nothing and counts one ERR_BOUNDS per row."""
    env, rng = await _setup(dut, 0x4B37)
    for rows, row_en in ((1, 1), (0b11, 0b11)):
        if rows > 1 and B_MAX < 2:
            continue
        d = descriptor(rng, transposed=True, rows=rows, k=32)
        await _one(env, rng, d, 32, row_en)
        assert env.err == bin(rows & ((1 << B_MAX) - 1)).count("1"), "one count per row"
        d = descriptor(rng, transposed=False, rows=rows, k=32)
        await _one(env, rng, d, 4000, row_en)
        d = descriptor(rng, transposed=True, rows=rows, k=32)
        await _one(env, rng, d, 31, row_en)  # the last position the capacity allows
        assert env.err == 0 and env.beats, "POS = k - 1 is inside the capacity"


@cocotb.test()
async def test_source_range_error(dut):
    """A source range past the end of VSRAM reads zeros there and counts one event per row."""
    env, rng = await _setup(dut, 0x4B38)
    for transposed in (True, False):
        for vs_src in (ELEMS - HEAD_DIM + 1, ELEMS - 8, ELEMS + 24):
            d = descriptor(rng, transposed=transposed, vs_src=vs_src)
            await _one(env, rng, d, 11, 1)
            assert env.err == 1, f"vs_src {vs_src}: ERR_BOUNDS {env.err}"


@cocotb.test()
async def test_write_port_backpressure(dut):
    """Random and long write stalls: every beat still arrives once, in order, unchanged."""
    env, rng = await _setup(dut, 0x4B39)
    patterns = (
        lambda c: c % 2,
        lambda c: c % 7 < 2,
        lambda c: (c // 16) % 2,
        lambda c: rng.random() < 0.3,
    )
    for i, pattern in enumerate(patterns):
        d = descriptor(rng, transposed=bool(i % 2))
        await _one(env, rng, d, 33 * i + 1, 1, ready=pattern, timeout=20000)


@cocotb.test()
async def test_back_to_back(dut):
    """Descriptors issued one after another: one done pulse each, busy low in between."""
    env, rng = await _setup(dut, 0x4B3A)
    for i in range(4):
        d = descriptor(rng, transposed=bool(i % 2))
        _fill_window(env, rng, d)
        scales = [_scale(rng) for _ in range(B_MAX)]
        want_mem, _ = reference(env, d, 40 + i, 1, scales)
        env.mem = bytearray(env.image)
        env.beats.clear()
        seen = len(env.done_cycles)
        await env.run(d, 40 + i, 1, scales)
        assert len(env.done_cycles) == seen + 1, "more than one done pulse"
        assert bytes(env.mem) == want_mem, f"descriptor {i} image"
        assert int(dut.busy.value) == 0, "busy stayed high after done"


@cocotb.test()
async def test_idle_is_quiet(dut):
    """Without a descriptor the module reads nothing, writes nothing and stays idle."""
    env, _ = await _setup(dut, 0x4B3B)
    for _ in range(32):
        await env.step(1)
    assert env.beats == [] and env.err == 0 and env.done_cycles == []
    assert env.busy_cycles == 0, "busy without a descriptor"
