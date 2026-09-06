"""cocotb tests of qcore_requant: bit-exact requantization against numerics.requant and
numerics.requant_rows (zero scales, shift clamps, every saturation stage, negative
extremes, bias and accumulate adds), the VSRAM word writes and read-modify-write on
port B, ARGMAX (strict greater, ties), DUMP beats, the tracked absmax, partial tiles,
two rows, EMBED, empty descriptors, bounds, event counts and the drain timing."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import cocotb
import numpy as np
import qc_numerics as qn
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly
from qc_numerics import SFLOAT_ONE, SFLOAT_ZERO, SFloat, Stats
from quettos import numerics

OP_GEMV = 0x10
OP_EMBED = 0x11
OUT_VSRAM, OUT_ARGMAX, OUT_ARGMAX_DUMP, OUT_VSRAM_DUMP = 0, 1, 2, 3
VSRAM_MODES = (OUT_VSRAM, OUT_VSRAM_DUMP)
ARGMAX_MODES = (OUT_ARGMAX, OUT_ARGMAX_DUMP)
DUMP_MODES = (OUT_ARGMAX_DUMP, OUT_VSRAM_DUMP)
MASK32 = (1 << 32) - 1
MASK40 = (1 << 40) - 1
INT32_MIN = -(1 << 31)


def _u(sig) -> int:
    return sig.value.to_unsigned()


def _b(sig) -> int:
    return int(sig.value)


# --------------------------------------------------------------------------- geometry


@dataclass(frozen=True)
class Geometry:
    wb: int
    b_max: int
    acc_w: int
    words: int

    @property
    def epb(self) -> int:
        return self.wb // 4

    @property
    def elems(self) -> int:
        return self.words * 8

    @classmethod
    def of(cls, dut) -> Geometry:
        wb = len(dut.d_wr_strb)
        b_max = len(dut.cmd_rows)
        return cls(wb, b_max, len(dut.acc_flat) // (wb * b_max), 1 << len(dut.vsb_addr))


# --------------------------------------------------------------------------- descriptors


@dataclass
class Desc:
    """One GEMV / EMBED descriptor as the requant sees it, with the data the rows would produce."""

    op: int = OP_GEMV
    out_mode: int = OUT_VSRAM
    accumulate: bool = False
    unit_meta: bool = False
    track_absmax: bool = False
    n: int = 0
    vs_dst: int = 0
    sreg_dst: int = 0
    sh0: int = 0
    sh1: int = 0
    addr_c: int = 0
    rows: int = 1
    sx: list[SFloat] = field(default_factory=list)  # per physical row
    acc: dict[int, list[int]] = field(default_factory=dict)  # row -> N accumulators
    meta: list[tuple[int, SFloat]] = field(default_factory=list)  # (bias_q, Sw) per channel
    emb: tuple[int, SFloat] = (0, SFLOAT_ONE)  # the EMBED record

    def part_rows(self, b_max: int) -> list[int]:
        return [r for r in range(b_max) if (self.rows >> r) & 1]

    def tiles(self, wb: int) -> int:
        return -(-self.n // wb)

    @property
    def is_embed(self) -> bool:
        return self.op == OP_EMBED

    def sw_bias(self, c: int) -> tuple[SFloat, int]:
        """(Sw, bias_q) of channel c as the hardware applies them."""
        if self.unit_meta:
            return SFLOAT_ONE, 0
        if self.is_embed:
            return self.emb[1], 0
        bias, sw = self.meta[c]
        return sw, bias

    def records(self, wb: int) -> list[int]:
        """The meta side-stream records the stream controller pushes: nvalid per tile."""
        if self.unit_meta:
            return []
        if self.is_embed:
            return [qn.meta_record56(self.emb[0], self.emb[1])]
        return [
            qn.meta_record56(*self.meta[t * wb + j])
            for t in range(self.tiles(wb))
            for j in range(min(wb, self.n - t * wb))
        ]


# --------------------------------------------------------------------------- reference model


def _elem_get(word: int, slot: int) -> int:
    return qn.to_signed((word >> (32 * slot)) & MASK32, 32)


def _elem_set(word: int, slot: int, value: int) -> int:
    return (word & ~(MASK32 << (32 * slot))) | ((value & MASK32) << (32 * slot))


def vs_read(bank: list[int], start: int, count: int, g: Geometry) -> tuple[list[int], int]:
    """isa_sim.Machine.vs_read: elements past the end read 0 and count one bounds error."""
    err = 1 if start + count > g.elems else 0
    out = []
    for e in range(start, start + count):
        out.append(_elem_get(bank[e >> 3], e & 7) if e < g.elems else 0)
    return out, err


def vs_write(bank: list[int], start: int, values: list[int], g: Geometry) -> int:
    err = 1 if start + len(values) > g.elems else 0
    for i, v in enumerate(values):
        e = start + i
        if e < g.elems:
            bank[e >> 3] = _elem_set(bank[e >> 3], e & 7, v)
    return err


@dataclass
class Expect:
    y: dict[int, list[int]]
    vsram: dict[int, list[int]]
    argmax: list[tuple[int, int, int]]  # (row, idx, val), rows ascending
    sreg: list[tuple[int, int, int]]  # (row, sreg_dst, absmax)
    beats: list[tuple[int, int, bytes]]  # (addr, strb, strobed bytes) in order
    stats: Stats
    err_bounds: int


def reference(d: Desc, g: Geometry, vsram: dict[int, list[int]]) -> Expect:
    """numerics.requant per element with isa_sim's output rules, on a copy of the VSRAM."""
    vs = {r: list(w) for r, w in vsram.items()}
    stats = Stats()
    err_bounds = 0
    s1 = d.sh0
    rows = d.part_rows(g.b_max)
    ys: dict[int, list[int]] = {}
    argmax: list[tuple[int, int, int]] = []
    sreg: list[tuple[int, int, int]] = []
    beats: list[tuple[int, int, bytes]] = []
    if d.n == 0 or not rows:
        return Expect(ys, vs, argmax, sreg, beats, stats, err_bounds)
    if s1 > 63:
        stats.err_shift += d.n * len(rows)
        s1 = 63
    for r in rows:
        old = None
        if d.accumulate:
            old, err = vs_read(vs[r], d.vs_dst, d.n, g)
            err_bounds += err
        y = []
        for c in range(d.n):
            sw, bias = d.sw_bias(c)
            o = None if old is None else old[c]
            y.append(qn.requant(d.acc[r][c], sw, d.sx[r], s1, d.sh1, bias, o, stats))
        ys[r] = y
        if d.out_mode in VSRAM_MODES:
            err_bounds += vs_write(vs[r], d.vs_dst, y, g)
        if d.out_mode in ARGMAX_MODES:
            idx = numerics.argmax(np.array(y, dtype=np.int64))
            argmax.append((r, idx, y[idx]))
        if d.out_mode in DUMP_MODES:
            base = d.addr_c + r * 4 * d.n
            raw = np.array(y, dtype="<i4").tobytes()
            for b in range(-(-4 * d.n // g.wb)):
                chunk = raw[b * g.wb : (b + 1) * g.wb]
                beats.append((base + b * g.wb, (1 << len(chunk)) - 1, chunk))
        if d.track_absmax:
            sreg.append((r, d.sreg_dst, numerics.absmax(np.array(y, dtype=np.int64))))
    return Expect(ys, vs, argmax, sreg, beats, stats, err_bounds)


def reference_rows(d: Desc, g: Geometry, vsram: dict[int, list[int]]) -> tuple[dict, Stats]:
    """The same outputs through numerics.requant_rows (block form), for the cross-check."""
    stats = Stats()
    s1 = d.sh0
    rows = d.part_rows(g.b_max)
    if s1 > 63:
        stats.err_shift += d.n * len(rows)
        s1 = 63
    sw = [d.sw_bias(c) for c in range(d.n)]
    sw_m = np.array([s.m for s, _ in sw], dtype=np.int64)
    sw_e = np.array([s.e for s, _ in sw], dtype=np.int64)
    bias = np.array([b for _, b in sw], dtype=np.int64)
    out = {}
    for r in rows:
        old = None
        if d.accumulate:
            old = np.array(vs_read(vsram[r], d.vs_dst, d.n, g)[0], dtype=np.int64)[None, :]
        y = numerics.requant_rows(
            np.array(d.acc[r], dtype=np.int64)[None, :],
            sw_m,
            sw_e,
            np.array([d.sx[r].m], dtype=np.int64),
            np.array([d.sx[r].e], dtype=np.int64),
            s1,
            d.sh1,
            bias_q=bias,
            old=old,
            stats=stats,
        )
        out[r] = [int(v) for v in y[0]]
    return out, stats


# --------------------------------------------------------------------------- environment


@dataclass
class Sample:
    acc_ready: int = 0
    meta_ready: int = 0
    vsb_en: int = 0
    vsb_we: int = 0
    vsb_addr: int = 0
    vsb_row: int = 0
    vsb_wdata: int = 0
    d_wr_valid: int = 0
    d_wr_addr: int = 0
    d_wr_strb: int = 0
    d_wr_data: int = 0
    sreg_wr_en: int = 0
    sreg: tuple[int, int, int] = (0, 0, 0)
    argmax_we: int = 0
    argmax: tuple[int, int] = (0, 0)
    sat_inc: int = 0
    err_shift_inc: int = 0
    err_bounds_inc: int = 0
    done: int = 0
    busy: int = 0


@dataclass
class Result:
    vsram: dict[int, list[int]]
    argmax: list[tuple[int, int, int]]
    sreg: list[tuple[int, int, int]]
    beats: list[tuple[int, int, int]]  # (addr, strb, data)
    sat: int
    err_shift: int
    err_bounds: int
    cycles: int
    handshakes: list[int]
    first_write: int | None
    meta_left: int


class Env:
    """Drives the issue bundle, the accumulator handoff, the meta stream, VSRAM port B
    (per-bank READ_FIRST words behind an unregistered row-select mux) and the dump
    port; samples every output in the read-only phase before each rising edge."""

    def __init__(self, dut, g: Geometry, rng: random.Random) -> None:
        self.dut, self.g, self.rng = dut, g, rng
        self.vsram = {r: [rng.getrandbits(256) for _ in range(g.words)] for r in range(g.b_max)}
        self.rd_b = {r: 0 for r in range(g.b_max)}
        self.meta_gap = 0.0  # probability of holding meta_valid low with a record waiting
        self.tile_gap = (0, 0)  # cycles between a tile's acceptance and the next offer
        self.ready_p = 1.0  # probability of d_wr_ready per cycle
        self.cycle = 0

    def idle(self) -> None:
        d = self.dut
        d.cmd_valid_gemv.value = 0
        d.acc_valid.value = 0
        d.meta_valid.value = 0
        d.d_wr_ready.value = 1
        d.vsb_rdata.value = 0
        for name in (
            "cmd_op",
            "cmd_out_mode",
            "cmd_accumulate",
            "cmd_unit_meta",
            "cmd_track_absmax",
            "cmd_n",
            "cmd_vs_dst",
            "cmd_sreg_dst",
            "cmd_sh0",
            "cmd_sh1",
            "cmd_imm32",
            "cmd_rows",
            "cmd_sx_m",
            "cmd_sx_e",
            "acc_flat",
            "acc_tile",
            "acc_nvalid",
            "acc_last",
            "meta_data",
        ):
            getattr(d, name).value = 0

    def sample(self) -> Sample:
        d = self.dut
        s = Sample()
        s.acc_ready = _b(d.acc_ready)
        s.meta_ready = _b(d.meta_ready)
        s.vsb_en = _b(d.vsb_en)
        s.vsb_we = _u(d.vsb_we)
        if s.vsb_en or s.vsb_we:
            s.vsb_addr = _u(d.vsb_addr)
            s.vsb_row = _u(d.vsb_row)
            if s.vsb_we:
                s.vsb_wdata = _u(d.vsb_wdata)
        else:
            s.vsb_row = _u(d.vsb_row)
        s.d_wr_valid = _b(d.d_wr_valid)
        if s.d_wr_valid:
            s.d_wr_addr = _u(d.d_wr_addr)
            s.d_wr_strb = _u(d.d_wr_strb)
            s.d_wr_data = _u(d.d_wr_data)
        s.sreg_wr_en = _b(d.sreg_wr_en)
        if s.sreg_wr_en:
            s.sreg = (_u(d.sreg_wr_row), _u(d.sreg_wr_idx), _u(d.sreg_wr_data))
        s.argmax_we = _b(d.argmax_we)
        if s.argmax_we:
            s.argmax = (_u(d.argmax_tok), qn.to_signed(_u(d.argmax_val), 32))
        s.sat_inc = _u(d.sat_inc)
        s.err_shift_inc = _u(d.err_shift_inc)
        s.err_bounds_inc = _u(d.err_bounds_inc)
        s.done = _b(d.done)
        s.busy = _b(d.busy)
        return s

    def _drive_cmd(self, d: Desc) -> None:
        g, dut = self.g, self.dut
        dut.cmd_op.value = d.op
        dut.cmd_out_mode.value = d.out_mode
        dut.cmd_accumulate.value = int(d.accumulate)
        dut.cmd_unit_meta.value = int(d.unit_meta)
        dut.cmd_track_absmax.value = int(d.track_absmax)
        dut.cmd_n.value = d.n
        dut.cmd_vs_dst.value = d.vs_dst
        dut.cmd_sreg_dst.value = d.sreg_dst
        dut.cmd_sh0.value = d.sh0
        dut.cmd_sh1.value = qn.from_signed(d.sh1, 8)
        dut.cmd_imm32.value = d.addr_c
        dut.cmd_rows.value = d.rows & ((1 << g.b_max) - 1)
        sxm = sxe = 0
        for r in range(g.b_max):
            s = d.sx[r] if r < len(d.sx) else SFLOAT_ZERO
            sxm |= s.m << (16 * r)
            sxe |= qn.from_signed(s.e, 8) << (8 * r)
        dut.cmd_sx_m.value = sxm
        dut.cmd_sx_e.value = sxe

    def _tiles(self, d: Desc) -> list[tuple[int, int, int, int]]:
        """(tile, nvalid, last, acc_flat) per tile; lanes the drain must ignore hold noise."""
        g, rng = self.g, self.rng
        rows = set(d.part_rows(g.b_max))
        out = []
        for t in range(d.tiles(g.wb)):
            nvalid = min(g.wb, d.n - t * g.wb)
            flat = 0
            for r in range(g.b_max):
                for j in range(g.wb):
                    if r in rows and j < nvalid:
                        v = d.acc[r][t * g.wb + j] & MASK40
                    else:
                        v = rng.getrandbits(g.acc_w)
                    flat |= v << ((r * g.wb + j) * g.acc_w)
            out.append((t, nvalid, int(t == d.tiles(g.wb) - 1), flat))
        return out

    async def run(self, d: Desc, max_cycles: int = 100000) -> Result:
        dut, g, rng = self.dut, self.g, self.rng
        clk = dut.clk
        meta_q = d.records(g.wb)
        tiles = self._tiles(d)
        tile_idx = 0
        pend = None
        draining = False  # the DUT still reads the tile it accepted: hold acc_flat
        tile_wait = 0
        drv_meta = drv_acc = drv_ready = 0
        res = Result(self.vsram, [], [], [], 0, 0, 0, 0, [], None, 0)
        last_read: tuple[int, int] | None = None
        start_cycle = self.cycle

        await FallingEdge(clk)
        self._drive_cmd(d)
        dut.cmd_valid_gemv.value = 1
        dut.acc_valid.value = 0
        dut.meta_valid.value = 0
        dut.d_wr_ready.value = 0
        await ReadOnly()
        s = self.sample()
        assert s.busy == 0 and s.done == 0, "requant busy before issue"
        while True:
            await FallingEdge(clk)
            self.cycle += 1
            dut.cmd_valid_gemv.value = 0
            # effects of the rising edge that just passed
            if drv_meta and s.meta_ready:
                meta_q.pop(0)
            if drv_acc and s.acc_ready:
                res.handshakes.append(self.cycle - 1)
                pend = None
                draining = True
            elif draining and s.acc_ready:
                # the final read of the tile: a row swaps its sets the cycle after
                draining = False
                tile_wait = rng.randint(*self.tile_gap)
            if last_read is not None:
                assert s.vsb_row == last_read[1], (
                    f"cycle {self.cycle}: vsb_row {s.vsb_row} changed the cycle after a read "
                    f"of bank {last_read[1]}"
                )
                last_read = None
            if s.vsb_en or s.vsb_we:
                bank = s.vsb_row
                assert bank < g.b_max, f"vsb_row {bank} outside the rows"
                assert not (s.vsb_en and s.vsb_we), "port B read and write in one cycle"
                if s.vsb_en:
                    self.rd_b[bank] = self.vsram[bank][s.vsb_addr]
                    last_read = (self.cycle, bank)
                else:
                    w = self.vsram[bank][s.vsb_addr]
                    for i in range(8):
                        if (s.vsb_we >> i) & 1:
                            w = _elem_set(w, i, (s.vsb_wdata >> (32 * i)) & MASK32)
                    self.vsram[bank][s.vsb_addr] = w
                    if res.first_write is None:
                        res.first_write = self.cycle - 1
            if s.d_wr_valid and drv_ready:
                res.beats.append((s.d_wr_addr, s.d_wr_strb, s.d_wr_data))
            if s.sreg_wr_en:
                res.sreg.append(s.sreg)
            if s.argmax_we:
                res.argmax.append((s.argmax[0], s.argmax[1]))
            res.sat += s.sat_inc
            res.err_shift += s.err_shift_inc
            res.err_bounds += s.err_bounds_inc
            if s.done:
                assert s.busy == 0, "busy high in the done cycle"
                break
            assert self.cycle - start_cycle < max_cycles, "descriptor did not finish"
            # inputs for the next cycle
            dut.vsb_rdata.value = self.rd_b[_u(dut.vsb_row) % g.b_max]
            if meta_q and rng.random() >= self.meta_gap:
                drv_meta = 1
                dut.meta_data.value = meta_q[0]
            else:
                drv_meta = 0
                dut.meta_data.value = rng.getrandbits(56)
            dut.meta_valid.value = drv_meta
            if pend is None and not draining and tile_idx < len(tiles):
                if tile_wait == 0:
                    pend = tiles[tile_idx]
                    tile_idx += 1
                    dut.acc_tile.value = pend[0]
                    dut.acc_nvalid.value = pend[1]
                    dut.acc_last.value = pend[2]
                    dut.acc_flat.value = pend[3]
                else:
                    tile_wait -= 1
            drv_acc = int(pend is not None)
            dut.acc_valid.value = drv_acc
            drv_ready = int(rng.random() < self.ready_p)
            dut.d_wr_ready.value = drv_ready
            await ReadOnly()
            s = self.sample()
        res.cycles = self.cycle - start_cycle
        res.meta_left = len(meta_q)
        await FallingEdge(clk)
        dut.acc_valid.value = 0
        dut.meta_valid.value = 0
        return res


# --------------------------------------------------------------------------- checks


def check(d: Desc, g: Geometry, exp: Expect, res: Result, tag: str) -> None:
    assert res.meta_left == 0, f"{tag}: {res.meta_left} meta records not consumed"
    for r in range(g.b_max):
        got, want = res.vsram[r], exp.vsram[r]
        bad = [w for w in range(g.words) if got[w] != want[w]]
        assert not bad, (
            f"{tag}: VSRAM bank {r} differs at words {bad[:8]}: "
            f"got {got[bad[0]]:#066x} want {want[bad[0]]:#066x}"
        )
    assert res.argmax == [(i, v) for _, i, v in exp.argmax], (
        f"{tag}: ARGMAX writes {res.argmax} != {[(i, v) for _, i, v in exp.argmax]}"
    )
    assert res.sreg == exp.sreg, f"{tag}: SREG writes {res.sreg} != {exp.sreg}"
    got_beats = []
    for addr, strb, data in res.beats:
        raw = data.to_bytes(g.wb, "little")
        n = strb.bit_length()
        assert strb == (1 << n) - 1 and n % 4 == 0, f"{tag}: dump strobe {strb:#x} not a prefix"
        got_beats.append((addr, strb, raw[:n]))
    # rows drain tile by tile, so beats of different rows interleave; within a row
    # they ascend (each row's region is a contiguous address range)
    assert sorted(got_beats) == sorted(exp.beats), (
        f"{tag}: dump beats differ ({len(got_beats)} vs {len(exp.beats)}): "
        f"got {[(a, s, b.hex()) for a, s, b in sorted(got_beats)[:4]]} "
        f"want {[(a, s, b.hex()) for a, s, b in sorted(exp.beats)[:4]]}"
    )
    for r in d.part_rows(g.b_max):
        lo, hi = d.addr_c + r * 4 * d.n, d.addr_c + (r + 1) * 4 * d.n
        addrs = [a for a, _, _ in got_beats if lo <= a < hi]
        assert addrs == sorted(addrs), f"{tag}: row {r} beats out of order: {addrs}"
    assert res.sat == exp.stats.sat, f"{tag}: SAT_REQ {res.sat} != {exp.stats.sat}"
    assert res.err_shift == exp.stats.err_shift, (
        f"{tag}: ERR_SHIFT {res.err_shift} != {exp.stats.err_shift}"
    )
    assert res.err_bounds == exp.err_bounds, (
        f"{tag}: ERR_BOUNDS {res.err_bounds} != {exp.err_bounds}"
    )


# --------------------------------------------------------------------------- generators


def sfloat(rng: random.Random, e: int | None = None, zero_p: float = 0.0) -> SFloat:
    if rng.random() < zero_p:
        return SFLOAT_ZERO
    if e is None:
        e = rng.randrange(-128, 128)
    return SFloat(rng.randrange(1 << 15, 1 << 16), max(-128, min(127, e)))


def rand_acc(rng: random.Random) -> int:
    k = rng.random()
    if k < 0.25:
        return rng.randrange(-(1 << 39), 1 << 39)
    if k < 0.35:
        return rng.choice([-(1 << 39), (1 << 39) - 1, 0, -1, 1, 1 << 38, -(1 << 38)])
    if k < 0.6:
        return rng.randrange(-(1 << 32), 1 << 32)
    if k < 0.8:
        return rng.randrange(-(1 << 20), 1 << 20)
    return rng.randrange(-(1 << 8), 1 << 8)


def rand_bias(rng: random.Random) -> int:
    k = rng.random()
    if k < 0.4:
        return 0
    if k < 0.6:
        return rng.choice([INT32_MIN, MASK32 >> 1, INT32_MIN + 1, -1, 1])
    if k < 0.8:
        return rng.randrange(INT32_MIN, 1 << 31)
    return rng.randrange(-(1 << 16), 1 << 16)


def rand_s_target(rng: random.Random) -> int:
    k = rng.random()
    if k < 0.75:
        return rng.randrange(0, 64)
    if k < 0.85:
        return rng.choice([0, 63, 1, 62])
    if k < 0.93:
        return rng.randrange(-40, 0)
    return rng.randrange(64, 100)


def rand_sh0(rng: random.Random) -> int:
    k = rng.random()
    if k < 0.6:
        return rng.randrange(8, 25)
    if k < 0.85:
        return rng.randrange(0, 64)
    if k < 0.92:
        return rng.choice([0, 63, 64, 255])
    return rng.randrange(64, 256)


def rand_n(rng: random.Random, g: Geometry, max_tiles: int = 5) -> int:
    k = rng.random()
    if k < 0.3:
        return rng.randrange(1, g.wb * max_tiles + 1)
    if k < 0.5:
        return g.wb * rng.randrange(1, max_tiles + 1)
    if k < 0.7:
        return rng.randrange(1, 2 * g.wb)
    if k < 0.85:
        return rng.choice([1, 2, 7, 8, 9, g.epb, g.epb + 1, g.wb - 1, g.wb + 1, 2 * g.wb - 1])
    return rng.randrange(1, 9)


def rand_vs_dst(rng: random.Random, g: Geometry, n: int) -> int:
    k = rng.random()
    if k < 0.8:
        return 8 * rng.randrange(0, max(1, (g.elems - n) // 8 + 1))
    if k < 0.9:
        return 8 * rng.randrange(max(0, (g.elems - n) // 8 - 4), g.words)
    return 8 * rng.randrange(0, g.words)


def gemv_desc(rng: random.Random, g: Geometry, **fixed) -> Desc:
    n = fixed.pop("n", None) or rand_n(rng, g, fixed.pop("max_tiles", 5))
    rows = fixed.pop("rows", None) or rng.randrange(1, 1 << g.b_max)
    d = Desc(
        op=OP_GEMV,
        out_mode=fixed.pop("out_mode", rng.randrange(4)),
        accumulate=fixed.pop("accumulate", rng.random() < 0.3),
        unit_meta=fixed.pop("unit_meta", rng.random() < 0.15),
        track_absmax=fixed.pop("track_absmax", rng.random() < 0.5),
        n=n,
        vs_dst=fixed.pop("vs_dst", None) or rand_vs_dst(rng, g, n),
        sreg_dst=fixed.pop("sreg_dst", rng.randrange(256)),
        sh0=fixed.pop("sh0", rand_sh0(rng)),
        sh1=fixed.pop("sh1", rng.randrange(-128, 128)),
        addr_c=fixed.pop("addr_c", 64 * rng.randrange(1, 1 << 20)),
        rows=rows,
    )
    assert not fixed, f"unknown fields {fixed}"
    if d.out_mode not in DUMP_MODES:
        d.addr_c = 0
    sxe = rng.randrange(-128, 128)
    d.sx = [sfloat(rng, sxe + rng.randrange(-2, 3), zero_p=0.06) for _ in range(g.b_max)]
    tiles_n = d.tiles(g.wb) * g.wb
    d.meta = []
    for _ in range(tiles_n):
        s_t = rand_s_target(rng)
        swe = d.sh1 - sxe - s_t
        d.meta.append((rand_bias(rng), sfloat(rng, swe, zero_p=0.05)))
    d.acc = {r: [rand_acc(rng) for _ in range(n)] for r in d.part_rows(g.b_max)}
    return d


def embed_desc(rng: random.Random, g: Geometry, k: int | None = None, **fixed) -> Desc:
    k = k or rand_n(rng, g)
    s1 = rng.randrange(8, 25)
    frac_x = rng.randrange(0, 31)
    d = Desc(
        op=OP_EMBED,
        out_mode=fixed.pop("out_mode", rng.randrange(4)),
        track_absmax=fixed.pop("track_absmax", rng.random() < 0.5),
        n=k,
        vs_dst=fixed.pop("vs_dst", None) or rand_vs_dst(rng, g, k),
        sreg_dst=rng.randrange(32),
        sh0=s1,
        sh1=numerics.sbias_for(frac_x, s1, 24),
        addr_c=64 * rng.randrange(1, 1 << 20),
        rows=fixed.pop("rows", None) or rng.randrange(1, 1 << g.b_max),
    )
    assert not fixed, f"unknown fields {fixed}"
    if d.out_mode not in DUMP_MODES:
        d.addr_c = 0
    d.sx = [SFLOAT_ONE] * g.b_max
    d.emb = (rand_bias(rng), sfloat(rng, rng.randrange(-40, 10), zero_p=0.05))
    d.acc = {r: [rng.randrange(-128, 128) << 24 for _ in range(k)] for r in d.part_rows(g.b_max)}
    return d


# --------------------------------------------------------------------------- tests


async def setup(dut, seed: int) -> tuple[Env, Geometry, random.Random]:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    g = Geometry.of(dut)
    rng = random.Random(seed)
    env = Env(dut, g, rng)
    env.idle()
    dut.rst.value = 1
    await FallingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    await FallingEdge(dut.clk)
    return env, g, rng


async def run_and_check(env: Env, d: Desc, tag: str) -> Result:
    before = {r: list(w) for r, w in env.vsram.items()}
    exp = reference(d, env.g, before)
    res = await env.run(d)
    check(d, env.g, exp, res, tag)
    return res


@cocotb.test()
async def test_directed_corners(dut):
    """S = 0 / 63 / clamped, zero scales, every saturation stage, negative extremes, argmax ties."""
    env, g, rng = await setup(dut, 0x1001)
    wb = g.wb

    def base(**kw) -> Desc:
        d = gemv_desc(rng, g, n=kw.pop("n", wb), rows=1, **kw)
        d.sx = [SFloat(1 << 15, 0)] * g.b_max
        d.sh0 = 16
        return d

    # S exactly 0, 63, -1 (clamped, counted) and 64 (clamped, counted) per channel
    d = base(unit_meta=False, accumulate=False, out_mode=OUT_VSRAM)
    d.sh1 = 0
    targets = [0, 63, -1, 64, 1, 62, -100, 127]
    d.meta = [(0, SFloat(0xFFFF, -targets[i % len(targets)])) for i in range(wb)]
    d.acc[0] = [rng.choice([(1 << 39) - 1, -(1 << 39), 12345, -1]) for _ in range(wb)]
    await run_and_check(env, d, "S corners")

    # zero scales: Sw_m == 0 channels and an Sx_m == 0 row give 0 with no event, bias still adds
    d = base(unit_meta=False, accumulate=True, out_mode=OUT_VSRAM_DUMP)
    d.meta = [
        (rand_bias(rng), SFLOAT_ZERO if i % 3 == 0 else SFloat(0x8000, -20)) for i in range(wb)
    ]
    d.sh0 = 200  # counted once per element, on zero-scale channels too
    await run_and_check(env, d, "Sw zero")
    d = base(unit_meta=False, accumulate=False, out_mode=OUT_ARGMAX_DUMP)
    d.sx = [SFLOAT_ZERO] * g.b_max
    d.meta = [(rand_bias(rng), SFloat(0xFFFF, rng.randrange(-100, 100))) for _ in range(wb)]
    await run_and_check(env, d, "Sx zero")

    # saturation at each stage: stage 1 (s1 = 0, full-scale acc), stage 2 (S small),
    # bias add and accumulate add at the int32 limits
    d = base(unit_meta=False, accumulate=True, out_mode=OUT_VSRAM)
    d.sh0 = 0
    d.sh1 = 0
    d.meta = [(rng.choice([INT32_MIN, MASK32 >> 1]), SFloat(0xFFFF, 0)) for _ in range(wb)]
    d.acc[0] = [rng.choice([(1 << 39) - 1, -(1 << 39), 1 << 38, -(1 << 38)]) for _ in range(wb)]
    for r in range(g.b_max):
        for w in range(d.vs_dst // 8, d.vs_dst // 8 + wb // 8 + 1):
            env.vsram[r][w] = rng.choice([0x7FFFFFFF, 0x80000000]) * ((1 << 256) - 1) // 0xFFFFFFFF
    await run_and_check(env, d, "saturations")

    # negative extremes through a unit scale and s1 = 63
    d = base(unit_meta=True, accumulate=False, out_mode=OUT_VSRAM_DUMP)
    d.sh0 = 63
    d.sh1 = 127
    d.acc[0] = [-(1 << 39)] * (wb // 2) + [(1 << 39) - 1] * (wb - wb // 2)
    await run_and_check(env, d, "extremes s1=63")
    d = base(unit_meta=True, out_mode=OUT_ARGMAX)
    d.sh0 = 0
    d.sh1 = -128
    d.acc[0] = [-(1 << 39) + i for i in range(wb)]
    await run_and_check(env, d, "extremes S=-128")

    # argmax: ties resolve to the lowest index; an all-equal vector picks 0; -2^31 outputs
    d = base(unit_meta=True, out_mode=OUT_ARGMAX_DUMP, n=2 * wb + 3)
    d.sh1 = -24
    d.acc[0] = [7 << 24] * d.n
    d.acc[0][wb + 1] = 9 << 24
    d.acc[0][2 * wb] = 9 << 24
    await run_and_check(env, d, "argmax tie")
    d = base(unit_meta=True, out_mode=OUT_ARGMAX, n=wb + 5)
    d.sh1 = 0
    d.acc[0] = [-(1 << 39)] * d.n
    await run_and_check(env, d, "argmax all -2^31")
    d = base(unit_meta=True, out_mode=OUT_ARGMAX, n=5)
    d.acc[0] = [0] * 5
    await run_and_check(env, d, "argmax zeros")


@cocotb.test()
async def test_random_gemv(dut):
    """Random GEMV descriptors over every mode, accumulate, unit_meta, rows and partial tiles,
    with random gaps on the meta stream, the tiles and the dump port."""
    env, g, rng = await setup(dut, 0x2002)
    for i in range(500):
        env.meta_gap = rng.choice([0.0, 0.0, 0.2, 0.6, 0.9])
        env.tile_gap = rng.choice([(0, 0), (0, 3), (1, 20)])
        env.ready_p = rng.choice([1.0, 1.0, 0.7, 0.3, 0.05])
        d = gemv_desc(rng, g, max_tiles=rng.choice([2, 5, 9]))
        await run_and_check(env, d, f"gemv {i}: {d.n} ch rows {d.rows:#x} mode {d.out_mode}")


@cocotb.test()
async def test_random_embed(dut):
    """EMBED descriptors: one record for every row, arriving before or after the first tile."""
    env, g, rng = await setup(dut, 0x3003)
    for i in range(120):
        env.meta_gap = rng.choice([0.0, 0.5, 0.95])
        env.tile_gap = rng.choice([(0, 0), (0, 4)])
        env.ready_p = rng.choice([1.0, 0.5])
        d = embed_desc(rng, g)
        await run_and_check(env, d, f"embed {i}: k {d.n} rows {d.rows:#x} mode {d.out_mode}")


@cocotb.test()
async def test_dump_beats_and_rows(dut):
    """DUMP beat addresses, strobes and contents for partial beats and every row combination."""
    env, g, rng = await setup(dut, 0x4004)
    env.ready_p = 0.5
    for rows in range(1, 1 << g.b_max):
        for n in (1, 2, g.epb - 1, g.epb, g.epb + 1, g.wb - 1, g.wb, g.wb + 1, 2 * g.wb + 5):
            for mode in DUMP_MODES:
                d = gemv_desc(rng, g, n=n, rows=rows, out_mode=mode, accumulate=rng.random() < 0.5)
                await run_and_check(env, d, f"dump rows {rows:#x} n {n} mode {mode}")


@cocotb.test()
async def test_bounds(dut):
    """Ranges past the VSRAM: dropped words, zero old values, one count per operand and row."""
    env, g, rng = await setup(dut, 0x5005)
    for i in range(24):
        n = rng.choice([8, 9, g.wb, g.wb + 8, 3 * g.wb])
        vs_dst = 8 * rng.randrange(max(0, g.words - n // 8 - 2), g.words)
        d = gemv_desc(
            rng,
            g,
            n=n,
            vs_dst=vs_dst,
            accumulate=bool(i & 1),
            out_mode=rng.choice([OUT_VSRAM, OUT_VSRAM_DUMP, OUT_ARGMAX]),
        )
        await run_and_check(env, d, f"bounds {i}: vs_dst {vs_dst} n {n}")


@cocotb.test()
async def test_empty(dut):
    """N == 0 and an empty row set finish with no writes, no events and no meta consumed."""
    env, g, rng = await setup(dut, 0x6006)
    d = gemv_desc(rng, g, n=8, rows=1, out_mode=OUT_VSRAM_DUMP, track_absmax=True)
    d.n = 0
    d.meta = []
    d.acc = {r: [] for r in d.acc}
    res = await run_and_check(env, d, "n = 0")
    assert res.cycles < 8
    d = gemv_desc(rng, g, n=g.wb, out_mode=OUT_ARGMAX_DUMP, track_absmax=True)
    d.rows = 0
    d.meta = []
    d.acc = {}
    res = await run_and_check(env, d, "rows = 0")
    assert res.cycles < 8
    # and a normal descriptor right after
    await run_and_check(env, gemv_desc(rng, g), "after empty")


@cocotb.test()
async def test_requant_rows_crosscheck(dut):
    """One large descriptor per row set against numerics.requant_rows in block form."""
    env, g, rng = await setup(dut, 0x7007)
    for rows in range(1, 1 << g.b_max):
        d = gemv_desc(rng, g, n=6 * g.wb + 3, rows=rows, out_mode=OUT_VSRAM_DUMP)
        d.accumulate = True
        before = {r: list(w) for r, w in env.vsram.items()}
        exp = reference(d, g, before)
        y_rows, stats_rows = reference_rows(d, g, before)
        assert y_rows == exp.y and stats_rows == exp.stats, "requant and requant_rows disagree"
        res = await env.run(d)
        check(d, g, exp, res, f"rows {rows:#x}")


@cocotb.test()
async def test_throughput_and_latency(dut):
    """Back-to-back tiles drain at one element per cycle; the first word write follows the
    first handshake by a fixed latency; done follows the last write."""
    env, g, rng = await setup(dut, 0x8008)
    tiles = 6
    for rows in range(1, 1 << g.b_max):
        d = gemv_desc(rng, g, n=tiles * g.wb, rows=rows, out_mode=OUT_VSRAM_DUMP)
        d.accumulate = True
        d.unit_meta = False
        res = await run_and_check(env, d, f"throughput rows {rows:#x}")
        nrows = len(d.part_rows(g.b_max))
        gaps = [b - a for a, b in zip(res.handshakes, res.handshakes[1:], strict=False)]
        assert len(res.handshakes) == tiles
        assert all(gap == nrows * g.wb + 1 for gap in gaps), f"tile periods {gaps}"
        latency = res.first_write - res.handshakes[0]
        per_elem = res.cycles / (tiles * g.wb * nrows)
        dut._log.info(
            f"rows {nrows}: tile period {gaps[0]} cycles, first write {latency} cycles after "
            f"the handshake, {res.cycles} cycles for {tiles} tiles ({per_elem:.2f} per element)"
        )
        assert latency == 16, f"first write latency {latency}"
        assert res.cycles <= tiles * g.wb * nrows + 40
