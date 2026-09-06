"""cocotb tests of qcore_gemv_wrap: GEMV and EMBED descriptors through the assembled path.

The arbiter, the stream controller, the rows with their VSRAMs and the requant
execute descriptors against the QMEM bus model; every output element (VSRAM
word, dump byte, ARGMAX CSR, tracked absmax) is compared with numerics.requant
over the exact integer matmul, and the saturation / shift-clamp counts with
numerics.Stats.  Inputs are driven and outputs sampled at falling edges.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import cocotb
import numpy as np
import qc_numerics as qn
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge
from qc_qmem import QmemModel
from qc_stream import reset, value
from quettos import isa
from quettos.numerics import SFLOAT_ONE, SFLOAT_ZERO, SFloat, Stats

OP_GEMV, OP_EMBED = int(isa.Opcode.GEMV), int(isa.Opcode.EMBED)
OUT_VSRAM, OUT_ARGMAX, OUT_ARGMAX_DUMP, OUT_VSRAM_DUMP = 0, 1, 2, 3
LAT = 32
WEIGHT_BASE, META_BASE, DUMP_BASE = 0x0001_0000, 0x0008_0000, 0x000C_0000
EMB_BASE, EMB_META = 0x0010_0000, 0x0020_0000
MASK32 = 0xFFFF_FFFF
ISSUE_GAP = 4  # cycles from done to the next issue


@dataclass
class Desc:
    """One GEMV / EMBED descriptor with the data placed in QMEM and VSRAM for it."""

    op: int = OP_GEMV
    n: int = 1
    k: int = 1
    k_stride: int = 1
    rows: int = 1
    vs_src: int = 0
    vs_dst: int = 1024
    out_mode: int = OUT_VSRAM
    accumulate: bool = False
    unit_meta: bool = False
    track_absmax: bool = False
    s1: int = 0
    sbias: int = 0
    sreg_dst: int = 0
    tok: int = 0
    addr_a: int = WEIGHT_BASE
    addr_m: int = META_BASE
    addr_c: int = DUMP_BASE
    sx: list[SFloat] = field(default_factory=list)  # per physical row
    meta: list[tuple[int, SFloat]] = field(default_factory=list)  # (bias_q, Sw) per channel
    w: np.ndarray | None = None  # GEMV: [tiles*WB][K] int8 (padded channels zero)
    q: np.ndarray | None = None  # EMBED: the K gathered int8 values
    emb: tuple[int, SFloat] = (0, SFLOAT_ONE)

    @property
    def embed(self) -> bool:
        return self.op == OP_EMBED

    @property
    def n_out(self) -> int:
        return self.k if self.embed else self.n

    def tiles(self, wb: int) -> int:
        return -(-self.n_out // wb)

    def part_rows(self, b_max: int) -> list[int]:
        return [r for r in range(b_max) if (self.rows >> r) & 1]

    def sw_bias(self, c: int) -> tuple[SFloat, int]:
        if self.unit_meta:
            return SFLOAT_ONE, 0
        if self.embed:
            return self.emb[1], 0
        bias, sw = self.meta[c]
        return sw, bias

    def weight_beats(self, wb: int) -> int:
        """Beats the rows accept: K per tile, or one pseudo-beat per EMBED tile."""
        return self.tiles(wb) if self.embed else self.tiles(wb) * self.k

    def rd_beats(self, wb: int) -> int:
        if self.embed:
            return self.k + 1
        return self.tiles(wb) * (self.k + (0 if self.unit_meta else 8))


def _sfloat(rng: random.Random, lo: int = -22, hi: int = -6) -> SFloat:
    return SFloat(rng.randrange(1 << 15, 1 << 16), rng.randrange(lo, hi + 1))


# --------------------------------------------------------------------------- bench


class Bench:
    def __init__(self, dut, seed: int) -> None:
        self.dut = dut
        self.rng = random.Random(seed)
        self.wb = len(dut.wr_strb)
        self.b_max = len(dut.cmd_rows)
        self.words = 1 << len(dut.g_row[0].u_vsram.addr_a)
        self.model = QmemModel(dut, wb=self.wb, latency=LAT)
        self.vs: dict[tuple[int, int], int] = {}  # (bank, word) -> 256-bit word
        for name in (
            "cmd_valid_gemv", "cmd_op", "cmd_out_mode", "cmd_accumulate", "cmd_unit_meta",
            "cmd_track_absmax", "cmd_addr_a", "cmd_addr_m", "cmd_imm32", "cmd_n", "cmd_k",
            "cmd_k_stride", "cmd_vs_src", "cmd_vs_dst", "cmd_sreg_dst", "cmd_src_row",
            "cmd_dst_row", "cmd_sh0", "cmd_sh1", "cmd_rows", "cmd_tok", "cmd_sx_m", "cmd_sx_e",
            "f_req_valid", "f_req_addr", "f_req_len", "f_req_tag", "dq_count", "v_req_valid",
            "v_req_addr", "v_req_len", "v_req_tag", "k_wr_valid", "k_wr_addr", "k_wr_data",
            "k_wr_strb", "sreg_rd_en", "sreg_rd_idx",
        ):  # fmt: skip
            getattr(dut, name).value = 0

    # ---- VSRAM contents (deposited into the RAM arrays and mirrored)

    def vs_word(self, bank: int, w: int) -> int:
        return self.vs.get((bank, w), 0)

    def vs_set_word(self, bank: int, w: int, val: int) -> None:
        self.vs[(bank, w)] = val
        self.dut.g_row[bank].u_vsram.mem[w].value = val

    def vs_elem(self, bank: int, e: int) -> int:
        return (self.vs_word(bank, e // 8) >> (32 * (e % 8))) & MASK32

    def vs_fill(self, bank: int, start: int, count: int) -> None:
        for w in range(start // 8, -(-(start + count) // 8)):
            self.vs_set_word(bank, w, self.rng.getrandbits(256))

    def vs_read_word(self, bank: int, w: int) -> int:
        return value(self.dut.g_row[bank].u_vsram.mem[w])

    def sreg_read(self, bank: int, idx: int) -> int:
        return value(self.dut.g_row[bank].u_row.sreg[idx])

    # ---- descriptor placement

    def place(self, d: Desc) -> None:
        wb, rng = self.wb, self.rng
        for r in d.part_rows(self.b_max):
            self.vs_fill(r, d.vs_src, max(d.k, 1))
            self.vs_fill(r, d.vs_dst, d.n_out)
        if d.embed:
            base = d.addr_a + (d.tok // wb) * d.k * wb
            self.model.write_bytes(base, bytes(rng.getrandbits(8) for _ in range(d.k * wb)))
            for i in range(d.k):
                self.model.write_bytes(base + i * wb + d.tok % wb, bytes([int(d.q[i]) & 0xFF]))
            self.model.write_bytes(d.addr_m + d.tok * 8, qn.meta_bytes(d.emb[0], d.emb[1]))
            return
        for t in range(d.tiles(wb)):
            for kk in range(d.k):
                lanes = d.w[t * wb : (t + 1) * wb, kk]
                self.model.write_bytes(
                    d.addr_a + (t * d.k_stride + kk) * wb, lanes.astype(np.int8).tobytes()
                )
            for j in range(wb):
                bias, sw = d.meta[t * wb + j]
                self.model.write_bytes(d.addr_m + (t * wb + j) * 8, qn.meta_bytes(bias, sw))

    def gemv(self, n: int, k: int, rows: int = 1, **kw) -> Desc:
        wb, rng = self.wb, self.rng
        d = Desc(op=OP_GEMV, n=n, k=k, rows=rows, **kw)
        d.k_stride = kw.get("k_stride", k + rng.choice([0, 0, 3]))
        d.vs_src = kw.get("vs_src", rng.randrange(0, 512))
        d.vs_dst = kw.get("vs_dst", rng.randrange(1024, 8192) & ~7)
        d.sx = [_sfloat(rng) for _ in range(self.b_max)]
        tiles = d.tiles(wb)
        d.w = np.zeros((tiles * wb, k), dtype=np.int64)
        d.w[:n, :] = rng.choice([1, 1, 4]) * np.array(
            [[rng.randrange(-32, 32) for _ in range(k)] for _ in range(n)]
        )
        d.w = np.clip(d.w, -128, 127)
        d.meta = []
        for c in range(tiles * wb):
            if c >= n:
                d.meta.append((0, SFLOAT_ZERO))
            elif rng.random() < 0.03:
                d.meta.append((rng.randrange(-1000, 1000), SFLOAT_ZERO))
            else:
                d.meta.append((rng.randrange(-(1 << 20), 1 << 20), _sfloat(rng)))
        d.s1 = kw.get("s1", rng.randrange(0, 12))
        d.sbias = kw.get("sbias", rng.randrange(-40, -10) + rng.randrange(0, 30))
        d.sreg_dst = rng.randrange(0, 32)
        d.addr_a = WEIGHT_BASE + rng.randrange(0, 8) * 0x8000
        d.addr_m = META_BASE + rng.randrange(0, 8) * 0x4000
        d.addr_c = DUMP_BASE + rng.randrange(0, 8) * 0x4000
        return d

    def embed(self, k: int, tok: int, rows: int = 1, **kw) -> Desc:
        rng = self.rng
        d = Desc(
            op=OP_EMBED,
            n=k,
            k=k,
            k_stride=0,
            rows=rows,
            addr_a=EMB_BASE,
            addr_m=EMB_META,
            tok=tok,
            **kw,
        )
        d.vs_src = 0
        d.vs_dst = rng.randrange(1024, 8192) & ~7
        d.sx = [SFLOAT_ONE] * self.b_max
        d.q = np.array([rng.randrange(-128, 128) for _ in range(k)], dtype=np.int64)
        d.emb = (0, _sfloat(rng))
        d.s1 = rng.randrange(8, 25)
        d.sbias = -(rng.randrange(8, 16) + d.s1) + 24
        d.sreg_dst = rng.randrange(0, 32)
        return d

    # ---- execution

    async def run(self, d: Desc, tag: str, max_cycles: int = 40000) -> None:
        dut, wb = self.dut, self.wb
        self.place(d)
        exp = self.expect(d)
        rd0 = self.model.rd_beats
        sx_m = sum(s.m << (16 * r) for r, s in enumerate(d.sx))
        sx_e = sum(qn.from_signed(s.e, 8) << (8 * r) for r, s in enumerate(d.sx))
        await FallingEdge(dut.clk)
        dut.cmd_op.value = d.op
        dut.cmd_out_mode.value = d.out_mode
        dut.cmd_accumulate.value = int(d.accumulate)
        dut.cmd_unit_meta.value = int(d.unit_meta)
        dut.cmd_track_absmax.value = int(d.track_absmax)
        dut.cmd_addr_a.value = d.addr_a
        dut.cmd_addr_m.value = d.addr_m
        dut.cmd_imm32.value = d.addr_c
        dut.cmd_n.value = d.n_out
        dut.cmd_k.value = d.k
        dut.cmd_k_stride.value = d.k_stride
        dut.cmd_vs_src.value = d.vs_src
        dut.cmd_vs_dst.value = d.vs_dst
        dut.cmd_sreg_dst.value = d.sreg_dst
        dut.cmd_src_row.value = 0
        dut.cmd_dst_row.value = 0
        dut.cmd_sh0.value = d.s1
        dut.cmd_sh1.value = qn.from_signed(d.sbias, 8)
        dut.cmd_rows.value = d.rows
        dut.cmd_tok.value = d.tok
        dut.cmd_sx_m.value = sx_m
        dut.cmd_sx_e.value = sx_e
        dut.cmd_valid_gemv.value = 1
        await FallingEdge(dut.clk)
        dut.cmd_valid_gemv.value = 0
        sat = err_shift = err_bounds = beats = cycles = 0
        argmax = None
        while True:
            cycles += 1
            assert cycles < max_cycles, f"{tag}: no done within {max_cycles} cycles"
            sat += value(dut.sat_inc)
            err_shift += value(dut.err_shift_inc)
            err_bounds += value(dut.err_bounds_inc)
            beats += value(dut.gemv_beat)
            if value(dut.argmax_we):
                argmax = (value(dut.argmax_tok), value(dut.argmax_val))
            if value(dut.done_gemv):
                break
            await FallingEdge(dut.clk)
        for _ in range(ISSUE_GAP):
            await FallingEdge(dut.clk)
            beats += value(dut.gemv_beat)
        # The dispatcher issues only while the stream controller is idle: a
        # partial tile's padded meta beats may still be draining after done.
        idle_wait = 0
        while value(dut.stream_busy):
            await FallingEdge(dut.clk)
            idle_wait += 1
            assert idle_wait < 64, f"{tag}: stream busy long after done"
        assert value(dut.requant_busy) == 0, f"{tag}: requant busy after done"
        assert beats == d.weight_beats(wb), (
            f"{tag}: {beats} beats accepted, expected {d.weight_beats(wb)}"
        )
        assert self.model.rd_beats - rd0 == d.rd_beats(wb), (
            f"{tag}: {self.model.rd_beats - rd0} read beats"
        )
        assert err_bounds == 0, f"{tag}: {err_bounds} bounds errors"
        self.check(d, exp, tag, sat, err_shift, argmax)
        self.dut._log.info("%s: %d cycles, %d beats", tag, cycles, beats)

    # ---- reference

    def expect(self, d: Desc) -> dict:
        stats = Stats()
        out: dict[int, list[int]] = {}
        for r in d.part_rows(self.b_max):
            ys = []
            if d.embed:
                accs = [int(q) << 24 for q in d.q]
            else:
                a = np.array(
                    [
                        qn.to_signed(self.vs_elem(r, d.vs_src + kk) & 0xFFFF, 16)
                        for kk in range(d.k)
                    ],
                    dtype=np.int64,
                )
                accs = [int(np.dot(d.w[c], a)) for c in range(d.n)]
            for c, acc in enumerate(accs):
                sw, bias = d.sw_bias(c)
                old = qn.to_signed(self.vs_elem(r, d.vs_dst + c), 32) if d.accumulate else None
                ys.append(qn.requant(acc, sw, d.sx[r], d.s1, d.sbias, bias, old, stats))
            out[r] = ys
        return {"y": out, "sat": stats.sat, "err_shift": stats.err_shift}

    def check(self, d: Desc, exp: dict, tag: str, sat: int, err_shift: int, argmax) -> None:
        vsram = d.out_mode in (OUT_VSRAM, OUT_VSRAM_DUMP)
        dump = d.out_mode in (OUT_ARGMAX_DUMP, OUT_VSRAM_DUMP)
        rows = d.part_rows(self.b_max)
        for r in rows:
            ys = exp["y"][r]
            # every word of the destination range: written elements, untouched others
            for w in range(d.vs_dst // 8, -(-(d.vs_dst + d.n_out) // 8)):
                want = self.vs_word(r, w)
                if vsram:
                    for slot in range(8):
                        c = w * 8 + slot - d.vs_dst
                        if 0 <= c < d.n_out:
                            want = (want & ~(MASK32 << (32 * slot))) | (
                                (ys[c] & MASK32) << (32 * slot)
                            )
                got = self.vs_read_word(r, w)
                assert got == want, (
                    f"{tag}: row {r} word {w}: got {got:#066x}, expected {want:#066x}"
                )
                self.vs[(r, w)] = got
            if dump:
                got = self.model.read_bytes(d.addr_c + r * 4 * d.n_out, 4 * d.n_out)
                want = b"".join((y & MASK32).to_bytes(4, "little") for y in ys)
                assert got == want, f"{tag}: row {r} dump differs"
            if d.track_absmax:
                amax = max((abs(y) for y in ys), default=0)
                got = self.sreg_read(r, d.sreg_dst)
                assert got == amax, f"{tag}: row {r} absmax {got} != {amax}"
        if d.out_mode in (OUT_ARGMAX, OUT_ARGMAX_DUMP):
            ys = exp["y"][rows[-1]]
            best = max(range(len(ys)), key=lambda c: (ys[c], -c))
            assert argmax == (best, ys[best] & MASK32), (
                f"{tag}: argmax {argmax} != {(best, ys[best] & MASK32)}"
            )
        assert sat == exp["sat"], f"{tag}: SAT {sat} != {exp['sat']}"
        assert err_shift == exp["err_shift"], f"{tag}: ERR_SHIFT {err_shift} != {exp['err_shift']}"


async def _start(dut, seed: int) -> Bench:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    bench = Bench(dut, seed)
    await reset(dut, dut.rst)
    cocotb.start_soon(bench.model.run())
    return bench


# --------------------------------------------------------------------------- tests


@cocotb.test()
async def test_directed(dut):
    """Full, partial and multi-burst tiles, two rows, accumulate, every output mode."""
    b = await _start(dut, 1)
    cases = [
        ("full tile", b.gemv(16, 8)),
        ("partial tile with meta", b.gemv(17, 3)),
        ("one channel", b.gemv(1, 1)),
        ("two bursts, accumulate, both rows", b.gemv(5, 70, rows=3, accumulate=True)),
        ("argmax and dump", b.gemv(33, 20, rows=2, out_mode=OUT_ARGMAX_DUMP)),
        (
            "three tiles, vsram and dump, absmax",
            b.gemv(40, 130, rows=3, out_mode=OUT_VSRAM_DUMP, track_absmax=True),
        ),
        ("unit meta", b.gemv(20, 9, unit_meta=True)),
        ("argmax only", b.gemv(31, 5, out_mode=OUT_ARGMAX, track_absmax=True)),
        ("embed 37", b.embed(37, 1234)),
        ("embed 16", b.embed(16, 15)),
        ("embed 1, both rows", b.embed(1, 0, rows=3)),
        ("embed 50, dump", b.embed(50, 3, out_mode=OUT_VSRAM_DUMP)),
    ]
    for name, d in cases:
        await b.run(d, name)


@cocotb.test()
async def test_random(dut):
    """Random GEMV and EMBED descriptors back to back."""
    b = await _start(dut, 2)
    await _random_sequence(b, 40)


@cocotb.test()
async def test_random_slow_memory(dut):
    """The same mix with one memory beat every three cycles."""
    b = await _start(dut, 3)
    b.model.bw_div = 3
    await _random_sequence(b, 24)


async def _random_sequence(b: Bench, count: int) -> None:
    for i in range(count):
        if b.rng.random() < 0.2:
            d = b.embed(
                b.rng.randrange(1, 60), b.rng.randrange(0, 2000), rows=b.rng.choice([1, 2, 3])
            )
        else:
            d = b.gemv(
                b.rng.randrange(1, 40),
                b.rng.randrange(1, 80),
                rows=b.rng.choice([1, 1, 2, 3]),
                out_mode=b.rng.choice(
                    [OUT_VSRAM, OUT_VSRAM, OUT_ARGMAX, OUT_ARGMAX_DUMP, OUT_VSRAM_DUMP]
                ),
                accumulate=b.rng.random() < 0.3,
                unit_meta=b.rng.random() < 0.15,
                track_absmax=b.rng.random() < 0.3,
            )
        await b.run(d, f"random {i}")
