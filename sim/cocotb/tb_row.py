"""cocotb tests of qcore_row on the tiny configuration: random GEMVs over any vs_src
alignment and K with stream gaps and requant backpressure, full-rate streaming with the
tile-end rule, ranges past the end of the VSRAM, EMBED pseudo-beats, a non-participating
row, two rows fed the same beats in lockstep, and the SREG bank.

The bench models VSRAM port A (registered read, one cycle) and the requant's
accumulator reader (acc_ready per docs/RTL.md section 2.4); everything is driven
and sampled at falling edges."""

from __future__ import annotations

import random
from dataclasses import dataclass

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly
from qc_numerics import int8s_to_beat, to_signed
from qc_stream import reset

WORD_BITS = 256
ELEMS = 8


@dataclass
class Beat:
    data: int
    k: int
    tile: int
    start: bool
    end: bool
    nvalid: int
    last: bool
    embed: bool


@dataclass
class Desc:
    active: bool
    vs_src: int
    k: int
    beats: list[Beat]
    expected: list[list[int]]  # per tile, WB lanes


class RowEnv:
    """The row's surroundings: VSRAM port A, the weight stream, the requant reader."""

    def __init__(self, dut) -> None:
        self.dut = dut
        self.wb = len(dut.ws_data) // 8
        self.acc_w = len(dut.acc_flat) // self.wb
        self.words = 1 << len(dut.vsa_addr)
        self.mem: dict[int, int] = {}
        self.cycle = 0
        self.rd_prev: int | None = None
        self.acc_ready = 1

    # ---- VSRAM contents

    def fill(self, rng: random.Random, lo: int, hi: int) -> None:
        for w in range(max(0, lo), min(hi, self.words)):
            self.mem[w] = rng.getrandbits(WORD_BITS)

    def act(self, e: int) -> int:
        """A[k]: the low 16 bits of element e, sign-extended; 0 past the end."""
        if e >= self.words * ELEMS:
            return 0
        v = (self.mem.get(e // ELEMS, 0) >> (32 * (e % ELEMS))) & 0xFFFF
        return to_signed(v, 16)

    # ---- descriptors

    def gemv(self, rng: random.Random, vs_src: int, k: int, n: int, active: bool = True) -> Desc:
        wb = self.wb
        tiles = -(-n // wb)
        w = np.array(
            [[[rng.randint(-128, 127) for _ in range(wb)] for _ in range(k)] for _ in range(tiles)],
            dtype=np.int64,
        )
        a = np.array([self.act(vs_src + i) for i in range(k)], dtype=np.int64)
        beats = []
        for t in range(tiles):
            for i in range(k):
                beats.append(
                    Beat(
                        data=int8s_to_beat(w[t, i]),
                        k=i,
                        tile=t,
                        start=i == 0,
                        end=i == k - 1,
                        nvalid=min(wb, n - t * wb),
                        last=(t == tiles - 1 and i == k - 1),
                        embed=False,
                    )
                )
        expected = [[int(x) for x in (w[t] * a[:, None]).sum(axis=0)] for t in range(tiles)]
        return Desc(active, vs_src, k, beats, expected)

    def embed(self, rng: random.Random, k: int) -> Desc:
        wb = self.wb
        q = [rng.randint(-128, 127) for _ in range(k)]
        tiles = -(-k // wb)
        beats, expected = [], []
        for t in range(tiles):
            lanes = q[t * wb : (t + 1) * wb]
            lanes += [0] * (wb - len(lanes))
            beats.append(
                Beat(
                    data=int8s_to_beat(lanes),
                    k=0,
                    tile=t,
                    start=True,
                    end=True,
                    nvalid=min(wb, k - t * wb),
                    last=(t == tiles - 1),
                    embed=True,
                )
            )
            expected.append([v << 24 for v in lanes])
        return Desc(True, 0, k, beats, expected)

    # ---- signal helpers

    def flat(self) -> list[int]:
        v = int(self.dut.acc_flat.value.to_unsigned())
        m = (1 << self.acc_w) - 1
        return [to_signed((v >> (j * self.acc_w)) & m, self.acc_w) for j in range(self.wb)]

    def _drive_beat(self, b: Beat | None, rng: random.Random) -> None:
        d = self.dut
        if b is None:
            d.ws_valid.value = 0
            d.ws_data.value = rng.getrandbits(self.wb * 8)
            d.ws_k.value = rng.getrandbits(16)
            d.ws_tile.value = rng.getrandbits(20)
            d.ws_tile_start.value = rng.getrandbits(1)
            d.ws_tile_end.value = rng.getrandbits(1)
            d.ws_nvalid.value = rng.getrandbits(len(d.ws_nvalid))
            d.ws_last.value = rng.getrandbits(1)
            d.ws_embed.value = rng.getrandbits(1)
            return
        d.ws_valid.value = 1
        d.ws_data.value = b.data
        d.ws_k.value = b.k
        d.ws_tile.value = b.tile
        d.ws_tile_start.value = int(b.start)
        d.ws_tile_end.value = int(b.end)
        d.ws_nvalid.value = b.nvalid
        d.ws_last.value = int(b.last)
        d.ws_embed.value = int(b.embed)

    # ---- the descriptor engine

    async def run(
        self,
        desc: Desc,
        rng: random.Random,
        p_gap: float = 0.0,
        p_bp: float = 0.0,
        max_cycles: int = 50000,
    ) -> list[int]:
        """Issue ``desc``, stream its beats, drain every tile; returns the accept cycles.

        Cycle x runs from rising edge x to rising edge x + 1: its inputs are written at
        the falling edge inside it, the combinational handshake (ws_ready, ev_beat) is
        read once those have settled (ReadOnly), and its registered outputs are read at
        the next falling edge, after rising edge x + 1.
        """
        dut = self.dut
        tiles = len(desc.expected)
        nvs = [b.nvalid for b in desc.beats if b.end]
        w0 = desc.vs_src >> 3
        wlast = (desc.vs_src + max(desc.k, 1) - 1) >> 3
        overflow = desc.vs_src + desc.k > self.words * ELEMS
        x = self.cycle
        issue = x

        bi = 0
        presented: Beat | None = None
        accepted = False
        gap = 0
        accepts: list[int] = []
        last_accept = -1
        stream_done = False
        waiting: int | None = None  # tile on acc_flat awaiting its transfer
        transfer: tuple[int, int] | None = None  # (tile, nvalid) transferring at the next edge
        reads: dict[int, int] = {}
        low: set[int] = set()
        final = -1
        drained = 0
        finish_at: int | None = None

        async def drive(x: int, accv: int) -> None:
            nonlocal presented, accepted, transfer, waiting
            dut.cmd_valid_gemv.value = int(x == issue)
            dut.row_active.value = int(desc.active)
            dut.cmd_vs_src.value = desc.vs_src
            dut.cmd_k.value = desc.k
            if x in low:
                ar = 0
            elif x == final:
                ar = 1
            else:
                ar = 0 if rng.random() < p_bp else 1
            dut.acc_ready.value = ar
            self.acc_ready = ar
            transfer = None
            if accv and ar and waiting is not None:
                transfer = (waiting, nvs[waiting])
                waiting = None
            if x == issue:
                presented = None  # the row becomes active the cycle after the pulse
                self._drive_beat(None, rng)
            elif accepted or presented is None:
                if bi < len(desc.beats):
                    if gap_state[0] == 0 and p_gap and rng.random() < p_gap:
                        gap_state[0] = rng.randint(1, 4)
                    if gap_state[0]:
                        gap_state[0] -= 1
                        presented = None
                    else:
                        presented = desc.beats[bi]
                else:
                    presented = None
                self._drive_beat(presented, rng)
            await ReadOnly()
            ready = int(dut.ws_ready.value)
            ev = int(dut.ev_beat.value)
            accepted = presented is not None and ready == 1
            assert ev == (1 if accepted and desc.active else 0), f"cycle {x}: ev_beat {ev}"
            if not desc.active:
                assert ready == 1, "a non-participating row stalled the stream"
            if presented is not None and desc.active and presented.end and (accv or not ar):
                assert ready == 0, (
                    f"cycle {x}: tile-end beat accepted while acc_valid || !acc_ready"
                )

        gap_state = [gap]
        await drive(x, int(dut.acc_valid.value))
        while True:
            await FallingEdge(dut.clk)
            x += 1
            self.cycle = x
            assert x - issue < max_cycles, f"descriptor did not finish in {max_cycles} cycles"

            accv = int(dut.acc_valid.value)
            errb = int(dut.err_bounds.value)
            ven = int(dut.vsa_en.value)
            vaddr = int(dut.vsa_addr.value.to_unsigned())
            want_err = 1 if (x == issue + 1 and desc.active and overflow) else 0
            assert errb == want_err, f"cycle {x}: err_bounds {errb} != {want_err}"
            if not desc.active:
                assert accv == 0 and ven == 0, "a non-participating row acted"
            if ven:
                assert vaddr < self.words
                assert w0 <= vaddr <= wlast, f"cycle {x}: read of word {vaddr} outside the range"
                assert not stream_done or x <= last_accept, f"cycle {x}: read after the last beat"

            if accepted:
                b = desc.beats[bi]
                bi += 1
                if desc.active:
                    accepts.append(x - 1)
                    last_accept = x - 1
                    if b.end:
                        t = b.tile
                        assert accv == 1, f"cycle {x}: acc_valid low the cycle after tile {t}"
                        got = self.flat()
                        assert got == desc.expected[t], f"tile {t} at swap: {got}"
                        assert int(dut.acc_tile.value) == t, f"acc_tile != {t}"
                        assert int(dut.acc_nvalid.value) == nvs[t], f"acc_nvalid != {nvs[t]}"
                        assert int(dut.acc_last.value) == int(t == tiles - 1), "acc_last"
                        waiting = t
                    if b.last:
                        stream_done = True
                elif bi == len(desc.beats):
                    finish_at = x + 3
            if transfer is not None:
                t, nv = transfer
                assert accv == 0, f"cycle {x}: acc_valid still high after the transfer"
                for i in range(nv):
                    reads[x + i] = t
                low = set(range(x, x + nv - 1))
                final = x + nv - 1
            if x in reads:
                t = reads.pop(x)
                got = self.flat()
                assert got == desc.expected[t], f"tile {t} drain read at cycle {x}: {got}"
                if not reads:
                    drained += 1
                    if drained == tiles:
                        finish_at = x + 3

            # VSRAM port A: the word addressed in the previous cycle is on rd_a now
            if self.rd_prev is not None:
                dut.vsa_rdata.value = self.mem.get(self.rd_prev, 0)
            self.rd_prev = vaddr if ven else None

            if finish_at is not None and x >= finish_at:
                assert bi == len(desc.beats) and not reads and waiting is None
                dut.cmd_valid_gemv.value = 0
                self._drive_beat(None, rng)
                return accepts
            await drive(x, accv)


async def _setup(dut) -> RowEnv:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for name in (
        "cmd_valid_gemv",
        "row_active",
        "cmd_vs_src",
        "cmd_k",
        "ws_valid",
        "ws_data",
        "ws_k",
        "ws_tile",
        "ws_tile_start",
        "ws_tile_end",
        "ws_nvalid",
        "ws_last",
        "ws_embed",
        "vsa_rdata",
        "acc_ready",
        "sreg_rd_en",
        "sreg_rd_idx",
        "sreg_wr_en",
        "sreg_wr_idx",
        "sreg_wr_data",
    ):
        getattr(dut, name).value = 0
    await reset(dut, dut.rst)
    env = RowEnv(dut)
    env.dut.acc_ready.value = 1
    return env


def _expected_accepts(first: int, k: int, nvalids: list[int]) -> list[int]:
    """Accept cycles at full rate: one per cycle, a tile end waiting for the previous drain."""
    out = []
    c = first
    e_prev = None
    for nv in nvalids:
        for i in range(k):
            if i == k - 1 and e_prev is not None:
                c = max(c, e_prev[0] + e_prev[1] + 1)
            out.append(c)
            if i == k - 1:
                e_prev = (c, nv)
            c += 1
    return out


@cocotb.test()
async def test_gemv_random(dut):
    """Random vs_src alignment, K and N (partial last tiles) with stream gaps and requant
    backpressure: every tile bit-exact at the swap and through its drain."""
    env = await _setup(dut)
    rng = random.Random(0x0A1)
    env.fill(rng, 0, env.words)
    for i in range(24):
        k = rng.choice([1, 2, 3, 7, 8, 9, 15, 16, 17, 24, rng.randint(1, 120)])
        n = rng.choice([1, env.wb - 1, env.wb, env.wb + 1, rng.randint(1, 5 * env.wb)])
        vs_src = rng.randint(0, env.words * ELEMS - k)
        d = env.gemv(rng, vs_src, k, n)
        acc = await env.run(d, rng, p_gap=0.25 * (i % 3), p_bp=0.3 * (i % 2))
        assert len(acc) == len(d.beats)


@cocotb.test()
async def test_gemv_full_rate(dut):
    """No gaps, no backpressure: one beat per cycle inside a tile for aligned and unaligned
    vs_src and every K mod 8, tile ends spaced by max(K, nvalid + 1)."""
    env = await _setup(dut)
    rng = random.Random(0xF11)
    env.fill(rng, 0, env.words)
    for k in [1, 2, 3, 5, 8, 9, 10, 15, 16, 17, 33, 64]:
        for vs_src in [0, 7, 8, 13, 1024 + 6]:
            n = rng.choice([env.wb, 2 * env.wb + 1, 3 * env.wb, 4 * env.wb - 3])
            d = env.gemv(rng, vs_src, k, n)
            acc = await env.run(d, rng)
            nvalids = [min(env.wb, n - t * env.wb) for t in range(len(d.expected))]
            want = _expected_accepts(acc[0], k, nvalids)
            assert acc == want, f"K={k} vs_src={vs_src} N={n}: accepts {acc} != {want}"


@cocotb.test()
async def test_bounds(dut):
    """A range past VSRAM_WORDS*8: err_bounds pulses once, the missing elements read as 0
    and the row never reads a word past the end."""
    env = await _setup(dut)
    rng = random.Random(0xB0D)
    env.fill(rng, env.words - 8, env.words)
    top = env.words * ELEMS
    for vs_src, k in [(top - 5, 30), (top - 1, 1), (top - 16, 17), (top - 64, 64), (top - 3, 100)]:
        d = env.gemv(rng, vs_src, k, 2 * env.wb - 1)
        await env.run(d, rng, p_gap=0.2)
    d = env.gemv(rng, top - 64, 64, env.wb)
    await env.run(d, rng)


@cocotb.test()
async def test_embed(dut):
    """EMBED pseudo-beats are accepted without an activation word and load sext(q) << 24;
    every pseudo-beat is a tile with its own handoff."""
    env = await _setup(dut)
    rng = random.Random(0xE3B)
    env.fill(rng, 0, 4)
    for k in [1, env.wb - 1, env.wb, env.wb + 1, 3 * env.wb, 100, 896 // 4]:
        d = env.embed(rng, k)
        await env.run(d, rng, p_gap=0.2, p_bp=0.2)


@cocotb.test()
async def test_inactive_row(dut):
    """A non-participating row accepts every beat, reads nothing, raises nothing, and
    participates normally in the next descriptor."""
    env = await _setup(dut)
    rng = random.Random(0x1A0)
    env.fill(rng, 0, env.words)
    top = env.words * ELEMS
    for vs_src, k in [(0, 16), (top - 4, 40), (100, 3)]:
        d = env.gemv(rng, vs_src, k, 2 * env.wb, active=False)
        await env.run(d, rng, p_gap=0.3)
        d = env.gemv(rng, vs_src if vs_src + k <= top else 8, k, env.wb + 2)
        await env.run(d, rng, p_gap=0.3, p_bp=0.3)


@cocotb.test()
async def test_two_rows_lockstep(dut):
    """Two rows fed the same beats under the same gaps and backpressure accept every beat
    in the same cycle (relative to issue) while producing their own bit-exact sums."""
    env = await _setup(dut)
    traces = []
    for row in range(2):
        env.mem = {}
        env.fill(random.Random(0x5EED + row), 0, env.words)
        seq = random.Random(0x5E0)  # the shared beat, gap and backpressure sequence
        row_trace = []
        for _ in range(6):
            k = seq.choice([1, 3, 8, 9, 17, 40])
            n = seq.choice([env.wb, 2 * env.wb - 1, 3 * env.wb + 1])
            vs_src = seq.randint(0, 200)
            d = env.gemv(seq, vs_src, k, n)
            base = env.cycle
            acc = await env.run(d, seq, p_gap=0.3, p_bp=0.3)
            row_trace.append([c - base for c in acc])
        traces.append(row_trace)
    assert traces[0] == traces[1], "the two rows did not accept beats in lockstep"


@cocotb.test()
async def test_sreg_bank(dut):
    """32 registers: writes land the next cycle, reads return data the cycle after the
    enable and hold; an index at or above 32 reads 0, drops the write and pulses sreg_err."""
    env = await _setup(dut)
    rng = random.Random(0x5E6)
    d = env.dut
    words = [rng.getrandbits(32) for _ in range(32)]
    for i, w in enumerate(words):
        d.sreg_wr_en.value = 1
        d.sreg_wr_idx.value = i
        d.sreg_wr_data.value = w
        await FallingEdge(d.clk)
        assert int(d.sreg_err.value) == 0
    d.sreg_wr_en.value = 0
    for i in rng.sample(range(32), 32):
        d.sreg_rd_en.value = 1
        d.sreg_rd_idx.value = i
        await FallingEdge(d.clk)
        d.sreg_rd_en.value = 0
        d.sreg_rd_idx.value = rng.randrange(256)
        assert int(d.sreg_rd_data.value) == words[i], f"SREG[{i}]"
        await FallingEdge(d.clk)
        assert int(d.sreg_rd_data.value) == words[i], "read data must hold"
    # out-of-range read: 0 and one error pulse
    for bad in [32, 33, 100, 255]:
        d.sreg_rd_en.value = 1
        d.sreg_rd_idx.value = bad
        await FallingEdge(d.clk)
        d.sreg_rd_en.value = 0
        assert int(d.sreg_rd_data.value) == 0 and int(d.sreg_err.value) == 1
        await FallingEdge(d.clk)
        assert int(d.sreg_err.value) == 0
    # out-of-range write: dropped, one error pulse; in-range contents untouched
    for bad in [32, 64, 255]:
        d.sreg_wr_en.value = 1
        d.sreg_wr_idx.value = bad
        d.sreg_wr_data.value = 0xDEADBEEF
        await FallingEdge(d.clk)
        d.sreg_wr_en.value = 0
        assert int(d.sreg_err.value) == 1
        await FallingEdge(d.clk)
        assert int(d.sreg_err.value) == 0
    for i in range(32):
        d.sreg_rd_en.value = 1
        d.sreg_rd_idx.value = i
        await FallingEdge(d.clk)
        d.sreg_rd_en.value = 0
        assert int(d.sreg_rd_data.value) == words[i]
    # a write followed by a read of the same register the next cycle
    d.sreg_wr_en.value = 1
    d.sreg_wr_idx.value = 5
    d.sreg_wr_data.value = 0x12345678
    await FallingEdge(d.clk)
    d.sreg_wr_en.value = 0
    d.sreg_rd_en.value = 1
    d.sreg_rd_idx.value = 5
    await FallingEdge(d.clk)
    d.sreg_rd_en.value = 0
    assert int(d.sreg_rd_data.value) == 0x12345678
