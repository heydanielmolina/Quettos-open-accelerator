"""cocotb tests of qcore_mac_lane_group: random tiles bit-exact against integer math,
a fresh sum on every tile_start, the hold set through the following tile, idle cycles,
EMBED loads and the accumulator extremes."""

from __future__ import annotations

import random

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, Timer
from qc_numerics import from_signed, int8s_to_beat, to_signed

LANES = 8


def _acc_w(dut) -> int:
    return len(dut.acc_drain) // LANES


def _drain(dut) -> list[int]:
    w = _acc_w(dut)
    v = int(dut.acc_drain.value.to_unsigned())
    return [to_signed((v >> (i * w)) & ((1 << w) - 1), w) for i in range(LANES)]


def _sums(w: np.ndarray, a: np.ndarray) -> list[int]:
    """acc[i] = sum_k w[k, i] * a[k] in exact int64 arithmetic."""
    return [int(x) for x in (w.astype(np.int64) * a.astype(np.int64)[:, None]).sum(axis=0)]


async def _setup(dut, seed: int) -> random.Random:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.en.value = 0
    dut.tile_start.value = 0
    dut.embed.value = 0
    dut.buf_sel.value = 0
    dut.w.value = 0
    dut.a.value = 0
    await FallingEdge(dut.clk)
    return random.Random(seed)


async def _beat(dut, w8, a: int, tile_start: bool, tile_end: bool, embed: bool = False) -> None:
    """Consume one beat at the next rising edge, then set buf_sel as the row does
    (buf_sel <= tile_end on every consumed beat)."""
    dut.en.value = 1
    dut.tile_start.value = int(tile_start)
    dut.embed.value = int(embed)
    dut.w.value = int8s_to_beat(w8)
    dut.a.value = from_signed(int(a), 16)
    await FallingEdge(dut.clk)
    dut.en.value = 0
    dut.buf_sel.value = int(tile_end)


async def _idle(dut, rng: random.Random, n: int) -> None:
    """n cycles with en low and random garbage on every other input."""
    for _ in range(n):
        dut.en.value = 0
        dut.tile_start.value = rng.getrandbits(1)
        dut.embed.value = rng.getrandbits(1)
        dut.w.value = rng.getrandbits(64)
        dut.a.value = rng.getrandbits(16)
        await FallingEdge(dut.clk)
    dut.tile_start.value = 0
    dut.embed.value = 0


async def _settled(dut) -> list[int]:
    """acc_drain after the values written at this falling edge have propagated."""
    await Timer(1, "ns")
    return _drain(dut)


async def _run_tile(dut, rng: random.Random, w: np.ndarray, a: np.ndarray, prev, gaps: int):
    """Stream one tile; while it runs acc_drain must show ``prev`` (the hold set);
    the cycle after its last beat acc_drain must show the tile's sums."""
    k = w.shape[0]
    for i in range(k):
        if gaps and rng.random() < 0.3:
            await _idle(dut, rng, rng.randint(1, 3))
            if prev is not None:
                assert await _settled(dut) == prev, "hold set changed during an idle cycle"
        await _beat(dut, w[i], int(a[i]), i == 0, i == k - 1)
        got = await _settled(dut)
        if i < k - 1:
            if prev is not None:
                assert got == prev, f"beat {i}: hold set changed to {got}, expected {prev}"
    sums = _sums(w, a)
    assert got == sums, f"tile sums {got} != {sums}"
    return sums


@cocotb.test()
async def test_random_tiles(dut):
    """Random K, weights and activations over many tiles: every tile bit-exact, every
    tile starting from zero, the previous tile held on acc_drain throughout."""
    rng = await _setup(dut, 0xA11)
    prev = None
    for t in range(60):
        k = rng.choice([1, 2, 3, 7, 8, 9, 16, 17, rng.randint(1, 64)])
        w = np.array([[rng.randint(-128, 127) for _ in range(LANES)] for _ in range(k)])
        a = np.array([rng.randint(-32768, 32767) for _ in range(k)])
        prev = await _run_tile(dut, rng, w, a, prev, gaps=t % 2)


@cocotb.test()
async def test_embed_beats(dut):
    """An EMBED beat loads sext(w[i]) << 24 regardless of a; the following GEMV tile
    keeps it on acc_drain until that tile ends."""
    rng = await _setup(dut, 0xE3B)
    prev = None
    for _ in range(20):
        q = [rng.choice([-128, -1, 0, 1, 127, rng.randint(-128, 127)]) for _ in range(LANES)]
        await _beat(dut, q, rng.randint(-32768, 32767), True, True, embed=True)
        got = await _settled(dut)
        want = [v << 24 for v in q]
        assert got == want, f"embed {got} != {want}"
        k = rng.randint(1, 12)
        w = np.array([[rng.randint(-128, 127) for _ in range(LANES)] for _ in range(k)])
        a = np.array([rng.randint(-32768, 32767) for _ in range(k)])
        prev = await _run_tile(dut, rng, w, a, want, gaps=1)
        assert prev == _sums(w, a)


@cocotb.test()
async def test_extremes(dut):
    """Corner operands over long tiles: the largest positive and negative products,
    alternating signs, and a sum that lands exactly on zero."""
    rng = await _setup(dut, 0xE37)
    k = 4000
    cases = []
    w = np.full((k, LANES), -128)
    a = np.full(k, -32768)
    cases.append((w, a))  # +2^22 per beat
    w = np.full((k, LANES), -128)
    a = np.full(k, 32767)
    cases.append((w, a))  # most negative product
    w = np.array([[127 if (i + j) % 2 else -128 for j in range(LANES)] for i in range(k)])
    a = np.array([32767 if i % 2 else -32768 for i in range(k)])
    cases.append((w, a))
    w = np.array([[100] * LANES if i % 2 else [-100] * LANES for i in range(k)])
    a = np.full(k, 30000)
    cases.append((w, a))  # sums to zero
    prev = None
    for w, a in cases:
        prev = await _run_tile(dut, rng, w, a, prev, gaps=0)
        assert max(abs(v) for v in prev) < 1 << (_acc_w(dut) - 1)
    assert prev == [0] * LANES
