"""cocotb tests of qcore_vsram: every word through both ports, read latency and hold,
element write strobes, READ_FIRST on port B, and concurrent cross-port traffic."""

from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge

W = 256
NE = 8
EW = W // NE
MASK_W = (1 << W) - 1


def _words(dut) -> int:
    return 1 << len(dut.addr_a)


def _rand_word(rng: random.Random) -> int:
    return rng.getrandbits(W)


def _rd(sig) -> int:
    return int(sig.value.to_unsigned())


async def _setup(dut) -> random.Random:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.en_a.value = 0
    dut.addr_a.value = 0
    dut.en_b.value = 0
    dut.we_b.value = 0
    dut.addr_b.value = 0
    dut.wd_b.value = 0
    await FallingEdge(dut.clk)
    return random.Random(0x5EED)


async def _write(dut, addr: int, word: int, strobes: int = 0xFF) -> None:
    dut.we_b.value = strobes
    dut.addr_b.value = addr
    dut.wd_b.value = word
    await FallingEdge(dut.clk)
    dut.we_b.value = 0


async def _fill(dut, rng: random.Random, words: int) -> list[int]:
    data = [_rand_word(rng) for _ in range(words)]
    for a, w in enumerate(data):
        dut.we_b.value = 0xFF
        dut.addr_b.value = a
        dut.wd_b.value = w
        await FallingEdge(dut.clk)
    dut.we_b.value = 0
    return data


@cocotb.test()
async def test_every_word_both_ports(dut):
    """Write every word on port B, read every word back on port A and on port B, one per cycle."""
    rng = await _setup(dut)
    words = _words(dut)
    data = await _fill(dut, rng, words)
    for port, en, addr, rd in (
        ("A", dut.en_a, dut.addr_a, dut.rd_a),
        ("B", dut.en_b, dut.addr_b, dut.rd_b),
    ):
        for a in range(words + 1):
            if a >= 1:  # the word addressed one cycle earlier is on the output now
                got = _rd(rd)
                assert got == data[a - 1], f"port {port} word {a - 1}: {got:#x} != {data[a - 1]:#x}"
            if a < words:
                en.value = 1
                addr.value = a
            else:
                en.value = 0
            await FallingEdge(dut.clk)


@cocotb.test()
async def test_read_latency_and_hold(dut):
    """A read returns the word one cycle after the enabled cycle; the output holds when disabled."""
    rng = await _setup(dut)
    words = _words(dut)
    a0, a1 = 5, words - 1
    w0, w1 = _rand_word(rng), _rand_word(rng)
    await _write(dut, a0, w0)
    await _write(dut, a1, w1)
    for en, addr, rd in ((dut.en_a, dut.addr_a, dut.rd_a), (dut.en_b, dut.addr_b, dut.rd_b)):
        en.value = 1
        addr.value = a0
        await FallingEdge(dut.clk)
        assert _rd(rd) == w0
        en.value = 0
        addr.value = a1  # address changes without enable: the output must hold
        for _ in range(4):
            await FallingEdge(dut.clk)
            assert _rd(rd) == w0
        en.value = 1
        await FallingEdge(dut.clk)
        en.value = 0
        assert _rd(rd) == w1


@cocotb.test()
async def test_element_strobes(dut):
    """we_b[i] writes only element i (bits [32i+31:32i]); other elements keep their value."""
    rng = await _setup(dut)
    words = _words(dut)
    addr = rng.randrange(words)
    old = _rand_word(rng)
    await _write(dut, addr, old)
    cur = old
    patterns = [1 << i for i in range(NE)] + [rng.getrandbits(NE) for _ in range(24)] + [0, 0xFF]
    for strobes in patterns:
        new = _rand_word(rng)
        await _write(dut, addr, new, strobes)
        for i in range(NE):
            lane = ((1 << EW) - 1) << (EW * i)
            if (strobes >> i) & 1:
                cur = (cur & ~lane) | (new & lane)
        dut.en_a.value = 1
        dut.addr_a.value = addr
        await FallingEdge(dut.clk)
        dut.en_a.value = 0
        got = _rd(dut.rd_a)
        assert got == cur, f"strobes {strobes:#04x}: {got:#x} != {cur:#x}"


@cocotb.test()
async def test_port_b_read_first(dut):
    """A read and a write of the same word on port B in one cycle return the old word."""
    rng = await _setup(dut)
    addr = 77
    old, new = _rand_word(rng), _rand_word(rng)
    await _write(dut, addr, old)
    dut.en_b.value = 1
    dut.we_b.value = 0xFF
    dut.addr_b.value = addr
    dut.wd_b.value = new
    await FallingEdge(dut.clk)
    dut.we_b.value = 0
    assert _rd(dut.rd_b) == old, "READ_FIRST: the read must return the word before the write"
    await FallingEdge(dut.clk)
    dut.en_b.value = 0
    assert _rd(dut.rd_b) == new, "the write lands for the next read"


@cocotb.test()
async def test_cross_port_concurrent(dut):
    """Port A reads word X while port B writes Y != X; both complete; Y reads new next cycle."""
    rng = await _setup(dut)
    words = _words(dut)
    data = await _fill(dut, rng, 64)
    for _ in range(200):
        x = rng.randrange(64)
        y = rng.randrange(64)
        if y == x:
            y = (y + 1) % 64
        new = _rand_word(rng)
        dut.en_a.value = 1
        dut.addr_a.value = x
        dut.we_b.value = 0xFF
        dut.addr_b.value = y
        dut.wd_b.value = new
        await FallingEdge(dut.clk)
        dut.we_b.value = 0
        assert _rd(dut.rd_a) == data[x]
        data[y] = new
        dut.addr_a.value = y
        await FallingEdge(dut.clk)
        assert _rd(dut.rd_a) == new
    dut.en_a.value = 0
    # port A and port B read different words in the same cycle
    for _ in range(100):
        x, y = rng.randrange(64), rng.randrange(64)
        dut.en_a.value = 1
        dut.addr_a.value = x
        dut.en_b.value = 1
        dut.addr_b.value = y
        await FallingEdge(dut.clk)
        assert _rd(dut.rd_a) == data[x] and _rd(dut.rd_b) == data[y]
    dut.en_a.value = 0
    dut.en_b.value = 0
    assert words >= 64
