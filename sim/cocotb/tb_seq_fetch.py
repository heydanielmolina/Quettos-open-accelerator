"""cocotb tests of qcore_seq_fetch: descriptor bursts through the arbiter's fetch port at three
memory latencies with request and consumer backpressure, the reserved-slot rule, a queue that
never overflows, the leading-descriptor skip of an unaligned restart at every position in a beat,
step mode across the beat boundaries, the write fence that holds the prefetch, the flush that
discards beats still in flight, and the FETCH_BEATS strobe."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import cocotb
import qc_stream
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge
from quettos import isa

TAG_FETCH = 0
DESC_B = isa.DESC_BYTES
BASE = 0x0000_4000


@dataclass
class _Req:
    deliver: int
    addr: int
    last: bool


@dataclass
class FetchBus:
    """QMEM read port seen by qcore_seq_fetch: fixed latency, one beat per cycle, in order."""

    dut: Any
    wb: int
    latency: int = 32
    window: int = 64
    ready_pattern: Any = 1
    mem: bytearray = field(default_factory=lambda: bytearray(1 << 20))
    beats: int = 0
    requests: list[tuple[int, int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._t = 0
        self._pending: list[_Req] = []
        self._last_deliver = -1
        self._prev = (0, 0, 0, 0, 0)
        for name in ("f_req_ready", "fd_valid", "fd_data", "fd_last"):
            getattr(self.dut, name).value = 0

    def write(self, addr: int, data: bytes) -> None:
        self.mem[addr : addr + len(data)] = data

    def beat(self, addr: int) -> int:
        return int.from_bytes(self.mem[addr : addr + self.wb], "little")

    @property
    def outstanding(self) -> int:
        return len(self._pending)

    def _sample(self) -> tuple[int, int, int, int, int]:
        valid = int(self.dut.f_req_valid.value)
        if not valid:
            return (0, 0, 0, 0, 0)
        return (
            1,
            qc_stream.value(self.dut.f_req_addr),
            qc_stream.value(self.dut.f_req_len),
            qc_stream.value(self.dut.f_req_tag),
            0,
        )

    def _accept(self, s: tuple[int, int, int, int, int]) -> None:
        _, addr, length, tag, _ = s
        assert tag == TAG_FETCH, f"fetch request carries tag {tag}, not TAG_FETCH"
        assert length >= 1, "a fetch request of zero beats"
        self.requests.append((addr, length, tag))
        for i in range(length):
            deliver = max(self._t + self.latency + i, self._last_deliver + 1)
            self._last_deliver = deliver
            self._pending.append(_Req(deliver, addr + i * self.wb, i == length - 1))

    async def run(self) -> None:
        clk = self.dut.clk
        while True:
            await FallingEdge(clk)
            self._t += 1
            prev = self._prev
            if prev[0] and prev[4]:
                self._accept(prev)
            cur = self._sample()
            nxt = self._t + 1
            if self._pending and self._pending[0].deliver <= nxt:
                b = self._pending.pop(0)
                self.dut.fd_valid.value = 1
                self.dut.fd_data.value = self.beat(b.addr)
                self.dut.fd_last.value = int(b.last)
                self.beats += 1
            else:
                self.dut.fd_valid.value = 0
                self.dut.fd_last.value = 0
            p = self.ready_pattern
            ready = int(bool(p(self._t) if callable(p) else p)) and len(self._pending) < self.window
            self.dut.f_req_ready.value = ready
            self._prev = (cur[0], cur[1], cur[2], cur[3], ready)


class Queue:
    """Consumer of the descriptor queue: drives dq_ready and records every popped word."""

    def __init__(self, dut, pattern: Any = 1) -> None:
        self.dut = dut
        self.pattern = pattern
        self.words: list[int] = []
        self.max_count = 0
        self.stop = False

    async def run(self) -> None:
        clk = self.dut.clk
        cycle = 0
        while not self.stop:
            await RisingEdge(clk)
            self.max_count = max(self.max_count, qc_stream.value(self.dut.dq_count))
            if int(self.dut.dq_valid.value) and int(self.dut.dq_ready.value):
                self.words.append(qc_stream.value(self.dut.dq_desc))
            await FallingEdge(clk)
            p = self.pattern
            self.dut.dq_ready.value = int(bool(p(cycle) if callable(p) else p))
            cycle += 1


def program(rng: random.Random, count: int) -> tuple[bytes, list[int]]:
    """``count`` random 32-byte descriptor blobs and the 256-bit word of each."""
    raw = bytes(rng.getrandbits(8) for _ in range(count * DESC_B))
    words = [int.from_bytes(raw[i * DESC_B : (i + 1) * DESC_B], "little") for i in range(count)]
    return raw, words


async def _setup(dut, seed: int = 0x5EE, latency: int = 32, **kw):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for sig in (
        dut.fetch_start,
        dut.fetch_pc,
        dut.fetch_step,
        dut.fetch_flush,
        dut.fetch_hold,
        dut.dq_ready,
    ):
        sig.value = 0
    wb = len(dut.fd_data.value) // 8
    bus = FetchBus(dut, wb, latency=latency, **kw)
    await qc_stream.reset(dut, dut.rst)
    cocotb.start_soon(bus.run())
    return bus, random.Random(seed), wb


async def restart(dut, pc: int, step: bool = False) -> None:
    """Pulse fetch_flush and fetch_start at ``pc`` the way the dispatcher does."""
    dut.fetch_step.value = int(step)
    dut.fetch_pc.value = pc
    dut.fetch_flush.value = 1
    dut.fetch_start.value = 1
    await FallingEdge(dut.clk)
    dut.fetch_flush.value = 0
    dut.fetch_start.value = 0


async def collect(dut, q: Queue, count: int, timeout: int = 20000) -> list[int]:
    """Wait until ``count`` descriptors have been popped since the call."""
    have = len(q.words)
    for _ in range(timeout):
        if len(q.words) >= have + count:
            return q.words[have : have + count]
        await FallingEdge(dut.clk)
    raise TimeoutError(f"only {len(q.words) - have} of {count} descriptors arrived")


@cocotb.test()
async def test_fetch_in_order_at_three_latencies(dut):
    """Descriptors arrive in program order at LAT 1, 32 and 200 with the queue always fed."""
    bus, rng, _ = await _setup(dut, latency=1)
    q = Queue(dut, 1)
    cocotb.start_soon(q.run())
    raw, words = program(rng, 24)
    bus.write(BASE, raw)
    for latency in (1, 32, 200):
        bus.latency = latency
        await restart(dut, BASE)
        got = await collect(dut, q, len(words))
        assert got == words, f"LAT {latency}: descriptor order or content differs"
        assert bus.requests, f"LAT {latency}: no fetch request issued"


@cocotb.test()
async def test_fetch_with_request_and_consumer_backpressure(dut):
    """A stalling arbiter and a slow consumer change nothing but the rate."""
    bus, rng, _ = await _setup(dut, latency=17, ready_pattern=lambda t: (t % 5) != 0)
    q = Queue(dut, lambda c: (c % 7) < 2)
    cocotb.start_soon(q.run())
    raw, words = program(rng, 20)
    bus.write(BASE, raw)
    await restart(dut, BASE)
    got = await collect(dut, q, len(words), timeout=60000)
    assert got == words, "descriptor order or content changed under backpressure"


@cocotb.test()
async def test_reserved_slots_and_no_overflow(dut):
    """At most two bursts are outstanding and the queue never takes more than it reserved."""
    bus, rng, _ = await _setup(dut, latency=3)
    q = Queue(dut, lambda c: (c % 40) == 0)  # a very slow consumer fills the queue
    cocotb.start_soon(q.run())
    raw, words = program(rng, 12)
    bus.write(BASE, raw)
    await restart(dut, BASE)
    beats_per_burst = qc_stream.value(dut.f_req_len)
    worst = 0
    for _ in range(600):
        await FallingEdge(dut.clk)
        worst = max(worst, bus.outstanding)
        assert bus.outstanding <= 2 * beats_per_burst, (
            f"{bus.outstanding} beats in flight, more than two bursts of {beats_per_burst}"
        )
    assert worst > 0, "no request was ever outstanding"
    assert 8 <= q.max_count <= 12, f"dq_count reached {q.max_count}"
    while len(q.words) < len(words):
        await collect(dut, q, 1, timeout=40000)
    assert q.words[: len(words)] == words, "a backed-up queue reordered or lost descriptors"


@cocotb.test()
async def test_step_mode_fetches_one_descriptor(dut):
    """With fetch_step high a restart issues exactly one request; PC moves between steps."""
    bus, rng, _ = await _setup(dut, latency=8)
    q = Queue(dut, 1)
    cocotb.start_soon(q.run())
    raw, words = program(rng, 6)
    bus.write(BASE, raw)
    for i, want in enumerate(words):
        before = len(bus.requests)
        await restart(dut, BASE + i * DESC_B, step=True)
        got = await collect(dut, q, 1)
        assert got == [want], f"step {i}: wrong descriptor"
        await qc_stream.cycles(dut.clk, 60)
        issued = len(bus.requests) - before
        assert issued == 1, f"step {i}: {issued} requests issued, expected 1"
    dut.fetch_step.value = 0


@cocotb.test()
async def test_flush_discards_beats_in_flight(dut):
    """A restart at a new PC drops the queue and the beats of requests still outstanding."""
    bus, rng, _ = await _setup(dut, latency=120)
    q = Queue(dut, 1)
    cocotb.start_soon(q.run())
    raw, words = program(rng, 16)
    bus.write(BASE, raw)
    await restart(dut, BASE)
    await qc_stream.cycles(dut.clk, 40)  # requests in flight, nothing returned yet
    assert bus.outstanding > 0, "the test needs beats in flight"
    n = len(q.words)
    await restart(dut, BASE + 8 * DESC_B)
    await FallingEdge(dut.clk)
    assert qc_stream.value(dut.dq_count) == 0, "the flush must empty the queue"
    assert len(q.words) == n, "a descriptor was popped across the flush"
    got = await collect(dut, q, 8)
    assert got == words[8:16], "stale beats reached the queue after a flush"


@cocotb.test()
async def test_restart_inside_a_beat(dut):
    """A 32-byte-aligned PC inside a wider beat drops the descriptors before it."""
    bus, rng, _ = await _setup(dut, latency=6)
    q = Queue(dut, 1)
    cocotb.start_soon(q.run())
    raw, words = program(rng, 12)
    bus.write(BASE, raw)
    for first in (1, 2, 3, 5, 0):
        await restart(dut, BASE + first * DESC_B)
        got = await collect(dut, q, 4)
        assert got == words[first : first + 4], f"restart at descriptor {first}"
        assert bus.requests[-1][0] % bus.wb == 0, "a fetch request is not beat aligned"


@cocotb.test()
async def test_restart_at_every_position_in_a_beat(dut):
    """Every 32-byte PC of a beat restarts the program there, whatever else the beat holds."""
    bus, rng, _ = await _setup(dut, latency=9)
    q = Queue(dut, 1)
    cocotb.start_soon(q.run())
    raw, words = program(rng, 20)
    bus.write(BASE, raw)
    for first in range(16):
        await restart(dut, BASE + first * DESC_B)
        got = await collect(dut, q, 4)
        assert got == words[first : first + 4], f"restart at descriptor {first}: {got[0]:#x}"
        for addr, _, _ in bus.requests:
            assert addr % bus.wb == 0, "a fetch request is not beat aligned"


@cocotb.test()
async def test_step_across_a_beat_boundary(dut):
    """One request per step, the descriptor at PC each time, walking over the beat boundaries."""
    bus, rng, _ = await _setup(dut, latency=11)
    q = Queue(dut, 1)
    cocotb.start_soon(q.run())
    raw, words = program(rng, 16)
    bus.write(BASE, raw)
    for i, want in enumerate(words[:14]):
        before = len(bus.requests)
        await restart(dut, BASE + i * DESC_B, step=True)
        got = await collect(dut, q, 1)
        assert got == [want], f"step {i}: descriptor {got[0]:#x}, expected {want:#x}"
        await qc_stream.cycles(dut.clk, 40)
        issued = len(bus.requests) - before
        assert issued == 1, f"step {i}: {issued} requests issued, expected 1"
    dut.fetch_step.value = 0


@cocotb.test()
async def test_fetch_hold_stops_new_requests(dut):
    """fetch_hold is the prefetch's share of the auto-fence: no request leaves while it is high."""
    bus, rng, _ = await _setup(dut, latency=4)
    q = Queue(dut, lambda c: (c % 6) == 0)
    cocotb.start_soon(q.run())
    raw, words = program(rng, 24)
    bus.write(BASE, raw)
    await restart(dut, BASE)
    await collect(dut, q, 2)
    dut.fetch_hold.value = 1
    await qc_stream.cycles(dut.clk, 8)  # a request already presented is still granted
    held = len(bus.requests)
    for _ in range(200):
        await FallingEdge(dut.clk)
        assert len(bus.requests) == held, "a fetch request was issued while fetch_hold was high"
    assert bus.outstanding == 0, "the fetch never went quiet under fetch_hold"
    popped = len(q.words)
    assert popped < len(words), "the whole program was popped before the hold took effect"
    dut.fetch_hold.value = 0
    await collect(dut, q, len(words) - popped)
    assert len(bus.requests) > held, "the prefetch did not resume when fetch_hold fell"
    assert q.words[: len(words)] == words, "the held prefetch lost or reordered descriptors"


@cocotb.test()
async def test_fetch_beat_strobe(dut):
    """ev_fetch_beat pulses once per returned beat, one cycle behind it."""
    bus, rng, _ = await _setup(dut, latency=5)
    q = Queue(dut, 1)
    cocotb.start_soon(q.run())
    raw, _ = program(rng, 10)
    bus.write(BASE, raw)
    strobes = 0
    await restart(dut, BASE)
    for _ in range(300):
        await RisingEdge(dut.clk)
        strobes += int(dut.ev_fetch_beat.value)
        await FallingEdge(dut.clk)
    q.stop = True  # let the queue fill so the fetch drains and stops requesting
    await FallingEdge(dut.clk)
    dut.dq_ready.value = 0
    for _ in range(200):
        await RisingEdge(dut.clk)
        strobes += int(dut.ev_fetch_beat.value)
        await FallingEdge(dut.clk)
    assert bus.outstanding == 0, "the fetch never went quiet"
    assert strobes == bus.beats, f"{strobes} strobes for {bus.beats} returned beats"
    assert strobes > 10, "the test returned too few beats to be meaningful"
