"""cocotb tests of qcore_mem_arb: read priority with the reserved fetch slot, tag routing of
returned beats, the write mux with the outstanding-ack counter, and the PERF strobes, against
the QMEM bus model at latencies 1, 32 and 200 with response gaps."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import cocotb
import qc_stream
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from cocotb.types import Logic
from qc_qmem import QmemModel
from qc_stream import ValidReadyDriver

WB = 16
DW = WB * 8
MAX_BURST = 64
TAG_FETCH, TAG_WEIGHT, TAG_META, TAG_VPU = 0, 1, 2, 3  # qcore_pkg read tags
REQUESTER_OF_TAG = {TAG_FETCH: "f", TAG_WEIGHT: "s", TAG_META: "s", TAG_VPU: "v"}
READ_BASE = 0x0010_0000  # read image; writes go elsewhere so read data stays stable
WRITE_BASE = 0x0020_0000
TIMEOUT = 200_000

Req = tuple[int, int, int]  # (addr, len, tag)


def _val(sig) -> int:
    """Unsigned value of a signal, one bit (``Logic``) or wider (``LogicArray``)."""
    v = sig.value
    return int(v) if isinstance(v, Logic) else int(v.to_unsigned())


class Monitor:
    """Records every transfer of a valid/ready port at rising edges (``ready`` None: valid only)."""

    def __init__(self, clk, valid, ready, fields: Mapping[str, object]) -> None:
        self.clk, self.valid, self.ready, self.fields = clk, valid, ready, dict(fields)
        self.seen: list[dict[str, int]] = []

    async def run(self) -> None:
        while True:
            await RisingEdge(self.clk)
            if int(self.valid.value) == 1 and (self.ready is None or int(self.ready.value) == 1):
                self.seen.append({n: _val(s) for n, s in self.fields.items()})


class PulseCounter:
    """Counts rising edges at which ``sig`` is 1 and sums ``bytes_sig`` on those edges."""

    def __init__(self, clk, sig, bytes_sig=None) -> None:
        self.clk, self.sig, self.bytes_sig = clk, sig, bytes_sig
        self.count = 0
        self.total = 0

    async def run(self) -> None:
        while True:
            await RisingEdge(self.clk)
            if int(self.sig.value) == 1:
                self.count += 1
                if self.bytes_sig is not None:
                    self.total += _val(self.bytes_sig)
            elif self.bytes_sig is not None:
                assert _val(self.bytes_sig) == 0, "ev_wr_bytes must be 0 without ev_wr_beat"


class Bench:
    """The DUT with the QMEM model, one driver per requester and one monitor per sink."""

    def __init__(self, dut, model: QmemModel) -> None:
        self.dut, self.model = dut, model
        self.req = {
            p: ValidReadyDriver(
                dut.clk,
                getattr(dut, f"{p}_req_valid"),
                getattr(dut, f"{p}_req_ready"),
                {
                    "addr": getattr(dut, f"{p}_req_addr"),
                    "len": getattr(dut, f"{p}_req_len"),
                    "tag": getattr(dut, f"{p}_req_tag"),
                },
            )
            for p in "svf"
        }
        self.wr = {
            p: ValidReadyDriver(
                dut.clk,
                getattr(dut, f"{p}_wr_valid"),
                getattr(dut, f"{p}_wr_ready"),
                {
                    "addr": getattr(dut, f"{p}_wr_addr"),
                    "data": getattr(dut, f"{p}_wr_data"),
                    "strb": getattr(dut, f"{p}_wr_strb"),
                },
            )
            for p in "kd"
        }
        self.sink = {}
        for tag, sig in (
            (TAG_FETCH, dut.rdf_valid),
            (TAG_WEIGHT, dut.rdw_valid),
            (TAG_META, dut.rdm_valid),
            (TAG_VPU, dut.rdv_valid),
        ):
            mon = Monitor(dut.clk, sig, None, {"data": dut.rdd_data, "last": dut.rdd_last})
            cocotb.start_soon(mon.run())
            self.sink[tag] = mon
        self.rd_events = PulseCounter(dut.clk, dut.ev_rd_beat)
        self.wr_events = PulseCounter(dut.clk, dut.ev_wr_beat, dut.ev_wr_bytes)
        cocotb.start_soon(self.rd_events.run())
        cocotb.start_soon(self.wr_events.run())

    async def send_reqs(self, p: str, reqs: Sequence[Req]) -> None:
        for addr, length, tag in reqs:
            await self.req[p].send({"addr": addr, "len": length, "tag": tag})

    async def send_writes(self, p: str, writes: Sequence[tuple[int, int, int]]) -> None:
        for addr, data, strb in writes:
            await self.wr[p].send({"addr": addr, "data": data, "strb": strb})

    async def drain(self, extra: int = 8) -> None:
        """Wait until every requester is idle and the model has nothing in flight."""
        for _ in range(TIMEOUT):
            await FallingEdge(self.dut.clk)
            idle = all(int(d.valid.value) == 0 for d in (*self.req.values(), *self.wr.values()))
            if idle and self.model.outstanding == 0 and self.model.writes_outstanding == 0:
                break
        else:
            raise AssertionError("bus did not drain")
        await qc_stream.cycles(self.dut.clk, extra)

    def expected_beats(self) -> dict[int, list[dict[str, int]]]:
        """Per tag, the beats of the accepted requests in acceptance order."""
        out: dict[int, list[dict[str, int]]] = {t: [] for t in range(4)}
        for addr, length, tag in self.model.requests:
            for i in range(length):
                out[tag].append(
                    {"data": self.model.beat(addr + i * WB), "last": int(i == length - 1)}
                )
        return out

    def check_reads(self) -> None:
        exp = self.expected_beats()
        for tag in range(4):
            got = self.sink[tag].seen
            assert len(got) == len(exp[tag]), f"tag {tag}: {len(got)} of {len(exp[tag])} beats"
            for i, (g, e) in enumerate(zip(got, exp[tag], strict=True)):
                assert g == e, f"tag {tag} beat {i}: {g} != {e}"
        total = sum(len(v) for v in exp.values())
        assert total == self.model.rd_beats
        assert self.rd_events.count == total, f"ev_rd_beat {self.rd_events.count} != {total}"

    def check_writes(self, writes: Sequence[tuple[int, int, int]]) -> None:
        """Every write landed with its strobes; the strobes and beats match the PERF strobes."""
        for addr, data, strb in writes:
            raw = data.to_bytes(WB, "little")
            got = self.model.read_bytes(addr, WB)
            for j in range(WB):
                if (strb >> j) & 1:
                    assert got[j] == raw[j], f"write {addr:#x} byte {j}"
        assert self.model.wr_beats == len(writes)
        assert self.wr_events.count == self.model.wr_beats, "ev_wr_beat count"
        assert self.wr_events.total == self.model.wr_bytes, "ev_wr_bytes sum"
        assert _val(self.dut.wr_idle) == 1


async def _setup(dut, latency: int, bw_div: int = 1, wr_ready=1) -> tuple[Bench, random.Random]:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for p in "svf":
        for f in ("valid", "addr", "len", "tag"):
            getattr(dut, f"{p}_req_{f}").value = 0
    for p in "kd":
        for f in ("valid", "addr", "data", "strb"):
            getattr(dut, f"{p}_wr_{f}").value = 0
    dut.dq_count.value = 0
    model = QmemModel(dut, wb=WB, latency=latency, bw_div=bw_div, wr_ready_pattern=wr_ready)
    rng = random.Random(0xA5B + latency)
    model.write_bytes(READ_BASE, rng.randbytes(1 << 16))
    await qc_stream.reset(dut, dut.rst)
    cocotb.start_soon(model.run())
    bench = Bench(dut, model)
    await FallingEdge(dut.clk)
    return bench, rng


def _bursts(rng: random.Random, tag: int, count: int, max_len: int) -> list[Req]:
    return [
        (READ_BASE + rng.randrange(0, (1 << 16) - 256 * WB), rng.randrange(1, max_len + 1), tag)
        for _ in range(count)
    ]


async def _run_pattern(bench: Bench, per_requester: dict[str, list[Req]]) -> None:
    tasks = [cocotb.start_soon(bench.send_reqs(p, reqs)) for p, reqs in per_requester.items()]
    for t in tasks:
        await t
    await bench.drain()


@cocotb.test()
async def fetch_slot_alternates_with_full_bursts(dut):
    """Stream 64-beat bursts against a waiting fetch: grants alternate s, f, s, f (dq_count < 4)."""
    bench, _ = await _setup(dut, latency=32)
    s = [(READ_BASE + i * 64 * WB, 64, TAG_WEIGHT) for i in range(6)]
    f = [(READ_BASE + 0x8000 + i * 2 * WB, 2, TAG_FETCH) for i in range(6)]
    await _run_pattern(bench, {"s": s, "f": f})
    tags = [t for _, _, t in bench.model.requests]
    assert tags == [TAG_WEIGHT, TAG_FETCH] * 6, tags
    assert bench.model.requests == [r for pair in zip(s, f, strict=True) for r in pair]
    bench.check_reads()


@cocotb.test()
async def fetch_waits_while_queue_is_deep(dut):
    """With dq_count >= 4 the stream keeps the port; the fetch is granted once it goes idle."""
    bench, _ = await _setup(dut, latency=32)
    dut.dq_count.value = 4
    s = [(READ_BASE + i * 64 * WB, 64, TAG_WEIGHT) for i in range(6)]
    f = [(READ_BASE + 0x8000 + i * 2 * WB, 2, TAG_FETCH) for i in range(6)]
    await _run_pattern(bench, {"s": s, "f": f})
    tags = [t for _, _, t in bench.model.requests]
    assert tags == [TAG_WEIGHT] * 6 + [TAG_FETCH] * 6, tags
    bench.check_reads()
    # a queue that drops below 4 hands the slot to the fetch at the next opportunity
    dut.dq_count.value = 3
    s2 = [(READ_BASE + 0x4000 + i * 64 * WB, 64, TAG_WEIGHT) for i in range(3)]
    f2 = [(READ_BASE + 0xC000 + i * 2 * WB, 2, TAG_FETCH) for i in range(3)]
    await _run_pattern(bench, {"s": s2, "f": f2})
    tags = [t for _, _, t in bench.model.requests[12:]]
    assert tags == [TAG_WEIGHT, TAG_FETCH] * 3, tags
    bench.check_reads()


@cocotb.test()
async def vpu_over_fetch_without_stream_beats(dut):
    """Without stream traffic the fetch slot never opens: VPU requests win while they are valid."""
    bench, _ = await _setup(dut, latency=32)
    v = [(READ_BASE + i * 8 * WB, 8, TAG_VPU) for i in range(4)]
    f = [(READ_BASE + 0x8000 + i * WB, 1, TAG_FETCH) for i in range(2)]
    await _run_pattern(bench, {"v": v, "f": f})
    tags = [t for _, _, t in bench.model.requests]
    assert tags == [TAG_VPU] * 4 + [TAG_FETCH] * 2, tags
    bench.check_reads()


@cocotb.test()
async def reserved_fetch_beats_vpu(dut):
    """After a full stream burst the reserved fetch is granted before a waiting VPU request."""
    bench, _ = await _setup(dut, latency=32)
    s = [(READ_BASE, 64, TAG_WEIGHT)]
    v = [(READ_BASE + 0x4000, 8, TAG_VPU)]
    f = [(READ_BASE + 0x8000, 2, TAG_FETCH)]
    await _run_pattern(bench, {"s": s, "v": v, "f": f})
    tags = [t for _, _, t in bench.model.requests]
    assert tags == [TAG_WEIGHT, TAG_FETCH, TAG_VPU], tags
    bench.check_reads()


@cocotb.test()
async def fetch_gap_is_bounded(dut):
    """With 48-beat stream bursts the fetch is granted every 96 stream beats (two bursts)."""
    bench, _ = await _setup(dut, latency=32)
    s = [(READ_BASE + i * 48 * WB, 48, TAG_WEIGHT) for i in range(8)]
    f = [(READ_BASE + 0x8000 + i * 2 * WB, 2, TAG_FETCH) for i in range(4)]
    await _run_pattern(bench, {"s": s, "f": f})
    tags = [t for _, _, t in bench.model.requests]
    assert tags == [TAG_WEIGHT, TAG_WEIGHT, TAG_FETCH] * 4, tags
    gap, worst = 0, 0
    for _, length, tag in bench.model.requests:
        if tag == TAG_FETCH:
            worst, gap = max(worst, gap), 0
        else:
            gap += length
    assert worst == 96 and worst <= MAX_BURST + 48 - 1
    bench.check_reads()


async def _routing(dut, latency: int, bw_div: int) -> None:
    bench, rng = await _setup(dut, latency=latency, bw_div=bw_div)
    per: dict[str, list[Req]] = {"s": [], "v": [], "f": []}
    for _ in range(48):
        tag = rng.choice((TAG_FETCH, TAG_WEIGHT, TAG_META, TAG_VPU))
        per[REQUESTER_OF_TAG[tag]] += _bursts(rng, tag, 1, 2 if tag == TAG_FETCH else 8)
    dq = cocotb.start_soon(_wiggle_dq(dut, rng))
    await _run_pattern(bench, per)
    dq.cancel()
    accepted = {p: [r for r in bench.model.requests if REQUESTER_OF_TAG[r[2]] == p] for p in per}
    assert accepted == per, "each requester's requests are accepted in its own order"
    bench.check_reads()
    dut._log.info(
        "routing LAT=%d bw_div=%d: %d requests, %d beats",
        latency,
        bw_div,
        len(bench.model.requests),
        bench.model.rd_beats,
    )


async def _wiggle_dq(dut, rng: random.Random) -> None:
    while True:
        dut.dq_count.value = rng.randrange(0, 9)
        await qc_stream.cycles(dut.clk, rng.randrange(1, 40))


@cocotb.test()
async def routing_lat1(dut):
    """Every beat reaches the sink of its tag, in order and bit-exact, at latency 1."""
    await _routing(dut, 1, 1)


@cocotb.test()
async def routing_lat32(dut):
    """Every beat reaches the sink of its tag, in order and bit-exact, at latency 32."""
    await _routing(dut, 32, 1)


@cocotb.test()
async def routing_lat200_gaps(dut):
    """Latency 200 with one beat every 3 cycles: routing and order hold."""
    await _routing(dut, 200, 3)


def _writes(rng: random.Random, base: int, count: int) -> list[tuple[int, int, int]]:
    # The first two strobes are the extremes: every byte and one byte. A random
    # draw reaches an all-ones strobe about once in 2**WB writes, so a truncated
    # byte count would otherwise pass the whole bench.
    writes = [
        (base + i * 64 + rng.randrange(0, 48), rng.getrandbits(DW), rng.randrange(1, 1 << WB))
        for i in range(count)
    ]
    if count > 0:
        writes[0] = (writes[0][0], writes[0][1], (1 << WB) - 1)
    if count > 1:
        writes[1] = (writes[1][0], writes[1][1], 1)
    return writes


@cocotb.test()
async def write_mux_priority_and_acks(dut):
    """KV writes win over dump writes, every strobed byte lands, wr_idle tracks the acks."""
    rng0 = random.Random(7)
    bench, rng = await _setup(dut, latency=32, wr_ready=lambda t: rng0.random() < 0.6)
    k = _writes(rng, WRITE_BASE, 12)
    d = _writes(rng, WRITE_BASE + 0x10000, 12)
    idle_seen_low = False

    async def watch_idle() -> None:
        nonlocal idle_seen_low
        while True:
            await FallingEdge(dut.clk)
            if bench.model.writes_outstanding > 0:
                assert _val(dut.wr_idle) == 0, "wr_idle high with writes awaiting their ack"
                idle_seen_low = True

    watcher = cocotb.start_soon(watch_idle())
    tk = cocotb.start_soon(bench.send_writes("k", k))
    td = cocotb.start_soon(bench.send_writes("d", d))
    await tk
    await td
    await bench.drain(extra=40)
    watcher.cancel()
    assert idle_seen_low
    accepted = bench.model.writes
    assert accepted[0][0] == k[0][0], "the KV writer is accepted first"
    k_order = [w for w in accepted if w[0] < WRITE_BASE + 0x10000]
    d_order = [w for w in accepted if w[0] >= WRITE_BASE + 0x10000]
    assert [w[0] for w in k_order] == [w[0] for w in k]
    assert [w[0] for w in d_order] == [w[0] for w in d]
    # while the KV writer had a write pending, no dump write was accepted ahead of it
    assert [w[0] for w in accepted[: len(k)]] == [w[0] for w in k]
    bench.check_writes(k + d)


@cocotb.test()
async def wr_idle_falls_in_the_cycle_the_write_is_accepted(dut):
    """A read granted in the cycle a write is accepted must not pass it, so wr_idle is already low.

    The stage registers an accepted write on the next edge, so a wr_idle built from the stage and
    the ack counters alone still reads idle in the cycle it takes the write -- and the descriptor
    fetch that wr_idle releases is granted in that same cycle, ahead of the write.
    """
    bench, rng = await _setup(dut, latency=32)
    await FallingEdge(dut.clk)
    assert _val(dut.wr_idle) == 1, "the write path is not idle before any write"
    for p in "kd":
        addr, data, strb = _writes(rng, WRITE_BASE + (0x10000 if p == "d" else 0), 1)[0]
        getattr(dut, f"{p}_wr_addr").value = addr
        getattr(dut, f"{p}_wr_data").value = data
        getattr(dut, f"{p}_wr_strb").value = strb
        getattr(dut, f"{p}_wr_valid").value = 1
        await Timer(1, unit="ns")  # the same cycle, before the edge that takes the write
        assert _val(getattr(dut, f"{p}_wr_ready")) == 1, f"the stage did not take the {p} write"
        assert _val(dut.wr_idle) == 0, (
            f"wr_idle high in the cycle the {p} write is accepted: a fetch granted now "
            "would pass the write"
        )
        await FallingEdge(dut.clk)
        getattr(dut, f"{p}_wr_valid").value = 0
        assert _val(dut.wr_idle) == 0, "wr_idle high with the write in the stage"
        for _ in range(200):
            await FallingEdge(dut.clk)
            if _val(dut.wr_idle) == 1:
                break
        else:
            raise AssertionError(f"wr_idle never rose again after the {p} write")
    await bench.drain()
    assert bench.model.wr_beats == 2, f"{bench.model.wr_beats} writes reached the memory"


@cocotb.test()
async def wr_idle_timing(dut):
    """wr_idle is low from the cycle a write is accepted and rises the cycle after its ack."""
    bench, rng = await _setup(dut, latency=32)
    assert _val(dut.wr_idle) == 1
    w = _writes(rng, WRITE_BASE, 1)[0]
    await bench.send_writes("k", [w])
    assert _val(dut.wr_idle) == 0
    low = 0
    while _val(dut.wr_idle) == 0:
        low += 1
        await FallingEdge(dut.clk)
        assert low < 100
    # accepted by the stage, then by the memory, then the ack after the latency
    assert 32 <= low <= 36, low
    await bench.drain()
    bench.check_writes([w])


@cocotb.test()
async def random_traffic(dut):
    """Random valid patterns on all five requesters with a wandering dq_count and response gaps."""
    rng0 = random.Random(11)
    bench, rng = await _setup(dut, latency=32, bw_div=2, wr_ready=lambda t: rng0.random() < 0.7)
    per: dict[str, list[Req]] = {"s": [], "v": [], "f": []}
    for _ in range(80):
        tag = rng.choice((TAG_FETCH, TAG_WEIGHT, TAG_META, TAG_VPU))
        per[REQUESTER_OF_TAG[tag]] += _bursts(rng, tag, 1, 2 if tag == TAG_FETCH else 16)
    k = _writes(rng, WRITE_BASE, 20)
    d = _writes(rng, WRITE_BASE + 0x10000, 20)

    async def gappy(p: str, reqs: list[Req]) -> None:
        for r in reqs:
            await qc_stream.cycles(dut.clk, rng.randrange(0, 6))
            await bench.send_reqs(p, [r])

    async def gappy_writes(p: str, ws: list[tuple[int, int, int]]) -> None:
        for w in ws:
            await qc_stream.cycles(dut.clk, rng.randrange(0, 10))
            await bench.send_writes(p, [w])

    dq = cocotb.start_soon(_wiggle_dq(dut, rng))
    tasks = [cocotb.start_soon(gappy(p, reqs)) for p, reqs in per.items()]
    tasks += [cocotb.start_soon(gappy_writes("k", k)), cocotb.start_soon(gappy_writes("d", d))]
    for t in tasks:
        await t
    await bench.drain(extra=40)
    dq.cancel()
    accepted = {p: [r for r in bench.model.requests if REQUESTER_OF_TAG[r[2]] == p] for p in per}
    assert accepted == per
    bench.check_reads()
    bench.check_writes(k + d)
