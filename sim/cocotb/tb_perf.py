"""cocotb tests of qcore_perf: the sixteen 64-bit counters against a Python model of the same
event stream, the clear and snapshot semantics, the byte and bulk adds, and the exclusive-bucket
invariant BUSY = MAC_ACTIVE + STALL_MEM + STALL_VPU + STALL_KV + STALL_SEQ + STALL_DRAIN."""

from __future__ import annotations

import os
import random

import cocotb
import qc_stream
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, Timer
from quettos import isa

WB = int(os.environ["QC_WB"])
NC = isa.PERF_COUNT
IDX = isa.PERF_INDEX
BUCKETS = ("MAC_ACTIVE", "STALL_MEM", "STALL_VPU", "STALL_KV", "STALL_SEQ", "STALL_DRAIN")
MASK64 = (1 << 64) - 1

# Every event port, with the width the module declares.
PORTS = {
    "clear": 1,
    "snapshot": 1,
    "ev_cycle": 1,
    "ev_busy": 1,
    "ev_bucket": 6,
    "ev_rd_beat": 1,
    "ev_wr_beat": 1,
    "ev_wr_bytes": 8,
    "ev_wt_valid": 1,
    "ev_wt_bytes": 40,
    "ev_macs_valid": 1,
    "ev_macs": 40,
    "ev_desc": 1,
    "ev_fetch_beat": 1,
}


class Model:
    """The sixteen counters as docs/RTL.md 3.18 defines them, live set and snapshot."""

    def __init__(self) -> None:
        self.live = [0] * NC
        self.snap = [0] * NC

    def step(self, ev: dict) -> None:
        inc = [0] * NC
        bucket = ev.get("ev_bucket", 0)
        inc[IDX["CYCLES"]] = ev.get("ev_cycle", 0)
        inc[IDX["BUSY"]] = ev.get("ev_busy", 0)
        for i, name in enumerate(BUCKETS):
            inc[IDX[name]] = (bucket >> i) & 1
        inc[IDX["RD_BEATS"]] = ev.get("ev_rd_beat", 0)
        inc[IDX["RD_BYTES"]] = WB if ev.get("ev_rd_beat", 0) else 0
        inc[IDX["WT_BYTES"]] = ev.get("ev_wt_bytes", 0) if ev.get("ev_wt_valid", 0) else 0
        inc[IDX["WR_BEATS"]] = ev.get("ev_wr_beat", 0)
        inc[IDX["WR_BYTES"]] = ev.get("ev_wr_bytes", 0)
        inc[IDX["MACS"]] = ev.get("ev_macs", 0) if ev.get("ev_macs_valid", 0) else 0
        inc[IDX["DESCRIPTORS"]] = ev.get("ev_desc", 0)
        inc[IDX["FETCH_BEATS"]] = ev.get("ev_fetch_beat", 0)
        nxt = [(self.live[i] + inc[i]) & MASK64 for i in range(NC)]
        if ev.get("clear", 0):
            self.live = [0] * NC
            self.snap = [0] * NC
        else:
            self.live = nxt
            if ev.get("snapshot", 0):
                self.snap = list(nxt)


def snapshot_of(dut) -> list[int]:
    """The sixteen counters the module presents on ``perf_snap``."""
    word = qc_stream.value(dut.perf_snap)
    return [(word >> (64 * i)) & MASK64 for i in range(NC)]


def check(dut, model: Model, where: str) -> None:
    got = snapshot_of(dut)
    for i in range(NC):
        name = next(n for n, j in IDX.items() if j == i)
        assert got[i] == model.snap[i], (
            f"{where}: PERF[{i}] {name} is {got[i]}, want {model.snap[i]}"
        )
    buckets = sum(got[IDX[n]] for n in BUCKETS)
    assert buckets == got[IDX["BUSY"]], (
        f"{where}: buckets sum to {buckets}, BUSY is {got[IDX['BUSY']]}"
    )


async def tick(dut, model: Model, **ev: int) -> None:
    """Drive one cycle of events; the module and the model both take them at the same edge."""
    for name in PORTS:
        getattr(dut, name).value = ev.get(name, 0)
    model.step(ev)
    await FallingEdge(dut.clk)


async def _setup(dut, seed: int) -> tuple[Model, random.Random]:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for name in PORTS:
        getattr(dut, name).value = 0
    await qc_stream.reset(dut, dut.rst)
    return Model(), random.Random(seed)


def busy_cycle(rng: random.Random, **ev: int) -> dict:
    """A legal busy cycle: ev_cycle and ev_busy high with exactly one bucket bit."""
    ev.update(ev_cycle=1, ev_busy=1, ev_bucket=1 << rng.randrange(6))
    return ev


@cocotb.test()
async def test_reset_state(dut):
    """Out of reset every counter and every snapshot half reads zero."""
    model, _ = await _setup(dut, 0x9E01)
    assert snapshot_of(dut) == [0] * NC
    for _ in range(4):
        await tick(dut, model, snapshot=1)
    check(dut, model, "idle")


@cocotb.test()
async def test_level_counters_and_buckets(dut):
    """CYCLES, BUSY and the six buckets count the cycles their strobes are high."""
    model, rng = await _setup(dut, 0x9E02)
    for b in range(6):
        for _ in range(b + 1):
            await tick(dut, model, ev_cycle=1, ev_busy=1, ev_bucket=1 << b)
    for _ in range(5):  # idle cycles belong to no bucket and to no counter
        await tick(dut, model)
    await tick(dut, model, snapshot=1)
    check(dut, model, "buckets")
    got = snapshot_of(dut)
    assert got[IDX["CYCLES"]] == 21 and got[IDX["BUSY"]] == 21
    for b, name in enumerate(BUCKETS):
        assert got[IDX[name]] == b + 1, f"{name} counted {got[IDX[name]]}"


@cocotb.test()
async def test_beat_and_byte_counters(dut):
    """RD_BYTES adds WB per read beat; WR_BYTES adds the strobed-byte count of its beat."""
    model, rng = await _setup(dut, 0x9E03)
    beats = 0
    written = 0
    for _ in range(64):
        rd = rng.randrange(2)
        wr = rng.randrange(2)
        n = rng.randrange(1, WB + 1) if wr else 0
        beats += rd
        written += n
        await tick(
            dut,
            model,
            **busy_cycle(rng, ev_rd_beat=rd, ev_wr_beat=wr, ev_wr_bytes=n, ev_fetch_beat=rd),
        )
    await tick(dut, model, snapshot=1)
    check(dut, model, "traffic")
    got = snapshot_of(dut)
    assert got[IDX["RD_BEATS"]] == beats and got[IDX["RD_BYTES"]] == beats * WB
    assert got[IDX["WR_BYTES"]] == written
    assert got[IDX["FETCH_BEATS"]] == beats


@cocotb.test()
async def test_bulk_adds(dut):
    """WT_BYTES and MACS take their 40-bit bulk value only in the cycles their valid is high."""
    model, rng = await _setup(dut, 0x9E04)
    wt = 0
    macs = 0
    for _ in range(48):
        wv = rng.randrange(2)
        mv = rng.randrange(2)
        wb_bytes = rng.randrange(1 << 40)
        mv_macs = rng.randrange(1 << 40)
        wt += wb_bytes if wv else 0
        macs += mv_macs if mv else 0
        await tick(
            dut,
            model,
            **busy_cycle(
                rng,
                ev_wt_valid=wv,
                ev_wt_bytes=wb_bytes,
                ev_macs_valid=mv,
                ev_macs=mv_macs,
            ),
        )
    await tick(dut, model, snapshot=1)
    check(dut, model, "bulk")
    got = snapshot_of(dut)
    assert got[IDX["WT_BYTES"]] == wt and got[IDX["MACS"]] == macs
    assert got[IDX["WT_BYTES"]] > (1 << 40), "the bulk adds must accumulate past one event"


@cocotb.test()
async def test_snapshot_includes_its_own_cycle(dut):
    """A snapshot carries the events of the cycle it is pulsed in: a HALT retire counts itself."""
    model, rng = await _setup(dut, 0x9E05)
    for _ in range(3):
        await tick(dut, model, **busy_cycle(rng, ev_desc=1))
    await tick(dut, model, snapshot=1, **busy_cycle(rng, ev_desc=1))
    check(dut, model, "snapshot")
    assert snapshot_of(dut)[IDX["DESCRIPTORS"]] == 4, "the retire of the snapshot cycle is missing"


@cocotb.test()
async def test_snapshot_holds_until_the_next_pulse(dut):
    """perf_snap is stable while the live counters run on, and moves only on the next snapshot."""
    model, rng = await _setup(dut, 0x9E06)
    for _ in range(6):
        await tick(dut, model, **busy_cycle(rng, ev_desc=1))
    await tick(dut, model, snapshot=1)
    held = snapshot_of(dut)
    for _ in range(20):
        await tick(dut, model, **busy_cycle(rng, ev_desc=1, ev_rd_beat=1))
        assert snapshot_of(dut) == held, "perf_snap moved between snapshots"
    await tick(dut, model, snapshot=1)
    check(dut, model, "second snapshot")
    assert snapshot_of(dut) != held, "the second snapshot did not take the new values"


@cocotb.test()
async def test_clear_zeroes_both_sets(dut):
    """START's clear zeroes the live counters and the snapshot, and counts nothing of its cycle."""
    model, rng = await _setup(dut, 0x9E07)
    for _ in range(12):
        await tick(dut, model, **busy_cycle(rng, ev_desc=1, ev_rd_beat=1))
    await tick(dut, model, snapshot=1)
    assert any(snapshot_of(dut)), "nothing counted"
    await tick(dut, model, clear=1, **busy_cycle(rng, ev_desc=1))
    check(dut, model, "clear")
    assert snapshot_of(dut) == [0] * NC
    await tick(dut, model, snapshot=1)
    check(dut, model, "after clear")
    assert snapshot_of(dut) == [0] * NC, "the live counters kept a pre-clear value"


@cocotb.test()
async def test_random_stream(dut):
    """A long random legal stream, with snapshots and clears, against the model cycle by cycle."""
    model, rng = await _setup(dut, 0x9E08)
    for _ in range(1500):
        busy = rng.random() < 0.8
        ev = {
            "ev_rd_beat": rng.randrange(2),
            "ev_wr_beat": 0,
            "ev_wr_bytes": 0,
            "ev_desc": 1 if rng.random() < 0.05 else 0,
            "ev_fetch_beat": rng.randrange(2),
        }
        if rng.random() < 0.4:
            ev["ev_wr_beat"] = 1
            ev["ev_wr_bytes"] = rng.randrange(1, WB + 1)
        if rng.random() < 0.05:
            ev["ev_wt_valid"] = 1
            ev["ev_wt_bytes"] = rng.randrange(1 << 40)
        if rng.random() < 0.05:
            ev["ev_macs_valid"] = 1
            ev["ev_macs"] = rng.randrange(1 << 40)
        if busy:
            ev = busy_cycle(rng, **ev)
        if rng.random() < 0.02:
            ev["snapshot"] = 1
        elif rng.random() < 0.01:
            ev["clear"] = 1
        await tick(dut, model, **ev)
        if ev.get("snapshot") or ev.get("clear"):
            check(dut, model, "random stream")
    await tick(dut, model, snapshot=1)
    check(dut, model, "random stream end")


@cocotb.test()
async def test_buckets_are_exclusive(dut):
    """Every busy cycle lands in exactly one bucket, so the six counters sum to BUSY.

    Each bucket gets a different number of cycles with idle cycles between them, so a cycle
    claimed by two buckets, or one bucket folded into another, shows up both as a wrong
    per-bucket count and as a sum above BUSY. The module's own one-hot detector is read back
    every cycle; it stops the simulation through $error if a stream ever violates the rule.
    """
    counts = (3, 5, 7, 11, 13, 17)
    model, _ = await _setup(dut, 0x9E09)
    for b, n in enumerate(counts):
        for _ in range(n):
            await tick(dut, model, ev_cycle=1, ev_busy=1, ev_bucket=1 << b)
            assert qc_stream.value(dut.bucket_ones) == 1, f"{BUCKETS[b]}: bucket bits set"
        for _ in range(2):
            await tick(dut, model)
            assert qc_stream.value(dut.bucket_ones) == 0, "a bucket bit outside a busy cycle"
    await tick(dut, model, snapshot=1)
    check(dut, model, "exclusive buckets")
    got = snapshot_of(dut)
    for b, name in enumerate(BUCKETS):
        assert got[IDX[name]] == counts[b], f"{name} counted {got[IDX[name]]}, want {counts[b]}"
    total = sum(counts)
    assert sum(got[IDX[n]] for n in BUCKETS) == total, "the buckets do not sum to the busy cycles"
    assert got[IDX["BUSY"]] == total and got[IDX["CYCLES"]] == total
    assert qc_stream.value(dut.bucket_sum) == total, "the module's own sum disagrees"


async def _delta(dut, model: Model, **ev: int) -> dict[str, int]:
    """The counters one cycle of ``ev`` adds, as ``{name: increment}`` over the empty cycle."""
    await tick(dut, model, snapshot=1)
    before = snapshot_of(dut)
    await tick(dut, model, snapshot=1, **ev)
    after = snapshot_of(dut)
    return {n: after[i] - before[i] for n, i in IDX.items() if after[i] != before[i]}


@cocotb.test()
async def test_every_counter_has_exactly_one_source(dut):
    """Each event port moves the counters the source table of docs/RTL.md 3.18 names for it.

    A cycle is driven with one port set on top of a legal busy cycle and the whole
    snapshot is differenced, so a counter wired to a second strobe, or a strobe
    reaching a counter it does not own, shows up as an extra key.
    """
    model, _ = await _setup(dut, 0x9E0A)
    for i, name in enumerate(BUCKETS):
        got = await _delta(dut, model, ev_cycle=1, ev_busy=1, ev_bucket=1 << i)
        assert got == {"CYCLES": 1, "BUSY": 1, name: 1}, f"bucket {name}: {got}"
    live = {"CYCLES": 1, "BUSY": 1, "MAC_ACTIVE": 1}
    cases: list[tuple[dict[str, int], dict[str, int]]] = [
        ({"ev_rd_beat": 1}, {"RD_BEATS": 1, "RD_BYTES": WB}),
        ({"ev_wr_beat": 1, "ev_wr_bytes": 5}, {"WR_BEATS": 1, "WR_BYTES": 5}),
        ({"ev_wr_bytes": 7}, {"WR_BYTES": 7}),  # the arbiter drives 0 without a beat
        ({"ev_wt_valid": 1, "ev_wt_bytes": 1 << 33}, {"WT_BYTES": 1 << 33}),
        ({"ev_wt_bytes": 1 << 33}, {}),  # a bulk value without its valid adds nothing
        ({"ev_macs_valid": 1, "ev_macs": 1 << 35}, {"MACS": 1 << 35}),
        ({"ev_macs": 1 << 35}, {}),
        ({"ev_desc": 1}, {"DESCRIPTORS": 1}),
        ({"ev_fetch_beat": 1}, {"FETCH_BEATS": 1}),
    ]
    for ev, want in cases:
        got = await _delta(dut, model, ev_cycle=1, ev_busy=1, ev_bucket=1, **ev)
        assert got == {**live, **want}, f"{ev}: moved {got}, expected {live | want}"
    check(dut, model, "one source per counter")


@cocotb.test()
async def test_the_one_hot_detector_counts_every_claim(dut):
    """``bucket_ones`` is the population count the module's own check compares against one.

    A bucket mask is presented and withdrawn between two rising edges, so the
    registered check never samples an illegal one: what is proved here is that a
    second claim is counted as a second claim rather than folded into the first.
    A mask that really reached a rising edge stops the simulation through the
    module's ``$error``, and the test that produced it fails with it.
    """
    model, _ = await _setup(dut, 0x9E0B)
    masks = ((0, 0), (0b1, 1), (0b100000, 1), (0b11, 2), (0b010010, 2), (0b111111, 6))
    for mask, ones in masks:
        await FallingEdge(dut.clk)
        dut.ev_bucket.value = mask
        await Timer(1, "ns")
        got = qc_stream.value(dut.bucket_ones)
        dut.ev_bucket.value = 0
        await Timer(1, "ns")
        assert got == ones, f"mask {mask:#08b}: bucket_ones {got}, expected {ones}"
        assert qc_stream.value(dut.bucket_ones) == 0, "the mask outlived its half cycle"
    await tick(dut, model, snapshot=1)
    check(dut, model, "detector")
    assert snapshot_of(dut) == [0] * NC, "an idle cycle counted"
