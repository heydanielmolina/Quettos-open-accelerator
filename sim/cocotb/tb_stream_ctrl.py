"""cocotb tests of qcore_stream_ctrl: GEMV weight and meta streams at latencies 1, 32 and 200
with random response gaps and backpressure, the FIFO reservation bound, weight-port utilization
at latency 32, the EMBED gather, unit_meta, empty and back-to-back descriptors. The QMEM bus
model answers the controller's request port directly and routes its beats by tag."""

from __future__ import annotations

import random
from collections.abc import Mapping

import cocotb
import qc_numerics
import qc_stream
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge
from cocotb.types import Logic
from qc_qmem import QmemModel
from qc_stream import ReadySource
from quettos import isa
from quettos.numerics import SFloat

WB = 16
DW = WB * 8
FIFO_BEATS = 128
META_FIFO_BEATS = 16
MAX_BURST = 64
TAG_WEIGHT, TAG_META = 1, 2  # qcore_pkg read tags
OP_GEMV, OP_EMBED = int(isa.Opcode.GEMV), int(isa.Opcode.EMBED)
WEIGHT_BASE = 0x0020_0000
META_BASE = 0x0030_0000
TIMEOUT = 300_000
WS_FIELDS = ("data", "k", "tile", "tile_start", "tile_end", "nvalid", "last", "embed")


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


# ---- QMEM model on the controller's own ports


class _Const:
    """A signal stand-in that reads as 0: the write channel the controller does not have."""

    value = 0


class _Sink:
    """A signal stand-in that accepts writes."""

    def __init__(self) -> None:
        self.value = 0


class _Route:
    """Holds rd_data_valid and rd_data_tag and drives rdw_valid / rdm_valid from them."""

    def __init__(self, dut) -> None:
        self.dut, self.valid, self.tag = dut, 0, 0

    def apply(self) -> None:
        self.dut.rdw_valid.value = int(self.valid == 1 and self.tag == TAG_WEIGHT)
        self.dut.rdm_valid.value = int(self.valid == 1 and self.tag == TAG_META)


class _RouteField:
    def __init__(self, route: _Route, name: str) -> None:
        self._route, self._name = route, name

    @property
    def value(self) -> int:
        return getattr(self._route, self._name)

    @value.setter
    def value(self, v: int) -> None:
        setattr(self._route, self._name, int(v))
        self._route.apply()


class StreamQmem(QmemModel):
    """QmemModel bound to s_req_* / rdw_valid / rdm_valid / rd_data / rd_data_last."""

    def __post_init__(self) -> None:
        route = _Route(self.dut)
        self._map = {
            "rd_req_valid": self.dut.s_req_valid,
            "rd_req_ready": self.dut.s_req_ready,
            "rd_req_addr": self.dut.s_req_addr,
            "rd_req_len": self.dut.s_req_len,
            "rd_req_tag": self.dut.s_req_tag,
            "rd_data_valid": _RouteField(route, "valid"),
            "rd_data_tag": _RouteField(route, "tag"),
            "rd_data": self.dut.rd_data,
            "rd_data_last": self.dut.rd_data_last,
            "wr_valid": _Const(),
            "wr_addr": _Const(),
            "wr_data": _Const(),
            "wr_strb": _Const(),
            "wr_ready": _Sink(),
            "wr_ack": _Sink(),
        }
        super().__post_init__()

    def _sig(self, name: str):
        return self._map[name]


# ---- descriptor, expected streams


class Desc:
    """The command bundle of one GEMV / EMBED as the dispatcher presents it."""

    def __init__(
        self,
        op: int,
        addr_a: int,
        addr_m: int,
        n: int,
        k: int,
        k_stride: int,
        unit_meta: int = 0,
        tok: int = 0,
    ) -> None:
        self.op, self.addr_a, self.addr_m = op, addr_a, addr_m
        self.n, self.k, self.k_stride = n, k, k_stride
        self.unit_meta, self.tok = unit_meta, tok

    @property
    def embed(self) -> bool:
        return self.op == OP_EMBED

    @property
    def n_eff(self) -> int:
        return self.k if self.embed else self.n

    @property
    def tiles(self) -> int:
        return -(-self.n_eff // WB)

    def nvalid(self, t: int) -> int:
        return min(WB, self.n_eff - t * WB)

    def meta(self) -> bool:
        return self.embed or not self.unit_meta

    def requests(self) -> list[tuple[int, int, int]]:
        """The bursts the controller issues, in order: (addr, len, tag)."""
        out: list[tuple[int, int, int]] = []
        if self.n_eff == 0 or self.k == 0:
            return out
        for t in range(1 if self.embed else self.tiles):  # an EMBED streams one tile
            if self.embed:  # the row's meta record comes first
                base = (self.addr_a + (self.tok // WB) * self.k * WB) & 0xFFFFFFFF
                out.append(((self.addr_m + self.tok * 8) & 0xFFFFFFFF, 1, TAG_META))
            else:
                base = (self.addr_a + t * self.k_stride * WB) & 0xFFFFFFFF
            rem, addr = self.k, base
            while rem:
                length = min(MAX_BURST, rem)
                out.append((addr, length, TAG_WEIGHT))
                addr, rem = (addr + length * WB) & 0xFFFFFFFF, rem - length
            if self.meta() and not self.embed:
                out.append(((self.addr_m + t * WB * 8) & 0xFFFFFFFF, 8, TAG_META))
        return out

    def beats(self, mem: QmemModel) -> list[dict[str, int]]:
        """The weight stream: one dict of WS_FIELDS per beat."""
        out: list[dict[str, int]] = []
        if self.n_eff == 0 or self.k == 0:
            return out
        if self.embed:
            base = self.addr_a + (self.tok // WB) * self.k * WB + self.tok % WB
            q = bytes(mem.read_bytes(base + i * WB, 1)[0] for i in range(self.k))
            for p in range(self.tiles):
                chunk = q[p * WB : (p + 1) * WB].ljust(WB, b"\0")
                out.append(
                    {
                        "data": int.from_bytes(chunk, "little"),
                        "k": 0,
                        "tile": p,
                        "tile_start": 1,
                        "tile_end": 1,
                        "nvalid": self.nvalid(p),
                        "last": int(p == self.tiles - 1),
                        "embed": 1,
                    }
                )
            return out
        for t in range(self.tiles):
            for kk in range(self.k):
                out.append(
                    {
                        "data": mem.beat(self.addr_a + (t * self.k_stride + kk) * WB),
                        "k": kk,
                        "tile": t,
                        "tile_start": int(kk == 0),
                        "tile_end": int(kk == self.k - 1),
                        "nvalid": self.nvalid(t),
                        "last": int(t == self.tiles - 1 and kk == self.k - 1),
                        "embed": 0,
                    }
                )
        return out

    def records(self, mem: QmemModel) -> list[int]:
        """The meta side-stream: 56-bit records in channel order, nvalid per tile."""
        if self.n_eff == 0 or self.k == 0 or not self.meta():
            return []
        if self.embed:
            return [_record_at(mem, self.addr_m + self.tok * 8)]
        return [
            _record_at(mem, self.addr_m + (t * WB + j) * 8)
            for t in range(self.tiles)
            for j in range(self.nvalid(t))
        ]


def _record_at(mem: QmemModel, addr: int) -> int:
    raw = mem.read_bytes(addr, isa.META_BYTES)
    bias = int.from_bytes(raw[0:4], "little", signed=True)
    m = int.from_bytes(raw[4:6], "little")
    e = int.from_bytes(raw[6:7], "little", signed=True)
    return qc_numerics.meta_record56(bias, SFloat(m, e))


def _fill_meta(mem: QmemModel, rng: random.Random, addr: int, count: int) -> None:
    """``count`` random meta records at ``addr`` through the harness packing."""
    blob = b"".join(
        qc_numerics.meta_bytes(
            rng.randrange(-(1 << 31), 1 << 31),
            SFloat(rng.randrange(1 << 15, 1 << 16), rng.randrange(-128, 128)),
        )
        for _ in range(count)
    )
    mem.write_bytes(addr, blob)


# ---- bench


class Bench:
    def __init__(self, dut, model: QmemModel, rng: random.Random) -> None:
        self.dut, self.model, self.rng = dut, model, rng
        self.ws = Monitor(
            dut.clk, dut.ws_valid, dut.ws_ready, {f: getattr(dut, f"ws_{f}") for f in WS_FIELDS}
        )
        self.meta = Monitor(dut.clk, dut.meta_valid, dut.meta_ready, {"data": dut.meta_data})
        self.ws_ready = ReadySource(dut.clk, dut.ws_ready, 1)
        self.meta_ready = ReadySource(dut.clk, dut.meta_ready, 1)
        self.max_w = 0
        self.max_m = 0
        self.busy_cycles = 0
        self.expected_beats = 0
        self.check_done = False  # stream_done is checked only while a descriptor is in flight
        for c in (self.ws.run(), self.meta.run(), self.ws_ready.run(), self.meta_ready.run()):
            cocotb.start_soon(c)
        cocotb.start_soon(self._watch())

    async def _watch(self) -> None:
        """Per-cycle invariants: reservation bound, payload hold, stream_done, busy count."""
        prev_ws = prev_meta = None
        while True:
            await RisingEdge(self.dut.clk)
            ws_v, ws_r = int(self.dut.ws_valid.value), int(self.dut.ws_ready.value)
            cur_ws = {f: _val(getattr(self.dut, f"ws_{f}")) for f in WS_FIELDS}
            if prev_ws is not None:
                assert cur_ws == prev_ws, "ws payload changed while valid and not ready"
            prev_ws = cur_ws if ws_v and not ws_r else None
            m_v, m_r = int(self.dut.meta_valid.value), int(self.dut.meta_ready.value)
            cur_meta = _val(self.dut.meta_data)
            if prev_meta is not None:
                assert cur_meta == prev_meta, "meta payload changed while valid and not ready"
            prev_meta = cur_meta if m_v and not m_r else None
            await FallingEdge(self.dut.clk)
            held_w = _val(self.dut.w_cnt) + _val(self.dut.w_ovalid) + _val(self.dut.w_outstanding)
            held_m = _val(self.dut.m_cnt) + _val(self.dut.m_ovalid) + _val(self.dut.m_outstanding)
            assert held_w <= FIFO_BEATS, f"weight reservation exceeds the FIFO: {held_w}"
            assert held_m <= META_FIFO_BEATS, f"meta reservation exceeds the FIFO: {held_m}"
            assert _val(self.dut.w_cnt) + _val(self.dut.w_ovalid) <= FIFO_BEATS
            self.max_w = max(self.max_w, held_w)
            self.max_m = max(self.max_m, held_m)
            if _val(self.dut.busy):
                self.busy_cycles += 1
            done = _val(self.dut.stream_done)
            if self.check_done:
                assert done == int(len(self.ws.seen) == self.expected_beats), (
                    f"stream_done {done} with {len(self.ws.seen)} of {self.expected_beats} beats;"
                    f" requests {self.model.requests}; tiles {[b['tile'] for b in self.ws.seen]}"
                )

    async def issue(self, d: Desc) -> None:
        self.check_done = False
        self.ws.seen.clear()
        self.meta.seen.clear()
        self.model.requests.clear()
        self.busy_cycles = 0
        self.expected_beats = len(d.beats(self.model))
        self.dut.cmd_op.value = d.op
        self.dut.cmd_addr_a.value = d.addr_a
        self.dut.cmd_addr_m.value = d.addr_m
        self.dut.cmd_n.value = d.n
        self.dut.cmd_k.value = d.k
        self.dut.cmd_k_stride.value = d.k_stride
        self.dut.cmd_unit_meta.value = d.unit_meta
        self.dut.cmd_tok.value = d.tok
        await qc_stream.pulse(self.dut.clk, self.dut.cmd_valid_gemv)
        self.check_done = True

    async def finish(self, d: Desc) -> None:
        """Wait for stream_done, every record and busy low; then compare everything."""
        recs = d.records(self.model)
        for _ in range(TIMEOUT):
            await FallingEdge(self.dut.clk)
            done = _val(self.dut.stream_done) and not _val(self.dut.busy)
            if done and len(self.meta.seen) >= len(recs):
                break
        else:
            raise AssertionError(
                f"stream did not finish: {len(self.ws.seen)} beats, {len(self.meta.seen)} records"
            )
        await qc_stream.cycles(self.dut.clk, 4)
        assert self.model.requests == d.requests(), (
            f"requests {self.model.requests[:6]}... != {d.requests()[:6]}..."
        )
        exp = d.beats(self.model)
        got = self.ws.seen
        assert len(got) == len(exp), f"{len(got)} beats, expected {len(exp)}"
        for i, (g, e) in enumerate(zip(got, exp, strict=True)):
            assert g == e, f"beat {i}: {g} != {e}"
        got_r = [r["data"] for r in self.meta.seen]
        assert got_r == recs, f"{len(got_r)} records, expected {len(recs)}"
        assert _val(self.dut.ws_valid) == 0 and _val(self.dut.meta_valid) == 0
        assert self.model.outstanding == 0
        self.check_done = False

    async def run(self, d: Desc) -> None:
        await self.issue(d)
        await self.finish(d)


async def _setup(dut, latency: int, bw_div: int = 1, seed: int = 1) -> Bench:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for name in (
        "cmd_valid_gemv",
        "cmd_op",
        "cmd_addr_a",
        "cmd_addr_m",
        "cmd_n",
        "cmd_k",
        "cmd_k_stride",
        "cmd_unit_meta",
        "cmd_tok",
        "ws_ready",
        "meta_ready",
    ):
        getattr(dut, name).value = 0
    rng = random.Random(seed)
    model = StreamQmem(dut, wb=WB, latency=latency, bw_div=bw_div)
    model.write_bytes(WEIGHT_BASE, rng.randbytes(1 << 18))
    _fill_meta(model, rng, META_BASE, 1 << 12)
    await qc_stream.reset(dut, dut.rst)
    cocotb.start_soon(model.run())
    bench = Bench(dut, model, rng)
    await FallingEdge(dut.clk)
    return bench


def _random_ready(rng: random.Random, p: float):
    return lambda t: rng.random() < p


async def _gemv_case(dut, latency: int, bw_div: int, ws_p: float, meta_p: float, seed: int) -> None:
    bench = await _setup(dut, latency, bw_div, seed)
    bench.ws_ready.pattern = _random_ready(bench.rng, ws_p)
    bench.meta_ready.pattern = _random_ready(bench.rng, meta_p)
    # 4 tiles, the last with 5 channels; K spans two bursts; tiles 128 beats apart
    await bench.run(Desc(OP_GEMV, WEIGHT_BASE, META_BASE, n=3 * WB + 5, k=100, k_stride=128))
    dut._log.info(
        "gemv LAT=%d bw_div=%d: %d beats, %d records, max held weight %d meta %d, busy %d",
        latency,
        bw_div,
        len(bench.ws.seen),
        len(bench.meta.seen),
        bench.max_w,
        bench.max_m,
        bench.busy_cycles,
    )


@cocotb.test()
async def gemv_lat1(dut):
    """Latency 1 with random ws_ready / meta_ready: beats and records in order, flags right."""
    await _gemv_case(dut, 1, 1, 0.7, 0.5, seed=1)


@cocotb.test()
async def gemv_lat32(dut):
    """Latency 32 with random ws_ready / meta_ready."""
    await _gemv_case(dut, 32, 1, 0.7, 0.5, seed=2)


@cocotb.test()
async def gemv_lat200(dut):
    """Latency 200: the in-flight window throttles the stream; everything still arrives in order."""
    await _gemv_case(dut, 200, 1, 0.9, 0.9, seed=3)


@cocotb.test()
async def gemv_response_gaps(dut):
    """One beat every 3 cycles from the memory, with random backpressure on both streams."""
    await _gemv_case(dut, 32, 3, 0.6, 0.4, seed=4)


@cocotb.test()
async def utilization_lat32(dut):
    """Weight beats per busy cycle at latency 32 with the rows always ready: at least 0.98."""
    bench = await _setup(dut, 32)
    d = Desc(OP_GEMV, WEIGHT_BASE, META_BASE, n=4 * WB, k=1024, k_stride=1024)
    await bench.run(d)
    beats = len(bench.ws.seen)
    ratio = beats / bench.busy_cycles
    dut._log.info(
        "utilization LAT=32: %d weight beats in %d busy cycles = %.4f (max held %d)",
        beats,
        bench.busy_cycles,
        ratio,
        bench.max_w,
    )
    assert ratio >= 0.98, ratio
    d2 = Desc(OP_GEMV, WEIGHT_BASE, META_BASE, n=4 * WB, k=1024, k_stride=1024, unit_meta=1)
    await bench.run(d2)
    ratio2 = len(bench.ws.seen) / bench.busy_cycles
    dut._log.info(
        "utilization LAT=32 unit_meta: %d beats in %d busy cycles = %.4f",
        len(bench.ws.seen),
        bench.busy_cycles,
        ratio2,
    )
    assert ratio2 >= 0.98, ratio2


@cocotb.test()
async def backpressure_fills_the_fifo(dut):
    """With the rows stalled the controller fills exactly FIFO_BEATS beats and stops requesting."""
    bench = await _setup(dut, 32)
    bench.ws_ready.pattern = 0
    d = Desc(OP_GEMV, WEIGHT_BASE, META_BASE, n=WB, k=5 * MAX_BURST, k_stride=5 * MAX_BURST)
    await bench.issue(d)
    await qc_stream.cycles(dut.clk, 400)
    held = _val(dut.w_cnt) + _val(dut.w_ovalid)
    assert held == FIFO_BEATS, f"FIFO holds {held}, expected {FIFO_BEATS}"
    assert _val(dut.w_outstanding) == 0 and bench.model.outstanding == 0
    assert _val(dut.s_req_valid) == 0, "no request while the FIFO has no room"
    assert bench.model.rd_beats == FIFO_BEATS
    bench.ws_ready.pattern = _random_ready(bench.rng, 0.5)
    await bench.finish(d)
    assert bench.max_w == FIFO_BEATS


@cocotb.test()
async def embed_gather(dut):
    """EMBED: lane TOK % WB of each beat packed into pseudo-beats, zero fill, one meta record."""
    bench = await _setup(dut, 32)
    bench.ws_ready.pattern = _random_ready(bench.rng, 0.7)
    cases = [
        (37, 1234),
        (16, 15),
        (1, 0),
        (50, 16 * 77 + 7),
        (3 * WB, 16 * 3),
        (2, 151935),
        (300, 5),
    ]  # 300 bytes: five bursts, more beats than the FIFO holds
    for k, tok in cases:
        d = Desc(OP_EMBED, WEIGHT_BASE, META_BASE, n=k, k=k, k_stride=0, tok=tok)
        await bench.run(d)
        assert all(b["embed"] == 1 for b in bench.ws.seen)
        assert len(bench.ws.seen) == -(-k // WB)
        dut._log.info(
            "embed K=%d TOK=%d: %d pseudo-beats, %d requests",
            k,
            tok,
            len(bench.ws.seen),
            len(bench.model.requests),
        )


@cocotb.test()
async def unit_meta_has_no_meta_traffic(dut):
    """unit_meta: weight bursts only, meta_valid never rises."""
    bench = await _setup(dut, 32)
    d = Desc(OP_GEMV, WEIGHT_BASE, META_BASE, n=2 * WB + 3, k=20, k_stride=20, unit_meta=1)
    await bench.run(d)
    assert all(t == TAG_WEIGHT for _, _, t in bench.model.requests)
    assert bench.meta.seen == []


@cocotb.test()
async def empty_descriptor(dut):
    """N == 0 or K == 0: no request, no beat, stream_done the cycle after issue."""
    bench = await _setup(dut, 32)
    for d in (
        Desc(OP_GEMV, WEIGHT_BASE, META_BASE, n=0, k=20, k_stride=20),
        Desc(OP_GEMV, WEIGHT_BASE, META_BASE, n=20, k=0, k_stride=0),
        Desc(OP_EMBED, WEIGHT_BASE, META_BASE, n=0, k=0, k_stride=0, tok=5),
    ):
        await bench.issue(d)
        assert _val(dut.stream_done) == 1 and _val(dut.busy) == 0
        await bench.finish(d)
        assert bench.model.requests == []


@cocotb.test()
async def back_to_back_descriptors(dut):
    """A partial tile with meta, an EMBED, a one-beat tile and a one-channel GEMV in sequence."""
    bench = await _setup(dut, 32, seed=9)
    bench.ws_ready.pattern = _random_ready(bench.rng, 0.8)
    bench.meta_ready.pattern = _random_ready(bench.rng, 0.6)
    seq = [
        Desc(OP_GEMV, WEIGHT_BASE, META_BASE, n=2 * WB + 1, k=70, k_stride=70),
        Desc(OP_EMBED, WEIGHT_BASE + 0x1000, META_BASE + 0x100, n=21, k=21, k_stride=0, tok=33),
        Desc(OP_GEMV, WEIGHT_BASE + 0x2000, META_BASE + 0x200, n=WB, k=1, k_stride=1),
        Desc(OP_GEMV, WEIGHT_BASE + 0x3000, META_BASE + 0x300, n=1, k=3, k_stride=8),
        Desc(OP_GEMV, WEIGHT_BASE + 0x4000, META_BASE + 0x400, n=WB + 1, k=64, k_stride=64),
        Desc(
            OP_GEMV,
            WEIGHT_BASE + 0x5000,
            META_BASE + 0x500,
            n=2 * WB,
            k=128,
            k_stride=200,
            unit_meta=1,
        ),
    ]
    for d in seq:
        await bench.run(d)
        # the next issue follows the dispatcher's minimum spacing after done
        await qc_stream.cycles(dut.clk, 3)


@cocotb.test()
async def contiguous_tiles_exact_bursts(dut):
    """k_stride == K with K a multiple of MAX_BURST: back-to-back full bursts across tiles."""
    bench = await _setup(dut, 32, seed=12)
    bench.meta_ready.pattern = _random_ready(bench.rng, 0.3)
    d = Desc(OP_GEMV, WEIGHT_BASE, META_BASE, n=3 * WB, k=3 * MAX_BURST, k_stride=3 * MAX_BURST)
    await bench.run(d)
    assert all(length == MAX_BURST or tag == TAG_META for _, length, tag in bench.model.requests)
