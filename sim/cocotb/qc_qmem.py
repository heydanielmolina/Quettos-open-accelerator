"""Python model of the QMEM memory bus for cocotb tests (``docs/RTL.md``, QMEM).

``QmemModel`` answers a DUT's ``rd_req`` / ``rd_data`` / ``wr`` / ``wr_ack``
ports from a byte-addressed sparse memory: fixed read latency, one beat per
cycle (``bw_div`` beats every ``bw_div`` cycles), in-order return across
tags, an in-flight window that throttles ``rd_req_ready``, write acks after
the same latency.  A read takes its bytes when the request is accepted, so a
write accepted while the burst is in flight leaves the returned data alone.
Everything is sampled and driven on falling clock edges.
Counters mirror the PERF byte counters (``rd_beats``, ``rd_bytes``,
``wr_beats``, ``wr_bytes`` = strobed bytes).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cocotb.triggers import FallingEdge

PAGE = 4096


@dataclass
class _Beat:
    deliver: int  # rising-edge index at which the DUT samples the beat
    data: int  # the beat as memory held it when the request was accepted
    tag: int
    last: bool


@dataclass
class _Sample:
    rd_valid: int = 0
    rd_addr: int = 0
    rd_len: int = 0
    rd_tag: int = 0
    rd_ready: int = 0
    wr_valid: int = 0
    wr_addr: int = 0
    wr_data: int = 0
    wr_strb: int = 0
    wr_ready: int = 0


@dataclass
class QmemModel:
    """Bus model on the ``<prefix>rd_req_*`` / ``rd_data*`` / ``wr_*`` / ``wr_ack`` ports."""

    dut: Any
    wb: int
    latency: int = 32
    window: int = 64
    bw_div: int = 1
    prefix: str = ""
    wr_ready_pattern: Any = 1
    pages: dict[int, bytearray] = field(default_factory=dict)
    rd_beats: int = 0
    rd_bytes: int = 0
    wr_beats: int = 0
    wr_bytes: int = 0
    requests: list[tuple[int, int, int]] = field(default_factory=list)
    writes: list[tuple[int, int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._t = 0
        self._pending: deque[_Beat] = deque()
        self._last_deliver = -1
        self._acks: deque[int] = deque()
        self._prev = _Sample()
        for name in (
            "rd_req_ready",
            "rd_data_valid",
            "rd_data",
            "rd_data_tag",
            "rd_data_last",
            "wr_ready",
            "wr_ack",
        ):
            self._sig(name).value = 0

    # ---- memory contents

    def _sig(self, name: str) -> Any:
        return getattr(self.dut, self.prefix + name)

    def write_bytes(self, addr: int, data: bytes) -> None:
        for i, b in enumerate(data):
            a = addr + i
            self.pages.setdefault(a // PAGE, bytearray(PAGE))[a % PAGE] = b

    def read_bytes(self, addr: int, n: int) -> bytes:
        out = bytearray(n)
        for i in range(n):
            a = addr + i
            page = self.pages.get(a // PAGE)
            if page is not None:
                out[i] = page[a % PAGE]
        return bytes(out)

    def load_image(self, path: str | Path, base: int = 0) -> None:
        self.write_bytes(base, Path(path).read_bytes())

    def beat(self, addr: int) -> int:
        """The ``wb``-byte beat at byte address ``addr``: byte ``j`` in bits ``[8j+7:8j]``."""
        return int.from_bytes(self.read_bytes(addr, self.wb), "little")

    @property
    def outstanding(self) -> int:
        return len(self._pending)

    @property
    def writes_outstanding(self) -> int:
        return len(self._acks)

    # ---- bus protocol

    def _sample(self) -> _Sample:
        s = _Sample()
        s.rd_valid = int(self._sig("rd_req_valid").value)
        if s.rd_valid:
            s.rd_addr = int(self._sig("rd_req_addr").value.to_unsigned())
            s.rd_len = int(self._sig("rd_req_len").value.to_unsigned())
            s.rd_tag = int(self._sig("rd_req_tag").value.to_unsigned())
        s.wr_valid = int(self._sig("wr_valid").value)
        if s.wr_valid:
            s.wr_addr = int(self._sig("wr_addr").value.to_unsigned())
            s.wr_data = int(self._sig("wr_data").value.to_unsigned())
            s.wr_strb = int(self._sig("wr_strb").value.to_unsigned())
        return s

    def _accept_read(self, s: _Sample) -> None:
        self.requests.append((s.rd_addr, s.rd_len, s.rd_tag))
        for i in range(s.rd_len):
            deliver = max(self._t + self.latency + i, self._last_deliver + self.bw_div)
            self._last_deliver = deliver
            self._pending.append(
                _Beat(deliver, self.beat(s.rd_addr + i * self.wb), s.rd_tag, i == s.rd_len - 1)
            )

    def _accept_write(self, s: _Sample) -> None:
        data = s.wr_data.to_bytes(self.wb, "little")
        strobed = 0
        for j in range(self.wb):
            if (s.wr_strb >> j) & 1:
                self.write_bytes(s.wr_addr + j, data[j : j + 1])
                strobed += 1
        self.writes.append((s.wr_addr, s.wr_strb, s.wr_data))
        self.wr_beats += 1
        self.wr_bytes += strobed
        self._acks.append(self._t + self.latency)

    async def run(self) -> None:
        """Serve the bus forever; start with ``cocotb.start_soon(model.run())``."""
        clk = self.dut.clk
        while True:
            await FallingEdge(clk)
            self._t += 1
            prev = self._prev
            if prev.rd_valid and prev.rd_ready:
                self._accept_read(prev)
            if prev.wr_valid and prev.wr_ready:
                self._accept_write(prev)
            cur = self._sample()
            nxt = self._t + 1
            # read data for the next rising edge
            if self._pending and self._pending[0].deliver <= nxt:
                b = self._pending.popleft()
                self._sig("rd_data_valid").value = 1
                self._sig("rd_data").value = b.data
                self._sig("rd_data_tag").value = b.tag
                self._sig("rd_data_last").value = int(b.last)
                self.rd_beats += 1
                self.rd_bytes += self.wb
            else:
                self._sig("rd_data_valid").value = 0
                self._sig("rd_data_last").value = 0
            # write ack for the next rising edge
            if self._acks and self._acks[0] <= nxt:
                self._acks.popleft()
                self._sig("wr_ack").value = 1
            else:
                self._sig("wr_ack").value = 0
            # readies for the next rising edge
            cur.rd_ready = int(len(self._pending) < self.window)
            p = self.wr_ready_pattern
            cur.wr_ready = int(bool(p(self._t) if callable(p) else p))
            self._sig("rd_req_ready").value = cur.rd_ready
            self._sig("wr_ready").value = cur.wr_ready
            self._prev = cur
