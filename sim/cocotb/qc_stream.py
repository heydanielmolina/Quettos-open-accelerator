"""Helpers for driving and observing valid/ready ports and pulses from cocotb.

Every helper works on falling clock edges: inputs are driven at a falling
edge so they are stable at the next rising edge, and outputs are sampled at a
falling edge, after the previous rising edge settled.  ``fields`` map a
logical name to a DUT signal (``{"data": dut.ws_data, ...}``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cocotb.handle import LogicObject
from cocotb.triggers import FallingEdge, RisingEdge


def value(sig: Any) -> int:
    """The unsigned integer value of a signal (a Logic for 1-bit handles, else a LogicArray)."""
    v = sig.value
    return int(v.to_unsigned()) if hasattr(v, "to_unsigned") else int(v)


def signed_value(sig: Any, bits: int) -> int:
    """The two's-complement value of the low ``bits`` of a signal."""
    v = value(sig) & ((1 << bits) - 1)
    return v - (1 << bits) if v >> (bits - 1) else v


async def cycles(clk: LogicObject, n: int) -> None:
    """Wait ``n`` falling edges."""
    for _ in range(n):
        await FallingEdge(clk)


async def reset(dut: Any, rst: LogicObject, cycles_high: int = 2) -> None:
    """Hold the synchronous reset for ``cycles_high`` rising edges; release at a falling edge."""
    rst.value = 1
    for _ in range(cycles_high):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    rst.value = 0


async def pulse(clk: LogicObject, sig: LogicObject) -> None:
    """Assert ``sig`` for one clock (driven at a falling edge, released at the next)."""
    sig.value = 1
    await FallingEdge(clk)
    sig.value = 0


class ValidReadyDriver:
    """Source side of a valid/ready port: ``send`` holds one transfer until it is accepted."""

    def __init__(
        self,
        clk: LogicObject,
        valid: LogicObject,
        ready: LogicObject,
        fields: Mapping[str, LogicObject],
    ) -> None:
        self.clk, self.valid, self.ready, self.fields = clk, valid, ready, dict(fields)
        self.valid.value = 0
        self.sent: list[dict[str, int]] = []

    async def send(self, values: Mapping[str, int], timeout: int = 100000) -> int:
        """Drive ``values`` with valid high until ``ready``; returns the cycles waited."""
        for name, v in values.items():
            self.fields[name].value = int(v)
        self.valid.value = 1
        waited = 0
        while True:
            await RisingEdge(self.clk)
            waited += 1
            if int(self.ready.value) == 1:
                break
            if waited > timeout:
                raise TimeoutError(f"valid/ready transfer not accepted within {timeout} cycles")
        await FallingEdge(self.clk)
        self.valid.value = 0
        self.sent.append(dict(values))
        return waited


class ValidReadyMonitor:
    """Sink side of a valid/ready port: records every accepted transfer at rising edges."""

    def __init__(
        self,
        clk: LogicObject,
        valid: LogicObject,
        ready: LogicObject | None,
        fields: Mapping[str, LogicObject],
    ) -> None:
        self.clk, self.valid, self.ready, self.fields = clk, valid, ready, dict(fields)
        self.seen: list[dict[str, int]] = []

    async def run(self) -> None:
        """Sample forever; start it with ``cocotb.start_soon``."""
        while True:
            await RisingEdge(self.clk)
            if int(self.valid.value) == 1 and (self.ready is None or int(self.ready.value) == 1):
                self.seen.append({n: value(s) for n, s in self.fields.items()})


class ReadySource:
    """Drives a ``ready`` input: ``1`` always, ``0`` never, or a callable evaluated per cycle."""

    def __init__(self, clk: LogicObject, ready: LogicObject, pattern: Any = 1) -> None:
        self.clk, self.ready, self.pattern = clk, ready, pattern
        self.cycle = 0

    async def run(self) -> None:
        while True:
            await FallingEdge(self.clk)
            p = self.pattern(self.cycle) if callable(self.pattern) else self.pattern
            self.ready.value = int(bool(p))
            self.cycle += 1
