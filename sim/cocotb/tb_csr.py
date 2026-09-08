"""cocotb tests of qcore_csr: every mapped word through the host port, the CTRL pulses and their
suppression rules, the sticky STATUS bits with the fault fields and their write-one-to-clear, the
PC hand-off between the core and the host, the four event counters, the ARGMAX registers and the
thirty-two PERF halves."""

from __future__ import annotations

import random

import cocotb
import qc_stream
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge
from quettos import isa
from quettos.isa import Fault

CSR = {c.name: c.word for c in isa.CSRS}
CTRL, STATUS, PC = CSR["CTRL"], CSR["STATUS"], CSR["PC"]
ROW_EN, TOK, POS = CSR["ROW_EN"], CSR["TOK"], CSR["POS"]
ARGMAX_TOK, ARGMAX_VAL = CSR["ARGMAX_TOK"], CSR["ARGMAX_VAL"]
COUNTERS = ("SAT_REQ", "SAT_VPU", "ERR_SHIFT", "ERR_BOUNDS")
RW_WORDS = (PC, ROW_EN, TOK, POS)
MASK32 = (1 << 32) - 1
UNMAPPED = tuple(w for w in range(isa.CSR_WORDS) if w not in set(CSR.values()))


class Host:
    """The host side of the CSR port: one write or one read per cycle, reads a cycle late."""

    def __init__(self, dut) -> None:
        self.dut = dut

    async def write(self, word: int, value: int) -> None:
        self.dut.csr_we.value = 1
        self.dut.csr_addr.value = word
        self.dut.csr_wdata.value = value & MASK32
        await FallingEdge(self.dut.clk)
        self.dut.csr_we.value = 0

    async def read(self, word: int) -> int:
        self.dut.csr_re.value = 1
        self.dut.csr_addr.value = word
        await FallingEdge(self.dut.clk)
        self.dut.csr_re.value = 0
        return qc_stream.value(self.dut.csr_rdata)

    async def idle(self, n: int = 1) -> None:
        await qc_stream.cycles(self.dut.clk, n)


async def _setup(dut, seed: int = 0xC5B) -> tuple[Host, random.Random]:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for sig in (
        dut.csr_we,
        dut.csr_re,
        dut.csr_addr,
        dut.csr_wdata,
        dut.pc_set,
        dut.pc_set_val,
        dut.busy_i,
        dut.done_set,
        dut.step_halted_set,
        dut.err_set,
        dut.fault_code,
        dut.fault_op,
        dut.argmax_we,
        dut.argmax_tok,
        dut.argmax_val,
        dut.sat_req_inc,
        dut.sat_vpu_inc,
        dut.err_shift_inc,
        dut.err_bounds_inc,
        dut.perf_snap,
    ):
        sig.value = 0
    await qc_stream.reset(dut, dut.rst)
    return Host(dut), random.Random(seed)


async def _set_pulse(dut, name: str, **extra: int) -> None:
    """Raise one of the dispatcher's set pulses (with any extra payload) for one cycle."""
    for k, v in extra.items():
        getattr(dut, k).value = v
    getattr(dut, name).value = 1
    await FallingEdge(dut.clk)
    getattr(dut, name).value = 0


def _rdata(dut) -> int:
    """The read-data register as it stands now, without issuing a read."""
    return qc_stream.value(dut.csr_rdata)


@cocotb.test()
async def test_reset_state_and_unmapped_words(dut):
    """Out of reset every word reads 0 except ISA_VERSION; ro and unmapped words ignore writes."""
    host, _ = await _setup(dut)
    for word in range(isa.CSR_WORDS):
        got = await host.read(word)
        want = isa.ISA_VERSION if word == CSR["ISA_VERSION"] else 0
        assert got == want, f"word {word} reads {got:#x}, expected {want:#x}"
    for word in (CSR["ISA_VERSION"], CSR["SAT_REQ"], ARGMAX_VAL, CSR["PERF0_LO"], *UNMAPPED[:4]):
        await host.write(word, 0xDEADBEEF)
        got = await host.read(word)
        want = isa.ISA_VERSION if word == CSR["ISA_VERSION"] else 0
        assert got == want, f"write to read-only word {word} took effect"
    assert qc_stream.value(dut.start) == 0 and qc_stream.value(dut.step) == 0
    assert qc_stream.value(dut.abort_run) == 0


@cocotb.test()
async def test_rw_registers_round_trip(dut):
    """PC, ROW_EN, TOK and POS take every bit, drive their outputs and read back."""
    host, rng = await _setup(dut)
    outs = {PC: dut.pc_q, ROW_EN: dut.row_en_q, TOK: dut.tok_q, POS: dut.pos_q}
    for value in (0xFFFFFFFF, 0x00000001, 0x80000000, 0xA5A5A5A5, 0):
        for word in RW_WORDS:
            await host.write(word, value)
        for word in RW_WORDS:
            assert qc_stream.value(outs[word]) == value, f"word {word} output"
            assert await host.read(word) == value, f"word {word} read back"
    for _ in range(40):
        word = rng.choice(RW_WORDS)
        value = rng.getrandbits(32)
        await host.write(word, value)
        assert await host.read(word) == value
        for other in RW_WORDS:  # a write reaches exactly one register
            if other != word:
                assert qc_stream.value(outs[other]) != value or value == 0


@cocotb.test()
async def test_read_latency_and_hold(dut):
    """csr_rdata carries the addressed word one cycle after csr_re and holds it until the next."""
    host, _ = await _setup(dut)
    await host.write(TOK, 0x1234_5678)
    await host.write(POS, 0x0000_0042)
    assert await host.read(TOK) == 0x1234_5678
    dut.csr_addr.value = POS  # the address moves with no enable: the output must hold
    for _ in range(4):
        await FallingEdge(dut.clk)
        assert _rdata(dut) == 0x1234_5678
    assert await host.read(POS) == 0x42
    # a read and a write of the same word in one cycle return the value before the write
    dut.csr_re.value = 1
    await host.write(TOK, 0xFFFF_0000)
    dut.csr_re.value = 0
    assert _rdata(dut) == 0x1234_5678
    assert await host.read(TOK) == 0xFFFF_0000


@cocotb.test()
async def test_ctrl_pulses(dut):
    """CTRL is write-one-to-pulse: one cycle, the cycle after the write, and reads back as 0."""
    host, _ = await _setup(dut)
    pulses = {"START": dut.start, "STEP": dut.step, "ABORT": dut.abort_run}
    for name, sig in pulses.items():
        dut.busy_i.value = 1 if name == "ABORT" else 0
        await host.write(CTRL, 1 << isa.CTRL_BITS[name])
        assert qc_stream.value(sig) == 1, f"{name}: no pulse the cycle after the write"
        for other, osig in pulses.items():
            if other != name:
                assert qc_stream.value(osig) == 0, f"{name} also pulsed {other}"
        await FallingEdge(dut.clk)
        assert qc_stream.value(sig) == 0, f"{name}: pulse wider than one cycle"
        assert await host.read(CTRL) == 0, "CTRL reads as zero"
    # START and STEP are suppressed while busy, ABORT only acts while busy
    dut.busy_i.value = 1
    for name in ("START", "STEP"):
        await host.write(CTRL, 1 << isa.CTRL_BITS[name])
        assert qc_stream.value(pulses[name]) == 0, f"{name} must be ignored while busy"
    dut.busy_i.value = 0
    await host.write(CTRL, 1 << isa.CTRL_BITS["ABORT"])
    assert qc_stream.value(dut.abort_run) == 0, "ABORT must be ignored while idle"
    # START wins over STEP in one write; a write of every bit at once still pulses once
    await host.write(CTRL, 0b011)
    assert qc_stream.value(dut.start) == 1 and qc_stream.value(dut.step) == 0
    dut.busy_i.value = 1
    await host.write(CTRL, 0xFFFF_FFFF)
    assert qc_stream.value(dut.abort_run) == 1
    assert qc_stream.value(dut.start) == 0 and qc_stream.value(dut.step) == 0
    dut.busy_i.value = 0
    await host.write(CTRL, 0)  # no bit set: no pulse
    for sig in pulses.values():
        assert qc_stream.value(sig) == 0


@cocotb.test()
async def test_status_bits_and_fault_fields(dut):
    """DONE, STEP_HALTED and ERR are sticky, BUSY is the live level, the fault fields latch."""
    host, _ = await _setup(dut)
    assert await host.read(STATUS) == 0
    dut.busy_i.value = 1
    assert await host.read(STATUS) == isa.status_word(busy=True)
    dut.busy_i.value = 0
    await _set_pulse(dut, "done_set")
    assert await host.read(STATUS) == isa.status_word(done=True)
    await host.idle(3)
    assert await host.read(STATUS) == isa.status_word(done=True), "DONE is sticky"
    await _set_pulse(dut, "step_halted_set")
    assert await host.read(STATUS) == isa.status_word(done=True, step_halted=True)
    for fault in Fault:
        for op in (int(isa.Opcode.GEMV), int(isa.Opcode.KVWRITE), 0x77, 0xFF):
            await _set_pulse(dut, "err_set", fault_code=int(fault), fault_op=op)
            word = await host.read(STATUS)
            assert isa.status_fault(word) == (fault, op), f"{fault.name}/{op:#x}: {word:#x}"
            assert word == isa.status_word(
                done=True, step_halted=True, err=True, fault=fault, fault_op=op
            )
    last = list(Fault)[-1]  # the pulse the fault fields still hold
    # write one to clear: only the bits written, and the fault fields go with ERR
    await host.write(STATUS, 1 << isa.STATUS_BITS["DONE"])
    assert await host.read(STATUS) == isa.status_word(
        step_halted=True, err=True, fault=last, fault_op=0xFF
    )
    await host.write(STATUS, 1 << isa.STATUS_BITS["ERR"])
    assert await host.read(STATUS) == isa.status_word(step_halted=True)
    await host.write(STATUS, 1 << isa.STATUS_BITS["STEP_HALTED"])
    assert await host.read(STATUS) == 0
    # a set pulse in the cycle of a clearing write wins
    dut.done_set.value = 1
    await host.write(STATUS, 0xFFFF_FFFF)
    dut.done_set.value = 0
    assert await host.read(STATUS) == isa.status_word(done=True)


@cocotb.test()
async def test_start_and_step_clear_the_status_bits(dut):
    """START and STEP clear DONE, STEP_HALTED, ERR and the fault fields; ABORT does not."""
    host, _ = await _setup(dut)
    for name in ("START", "STEP"):
        await _set_pulse(dut, "done_set")
        await _set_pulse(dut, "err_set", fault_code=int(Fault.ROW), fault_op=0x20)
        await _set_pulse(dut, "step_halted_set")
        assert isa.status_fault(await host.read(STATUS)) == (Fault.ROW, 0x20)
        await host.write(CTRL, 1 << isa.CTRL_BITS[name])
        await host.idle(2)
        assert await host.read(STATUS) == 0, f"{name} must clear the status bits"
    await _set_pulse(dut, "done_set")
    dut.busy_i.value = 1
    await host.write(CTRL, 1 << isa.CTRL_BITS["ABORT"])
    await host.idle(2)
    dut.busy_i.value = 0
    assert await host.read(STATUS) == isa.status_word(done=True), "ABORT keeps the status bits"


@cocotb.test()
async def test_pc_belongs_to_the_core_while_it_runs(dut):
    """pc_set beats a host write in the same cycle; the host may only write PC while idle."""
    host, rng = await _setup(dut)
    await host.write(PC, 0x0000_0640)
    assert qc_stream.value(dut.pc_q) == 0x640
    for _ in range(8):  # the retire update
        want = qc_stream.value(dut.pc_q) + isa.DESC_BYTES
        await _set_pulse(dut, "pc_set", pc_set_val=want)
        assert qc_stream.value(dut.pc_q) == want
    dut.pc_set.value = 1
    dut.pc_set_val.value = 0x1000
    await host.write(PC, 0x2000)  # the core wins the same cycle
    dut.pc_set.value = 0
    assert qc_stream.value(dut.pc_q) == 0x1000
    dut.busy_i.value = 1
    for _ in range(4):
        await host.write(PC, rng.getrandbits(32))
        assert qc_stream.value(dut.pc_q) == 0x1000, "a host PC write while busy must be ignored"
    await _set_pulse(dut, "pc_set", pc_set_val=0x1020)
    assert qc_stream.value(dut.pc_q) == 0x1020, "the core still advances PC while busy"
    for word in (ROW_EN, TOK, POS):  # the per-token registers are the host's while idle only
        await host.write(word, 0xFFFF_FFFF)
        assert await host.read(word) == 0
    dut.busy_i.value = 0
    await host.write(TOK, 0x99)
    assert await host.read(TOK) == 0x99


@cocotb.test()
async def test_event_counters(dut):
    """The four counters add their per-cycle counts and clear on START, not on STEP."""
    host, rng = await _setup(dut)
    incs = {
        "SAT_REQ": dut.sat_req_inc,
        "SAT_VPU": dut.sat_vpu_inc,
        "ERR_SHIFT": dut.err_shift_inc,
        "ERR_BOUNDS": dut.err_bounds_inc,
    }
    totals = dict.fromkeys(COUNTERS, 0)
    for _ in range(300):
        step = {n: rng.randrange(0, 5) for n in COUNTERS}
        for n, sig in incs.items():
            sig.value = step[n]
        await FallingEdge(dut.clk)
        for n in COUNTERS:
            totals[n] += step[n]
    for sig in incs.values():
        sig.value = 0
    await FallingEdge(dut.clk)
    for n in COUNTERS:
        got = await host.read(CSR[n])
        assert got == totals[n], f"{n}: {got} != {totals[n]}"
    dut.sat_req_inc.value = 255  # the ports are counts, not masks
    await FallingEdge(dut.clk)
    dut.sat_req_inc.value = 0
    await FallingEdge(dut.clk)
    assert await host.read(CSR["SAT_REQ"]) == totals["SAT_REQ"] + 255
    await host.write(CTRL, 1 << isa.CTRL_BITS["STEP"])
    await host.idle(2)
    assert await host.read(CSR["SAT_VPU"]) == totals["SAT_VPU"], "STEP keeps the counters"
    await host.write(CTRL, 1 << isa.CTRL_BITS["START"])
    await host.idle(2)
    for n in COUNTERS:
        assert await host.read(CSR[n]) == 0, f"{n}: START must clear the counters"


@cocotb.test()
async def test_argmax_registers(dut):
    """argmax_we writes both words; they survive START and ignore host writes."""
    host, rng = await _setup(dut)
    for _ in range(16):
        tok, val = rng.getrandbits(32), rng.getrandbits(32)
        await _set_pulse(dut, "argmax_we", argmax_tok=tok, argmax_val=val)
        assert await host.read(ARGMAX_TOK) == tok
        assert await host.read(ARGMAX_VAL) == val
    last_tok = await host.read(ARGMAX_TOK)
    await host.write(ARGMAX_TOK, 0)
    await host.write(ARGMAX_VAL, 0)
    assert await host.read(ARGMAX_TOK) == last_tok
    await host.write(CTRL, 1 << isa.CTRL_BITS["START"])
    await host.idle(2)
    assert await host.read(ARGMAX_TOK) == last_tok, "START must not clear the ARGMAX result"


@cocotb.test()
async def test_perf_halves(dut):
    """PERF[i] reads as two words: the low half at PERF_BASE + 2i, the high half one word up."""
    host, rng = await _setup(dut)
    for _ in range(3):
        values = [rng.getrandbits(64) for _ in range(isa.PERF_COUNT)]
        snap = 0
        for i, v in enumerate(values):
            snap |= v << (64 * i)
        dut.perf_snap.value = snap
        await FallingEdge(dut.clk)
        for i, v in enumerate(values):
            lo, hi = isa.perf_words(i)
            assert await host.read(lo) == v & MASK32, f"PERF{i}_LO"
            assert await host.read(hi) == (v >> 32) & MASK32, f"PERF{i}_HI"
    dut.perf_snap.value = 0
    await FallingEdge(dut.clk)
    for i in range(isa.PERF_COUNT):
        lo, hi = isa.perf_words(i)
        assert await host.read(lo) == 0 and await host.read(hi) == 0
    for word in UNMAPPED:  # the words around the PERF block stay zero
        assert await host.read(word) == 0
