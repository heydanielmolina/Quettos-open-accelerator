"""cocotb tests of qcore_seq_dispatch against sw/quettos/isa_sim.py.

The bench stands in for everything around the dispatcher: the CSR file (PC, the per-token
registers and the CTRL pulses), the descriptor queue of qcore_seq_fetch, the SREG banks and the
three execution units with their done pulses, the weight stream's busy / done levels and the
arbiter's wr_idle.  Every issued bundle is compared field by field against the decode of the same
descriptor by ``quettos.isa`` and the POS-derived extents of ``quettos.isa_sim``, and a sweep over
shapes, positions and row sets checks the decoded extents, the MAC and weight-byte totals and the
bounds events of every POS-derived case against the simulator.
"""

from __future__ import annotations

import dataclasses
import random

import cocotb
import qc_stream
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge
from quettos import isa, isa_sim
from quettos.isa import Fault, Opcode, OutMode

BASE = 0x0000_1000
MASK32 = (1 << 32) - 1
BUCKETS = ("MAC_ACTIVE", "STALL_MEM", "STALL_VPU", "STALL_KV", "STALL_SEQ", "STALL_DRAIN")

# The bundle fields that are a plain descriptor field, as {port suffix: descriptor field}.
PLAIN = {
    "out_mode": "out_mode",
    "accumulate": "accumulate",
    "unit_meta": "unit_meta",
    "track_absmax": "track_absmax",
    "addr_a": "addr_a",
    "addr_m": "addr_m",
    "imm32": "imm32",
    "vs_src": "vs_src",
    "vs_dst": "vs_dst",
    "vs_aux": "vs_aux",
    "sreg_dst": "sreg_dst",
    "src_row": "src_row",
    "dst_row": "dst_row",
    "sh0": "sh0",
}
BUNDLE = (
    list(PLAIN)
    + ["op", "n", "k", "k_stride", "len", "sh1", "sqrt_m", "sqrt_e", "rows", "pos", "tok"]
    + ["vq_w8", "vq_use_tracked", "vq_group", "vq_scale_mul", "kv_transposed"]
    + ["sx_m", "sx_e", "sreg_u32"]
)

VPU_OPS = (
    Opcode.VRMSNORM,
    Opcode.VQUANT,
    Opcode.VROPE,
    Opcode.VSILUMUL,
    Opcode.VSOFTMAX,
    Opcode.VSUBC,
)
FENCED = (
    Opcode.GEMV,
    Opcode.EMBED,
    Opcode.VRMSNORM,
    Opcode.VROPE,
    Opcode.VSOFTMAX,
    Opcode.VSUBC,
    Opcode.FENCE,
    Opcode.HALT,
)


def sreg_word(row: int, idx: int) -> int:
    """The 32-bit word the bench's SREG banks hold at ``[row][idx]``."""
    return ((0x8000 + 0x111 * idx + 0x7 * row) & 0xFFFF) | (((idx + row) & 0xFF) << 16)


def participating(d: isa.Descriptor, row_en: int, b_max: int) -> list[int]:
    return [r for r in range(b_max) if (d.row_mask >> r) & 1 and (row_en >> r) & 1]


def expect(d: isa.Descriptor, *, pos: int, tok: int, row_en: int, b_max: int, wb: int) -> dict:
    """The cmd_* bundle qcore_seq_dispatch must present for ``d`` (docs/RTL.md 2.2)."""
    rows = participating(d, row_en, b_max)
    if d.opcode == Opcode.GEMV:
        n, k, _ = isa_sim.gemv_dims(d, pos, wb)
    elif d.opcode == Opcode.EMBED:
        n, k = d.k, d.k
    else:
        n, k = d.n, d.k
    want = {name: int(getattr(d, field)) for name, field in PLAIN.items()}
    want.update(
        op=int(d.opcode),
        n=n,
        k=k,
        k_stride=d.k,
        len=isa_sim.softmax_len(d, pos)[0],
        sh1=d.sh1 & 0xFF,
        sqrt_m=d.addr_m & 0xFFFF,
        sqrt_e=(d.addr_m >> 16) & 0xFF,
        rows=sum(1 << r for r in rows),
        pos=pos,
        tok=tok,
        vq_w8=int(bool(d.flags & isa.VquantFlag.W8)),
        vq_use_tracked=int(bool(d.flags & isa.VquantFlag.USE_TRACKED)),
        vq_group=int(bool(d.flags & isa.VquantFlag.GROUP)),
        vq_scale_mul=int(bool(d.flags & isa.VquantFlag.SCALE_MUL)),
        kv_transposed=int(bool(d.flags & isa.KvwriteFlag.TRANSPOSED)),
    )
    tracked = d.opcode == Opcode.VQUANT and (d.flags & isa.VquantFlag.USE_TRACKED)
    grouped = d.flags & isa.VquantFlag.GROUP
    if d.opcode in (Opcode.GEMV, Opcode.KVWRITE) or (tracked and not grouped):
        words = {r: sreg_word(d.src_row + r, d.sreg_src) for r in rows}
    elif d.opcode == Opcode.EMBED:
        words = {r: 0xF1_8000 for r in rows}
    else:
        words = {}
    want["sx_m"] = {r: w & 0xFFFF for r, w in words.items()}
    want["sx_e"] = {r: (w >> 16) & 0xFF for r, w in words.items()}
    want["sreg_u32"] = {
        r: (w if d.opcode not in (Opcode.EMBED,) else None) for r, w in words.items()
    }
    return want


def expect_events(d: isa.Descriptor, *, pos: int, row_en: int, b_max: int, wb: int) -> dict:
    """ERR_BOUNDS, MACS and WT_BYTES of one descriptor, counted as isa_sim counts them."""
    rows = len(participating(d, row_en, b_max))
    errs = macs = wt = 0
    if d.opcode == Opcode.GEMV:
        n, k, e = isa_sim.gemv_dims(d, pos, wb)
        errs = e * rows
        if n and k:
            tiles = -(-n // wb)
            macs = rows * tiles * wb * k
            if not (d.n_from_pos or d.k_from_pos):
                wt = rows * (tiles * k * wb + (0 if d.unit_meta else tiles * wb * 8))
    elif d.opcode == Opcode.EMBED:
        if d.k:
            wt = rows * (d.k + isa.META_BYTES)
    elif d.opcode == Opcode.VSOFTMAX:
        errs = isa_sim.softmax_len(d, pos)[1] * rows
    return {"err_bounds": errs, "macs": macs, "wt_bytes": wt}


class Env:
    """The dispatcher's surroundings: CSR registers, the descriptor queue and the three units."""

    def __init__(
        self,
        dut,
        *,
        b_max: int,
        wb: int,
        gemv_beats: int = 4,
        gemv_drain: int = 3,
        vpu_cycles: int = 5,
        kv_cycles: int = 4,
        dq_delay: int = 0,
        beat_on_embed: bool = True,
    ) -> None:
        self.dut = dut
        self.b_max, self.wb = b_max, wb
        self.gemv_beats, self.gemv_drain = gemv_beats, gemv_drain
        self.vpu_cycles, self.kv_cycles = vpu_cycles, kv_cycles
        self.dq_delay, self.beat_on_embed = dq_delay, beat_on_embed
        self.program: list[isa.Descriptor | int] = []
        self.base = BASE
        self.pc = BASE
        self.row_en, self.tok, self.pos = 1, 0, 0
        self.wr_idle = 1
        self.idx, self.fetched, self.armed, self.step_level = 0, 0, False, 0
        self.wait = 0
        self.cycle = 0
        self.unit = ""  # "" | "beats" | "drain" | "vpu" | "kv"
        self.sdone = 0  # stream_done, a level that holds until the next GEMV issue
        self.cnt = 0
        self.embed = False
        # observations
        self.issues: list[dict] = []
        self.retires: list[dict] = []
        self.pops: list[int] = []
        self.trace: list[tuple[int, int]] = []  # (busy, ev_bucket) per cycle
        self.sreg_reads: list[tuple[int, int]] = []
        self.fetch_starts: list[int] = []
        self.aborts: list[int] = []
        self.flushes = 0
        self.snapshots = 0
        self.clears = 0
        self.err_bounds = 0
        self.macs = 0
        self.wt_bytes = 0
        self.descriptors = 0
        self.done = 0
        self.step_halted = 0
        self.err = 0
        self.fault: tuple[int, int] | None = None
        self._prev_pop = False

    # ---- helpers

    def load(self, program, base: int = BASE) -> None:
        self.program = list(program)
        self.base = base
        self.pc = base

    def word(self, i: int) -> int:
        d = self.program[i]
        return d if isinstance(d, int) else d.word()

    def bundle(self) -> dict:
        """A snapshot of every cmd_* output this cycle."""
        dut, b = self.dut, {}
        for name in BUNDLE:
            if name in ("sx_m", "sx_e", "sreg_u32"):
                continue
            b[name] = qc_stream.value(getattr(dut, f"cmd_{name}"))
        sx_m, sx_e, u32 = (
            qc_stream.value(dut.cmd_sx_m),
            qc_stream.value(dut.cmd_sx_e),
            qc_stream.value(dut.cmd_sreg_u32),
        )
        b["sx_m"] = {r: (sx_m >> (16 * r)) & 0xFFFF for r in range(self.b_max)}
        b["sx_e"] = {r: (sx_e >> (8 * r)) & 0xFF for r in range(self.b_max)}
        b["sreg_u32"] = {r: (u32 >> (32 * r)) & MASK32 for r in range(self.b_max)}
        return b

    # ---- the bench clock domain

    async def run(self) -> None:
        dut = self.dut
        while True:
            await FallingEdge(dut.clk)
            self.cycle += 1
            self._observe()
            self._drive()

    def _observe(self) -> None:
        dut = self.dut
        busy = int(dut.busy.value)
        self.trace.append((busy, qc_stream.value(dut.ev_bucket)))
        if self._prev_pop:
            self.pops.append(self.cycle - 1)
            self.idx += 1
            self.fetched += 1
            self.wait = self.dq_delay
        self._prev_pop = bool(int(dut.dq_valid.value) and int(dut.dq_ready.value))
        if int(dut.fetch_flush.value):
            self.flushes += 1
            self.armed = False
        if int(dut.fetch_start.value):
            pc = qc_stream.value(dut.fetch_pc)
            self.fetch_starts.append(pc)
            self.idx = (pc - self.base) // isa.DESC_BYTES
            self.fetched, self.armed, self.wait = 0, True, self.dq_delay
        self.step_level = int(dut.fetch_step.value)
        if int(dut.abort_run.value):
            self.aborts.append(self.cycle)
        if int(dut.sreg_rd_en.value):
            self.sreg_reads.append(
                (qc_stream.value(dut.sreg_rd_row), qc_stream.value(dut.sreg_rd_idx))
            )
        kinds = {
            "gemv": int(dut.cmd_valid_gemv.value),
            "vpu": int(dut.cmd_valid_vpu.value),
            "kv": int(dut.cmd_valid_kv.value),
        }
        for kind, on in kinds.items():
            if on:
                self.issues.append({"cycle": self.cycle, "kind": kind, **self.bundle()})
                self.embed = qc_stream.value(dut.cmd_op) == int(Opcode.EMBED)
                self.unit = "beats" if kind == "gemv" else kind
                self.cnt = {
                    "gemv": self.gemv_beats,
                    "vpu": self.vpu_cycles,
                    "kv": self.kv_cycles,
                }[kind]
        if int(dut.ev_desc.value):
            self.descriptors += 1
        if int(dut.pc_set.value):
            self.retires.append({"cycle": self.cycle, "pc": self.pc})
            self.pc = qc_stream.value(dut.pc_set_val)
        self.err_bounds += qc_stream.value(dut.err_bounds_inc)
        if int(dut.ev_macs_valid.value):
            self.macs += qc_stream.value(dut.ev_macs)
        if int(dut.ev_wt_valid.value):
            self.wt_bytes += qc_stream.value(dut.ev_wt_bytes)
        self.done += int(dut.done_set.value)
        self.step_halted += int(dut.step_halted_set.value)
        if int(dut.err_set.value):
            self.err += 1
            self.fault = (qc_stream.value(dut.fault_code), qc_stream.value(dut.fault_op))
        self.snapshots += int(dut.perf_snapshot.value)
        self.clears += int(dut.perf_clear.value)

    def _drive(self) -> None:
        dut = self.dut
        dut.pc_q.value = self.pc & MASK32
        dut.row_en_q.value = self.row_en & ((1 << self.b_max) - 1)
        dut.tok_q.value = self.tok & MASK32
        dut.pos_q.value = self.pos & MASK32
        dut.wr_idle.value = self.wr_idle
        # descriptor queue
        ready = self.armed and self.idx < len(self.program)
        if self.step_level and self.fetched >= 1:
            ready = False
        if self.wait > 0:
            self.wait -= 1
            ready = False
        dut.dq_valid.value = int(ready)
        dut.dq_desc.value = self.word(self.idx) if ready else 0
        # SREG banks: data one cycle after the enable
        if int(dut.sreg_rd_en.value):
            row, idx = qc_stream.value(dut.sreg_rd_row), qc_stream.value(dut.sreg_rd_idx)
            dut.sreg_rd_data.value = sreg_word(row, idx)
        # execution units
        done_gemv = done_vpu = done_kv = 0
        beat = busy = 0
        if self.unit == "beats":
            busy, beat = 1, int(self.beat_on_embed or not self.embed)
            self.sdone = 0
            self.cnt -= 1
            if self.cnt <= 0:
                self.unit, self.cnt = "drain", self.gemv_drain
        elif self.unit == "drain":
            busy, self.sdone = 1, 1
            if self.cnt == self.gemv_drain:
                done_gemv = 1
            self.cnt -= 1
            if self.cnt <= 0:
                self.unit = ""
        elif self.unit in ("vpu", "kv"):
            self.cnt -= 1
            if self.cnt <= 0:
                done_vpu, done_kv = int(self.unit == "vpu"), int(self.unit == "kv")
                self.unit = ""
        stream_done = self.sdone
        dut.done_gemv.value = done_gemv
        dut.done_vpu.value = done_vpu
        dut.done_kv.value = done_kv
        dut.gemv_beat.value = beat
        dut.stream_busy.value = busy
        dut.stream_done.value = stream_done


async def setup(dut, *, seed: int = 0xD15, **kw) -> tuple[Env, random.Random]:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    b_max = len(dut.cmd_rows.value)
    for sig in (
        dut.start,
        dut.step,
        dut.abort_run,
        dut.pc_q,
        dut.row_en_q,
        dut.tok_q,
        dut.pos_q,
        dut.dq_valid,
        dut.dq_desc,
        dut.sreg_rd_data,
        dut.done_gemv,
        dut.done_vpu,
        dut.done_kv,
        dut.wr_idle,
        dut.gemv_beat,
        dut.stream_done,
        dut.stream_busy,
    ):
        sig.value = 0
    env = Env(dut, b_max=b_max, wb=int(kw.pop("wb", 16)), **kw)
    await qc_stream.reset(dut, dut.rst)
    cocotb.start_soon(env.run())
    return env, random.Random(seed)


async def pulse(dut, name: str) -> None:
    sig = getattr(dut, name)
    sig.value = 1
    await FallingEdge(dut.clk)
    sig.value = 0


async def wait_idle(dut, timeout: int = 40000) -> None:
    """Wait for busy to fall, then two more cycles so the bench sees the ending cycle."""
    for _ in range(timeout):
        await FallingEdge(dut.clk)
        if not int(dut.busy.value):
            await qc_stream.cycles(dut.clk, 2)
            return
    raise TimeoutError("the dispatcher never went idle")


async def run_program(dut, env: Env, program, *, pc: int = BASE, timeout: int = 40000) -> None:
    """Load ``program`` at ``pc``, pulse START and wait for the run to end."""
    env.load(program, pc)
    await FallingEdge(dut.clk)
    await pulse(dut, "start")
    await wait_idle(dut, timeout)


def check_bundle(got: dict, want: dict, where: str) -> None:
    for name in BUNDLE:
        w = want[name]
        if isinstance(w, dict):
            for r, v in w.items():
                if v is None:
                    continue
                assert got[name][r] == v, (
                    f"{where}: cmd_{name}[{r}] = {got[name][r]:#x}, expected {v:#x}"
                )
        else:
            assert got[name] == w, f"{where}: cmd_{name} = {got[name]:#x}, expected {w:#x}"


def kind_of(d: isa.Descriptor) -> str:
    if d.opcode in (Opcode.GEMV, Opcode.EMBED):
        return "gemv"
    if d.opcode == Opcode.KVWRITE:
        return "kv"
    return "vpu"


def zero_work(d: isa.Descriptor, *, pos: int, row_en: int, b_max: int, wb: int) -> bool:
    if d.opcode in (Opcode.NOP, Opcode.HALT, Opcode.FENCE):
        return False
    if not participating(d, row_en, b_max):
        return True
    if d.opcode == Opcode.GEMV:
        n, k, _ = isa_sim.gemv_dims(d, pos, wb)
        return n == 0 or k == 0
    if d.opcode == Opcode.EMBED:
        return d.k == 0
    return d.opcode in VPU_OPS and d.n == 0


def issuing(program, *, pos: int, row_en: int, b_max: int, wb: int) -> list[isa.Descriptor]:
    return [
        d
        for d in program
        if d.opcode not in (Opcode.NOP, Opcode.HALT, Opcode.FENCE)
        and not zero_work(d, pos=pos, row_en=row_en, b_max=b_max, wb=wb)
    ]


def every_opcode(rows: int = 1) -> list[isa.Descriptor]:
    """One descriptor of every opcode, shaped the way a compiled program shapes it."""
    vq = int(isa.VquantFlag.W8 | isa.VquantFlag.USE_TRACKED)
    return [
        isa.Descriptor(opcode=Opcode.NOP, row_mask=rows),
        isa.Descriptor(
            opcode=Opcode.EMBED,
            row_mask=rows,
            addr_a=0x2000,
            addr_m=0x3000,
            k=48,
            n=48,
            vs_dst=0,
            sh0=16,
            sh1=-8,
        ),
        isa.Descriptor(
            opcode=Opcode.VRMSNORM,
            row_mask=rows,
            addr_a=0x4000,
            addr_m=0x0007_8123,
            n=48,
            vs_src=0,
            vs_dst=64,
            sh0=20,
            sh1=12,
            imm32=5,
            sreg_dst=3,
            track_absmax=True,
        ),
        isa.Descriptor(
            opcode=Opcode.VQUANT,
            row_mask=rows,
            flags=vq,
            n=48,
            vs_src=64,
            vs_dst=128,
            sreg_src=3,
            sreg_dst=1,
            sh0=15,
        ),
        isa.Descriptor(
            opcode=Opcode.GEMV,
            row_mask=rows,
            addr_a=0x1_0000,
            addr_m=0x2_0000,
            n=64,
            k=48,
            vs_src=128,
            vs_dst=192,
            sreg_src=1,
            sh0=12,
            sh1=-3,
            out_mode=OutMode.VSRAM,
            track_absmax=True,
            sreg_dst=4,
        ),
        isa.Descriptor(opcode=Opcode.VROPE, row_mask=rows, addr_a=0x3_0000, n=64, vs_src=192),
        isa.Descriptor(
            opcode=Opcode.KVWRITE,
            row_mask=rows,
            flags=int(isa.KvwriteFlag.TRANSPOSED),
            addr_a=0x4_0000,
            addr_m=0x5_0000,
            k=2048,
            vs_src=192,
            sreg_src=1,
        ),
        isa.Descriptor(
            opcode=Opcode.GEMV,
            row_mask=rows,
            n_from_pos=True,
            addr_a=0x6_0000,
            addr_m=0x7_0000,
            n=2048,
            k=64,
            vs_src=192,
            vs_dst=256,
            sreg_src=1,
            sh0=9,
            sh1=-2,
        ),
        isa.Descriptor(
            opcode=Opcode.VSOFTMAX,
            row_mask=rows,
            len_from_pos=True,
            addr_a=0x5_0000,
            n=2048,
            vs_src=256,
            vs_dst=512,
            sh0=24,
            sreg_dst=6,
        ),
        isa.Descriptor(
            opcode=Opcode.GEMV,
            row_mask=rows,
            k_from_pos=True,
            unit_meta=True,
            addr_a=0x8_0000,
            n=64,
            k=2048,
            vs_src=512,
            vs_dst=576,
            sreg_src=6,
            sh0=14,
            sh1=-1,
        ),
        isa.Descriptor(
            opcode=Opcode.VSILUMUL,
            row_mask=rows,
            n=64,
            vs_src=576,
            vs_aux=640,
            vs_dst=704,
            sh0=13,
            sh1=6,
            sreg_dst=2,
            track_absmax=True,
        ),
        isa.Descriptor(
            opcode=Opcode.VSUBC, row_mask=rows, addr_a=0x9_0000, n=64, vs_src=704, vs_dst=768
        ),
        isa.Descriptor(
            opcode=Opcode.GEMV,
            row_mask=rows,
            addr_a=0xA_0000,
            addr_m=0xB_0000,
            n=96,
            k=64,
            vs_src=768,
            vs_dst=832,
            sreg_src=2,
            sh0=11,
            sh1=-4,
            out_mode=OutMode.ARGMAX_DUMP,
            imm32=0xC_0000,
        ),
        isa.Descriptor(opcode=Opcode.FENCE),
        isa.Descriptor(opcode=Opcode.HALT),
    ]


@cocotb.test()
async def test_program_of_every_opcode(dut):
    """Every opcode issues in program order with the bundle isa.decode says it carries."""
    env, _ = await setup(dut, wb=16)
    env.row_en = (1 << env.b_max) - 1
    env.pos, env.tok = 63, 7
    program = every_opcode(rows=(1 << env.b_max) - 1)
    await run_program(dut, env, program)
    want = issuing(program, pos=env.pos, row_en=env.row_en, b_max=env.b_max, wb=env.wb)
    assert len(env.issues) == len(want), (
        f"{len(env.issues)} issues for {len(want)} issuing descriptors"
    )
    for i, (got, d) in enumerate(zip(env.issues, want, strict=True)):
        assert got["kind"] == kind_of(d), f"descriptor {i}: issued on the {got['kind']} port"
        check_bundle(
            got,
            expect(d, pos=env.pos, tok=env.tok, row_en=env.row_en, b_max=env.b_max, wb=env.wb),
            f"descriptor {i} ({d.opcode.name})",
        )
    assert env.descriptors == len(program), f"{env.descriptors} descriptors retired"
    for i, r in enumerate(env.retires):
        assert r["pc"] == BASE + i * isa.DESC_BYTES, f"retire {i} at PC {r['pc']:#x}"
    assert env.pc == BASE + len(program) * isa.DESC_BYTES, "PC past the HALT"
    assert (env.done, env.err, env.step_halted) == (1, 0, 0), "the run did not end on HALT"
    assert env.snapshots == 1 and env.clears == 1, "one PERF clear and one snapshot per run"


@cocotb.test()
async def test_issue_follows_a_pop_by_three_cycles(dut):
    """One descriptor is in flight and its issue pulse follows the pop by at least three cycles."""
    env, _ = await setup(dut, wb=16)
    program = [d for d in every_opcode() if d.opcode not in (Opcode.NOP, Opcode.FENCE)]
    await run_program(dut, env, program)
    assert len(env.pops) == len(program), f"{len(env.pops)} pops for {len(program)} descriptors"
    for i, issue in enumerate(env.issues):
        pop = max(p for p in env.pops if p < issue["cycle"])
        assert issue["cycle"] - pop >= 3, (
            f"issue {i}: pulsed {issue['cycle'] - pop} cycles after its pop"
        )
    for a, b in zip(env.issues, env.issues[1:], strict=False):
        assert b["cycle"] > a["cycle"], "two issues in one cycle"
    for a, b in zip(env.pops, env.pops[1:], strict=False):
        assert b > a, "two descriptors popped in one cycle"
    for issue, retire in zip(env.issues, env.retires, strict=False):
        assert retire["cycle"] > issue["cycle"], "a retire before its issue"
    assert env.err == 0


@cocotb.test()
async def test_pos_derived_fields(dut):
    """cmd_n, cmd_k, cmd_len and the ERR_BOUNDS count follow gemv_dims and softmax_len."""
    env, _ = await setup(dut, wb=16)
    program = [d for d in every_opcode() if d.opcode in (Opcode.GEMV, Opcode.VSOFTMAX)]
    program.append(isa.Descriptor(opcode=Opcode.HALT))
    for pos in (0, 1, 63, 64, 2047):
        env.issues.clear()
        env.err_bounds = env.macs = env.wt_bytes = 0
        env.pos = pos
        await run_program(dut, env, program)
        want = issuing(program, pos=pos, row_en=env.row_en, b_max=env.b_max, wb=env.wb)
        assert len(env.issues) == len(want), f"POS {pos}: {len(env.issues)} issues"
        errs = 0
        for got, d in zip(env.issues, want, strict=True):
            check_bundle(
                got,
                expect(d, pos=pos, tok=env.tok, row_en=env.row_en, b_max=env.b_max, wb=env.wb),
                f"POS {pos} ({d.opcode.name})",
            )
        for d in program:
            errs += expect_events(d, pos=pos, row_en=env.row_en, b_max=env.b_max, wb=env.wb)[
                "err_bounds"
            ]
        assert env.err_bounds == errs, f"POS {pos}: ERR_BOUNDS {env.err_bounds}, expected {errs}"


@cocotb.test()
async def test_macs_and_wt_bytes_per_row(dut):
    """MACS and WT_BYTES are the per-row totals isa_sim counts, for one row and for two."""
    env, _ = await setup(dut, wb=16)
    for rows in (1, (1 << env.b_max) - 1):
        for pos in (0, 63, 2047):
            env.row_en = rows
            env.pos = pos
            env.macs = env.wt_bytes = env.err_bounds = 0
            program = every_opcode(rows=rows)
            await run_program(dut, env, program)
            want = {"macs": 0, "wt_bytes": 0, "err_bounds": 0}
            for d in program:
                for key, v in expect_events(
                    d, pos=pos, row_en=rows, b_max=env.b_max, wb=env.wb
                ).items():
                    want[key] += v
            assert env.macs == want["macs"], (
                f"rows {rows:#x} POS {pos}: MACS {env.macs} != {want['macs']}"
            )
            assert env.wt_bytes == want["wt_bytes"], (
                f"rows {rows:#x} POS {pos}: WT_BYTES {env.wt_bytes} != {want['wt_bytes']}"
            )
            assert env.err_bounds == want["err_bounds"], (
                f"rows {rows:#x} POS {pos}: ERR_BOUNDS {env.err_bounds} != {want['err_bounds']}"
            )
    env.row_en = 1


def bucket_name(mask: int) -> str:
    return ", ".join(n for i, n in enumerate(BUCKETS) if (mask >> i) & 1) or "none"


def bucket_counts(env: Env, first: int, last: int) -> dict[str, int]:
    """How many cycles in ``[first, last]`` fell into each bucket (env cycle numbers)."""
    out = dict.fromkeys(BUCKETS, 0)
    for busy, mask in env.trace[first - 1 : last]:
        assert busy, "a cycle of the descriptor was not busy"
        for i, name in enumerate(BUCKETS):
            out[name] += (mask >> i) & 1
    return out


@cocotb.test()
async def test_auto_fence_ordering(dut):
    """Only the QMEM-reading opcodes, FENCE and HALT wait for wr_idle; the wait is STALL_KV."""
    env, _ = await setup(dut, wb=16)
    env.wr_idle = 0
    by_op = {d.opcode: d for d in every_opcode()}
    free = [by_op[op] for op in (Opcode.VQUANT, Opcode.VSILUMUL, Opcode.KVWRITE)]
    program = [*free, by_op[Opcode.GEMV], isa.Descriptor(opcode=Opcode.HALT)]
    env.load(program)
    await FallingEdge(dut.clk)
    await pulse(dut, "start")
    await qc_stream.cycles(dut.clk, 120)
    assert len(env.issues) == len(free), (
        f"{len(env.issues)} issues with wr_idle low, expected {len(free)} unfenced descriptors"
    )
    assert all(i["op"] != int(Opcode.GEMV) for i in env.issues), "a GEMV issued through the fence"
    assert int(dut.busy.value), "the dispatcher gave up on the fence"
    assert qc_stream.value(dut.ev_bucket) == 1 << BUCKETS.index("STALL_KV"), (
        f"waiting on wr_idle is {bucket_name(qc_stream.value(dut.ev_bucket))}, not STALL_KV"
    )
    env.wr_idle = 1
    await wait_idle(dut)
    assert len(env.issues) == len(free) + 1, "the GEMV never issued after wr_idle rose"
    assert env.descriptors == len(program) and env.done == 1


@cocotb.test()
async def test_gemv_buckets_and_stream_idle_gate(dut):
    """A GEMV retires only once the stream is idle; its cycles split MAC_ACTIVE / DRAIN / MEM."""
    env, _ = await setup(dut, wb=16, gemv_beats=6, gemv_drain=5)
    gemv = next(d for d in every_opcode() if d.opcode == Opcode.GEMV)
    embed = next(d for d in every_opcode() if d.opcode == Opcode.EMBED)
    for d, macs in ((gemv, 6), (embed, 0)):
        env.issues.clear()
        env.retires.clear()
        await run_program(dut, env, [d, isa.Descriptor(opcode=Opcode.HALT)])
        issue = env.issues[0]["cycle"]
        retire = env.retires[0]["cycle"]
        assert retire == issue + env.gemv_beats + env.gemv_drain + 1, (
            f"{d.opcode.name}: retired {retire - issue} cycles after the issue"
        )
        assert retire > issue + env.gemv_beats + 2, (
            f"{d.opcode.name}: retired on done_gemv without waiting for the stream to go idle"
        )
        # On the issue cycle the descriptor is waiting for its first beat.
        head = env.trace[issue - 1][1]
        assert head == 1 << BUCKETS.index("STALL_MEM"), (
            f"{d.opcode.name}: the issue cycle is {bucket_name(head)}, not STALL_MEM"
        )
        counts = bucket_counts(env, issue + 1, retire - 1)
        assert counts["MAC_ACTIVE"] == macs, f"{d.opcode.name}: MAC_ACTIVE {counts['MAC_ACTIVE']}"
        assert counts["STALL_DRAIN"] == env.gemv_drain, (
            f"{d.opcode.name}: STALL_DRAIN {counts['STALL_DRAIN']}"
        )
        assert counts["STALL_MEM"] == env.gemv_beats - macs, (
            f"{d.opcode.name}: STALL_MEM {counts['STALL_MEM']}"
        )
        assert counts["STALL_VPU"] == 0 and counts["STALL_KV"] == 0


@cocotb.test()
async def test_zero_work_retires_without_an_issue(dut):
    """N == 0, K == 0, an empty row set and a V op with n == 0 retire with nothing issued."""
    env, _ = await setup(dut, wb=16)
    env.pos = 40
    program = [
        isa.Descriptor(opcode=Opcode.GEMV, n=0, k=48, addr_a=0x1000, addr_m=0x2000),
        isa.Descriptor(opcode=Opcode.GEMV, n=64, k=0, addr_a=0x1000, addr_m=0x2000),
        isa.Descriptor(opcode=Opcode.VRMSNORM, n=0, addr_a=0x4000, vs_dst=64),
        isa.Descriptor(opcode=Opcode.GEMV, row_mask=0, n=64, k=48, addr_a=0x1000),
        isa.Descriptor(opcode=Opcode.EMBED, k=0, addr_a=0x2000, addr_m=0x3000),
        isa.Descriptor(opcode=Opcode.GEMV, n_from_pos=True, n=0, k=48, addr_a=0x1000),
        isa.Descriptor(opcode=Opcode.VSOFTMAX, n=0, len_from_pos=True, addr_a=0x5000),
        isa.Descriptor(opcode=Opcode.HALT),
    ]
    await run_program(dut, env, program)
    assert env.issues == [], f"{len(env.issues)} issues for a program with no work"
    assert env.descriptors == len(program), "a zero-work descriptor did not retire"
    assert (env.macs, env.wt_bytes) == (0, 0), "zero work counted MACS or WT_BYTES"
    want = sum(
        expect_events(d, pos=env.pos, row_en=env.row_en, b_max=env.b_max, wb=env.wb)["err_bounds"]
        for d in program
    )
    assert env.err_bounds == want, f"ERR_BOUNDS {env.err_bounds}, expected {want}"
    assert env.done == 1 and env.err == 0


@cocotb.test()
async def test_step_mode(dut):
    """CTRL.STEP runs exactly one descriptor, snapshots PERF and sets STEP_HALTED (DONE at HALT)."""
    env, _ = await setup(dut, wb=16)
    program = every_opcode()
    env.load(program)
    for i, d in enumerate(program):
        before = (env.descriptors, env.snapshots, env.pc)
        await FallingEdge(dut.clk)
        await pulse(dut, "step")
        await wait_idle(dut)
        assert env.descriptors == before[0] + 1, f"step {i}: {d.opcode.name} did not retire"
        assert env.snapshots == before[1] + 1, f"step {i}: PERF was not snapshotted"
        assert env.pc == before[2] + isa.DESC_BYTES, f"step {i}: PC {env.pc:#x}"
        assert env.clears == 0, "STEP must not clear the counters"
        if d.opcode == Opcode.HALT:
            assert env.done == 1 and env.step_halted == i, "HALT must set DONE, not STEP_HALTED"
        else:
            assert env.step_halted == i + 1, f"step {i}: STEP_HALTED not set"
            assert env.done == 0, f"step {i}: DONE set before HALT"
    assert env.err == 0


@cocotb.test()
async def test_fault_halt(dut):
    """An unknown opcode, an out-of-range row and a misaligned PC each stop the program."""
    env, _ = await setup(dut, wb=16)
    env.row_en = (1 << env.b_max) - 1
    unknown = (isa.Descriptor(opcode=Opcode.GEMV).word() & ~0xFF) | 0x77
    gemv = next(d for d in every_opcode() if d.opcode == Opcode.GEMV)
    bad_row = isa.Descriptor(
        opcode=Opcode.GEMV, row_mask=(1 << env.b_max) - 1, src_row=1, n=64, k=48, addr_a=0x1000
    )
    cases = (
        (
            [isa.Descriptor(opcode=Opcode.NOP), unknown, isa.Descriptor(opcode=Opcode.HALT)],
            BASE + isa.DESC_BYTES,
            Fault.OPCODE,
            0x77,
            1,
            0,
        ),
        (
            [gemv, bad_row, isa.Descriptor(opcode=Opcode.HALT)],
            BASE + isa.DESC_BYTES,
            Fault.ROW,
            int(Opcode.GEMV),
            1,
            1,
        ),
    )
    for program, pc, code, opbyte, descs, issues in cases:
        env.issues.clear()
        env.descriptors = env.done = env.err = env.snapshots = 0
        env.fault = None
        await run_program(dut, env, program)
        assert env.fault == (int(code), opbyte), f"{code.name}: fault {env.fault}"
        assert env.pc == pc, f"{code.name}: PC {env.pc:#x}, expected {pc:#x} (the faulting one)"
        assert env.descriptors == descs, f"{code.name}: {env.descriptors} descriptors retired"
        assert len(env.issues) == issues, f"{code.name}: {len(env.issues)} issues"
        assert (env.done, env.err, env.snapshots) == (1, 1, 1), f"{code.name}: status"
    # a misaligned PC faults before the first fetch request
    env.descriptors = env.done = env.err = env.snapshots = 0
    env.fault = None
    starts = len(env.fetch_starts)
    env.load([isa.Descriptor(opcode=Opcode.HALT)], BASE)
    env.pc = BASE + 16
    await FallingEdge(dut.clk)
    await pulse(dut, "start")
    await wait_idle(dut)
    assert env.fault == (int(Fault.PC_ALIGN), 0), f"PC_ALIGN: fault {env.fault}"
    assert env.pc == BASE + 16, "a PC_ALIGN fault must leave PC where it is"
    assert len(env.fetch_starts) == starts, "a fetch was started with a misaligned PC"
    assert (env.done, env.err, env.snapshots, env.descriptors) == (1, 1, 1, 0)


@cocotb.test()
async def test_abort_lets_the_descriptor_in_flight_retire(dut):
    """ABORT stops further issue, retires what is in flight and ends the run with DONE."""
    env, _ = await setup(dut, wb=16, vpu_cycles=12)
    program = [d for d in every_opcode() if d.opcode in VPU_OPS] * 2
    program.append(isa.Descriptor(opcode=Opcode.HALT))
    env.load(program)
    await FallingEdge(dut.clk)
    await pulse(dut, "start")
    while len(env.issues) < 2:
        await FallingEdge(dut.clk)
    await pulse(dut, "abort_run")
    issued = len(env.issues)
    retired = env.descriptors
    await wait_idle(dut)
    assert len(env.issues) == issued, "a descriptor issued after ABORT"
    assert env.descriptors == retired + 1, "the descriptor in flight did not retire"
    assert (env.done, env.err, env.step_halted) == (1, 0, 0), "ABORT must end the run with DONE"
    assert env.snapshots == 1, "ABORT must snapshot PERF once"
    assert env.pc == BASE + env.descriptors * isa.DESC_BYTES, "PC past the aborted descriptor"


@cocotb.test()
async def test_sreg_reads_per_participating_row(dut):
    """GEMV, KVWRITE and a tracked VQUANT read one SREG word per participating row, ascending."""
    env, _ = await setup(dut, wb=16)
    env.row_en = (1 << env.b_max) - 1
    rows = (1 << env.b_max) - 1
    program = [
        d
        for d in every_opcode(rows=rows)
        if d.opcode in (Opcode.GEMV, Opcode.KVWRITE, Opcode.VQUANT, Opcode.EMBED, Opcode.VROPE)
    ]
    program.append(isa.Descriptor(opcode=Opcode.HALT))
    await run_program(dut, env, program)
    want: list[tuple[int, int]] = []
    for d in program:
        tracked = d.opcode == Opcode.VQUANT and (d.flags & isa.VquantFlag.USE_TRACKED)
        if d.opcode in (Opcode.GEMV, Opcode.KVWRITE) or tracked:
            want += [(d.src_row + r, d.sreg_src) for r in participating(d, env.row_en, env.b_max)]
    assert env.sreg_reads == want, f"SREG reads {env.sreg_reads}, expected {want}"
    env.row_en = 1


@cocotb.test()
async def test_stall_bucket_invariant(dut):
    """Every busy cycle of a random program lands in exactly one bucket and no idle cycle does."""
    env, rng = await setup(dut, wb=16, dq_delay=3)
    pool = [d for d in every_opcode() if d.opcode not in (Opcode.HALT,)]
    pool += [isa.Descriptor(opcode=Opcode.NOP), isa.Descriptor(opcode=Opcode.FENCE)]
    for trial in range(4):
        env.pos = rng.choice((0, 1, 63, 64, 2047))
        env.trace.clear()
        program = [rng.choice(pool) for _ in range(18)]
        program.append(isa.Descriptor(opcode=Opcode.HALT))
        await run_program(dut, env, program)
        busy_cycles = sum(busy for busy, _ in env.trace)
        total = 0
        for cycle, (busy, mask) in enumerate(env.trace, start=1):
            if busy:
                assert mask and not (mask & (mask - 1)), (
                    f"trial {trial} cycle {cycle}: buckets {bucket_name(mask)}"
                )
                total += 1
            else:
                assert mask == 0, f"trial {trial} cycle {cycle}: bucket while idle"
        assert total == busy_cycles, "BUSY is not the sum of the six buckets"
        assert busy_cycles > 100, "the trial was too short to be meaningful"


@cocotb.test()
async def test_the_issue_cycle_of_a_gemv_is_stall_mem(dut):
    """The stream re-evaluates stream_done at the command pulse, so on the issue cycle the level
    still belongs to the descriptor before it; the new one is waiting for its first beat."""
    env, _ = await setup(dut, wb=16, gemv_beats=6, gemv_drain=5)
    gemv = next(d for d in every_opcode() if d.opcode == Opcode.GEMV)
    embed = next(d for d in every_opcode() if d.opcode == Opcode.EMBED)
    program = [gemv, gemv, embed, gemv, isa.Descriptor(opcode=Opcode.HALT)]
    await run_program(dut, env, program)
    assert len(env.issues) == 4, f"{len(env.issues)} issues"
    for i, (issue, retire) in enumerate(zip(env.issues, env.retires, strict=False)):
        macs = env.gemv_beats if issue["op"] == int(Opcode.GEMV) else 0
        head = env.trace[issue["cycle"] - 1][1]
        assert head == 1 << BUCKETS.index("STALL_MEM"), (
            f"descriptor {i}: the issue cycle is {bucket_name(head)}, not STALL_MEM"
        )
        # pc_set is registered, so the descriptor is in flight from its issue cycle to the
        # cycle before the retire is observed.
        counts = bucket_counts(env, issue["cycle"], retire["cycle"] - 1)
        assert counts["MAC_ACTIVE"] == macs, f"descriptor {i}: MAC_ACTIVE {counts['MAC_ACTIVE']}"
        assert counts["STALL_MEM"] == 1 + env.gemv_beats - macs, (
            f"descriptor {i}: STALL_MEM {counts['STALL_MEM']}"
        )
        assert counts["STALL_DRAIN"] == env.gemv_drain, (
            f"descriptor {i}: STALL_DRAIN {counts['STALL_DRAIN']}"
        )
        assert counts["STALL_VPU"] == 0 and counts["STALL_KV"] == 0 and counts["STALL_SEQ"] == 0


@cocotb.test()
async def test_a_fault_declares_done_only_after_the_writes_land(dut):
    """A fault stops the program at once and waits on the write fence before DONE, as HALT does."""
    env, _ = await setup(dut, wb=16)
    env.wr_idle = 0
    unknown = (isa.Descriptor(opcode=Opcode.GEMV).word() & ~0xFF) | 0x77
    program = [isa.Descriptor(opcode=Opcode.NOP), unknown, isa.Descriptor(opcode=Opcode.HALT)]
    env.load(program)
    await FallingEdge(dut.clk)
    await pulse(dut, "start")
    await qc_stream.cycles(dut.clk, 60)
    assert int(dut.busy.value), "the fault stop gave up on the write fence"
    assert (env.done, env.err, env.snapshots) == (0, 0, 0), "DONE before the writes were acked"
    assert qc_stream.value(dut.ev_bucket) == 1 << BUCKETS.index("STALL_KV"), (
        f"the fence a fault ends on is {bucket_name(qc_stream.value(dut.ev_bucket))}, not STALL_KV"
    )
    env.wr_idle = 1
    await wait_idle(dut)
    assert (env.done, env.err, env.snapshots) == (1, 1, 1), "the fault never ended the run"
    assert env.fault == (int(Fault.OPCODE), 0x77), f"fault {env.fault}"
    assert env.pc == BASE + isa.DESC_BYTES, f"PC {env.pc:#x}, expected the faulting descriptor"
    assert env.descriptors == 1, f"{env.descriptors} descriptors retired"


@cocotb.test()
async def test_an_abort_declares_done_only_after_the_writes_land(dut):
    """ABORT stops issuing at once; DONE waits for the writes of the descriptor that retired."""
    env, _ = await setup(dut, wb=16, kv_cycles=8)
    env.wr_idle = 0
    kv = next(d for d in every_opcode() if d.opcode == Opcode.KVWRITE)
    program = [kv, kv, kv, isa.Descriptor(opcode=Opcode.HALT)]
    env.load(program)
    await FallingEdge(dut.clk)
    await pulse(dut, "start")
    while not env.issues:
        await FallingEdge(dut.clk)
    await pulse(dut, "abort_run")
    issued = len(env.issues)
    await qc_stream.cycles(dut.clk, 60)
    assert len(env.issues) == issued, "a descriptor issued after ABORT"
    assert env.descriptors == 1, "the descriptor in flight did not retire promptly"
    assert int(dut.busy.value), "the ABORT stop gave up on the write fence"
    assert env.done == 0, "ABORT declared DONE with writes unacknowledged"
    assert qc_stream.value(dut.ev_bucket) == 1 << BUCKETS.index("STALL_KV"), (
        f"the fence an ABORT ends on is {bucket_name(qc_stream.value(dut.ev_bucket))}, not STALL_KV"
    )
    env.wr_idle = 1
    await wait_idle(dut)
    assert (env.done, env.err, env.snapshots) == (1, 0, 1), "ABORT must end the run with DONE"
    assert env.pc == BASE + isa.DESC_BYTES, f"PC {env.pc:#x}, expected past the aborted descriptor"


@cocotb.test()
async def test_abort_stops_a_descriptor_waiting_on_the_fence(dut):
    """A descriptor popped but held at the auto-fence is not issued: ABORT leaves PC on it."""
    env, _ = await setup(dut, wb=16)
    env.wr_idle = 0
    gemv = next(d for d in every_opcode() if d.opcode == Opcode.GEMV)
    env.load([gemv, isa.Descriptor(opcode=Opcode.HALT)])
    await FallingEdge(dut.clk)
    await pulse(dut, "start")
    await qc_stream.cycles(dut.clk, 40)
    assert len(env.pops) == 1 and env.issues == [], "the GEMV is not held at the fence"
    await pulse(dut, "abort_run")
    await qc_stream.cycles(dut.clk, 40)
    assert env.issues == [], "the GEMV issued after ABORT"
    assert int(dut.busy.value), "the ABORT stop gave up on the write fence"
    env.wr_idle = 1
    await wait_idle(dut)
    assert env.issues == [], "the GEMV issued once the fence opened"
    assert env.descriptors == 0, f"{env.descriptors} descriptors retired"
    assert env.pc == BASE, f"PC {env.pc:#x}, expected the descriptor that never ran"
    assert (env.done, env.err, env.snapshots) == (1, 0, 1)


@cocotb.test()
async def test_abort_at_every_point_of_a_descriptor_setup(dut):
    """Wherever ABORT lands, nothing is issued after it and PC ends on a retired boundary."""
    env, _ = await setup(dut, wb=16, dq_delay=2)
    program = [
        next(d for d in every_opcode() if d.opcode == Opcode.GEMV),
        next(d for d in every_opcode() if d.opcode == Opcode.KVWRITE),
        next(d for d in every_opcode() if d.opcode == Opcode.EMBED),
        isa.Descriptor(opcode=Opcode.HALT),
    ]
    for offset in range(1, 46):
        env.issues.clear()
        env.aborts.clear()
        env.descriptors = env.done = env.err = env.snapshots = 0
        env.load(program)
        await FallingEdge(dut.clk)
        await pulse(dut, "start")
        await qc_stream.cycles(dut.clk, offset)
        await pulse(dut, "abort_run")
        await wait_idle(dut)
        abort = env.aborts[0]
        late = [i["cycle"] for i in env.issues if i["cycle"] > abort + 1]
        assert not late, f"offset {offset}: issued at {late} after ABORT in cycle {abort}"
        assert env.descriptors in (len(env.issues), len(env.issues) + 1), (
            f"offset {offset}: {env.descriptors} retired for {len(env.issues)} issues"
        )
        assert env.pc == BASE + env.descriptors * isa.DESC_BYTES, f"offset {offset}: PC"
        assert (env.done, env.err, env.snapshots) == (1, 0, 1), f"offset {offset}: status"


@cocotb.test()
async def test_the_prefetch_takes_the_write_fence(dut):
    """fetch_hold holds the descriptor prefetch exactly while a write is unacknowledged."""
    env, _ = await setup(dut, wb=16)
    for idle in (1, 0, 1, 0):
        env.wr_idle = idle
        await qc_stream.cycles(dut.clk, 2)
        assert int(dut.fetch_hold.value) == (not idle), (
            f"fetch_hold {int(dut.fetch_hold.value)} with wr_idle {idle}"
        )


# A GEMV whose executed extents both come from POS: at a position past its capacity it raises two
# bounds events per participating row, which is what the tests below count.
CAPPED_GEMV = isa.Descriptor(
    opcode=Opcode.GEMV,
    n_from_pos=True,
    k_from_pos=True,
    addr_a=0x1_0000,
    addr_m=0x2_0000,
    n=16,
    k=8,
    vs_src=0,
    vs_dst=64,
    sreg_src=1,
    sh0=12,
    sh1=-3,
)


@cocotb.test()
async def test_an_unexecuted_descriptor_counts_no_bounds_event(dut):
    """ERR_BOUNDS follows the descriptors the core commits to, not the ones an ABORT discards."""
    env, _ = await setup(dut, wb=16)
    env.pos = 2047
    halt = isa.Descriptor(opcode=Opcode.HALT)
    per = expect_events(CAPPED_GEMV, pos=env.pos, row_en=env.row_en, b_max=env.b_max, wb=env.wb)[
        "err_bounds"
    ]
    assert per > 0, "the shape must raise a bounds event when it runs"

    # It runs: the events of one descriptor are counted once.
    env.err_bounds = 0
    await run_program(dut, env, [CAPPED_GEMV, halt])
    assert env.err_bounds == per, f"ERR_BOUNDS {env.err_bounds} for one executed descriptor"

    # An ABORT holds it at the auto-fence: it is never issued, so it counts nothing.
    env.issues.clear()
    env.err_bounds = env.descriptors = env.done = env.err = env.snapshots = 0
    env.wr_idle = 0
    env.load([CAPPED_GEMV, halt])
    await FallingEdge(dut.clk)
    await pulse(dut, "start")
    await qc_stream.cycles(dut.clk, 40)
    assert env.issues == [] and len(env.pops) > 0, "the GEMV is not held at the fence"
    await pulse(dut, "abort_run")
    env.wr_idle = 1
    await wait_idle(dut)
    assert env.issues == [], "the GEMV issued after ABORT"
    assert env.descriptors == 0, f"{env.descriptors} descriptors retired"
    assert env.err_bounds == 0, (
        f"ERR_BOUNDS {env.err_bounds} for a descriptor the ABORT stopped before its issue"
    )
    assert env.pc == BASE, "PC must stay on the descriptor that never ran"

    # Wherever the ABORT lands, the count is the count of the descriptors that issued.
    for offset in range(1, 30):
        env.issues.clear()
        env.err_bounds = env.descriptors = env.done = env.err = env.snapshots = 0
        env.load([CAPPED_GEMV, halt])
        await FallingEdge(dut.clk)
        await pulse(dut, "start")
        await qc_stream.cycles(dut.clk, offset)
        await pulse(dut, "abort_run")
        await wait_idle(dut)
        want = per * len(env.issues)
        assert env.err_bounds == want, (
            f"offset {offset}: ERR_BOUNDS {env.err_bounds} for {len(env.issues)} issued "
            f"descriptors, expected {want}"
        )


@cocotb.test()
async def test_a_zero_work_descriptor_still_counts_its_bounds_event(dut):
    """A capped VSOFTMAX with n == 0 retires without an issue and still counts its clamped len."""
    env, _ = await setup(dut, wb=16)
    env.pos = 2047
    program = [
        isa.Descriptor(opcode=Opcode.VSOFTMAX, n=0, len_from_pos=True, addr_a=0x5000),
        isa.Descriptor(opcode=Opcode.HALT),
    ]
    env.err_bounds = 0
    await run_program(dut, env, program)
    want = sum(
        expect_events(d, pos=env.pos, row_en=env.row_en, b_max=env.b_max, wb=env.wb)["err_bounds"]
        for d in program
    )
    assert want > 0, "the shape must raise a bounds event"
    assert env.issues == [], "a descriptor with no work issued"
    assert env.err_bounds == want, f"ERR_BOUNDS {env.err_bounds}, expected {want}"


@cocotb.test()
async def test_a_stepped_descriptor_halts_on_the_write_fence(dut):
    """STEP_HALTED waits on the same write fence every other stop takes."""
    env, _ = await setup(dut, wb=16, kv_cycles=6)
    kv = next(d for d in every_opcode() if d.opcode == Opcode.KVWRITE)
    env.load([kv, isa.Descriptor(opcode=Opcode.HALT)])
    env.wr_idle = 0
    await FallingEdge(dut.clk)
    await pulse(dut, "step")
    await qc_stream.cycles(dut.clk, 60)
    assert len(env.issues) == 1, "the stepped KVWRITE did not issue"
    assert env.descriptors == 1, "the stepped KVWRITE did not retire"
    assert int(dut.busy.value), "the step halt gave up on the write fence"
    assert (env.step_halted, env.done, env.snapshots) == (0, 0, 0), (
        "STEP_HALTED with the descriptor's writes unacknowledged"
    )
    assert qc_stream.value(dut.ev_bucket) == 1 << BUCKETS.index("STALL_KV"), (
        f"the fence a step ends on is {bucket_name(qc_stream.value(dut.ev_bucket))}, not STALL_KV"
    )
    env.wr_idle = 1
    await wait_idle(dut)
    assert (env.step_halted, env.done, env.err) == (1, 0, 0), "the step never set STEP_HALTED"
    assert env.snapshots == 1, "the step must snapshot PERF once"
    assert env.pc == BASE + isa.DESC_BYTES, f"PC {env.pc:#x}"


@cocotb.test()
async def test_a_step_an_abort_cuts_short_ends_on_done(dut):
    """An ABORT written during a step ends the run on the fence with DONE, not STEP_HALTED."""
    env, _ = await setup(dut, wb=16, kv_cycles=8)
    kv = next(d for d in every_opcode() if d.opcode == Opcode.KVWRITE)
    env.load([kv, isa.Descriptor(opcode=Opcode.HALT)])
    env.wr_idle = 0
    await FallingEdge(dut.clk)
    await pulse(dut, "step")
    while not env.issues:
        await FallingEdge(dut.clk)
    await pulse(dut, "abort_run")
    await qc_stream.cycles(dut.clk, 40)
    assert int(dut.busy.value), "the ABORT stop gave up on the write fence"
    assert (env.done, env.step_halted) == (0, 0), "the run ended with writes unacknowledged"
    env.wr_idle = 1
    await wait_idle(dut)
    assert (env.done, env.step_halted, env.err) == (1, 0, 0), "an aborted step must set DONE"
    assert env.descriptors == 1 and env.snapshots == 1


# The sweep: every shape whose extents come from POS, every position that lands on a tile edge or
# a capacity, and every non-empty row set of the tiny configuration.
SWEEP_POSITIONS = (0, 1, 15, 16, 17, 31, 63, 64, 65, 127, 2047, 4095)
SWEEP_ROWS = ((1, 1), (1, 3), (2, 3), (3, 3), (3, 1), (3, 2))  # (row_mask, ROW_EN)


def pos_shapes() -> list[isa.Descriptor]:
    """GEMV, EMBED and VSOFTMAX shapes covering every POS-derived extent and its capacity."""
    out: list[isa.Descriptor] = []
    for n, k in ((1, 1), (16, 1), (17, 15), (48, 64), (64, 33)):
        out.append(
            isa.Descriptor(
                opcode=Opcode.GEMV,
                addr_a=0x1_0000,
                addr_m=0x2_0000,
                n=n,
                k=k,
                vs_src=0,
                vs_dst=64,
                sreg_src=1,
                sh0=12,
                sh1=-3,
            )
        )
        out.append(dataclasses.replace(out[-1], unit_meta=True))
    for cap in (16, 64, 2048):
        out.append(
            isa.Descriptor(
                opcode=Opcode.GEMV,
                n_from_pos=True,
                addr_a=0x6_0000,
                addr_m=0x7_0000,
                n=cap,
                k=64,
                vs_src=0,
                vs_dst=64,
                sreg_src=1,
                sh0=9,
                sh1=-2,
            )
        )
        out.append(
            isa.Descriptor(
                opcode=Opcode.GEMV,
                k_from_pos=True,
                unit_meta=True,
                addr_a=0x8_0000,
                n=64,
                k=cap,
                vs_src=0,
                vs_dst=64,
                sreg_src=1,
                sh0=14,
                sh1=-1,
            )
        )
        out.append(
            isa.Descriptor(
                opcode=Opcode.GEMV,
                n_from_pos=True,
                k_from_pos=True,
                addr_a=0x9_0000,
                addr_m=0xA_0000,
                n=cap,
                k=cap,
                vs_src=0,
                vs_dst=64,
                sreg_src=1,
                sh0=10,
                sh1=-2,
            )
        )
    for k in (1, 8, 48, 64):
        out.append(
            isa.Descriptor(
                opcode=Opcode.EMBED, addr_a=0x2000, addr_m=0x3000, k=k, n=k, sh0=16, sh1=-8
            )
        )
    for n in (16, 64, 2048):
        out.append(
            isa.Descriptor(
                opcode=Opcode.VSOFTMAX,
                len_from_pos=True,
                addr_a=0x5_0000,
                n=n,
                vs_dst=512,
                sh0=24,
                sreg_dst=6,
            )
        )
    for imm in (0, 1, 64, 200):
        out.append(
            isa.Descriptor(
                opcode=Opcode.VSOFTMAX,
                addr_a=0x5_0000,
                n=64,
                imm32=imm,
                vs_dst=512,
                sh0=24,
                sreg_dst=6,
            )
        )
    return out


@cocotb.test()
async def test_pos_derived_sweep_against_the_simulator(dut):
    """Every shape at every position and row set: the decoded extents, the MAC and weight-byte
    totals and the bounds events are the ones quettos.isa_sim counts."""
    env, _ = await setup(dut, wb=16, gemv_beats=2, gemv_drain=1, vpu_cycles=2)
    shapes = pos_shapes()
    checked = 0
    for pos in SWEEP_POSITIONS:
        for row_mask, row_en in SWEEP_ROWS:
            program = [dataclasses.replace(d, row_mask=row_mask) for d in shapes]
            program.append(isa.Descriptor(opcode=Opcode.HALT))
            env.pos, env.row_en = pos, row_en
            env.issues.clear()
            env.macs = env.wt_bytes = env.err_bounds = env.done = env.err = 0
            await run_program(dut, env, program)
            want = issuing(program, pos=pos, row_en=row_en, b_max=env.b_max, wb=env.wb)
            totals = {"err_bounds": 0, "macs": 0, "wt_bytes": 0}
            for d in program:
                for key, v in expect_events(
                    d, pos=pos, row_en=row_en, b_max=env.b_max, wb=env.wb
                ).items():
                    totals[key] += v
            where = f"POS {pos} row_mask {row_mask:#x} ROW_EN {row_en:#x}"
            assert len(env.issues) == len(want), f"{where}: {len(env.issues)} issues"
            for i, (got, d) in enumerate(zip(env.issues, want, strict=True)):
                check_bundle(
                    got,
                    expect(d, pos=pos, tok=env.tok, row_en=row_en, b_max=env.b_max, wb=env.wb),
                    f"{where} descriptor {i} ({d.opcode.name})",
                )
                checked += 1
            assert env.macs == totals["macs"], f"{where}: MACS {env.macs} != {totals['macs']}"
            assert env.wt_bytes == totals["wt_bytes"], (
                f"{where}: WT_BYTES {env.wt_bytes} != {totals['wt_bytes']}"
            )
            assert env.err_bounds == totals["err_bounds"], (
                f"{where}: ERR_BOUNDS {env.err_bounds} != {totals['err_bounds']}"
            )
            assert (env.done, env.err) == (1, 0), f"{where}: the sweep run did not end on HALT"
    assert checked == len(shapes) * len(SWEEP_POSITIONS) * len(SWEEP_ROWS), (
        f"{checked} descriptors checked"
    )
    env.row_en = 1
