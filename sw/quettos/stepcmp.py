"""Whole compiled programs, descriptor by descriptor: ``qcore_top`` against the ISA simulator.

The compiler writes ``dump_plan.json`` beside ``decode.prog`` and
``prefill.prog``: for every descriptor, the VSRAM range it writes, the scale
registers, the memory regions and the CSRs.  The Verilator harness runs a
program one ``CTRL.STEP`` per descriptor and writes exactly that state out
after each one (``--step --dump-ops``); :mod:`quettos.isa_sim` runs the same
program on the same image and captures the same state.  This module runs both
over one compiled image -- every descriptor of ``prefill.prog`` at the prompt's
positions and of ``decode.prog`` at the generated ones -- and reports the first
descriptor whose effects differ, as its index, opcode, listing line and
element.

What is compared after every descriptor: the VSRAM range the plan names, the
scale registers, the memory regions (as bytes, or as their FNV-1a hash when a
region is larger than ``--dump-bytes``), the ARGMAX CSRs, ``PC``, the three
PERF counters the simulator models and the four event counters, all since the
token's first descriptor.  ``isa_sim.check_plan`` runs over the simulator's own
writes first, so the plan is known to name everything each descriptor wrote and
comparing the planned state is comparing all of it.

Where :mod:`quettos.compare` runs four short programs and compares whole VSRAM
banks, this runs the compiler's programs end to end -- every layer, the final
norm and the LM head included -- and compares what each descriptor is supposed
to have touched.  A whole-run match becomes a diagnosis: the report names the
descriptor.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from quettos import compare, compiler, isa, isa_sim, numerics, synthetic
from quettos.compare import CONFIGS, Config
from quettos.isa import Descriptor, Opcode

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "build"

#: The first line of this module's documentation, which the CLI prints as its
#: description; a build that strips docstrings still names the module.
SUMMARY = (
    __doc__.splitlines()[0]
    if __doc__
    else "whole compiled programs, descriptor by descriptor, against the ISA simulator"
)

#: An id no vocabulary holds, so the decode loop runs its full length.
EOS_OFF = 999_999_999

#: Memory regions up to this size travel as bytes, so a difference is reported
#: at the byte it is in; a larger one travels as its FNV-1a hash alone.
DUMP_BYTES = 1 << 16

#: The counters the simulator models, and the four event counters.
PERF_COMPARED = ("DESCRIPTORS", "MACS", "WT_BYTES")
EVENTS = ("SAT_REQ", "SAT_VPU", "ERR_SHIFT", "ERR_BOUNDS")

#: FNV-1a 64, the hash ``sim/verilator/main.cpp`` takes over a dumped region.
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def fnv1a64(data: bytes) -> int:
    """FNV-1a over ``data``, the hash the harness writes for a dumped memory region."""
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK64
    return h


def sreg_word(value: Any) -> int:
    """The 32-bit word an SREG entry holds: an sfloat pair, or a tracked absmax."""
    return isa.sfloat_imm(value) if isinstance(value, numerics.SFloat) else int(value) & 0xFFFFFFFF


# --------------------------------------------------------------------------- one descriptor


@dataclass(frozen=True)
class Region:
    """A dumped memory region: its address, its hash, and its bytes when they were carried."""

    addr: int
    size: int
    digest: int
    data: bytes | None


@dataclass(frozen=True)
class Effects:
    """What one descriptor left behind, in the form both machines report it.

    ``vsram`` is the plan's range as signed int32 starting at ``vsram_start``
    (``None`` when the descriptor writes no VSRAM), ``sreg`` the listed
    registers as 32-bit words, ``mem`` the listed regions by name, ``csr`` the
    listed registers, ``pc`` the address of the next descriptor, and ``perf`` /
    ``events`` the counters since the token's first descriptor.
    """

    index: int
    op: str
    name: str
    pc: int
    vsram_start: int
    vsram: list[int] | None
    sreg: dict[int, int]
    mem: dict[str, Region]
    csr: dict[str, int]
    perf: dict[str, int]
    events: dict[str, int]

    @classmethod
    def from_reference(cls, rec: isa_sim.StepRecord, pc: int, counters: dict[str, int]) -> Effects:
        """The simulator's side: one :class:`quettos.isa_sim.StepRecord` and its counters."""
        vs = rec.entry.get("vsram")
        regions = {r["name"]: r for r in rec.entry.get("mem", [])}
        return cls(
            index=rec.index,
            op=rec.descriptor.opcode.name,
            name=str(rec.entry.get("name")),
            pc=pc,
            vsram_start=0 if vs is None else int(vs["start"]),
            vsram=None if rec.vsram is None else [int(v) for v in rec.vsram],
            sreg={int(i): sreg_word(v) for i, v in rec.sreg.items()},
            mem={
                name: Region(int(regions[name]["addr"]), len(data), fnv1a64(data), data)
                for name, data in rec.mem.items()
            },
            csr=dict(rec.csr),
            perf={k: counters[k] for k in PERF_COMPARED},
            events={k: counters[k] for k in EVENTS},
        )

    @classmethod
    def from_rtl(cls, entry: dict[str, Any]) -> Effects:
        """The RTL's side: one record of a ``--dump-ops`` file."""
        vs = entry.get("vsram")
        vs = vs if isinstance(vs, dict) else None
        return cls(
            index=int(entry["index"]),
            op=str(entry["op"]),
            name=str(entry["name"]),
            pc=int(entry["pc"]),
            vsram_start=0 if vs is None else int(vs["start"]),
            vsram=None if vs is None else [int(v) for v in vs["values"]],
            sreg={int(i): int(v) for i, v in entry["sreg"].items()},
            mem={
                name: Region(
                    int(r["addr"]),
                    int(r["size"]),
                    int(r["fnv1a64"]),
                    bytes.fromhex(r["hex"]) if "hex" in r else None,
                )
                for name, r in entry["mem"].items()
            },
            csr={k: int(v) for k, v in entry["csr"].items()},
            perf={k: int(entry["perf"][k]) for k in PERF_COMPARED},
            events={k: int(entry["events"][k]) for k in EVENTS},
        )


#: ``(what, element, expected, got)`` for one thing two machines disagree on.
Difference = Iterator[tuple[str, "int | None", Any, Any]]


def _first_difference(expected: Sequence[int], got: Sequence[int]) -> int | None:
    if len(expected) != len(got):
        return min(len(expected), len(got))
    for i, (e, g) in enumerate(zip(expected, got, strict=True)):
        if e != g:
            return i
    return None


def differences(exp: Effects, got: Effects) -> Difference:
    """``(what, element, expected, got)`` for everything the two disagree on."""
    named = (("op", exp.op, got.op), ("name", exp.name, got.name), ("PC", exp.pc, got.pc))
    for what, e, g in named:
        if e != g:
            yield what, None, e, g
    if (exp.vsram is None) != (got.vsram is None):
        yield "VSRAM range", None, exp.vsram is not None, got.vsram is not None
    elif exp.vsram is not None and got.vsram is not None:
        i = _first_difference(exp.vsram, got.vsram)
        if i is not None:
            yield (
                f"VSRAM element {exp.vsram_start + i}",
                exp.vsram_start + i,
                exp.vsram[i] if i < len(exp.vsram) else None,
                got.vsram[i] if i < len(got.vsram) else None,
            )
    if set(exp.sreg) != set(got.sreg):
        yield "SREG indices", None, sorted(exp.sreg), sorted(got.sreg)
    else:
        for i in sorted(exp.sreg):
            if exp.sreg[i] != got.sreg[i]:
                yield f"SREG[{i}]", i, hex(exp.sreg[i]), hex(got.sreg[i])
    if set(exp.mem) != set(got.mem):
        yield "memory regions", None, sorted(exp.mem), sorted(got.mem)
    else:
        for name in sorted(exp.mem):
            yield from _region_difference(name, exp.mem[name], got.mem[name])
    for label, e_table, g_table in (
        ("", exp.csr, got.csr),
        ("PERF ", exp.perf, got.perf),
        ("", exp.events, got.events),
    ):
        for key, value in e_table.items():
            if key in g_table and value != g_table[key]:
                yield f"{label}{key}", None, value, g_table[key]


def _region_difference(name: str, exp: Region, got: Region) -> Difference:
    if exp.size != got.size or exp.addr != got.addr:
        yield f"{name} extent", None, (hex(exp.addr), exp.size), (hex(got.addr), got.size)
    elif exp.data is not None and got.data is not None:
        i = _first_difference(exp.data, got.data)
        if i is not None:
            yield f"{name} byte at 0x{exp.addr + i:08x}", i, exp.data[i], got.data[i]
    elif exp.digest != got.digest:
        # The region was too large to carry; --dump-bytes N raises the limit.
        yield f"{name} hash over {exp.size} bytes", None, hex(exp.digest), hex(got.digest)


# --------------------------------------------------------------------------- the report


@dataclass(frozen=True)
class Mismatch:
    """Where the two machines first disagree, and on what.

    ``index`` is the descriptor of the program named by ``program``, with its
    opcode, its dataflow name and its listing line; ``element`` is the first
    differing element of ``what``.  An ``index`` of -1 carries a problem with
    the dump plan itself, found before the two were compared.
    """

    program: str
    pos: int
    index: int
    opcode: str
    name: str
    listing: str
    what: str
    element: int | None
    expected: Any
    got: Any

    def __str__(self) -> str:
        if self.index < 0:
            return f"{self.program} POS {self.pos}: {self.what}: {self.expected}"
        where = f"{self.program} POS {self.pos} descriptor {self.index} {self.opcode} {self.name}"
        at = "" if self.element is None else f" element {self.element}"
        return (
            f"{where}: {self.what}{at}: isa_sim {self.expected}, RTL {self.got}\n"
            f"      {self.listing}"
        )


@dataclass
class Result:
    """One image at one configuration: what both machines produced over the same tokens."""

    image: Path
    config: Config
    prompt: list[int]
    reference_ids: list[int]
    rtl_ids: list[int]
    tokens: int
    descriptors: int
    events: dict[str, int]
    seconds: float
    mismatches: list[Mismatch] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches and self.reference_ids == self.rtl_ids


# --------------------------------------------------------------------------- the simulator


@dataclass
class TokenRun:
    """One token on one machine: the program it ran, its position, and its descriptors."""

    program: str
    pos: int
    tok: int
    effects: list[Effects]
    records: list[isa_sim.StepRecord] = field(default_factory=list)


def _capture(
    m: isa_sim.Machine, index: int, d: Descriptor, entry: dict[str, Any]
) -> isa_sim.StepRecord:
    """The state one descriptor's dump-plan entry names, read out of the simulator.

    The same capture :func:`quettos.isa_sim.record_program` performs; this
    module runs its own so it can read the counters at the same retire, and
    ``sw/tests/test_stepcmp.py`` holds the two side by side.
    """
    if entry.get("op") != d.opcode.name:
        raise ValueError(f"descriptor {index} is {d.opcode.name}, dump plan says {entry.get('op')}")
    row = d.src_row if d.opcode == Opcode.VROPE else d.dst_row
    vs = entry.get("vsram")
    vsram = None
    if vs is not None:
        vsram = m.vsram[row, vs["start"] : vs["start"] + vs["count"]].astype(np.int32)
    sreg = {int(i): m.sreg[row][int(i)] for i in entry.get("sreg", [])}
    mem = {r["name"]: bytes(m.mem[r["addr"] : r["addr"] + r["size"]]) for r in entry.get("mem", [])}
    csr = {name: m.csr[name] for name in entry.get("csr", [])}
    writes = {w.key() for w in (m.log or [])}
    return isa_sim.StepRecord(index, d, entry, vsram, sreg, mem, csr, writes)


def record_token(
    m: isa_sim.Machine,
    source: Sequence[Descriptor],
    plan: Sequence[dict[str, Any]],
    tok: int,
    pos: int,
    *,
    pc: int,
) -> tuple[list[isa_sim.StepRecord], list[Effects]]:
    """Run one token in step mode; return every descriptor's record and its effects."""
    records: list[isa_sim.StepRecord] = []
    effects: list[Effects] = []
    m.log = []

    def retire(index: int, d: Descriptor) -> None:
        if index >= len(plan):
            raise ValueError(f"descriptor {index} has no dump-plan entry")
        rec = _capture(m, index, d, plan[index])
        counters = {name: m.perf_value(name) for name in PERF_COMPARED}
        counters.update({name: int(m.csr[name]) for name in EVENTS})
        records.append(rec)
        effects.append(Effects.from_reference(rec, m.csr["PC"], counters))
        assert m.log is not None
        m.log.clear()

    try:
        isa_sim.run_token(m, source, tok, pos, pc=pc, on_retire=retire)
    finally:
        m.log = None
    return records, effects


def reference(
    image_dir: Path, cfg: Config, prompt: Sequence[int], max_new: int
) -> tuple[list[TokenRun], list[int]]:
    """The prefill/decode loop on :mod:`quettos.isa_sim`, every descriptor captured.

    ``prefill.prog`` runs at every prompt position but the last and
    ``decode.prog`` from there on, each decode step taking the id the step
    before it produced -- the loop ``sim/verilator/main.cpp`` drives.
    """
    layout = compiler.load_layout(image_dir)
    addr = {k: int(v["addr"]) for k, v in layout["programs"].items()}
    m = isa_sim.Machine.from_file(image_dir / layout["image"]["file"], **cfg.widths)
    programs = isa_sim.Programs.from_dir(image_dir)
    source = {"decode": programs.decode, "prefill": programs.prefill}
    plans = {"decode": programs.decode_plan, "prefill": programs.prefill_plan}

    runs: list[TokenRun] = []
    gen: list[int] = []

    def step(which: str, tok: int, pos: int) -> None:
        recs, eff = record_token(m, source[which], plans[which], tok, pos, pc=addr[which])
        runs.append(TokenRun(which, pos, tok, eff, recs))

    for i in range(len(prompt) - 1):
        step("prefill", int(prompt[i]), i)
    tok = int(prompt[-1])
    for j in range(max_new):
        step("decode", tok, len(prompt) - 1 + j)
        tok = int(m.csr["ARGMAX_TOK"])
        gen.append(tok)
    return runs, gen


# --------------------------------------------------------------------------- the RTL


def run_rtl(
    image_dir: Path,
    cfg: Config,
    prompt: Sequence[int],
    max_new: int,
    *,
    lat: int = 32,
    bw_div: int = 1,
    dump_bytes: int = DUMP_BYTES,
    out_dir: Path | None = None,
    quiet: bool = True,
) -> tuple[dict[tuple[str, int], list[Effects]], list[int], dict[str, Any]]:
    """Run the same loop on ``qcore_top`` in step mode and read the per-descriptor dumps back.

    The harness is given ``--allow-sat``: this comparison checks the four event
    counters against the simulator after every descriptor, which is the stronger
    statement, so a run is not failed for having raised one.
    """
    out_dir = BUILD / "stepcmp" if out_dir is None else out_dir
    steps = out_dir / f"steps-{image_dir.name}-w{cfg.wb}-lat{lat}-bw{bw_div}"
    if steps.is_dir():
        for old in steps.glob("*.json"):
            old.unlink()
    steps.mkdir(parents=True, exist_ok=True)
    perf_json = out_dir / f"perf-{image_dir.name}-w{cfg.wb}.json"
    cmd = [
        str(compare.harness_binary(cfg, quiet=quiet)),
        "--image",
        str(image_dir),
        "--step",
        "--dump-ops",
        "--dump-dir",
        str(steps),
        "--dump-bytes",
        str(dump_bytes),
        "--perf-json",
        str(perf_json),
        "--prompt-ids",
        ",".join(str(int(i)) for i in prompt),
        "--max-new",
        str(max_new),
        "--eos",
        str(EOS_OFF),
        "--lat",
        str(lat),
        "--bw-div",
        str(bw_div),
        "--allow-sat",
    ]
    if quiet:
        cmd.append("--quiet")
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"qcore_sim exited {proc.returncode}\n{' '.join(cmd)}\n{proc.stdout}{proc.stderr}"
        )
    dumps: dict[tuple[str, int], list[Effects]] = {}
    for path in steps.glob("*.json"):
        which, _, pos = path.stem.rpartition("_pos")
        dumps[(which, int(pos))] = [Effects.from_rtl(e) for e in json.loads(path.read_text())]
    run = json.loads(perf_json.read_text())
    ids = [int(t["out"]) for t in run["tokens"] if int(t.get("out", -1)) >= 0]
    return dumps, ids, run


# --------------------------------------------------------------------------- the comparison


def compare_token(run: TokenRun, got: list[Effects] | None) -> list[Mismatch]:
    """Every difference in the first descriptor of one token that differs; empty when it matches."""

    def mismatch(index: int, what: str, element: int | None, e: Any, g: Any) -> Mismatch:
        d = run.records[index].descriptor if index < len(run.records) else None
        return Mismatch(
            program=run.program,
            pos=run.pos,
            index=index,
            opcode="-" if d is None else d.opcode.name,
            name="" if index >= len(run.effects) else run.effects[index].name,
            listing="" if d is None else isa.disassemble_one(d),
            what=what,
            element=element,
            expected=e,
            got=g,
        )

    if got is None:
        return [mismatch(0, "the RTL wrote no records for this token", None, len(run.effects), 0)]
    if len(run.effects) != len(got):
        at = min(len(run.effects), len(got))
        return [mismatch(at, "descriptors", None, len(run.effects), len(got))]
    for exp, rtl in zip(run.effects, got, strict=True):
        found = [mismatch(exp.index, what, el, e, g) for what, el, e, g in differences(exp, rtl)]
        if found:
            return found
    return []


def check_image(
    image_dir: Path | str,
    cfg: Config,
    *,
    prompt: Sequence[int],
    max_new: int = 1,
    lat: int = 32,
    bw_div: int = 1,
    dump_bytes: int = DUMP_BYTES,
    out_dir: Path | None = None,
    quiet: bool = True,
) -> Result:
    """Run one compiled image on both machines and compare every descriptor of every token."""
    image_dir = Path(image_dir)
    layout = compiler.load_layout(image_dir)
    if int(layout["wb"]) != cfg.wb:
        raise ValueError(f"{image_dir} was compiled for WB={layout['wb']}, not {cfg.wb}")
    t0 = time.perf_counter()
    runs, gen = reference(image_dir, cfg, prompt, max_new)
    dumps, rtl_ids, run = run_rtl(
        image_dir,
        cfg,
        prompt,
        max_new,
        lat=lat,
        bw_div=bw_div,
        dump_bytes=dump_bytes,
        out_dir=out_dir,
        quiet=quiet,
    )
    mismatches: list[Mismatch] = []
    compared = 0
    for tr in runs:
        for problem in isa_sim.check_plan(tr.records):
            mismatches.append(
                Mismatch(
                    tr.program,
                    tr.pos,
                    -1,
                    "",
                    "",
                    "",
                    "the dump plan misses a write",
                    None,
                    problem,
                    "",
                )
            )
        found = compare_token(tr, dumps.get((tr.program, tr.pos)))
        mismatches += found
        # A run stops at the descriptor it first differs on: the ones before it
        # are compared, the ones after are downstream of that difference.
        if found:
            compared += max(found[0].index, 0)
        elif not mismatches:
            compared += len(tr.effects)
        if mismatches:
            break
    return Result(
        image=image_dir,
        config=cfg,
        prompt=[int(t) for t in prompt],
        reference_ids=gen,
        rtl_ids=rtl_ids,
        tokens=len(runs),
        descriptors=compared,
        events={k: int(run["events"][k]) for k in EVENTS},
        seconds=time.perf_counter() - t0,
        mismatches=mismatches,
    )


# --------------------------------------------------------------------------- the images


def distinct_shapes(count: int, *, seed: int = 0) -> list[synthetic.Shape]:
    """``count`` random tiny shapes, no two sharing a hidden size, vocabulary and head count."""
    rng = np.random.default_rng(seed)
    picked: list[synthetic.Shape] = []
    seen: set[tuple[int, int, int, int]] = set()
    while len(picked) < count:
        shape = synthetic.random_shape(rng)
        key = (shape.hidden, shape.vocab, shape.heads, shape.kv_heads)
        if key in seen:
            continue
        seen.add(key)
        picked.append(shape)
    return picked


#: ``--layers`` value that compiles every decoder layer the checkpoint has.
ALL_LAYERS = "all"


def layer_counts(text: str) -> list[int | None]:
    """Parse a ``--layers`` list: numbers, and :data:`ALL_LAYERS` for the complete model."""
    return [None if v == ALL_LAYERS else int(v) for v in text.split(",") if v]


def model_image(
    alias: str, layers: int | None, cfg: Config, out_dir: Path, *, max_ctx: int, a_bits: int = 16
) -> Path:
    """Compile a downloaded model for ``cfg``: its first ``layers`` layers, or all of them.

    The quantized checkpoint is ``build/quant/<name>.npz``
    (``uv run quettos quantize <alias>``); the compile is the ordinary one, with
    ``--layers`` when ``layers`` is a number and a context short enough that the
    KV regions travel as bytes.  ``layers=None`` compiles the complete model,
    every decoder layer, the final norm and the LM head.  The checkpoint is
    looked for before the spec is read, so a tree that has not built one says so
    instead of downloading a model.
    """
    from quettos import model as model_io
    from quettos import quantize

    name = model_io.sanitize_name(model_io.resolve_repo_id(alias))
    path = quantize.default_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run: uv run quettos quantize {alias}")
    spec = model_io.load_spec(alias)
    model = quantize.load(path)
    if layers is not None:
        model = compiler.truncate_layers(model, layers)
    compiled = compiler.compile(
        model,
        spec,
        out_dir=out_dir,
        max_ctx=max_ctx,
        wb=cfg.wb,
        a_bits=a_bits,
        vsram_words=cfg.vsram_words,
    )
    return Path(compiled.out_dir)


def token_ids(count: int, vocab: int, *, seed: int) -> list[int]:
    """``count`` distinct-looking prompt ids drawn from ``[0, vocab)``."""
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(0, vocab, count)]


# --------------------------------------------------------------------------- runs


def sweep(
    shapes: int,
    cfg: Config,
    *,
    seed: int = 0,
    prompt_len: int = 3,
    max_new: int = 1,
    max_ctx: int | None = None,
    lat: int = 32,
    bw_div: int = 1,
    dump_bytes: int = DUMP_BYTES,
    out_dir: Path | None = None,
    quiet: bool = True,
) -> list[Result]:
    """Every descriptor of both programs of ``shapes`` random tiny models, on both machines."""
    out_dir = BUILD / "stepcmp" if out_dir is None else out_dir
    ctx = 2 * cfg.wb if max_ctx is None else max_ctx
    results: list[Result] = []
    for i, shape in enumerate(distinct_shapes(shapes, seed=seed)):
        where = out_dir / f"shape{i}-w{cfg.wb}"
        image = compare.compile_shape(shape, seed + i, cfg, where, max_ctx=ctx)
        vocab = int(compiler.load_layout(image)["model"]["vocab"])
        results.append(
            check_image(
                image,
                cfg,
                prompt=token_ids(prompt_len, vocab, seed=seed + i),
                max_new=max_new,
                lat=lat,
                bw_div=bw_div,
                dump_bytes=dump_bytes,
                out_dir=out_dir,
                quiet=quiet,
            )
        )
    return results


def models(
    aliases: Sequence[str],
    layers: Sequence[int | None],
    cfg: Config,
    *,
    seed: int = 0,
    prompt_len: int = 3,
    max_new: int = 1,
    max_ctx: int = 128,
    lat: int = 32,
    bw_div: int = 1,
    dump_bytes: int = DUMP_BYTES,
    out_dir: Path | None = None,
    quiet: bool = True,
) -> list[Result]:
    """The same over real models: the first ``layers`` layers of each alias, or all of them.

    A ``None`` in ``layers`` is the complete model -- every decoder layer, the
    final norm and the LM head of the checkpoint, compiled at ``max_ctx`` and
    walked descriptor by descriptor on both machines.
    """
    out_dir = BUILD / "stepcmp" if out_dir is None else out_dir
    results: list[Result] = []
    for alias in aliases:
        for n in layers:
            where = out_dir / f"{alias}-l{ALL_LAYERS if n is None else n}-w{cfg.wb}"
            image = model_image(alias, n, cfg, where, max_ctx=max_ctx)
            vocab = int(compiler.load_layout(image)["model"]["vocab"])
            results.append(
                check_image(
                    image,
                    cfg,
                    prompt=token_ids(prompt_len, vocab, seed=seed),
                    max_new=max_new,
                    lat=lat,
                    bw_div=bw_div,
                    dump_bytes=dump_bytes,
                    out_dir=out_dir,
                    quiet=quiet,
                )
            )
    return results


# --------------------------------------------------------------------------- CLI


def report(results: list[Result]) -> int:
    """Print one line per image and the first differing descriptor of each; 1 on any mismatch."""
    head = f"{'model':<34} {'config':<32} {'tokens':>6} {'descriptors':>12} {'seconds':>8}"
    print(f"{head}  result")
    bad = 0
    total = 0
    for r in results:
        total += r.descriptors
        bad += 0 if r.ok else 1
        print(
            f"{r.image.name:<34} {str(r.config):<32} {r.tokens:>6} {r.descriptors:>12} "
            f"{r.seconds:>8.1f}  {'ok' if r.ok else 'MISMATCH'}"
        )
        for m in r.mismatches[:8]:
            print(f"    {m}")
        if r.reference_ids != r.rtl_ids:
            print(f"    generated ids: isa_sim {r.reference_ids}, RTL {r.rtl_ids}")
    events = {k: sum(r.events[k] for r in results) for k in EVENTS}
    print("\nevents: " + " ".join(f"{k}={v}" for k, v in events.items()))
    print(
        f"{len(results) - bad}/{len(results)} programs match sw/quettos/isa_sim.py, "
        f"{total} descriptors compared"
    )
    return 1 if bad else 0


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The options of ``python -m quettos.stepcmp``."""
    p.add_argument("--image", type=Path, help="a compiled model directory to compare on")
    p.add_argument("--wb", type=int, default=16, help="weight-port width of the RTL build")
    p.add_argument("--sweep", action="store_true", help="random tiny shapes instead of --image")
    p.add_argument("--shapes", type=int, default=5, help="how many random shapes to draw")
    p.add_argument("--models", default="", help="comma-separated aliases: qwen, smollm2")
    p.add_argument(
        "--layers",
        default="1,2",
        help=f"comma-separated layer counts for --models; {ALL_LAYERS!r} is the complete model",
    )
    p.add_argument("--prompt-len", type=int, default=3, help="prompt tokens: prefill steps + 1")
    p.add_argument("--max-new", type=int, default=1, help="decode steps after the prompt")
    p.add_argument("--max-ctx", type=int, default=None, help="KV positions the image holds")
    p.add_argument("--lat", type=int, default=32, help="QMEM read latency in cycles")
    p.add_argument("--bw-div", type=int, default=1, help="one returned beat every N cycles")
    p.add_argument("--seed", type=int, default=0, help="seed for the shapes and the prompt ids")
    p.add_argument(
        "--dump-bytes",
        type=int,
        default=DUMP_BYTES,
        help="memory regions up to this size travel as bytes, larger ones as a hash",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="where images and dumps are written")
    p.add_argument("--verbose", action="store_true", help="let the harness print its own output")
    return p


def run(a: argparse.Namespace) -> int:
    """Run the comparison the parsed options describe and print the table."""
    cfg = CONFIGS[a.wb]
    quiet = not a.verbose
    if a.models:
        results = models(
            [s for s in a.models.split(",") if s],
            layer_counts(a.layers),
            cfg,
            seed=a.seed,
            prompt_len=a.prompt_len,
            max_new=a.max_new,
            max_ctx=128 if a.max_ctx is None else a.max_ctx,
            lat=a.lat,
            bw_div=a.bw_div,
            dump_bytes=a.dump_bytes,
            out_dir=a.out_dir,
            quiet=quiet,
        )
    elif a.sweep:
        results = sweep(
            a.shapes,
            cfg,
            seed=a.seed,
            prompt_len=a.prompt_len,
            max_new=a.max_new,
            max_ctx=a.max_ctx,
            lat=a.lat,
            bw_div=a.bw_div,
            dump_bytes=a.dump_bytes,
            out_dir=a.out_dir,
            quiet=quiet,
        )
    elif a.image is not None:
        vocab = int(compiler.load_layout(a.image)["model"]["vocab"])
        results = [
            check_image(
                a.image,
                cfg,
                prompt=token_ids(a.prompt_len, vocab, seed=a.seed),
                max_new=a.max_new,
                lat=a.lat,
                bw_div=a.bw_div,
                dump_bytes=a.dump_bytes,
                out_dir=a.out_dir,
                quiet=quiet,
            )
        ]
    else:
        raise SystemExit("give --image, --sweep or --models")
    return report(results)


def main(argv: list[str] | None = None) -> int:
    p = add_arguments(argparse.ArgumentParser(prog="quettos.stepcmp", description=SUMMARY))
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
