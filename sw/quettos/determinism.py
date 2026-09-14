"""The same program run every way the machine allows: the values do not move.

``qcore_top`` is a synchronous design behind a memory model, so the only thing
that reaches a value it produces is the descriptor stream.  This module runs one
compiled program on the RTL under each setting that changes nothing else -- the
QMEM read latency, the rate the port returns beats at, the number of simulation
threads, the weight-port width, and Verilator's initial value for a variable no
reset reaches -- and compares what came back.  Cycle counts move with the
setting; generated ids, VSRAM elements, scale registers, KV bytes, CSRs and the
counters do not.

The comparison is :func:`quettos.compare.compare`, so a difference is reported
the same way it is there: the position, the descriptor, the field and the first
element it is in.  The reference is the RTL itself at the configuration every
measured run is taken at, ``--lat 32 --bw-div 1 --threads 1``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from quettos import compare, compiler, synthetic
from quettos.compare import Case, Config, Record

REPO = compare.REPO
HARNESS = compare.HARNESS
BUILD = compare.BUILD

#: The first line of this module's documentation, which the CLI prints as its
#: description; a build that strips docstrings still names the module.
SUMMARY = __doc__.splitlines()[0] if __doc__ else "one program run every way the machine allows"

#: Where compiled images and per-run records land.
OUT_DIR = BUILD / "determinism"

#: The program the state comparison runs: the whole decoder layer of the
#: compiled ``decode.prog``, at the six positions :func:`compare.positions`
#: names, which is the longest descriptor sequence the comparison has.
PROGRAM = "layer"


@dataclass(frozen=True)
class Setting:
    """One way of running the RTL: everything about a run except the program.

    ``x_initial`` is the Verilator build's initial value for a variable no reset
    reaches: ``fast`` is zeros, ``unique`` takes the value
    ``+verilator+rand+reset+<rand_reset>`` names -- 0 zeros, 1 ones, 2 a value
    drawn from ``+verilator+seed+<seed>``.
    """

    name: str
    lat: int = 32
    bw_div: int = 1
    threads: int = 1
    x_initial: str = "fast"
    rand_reset: int = 0
    seed: int = 1

    @property
    def make_args(self) -> list[str]:
        """What the harness Makefile is built with; the object directory hashes both."""
        return [f"THREADS={self.threads}", f"XINIT={self.x_initial}"]

    @property
    def plusargs(self) -> list[str]:
        """The runtime arguments the simulation kernel takes, empty for a ``fast`` build."""
        if self.x_initial == "fast":
            return []
        return [f"+verilator+rand+reset+{self.rand_reset}", f"+verilator+seed+{self.seed}"]

    def __str__(self) -> str:
        return self.name


#: The reference every other setting is compared against: the configuration
#: every measured run is taken at (``docs/PERFORMANCE.md``).
BASELINE = Setting("lat 32")


#: Timing: when the core sees its weights, and how Verilator partitions an
#: ``eval``. `--lat 200` is past the memory model's 64-beat in-flight window, so
#: the port is bandwidth-bound rather than saturated, and `--bw-div 2` halves the
#: bandwidth at the default latency.
def max_threads() -> int:
    """How many simulation threads this machine can actually run.

    Verilator refuses to run a model built for more threads than its runtime
    context has, so a build for four threads cannot execute on a two-core
    runner.  The property under test is that threading does not move a value,
    and two threads test it as well as four do.
    """
    return min(4, os.cpu_count() or 1)


def timing_settings(threads: int | None = None) -> tuple[Setting, ...]:
    """The timing sweep, with the threaded run sized to this machine.

    A machine that can only run one thread has nothing to compare a threaded
    run against, so the threaded setting is left out rather than run at one
    thread and reported as though it had been exercised.
    """
    n = max_threads() if threads is None else threads
    sweep = [
        Setting("lat 1", lat=1),
        Setting("lat 200", lat=200),
        Setting("bw-div 2", bw_div=2),
    ]
    if n > 1:
        sweep.append(Setting(f"threads {n}", threads=n))
    return tuple(sweep)


#: The timing sweep on this machine.  ``timing_settings`` sizes the threaded
#: run to the cores available, so the same sweep runs on a laptop and on a
#: two-core CI runner.
TIMING: tuple[Setting, ...] = timing_settings()

#: The record fields that name storage the RTL holds rather than a value the
#: program produced.  A run from an undefined-value start finds whatever the
#: start put in the elements a program has not written, so it is compared on
#: the ids, the memory the program writes and the registers the host reads.
BANK_FIELDS = ("VSRAM", "SREG")

#: The seeds an undefined-value start is drawn with by default.
X_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)


def x_initial(seeds: Sequence[int] = X_SEEDS) -> tuple[Setting, ...]:
    """Undefined-value starts: zeros, ones, and one random start per seed."""
    return (
        Setting("x-init zeros", x_initial="unique", rand_reset=0),
        Setting("x-init ones", x_initial="unique", rand_reset=1),
        *(Setting(f"x-init seed {s}", x_initial="unique", rand_reset=2, seed=s) for s in seeds),
    )


# --------------------------------------------------------------------------- running the RTL


@dataclass
class Run:
    """What one setting produced, or the line the simulator stopped on."""

    setting: Setting
    cycles: int = 0
    records: list[Record] = field(default_factory=list)
    mem: list[list[list[int]]] = field(default_factory=list)
    ids: list[int] = field(default_factory=list)
    stopped: str = ""

    @property
    def ok(self) -> bool:
        return not self.stopped


def harness_binary(cfg: Config, setting: Setting, *, quiet: bool = True) -> Path:
    """Build the harness for this configuration and setting if it is not built, and name it."""
    args = [*cfg.make_args, *setting.make_args]
    out = subprocess.DEVNULL if quiet else None
    subprocess.run(["make", "-C", str(HARNESS), "build", *args], cwd=REPO, check=True, stdout=out)
    where = subprocess.run(
        ["make", "-s", "--no-print-directory", "-C", str(HARNESS), "where", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(where.stdout.strip())


def _tag(setting: Setting) -> str:
    """A file-name-safe form of a setting's name."""
    return re.sub(r"[^A-Za-z0-9]+", "-", setting.name).strip("-")


def _stopped(proc: subprocess.CompletedProcess[str]) -> str:
    """The first line the simulator failed on, with the repository path taken off it."""
    for line in (proc.stdout + proc.stderr).splitlines():
        if "%Error" in line or line.startswith("FAIL:") or line.startswith("stopped:"):
            return line.replace(f"{REPO}/", "").strip()
    return f"qcore_sim exited {proc.returncode}"


def _records(blob: dict[str, Any]) -> list[Record]:
    """The per-descriptor records of a ``--bringup-json`` file, as the comparison reads them."""
    return [
        Record(
            pass_index=int(r["pass"]),
            pos=int(r["pos"]),
            index=int(r["index"]),
            pc=int(r["pc"]),
            status=int(r["status"]),
            argmax_tok=int(r["argmax_tok"]),
            argmax_val=int(r["argmax_val"]),
            events={name: int(r["events"][name]) for name in compare.EVENTS},
            perf={name: int(r["perf"][name]) for name in compare.PERF_COMPARED},
            vsram=[[int(v) for v in vals] for vals in r["vsram"]],
            sreg=[[int(v) for v in bank] for bank in r["sreg"]],
        )
        for r in blob["records"]
    ]


def run_case(
    image_dir: Path | str,
    c: Case,
    cfg: Config,
    setting: Setting,
    *,
    name: str = PROGRAM,
    out_dir: Path | None = None,
    quiet: bool = True,
) -> Run:
    """One case on ``qcore_top`` under one setting, one ``CTRL.STEP`` per descriptor.

    The same invocation :func:`quettos.compare.run_rtl` makes, plus the thread
    count the build was made with and the plusargs an undefined-value start
    needs.  A run the simulator stopped comes back with the line it stopped on
    instead of records.
    """
    out_dir = OUT_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{name}-w{cfg.wb}-{_tag(setting)}"
    prog_path = out_dir / f"{tag}.prog"
    json_path = out_dir / f"{tag}.json"
    prog_path.write_bytes(c.blob)
    json_path.unlink(missing_ok=True)

    cmd = [
        str(harness_binary(cfg, setting, quiet=quiet)),
        "--image",
        str(image_dir),
        "--program",
        str(prog_path),
        "--program-addr",
        str(c.addr),
        "--bringup-json",
        str(json_path),
        "--threads",
        str(setting.threads),
        "--lat",
        str(setting.lat),
        "--bw-div",
        str(setting.bw_div),
        "--step",
    ]
    for tok, pos in c.passes:
        cmd += ["--at", f"{tok}:{pos}"]
    for bank, index, word in c.sreg:
        cmd += ["--sreg", f"{bank}:{index}={word:#010x}"]
    for bank, start, count in c.ranges:
        cmd += ["--dump-vsram", f"{bank}:{start}:{count}"]
    for addr, size in c.mem:
        cmd += ["--dump-mem", f"{addr}:{size}"]
    if name in compare.EVENTFUL:
        cmd.append("--allow-sat")
    if quiet:
        cmd.append("--quiet")
    cmd += setting.plusargs

    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0 or not json_path.is_file():
        return Run(setting, stopped=_stopped(proc))
    blob = json.loads(json_path.read_text())
    mem: list[list[list[int]]] = [[] for _ in c.passes]
    for entry in blob["mem"]:
        mem[int(entry["pass"])].append([int(v) for v in entry["values"]])
    return Run(
        setting=setting,
        cycles=int(blob["run"]["clock_cycles"]),
        records=_records(blob),
        mem=mem,
    )


def run_generate(
    image_dir: Path | str,
    cfg: Config,
    setting: Setting,
    *,
    max_new: int,
    prompt: Sequence[int] | None = None,
    out_dir: Path | None = None,
    quiet: bool = True,
) -> Run:
    """The prefill/decode loop of a compiled image on ``qcore_top``: the ids and the cycles.

    End-of-sequence is disabled, so the loop runs its full length whatever the
    model generates and two runs are the same shape.
    """
    image_dir = Path(image_dir)
    out_dir = OUT_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = list(compare.prompt_ids(image_dir) if prompt is None else prompt)
    perf_path = out_dir / f"generate-{image_dir.name}-w{cfg.wb}-{_tag(setting)}.json"
    perf_path.unlink(missing_ok=True)

    cmd = [
        str(harness_binary(cfg, setting, quiet=quiet)),
        "--image",
        str(image_dir),
        "--perf-json",
        str(perf_path),
        "--max-new",
        str(max_new),
        "--prompt-ids",
        ",".join(str(i) for i in ids),
        "--eos",
        str(compare.EOS_OFF),
        "--threads",
        str(setting.threads),
        "--lat",
        str(setting.lat),
        "--bw-div",
        str(setting.bw_div),
    ]
    if quiet:
        cmd.append("--quiet")
    cmd += setting.plusargs

    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0 or not perf_path.is_file():
        return Run(setting, stopped=_stopped(proc))
    blob = json.loads(perf_path.read_text())
    return Run(
        setting=setting,
        cycles=int(blob["run"]["clock_cycles"]),
        ids=[int(t["out"]) for t in blob["tokens"] if t.get("out", -1) >= 0],
    )


# --------------------------------------------------------------------------- the comparison


@dataclass(frozen=True)
class Difference:
    """One value two runs of the same program disagree on, and where it is."""

    setting: str
    pos: int
    index: int
    opcode: str
    what: str
    element: int | None
    baseline: Any
    got: Any

    def __str__(self) -> str:
        at = f"POS {self.pos} descriptor {self.index} ({self.opcode})"
        where = f"{self.setting}: {at} {self.what}"
        if self.element is not None:
            where += f"[{self.element}]"
        return f"{where}: {self.baseline} at the baseline, {self.got} here"


def first_difference(base: Sequence[int], got: Sequence[int]) -> int | None:
    """The first index the two id sequences disagree at, or where one of them ends."""
    for i, (a, b) in enumerate(zip(base, got, strict=False)):
        if a != b:
            return i
    return None if len(base) == len(got) else min(len(base), len(got))


def differences(
    base: Run, got: Run, c: Case | None = None, *, label: str | None = None
) -> list[Difference]:
    """Every value the two runs disagree on: the ids, then the per-descriptor state."""
    name = got.setting.name if label is None else label
    out: list[Difference] = []
    i = first_difference(base.ids, got.ids)
    if i is not None:
        out.append(
            Difference(
                name,
                -1,
                -1,
                "-",
                "generated id",
                i,
                base.ids[i] if i < len(base.ids) else None,
                got.ids[i] if i < len(got.ids) else None,
            )
        )
    if c is not None:
        out += [
            Difference(name, m.pos, m.record, m.opcode, m.what, m.element, m.expected, m.got)
            for m in compare.compare(base.records, got.records, c, base.mem, got.mem)
        ]
    return out


def _moved(run: Run) -> Run:
    """A copy of a run with every value the comparison reads moved by one.

    The record order -- the pass, the position and the descriptor index -- is
    left alone, since a comparison that disagrees on it reports that and stops
    reading the record's fields.
    """
    return dataclasses.replace(
        run,
        ids=[i + 1 for i in run.ids],
        records=[
            dataclasses.replace(
                r,
                pc=r.pc + 1,
                status=r.status + 1,
                argmax_tok=r.argmax_tok + 1,
                argmax_val=r.argmax_val + 1,
                events={k: v + 1 for k, v in r.events.items()},
                perf={k: v + 1 for k, v in r.perf.items()},
                vsram=[[v + 1 for v in vals] for vals in r.vsram],
                sreg=[[v + 1 for v in bank] for bank in r.sreg],
            )
            for r in run.records
        ],
        mem=[[[v + 1 for v in vals] for vals in ranges] for ranges in run.mem],
    )


def kept_fields(base_case: Run, base_gen: Run, c: Case) -> frozenset[str]:
    """What a comparison with :data:`BANK_FIELDS` taken out of it still judges a run on.

    Measured rather than assumed, on the baseline's own records: every value the
    per-descriptor comparison reads is moved in a copy of them, and the fields
    that survive the filter are the ones an undefined start is held to.  The
    generated ids are compared by the generation run instead, where no filter
    reaches them, so they are deliberately not counted here: a set that always
    held one entry could never report the no-op this exists to catch.  An empty
    set is a comparison that has become a no-op, which :func:`check` refuses --
    the counterpart of the width check's own guard, where a pair of widths that
    share no position of a program raises rather than comparing nothing
    (:func:`width_cases`).
    """
    # Only the stepped state run is measured. The generation run compares ids
    # alone, so folding it in would keep the set non-empty under any filter and
    # hide the very no-op this exists to catch.
    diffs = differences(base_case, _moved(base_case), c)
    return frozenset(d.what for d in diffs if not d.what.startswith(BANK_FIELDS))


@dataclass
class Row:
    """One setting's outcome: what it cost, what it produced, and how it compared."""

    setting: str
    cycles: int
    ids: list[int] = field(default_factory=list)
    stopped: str = ""
    differences: list[Difference] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.stopped and not self.differences

    @property
    def result(self) -> str:
        if self.stopped:
            return "STOPPED"
        return "match" if not self.differences else "DIFFERS"


@dataclass
class Report:
    """One property, checked over a set of settings against one baseline."""

    name: str
    subject: str
    baseline: Row
    rows: list[Row] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.baseline.ok and all(r.ok for r in self.rows)

    @property
    def all_rows(self) -> list[Row]:
        return [self.baseline, *self.rows]


# --------------------------------------------------------------------------- the checks


def _case(image_dir: Path, cfg: Config, program: str, tok: int) -> Case:
    return compare.PROGRAMS[program](image_dir, cfg, tok)


def _row(
    base_case: Run, base_gen: Run, case_run: Run, gen_run: Run, c: Case, *, banks: bool = True
) -> Row:
    """One setting's row: the state run and the generation run, both against the baseline."""
    stopped = case_run.stopped or gen_run.stopped
    if stopped:
        return Row(str(case_run.setting), case_run.cycles or gen_run.cycles, stopped=stopped)
    diffs = differences(base_case, case_run, c) + differences(base_gen, gen_run)
    if not banks:
        diffs = [d for d in diffs if not d.what.startswith(BANK_FIELDS)]
    return Row(str(case_run.setting), case_run.cycles, list(gen_run.ids), differences=diffs)


def check(
    image_dir: Path | str,
    cfg: Config,
    settings: Sequence[Setting],
    *,
    name: str,
    program: str = PROGRAM,
    tok: int = 0,
    max_new: int = 2,
    prompt: Sequence[int] = (1, 2, 3),
    banks: bool = True,
    out_dir: Path | None = None,
    quiet: bool = True,
) -> Report:
    """One image, one configuration: the baseline run, then every setting against it.

    Each setting runs the program twice -- once descriptor by descriptor with
    every dump recorded, once as the image's prefill/decode loop -- so a row
    covers the dumped state and the ids it generates.  ``banks`` takes the
    vector SRAM and the scale registers out of the comparison, which is what a
    run from an undefined start is judged on (:data:`BANK_FIELDS`).
    """
    image_dir = Path(image_dir)
    c = _case(image_dir, cfg, program, tok)
    base_case = run_case(image_dir, c, cfg, BASELINE, name=program, out_dir=out_dir, quiet=quiet)
    base_gen = run_generate(
        image_dir, cfg, BASELINE, max_new=max_new, prompt=prompt, out_dir=out_dir, quiet=quiet
    )
    baseline = Row(
        str(BASELINE),
        base_case.cycles,
        list(base_gen.ids),
        stopped=base_case.stopped or base_gen.stopped,
    )
    if not banks and not baseline.stopped:
        kept = kept_fields(base_case, base_gen, c)
        if not kept:
            raise ValueError(
                f"{' and '.join(BANK_FIELDS)} cover every field {program!r} is compared on, "
                f"so {name!r} would hold a run to nothing"
            )
    rows: list[Row] = []
    for s in settings:
        if baseline.stopped:
            break
        run = run_case(image_dir, c, cfg, s, name=program, out_dir=out_dir, quiet=quiet)
        gen = run_generate(
            image_dir, cfg, s, max_new=max_new, prompt=prompt, out_dir=out_dir, quiet=quiet
        )
        rows.append(_row(base_case, base_gen, run, gen, c, banks=banks))
    return Report(name, f"{image_dir.name} {cfg} {program}", baseline, rows)


def check_timing(
    image_dir: Path | str, cfg: Config, *, settings: Sequence[Setting] | None = None, **kw: Any
) -> Report:
    """Latency 1 / 32 / 200, half bandwidth, and one against several simulation threads.

    The threaded run is sized to the machine by :func:`timing_settings`, and is
    left out where only one thread is available.
    """
    return check(
        image_dir, cfg, timing_settings() if settings is None else settings, name="timing", **kw
    )


def check_x_initial(
    image_dir: Path | str, cfg: Config, *, seeds: Sequence[int] = X_SEEDS, **kw: Any
) -> Report:
    """The same program from an undefined start: zeros, ones, and a random value per seed."""
    kw.setdefault("banks", False)
    return check(image_dir, cfg, x_initial(seeds), name="x-initial", **kw)


#: What belongs to the port width rather than to the arithmetic, so two widths
#: are not held to it: a partial last tile is zero-padded to the width and
#: streamed, so the padding is weight bytes the port carried and products the
#: array took.  Everything else -- the VSRAM elements, the scale registers,
#: ``PC``, ``STATUS``, the ARGMAX registers, the event counters and
#: ``DESCRIPTORS`` -- is held to the narrower width's value.
WIDTH_EXEMPT = frozenset({"PERF MACS", "PERF WT_BYTES"})


def width_cases(images: dict[int, Path], tok: int, *, program: str = "bringup") -> dict[int, Case]:
    """``program`` at each width, cut to the passes and the dumps the widths share.

    A width is its own compile, so three things about a case belong to the
    compile rather than to the arithmetic and are made common before the runs:

    - the **VSRAM window**, trimmed to the shortest of the compiles, whose used
      range grows with the context a width rounds up to;
    - the **passes**, cut to the positions every width names -- the positions of
      :func:`quettos.compare.positions` are the tile boundaries of the width
      that produced them, and the token of a kept position is the one the first
      width's pass sequence gives it, so both widths embed the same tokens in
      the same order;
    - the **memory regions**, kept only where every width puts the same bytes at
      the same address.  The dumped logits of the bring-up program are one int32
      per vocabulary entry at every width and stay; the KV cache of a layer
      program does not -- both halves are stored in the weight tiling of the
      port width (``compiler.kv_sizes``), so its address, its size and the order
      of its bytes are the width's own.  What the cache holds is compared
      through the arithmetic that reads it: the passes run in order on one
      machine, so the attention output of every position after the first is the
      cache the positions before it wrote.
    """
    cases = {wb: compare.PROGRAMS[program](images[wb], compare.CONFIGS[wb], tok) for wb in images}
    count = min(r[2] for c in cases.values() for r in c.ranges)
    shared_pos = set.intersection(*({p for _, p in c.passes} for c in cases.values()))
    if not shared_pos:
        raise ValueError(f"the widths {sorted(images)} share no position of {program!r}")
    first = cases[min(images)]
    passes = tuple((t, p) for t, p in first.passes if p in shared_pos)
    mem = tuple(sorted(set.intersection(*(set(c.mem) for c in cases.values()))))
    return {
        wb: dataclasses.replace(
            c,
            passes=passes,
            ranges=tuple((b, s, count) for b, s, _ in c.ranges),
            mem=mem,
        )
        for wb, c in cases.items()
    }


def check_width(
    images: dict[int, Path],
    *,
    program: str = "bringup",
    tok: int = 0,
    max_new: int = 4,
    prompt: Sequence[int] = (1, 2, 3),
    out_dir: Path | None = None,
    quiet: bool = True,
) -> Report:
    """One model compiled at each weight-port width: the values are the same, the cycles are not.

    A width is a separate compile of the same weights, so the two images have
    different layouts and different tilings and the same arithmetic.  Each width
    runs the image's own prefill/decode loop, for the ids and the cycles, and
    ``program`` descriptor by descriptor, for the VSRAM elements, the scale
    registers, the CSRs, the event counters and the memory regions of
    :func:`width_cases`.  ``bringup`` takes the embedding table and the tied
    head; ``layer`` takes a whole decoder layer of the compiled ``decode.prog``,
    at every position the two widths share, which is where the rotation, the
    softmax, the KV cache and the partial last tile of a real model are.
    """
    widths = sorted(images)
    cases = width_cases(images, tok, program=program)
    runs: dict[int, tuple[Run, Run]] = {}
    for wb in widths:
        cfg = compare.CONFIGS[wb]
        gen = run_generate(
            images[wb], cfg, BASELINE, max_new=max_new, prompt=prompt, out_dir=out_dir, quiet=quiet
        )
        state = run_case(
            images[wb], cases[wb], cfg, BASELINE, name=program, out_dir=out_dir, quiet=quiet
        )
        runs[wb] = (gen, state)

    base_wb = widths[0]
    base_gen, base_state = runs[base_wb]
    # The cycles column is the program the row names, as it is in :func:`check`,
    # and the ids column the generation run beside it.
    baseline = Row(
        f"WB {base_wb}",
        base_state.cycles or base_gen.cycles,
        list(base_gen.ids),
        stopped=base_gen.stopped or base_state.stopped,
    )
    rows: list[Row] = []
    for wb in widths[1:]:
        gen, state = runs[wb]
        row = Row(
            f"WB {wb}",
            state.cycles or gen.cycles,
            list(gen.ids),
            stopped=gen.stopped or state.stopped,
        )
        if not row.stopped and not baseline.stopped:
            label = f"WB {wb}"
            row.differences = differences(base_gen, gen, label=label) + [
                d
                for d in differences(base_state, state, cases[base_wb], label=label)
                if d.what not in WIDTH_EXEMPT
            ]
        rows.append(row)
    subject = " / ".join(images[wb].name for wb in widths) + f" {program}"
    return Report("width", subject, baseline, rows)


# --------------------------------------------------------------------------- images


def tiny_images(
    out_dir: Path | None = None, *, seed: int = 0, widths: Sequence[int] = (64, 128)
) -> dict[int, Path]:
    """One random tiny model, compiled at each width from the same weights.

    The synthetic model is a function of its shape and its seed, so the images
    hold the same integers and the same descriptors; only the tiling differs.
    Every width is compiled to the same context -- two weight-port tiles of the
    widest one, which is a whole number of tiles at each -- so the VSRAM map is
    the same map at every width and an element of it means the same thing.
    """
    out_dir = OUT_DIR if out_dir is None else Path(out_dir)
    shape = synthetic.random_shape(np.random.default_rng(seed))
    max_ctx = 2 * max(widths)
    return {
        wb: compare.compile_shape(
            shape,
            seed,
            compare.CONFIGS[wb],
            out_dir / f"tiny-s{seed}-w{wb}",
            max_ctx=max_ctx,
        )
        for wb in widths
    }


# --------------------------------------------------------------------------- CLI


def report(reports: Sequence[Report]) -> int:
    """Print every row and return 1 when any of them stopped or differed."""
    bad = 0
    for r in reports:
        print(f"\n{r.name}: {r.subject}")
        print(f"  {'setting':<20} {'cycles':>12}  {'ids':<28} result")
        for row in r.all_rows:
            ids = ",".join(str(i) for i in row.ids) if row.ids else "-"
            print(f"  {row.setting:<20} {row.cycles:>12}  {ids:<28} {row.result}")
            if row.stopped:
                print(f"      stopped: {row.stopped}")
            for d in row.differences[:8]:
                print(f"      {d}")
        bad += 0 if r.ok else 1
    total = sum(len(r.all_rows) for r in reports)
    matched = sum(1 for r in reports for row in r.all_rows if row.ok)
    print(f"\n{matched}/{total} runs produce the baseline's values")
    return 1 if bad else 0


CHECKS = ("timing", "width", "x-initial")


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The options of ``python -m quettos.determinism``."""
    p.add_argument(
        "--checks", default=",".join(CHECKS), help="comma-separated: " + ", ".join(CHECKS)
    )
    p.add_argument(
        "--image",
        type=Path,
        action="append",
        default=None,
        help="a compiled model directory; repeat it once per width for the width check "
        "(default: a random tiny model compiled at each width)",
    )
    p.add_argument("--widths", default="64,128", help="comma-separated WB values")
    p.add_argument(
        "--program", default=PROGRAM, help="comma-separated: " + ", ".join(compare.PROGRAMS)
    )
    p.add_argument("--tok", type=int, default=0, help="token id the first pass embeds")
    p.add_argument("--generate", type=int, default=2, help="tokens to generate per run")
    p.add_argument("--prompt", default="1,2,3", help="comma-separated prompt ids")
    p.add_argument("--seed", type=int, default=0, help="seed for the random tiny model")
    p.add_argument(
        "--x-seeds",
        default=",".join(str(s) for s in X_SEEDS),
        help="comma-separated seeds for the undefined-value starts",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None, help="where images and records are written"
    )
    p.add_argument("--verbose", action="store_true", help="let the harness print its own output")
    return p


def images_for(a: argparse.Namespace, widths: Sequence[int]) -> dict[int, Path]:
    """The compiled image per width: the ones named by ``--image``, else a random tiny model."""
    if not a.image:
        return tiny_images(a.out_dir, seed=a.seed, widths=widths)
    out: dict[int, Path] = {}
    for path in a.image:
        wb = int(compiler.load_layout(path)["wb"])
        out[wb] = Path(path)
    return out


def run(a: argparse.Namespace) -> int:
    """Run the checks the parsed options name and print their tables; 1 on any difference."""
    checks = tuple(name for name in a.checks.split(",") if name)
    for name in checks:
        if name not in CHECKS:
            raise SystemExit(f"unknown check {name!r}; give one of {', '.join(CHECKS)}")
    widths = [int(w) for w in a.widths.split(",")]
    images = images_for(a, widths)
    prompt = [int(v) for v in a.prompt.split(",") if v]
    programs = [p for p in a.program.split(",") if p]
    seeds = [int(s) for s in a.x_seeds.split(",") if s]
    quiet = not a.verbose
    first = min(images)
    cfg = compare.CONFIGS[first]
    reports: list[Report] = []
    for name in checks:
        for program in programs:
            if name == "width":
                if len(images) < 2:
                    raise SystemExit("the width check needs an image at two widths")
                reports.append(
                    check_width(
                        images,
                        program=program,
                        tok=a.tok,
                        max_new=a.generate,
                        prompt=prompt,
                        out_dir=a.out_dir,
                        quiet=quiet,
                    )
                )
            elif name == "timing":
                reports.append(
                    check_timing(
                        images[first],
                        cfg,
                        program=program,
                        tok=a.tok,
                        max_new=a.generate,
                        prompt=prompt,
                        out_dir=a.out_dir,
                        quiet=quiet,
                    )
                )
            else:
                reports.append(
                    check_x_initial(
                        images[first],
                        cfg,
                        seeds=seeds,
                        program=program,
                        tok=a.tok,
                        max_new=a.generate,
                        prompt=prompt,
                        out_dir=a.out_dir,
                        quiet=quiet,
                    )
                )
    return report(reports)


def main(argv: list[str] | None = None) -> int:
    p = add_arguments(argparse.ArgumentParser(prog="quettos.determinism", description=SUMMARY))
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
