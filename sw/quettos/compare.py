"""RTL against the ISA simulator: one bring-up program run on both, compared element by element.

The program is :func:`quettos.compiler.build_bringup`: ``EMBED``, the tied LM
head as a ``GEMV`` in ``ARGMAX_DUMP`` mode, and ``HALT`` -- three descriptors
lifted from the model's own ``decode.prog``.  It runs on :mod:`quettos.isa_sim`
and on ``qcore_top`` through the Verilator harness in ``sim/verilator``, and
every VSRAM element, SREG word, dumped logit, CSR and PERF counter the two
produce is compared; the first difference is reported with the descriptor, the
field and the element it is in.

``qcore_vpu_top`` owns the ``VQUANT`` that pairs an activation with its scale,
so the host loads that one scale register and the ``EMBED`` output shift is
raised to match it, which keeps the gathered row inside the int16 window a GEMV
activation is read through.  Both models are given the same scale; every other
input is the compiled image.  A v1 model ties the embedding and the LM head, so
the program is self-checking: ``ARGMAX_TOK == TOK``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from quettos import compiler, isa, isa_sim, numerics, synthetic
from quettos.compiler import Bringup
from quettos.isa import OutMode

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "sim" / "verilator"
BUILD = REPO / "build"

#: Counters the simulator models; the cycle and beat counters are the RTL's alone.
PERF_COMPARED = ("DESCRIPTORS", "MACS", "WT_BYTES")
EVENTS = ("SAT_REQ", "SAT_VPU", "ERR_SHIFT", "ERR_BOUNDS")
DUMP_MODES = (OutMode.ARGMAX_DUMP, OutMode.VSRAM_DUMP)


@dataclass(frozen=True)
class Config:
    """One RTL configuration: the widths the harness is built with."""

    wb: int
    b_max: int
    vl: int
    vsram_words: int

    @property
    def widths(self) -> dict[str, int]:
        """The widths the compiler and the simulator take from a configuration."""
        return {"wb": self.wb, "b_max": self.b_max, "vsram_words": self.vsram_words}

    @property
    def make_args(self) -> list[str]:
        return [
            f"WB={self.wb}",
            f"B_MAX={self.b_max}",
            f"VL={self.vl}",
            f"VSRAM_WORDS={self.vsram_words}",
        ]

    def __str__(self) -> str:
        return f"WB={self.wb} B_MAX={self.b_max} VSRAM_WORDS={self.vsram_words}"


CONFIGS: dict[int, Config] = {
    16: Config(16, 2, 2, 2048),
    64: Config(64, 1, 4, 4096),
    128: Config(128, 1, 4, 4096),
}


@dataclass
class Record:
    """One descriptor's worth of state, in the form both models report it."""

    index: int
    pc: int
    status: int
    argmax_tok: int
    argmax_val: int
    events: dict[str, int]
    perf: dict[str, int]
    vsram: list[list[int]]
    sreg: list[list[int]]


@dataclass(frozen=True)
class Mismatch:
    """Where the two models first disagree."""

    record: int
    opcode: str
    what: str
    element: int | None
    expected: Any
    got: Any

    def __str__(self) -> str:
        where = f"descriptor {self.record} ({self.opcode}) {self.what}"
        if self.element is not None:
            where += f"[{self.element}]"
        return f"{where}: isa_sim {self.expected}, RTL {self.got}"


# --------------------------------------------------------------------------- the simulator


def _sreg_word(value: Any) -> int:
    """The 32-bit word an SREG entry holds: an sfloat pair, or a tracked absmax."""
    return isa.sfloat_imm(value) if isinstance(value, numerics.SFloat) else int(value) & 0xFFFFFFFF


def reference(
    image_dir: Path, b: Bringup, cfg: Config, *, step: bool = True
) -> tuple[list[Record], list[int]]:
    """Run the bring-up program on :mod:`quettos.isa_sim` and record what the host would read."""
    layout = compiler.load_layout(image_dir)
    m = isa_sim.Machine.from_file(image_dir / layout["image"]["file"], **cfg.widths)
    m.mem[b.addr : b.addr + len(b.blob)] = b.blob
    for bank, index, word in b.sreg:
        m.sreg[bank][index] = isa.sfloat_from_imm(word)
    m.csr["TOK"], m.csr["POS"], m.csr["ROW_EN"] = b.tok, 0, 1
    m.csr["PC"] = b.addr

    records: list[Record] = []
    if step:
        for index, d in enumerate(b.program):
            isa_sim.step(m, d)
            records.append(_capture(m, index, b, cfg))
    else:
        isa_sim.run_program(m, list(b.program), pc=b.addr)
        records.append(_capture(m, 0, b, cfg))
    addr, size = b.dump
    dumped = np.frombuffer(bytes(m.mem[addr : addr + size]), dtype="<i4")
    return records, [int(v) for v in dumped]


def _capture(m: isa_sim.Machine, index: int, b: Bringup, cfg: Config) -> Record:
    return Record(
        index=index,
        pc=m.csr["PC"],
        status=m.csr["STATUS"],
        argmax_tok=m.csr["ARGMAX_TOK"],
        argmax_val=m.argmax_val(),
        events={name: int(m.csr[name]) for name in EVENTS},
        perf={name: m.perf_value(name) for name in PERF_COMPARED},
        vsram=[
            [int(v) for v in m.vsram[bank, start : start + count]]
            for bank, start, count in b.ranges
        ],
        sreg=[[_sreg_word(v) for v in m.sreg[bank]] for bank in range(cfg.b_max)],
    )


# --------------------------------------------------------------------------- the RTL


def harness_binary(cfg: Config, *, quiet: bool = False) -> Path:
    """Build the Verilator harness for ``cfg`` if it is not already built, and name the binary."""
    out = subprocess.DEVNULL if quiet else None
    subprocess.run(
        ["make", "-C", str(HARNESS), "build", *cfg.make_args], cwd=REPO, check=True, stdout=out
    )
    where = subprocess.run(
        ["make", "-s", "-C", str(HARNESS), "where", *cfg.make_args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(where.stdout.strip())


def run_rtl(
    image_dir: Path,
    b: Bringup,
    cfg: Config,
    *,
    lat: int = 32,
    bw_div: int = 1,
    step: bool = True,
    out_dir: Path | None = None,
    quiet: bool = True,
    allow_error: bool = False,
) -> tuple[list[Record], list[int], dict[str, Any]]:
    """Run the bring-up program on ``qcore_top`` through the harness and read its records back.

    ``allow_error`` keeps the records of a run the core stopped, which is what a
    program the hardware cannot execute produces.
    """
    out_dir = BUILD / "compare" if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"w{cfg.wb}-tok{b.tok}-lat{lat}-bw{bw_div}{'-step' if step else '-run'}"
    prog_path = out_dir / f"bringup-{tag}.prog"
    json_path = out_dir / f"bringup-{tag}.json"
    prog_path.write_bytes(b.blob)

    cmd = [
        str(harness_binary(cfg, quiet=quiet)),
        "--image",
        str(image_dir),
        "--program",
        str(prog_path),
        "--program-addr",
        str(b.addr),
        "--bringup-json",
        str(json_path),
        "--tok",
        str(b.tok),
        "--pos",
        "0",
        "--row-en",
        "1",
        "--lat",
        str(lat),
        "--bw-div",
        str(bw_div),
    ]
    if step:
        cmd.append("--step")
    for bank, index, word in b.sreg:
        cmd += ["--sreg", f"{bank}:{index}={word:#010x}"]
    for bank, start, count in b.ranges:
        cmd += ["--dump-vsram", f"{bank}:{start}:{count}"]
    cmd += ["--dump-mem", f"{b.dump[0]}:{b.dump[1]}"]
    if quiet:
        cmd.append("--quiet")
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0 and not (allow_error and json_path.exists()):
        raise RuntimeError(
            f"qcore_sim exited {proc.returncode}\n{' '.join(cmd)}\n{proc.stdout}{proc.stderr}"
        )
    blob = json.loads(json_path.read_text())
    records = [
        Record(
            index=int(r["index"]),
            pc=int(r["pc"]),
            status=int(r["status"]),
            argmax_tok=int(r["argmax_tok"]),
            argmax_val=int(r["argmax_val"]),
            events={name: int(r["events"][name]) for name in EVENTS},
            perf={name: int(r["perf"][name]) for name in PERF_COMPARED},
            vsram=[[int(v) for v in vals] for vals in r["vsram"]],
            sreg=[[int(v) for v in bank] for bank in r["sreg"]],
        )
        for r in blob["records"]
    ]
    mem = [int(v) for v in blob["mem"][0]["values"]] if blob["mem"] else []
    return records, mem, blob


# --------------------------------------------------------------------------- the comparison


def _first_difference(expected: list[int], got: list[int]) -> int | None:
    if len(expected) != len(got):
        return min(len(expected), len(got))
    for i, (e, g) in enumerate(zip(expected, got, strict=True)):
        if e != g:
            return i
    return None


def compare(
    ref: list[Record],
    got: list[Record],
    b: Bringup,
    ref_mem: list[int] | None = None,
    got_mem: list[int] | None = None,
) -> list[Mismatch]:
    """Every difference between the two runs, the first element of each named."""
    out: list[Mismatch] = []
    names = [d.opcode.name for d in b.program]
    if ref_mem is not None and got_mem is not None:
        # The DUMP region is read after HALT, whose auto-fence has acknowledged
        # every write the program issued.
        i = _first_difference(ref_mem, got_mem)
        if i is not None:
            dumper = next(k for k, d in enumerate(b.program) if d.out_mode in DUMP_MODES)
            out.append(
                Mismatch(
                    dumper,
                    names[dumper],
                    f"DUMP at 0x{b.dump[0]:08x} output",
                    i,
                    ref_mem[i] if i < len(ref_mem) else None,
                    got_mem[i] if i < len(got_mem) else None,
                )
            )
    if len(ref) != len(got):
        return [Mismatch(0, "-", "record count", None, len(ref), len(got))]
    for e, g in zip(ref, got, strict=True):
        op = names[e.index] if e.index < len(names) else "?"
        for what, ev, gv in (
            ("PC", e.pc, g.pc),
            ("STATUS", e.status, g.status),
            ("ARGMAX_TOK", e.argmax_tok, g.argmax_tok),
            ("ARGMAX_VAL", e.argmax_val, g.argmax_val),
        ):
            if ev != gv:
                out.append(Mismatch(e.index, op, what, None, ev, gv))
        for name in EVENTS:
            if e.events[name] != g.events[name]:
                out.append(Mismatch(e.index, op, name, None, e.events[name], g.events[name]))
        for name in PERF_COMPARED:
            if e.perf[name] != g.perf[name]:
                out.append(Mismatch(e.index, op, f"PERF {name}", None, e.perf[name], g.perf[name]))
        for (bank, start, _), ev_list, gv_list in zip(b.ranges, e.vsram, g.vsram, strict=True):
            i = _first_difference(ev_list, gv_list)
            if i is not None:
                out.append(
                    Mismatch(
                        e.index,
                        op,
                        f"VSRAM[{bank}] element {start + i}",
                        start + i,
                        ev_list[i] if i < len(ev_list) else None,
                        gv_list[i] if i < len(gv_list) else None,
                    )
                )
        for bank, (ev_list, gv_list) in enumerate(zip(e.sreg, g.sreg, strict=True)):
            i = _first_difference(ev_list, gv_list)
            if i is not None:
                out.append(
                    Mismatch(e.index, op, f"SREG[{bank}]", i, hex(ev_list[i]), hex(gv_list[i]))
                )
    return out


# --------------------------------------------------------------------------- runs


@dataclass
class Result:
    """One image, one configuration, one token: what both models produced."""

    image: Path
    config: Config
    tok: int
    shift: int
    argmax_tok: int
    argmax_val: int
    cycles: int
    perf: dict[str, int]
    mismatches: list[Mismatch] = field(default_factory=list)
    determinism: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.determinism


def check_image(
    image_dir: Path,
    cfg: Config,
    tok: int,
    *,
    timing: tuple[tuple[int, int], ...] = ((32, 1),),
    quiet: bool = True,
) -> Result:
    """Compare the RTL against the simulator on one image, and across the timing settings."""
    b = compiler.build_bringup(image_dir, tok, **cfg.widths)
    ref, ref_mem = reference(image_dir, b, cfg)
    baseline: tuple[list[Record], list[int]] | None = None
    determinism: list[str] = []
    got: list[Record] = []
    got_mem: list[int] = []
    cycles = 0
    for lat, bw_div in timing:
        got, got_mem, blob = run_rtl(image_dir, b, cfg, lat=lat, bw_div=bw_div, quiet=quiet)
        if baseline is None:
            baseline = (got, got_mem)
            cycles = int(blob["run"]["clock_cycles"])  # the first setting is the reported one
        elif (got, got_mem) != baseline:
            determinism.append(
                f"lat={lat} bw_div={bw_div} differs from lat={timing[0][0]} bw_div={timing[0][1]}"
            )
    last = got[-1]
    return Result(
        image=image_dir,
        config=cfg,
        tok=tok,
        shift=b.shift,
        argmax_tok=last.argmax_tok,
        argmax_val=last.argmax_val,
        cycles=cycles,
        perf=last.perf,
        mismatches=compare(ref, got, b, ref_mem, got_mem),
        determinism=determinism,
    )


def compile_shape(shape: synthetic.Shape, seed: int, cfg: Config, out_dir: Path) -> Path:
    """Build a random tiny model and compile it for ``cfg``; returns the image directory."""
    syn = synthetic.build(shape, seed=seed)
    # The context length is a whole number of weight-port tiles, so the tiny
    # models are compiled one tile deep at WB = 128.
    max_ctx = -(-synthetic.MAX_CTX // cfg.wb) * cfg.wb
    compiled = compiler.compile(
        syn.quant,
        syn.spec,
        out_dir=out_dir,
        max_ctx=max_ctx,
        wb=cfg.wb,
        a_bits=16,
        vsram_words=cfg.vsram_words,
    )
    return Path(compiled.out_dir)


def sweep(
    shapes: int,
    widths: list[int],
    *,
    seed: int = 0,
    tokens: int = 1,
    timing: tuple[tuple[int, int], ...] = ((32, 1), (1, 1), (200, 1), (32, 2)),
    out_dir: Path | None = None,
    quiet: bool = True,
) -> list[Result]:
    """Every shape at every width, each compared at every timing setting."""
    out_dir = BUILD / "compare" if out_dir is None else out_dir
    rng = np.random.default_rng(seed)
    picked = [synthetic.random_shape(rng) for _ in range(shapes)]
    results: list[Result] = []
    for i, shape in enumerate(picked):
        for wb in widths:
            cfg = CONFIGS[wb]
            image = compile_shape(shape, seed + i, cfg, out_dir / f"shape{i}-w{wb}")
            vocab = compiler.load_layout(image)["model"]["vocab"]
            for _ in range(tokens):
                tok = int(rng.integers(0, vocab))
                results.append(check_image(image, cfg, tok, timing=timing, quiet=quiet))
    return results


# --------------------------------------------------------------------------- CLI


def _report(results: list[Result]) -> int:
    print(f"{'model':<44} {'config':<32} {'tok':>6} {'argmax':>7} {'cycles':>10}  result")
    bad = 0
    for r in results:
        status = "ok" if r.ok else "MISMATCH"
        bad += 0 if r.ok else 1
        print(
            f"{r.image.name:<44} {str(r.config):<32} {r.tok:>6} {r.argmax_tok:>7} "
            f"{r.cycles:>10}  {status}"
        )
        for m in r.mismatches[:8]:
            print(f"    {m}")
        for d in r.determinism:
            print(f"    determinism: {d}")
    print(f"\n{len(results) - bad}/{len(results)} runs match sw/quettos/isa_sim.py")
    return 1 if bad else 0


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The comparison's options, shared by ``quettos compare`` and ``python -m quettos.compare``."""
    p.add_argument("--image", type=Path, help="a compiled model directory to compare on")
    p.add_argument("--wb", type=int, default=64, help="weight-port width of the RTL build")
    p.add_argument("--tok", type=int, default=0, help="token id the bring-up program embeds")
    p.add_argument("--lat", type=int, default=32, help="QMEM read latency in cycles")
    p.add_argument("--bw-div", type=int, default=1, help="one returned beat every N cycles")
    p.add_argument("--sweep", action="store_true", help="random shapes over several widths")
    p.add_argument("--shapes", type=int, default=5, help="how many random shapes to draw")
    p.add_argument("--widths", default="64,128", help="comma-separated WB values for --sweep")
    p.add_argument("--tokens", type=int, default=1, help="tokens per image in --sweep")
    p.add_argument("--seed", type=int, default=0, help="seed for the shapes and the token ids")
    p.add_argument(
        "--out-dir", type=Path, default=None, help="where images and records are written"
    )
    p.add_argument("--verbose", action="store_true", help="let the harness print its own output")
    return p


def run(a: argparse.Namespace) -> int:
    """Run the comparison the parsed options describe and print the table; 1 on any mismatch."""
    if a.sweep:
        results = sweep(
            a.shapes,
            [int(w) for w in a.widths.split(",")],
            seed=a.seed,
            tokens=a.tokens,
            out_dir=a.out_dir,
            quiet=not a.verbose,
        )
    elif a.image is not None:
        results = [
            check_image(
                a.image, CONFIGS[a.wb], a.tok, timing=((a.lat, a.bw_div),), quiet=not a.verbose
            )
        ]
    else:
        raise SystemExit("give --image or --sweep")
    return _report(results)


def main(argv: list[str] | None = None) -> int:
    p = add_arguments(
        argparse.ArgumentParser(prog="quettos.compare", description=__doc__.splitlines()[0])
    )
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
