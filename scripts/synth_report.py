#!/usr/bin/env python3
"""Write syn/reports/<block>.md from the output of the syn/synth_*.ys scripts.

Every figure in a report is read back out of the Yosys log the run just wrote:
the version string, the elaborated parameters of each configuration, the
`stat -tech xilinx` cell table, the hard-block instances with the source line
each was inferred from, and the `ltp` longest topological path with its cut
points and its endpoints. Nothing is copied by hand, so a report cannot drift
away from the RTL it describes.

Usage:
    uv run python scripts/synth_report.py            # run and rewrite the reports
    uv run python scripts/synth_report.py --check    # run and fail on any difference
    uv run python scripts/synth_report.py --script syn/synth_top.ys

`--check` regenerates into a temporary directory and diffs, so a stale report is
a failing command rather than a misleading page. The `## Notes (hand-written)`
section at the end of a report is the one part written by a person; it is
carried forward unchanged.

The README row that quotes the whole-core report is held to the same run: every
figure it states is read back off the row and required to equal what Yosys just
reported for the demo configuration, so the front page cannot drift away from
the page it points at.

Yosys's own output is passed through, minus the lines `is_noise` names: the
`check` pass that `synth_xilinx` runs between `proc` and its first `opt_clean`
reports the scaffolding elaboration has just built, and the next pass deletes
it. The full log under `build/synth/` keeps every line.
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYN_DIR = ROOT / "syn"
REPORT_DIR = SYN_DIR / "reports"
LOG_DIR = ROOT / "build" / "synth"
SHARED = SYN_DIR / "report.ys"
FRONT_PAGE = ROOT / "README.md"

# The README row that quotes syn/reports/qcore_top.md, and the figures it states.
# Each is read off the row and held to the demo configuration of the run.
FRONT_PAGE_ROW = "| `qcore_top` on xc7 |"
FRONT_PAGE_CELLS = ("DSP48E1", "RAMB36E1", "RAMB18E1", "RAM32M", "SRL16E")
FRONT_PAGE_TOTALS = ("LUTs", "flops", "cells", "estimated LCs")
FRONT_PAGE_FIGURE = re.compile(
    r"(\d[\d,]*) (" + "|".join(FRONT_PAGE_CELLS + FRONT_PAGE_TOTALS) + r")\b"
)

NOTES_HEADING = "## Notes (hand-written)"
DEFAULT_NOTES = "What dominates the area of this block, in a sentence.\n"

# stat rows that are not cells
_STAT_SKIP = {"wires", "ports", "cells", "memories", "processes", "submodules"}

# how the ltp cut points read in prose
_CUT_GROUPS = [
    ("the flops", ("FDRE", "FDSE", "FDCE", "FDPE", "FDRSE")),
    ("the clock buffer", ("BUFG", "BUFGCTRL")),
    ("the I/O buffers", ("IBUF", "OBUF", "IOBUF")),
    ("the hard blocks", ("DSP48E1", "RAMB36E1", "RAMB18E1", "RAM32M", "RAM64M", "RAM128X1D")),
]


class ReportError(RuntimeError):
    """A script or a log did not carry what a report needs."""


@dataclass
class Config:
    """One configuration section of one .ys script."""

    label: str
    params: list[tuple[str, str]] = field(default_factory=list)
    module: str = ""
    cells: list[tuple[str, int]] = field(default_factory=list)
    total_cells: int = 0
    est_lcs: int = 0
    ltp_len: int = 0
    ltp_path: list[tuple[int, str]] = field(default_factory=list)
    hard: dict[str, list[str]] = field(default_factory=dict)
    memories: list[tuple[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- run


# A wire the AST front end leaves behind: the per-bit `$mem2bits$` expansion of
# an array `$readmemh` fills, and the argument and result wires of an inlined
# `qcore_pkg::` function. `synth_xilinx` runs `check` once between `proc` and
# its first `opt_clean`, and that is the pass that prints these; its two later
# `check` passes print none, because `opt_clean` has deleted the wires by then.
# `scripts/lint.sh` runs `proc; opt; check -assert` over every module, so an
# undriven wire that outlives `opt` is still an error -- what these lines
# describe is the state between two passes, not the design.
_ELABORATION_WIRE = re.compile(
    r"^Warning: Wire \S*\$(?:mem2bits|func)\$\S+ \[\d+\] is used but has no driver\.$"
)


def is_noise(line: str) -> bool:
    """Whether a Yosys stderr line describes a pass in progress rather than the design."""
    return "Resizing cell port" in line or _ELABORATION_WIRE.match(line) is not None


def check_noise_is_scaffolding(script: Path, sections: list[tuple[str, list[str]]]) -> None:
    """The dropped warnings have to be the ones the next pass deletes.

    `is_noise` judges a stderr line by its wire name, and stderr carries no pass
    boundary. The log carries one: within a configuration, the first `opt_clean`
    is the pass that removes the elaboration scaffolding. A warning that matches
    after it names a wire that survived, so it fails the run rather than the
    console, and the filter cannot quietly grow into a real finding.
    """
    for label, lines in sections:
        start = next(
            (i for i, ln in enumerate(lines) if "Executing OPT_CLEAN pass" in ln), len(lines)
        )
        late = [ln for ln in lines[start:] if _ELABORATION_WIRE.match(ln)]
        if late:
            raise ReportError(
                f"{label}: {len(late)} undriven wire(s) outlive `opt_clean` in "
                f"{script.name}, starting with {late[0]}"
            )


def run_script(script: Path, log: Path) -> None:
    """Run one Yosys script, leaving its log at `log`."""
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = yosys_argv(script, log)
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    kept = [ln for ln in proc.stderr.splitlines() if not is_noise(ln)]
    if proc.returncode != 0:
        sys.stderr.write("\n".join(kept) + "\n")
        raise ReportError(f"{' '.join(argv)} failed with exit {proc.returncode}")
    for line in kept:
        print(line)


def yosys_argv(script: Path, log: Path) -> list[str]:
    """The exact command a report quotes."""
    return [
        "yosys",
        "-q",
        "-l",
        log.relative_to(ROOT).as_posix(),
        "-s",
        script.relative_to(ROOT).as_posix(),
    ]


# ------------------------------------------------------------------------- parse


def yosys_version(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Yosys ") and "(" in stripped:
            return stripped
    raise ReportError("no Yosys version line in the log")


def split_configs(text: str) -> list[tuple[str, list[str]]]:
    """Cut the log into the `==== qreport config <label>` sections."""
    lines = text.splitlines()
    marks = [
        (i, line.split()[-1])
        for i, line in enumerate(lines)
        if line.startswith("==== qreport config ")
    ]
    if not marks:
        raise ReportError("no `log ==== qreport config <label>` marker in the log")
    out = []
    for n, (start, label) in enumerate(marks):
        stop = marks[n + 1][0] if n + 1 < len(marks) else len(lines)
        out.append((label, lines[start:stop]))
    return out


def parse_params(lines: list[str]) -> list[tuple[str, str]]:
    """The parameters `chparam` set, as Yosys printed them before `hierarchy`."""
    seen: list[tuple[str, str]] = []
    names = set()
    for line in lines:
        if line.startswith("Executing HIERARCHY pass") or " Executing HIERARCHY pass" in line:
            break
        m = re.fullmatch(r"Parameter \\(\w+) = (-?\d+)", line.strip())
        if m and m.group(1) not in names:
            names.add(m.group(1))
            seen.append((m.group(1), m.group(2)))
    return seen


def region(lines: list[str], start: str, stop: str) -> list[str]:
    try:
        a = lines.index(start)
    except ValueError as exc:
        raise ReportError(f"missing marker {start!r}") from exc
    for b in range(a + 1, len(lines)):
        if lines[b].startswith(stop):
            return lines[a + 1 : b]
    raise ReportError(f"marker {start!r} is not closed by {stop!r}")


def parse_stat(lines: list[str]) -> tuple[str, list[tuple[str, int]], int, int]:
    """The `stat -tech xilinx` table: module, cell rows, total cells, estimated LCs."""
    heads = [ln for ln in lines if re.fullmatch(r"=== \S+ ===", ln.strip())]
    if len(heads) != 1:
        raise ReportError(f"expected one module in the stat table, found {len(heads)}")
    module = heads[0].strip().split()[1]
    if module == "design":
        raise ReportError("stat reported a hierarchy; the script must synthesize one top")
    rows: list[tuple[str, int]] = []
    total = est = 0
    after_total = False
    for line in lines[lines.index(heads[0]) + 1 :]:
        m = re.fullmatch(r"\s*Estimated number of LCs:\s+(\d+)\s*", line)
        if m:
            est = int(m.group(1))
            break
        m = re.fullmatch(r"\s*(\d+) cells\s*", line)
        if m:
            total = int(m.group(1))
            after_total = True
            continue
        m = re.fullmatch(r"\s*(\d+)\s+(\S+)\s*", line)
        if m and after_total and m.group(2) not in _STAT_SKIP:
            rows.append((m.group(2), int(m.group(1))))
    if not total or not rows:
        raise ReportError("could not read the stat cell table")
    return module, rows, total, est


def parse_ltp(lines: list[str]) -> tuple[int, list[tuple[int, str]]]:
    """The `ltp` length and the path entries, index and node name."""
    head = None
    length = 0
    for i, line in enumerate(lines):
        m = re.fullmatch(r"Longest topological path in (\S+) \(length=(\d+)\):", line.strip())
        if m:
            head, length = i, int(m.group(2))
            break
    if head is None:
        raise ReportError("no `Longest topological path` line in the ltp region")
    path = []
    for line in lines[head + 1 :]:
        m = re.fullmatch(r"\s*(\d+): (.*)", line)
        if not m:
            continue
        node = m.group(2).split(" (via ")[0].strip()
        path.append((int(m.group(1)), node))
    if len(path) != length + 1:
        raise ReportError(f"ltp printed {len(path)} entries for length {length}")
    return length, path


def parse_config(label: str, lines: list[str]) -> Config:
    cfg = Config(label=label, params=parse_params(lines))
    body = region(lines, "==== qreport stat", "==== qreport ")
    cfg.module, cfg.cells, cfg.total_cells, cfg.est_lcs = parse_stat(body)
    cfg.ltp_len, cfg.ltp_path = parse_ltp(region(lines, "==== qreport ltp", "==== qreport "))
    marks = [i for i, ln in enumerate(lines) if ln.startswith("==== qreport cells ")]
    for n, i in enumerate(marks):
        kind = lines[i].split()[-1]
        stop = marks[n + 1] if n + 1 < len(marks) else lines.index("==== qreport end")
        cfg.hard[kind] = [ln.strip() for ln in lines[i + 1 : stop] if ln.strip()]
    for line in lines:
        m = re.fullmatch(r"mapping memory (\S+) via (\S+)", line.strip())
        if m:
            cfg.memories.append((m.group(1), m.group(2)))
    check_hard_counts(cfg)
    return cfg


def check_hard_counts(cfg: Config) -> None:
    """The listed instances must account for the stat table exactly."""
    table = dict(cfg.cells)
    for kind, names in cfg.hard.items():
        if len(names) != table.get(kind, 0):
            raise ReportError(
                f"{cfg.label}: stat counts {table.get(kind, 0)} {kind}, select listed {len(names)}"
            )


# ---------------------------------------------------------------- attribution


_SRC = re.compile(r"\$(\w+)\$([A-Za-z0-9_./-]+\.sv:\d+)\$")


def attribute(name: str) -> str:
    """Where one hard-block instance came from, read out of its own name."""
    body = name.split("/", 1)[1] if "/" in name.split("$")[0] else name
    body = body.replace("$flatten\\", "").replace("\\", "")
    m = _SRC.search(body)
    if m:
        inst = body[: m.start()].rstrip(".")
        site = f"`${m.group(1)}` at `{m.group(2)}`"
        return f"`{index_star(inst)}`, {site}" if inst else site
    inst = re.sub(r"(\.\d+)+$", "", body)
    return f"`{index_star(inst)}`" if inst else "`?`"


def index_star(path: str) -> str:
    """Collapse generate indices so one row covers a whole array."""
    return re.sub(r"\[\d+\]", "[*]", path)


def hard_rows(cfg: Config) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    for kind, names in cfg.hard.items():
        groups: dict[str, int] = {}
        for name in names:
            key = attribute(name)
            groups[key] = groups.get(key, 0) + 1
        for key, count in sorted(groups.items(), key=lambda kv: (-kv[1], kv[0])):
            rows.append((kind, count, key))
    return rows


# ------------------------------------------------------------------- rendering


def cut_points(ltp_line: str) -> str:
    """Read the ltp cut points out of syn/report.ys and say them in prose."""
    types = re.findall(r"t:(\S+)", ltp_line)
    parts = []
    left = list(types)
    for phrase, members in _CUT_GROUPS:
        hit = [t for t in types if t in members]
        if hit:
            parts.append(f"{phrase} (" + ", ".join(f"`{t}`" for t in hit) + ")")
            left = [t for t in left if t not in hit]
    if left:
        parts.append(", ".join(f"`{t}`" for t in left))
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def landmark(node: str) -> str | None:
    """A path entry worth naming: a public signal, a port pad, or a source line."""
    if node.startswith("\\"):
        return node[1:].replace("$flatten\\", "").replace("\\", "")
    m = re.search(r"\$iopadmap\$(.+)$", node)
    if m:
        return m.group(1)
    m = re.search(r"[A-Za-z0-9_./-]+\.sv:\d+", node)
    return m.group(0) if m else None


def path_block(cfg: Config) -> list[str]:
    """Index 0, every named landmark in order, and the last index."""
    keep: list[tuple[int, str]] = []
    last = None
    for idx, node in cfg.ltp_path:
        mark = landmark(node)
        if idx == 0 or idx == cfg.ltp_len:
            keep.append((idx, mark or node))
            last = mark
        elif mark and mark != last:
            keep.append((idx, mark))
            last = mark
    width = len(str(cfg.ltp_len))
    return [f"{idx:>{width}}  {text}" for idx, text in keep]


def totals_sentence(cfg: Config) -> str:
    table = dict(cfg.cells)
    luts = sum(v for k, v in cfg.cells if re.fullmatch(r"LUT\d", k))
    flops = sum(v for k, v in cfg.cells if re.fullmatch(r"FD\w+", k))
    pads = table.get("IBUF", 0) + table.get("OBUF", 0)
    scope = table.get("$scopeinfo", 0)
    head = f"{cfg.total_cells} cells in total"
    if scope:
        head += f" ({scope} of them `$scopeinfo` hierarchy markers, which map to nothing)"
    parts = [f"{luts} LUTs", f"{flops} flops"]
    if table.get("CARRY4"):
        parts.append(f"{table['CARRY4']} `CARRY4`")
    if pads:
        parts.append(f"{pads} I/O pads")
    return f"{head}: " + ", ".join(parts) + f". Yosys estimates {cfg.est_lcs} LCs."


def para(text: str) -> list[str]:
    """One prose paragraph, wrapped so the file diffs a line at a time."""
    return textwrap.wrap(" ".join(text.split()), width=92)


def render(script: Path, log: Path, version: str, cfgs: list[Config], notes: str) -> str:
    module = cfgs[0].module
    ltp_line = next(ln for ln in SHARED.read_text().splitlines() if ln.startswith("ltp "))
    out: list[str] = []
    out.append(f"# {module} synthesis (xc7)")
    out.append("")
    out.extend(
        para(
            f"Every number below is read back out of `{log.relative_to(ROOT).as_posix()}` by "
            "`scripts/synth_report.py`, which `make synth` runs. The run comes first and this "
            "page is written from it, so the two cannot disagree. `--check` regenerates the "
            "page and requires it back byte for byte on the Yosys build named below; another "
            "build has its own LUT packing and path lengths, so it is held to the parameters "
            "and the hard-block inventory."
        )
    )
    out.append("")
    out.append(f"Tool: `{version}`")
    out.append("")
    out.append("```sh")
    out.append("mkdir -p build/synth")
    out.append(" ".join(yosys_argv(script, log)))
    out.append("```")
    out.append("")
    out.extend(
        para(
            "The cell counts are the `stat -tech xilinx` table after "
            "`synth_xilinx -family xc7`. A block synthesized on its own has its ports on pads, "
            "so `IBUF` and `OBUF` are in its totals; inside `qcore_top` they are internal wires."
        )
    )
    out.append("")
    out.extend(
        para(
            "The path depth is `ltp -noff` over the LUT fabric, cutting at "
            f"{cut_points(ltp_line)}, so it counts logic levels between registers and means the "
            "same thing in every block. "
            "Each path below lists its two endpoints and the named signals and source lines "
            "between them, with the position of each along the path."
        )
    )
    for cfg in cfgs:
        out.append("")
        out.append(f"## {cfg.label.capitalize()} configuration")
        out.append("")
        if cfg.params:
            out.extend(
                para("Parameters: " + ", ".join(f"`{k} = {v}`" for k, v in cfg.params) + ".")
            )
        else:
            out.append(f"`{cfg.module}` takes no parameters.")
        out.append("")
        out.append("| Cell | Count |")
        out.append("|---|---|")
        for name, count in cfg.cells:
            out.append(f"| `{name}` | {count} |")
        out.append("")
        out.extend(para(totals_sentence(cfg)))
        out.append("")
        rows = hard_rows(cfg)
        if rows:
            out.append("| Hard block | Count | Inferred from |")
            out.append("|---|---|---|")
            for kind, count, where in rows:
                out.append(f"| `{kind}` | {count} | {where} |")
        else:
            kinds = list(cfg.hard)
            names = ", ".join(f"`{k}`" for k in kinds[:-1]) + f" or `{kinds[-1]}`"
            out.extend(para(f"No hard blocks: the design has no {names} cell."))
        out.append("")
        if cfg.memories:
            out.append("Memories, as `memory_libmap` mapped them:")
            out.append("")
            for name, prim in cfg.memories:
                out.append(f"- `{name}` via `{prim}`")
        else:
            out.extend(
                para("`memory_libmap` mapped no memory: the block holds its state in flops.")
            )
        out.append("")
        cells = "cell" if cfg.ltp_len == 1 else "cells"
        out.append(f"Longest topological path through the LUT fabric: {cfg.ltp_len} {cells}.")
        out.append("")
        out.append("```")
        out.extend(path_block(cfg))
        out.append("```")
    out.append("")
    out.append(NOTES_HEADING)
    out.append("")
    out.append(notes.rstrip("\n"))
    return "\n".join(out) + "\n"


def keep_notes(path: Path) -> str:
    """Carry the hand-written tail of an existing report forward unchanged."""
    if not path.exists():
        return DEFAULT_NOTES
    text = path.read_text()
    if NOTES_HEADING not in text:
        return DEFAULT_NOTES
    body = text.split(NOTES_HEADING, 1)[1].lstrip("\n")
    return body if body.strip() else DEFAULT_NOTES


# ------------------------------------------------------------------- front page


def report_figures(cfg: Config) -> dict[str, int]:
    """The figures the README row quotes, as this run measured them."""
    table = dict(cfg.cells)
    figures = {name: table.get(name, 0) for name in FRONT_PAGE_CELLS}
    figures["LUTs"] = sum(v for k, v in cfg.cells if re.fullmatch(r"LUT\d", k))
    figures["flops"] = sum(v for k, v in cfg.cells if re.fullmatch(r"FD\w+", k))
    figures["cells"] = cfg.total_cells
    figures["estimated LCs"] = cfg.est_lcs
    return figures


def check_front_page(cfgs: list[Config]) -> list[str]:
    """Hold the README row that quotes the whole-core report to the run behind it.

    The row is the front page's copy of a generated table, which is the one
    place a synthesis figure can be typed. Every figure on it is read back and
    required to equal this run's demo configuration, and a figure the row stops
    stating fails too, so the check cannot be shortened away.
    """
    demo = next((c for c in cfgs if c.label == "demo"), None)
    if demo is None:
        return []
    rows = [ln for ln in FRONT_PAGE.read_text().splitlines() if ln.startswith(FRONT_PAGE_ROW)]
    if len(rows) != 1:
        return [f"README.md carries {len(rows)} rows starting `{FRONT_PAGE_ROW}`, expected one"]
    said = {unit: int(n.replace(",", "")) for n, unit in FRONT_PAGE_FIGURE.findall(rows[0])}
    figures = report_figures(demo)
    bad = [
        f"README.md's synthesis row says {said[unit]:,} {unit}, the run says {value:,}"
        for unit, value in figures.items()
        if unit in said and said[unit] != value
    ]
    absent = [unit for unit in figures if unit not in said]
    if absent:
        bad.append("README.md's synthesis row states no " + ", ".join(absent))
    return bad


# ------------------------------------------------------------------------ main


def build(script: Path) -> tuple[str, str, list[Config]]:
    """Run one script and return (report file name, report text, its configurations)."""
    log = LOG_DIR / f"{script.stem}.log"
    print(f"synth: {script.relative_to(ROOT)} -> {log.relative_to(ROOT)}")
    run_script(script, log)
    text = log.read_text()
    version = yosys_version(text)
    sections = split_configs(text)
    check_noise_is_scaffolding(script, sections)
    cfgs = [parse_config(label, lines) for label, lines in sections]
    modules = {c.module for c in cfgs}
    if len(modules) != 1:
        raise ReportError(f"{script.name}: configurations synthesize {sorted(modules)}")
    name = f"{cfgs[0].module}.md"
    page = render(script, log, version, cfgs, keep_notes(REPORT_DIR / name))
    check_facts_cover_params(name, cfgs, page)
    return name, page, cfgs


# Cell types whose count is an inference outcome, not an optimization detail: a
# different Yosys build may pack LUTs differently, but it must infer the same
# DSPs and memories from the same source. `--check` holds these to an exact
# match even when the tool that wrote the report was a different build.
HARD_BLOCKS = (
    "DSP48E1",
    "RAMB36E1",
    "RAMB18E1",
    "RAM32M",
    "RAM64M",
    "RAM128X1D",
)


def tool_banner() -> str:
    """The Yosys build this run uses, for the log."""
    out = subprocess.run(["yosys", "-V"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "yosys -V printed nothing"


def recorded_tool(text: str) -> str:
    """The tool version a recorded report was written with, empty if absent."""
    m = re.search(r"^Tool: `(.+)`$", text, re.M)
    return m.group(1) if m else ""


def design_facts(text: str) -> list[str]:
    """The lines a report must reproduce on any Yosys build.

    `render` wraps the parameter list to the page width, so a configuration with
    more parameters than one line holds carries the rest on continuation lines.
    The whole paragraph is joined back into a single fact before it is compared:
    every parameter is then held to the recorded value, and a build that happens
    to break the paragraph elsewhere still compares equal.
    """
    facts: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Parameters: "):
            paragraph = []
            while i < len(lines) and lines[i].strip():
                paragraph.append(lines[i].strip())
                i += 1
            facts.append(" ".join(paragraph))
            continue
        if line.startswith("# ") or line.startswith("## "):
            facts.append(line)
        elif line.startswith("| `") and any(f"`{b}`" in line for b in HARD_BLOCKS):
            facts.append(line)
        i += 1
    return facts


def check_facts_cover_params(name: str, cfgs: list[Config], text: str) -> None:
    """Every parameter the run elaborated has to reach the cross-build comparison.

    `design_facts` reads the rendered page, so a change to how `render` lays the
    parameters out could drop one from the comparison without changing any
    report. This fails the run instead: each `NAME = VALUE` the log carried is
    required back in the facts extracted from the page that was written from it.
    """
    facts = "\n".join(design_facts(text))
    missing = [f"{k} = {v}" for cfg in cfgs for k, v in cfg.params if f"`{k} = {v}`" not in facts]
    if missing:
        raise ReportError(
            f"{name}: the cross-build comparison does not cover " + ", ".join(missing)
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="regenerate into a temporary directory and fail on any difference",
    )
    ap.add_argument(
        "--script",
        action="append",
        default=None,
        metavar="PATH",
        help="one syn/synth_*.ys script (repeatable); the default is all of them",
    )
    args = ap.parse_args()

    scripts = (
        [Path(s) if Path(s).is_absolute() else ROOT / s for s in args.script]
        if args.script
        else sorted(SYN_DIR.glob("synth_*.ys"))
    )
    if not scripts:
        sys.stderr.write("synth-report: no syn/synth_*.ys script found\n")
        return 1

    started = time.monotonic()
    built: dict[str, str] = {}
    source: dict[str, Path] = {}
    configs: dict[str, list[Config]] = {}
    print(f"synth-report: {tool_banner()}")
    try:
        for script in scripts:
            try:
                name, text, cfgs = build(script)
            except ReportError as exc:
                raise ReportError(f"{script.name}: {exc}") from exc
            if name in built:
                raise ReportError(
                    f"{script.name} and {source[name].name} both write syn/reports/{name}"
                )
            built[name], source[name], configs[name] = text, script, cfgs
    except ReportError as exc:
        sys.stderr.write(f"synth-report: {exc}\n")
        return 1
    elapsed = time.monotonic() - started

    # The whole-core report is the one the front page quotes.
    front = check_front_page(configs.get("qcore_top.md", []))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.check:
        for name, text in built.items():
            (REPORT_DIR / name).write_text(text)
            print(f"synth-report: wrote syn/reports/{name}")
        for line in front:
            sys.stderr.write(f"synth-report: {line}\n")
        if front:
            return 1
        print(f"synth: {len(scripts)} script(s), {elapsed:.1f} s, OK")
        return 0

    failures: list[str] = list(front)
    cross: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in built.items():
            fresh = Path(tmp) / name
            fresh.write_text(text)
            recorded = REPORT_DIR / name
            if not recorded.exists():
                failures.append(f"syn/reports/{name} is missing")
                continue
            old_text = recorded.read_text()
            was, now = recorded_tool(old_text), recorded_tool(text)
            if was == now:
                if old_text != text:
                    failures.append(f"syn/reports/{name} does not match the run")
                    sys.stdout.writelines(
                        difflib.unified_diff(
                            old_text.splitlines(keepends=True),
                            text.splitlines(keepends=True),
                            fromfile=f"syn/reports/{name} (recorded)",
                            tofile=f"syn/reports/{name} (this run)",
                        )
                    )
                continue
            # A different Yosys build: cell packing and path lengths are its own,
            # so hold it to the design facts and say what drifted.
            old_facts, new_facts = design_facts(old_text), design_facts(text)
            if old_facts != new_facts:
                failures.append(
                    f"syn/reports/{name}: this Yosys infers a different design than the "
                    f"report records"
                )
                sys.stdout.writelines(
                    difflib.unified_diff(
                        [f"{ln}\n" for ln in old_facts],
                        [f"{ln}\n" for ln in new_facts],
                        fromfile=f"syn/reports/{name} (recorded, {was})",
                        tofile=f"syn/reports/{name} (this run, {now})",
                    )
                )
            else:
                cross.append(name)
    if args.script is None:
        for stale in sorted(REPORT_DIR.glob("*.md")):
            if stale.name not in built:
                failures.append(f"syn/reports/{stale.name} has no syn/synth_*.ys script")

    if failures:
        for line in failures:
            sys.stderr.write(f"synth-report: {line}\n")
        sys.stderr.write(
            "synth-report: FAILED -- rerun `uv run python scripts/synth_report.py` "
            "to rewrite the reports from this run\n"
        )
        return 1
    if cross:
        print(
            f"synth-report: {len(cross)} report(s) were written by another Yosys build; "
            "their hard-block inventory and parameters match this one"
        )
    print(f"synth: {len(scripts)} script(s), {elapsed:.1f} s, reports match, OK")
    return 0


if __name__ == "__main__":
    if shutil.which("yosys") is None:
        sys.stderr.write("synth-report: yosys is not on PATH\n")
        raise SystemExit(1)
    raise SystemExit(main())
