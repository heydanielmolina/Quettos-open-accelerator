"""Prefix reuse: the saved KV state, what it belongs to, and the tool-call demo.

An agent turn repeats a long head: the system message and the tool
descriptions are the same on every call, and only the user's turn changes.
``qcore_top`` can compute that head once, hand its KV region out through
``--kv-save`` and take it back through ``--kv-load``, and prefill only the
positions after it.  This module is the reader of that file (``sim/verilator/
prefix.hpp`` writes it) and the demo built on it: prefix, restore, generate,
and the same run again without reuse to hold the ids to.

``python -m quettos.prefix`` runs the demo end to end; ``make demo-toolcall``
is that command.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quettos import compare, compiler
from quettos.model import REPO_ROOT

#: The prefix file, as ``sim/verilator/prefix.hpp`` lays it out: the magic, the
#: fixed header, one u32 per covered position, then the KV region.
MAGIC = b"QKVPRFX1"
VERSION = 1
NAME_CHARS = 64
FIXED_BYTES = 176
HEADER_FMT = "<8sIIIIIIQQ64s64s"

#: The alternative question the shared-prefix check asks: the same tools, a
#: different user turn.  Nothing is generated from it -- it is rendered and
#: tokenized to show where two turns of one conversation stop agreeing.
OTHER_QUESTION = "Is it going to rain in Berlin tomorrow?"


class PrefixError(ValueError):
    """A prefix file that cannot be read, or does not belong to the run reading it."""


@dataclass(frozen=True)
class Header:
    """What a prefix file says its KV bytes belong to."""

    version: int
    isa_version: int
    wb: int
    max_ctx: int
    kv_base: int
    kv_size: int
    model: str
    image_sha256: str
    ids: tuple[int, ...]
    path: Path
    bytes: int

    @property
    def positions(self) -> int:
        return len(self.ids)


def read_header(path: str | Path) -> Header:
    """Parse a prefix file's header and check it against its own size.

    Reads the header alone: the KV payload is up to 26 MB and nothing on this
    side needs it, so the file is held to the size the header describes rather
    than loaded.
    """
    path = Path(path)
    data = path.read_bytes()[:FIXED_BYTES]
    size = path.stat().st_size
    if len(data) < FIXED_BYTES:
        raise PrefixError(f"{path} is {size} B, shorter than a {FIXED_BYTES} B prefix header")
    magic, version, header_bytes, isa, wb, max_ctx, positions, kv_base, kv_size, model, sha = (
        struct.unpack(HEADER_FMT, data)
    )
    if magic != MAGIC:
        raise PrefixError(f"{path} does not start with the prefix magic {MAGIC!r}")
    if version != VERSION:
        raise PrefixError(f"{path} is prefix format version {version}, not {VERSION}")
    if header_bytes != FIXED_BYTES + 4 * positions:
        raise PrefixError(
            f"{path} says its header is {header_bytes} B, not the "
            f"{FIXED_BYTES + 4 * positions} B of {positions} positions"
        )
    if size != header_bytes + kv_size:
        raise PrefixError(
            f"{path} is {size} B, not the {header_bytes + kv_size} B of a {header_bytes} B "
            f"header and a {kv_size} B KV region"
        )
    ids = struct.unpack(f"<{positions}I", path.read_bytes()[FIXED_BYTES:header_bytes])
    return Header(
        version=version,
        isa_version=isa,
        wb=wb,
        max_ctx=max_ctx,
        kv_base=kv_base,
        kv_size=kv_size,
        model=model.rstrip(b"\0").decode("utf-8"),
        image_sha256=sha.rstrip(b"\0").decode("utf-8"),
        ids=ids,
        path=path,
        bytes=size,
    )


def check_header(h: Header, layout: dict[str, Any], ids: Sequence[int]) -> list[str]:
    """The terms the file was written under, against the image and prompt it is read with.

    The same list the harness refuses on (``verify_prefix`` in
    ``sim/verilator/prefix.hpp``), read back on this side from the file the
    hardware wrote: a run of one is a check of the other.
    """
    bad: list[str] = []
    if h.isa_version != int(layout["isa_version"]):
        bad.append(f"ISA version {h.isa_version}, the image {layout['isa_version']}")
    if h.wb != int(layout["wb"]):
        bad.append(f"saved at WB={h.wb}, the image at WB={layout['wb']}")
    if h.max_ctx != int(layout["max_ctx"]):
        bad.append(f"saved at MAX_CTX={h.max_ctx}, the image at {layout['max_ctx']}")
    if h.model != str(layout["model"]["name"]):
        bad.append(f"model {h.model}, the image {layout['model']['name']}")
    if h.image_sha256 != str(layout["image"]["sha256"]):
        bad.append(f"image sha256 {h.image_sha256}, the image {layout['image']['sha256']}")
    kv_base = int(layout["bases"]["kv"])
    kv_size = int(layout["image"]["size"]) - kv_base
    if h.kv_base != kv_base or h.kv_size != kv_size:
        bad.append(f"KV region {h.kv_size} B at {h.kv_base}, the image {kv_size} B at {kv_base}")
    if h.positions == 0:
        bad.append("it covers no position")
    if h.positions >= len(ids):
        bad.append(f"it covers {h.positions} positions of a {len(ids)}-id prompt")
    for i, (saved, given) in enumerate(zip(h.ids, ids, strict=False)):
        if saved != given:
            bad.append(f"position {i} was token {saved}, this prompt has {given}")
            break
    return bad


# --------------------------------------------------------------------------- the boundary


def first_turn(ids: Sequence[int], eos_ids: Sequence[int]) -> int:
    """How many leading ids of a rendered prompt are its first turn.

    A chat template ends every turn with one of the model's own end-of-turn
    ids, so the first of those closes the system message -- the turn that
    carries the tool descriptions, and the one an agent repeats unchanged on
    every call.  That count is the prefix: the positions a later turn restores
    instead of recomputing.
    """
    for i, v in enumerate(ids):
        if v in set(eos_ids):
            return i + 1
    raise PrefixError(
        f"none of the {len(ids)} prompt ids is one of the model's end-of-turn ids "
        f"{list(eos_ids)}, so the prompt has no first turn to reuse"
    )


def shared_with_another_question(image: Path, layout: dict[str, Any], question: str) -> int:
    """How many leading ids this prompt shares with the same tools asked something else.

    The prefix is only worth saving if it belongs to the conversation rather
    than to the question, so the demo renders the prompt file's own tools with
    a different user turn through the same chat template and counts the ids the
    two renderings agree on.
    """
    from quettos.model import load_spec
    from quettos.tokenizer_io import encode, load_prompt, render_chat

    source = compare.prompt_source(layout)
    if source is None:
        raise PrefixError(f"{image}/layout.json does not name the prompt file it was compiled with")
    prompt = load_prompt(REPO_ROOT / source)
    spec = load_spec(str(layout["model"]["repo_id"]))
    other = encode(
        spec,
        render_chat(
            spec,
            [{"role": "user", "content": question}],
            tools=prompt.get("tools") or None,
            add_generation_prompt=True,
        ),
    )
    ids = compare.prompt_ids(image)
    n = 0
    while n < min(len(ids), len(other)) and ids[n] == other[n]:
        n += 1
    return n


# --------------------------------------------------------------------------- the report


def _cycles(perf: dict[str, Any], decode: bool) -> tuple[int, int]:
    """Cycles and token count of one pass of a run, from its per-token records."""
    rows = [t for t in perf["tokens"] if (str(t["pass"]) == "decode") == decode]
    return sum(int(t["cycles"]) for t in rows), len(rows)


def _ids(perf: dict[str, Any]) -> list[int]:
    return [int(t["out"]) for t in perf["tokens"] if int(t["out"]) >= 0]


def reuse_lines(
    reuse: dict[str, Any], prefix: dict[str, Any], baseline: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """The prefill a restore saved, and what has to hold of the two runs.

    Everything printed is a PERF counter of one of the three runs.  The check
    that matters is the last one: a restored prefix is exact or it is a defect,
    so the ids generated after the restore are held to the ids the same prompt
    generates with no reuse at all, one for one.
    """
    pre_cycles, pre_tokens = _cycles(prefix, decode=False)
    new_cycles, new_tokens = _cycles(reuse, decode=False)
    base_cycles, base_tokens = _cycles(baseline, decode=False)
    saved = base_cycles - new_cycles
    share = 100.0 * saved / base_cycles if base_cycles else 0.0
    restored = int(reuse["run"]["prefix"]["restored"])

    def row(name: str, tokens: int, cycles: int) -> str:
        per = cycles // tokens if tokens else 0
        return f"    {name:<34}{tokens:>7}{cycles:>16,}{per:>15,}"

    lines = [f"    {'prefill pass':<34}{'tokens':>7}{'cycles':>16}{'cycles/token':>15}"]
    lines.append(row("the prefix, computed once", pre_tokens, pre_cycles))
    lines.append(row("the user turn, after the restore", new_tokens, new_cycles))
    lines.append(row("the whole prompt, no reuse", base_tokens, base_cycles))
    lines.append(f"    {'-' * 72}")
    lines.append(
        f"    the restore of {restored} positions leaves {saved:,} of those "
        f"{base_cycles:,} prefill cycles unspent, {share:.1f}% of them"
    )

    bad: list[str] = []
    got, want = _ids(reuse), _ids(baseline)
    if got == want and got:
        lines.append("")
        lines.append(
            f"    the {len(got)} ids after the restore are the {len(want)} the same prompt "
            f"generates with no reuse, one for one"
        )
    else:
        i = next(
            (k for k, (a, b) in enumerate(zip(want, got, strict=False)) if a != b),
            min(len(want), len(got)),
        )
        bad.append(
            f"prefix reuse changed a generated id: id {i} is "
            f"{got[i] if i < len(got) else 'missing'} after the restore and "
            f"{want[i] if i < len(want) else 'missing'} without reuse -- reuse is exact or it is "
            f"a defect"
        )
    if restored == 0:
        bad.append("the run being summarized restored no prefix")
    if new_tokens + restored != base_tokens:
        bad.append(
            f"the restore covered {restored} positions and the run prefilled {new_tokens}, "
            f"which is not the {base_tokens} the run without reuse prefilled"
        )
    if pre_cycles + new_cycles != base_cycles:
        bad.append(
            f"the prefix cost {pre_cycles:,} cycles and the user turn {new_cycles:,}, which is "
            f"not the {base_cycles:,} of the same positions without reuse"
        )
    return lines, bad


def prefix_lines(
    h: Header, ids: Sequence[int], *, turn: str, shared: int, refused: str
) -> list[str]:
    """What the saved file is, as its own header describes it, and what refuses it."""
    return [
        f"    file       {h.path},",
        f"               {h.bytes:,} B: a {h.kv_size:,} B KV region and the "
        f"{h.bytes - h.kv_size:,} B record of what it",
        f"               belongs to -- {h.model} at WB={h.wb}, MAX_CTX={h.max_ctx}, image "
        f"sha256 {h.image_sha256[:16]},",
        "               and the token id of every position it covers",
        f"    covers     {h.positions} of the prompt's {len(ids)} ids: {turn}",
        f"    shared     the same turn with a different question after it opens on the same "
        f"{shared} ids,",
        "               so what the file covers belongs to the conversation, not to the question",
        "    refused    the same file with one byte of that record changed, offered to the "
        "same run:",
        *(f"               {line}" for line in textwrap.wrap(refused, 62)),
    ]


# --------------------------------------------------------------------------- the demo


@dataclass
class Stage:
    """One stage of the demo and the seconds it took, as ``demo-report`` reads them."""

    name: str
    seconds: float
    note: str


@dataclass
class Demo:
    """The run: where everything goes, and the stage timings collected on the way."""

    image: Path
    work: Path
    stages: list[Stage] = field(default_factory=list)

    @property
    def prefix_file(self) -> Path:
        return self.work / "prefix.kv"

    def perf(self, name: str) -> Path:
        return self.work / f"{name}.json"

    def stage(self, name: str, note: str, fn: Callable[[], Any]) -> Any:
        print(f"\n=== {name}: {note}", flush=True)
        t0 = time.perf_counter()
        out = fn()
        self.stages.append(Stage(name, time.perf_counter() - t0, note))
        return out

    def reused(self, name: str, note: str) -> None:
        print(f"\n=== {name}: {note}", flush=True)
        self.stages.append(Stage(name, 0.0, note))

    def write_stages(self) -> Path:
        path = self.work / "stages.jsonl"
        path.write_text(
            "".join(
                json.dumps({"name": s.name, "seconds": round(s.seconds, 3), "note": s.note}) + "\n"
                for s in self.stages
            ),
            encoding="utf-8",
        )
        return path


def _run_harness(binary: Path, args: list[str], *, expect: int = 0, capture: bool = False) -> str:
    """One harness run.

    A prefill pass is minutes of work, so the harness keeps the terminal: its
    tokens, cycles and utilization print as it produces them, the way they do
    under ``make demo``.  ``capture`` takes the output instead, for the one
    short run whose message this module reads.
    """
    cmd = [str(binary), *args]
    print(f"$ {' '.join(cmd)}", flush=True)
    sys.stdout.flush()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=capture,
        check=False,
    )
    if proc.returncode != expect:
        raise RuntimeError(f"qcore_sim exited {proc.returncode}, expected {expect}")
    return proc.stdout if capture else ""


def _refusal_check(binary: Path, demo: Demo) -> str:
    """The saved file with one byte of its record changed, offered to the same run.

    The point of the record is that a file which does not belong to a run is
    refused rather than restored, so the demo shows the refusal: the image
    SHA-256 in the header is changed and nothing else, and the run has to stop
    on it before it executes a cycle.
    """
    path = demo.work / "prefix-wrong-image.kv"
    data = bytearray(demo.prefix_file.read_bytes())
    data[112] ^= 0x01  # the first character of the image SHA-256
    path.write_bytes(bytes(data))
    out = _run_harness(
        binary,
        ["--image", str(demo.image), "--kv-load", str(path), "--max-new", "0", "--quiet"],
        expect=2,
        capture=True,
    )
    mark = "does not belong to this run: "
    if mark not in out:
        raise RuntimeError(f"the harness refused {path}, but not for the reason it should have")
    path.unlink()
    return out[out.index(mark) + len(mark) :].strip().splitlines()[0]


def run(a: argparse.Namespace) -> int:
    """Compute a prefix on the hardware, restore it, and generate from the turn after it."""
    from quettos import cli
    from quettos.model import load_spec
    from quettos.tokenizer_io import load_prompt

    cfg = compare.CONFIGS[a.wb]
    print(
        f"demo-toolcall: {a.model} at WB={cfg.wb}, prompt {a.prompt}, "
        f"up to {a.max_new} decode steps"
    )
    note = f"uv run quettos download {a.model}"
    print(f"\n=== checkpoint: {note}", flush=True)
    t0 = time.perf_counter()
    spec = load_spec(a.model)
    image = Path(a.image) if a.image else compare.BUILD / "images" / f"{spec.name}-toolcall"
    work = compare.BUILD / "demo" / image.name
    work.mkdir(parents=True, exist_ok=True)
    demo = Demo(image, work)
    demo.stages.append(Stage("checkpoint", time.perf_counter() - t0, note))

    quant = compare.BUILD / "quant" / f"{spec.name}.npz"
    if quant.is_file() and not a.fresh:
        demo.reused("quantize", f"{quant} is already built (--fresh rebuilds it)")
    else:
        demo.stage(
            "quantize",
            f"uv run quettos quantize {a.model}",
            lambda: cli.main(["quantize", a.model]),
        )

    compiled = image / compiler.FILES["layout"]
    ready = compiled.is_file() and not a.fresh
    if ready:
        layout = compiler.load_layout(image)
        ready = int(layout["wb"]) == cfg.wb and compare.prompt_source(layout) == a.prompt
    if ready:
        demo.reused("compile", f"{image} is already compiled for WB={cfg.wb} and {a.prompt}")
    else:
        demo.stage(
            "compile",
            f"uv run quettos compile {a.model} --wb {cfg.wb} --prompt {a.prompt}",
            lambda: cli.main(
                ["compile", a.model, "--wb", str(cfg.wb), "--prompt", a.prompt, "--out", str(image)]
            ),
        )
    layout = compiler.load_layout(image)
    ids = compare.prompt_ids(image)
    cut = first_turn(ids, [int(v) for v in layout["model"].get("eos_ids") or []])
    turn = "the first turn, the system message"
    if load_prompt(REPO_ROOT / a.prompt).get("tools"):
        turn += " that carries the tool descriptions"

    binary = demo.stage(
        "harness",
        f"make -C sim/verilator build {' '.join(cfg.make_args)}",
        lambda: compare.harness_binary(cfg),
    )

    common = ["--image", str(image), "--lat", str(a.lat), "--bw-div", str(a.bw_div)]
    demo.stage(
        "prefix",
        f"prefill the first {cut} ids on qcore_top and save the KV region",
        lambda: _run_harness(
            binary,
            [
                *common,
                "--prefix-len",
                str(cut),
                "--max-new",
                "0",
                "--kv-save",
                str(demo.prefix_file),
                "--perf-json",
                str(demo.perf("prefix")),
            ],
        ),
    )
    header = read_header(demo.prefix_file)
    bad = check_header(header, layout, ids)
    if bad:
        print("\ndemo-toolcall: FAILED")
        for line in bad:
            print(f"  {demo.prefix_file} does not belong to {image}: {line}")
        return 1
    refusal = demo.stage(
        "refusal",
        "the same file with one byte of its record changed, offered to the same run",
        lambda: _refusal_check(binary, demo),
    )
    shared = shared_with_another_question(image, layout, OTHER_QUESTION)

    demo.stage(
        "reuse",
        f"restore the prefix and prefill the {len(ids) - 1 - cut} positions after it",
        lambda: _run_harness(
            binary,
            [
                *common,
                "--kv-load",
                str(demo.prefix_file),
                "--max-new",
                str(a.max_new),
                "--perf-json",
                str(demo.perf("reuse")),
            ],
        ),
    )
    demo.stage(
        "no reuse",
        f"the same prompt again, all {len(ids) - 1} prefill positions on the hardware",
        lambda: _run_harness(
            binary,
            [
                *common,
                "--max-new",
                str(a.max_new),
                "--perf-json",
                str(demo.perf("baseline")),
            ],
        ),
    )

    stages = demo.write_stages()
    return cli.main(
        [
            "demo-report",
            "--image",
            str(image),
            "--perf",
            str(demo.perf("reuse")),
            "--stages",
            str(stages),
            "--prefix-perf",
            str(demo.perf("prefix")),
            "--baseline-perf",
            str(demo.perf("baseline")),
            "--prefix-file",
            str(demo.prefix_file),
            "--prefix-turn",
            turn,
            "--prefix-shared",
            str(shared),
            "--prefix-refused",
            refusal,
        ]
    )


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The demo's options (``make demo-toolcall`` passes TOOLCALL_ARGS through)."""
    p.add_argument("--model", default="qwen", help="alias (qwen, smollm2) or a repo id")
    p.add_argument(
        "--prompt",
        default="prompts/tool_call_weather.json",
        help="the prompt file compiled into the image; its first turn is the prefix",
    )
    p.add_argument("--max-new", type=int, default=20, help="decode steps after the restore")
    p.add_argument("--wb", type=int, default=64, choices=sorted(compare.CONFIGS), help="port width")
    p.add_argument("--image", default=None, help="the compiled image (default build/images/...)")
    p.add_argument("--lat", type=int, default=32, help="QMEM read latency in cycles")
    p.add_argument("--bw-div", type=int, default=1, help="one returned beat every N cycles")
    p.add_argument("--fresh", action="store_true", help="quantize and compile again")
    return p


def main(argv: list[str] | None = None) -> int:
    p = add_arguments(
        argparse.ArgumentParser(
            prog="python -m quettos.prefix",
            description="prefix reuse on qcore_top: compute a prefix once, restore it, generate",
        )
    )
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
