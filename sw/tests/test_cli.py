"""The demo's self-check: ``quettos demo-report`` and the record the harness writes for it.

The report is what turns a finished run into a verdict, so what it claims has
to be what it checked.  The tests here build a run record beside a compiled
directory and hold the report to it: every file ``layout.json`` records a
SHA-256 for against that hash, the prompt the run was given against the ids the
image was compiled with, and a run that generated nothing against nothing at
all.  The harness side -- the image, the prompt, the program each token ran and
the register names it writes and reads -- runs ``qcore_top`` on a random tiny
model and skips when Verilator is not on ``PATH``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest
from quettos import cli, compare, golden, isa, synthetic
from quettos.cli import main

REPO = Path(__file__).resolve().parents[2]

#: The checked-in record the fixture holds its ids to, and the prompt it is of.
RECORD = "models/smollm2-135m-instruct/expected_tokens.json"
SOURCE = "prompts/chat_short.json"

#: The two repository files a compiled layout names beside its own: the rope
#: table the compiler laid into the image and the lookup tables the numerics
#: were built from.
ROPE = "sw/quettos/tables/rope_theta1e5_2048.npy"
LUTS = "sw/quettos/tables/luts.json"

#: The tiny configuration and the shape the harness tests run.
TINY = compare.CONFIGS[16]
SHAPE = synthetic.SHAPES[0]


def recorded_ids() -> list[int]:
    report = json.loads((REPO / RECORD).read_text(encoding="utf-8"))
    return [int(v) for v in report["prompts"][SOURCE]["generated_ids"]]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tokens_bin(path: Path, table: dict[int, bytes]) -> int:
    """A ``tokens.bin`` as ``quettos.tokenizer_io`` writes one: u32 count, u16 length, bytes."""
    count = max(table) + 1
    chunks = [struct.pack("<I", count)]
    for i in range(count):
        b = table.get(i, b"")
        chunks.append(struct.pack("<H", len(b)) + b)
    data = b"".join(chunks)
    path.write_bytes(data)
    return len(data)


# --------------------------------------------------------------------------- a run to report on


#: The text the fixture's token table turns the recorded ids into, word for
#: word: what a demo of this model prints.
TEXT = ("The", " capital", " of", " France", " is", " Paris", ".", "<|im_end|>")


def write_image(directory: Path, prompt: list[int]) -> Path:
    """A compiled directory as ``layout.json`` describes one: every file it hashes.

    The compiler records a SHA-256 for the image, both descriptor programs, the
    dump plan, the token table, the prompt, the rope table, the lookup tables
    and the golden model's recorded continuation, so all nine are here -- the
    six written into the directory and the three the layout names from the
    repository root.
    """
    directory.mkdir(parents=True, exist_ok=True)
    blob = bytes((7 * i + 11) % 256 for i in range(4096))
    (directory / "image.bin").write_bytes(blob)
    programs = {
        "decode": bytes((3 * i + 1) % 256 for i in range(32)),
        "prefill": bytes((5 * i + 2) % 256 for i in range(32)),
    }
    for name, data in programs.items():
        (directory / f"{name}.prog").write_bytes(data)
    plan = {"format": "quettos-dump-plan", "isa_version": 1, "decode": [], "prefill": []}
    (directory / "dump_plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    ids = recorded_ids()
    nbytes = write_tokens_bin(
        directory / "tokens.bin",
        {i: t.encode("utf-8") for i, t in zip(ids, TEXT, strict=True)},
    )
    (directory / "prompt.tokens").write_text("".join(f"{i}\n" for i in prompt), encoding="utf-8")
    record = json.loads((REPO / RECORD).read_text(encoding="utf-8"))
    layout = {
        "format": "quettos-layout",
        "isa_version": 1,
        "wb": 64,
        "max_ctx": 2048,
        "a_bits": 16,
        "model": {
            "name": "smollm2-135m-instruct",
            "repo_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "layers": 30,
            "hidden": 576,
            "vocab": 49152,
        },
        "image": {
            "file": "image.bin",
            "size": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        },
        "programs": {
            "decode": {
                "file": "decode.prog",
                "addr": 0,
                "size": 32,
                "descriptors": 1475,
                "sha256": hashlib.sha256(programs["decode"]).hexdigest(),
            },
            "prefill": {
                "file": "prefill.prog",
                "addr": 64,
                "size": 32,
                "descriptors": 1472,
                "sha256": hashlib.sha256(programs["prefill"]).hexdigest(),
            },
        },
        "dump_plan": {
            "file": "dump_plan.json",
            "sha256": sha256_file(directory / "dump_plan.json"),
        },
        "tables": {
            "rope": {"file": ROPE, "rows": 2048, "sha256": sha256_file(REPO / ROPE)},
            "luts": {"file": LUTS, "sha256": sha256_file(REPO / LUTS)},
        },
        "tokens_bin": {
            "file": "tokens.bin",
            "bytes": nbytes,
            "count": max(ids) + 1,
            "sha256": sha256_file(directory / "tokens.bin"),
        },
        "prompt": {
            "file": "prompt.tokens",
            "source": SOURCE,
            "count": len(prompt),
            "sha256": golden.ids_sha256(prompt),
        },
        "expected_tokens": {
            "file": RECORD,
            "sha256": sha256_file(REPO / RECORD),
            "prompts": {
                key: {"count": len(v["generated_ids"]), "sha256": v["sha256"]}
                for key, v in record["prompts"].items()
            },
        },
    }
    (directory / "layout.json").write_text(json.dumps(layout, indent=1), encoding="utf-8")
    return directory


def token_records(image: Path, prompt: list[int], generated: list[int]) -> list[dict[str, Any]]:
    """One record per token the loop ran: the prompt prefilled, then the ids generated."""
    layout = json.loads((image / "layout.json").read_text(encoding="utf-8"))
    n_pre = int(layout["programs"]["prefill"]["descriptors"])
    n_dec = int(layout["programs"]["decode"]["descriptors"])
    tokens: list[dict[str, Any]] = []
    for i, tok in enumerate(prompt[:-1]):
        tokens.append(
            {
                "index": i,
                "pass": "prefill",
                "pos": i,
                "in": tok,
                "out": -1,
                "cycles": 1000,
                "mac_active": 800,
                "rd_bytes": 64000,
                "descriptors": n_pre,
            }
        )
    for j, tok in enumerate(generated):
        tokens.append(
            {
                "index": len(tokens),
                "pass": "decode",
                "pos": len(prompt) - 1 + j,
                "in": prompt[-1] if j == 0 else generated[j - 1],
                "out": tok,
                "cycles": 2000,
                "mac_active": 1600,
                "rd_bytes": 128000,
                "descriptors": n_dec,
            }
        )
    return tokens


def write_perf(
    path: Path,
    image: Path,
    prompt: list[int],
    generated: list[int],
    tokens: list[dict[str, Any]] | None = None,
) -> Path:
    """The record the harness writes: counters that add up over the tokens the run ran."""
    layout = json.loads((image / "layout.json").read_text(encoding="utf-8"))
    if tokens is None:
        tokens = token_records(image, prompt, generated)
    cycles = sum(int(t["cycles"]) for t in tokens)
    mac = sum(int(t["mac_active"]) for t in tokens)
    rd_bytes = sum(int(t["rd_bytes"]) for t in tokens)
    counters = {
        "CYCLES": cycles,
        "BUSY": cycles,
        "MAC_ACTIVE": mac,
        "STALL_MEM": cycles - mac,
        "STALL_VPU": 0,
        "STALL_KV": 0,
        "STALL_SEQ": 0,
        "STALL_DRAIN": 0,
        "RD_BEATS": rd_bytes // 64,
        "RD_BYTES": rd_bytes,
        "WT_BYTES": 12345,
        "WR_BEATS": 7,
        "WR_BYTES": 448,
        "MACS": 6789,
        "DESCRIPTORS": sum(int(t["descriptors"]) for t in tokens),
        "FETCH_BEATS": 11,
    }
    perf = {
        "format": "quettos-perf",
        "isa_version": 1,
        "model": layout["model"]["name"],
        "mode": "token",
        "build": {
            "top": "qcore_top",
            "wb": 64,
            "b_max": 1,
            "vl": 4,
            "vsram_words": 4096,
            "fifo_beats": 128,
            "acc_w": 40,
            "threads": 1,
        },
        "run": {
            "lat": 32,
            "bw_div": 1,
            "max_new": len(generated),
            "step": False,
            "status": "ok",
            "clock_cycles": cycles + 48 * len(tokens),
            "wall_seconds": 1.5,
            "mcycles_per_s": 2.7,
            "image": str((image / "image.bin").resolve()),
            "image_bytes": (image / "image.bin").stat().st_size,
            "prompt": {"from": str(image / "prompt.tokens"), "count": len(prompt), "ids": prompt},
        },
        "counters": counters,
        "events": {"SAT_REQ": 0, "SAT_VPU": 0, "ERR_SHIFT": 0, "ERR_BOUNDS": 0},
        "memory": {
            "rd_requests": 5,
            "rd_beats": counters["RD_BEATS"],
            "rd_bytes": rd_bytes,
            "wr_beats": counters["WR_BEATS"],
            "wr_bytes": counters["WR_BYTES"],
        },
        "tokens": tokens,
    }
    path.write_text(json.dumps(perf, indent=1), encoding="utf-8")
    return path


@pytest.fixture
def run_dir(tmp_path: Path) -> tuple[Path, Path]:
    """A compiled directory and the record of a clean run of it, both as they are written."""
    prompt = list(range(1, 38))
    image = write_image(tmp_path / "image", prompt)
    perf = write_perf(tmp_path / "perf.json", image, prompt, recorded_ids())
    return image, perf


def report(image: Path, perf: Path, *extra: str) -> int:
    return main(["demo-report", "--image", str(image), "--perf", str(perf), *extra])


def edit(path: Path, change) -> None:
    """Rewrite a JSON file through ``change``."""
    d = json.loads(path.read_text(encoding="utf-8"))
    change(d)
    path.write_text(json.dumps(d, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- the image


def test_clean_run_reports_ok(run_dir, capsys) -> None:
    """The fixture is a run the report accepts, so every failure below is the change it names."""
    image, perf = run_dir
    assert report(image, perf) == 0
    out = capsys.readouterr().out
    assert "demo: OK" in out
    assert f"{len(recorded_ids())} ids matching the recorded reference" in out
    assert (
        f"sha256 {json.loads((image / 'layout.json').read_text())['image']['sha256'][:16]}" in out
    )
    assert f"files      {len(HASHED)} of the {len(HASHED)} files" in out
    assert "'The capital of France is Paris.<|im_end|>'" in out


# --------------------------------------------------------------------------- every hashed file


#: Every entry of ``layout.json`` the compiler records a SHA-256 in, by the
#: dotted path to it (``sw/quettos/compiler.py``, the layout object).
HASHED = (
    "image",
    "programs.decode",
    "programs.prefill",
    "dump_plan",
    "tables.rope",
    "tables.luts",
    "tokens_bin",
    "prompt",
    "expected_tokens",
)

#: The six of them the compiler writes into the compiled directory; the other
#: three are repository files the layout names from the root.
IN_IMAGE = ("image", "programs.decode", "programs.prefill", "dump_plan", "tokens_bin", "prompt")


def entry_of(layout: dict[str, Any], dotted: str) -> dict[str, Any]:
    node: Any = layout
    for key in dotted.split("."):
        node = node[key]
    return node


def test_report_finds_every_hashed_entry(run_dir) -> None:
    """What the report walks for and what the compiler writes are the same set."""
    image, _ = run_dir
    layout = json.loads((image / "layout.json").read_text(encoding="utf-8"))
    assert {name for name, _ in cli._layout_hashed(layout)} == set(HASHED)


@pytest.mark.parametrize("dotted", HASHED)
def test_changed_file_fails_the_report(run_dir, capsys, dotted) -> None:
    """Every file the layout carries a hash for is hashed, not the image alone."""
    image, perf = run_dir
    layout = json.loads((image / "layout.json").read_text(encoding="utf-8"))
    named = entry_of(layout, dotted)["file"]
    edit(image / "layout.json", lambda d: entry_of(d, dotted).__setitem__("sha256", "0" * 64))
    assert report(image, perf) == 1
    out = capsys.readouterr().out
    assert named in out and "hashes to" in out


@pytest.mark.parametrize("dotted", IN_IMAGE)
def test_missing_file_fails_the_report(run_dir, capsys, dotted) -> None:
    """A missing file is reported as missing rather than passed over."""
    image, perf = run_dir
    layout = json.loads((image / "layout.json").read_text(encoding="utf-8"))
    (image / entry_of(layout, dotted)["file"]).unlink()
    assert report(image, perf) == 1
    assert "is not there" in capsys.readouterr().out


def test_doctored_token_table_fails_the_report(run_dir, capsys) -> None:
    """The table turns the hardware's ids into the sentence: a doctored one is a failed run.

    The ids are the hardware's either way, so the only thing a rewritten table
    moves is the text beside them -- which is why the file is hashed.
    """
    image, perf = run_dir
    ids = recorded_ids()
    doctored = dict(zip(ids, [t.encode("utf-8") for t in TEXT], strict=True))
    doctored[ids[5]] = b" Berlin"
    write_tokens_bin(image / "tokens.bin", doctored)
    assert report(image, perf) == 1
    out = capsys.readouterr().out
    assert "'The capital of France is Berlin.<|im_end|>'" in out, "the doctored headline"
    assert "tokens.bin hashes to" in out and "demo: FAILED" in out


def test_prompt_file_of_other_ids_fails_the_report(run_dir, capsys) -> None:
    """The prompt entry hashes the ids the file carries, which is the digest the compiler took."""
    image, perf = run_dir
    ids = [int(v) for v in (image / "prompt.tokens").read_text().split()]
    (image / "prompt.tokens").write_text("".join(f"{i + 1}\n" for i in ids), encoding="utf-8")
    assert report(image, perf) == 1
    assert "prompt.tokens hashes to" in capsys.readouterr().out


def test_prompt_file_that_is_not_ids_fails_the_report(run_dir, capsys) -> None:
    """A file the digest cannot be taken of ends in a verdict, not in a traceback."""
    image, perf = run_dir
    (image / "prompt.tokens").write_text("thirty-seven\n", encoding="utf-8")
    assert report(image, perf) == 1
    assert "is not one token id per line" in capsys.readouterr().out


def test_one_corrupted_byte_fails_the_report(run_dir, capsys) -> None:
    """The image the hardware executed is held to the SHA-256 layout.json carries."""
    image, perf = run_dir
    blob = bytearray((image / "image.bin").read_bytes())
    blob[1234] ^= 0x01
    (image / "image.bin").write_bytes(bytes(blob))
    assert report(image, perf) == 1
    out = capsys.readouterr().out
    assert "hashes to" in out and "layout.json describes" in out


def test_truncated_image_fails_the_report(run_dir, capsys) -> None:
    """A file of the wrong size is caught before its hash is."""
    image, perf = run_dir
    blob = (image / "image.bin").read_bytes()
    (image / "image.bin").write_bytes(blob[:-64])
    assert report(image, perf) == 1
    assert "layout.json describes" in capsys.readouterr().out


def test_unexecuted_image_fails_the_report(run_dir, capsys) -> None:
    """The report holds the run's own record of what it mapped to the directory it is given."""
    image, perf = run_dir
    edit(perf, lambda d: d["run"].__setitem__("image", "/tmp/somewhere-else/image.bin"))
    assert report(image, perf) == 1
    assert "the hardware executed /tmp/somewhere-else/image.bin" in capsys.readouterr().out


def test_run_record_without_an_image_fails_the_report(run_dir, capsys) -> None:
    """A record that does not say which image ran cannot be reported on."""
    image, perf = run_dir
    edit(perf, lambda d: d["run"].pop("image"))
    assert report(image, perf) == 1
    assert "does not name the image the hardware executed" in capsys.readouterr().out


# --------------------------------------------------------------------------- the generation


def test_empty_generation_fails_the_report(run_dir, capsys) -> None:
    """A run that generated nothing is a failure, not a match against an empty list."""
    image, perf = run_dir
    write_perf(perf, image, list(range(1, 38)), [])
    assert report(image, perf) == 1
    out = capsys.readouterr().out
    assert "generated no ids" in out
    assert "identical to" not in out


def test_empty_generation_fails_without_a_reference(run_dir, capsys) -> None:
    """``--no-reference`` drops the record, not the requirement that the run produced ids."""
    image, perf = run_dir
    write_perf(perf, image, list(range(1, 38)), [])
    assert report(image, perf, "--no-reference") == 1
    assert "generated no ids" in capsys.readouterr().out


def test_wrong_id_fails_the_report(run_dir, capsys) -> None:
    """The positive control's counterpart: an id the record does not carry."""
    image, perf = run_dir
    ids = recorded_ids()
    write_perf(perf, image, list(range(1, 38)), [ids[0] + 1, *ids[1:]])
    assert report(image, perf) == 1
    assert "MISMATCH" in capsys.readouterr().out


# --------------------------------------------------------------------------- the prompt


def test_uncompiled_prompt_fails_the_report(run_dir, capsys) -> None:
    """A run given other ids is neither labelled with the image's prompt nor held to its record."""
    image, perf = run_dir
    other = list(range(2, 39))
    write_perf(perf, image, other, recorded_ids())
    assert report(image, perf) == 1
    out = capsys.readouterr().out
    assert f"not the 37 of prompt.tokens, the ids of {SOURCE}" in out
    assert f"identical to {RECORD}" not in out


def test_prompt_of_another_length_fails_the_report(run_dir, capsys) -> None:
    """The count is part of the record: a shorter prompt is a different prompt."""
    image, perf = run_dir
    write_perf(perf, image, list(range(1, 30)), recorded_ids())
    assert report(image, perf) == 1
    assert "not the 37 of prompt.tokens" in capsys.readouterr().out


def test_run_record_without_a_prompt_fails_the_report(run_dir, capsys) -> None:
    """A record that does not carry its prompt cannot be labelled with one."""
    image, perf = run_dir
    edit(perf, lambda d: d["run"].pop("prompt"))
    assert report(image, perf) == 1
    assert "does not carry the prompt the hardware was given" in capsys.readouterr().out


def test_unprefilled_prompt_fails_the_report(run_dir, capsys) -> None:
    """The prompt record and the token records are the same run: they say the same length."""
    image, perf = run_dir
    prompt, ids = list(range(1, 38)), recorded_ids()
    short = token_records(image, prompt, ids)[1:]
    write_perf(perf, image, prompt, ids, tokens=short)
    assert report(image, perf) == 1
    assert "prefilled 36 ids of its 37-id prompt" in capsys.readouterr().out


# --------------------------------------------------------------------------- a file it cannot read


def test_perf_record_that_is_not_json_ends_in_a_verdict(run_dir, capsys) -> None:
    """The report's job is a verdict, so a file it cannot parse is one too."""
    image, perf = run_dir
    perf.write_text("{ this is not json", encoding="utf-8")
    assert report(image, perf) == 1
    out = capsys.readouterr().out
    assert "demo: FAILED" in out
    assert f"{perf}, the record of the run, is not JSON" in out


def test_layout_that_is_not_json_ends_in_a_verdict(run_dir, capsys) -> None:
    image, perf = run_dir
    (image / "layout.json").write_text("[1, 2", encoding="utf-8")
    assert report(image, perf) == 1
    assert "the compiled layout, is not JSON" in capsys.readouterr().out


def test_missing_perf_record_ends_in_a_verdict(run_dir, capsys) -> None:
    image, perf = run_dir
    perf.unlink()
    assert report(image, perf) == 1
    assert "cannot be read: No such file or directory" in capsys.readouterr().out


def test_perf_record_missing_a_field_ends_in_a_verdict(run_dir, capsys) -> None:
    """Valid JSON that is not the record the harness writes is reported, not raised."""
    image, perf = run_dir
    edit(perf, lambda d: d.pop("counters"))
    assert report(image, perf) == 1
    out = capsys.readouterr().out
    assert "demo: FAILED" in out and "KeyError('counters')" in out


def test_stage_line_that_is_not_a_timing_ends_in_a_verdict(run_dir, capsys, tmp_path) -> None:
    """The stage timings are an input file like the other two."""
    image, perf = run_dir
    stages = tmp_path / "stages.jsonl"
    stages.write_text('{"name": "compile", "seconds": 0.5}\nnot a stage\n', encoding="utf-8")
    assert (
        main(["demo-report", "--image", str(image), "--perf", str(perf), "--stages", str(stages)])
        == 1
    )
    assert f"{stages} line 2 is not a stage timing" in capsys.readouterr().out


# --------------------------------------------------------------------------- the program that ran


def test_faulted_token_is_held_to_the_decode_program(run_dir, capsys) -> None:
    """A token that faulted has no id, and it is still a decode token.

    Its ``out`` is -1, like a prefill token's, so the program it is held to is
    the one the record names rather than the one its id would imply.
    """
    image, perf = run_dir
    prompt, ids = list(range(1, 38)), recorded_ids()
    tokens = token_records(image, prompt, ids)
    tokens[-1] = {**tokens[-1], "out": -1, "descriptors": 3}
    write_perf(perf, image, prompt, ids[:-1], tokens=tokens)
    assert report(image, perf) == 1
    out = capsys.readouterr().out
    assert "retired 3 descriptors, not the program's 1,475" in out, "decode.prog's count"
    assert "not the program's 1,472" not in out, "prefill.prog's is the other program's count"
    assert "prefilled" not in out, "the token that faulted ran decode.prog, not prefill.prog"


# --------------------------------------------------------------------------- the harness itself


def needs_verilator() -> None:
    if shutil.which("verilator") is None:
        pytest.skip("verilator is not on PATH")


@pytest.fixture(scope="module")
def tiny_image(tmp_path_factory) -> Path:
    """A random tiny model compiled for the CI width."""
    needs_verilator()
    return compare.compile_shape(SHAPE, 1, TINY, tmp_path_factory.mktemp("cli-w16"))


@pytest.fixture(scope="module")
def harness() -> Path:
    needs_verilator()
    return compare.harness_binary(TINY, quiet=True)


def run_harness(binary: Path, image: Path, out: Path, *args: str) -> subprocess.CompletedProcess:
    cmd = [str(binary), "--image", str(image), "--perf-json", str(out), *args]
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


def test_run_records_the_image_and_the_prompt(harness, tiny_image, tmp_path) -> None:
    """perf.json says which image the hardware executed and which ids it was given."""
    out = tmp_path / "perf.json"
    proc = run_harness(harness, tiny_image, out, "--max-new", "1", "--prompt-ids", "3,5", "--quiet")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    run = json.loads(out.read_text())["run"]
    image_bin = tiny_image / "image.bin"
    assert Path(run["image"]) == image_bin.resolve()
    assert run["image_bytes"] == image_bin.stat().st_size
    assert run["prompt"]["ids"] == [3, 5]
    assert run["prompt"]["count"] == 2
    assert run["prompt"]["from"] == "--prompt-ids"


def test_image_the_layout_does_not_describe_stops_the_run(harness, tiny_image, tmp_path) -> None:
    """A truncated image.bin is refused before the first cycle."""
    copy = tmp_path / "image"
    shutil.copytree(tiny_image, copy)
    blob = (copy / "image.bin").read_bytes()
    (copy / "image.bin").write_bytes(blob[:-64])
    proc = run_harness(harness, copy, tmp_path / "perf.json", "--max-new", "1", "--prompt-ids", "3")
    assert proc.returncode == 2
    assert "layout.json describes" in proc.stderr


def test_unknown_register_name_fails_the_dump(harness, tiny_image, tmp_path) -> None:
    """A dump plan naming a register the CSR window does not carry stops the run."""
    copy = tmp_path / "image"
    shutil.copytree(tiny_image, copy)
    plan = json.loads((copy / "dump_plan.json").read_text())
    named = 0
    for entry in plan["decode"]:
        if entry["csr"]:
            entry["csr"] = ["ARGMAX_TOKEN"]
            named += 1
    assert named > 0, "the decode program dumps at least one CSR"
    (copy / "dump_plan.json").write_text(json.dumps(plan))
    proc = run_harness(
        harness,
        copy,
        tmp_path / "perf.json",
        "--max-new",
        "1",
        "--prompt-ids",
        "3",
        "--step",
        "--dump-ops",
        "--dump-dir",
        str(tmp_path / "steps"),
    )
    assert proc.returncode == 2
    assert 'names "ARGMAX_TOKEN"' in proc.stderr


def test_run_records_the_program_every_token_ran(harness, tiny_image, tmp_path) -> None:
    """Each token record names its own program, which is what the report groups and checks by."""
    out = tmp_path / "perf.json"
    proc = run_harness(harness, tiny_image, out, "--max-new", "2", "--prompt-ids", "3,5", "--quiet")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    tokens = json.loads(out.read_text())["tokens"]
    assert [t["pass"] for t in tokens] == ["prefill", "decode", "decode"]


def test_faulted_token_is_held_to_the_counters(harness, tiny_image, tmp_path) -> None:
    """A token that faulted is held to the traffic model and the bucket sum like every other one.

    The opcode byte of the second descriptor of ``decode.prog`` is set to a
    value the ISA does not define, so the core stops the program with
    ``FAULT = OPCODE`` partway through the only token of the run.
    """
    copy = tmp_path / "image"
    shutil.copytree(tiny_image, copy)
    layout = json.loads((copy / "layout.json").read_text())
    at = int(layout["programs"]["decode"]["addr"]) + isa.DESC_BYTES
    blob = bytearray((copy / "image.bin").read_bytes())
    blob[at] = 0x7F
    assert not isa.is_opcode(blob[at])
    (copy / "image.bin").write_bytes(bytes(blob))
    proc = run_harness(
        harness, copy, tmp_path / "perf.json", "--max-new", "1", "--prompt-ids", "3", "--quiet"
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "FAULT=OPCODE" in proc.stdout
    assert "descriptors, not the program's" in proc.stdout, "the counters were checked"
    tokens = json.loads((tmp_path / "perf.json").read_text())["tokens"]
    assert [(t["pass"], t["out"]) for t in tokens] == [("decode", -1)]


def test_descriptor_count_disagreement_fails_the_run(harness, tiny_image, tmp_path) -> None:
    """A token that did not retire its program's descriptors is reported, not passed over."""
    copy = tmp_path / "image"
    shutil.copytree(tiny_image, copy)
    layout = json.loads((copy / "layout.json").read_text())
    layout["programs"]["prefill"]["descriptors"] += 1
    (copy / "layout.json").write_text(json.dumps(layout))
    proc = run_harness(
        harness, copy, tmp_path / "perf.json", "--max-new", "1", "--prompt-ids", "3,5", "--quiet"
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "descriptors, not the program's" in proc.stdout
