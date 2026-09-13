"""Prefix reuse: the file format, the boundary rule, the report, and the RTL run.

``sim/verilator/prefix.hpp`` writes the prefix file and refuses one that does
not belong to the run restoring it; :mod:`quettos.prefix` reads the same file
on this side and drives the demo.  The tests below hold the two to each other:
the format tests build a header with :mod:`struct` and read it back, and the
RTL test runs a tiny compiled model on ``qcore_top`` three ways -- prefix,
restore, and the same prompt with no reuse -- and holds the ids and the cycles
of the restored run to the run that recomputed everything.  The RTL tests skip
when Verilator is not on ``PATH``.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest
from quettos import compare, compiler, prefix, synthetic
from quettos.model import REPO_ROOT

TINY = compare.CONFIGS[16]

#: A prompt long enough to cut in two with positions left on both sides.
PROMPT = (1, 2, 3, 4, 5, 6)
CUT = 3


def needs_verilator() -> None:
    if shutil.which("verilator") is None:
        pytest.skip("verilator is not on PATH")


# --------------------------------------------------------------------------- the format


def pack(
    *,
    magic: bytes = prefix.MAGIC,
    version: int = prefix.VERSION,
    isa: int = 1,
    wb: int = 64,
    max_ctx: int = 2048,
    kv_base: int = 1024,
    kv_size: int = 256,
    model: str = "a-model",
    sha: str = "ab" * 32,
    ids: tuple[int, ...] = (11, 22, 33),
    header_bytes: int | None = None,
    payload: int | None = None,
) -> bytes:
    """A prefix file laid out as ``sim/verilator/prefix.hpp`` writes it."""
    n = prefix.FIXED_BYTES + 4 * len(ids) if header_bytes is None else header_bytes
    head = struct.pack(
        prefix.HEADER_FMT,
        magic,
        version,
        n,
        isa,
        wb,
        max_ctx,
        len(ids),
        kv_base,
        kv_size,
        model.encode(),
        sha.encode(),
    )
    return (
        head + struct.pack(f"<{len(ids)}I", *ids) + bytes(kv_size if payload is None else payload)
    )


def write(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "prefix.kv"
    path.write_bytes(data)
    return path


def test_header_round_trips(tmp_path) -> None:
    """Every field the harness writes comes back out of the file."""
    h = prefix.read_header(write(tmp_path, pack()))
    assert (h.version, h.isa_version, h.wb, h.max_ctx) == (prefix.VERSION, 1, 64, 2048)
    assert (h.kv_base, h.kv_size) == (1024, 256)
    assert (h.model, h.image_sha256) == ("a-model", "ab" * 32)
    assert h.ids == (11, 22, 33) and h.positions == 3
    assert h.bytes == prefix.FIXED_BYTES + 12 + 256


@pytest.mark.parametrize(
    ("kwargs", "says"),
    [
        ({"magic": b"QKVPRFX0"}, "magic"),
        ({"version": 2}, "version 2"),
        ({"header_bytes": 999}, "says its header is 999 B"),
        ({"payload": 255}, "is 443 B"),
        ({"payload": 257}, "is 445 B"),
    ],
)
def test_file_that_is_not_a_prefix_is_refused(tmp_path, kwargs, says) -> None:
    """The file is held to its own header before anything is read out of it."""
    with pytest.raises(prefix.PrefixError, match=says):
        prefix.read_header(write(tmp_path, pack(**kwargs)))


def test_short_file_is_refused(tmp_path) -> None:
    with pytest.raises(prefix.PrefixError, match="shorter than"):
        prefix.read_header(write(tmp_path, pack()[:100]))


#: A layout the header of :func:`pack` belongs to.
LAYOUT = {
    "isa_version": 1,
    "wb": 64,
    "max_ctx": 2048,
    "model": {"name": "a-model"},
    "image": {"sha256": "ab" * 32, "size": 1280},
    "bases": {"kv": 1024},
}


def test_file_that_belongs_to_the_run_passes(tmp_path) -> None:
    h = prefix.read_header(write(tmp_path, pack()))
    assert prefix.check_header(h, LAYOUT, [11, 22, 33, 44]) == []


@pytest.mark.parametrize(
    ("kwargs", "ids", "says"),
    [
        ({"isa": 2}, [11, 22, 33, 44], "ISA version 2"),
        ({"wb": 128}, [11, 22, 33, 44], "saved at WB=128"),
        ({"max_ctx": 1024}, [11, 22, 33, 44], "saved at MAX_CTX=1024"),
        ({"model": "other"}, [11, 22, 33, 44], "model other"),
        ({"sha": "cd" * 32}, [11, 22, 33, 44], "image sha256 cdcd"),
        ({"kv_base": 512}, [11, 22, 33, 44], "KV region 256 B at 512"),
        ({"ids": ()}, [11, 22], "covers no position"),
        ({}, [11, 22, 33], "covers 3 positions of a 3-id prompt"),
        ({}, [11, 99, 33, 44], "position 1 was token 22"),
    ],
)
def test_every_term_the_bytes_were_computed_under_is_checked(tmp_path, kwargs, ids, says) -> None:
    """The same list the harness refuses on, read back on this side."""
    h = prefix.read_header(write(tmp_path, pack(**kwargs)))
    bad = prefix.check_header(h, LAYOUT, ids)
    assert any(says in line for line in bad), bad


def test_python_and_c_headers_are_the_same_layout() -> None:
    """The constants this module unpacks with are the ones the harness writes."""
    source = (REPO_ROOT / "sim" / "verilator" / "prefix.hpp").read_text(encoding="utf-8")
    assert f"PREFIX_FIXED_BYTES = {prefix.FIXED_BYTES}u" in source
    assert f"PREFIX_VERSION = {prefix.VERSION}u" in source
    assert f"PREFIX_NAME_CHARS = {prefix.NAME_CHARS}u" in source
    letters = ", ".join(f"'{c}'" for c in prefix.MAGIC.decode())
    assert f"PREFIX_MAGIC[8] = {{{letters}}}" in source
    assert struct.calcsize(prefix.HEADER_FMT) == prefix.FIXED_BYTES


# --------------------------------------------------------------------------- the boundary


def test_prefix_is_the_first_turn() -> None:
    """The first end-of-turn id closes the system message, and the prefix is up to it."""
    assert prefix.first_turn([5, 6, 2, 7, 8, 2, 9], [2]) == 3
    assert prefix.first_turn([5, 2], [2, 3]) == 2
    assert prefix.first_turn([5, 3, 2], [2, 3]) == 2


def test_prompt_with_no_turn_end_has_no_prefix() -> None:
    with pytest.raises(prefix.PrefixError, match="no first turn"):
        prefix.first_turn([5, 6, 7], [2])


def test_tool_call_prompt_cuts_at_its_system_turn(qwen) -> None:
    """The prompt the demo runs: its first turn is the system message with the tools in it."""
    from quettos.tokenizer_io import encode, prompt_tokens, render_prompt

    ids = prompt_tokens(qwen, REPO_ROOT / "prompts" / "tool_call_weather.json")
    cut = prefix.first_turn(ids, qwen.eos_ids)
    text = render_prompt(qwen, REPO_ROOT / "prompts" / "tool_call_weather.json")
    head = text[: text.index("<|im_start|>user")]
    assert "<tools>" in head and "get_weather" in head
    # The text before the user's turn is the system turn, its end-of-turn id and
    # the newline after it, so the token-level rule cuts one id inside it.
    assert encode(qwen, head) == ids[: cut + 1]
    assert 0 < cut < len(ids)


# --------------------------------------------------------------------------- the report


def perf(passes: list[tuple[str, int, int]], *, restored: int = 0) -> dict:
    """A run record shaped like the one the harness writes: ``(pass, cycles, out)``."""
    return {
        "run": {"prefix": {"restored": restored}},
        "tokens": [
            {"pass": p, "cycles": c, "out": o, "pos": i} for i, (p, c, o) in enumerate(passes)
        ],
    }


def three_runs(reuse_ids=(7, 8)) -> tuple[dict, dict, dict]:
    """A prefix pass, a restored pass and the same prompt with no reuse, all consistent."""
    prefix_run = perf([("prefill", 10, -1), ("prefill", 11, -1)])
    reuse = perf(
        [("prefill", 12, -1), *[("decode", 20, i) for i in reuse_ids]],
        restored=2,
    )
    baseline = perf(
        [
            ("prefill", 10, -1),
            ("prefill", 11, -1),
            ("prefill", 12, -1),
            *[("decode", 20, i) for i in (7, 8)],
        ]
    )
    return reuse, prefix_run, baseline


def test_two_passes_add_up_to_the_run_without_reuse() -> None:
    lines, bad = prefix.reuse_lines(*three_runs())
    assert bad == []
    assert any("21" in line for line in lines)  # the prefix cost, 10 + 11
    assert any("63.6%" in line for line in lines)  # 21 of 33 prefill cycles unspent


def test_generated_id_that_moved_is_a_defect() -> None:
    """Reuse is exact or it is a defect: a differing id fails the summary."""
    _, bad = prefix.reuse_lines(*three_runs(reuse_ids=(7, 9)))
    assert any("prefix reuse changed a generated id" in line for line in bad)
    assert any("id 1 is 9 after the restore and 8 without reuse" in line for line in bad)


def test_run_that_restored_nothing_is_not_reuse() -> None:
    reuse, prefix_run, baseline = three_runs()
    reuse["run"]["prefix"]["restored"] = 0
    _, bad = prefix.reuse_lines(reuse, prefix_run, baseline)
    assert any("restored no prefix" in line for line in bad)


def test_two_passes_cover_the_same_positions() -> None:
    """A prefix pass that did not run the positions the restore skipped is caught."""
    reuse, prefix_run, baseline = three_runs()
    prefix_run["tokens"] = prefix_run["tokens"][:1]
    _, bad = prefix.reuse_lines(reuse, prefix_run, baseline)
    assert any("not the 33 of the same positions without reuse" in line for line in bad)


# --------------------------------------------------------------------------- on qcore_top


@pytest.fixture(scope="module")
def tiny_image(tmp_path_factory) -> Path:
    """A random tiny model compiled for the CI width, two tiles of context deep."""
    return compare.compile_shape(
        synthetic.SHAPES[0], 0, TINY, tmp_path_factory.mktemp("img-prefix"), max_ctx=2 * TINY.wb
    )


def run_harness(binary: Path, args: list[str], *, expect: int = 0) -> str:
    proc = subprocess.run(
        [str(binary), *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert proc.returncode == expect, proc.stdout + proc.stderr
    return proc.stdout + proc.stderr


@pytest.fixture(scope="module")
def restored(tiny_image, tmp_path_factory) -> tuple[dict, dict, dict, Path]:
    """One tiny model run three ways on ``qcore_top``: prefix, restore, and no reuse."""
    needs_verilator()
    binary = compare.harness_binary(TINY, quiet=True)
    out = tmp_path_factory.mktemp("prefix-run")
    ids = ",".join(str(i) for i in PROMPT)
    common = ["--image", str(tiny_image), "--prompt-ids", ids, "--eos", str(compare.EOS_OFF)]
    run_harness(
        binary,
        [
            *common,
            "--prefix-len",
            str(CUT),
            "--max-new",
            "0",
            "--quiet",
            "--kv-save",
            str(out / "p.kv"),
            "--perf-json",
            str(out / "prefix.json"),
        ],
    )
    run_harness(
        binary,
        [
            *common,
            "--max-new",
            "2",
            "--quiet",
            "--kv-load",
            str(out / "p.kv"),
            "--perf-json",
            str(out / "reuse.json"),
        ],
    )
    run_harness(
        binary,
        [*common, "--max-new", "2", "--quiet", "--perf-json", str(out / "baseline.json")],
    )
    read = lambda name: json.loads((out / name).read_text())  # noqa: E731
    return read("prefix.json"), read("reuse.json"), read("baseline.json"), out / "p.kv"


def test_restored_run_generates_the_same_ids(restored) -> None:
    """The property the demo exists to show: reuse is exact, id for id."""
    _, reuse, baseline, _ = restored
    got = [int(t["out"]) for t in reuse["tokens"] if int(t["out"]) >= 0]
    want = [int(t["out"]) for t in baseline["tokens"] if int(t["out"]) >= 0]
    assert got and got == want


def test_restore_skips_the_positions_the_prefix_covers(restored) -> None:
    """The restored run starts at the position after the file, and costs those cycles only."""
    prefix_run, reuse, baseline, _ = restored
    assert [t["pos"] for t in prefix_run["tokens"]] == list(range(CUT))
    assert [t["pos"] for t in reuse["tokens"]] == list(range(CUT, len(PROMPT) + 1))
    assert int(reuse["run"]["prefix"]["restored"]) == CUT
    cycles = lambda p: {t["pos"]: t["cycles"] for t in p["tokens"]}  # noqa: E731
    base = cycles(baseline)
    for pos, c in {**cycles(prefix_run), **cycles(reuse)}.items():
        assert c == base[pos], f"position {pos} cost {c} cycles, {base[pos]} without reuse"


def test_two_passes_cost_what_one_pass_costs(restored) -> None:
    prefix_run, reuse, baseline, _ = restored
    lines, bad = prefix.reuse_lines(reuse, prefix_run, baseline)
    assert bad == [], bad
    assert lines


def test_saved_file_belongs_to_its_image(restored, tiny_image) -> None:
    prefix_run, _, _, path = restored
    header = prefix.read_header(path)
    layout = compiler.load_layout(tiny_image)
    assert prefix.check_header(header, layout, list(PROMPT)) == []
    assert header.ids == PROMPT[:CUT]
    assert header.bytes == prefix.FIXED_BYTES + 4 * CUT + header.kv_size


@pytest.mark.parametrize(
    ("offset", "says"),
    [
        (0, "does not start with the prefix magic"),
        (20, "saved at WB="),
        (48, "model "),
        (112, "image sha256 "),
        (prefix.FIXED_BYTES, "position 0 was token "),
    ],
)
def test_harness_refuses_a_foreign_file(restored, tiny_image, tmp_path, offset, says) -> None:
    """One byte of the record changed, and the run stops before it executes a cycle."""
    needs_verilator()
    _, _, _, path = restored
    data = bytearray(path.read_bytes())
    data[offset] ^= 0x01
    bad = tmp_path / "bad.kv"
    bad.write_bytes(bytes(data))
    out = run_harness(
        compare.harness_binary(TINY, quiet=True),
        [
            "--image",
            str(tiny_image),
            "--prompt-ids",
            ",".join(str(i) for i in PROMPT),
            "--kv-load",
            str(bad),
            "--max-new",
            "0",
            "--quiet",
        ],
        expect=2,
    )
    assert says in out, out


def test_harness_refuses_a_prefix_from_another_prompt(restored, tiny_image) -> None:
    """The file carries the token at every position it covers, so another prompt is refused."""
    needs_verilator()
    _, _, _, path = restored
    out = run_harness(
        compare.harness_binary(TINY, quiet=True),
        [
            "--image",
            str(tiny_image),
            "--prompt-ids",
            "1,2,9,4,5,6",
            "--kv-load",
            str(path),
            "--max-new",
            "0",
            "--quiet",
        ],
        expect=2,
    )
    assert "position 2 was token 3, this prompt has 9" in out


def test_prefix_run_generates_nothing(tiny_image) -> None:
    """``--prefix-len`` computes a prefix; a run that also generates is not one."""
    needs_verilator()
    out = run_harness(
        compare.harness_binary(TINY, quiet=True),
        ["--image", str(tiny_image), "--prefix-len", "2", "--max-new", "1", "--quiet"],
        expect=2,
    )
    assert "--max-new 0" in out
