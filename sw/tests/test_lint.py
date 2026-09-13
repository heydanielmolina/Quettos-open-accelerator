"""``scripts/lint.sh``: what it does with Icarus's output.

Elaborating an ``always_comb`` that reads a constant part-select, Icarus 13
prints ``sorry: constant selects in always_* processes are not fully supported
(the process will be sensitive to all bits in '<signal>')`` -- a note about its
own implicit sensitivity list, not about the design.  The lint counts those in
one line and fails on every other line Icarus prints, a ``warning:`` included,
which on its own leaves the Icarus exit status at 0.  The classification is
driven here with stub tools; the last test runs the real three over one top and
checks that a reader watching the command sees no diagnostic on screen.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LINT = REPO_ROOT / "scripts" / "lint.sh"
REAL_TOP = "qcore_vpu_top"  # the module all but three of the notes come from
TOOLS = ("verilator", "yosys", "iverilog")

#: The note itself, as Icarus writes it.
NOTE = (
    "rtl/qcore_pkg.sv:{line}: sorry: constant selects in always_* processes are not fully "
    "supported (the process will be sensitive to all bits in 'x[48:0]')."
)


def _tree(tmp_path: Path, iverilog: str) -> dict[str, str]:
    """One trivial module, an empty wrapper directory, and stub tools around them.

    ``iverilog`` is the body of the stub Icarus: the lint's own reading of what
    that prints is what these tests are about, so Verilator and Yosys are
    silent successes.
    """
    rtl = tmp_path / "rtl"
    rtl.mkdir()
    (rtl / "stub_mod.sv").write_text("module stub_mod;\nendmodule\n", encoding="utf-8")
    wrap = tmp_path / "wrappers"
    wrap.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("verilator", "exit 0\n"), ("yosys", "exit 0\n"), ("iverilog", iverilog)):
        tool = bin_dir / name
        tool.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        tool.chmod(0o755)
    return {
        "RTL_DIR": str(rtl),
        "WRAP_DIR": str(wrap),
        "VERILATOR": str(bin_dir / "verilator"),
        "YOSYS": str(bin_dir / "yosys"),
        "IVERILOG": str(bin_dir / "iverilog"),
    }


def _run(env: dict[str, str], **extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LINT)],
        cwd=REPO_ROOT,
        env={**os.environ, **env, **extra},
        capture_output=True,
        text=True,
        check=False,
    )


def _echo(*lines: str) -> str:
    return "".join(f'echo "{line}"\n' for line in lines) + "exit 0\n"


def test_notes_are_counted_and_never_printed(tmp_path) -> None:
    """Three notes go in, one summary line comes out, and the run is OK."""
    env = _tree(
        tmp_path, _echo(NOTE.format(line=277), NOTE.format(line=285), NOTE.format(line=290))
    )
    r = _run(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "sorry:" not in r.stdout and "sorry:" not in r.stderr
    assert (
        "lint: iverilog: 3 sensitivity-list note(s) at 3 select(s) in 1 file(s), no warnings"
        in r.stdout
    )
    assert "LINT_ICARUS_NOTES=1 prints them" in r.stdout
    assert r.stdout.rstrip().endswith("lint: OK")


def test_notes_are_printed_on_request(tmp_path) -> None:
    """LINT_ICARUS_NOTES=1 prints them, once each: they repeat across tops."""
    env = _tree(
        tmp_path, _echo(NOTE.format(line=277), NOTE.format(line=277), NOTE.format(line=290))
    )
    r = _run(env, LINT_ICARUS_NOTES="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "3 sensitivity-list note(s) at 2 select(s) in 1 file(s)" in r.stdout
    assert r.stdout.count(NOTE.format(line=277)) == 1
    assert r.stdout.count(NOTE.format(line=290)) == 1


def test_warning_fails_the_lint_though_icarus_exits_zero(tmp_path) -> None:
    """The line the count exists to keep visible: anything that is not the known note."""
    warning = "rtl/qcore_top.sv:9: warning: @* is sensitive to all bits in 'q[7:0]'."
    env = _tree(tmp_path, _echo(NOTE.format(line=277), warning))
    r = _run(env)
    assert r.returncode == 1
    assert warning in r.stdout
    assert "stub_mod: iverilog reported:" in r.stdout
    assert "no warnings" not in r.stdout  # the summary does not claim what it just contradicted
    assert "lint: FAILED" in r.stdout


def test_note_of_another_shape_is_not_swallowed(tmp_path) -> None:
    """Only the sensitivity-list note is known; a different `sorry:` is a finding."""
    other = "rtl/qcore_top.sv:9: sorry: constant user defined functions are not supported."
    env = _tree(tmp_path, _echo(other))
    r = _run(env)
    assert r.returncode == 1 and other in r.stdout


def test_failing_icarus_fails_with_its_output(tmp_path) -> None:
    env = _tree(tmp_path, 'echo "rtl/stub_mod.sv:1: error: no such module."\nexit 1\n')
    r = _run(env)
    assert r.returncode == 1
    assert "stub_mod: iverilog FAILED" in r.stdout and "no such module" in r.stdout


@pytest.mark.skipif(
    any(shutil.which(t) is None for t in TOOLS), reason="verilator / yosys / iverilog not on PATH"
)
def test_real_run_prints_no_diagnostic(tmp_path) -> None:
    """The claim as a reader checks it: the three parsers over one top, nothing on screen."""
    r = _run({}, TOPS=REAL_TOP)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "sorry:" not in r.stdout and "sorry:" not in r.stderr
    screen = (r.stdout + r.stderr).lower()
    assert "warning:" not in screen and "%warning" not in screen
    assert "error" not in screen
    assert ", no warnings" in r.stdout
    assert r.stdout.rstrip().endswith("lint: OK")
    notes = _run({}, TOPS=REAL_TOP, LINT_ICARUS_NOTES="1")
    assert notes.returncode == 0
    assert notes.stdout.count("sorry: constant selects in always_* processes") > 0
