"""pytest entries for qcore_seq_fetch.

``test_seq_fetch`` runs tb_seq_fetch on the tiny configuration (two beats per descriptor) and
``test_seq_fetch_wide`` on the demo width (two descriptors per beat, where the
leading-descriptor skip of an unaligned restart lives); ``test_seq_fetch_elaborates`` runs the
three parsers over the module in all three widths.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
import qc_runner

CONFIGS = {"tiny_w16": {"WB": 16}, "fpga_w64": {"WB": 64}, "sim_w128": {"WB": 128}}
SOURCES = ["qcore_pkg.sv", "qcore_seq_fetch.sv"]


def elaborate(top: str, params: dict[str, int], sources: list[str] | None = None) -> None:
    """Verilator, Yosys and Icarus over ``top`` with ``params``; zero warnings from all three."""
    for tool in ("verilator", "yosys", "iverilog"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} not on PATH")
    rtl = str(qc_runner.RTL)
    files = [str(qc_runner.RTL / s) for s in (sources or SOURCES)]

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(cmd, cwd=qc_runner.REPO, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"{' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}"
        return proc

    proc = run(
        ["verilator", "--lint-only", "-Wall", "-Wpedantic", "--top-module", top, f"-I{rtl}"]
        + [f"-G{k}={v}" for k, v in params.items()]
        + files
    )
    assert "%Warning" not in proc.stderr, proc.stderr
    chparam = " ".join(f"-set {k} {v}" for k, v in params.items())
    run(
        [
            "yosys",
            "-q",
            "-p",
            f"read_verilog -sv -defer -I{rtl} {' '.join(files)}; chparam {chparam} {top}; "
            f"hierarchy -check -top {top}; proc; opt; check -assert",
        ]
    )
    run(
        ["iverilog", "-g2012", f"-I{rtl}", "-s", top, "-o", "/dev/null"]
        + [f"-P{top}.{k}={v}" for k, v in params.items()]
        + files
    )


def test_seq_fetch() -> None:
    qc_runner.run(
        "qcore_seq_fetch",
        ["qcore_seq_fetch.sv"],
        "tb_seq_fetch",
        parameters={"WB": qc_runner.TINY["WB"], "DQ_DEPTH": 8},
    )


def test_seq_fetch_wide() -> None:
    qc_runner.run(
        "qcore_seq_fetch",
        ["qcore_seq_fetch.sv"],
        "tb_seq_fetch",
        parameters={"WB": 64, "DQ_DEPTH": 8},
    )


@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_seq_fetch_elaborates(cfg: str) -> None:
    elaborate("qcore_seq_fetch", CONFIGS[cfg])
