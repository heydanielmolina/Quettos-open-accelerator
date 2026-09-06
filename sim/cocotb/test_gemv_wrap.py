"""pytest entries for sim/cocotb/wrappers/qcore_gemv_wrap.sv: the GEMV path assembled.

``test_gemv_wrap_elaborates`` runs the three parsers over the wrapper in the tiny,
fpga_w64 and sim_w128 configurations (elaboration only, zero Verilator warnings);
``test_gemv_wrap`` builds it on the tiny configuration and runs tb_gemv_wrap.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import qc_runner

WRAP = Path(__file__).resolve().parent / "wrappers" / "qcore_gemv_wrap.sv"
SOURCES = [
    "qcore_vsram.sv",
    "qcore_mac_lane_group.sv",
    "qcore_row.sv",
    "qcore_mem_arb.sv",
    "qcore_stream_ctrl.sv",
    "qcore_requant.sv",
]
CONFIGS = {
    "tiny_w16": {"WB": 16, "B_MAX": 2, "VSRAM_WORDS": 2048},
    "fpga_w64": {"WB": 64, "B_MAX": 1, "VSRAM_WORDS": 4096},
    "sim_w128": {"WB": 128, "B_MAX": 1, "VSRAM_WORDS": 4096},
}


def _files() -> list[str]:
    return (
        [str(qc_runner.RTL / "qcore_pkg.sv")]
        + [str(qc_runner.RTL / s) for s in SOURCES]
        + [str(WRAP)]
    )


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=qc_runner.REPO, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"{' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}"
    return proc


@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_gemv_wrap_elaborates(cfg: str) -> None:
    for tool in ("verilator", "yosys", "iverilog"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} not on PATH")
    top, params, files, rtl = "qcore_gemv_wrap", CONFIGS[cfg], _files(), str(qc_runner.RTL)
    proc = _run(
        ["verilator", "--lint-only", "-Wall", "-Wpedantic", "--top-module", top, f"-I{rtl}"]
        + [f"-G{k}={v}" for k, v in params.items()]
        + files
    )
    assert "%Warning" not in proc.stderr, proc.stderr
    chparam = " ".join(f"-set {k} {v}" for k, v in params.items())
    _run(
        [
            "yosys",
            "-q",
            "-p",
            f"read_verilog -sv -defer -I{rtl} {' '.join(files)}; chparam {chparam} {top}; "
            f"hierarchy -check -top {top}; proc; opt; check -assert",
        ]
    )
    _run(
        ["iverilog", "-g2012", f"-I{rtl}", "-s", top, "-o", "/dev/null"]
        + [f"-P{top}.{k}={v}" for k, v in params.items()]
        + files
    )


def test_gemv_wrap() -> None:
    qc_runner.run(
        "qcore_gemv_wrap",
        [*SOURCES, str(WRAP)],
        "tb_gemv_wrap",
        parameters={
            "WB": qc_runner.TINY["WB"],
            "B_MAX": qc_runner.TINY["B_MAX"],
            "VSRAM_WORDS": qc_runner.TINY["VSRAM_WORDS"],
            "FIFO_BEATS": qc_runner.TINY["FIFO_BEATS"],
            "ACC_W": qc_runner.TINY["ACC_W"],
            "META_FIFO_BEATS": 16,
            "MAX_BURST": 64,
        },
    )
