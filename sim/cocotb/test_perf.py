"""pytest entry: builds qcore_perf on the tiny configuration and runs tb_perf.

The bench reads ``WB`` from the environment, so the entry passes it as ``QC_WB``
alongside the build parameter and clears ``build/cocotb/qcore_perf`` first.
"""

from __future__ import annotations

import shutil

import qc_runner


def test_perf() -> None:
    params = {"WB": qc_runner.TINY["WB"]}
    shutil.rmtree(qc_runner.BUILD / "qcore_perf", ignore_errors=True)
    qc_runner.run(
        "qcore_perf",
        ["qcore_perf.sv"],
        "tb_perf",
        parameters=params,
        extra_env={f"QC_{k}": str(v) for k, v in params.items()},
    )
