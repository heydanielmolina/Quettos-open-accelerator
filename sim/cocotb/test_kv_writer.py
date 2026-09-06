"""pytest entries: build qcore_kv_writer at WB = 16, 64 and 128 and run tb_kv_writer on each.

The bench reads its widths from the environment, so each entry passes them as ``QC_*``
alongside the build parameters and clears ``build/cocotb/qcore_kv_writer`` first, which
keeps every configuration a fresh build.
"""

from __future__ import annotations

import shutil

import pytest
import qc_runner

CONFIGS = {
    "w16": {"WB": 16, "B_MAX": 2, "VSRAM_WORDS": 2048},
    "w64": {"WB": 64, "B_MAX": 2, "VSRAM_WORDS": 2048},
    "w128": {"WB": 128, "B_MAX": 1, "VSRAM_WORDS": 4096},
}


@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_kv_writer(cfg: str) -> None:
    params = CONFIGS[cfg]
    shutil.rmtree(qc_runner.BUILD / "qcore_kv_writer", ignore_errors=True)
    qc_runner.run(
        "qcore_kv_writer",
        ["qcore_kv_writer.sv"],
        "tb_kv_writer",
        parameters=params,
        extra_env={f"QC_{k}": str(v) for k, v in params.items()},
    )
