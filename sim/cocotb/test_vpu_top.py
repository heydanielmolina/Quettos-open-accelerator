"""pytest entry: builds qcore_vpu_top with its lanes, tables and scalar unit, runs tb_vpu_top.

The vector unit is built on the tiny configuration (``WB = 16``, ``B_MAX = 2``,
``VL = 2``, ``VSRAM_WORDS = 2048``) and, in ``test_vpu_top_wide``, on the demo
widths, so both the two-lane and the four-lane operand windows are exercised.
The bench reads the widths from the environment and the ROM image paths come
from ``qc_runner``.
"""

from __future__ import annotations

import shutil

import pytest
import qc_runner

SOURCES = [
    "qcore_lut_rom.sv",
    "qcore_lut_interp.sv",
    "qcore_vpu_lane.sv",
    "qcore_vpu_scalar.sv",
    "qcore_vpu_top.sv",
]

CONFIGS = {
    "tiny": {"WB": 16, "B_MAX": 2, "VL": 2, "VSRAM_WORDS": 2048},
    "demo": {"WB": 64, "B_MAX": 1, "VL": 4, "VSRAM_WORDS": 4096},
}


@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_vpu_top(cfg: str) -> None:
    params = CONFIGS[cfg]
    shutil.rmtree(qc_runner.BUILD / "qcore_vpu_top", ignore_errors=True)
    qc_runner.run(
        "qcore_vpu_top",
        SOURCES,
        "tb_vpu_top",
        parameters=params,
        extra_env={f"QC_{k}": str(v) for k, v in params.items()},
    )
