"""pytest entry: builds qcore_row on the tiny configuration and runs tb_row."""

from __future__ import annotations

import qc_runner


def test_row() -> None:
    qc_runner.run(
        "qcore_row",
        ["qcore_mac_lane_group.sv", "qcore_row.sv"],
        "tb_row",
        parameters={
            "WB": qc_runner.TINY["WB"],
            "ACC_W": qc_runner.TINY["ACC_W"],
            "VSRAM_WORDS": qc_runner.TINY["VSRAM_WORDS"],
        },
    )
