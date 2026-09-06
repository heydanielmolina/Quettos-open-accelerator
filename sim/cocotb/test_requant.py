"""pytest entry: builds qcore_requant on the tiny configuration and runs tb_requant."""

from __future__ import annotations

import qc_runner


def test_requant() -> None:
    qc_runner.run(
        "qcore_requant",
        ["qcore_requant.sv"],
        "tb_requant",
        parameters={
            "WB": qc_runner.TINY["WB"],
            "B_MAX": qc_runner.TINY["B_MAX"],
            "ACC_W": qc_runner.TINY["ACC_W"],
            "VSRAM_WORDS": qc_runner.TINY["VSRAM_WORDS"],
        },
    )
