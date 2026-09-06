"""pytest entry: builds qcore_mem_arb on the tiny configuration and runs tb_mem_arb."""

from __future__ import annotations

import qc_runner


def test_mem_arb() -> None:
    qc_runner.run(
        "qcore_mem_arb",
        ["qcore_mem_arb.sv"],
        "tb_mem_arb",
        parameters={"WB": qc_runner.TINY["WB"], "MAX_BURST": 64},
    )
