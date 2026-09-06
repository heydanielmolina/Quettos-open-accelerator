"""pytest entries for qcore_seq_dispatch: tb_seq_dispatch on the tiny configuration (B_MAX = 2)
and a three-parser elaboration of the module in all three configurations."""

from __future__ import annotations

import pytest
import qc_runner
from test_seq_fetch import elaborate

CONFIGS = {
    "tiny_w16": {"WB": 16, "B_MAX": 2},
    "fpga_w64": {"WB": 64, "B_MAX": 1},
    "sim_w128": {"WB": 128, "B_MAX": 1},
}


def test_seq_dispatch() -> None:
    qc_runner.run(
        "qcore_seq_dispatch",
        ["qcore_seq_dispatch.sv"],
        "tb_seq_dispatch",
        parameters={"WB": qc_runner.TINY["WB"], "B_MAX": qc_runner.TINY["B_MAX"]},
    )


@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_seq_dispatch_elaborates(cfg: str) -> None:
    elaborate("qcore_seq_dispatch", CONFIGS[cfg], ["qcore_pkg.sv", "qcore_seq_dispatch.sv"])
