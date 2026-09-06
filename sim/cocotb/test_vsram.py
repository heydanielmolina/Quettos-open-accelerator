"""pytest entry: builds qcore_vsram on the tiny configuration and runs tb_vsram."""

from __future__ import annotations

import qc_runner


def test_vsram() -> None:
    qc_runner.run(
        "qcore_vsram",
        ["qcore_vsram.sv"],
        "tb_vsram",
        parameters={"WORDS": qc_runner.TINY["VSRAM_WORDS"]},
    )
