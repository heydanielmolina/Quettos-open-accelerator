"""pytest entry: builds qcore_stream_ctrl on the tiny configuration and runs tb_stream_ctrl."""

from __future__ import annotations

import qc_runner


def test_stream_ctrl() -> None:
    qc_runner.run(
        "qcore_stream_ctrl",
        ["qcore_stream_ctrl.sv"],
        "tb_stream_ctrl",
        parameters={
            "WB": qc_runner.TINY["WB"],
            "FIFO_BEATS": qc_runner.TINY["FIFO_BEATS"],
            "META_FIFO_BEATS": 16,
            "MAX_BURST": 64,
        },
    )
