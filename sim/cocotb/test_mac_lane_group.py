"""pytest entry: builds qcore_mac_lane_group on the tiny configuration, runs tb_mac_lane_group."""

from __future__ import annotations

import qc_runner


def test_mac_lane_group() -> None:
    qc_runner.run(
        "qcore_mac_lane_group",
        ["qcore_mac_lane_group.sv"],
        "tb_mac_lane_group",
        parameters={"ACC_W": qc_runner.TINY["ACC_W"]},
    )
