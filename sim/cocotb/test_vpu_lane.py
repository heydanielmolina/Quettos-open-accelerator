"""pytest entry: builds qcore_vpu_lane and runs tb_vpu_lane.

The lane takes no configuration parameter, so one build covers every RTL width.
"""

from __future__ import annotations

import qc_runner


def test_vpu_lane() -> None:
    qc_runner.run("qcore_vpu_lane", ["qcore_vpu_lane.sv"], "tb_vpu_lane")
