"""pytest entry: builds qcore_vpu_scalar with its two ROMs and runs tb_vpu_scalar.

The unit takes no width parameter; the ROM image paths come from ``qc_runner``.
"""

from __future__ import annotations

import qc_runner


def test_vpu_scalar() -> None:
    qc_runner.run(
        "qcore_vpu_scalar",
        ["qcore_lut_rom.sv", "qcore_lut_interp.sv", "qcore_vpu_scalar.sv"],
        "tb_vpu_scalar",
    )
