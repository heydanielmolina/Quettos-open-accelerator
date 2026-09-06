"""pytest entry: builds qcore_csr and runs tb_csr."""

from __future__ import annotations

import qc_runner


def test_csr() -> None:
    qc_runner.run("qcore_csr", ["qcore_csr.sv"], "tb_csr")
