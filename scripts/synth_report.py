#!/usr/bin/env python3
"""Turn a Yosys `stat -tech xilinx` / `ltp` log into syn/reports/xc7.md.

STUB (see docs/ROADMAP.md).

Planned usage:
    uv run python scripts/synth_report.py --log build/syn/xc7.log \
        --out syn/reports/xc7.md --part xc7a100t

Planned behaviour:
  * parse cell counts (LUT1..LUT6, FDRE/FDSE, CARRY4, DSP48E1, RAMB36E1,
    RAMB18E1) and the `ltp -noff` longest-path depth from the Yosys log;
  * compute percentages of XC7A100T (63,400 LUT6, 126,800 FF, 240 DSP48E1,
    135 RAMB36) and XC7A200T;
  * record the exact yosys command line and `yosys -V` string;
  * emit a Markdown table with the method notes: Yosys synth_xilinx results,
    post-synthesis (fmax from nextpnr when run; ltp logic levels as the depth
    metric); core only, excluding memory controller, host interface and clocking.
"""

import sys


def main() -> int:
    sys.stderr.write(
        "synth_report.py: not implemented yet (see docs/ROADMAP.md)\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
