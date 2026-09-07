"""pytest entries for qcore_lut_rom and qcore_lut_interp.

``test_lut_rom`` builds the ROM once per table with that table's
``rtl/gen/<name>.hex`` on ``ROM_FILE`` and runs the ROM coroutines of
``tb_lut``; ``test_lut_interp`` builds the interpolator once and runs the
arithmetic coroutines, which sweep the whole input domain of all four tables.
``test_rom_file_elaborates``, ``test_yosys_loads_the_hex`` and
``test_icarus_loads_the_hex`` take the ``ROM_FILE`` flow through the other two
parsers: Verilator loads the image in the cocotb build above, Yosys is made to
print the memory initialization it read and Icarus to print every entry it
holds, and both are compared against the checked-in file.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import qc_runner
from quettos import numerics

TABLES = {name: spec["entries"] for name, spec in numerics.TABLE_SPECS.items()}
BUILD = qc_runner.REPO / "build" / "lut"

ROM_TESTS = (
    "test_rom_every_entry_port_a",
    "test_rom_every_entry_port_b",
    "test_rom_both_ports_same_cycle",
    "test_rom_latency_and_hold",
    "test_rom_ends_and_segment_boundary",
)
INTERP_TESTS = (
    "test_interp_valid_and_hold",
    "test_interp_rounding_boundaries",
    "test_interp_product_extremes",
    "test_interp_landmark_fraction_sweeps",
    "test_interp_every_entry_every_fraction",
)


def hex_path(table: str) -> Path:
    return qc_runner.ROM_DIR / f"{table}.hex"


def hex_words(table: str) -> list[int]:
    """The ``{v[15:0], dv[15:0]}`` words of one checked-in image."""
    return [int(line, 16) for line in hex_path(table).read_text().split()]


def json_words(table: str) -> list[int]:
    """The same words rebuilt from ``sw/quettos/tables/luts.json``."""
    lut = getattr(numerics.load_tables(), table)
    return [(int(v) << 16) | (int(dv) & 0xFFFF) for v, dv in zip(lut.v, lut.dv, strict=True)]


def need(*tools: str) -> None:
    for tool in tools:
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} not on PATH")


def run_tool(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd, cwd=cwd or qc_runner.REPO, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, f"{' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}"
    return proc


# --------------------------------------------------------------------------- cocotb


@pytest.mark.parametrize("table", sorted(TABLES))
def test_lut_rom(table: str) -> None:
    qc_runner.run(
        "qcore_lut_rom",
        ["qcore_lut_rom.sv"],
        "tb_lut",
        parameters={"ENTRIES": TABLES[table], "ROM_FILE": f'"{hex_path(table)}"'},
        testcase=ROM_TESTS,
        extra_env={"QC_LUT_TABLE": table},
    )


def test_lut_interp() -> None:
    qc_runner.run("qcore_lut_interp", ["qcore_lut_interp.sv"], "tb_lut", testcase=INTERP_TESTS)


# ------------------------------------------------------------------- the ROM_FILE flow


@pytest.mark.parametrize("table", sorted(TABLES))
def test_rom_file_elaborates(table: str) -> None:
    """The three parsers elaborate the ROM with the image path on ``ROM_FILE``."""
    need("verilator", "yosys", "iverilog")
    rtl, top = str(qc_runner.RTL), "qcore_lut_rom"
    src = str(qc_runner.RTL / "qcore_lut_rom.sv")
    entries, image = TABLES[table], hex_path(table)
    proc = run_tool(
        [
            "verilator",
            "--lint-only",
            "-Wall",
            "-Wpedantic",
            "--top-module",
            top,
            f"-I{rtl}",
            f"-GENTRIES={entries}",
            f'-GROM_FILE="{image}"',
            src,
        ]
    )
    assert "%Warning" not in proc.stderr, proc.stderr
    run_tool(
        [
            "yosys",
            "-q",
            "-p",
            f"read_verilog -sv -defer -I{rtl} {src}; "
            f'chparam -set ENTRIES {entries} -set ROM_FILE "{image}" {top}; '
            f"hierarchy -check -top {top}; proc; opt; check -assert",
        ]
    )
    run_tool(
        [
            "iverilog",
            "-g2012",
            f"-I{rtl}",
            "-s",
            top,
            "-o",
            "/dev/null",
            f"-P{top}.ENTRIES={entries}",
            f'-P{top}.ROM_FILE="{image}"',
            src,
        ]
    )


@pytest.mark.parametrize("table", sorted(TABLES))
def test_yosys_loads_the_hex(table: str) -> None:
    """Yosys's memory initialization is the image, word for word.

    ``read_verilog -defer`` plus ``chparam`` is the one Yosys 0.65 form that
    reaches ``$readmemh``; the ``$meminit_v2`` cell it leaves behind carries the
    contents as one bit vector, least significant bit of word 0 first.
    """
    need("yosys")
    entries, image = TABLES[table], hex_path(table)
    out = BUILD / table
    out.mkdir(parents=True, exist_ok=True)
    design = out / "meminit.json"
    run_tool(
        [
            "yosys",
            "-q",
            "-p",
            f"read_verilog -sv -defer -I{qc_runner.RTL} {qc_runner.RTL / 'qcore_lut_rom.sv'}; "
            f'chparam -set ENTRIES {entries} -set ROM_FILE "{image}" qcore_lut_rom; '
            f"hierarchy -check -top qcore_lut_rom; proc; opt; write_json {design}",
        ]
    )
    cells = json.loads(design.read_text())["modules"]["qcore_lut_rom"]["cells"]
    inits = [c for c in cells.values() if c["type"] == "$meminit_v2"]
    assert len(inits) == 1, f"{table}: {len(inits)} memory initializations, expected 1"
    init = inits[0]
    assert int(init["parameters"]["WORDS"], 2) == entries
    assert int(init["parameters"]["WIDTH"], 2) == 32
    bits = init["connections"]["DATA"]
    assert len(bits) == entries * 32, f"{table}: {len(bits)} initialized bits"
    assert set(bits) <= {"0", "1"}, f"{table}: the initialization is not fully driven"
    got = [sum(int(bits[w * 32 + b]) << b for b in range(32)) for w in range(entries)]
    assert got == hex_words(table), f"{table}: Yosys read something other than {image}"
    assert got == json_words(table)


DUMP_BENCH = """`timescale 1ns/1ps
// Generated by sim/cocotb/test_lut.py. Prints every entry the ROM holds.
module qcore_lut_dump #(
  parameter int ENTRIES = 256,
  parameter     ROM_FILE = ""
);
  localparam int AW = $clog2(ENTRIES);
  reg clk = 1'b0;
  reg en_a = 1'b0;
  reg en_b = 1'b0;
  reg [AW-1:0] idx_a = {AW{1'b0}};
  reg [AW-1:0] idx_b = {AW{1'b0}};
  wire [15:0] v_a, dv_a, v_b, dv_b;
  integer i;

  qcore_lut_rom #(.ENTRIES(ENTRIES), .ROM_FILE(ROM_FILE)) u_rom (
    .clk(clk), .en_a(en_a), .idx_a(idx_a), .v_a(v_a), .dv_a(dv_a),
    .en_b(en_b), .idx_b(idx_b), .v_b(v_b), .dv_b(dv_b)
  );

  always #5 clk = ~clk;

  initial begin
    en_a = 1'b1;
    en_b = 1'b1;
    for (i = 0; i < ENTRIES; i = i + 1) begin
      idx_a = i[AW-1:0];
      idx_b = ENTRIES - 1 - i;
      @(posedge clk);
      #1;
      $display("E %0d %0d %0d %0d %0d %0d",
               i, v_a, $signed(dv_a), ENTRIES - 1 - i, v_b, $signed(dv_b));
    end
    $finish;
  end
endmodule
"""


@pytest.mark.parametrize("table", sorted(TABLES))
def test_icarus_loads_the_hex(table: str) -> None:
    """Icarus loads the image through ``-P<top>.ROM_FILE`` and returns every entry."""
    need("iverilog", "vvp")
    entries, image = TABLES[table], hex_path(table)
    out = BUILD / table
    out.mkdir(parents=True, exist_ok=True)
    bench = out / "dump.sv"
    bench.write_text(DUMP_BENCH)
    run_tool(
        [
            "iverilog",
            "-g2012",
            "-s",
            "qcore_lut_dump",
            "-o",
            str(out / "dump.vvp"),
            f"-Pqcore_lut_dump.ENTRIES={entries}",
            f'-Pqcore_lut_dump.ROM_FILE="{image}"',
            str(bench),
            str(qc_runner.RTL / "qcore_lut_rom.sv"),
        ]
    )
    sim = run_tool(["vvp", str(out / "dump.vvp")], cwd=out)
    rows = [ln.split()[1:] for ln in sim.stdout.splitlines() if ln.startswith("E ")]
    assert len(rows) == entries, f"{table}: {len(rows)} entries printed"
    lut = getattr(numerics.load_tables(), table)
    for row in rows:
        ia, va, dva, ib, vb, dvb = (int(x) for x in row)
        assert (va, dva) == (int(lut.v[ia]), int(lut.dv[ia])), f"{table} port A entry {ia}"
        assert (vb, dvb) == (int(lut.v[ib]), int(lut.dv[ib])), f"{table} port B entry {ib}"


def test_hex_images_match_the_tables() -> None:
    """The four images the build hands the RTL are the tables numerics loads."""
    for table in TABLES:
        assert hex_words(table) == json_words(table), table
        assert len(hex_words(table)) == TABLES[table], table
