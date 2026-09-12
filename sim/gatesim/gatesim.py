"""Gate-level equivalence: a Yosys netlist against the SystemVerilog it came from.

For every entry of CASES: Yosys maps the module to Xilinx 7-series cells and
writes the netlist, then one Icarus bench drives the source module and the
netlist from the same stimulus and compares the concatenation of their outputs
every cycle. A cycle in which the two differ is a mismatch and the run exits
non-zero. The netlist is simulated against Yosys's own cell models
(``$(yosys-config --datdir)/xilinx/cells_sim.v``), so the source side is read by
Icarus and the netlist side by Yosys: a construct the two front ends read
differently shows up as a mismatch instead of surviving to the bitstream.

The coverage table of ``sim/gatesim/README.md`` is written from the run: every
case, the parameters it is elaborated with and the cell count Yosys reported for
it. A full run checks that table and fails when it no longer matches;
``--write-table`` rewrites it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
README = Path(__file__).resolve().parent / "README.md"

# The coverage table of README.md is generated: the run writes the rows between
# these two markers, so a cell count on the page is one Yosys just reported.
TABLE_BEGIN = "<!-- gatesim:cases -->"
TABLE_END = "<!-- /gatesim:cases -->"

# Cycles the bench holds reset before the directed vectors start, and the extra
# settling cycles before the first comparison.
RESET_CYCLES = 4
SETTLE_CYCLES = 4
# The share of the comparison window the source outputs may still be X in: a
# module whose registers or memories come up unwritten spends its first cycles
# there, and a mid-run reset puts it back. Past a tenth the window is no longer
# worth calling a comparison.
MAX_X_SHARE = 0.10


@dataclass(frozen=True)
class Case:
    """One module in one configuration."""

    name: str
    top: str
    sources: tuple[str, ...]
    # Parameter values as Yosys, Icarus and the generated bench all read them:
    # an int, or the already-quoted absolute path `rom_image` builds for a ROM
    # image parameter (`ROM_FILE`, `ROM_FILE_<TABLE>`), which the RTL declares
    # untyped because Yosys 0.65 rejects `parameter string`.
    params: dict[str, int | str] = field(default_factory=dict)
    cycles: int = 4000
    reset: str | None = "rst"
    # Verilog driven verbatim on the cycles right after reset, one entry per
    # cycle; a signal an entry does not name holds its previous value.
    directed: tuple[str, ...] = ()
    # Verilog applied after the random draw of every random cycle, to steer the
    # wide inputs into the encodings the module decodes.
    shape: str = ""
    # Outputs that are constant by construction. Every other output port must
    # toggle, or the stimulus no longer reaches it and the case is failed.
    constants: tuple[str, ...] = ()
    seed: int = 1


# A descriptor whose fields are drawn from the ranges the compiler emits: a
# known opcode, one or two participating rows, an N and a K near the tile
# boundaries the dispatcher rounds to, and the POS-driven flags set half the
# time. Fields are placed by docs/ISA.md, "Descriptor bit layout".
DESC_SHAPE = """
      rv = $random(seed);                                             // opcode
      case (rv % 16)
        0:       dq_desc[7:0] = 8'd0;    // NOP
        1:       dq_desc[7:0] = 8'd1;    // HALT
        2, 3, 4: dq_desc[7:0] = 8'd16;   // GEMV
        5:       dq_desc[7:0] = 8'd17;   // EMBED
        6:       dq_desc[7:0] = 8'd32;   // VRMSNORM
        7:       dq_desc[7:0] = 8'd33;   // VQUANT
        8:       dq_desc[7:0] = 8'd34;   // VROPE
        9:       dq_desc[7:0] = 8'd35;   // VSILUMUL
        10:      dq_desc[7:0] = 8'd36;   // VSOFTMAX
        11:      dq_desc[7:0] = 8'd37;   // VSUBC
        12, 13:  dq_desc[7:0] = 8'd48;   // KVWRITE
        14:      dq_desc[7:0] = 8'd49;   // FENCE
        default: dq_desc[7:0] = rv[7:0]; // an unknown opcode must fault alike
      endcase
      rv = $random(seed); dq_desc[15:8]    = rv[7:0];                 // flags
      rv = $random(seed); dq_desc[23:16]   = 8'd1 << (rv % 2);        // row_mask
      rv = $random(seed); dq_desc[27:24]   = rv[3:0];                 // acc/meta/from_pos
      rv = $random(seed); dq_desc[119:96]  = (rv % 5) * 24'd16 + (rv >> 8) % 3;  // n
      rv = $random(seed); dq_desc[135:120] = (rv % 5) * 16'd16 + (rv >> 8) % 3;  // k
      rv = $random(seed); dq_desc[203:200] = rv % 2;                  // src_row
      rv = $random(seed); dq_desc[207:204] = rv % 2;                  // dst_row
      rv = $random(seed); pos_q = (rv % 4) ? (rv % 96) : rv;
      rv = $random(seed); pc_q  = {rv[26:0], 5'd0};                   // a fault stops the run
      rv = $random(seed); if ((rv % 64) == 0) row_en_q = ~row_en_q;
      row_en_q    = row_en_q | 1'b1;
      rv = $random(seed); abort_run   = (rv % 64) == 0;
      rv = $random(seed); start       = (rv % 8) == 0;
      rv = $random(seed); step        = (rv % 64) == 0;
      rv = $random(seed); wr_idle     = (rv % 8) != 0;
      rv = $random(seed); dq_valid    = (rv % 8) != 0;
      rv = $random(seed); stream_busy = (rv % 8) == 0;
      rv = $random(seed); stream_done = (rv % 6) == 0;
      rv = $random(seed); done_gemv   = (rv % 6) == 0;
      rv = $random(seed); done_vpu    = (rv % 6) == 0;
      rv = $random(seed); done_kv     = (rv % 6) == 0;
"""


def rom_image(table: str) -> str:
    """The `chparam` / instance value of one ROM_FILE parameter: a quoted image path."""
    return f'"{REPO / "rtl" / "gen" / f"{table}.hex"}"'


CASES: tuple[Case, ...] = (
    Case(
        name="mac_lane_group",
        top="qcore_mac_lane_group",
        sources=("rtl/qcore_mac_lane_group.sv",),
        params={"ACC_W": 40},
        cycles=4000,
        reset=None,
        directed=(
            "en = 1'b1; tile_start = 1'b1; embed = 1'b0; buf_sel = 1'b1;",
            "w = 64'hff80_7f01_8000_7fff; a = 16'h8000;",
            "tile_start = 1'b0; a = 16'h7fff;",
            "embed = 1'b1; tile_start = 1'b1;",
            "embed = 1'b0; buf_sel = 1'b0;",
        ),
    ),
    Case(
        name="mem_arb_wb64",
        top="qcore_mem_arb",
        sources=("rtl/qcore_pkg.sv", "rtl/qcore_mem_arb.sv"),
        params={"WB": 64, "MAX_BURST": 64},
        cycles=6000,
        directed=(
            "s_req_valid = 1'b1; s_req_len = 8'd64; rd_req_ready = 1'b1; dq_count = 4'd0;",
            "f_req_valid = 1'b1; f_req_len = 8'd1;",
            "k_wr_valid = 1'b1; k_wr_strb = {WB{1'b1}}; wr_ready = 1'b1;",
            "wr_ack = 1'b1; d_wr_valid = 1'b1;",
            "rd_data_valid = 1'b1; rd_data_tag = 4'd0; rd_data_last = 1'b1;",
            "rd_data_tag = 4'd1;",
            "rd_data_tag = 4'd2;",
            "rd_data_tag = 4'd3;",
        ),
        shape=(
            "rv = $random(seed); s_req_len = rv[7:0];\nrv = $random(seed); rd_data_tag = rv % 6;\n"
        ),
    ),
    Case(
        name="mem_arb_wb16",
        top="qcore_mem_arb",
        sources=("rtl/qcore_pkg.sv", "rtl/qcore_mem_arb.sv"),
        params={"WB": 16, "MAX_BURST": 8},
        cycles=6000,
        directed=(
            "s_req_valid = 1'b1; s_req_len = 8'd8; rd_req_ready = 1'b1; dq_count = 4'd0;",
            "f_req_valid = 1'b1; f_req_len = 8'd2;",
            "k_wr_valid = 1'b1; k_wr_strb = {WB{1'b1}}; wr_ready = 1'b1;",
            "wr_ack = 1'b1; d_wr_valid = 1'b1;",
            "rd_data_valid = 1'b1; rd_data_tag = 4'd0; rd_data_last = 1'b1;",
        ),
        shape=(
            "rv = $random(seed); s_req_len = rv[7:0];\nrv = $random(seed); rd_data_tag = rv % 6;\n"
        ),
    ),
    Case(
        name="seq_fetch_wb64",
        top="qcore_seq_fetch",
        sources=("rtl/qcore_pkg.sv", "rtl/qcore_seq_fetch.sv"),
        params={"WB": 64, "DQ_DEPTH": 8},
        cycles=8000,
        directed=(
            "fetch_start = 1'b1; fetch_pc = 32'h0000_0020; f_req_ready = 1'b1; dq_ready = 1'b1;",
            "fetch_start = 1'b0;",
            "fd_valid = 1'b1; fd_last = 1'b1;",
            "fetch_start = 1'b1; fetch_pc = 32'h0000_003f;",
            "fetch_start = 1'b0;",
            "fetch_start = 1'b1; fetch_pc = 32'hdead_beef;",
            "fetch_start = 1'b0;",
            "fetch_start = 1'b1; fetch_pc = 32'hffff_ffff;",
            "fetch_start = 1'b0; fetch_step = 1'b1;",
            "fetch_step = 1'b0; fetch_hold = 1'b1;",
            "fetch_hold = 1'b0; fetch_flush = 1'b1;",
            "fetch_flush = 1'b0;",
        ),
        # Every 32-byte-aligned and every misaligned PC is interesting, so half
        # the draws are small offsets around a beat boundary.
        shape="      rv = $random(seed); if (rv % 2) fetch_pc = {rv[27:0], 4'd0} ^ (rv >> 3);\n",
        # f_req_tag is TAG_FETCH and f_req_len is the beats per request; both
        # are constants of the configuration.
        constants=("f_req_tag", "f_req_len"),
    ),
    Case(
        name="seq_fetch_wb16",
        top="qcore_seq_fetch",
        sources=("rtl/qcore_pkg.sv", "rtl/qcore_seq_fetch.sv"),
        params={"WB": 16, "DQ_DEPTH": 8},
        cycles=8000,
        directed=(
            "fetch_start = 1'b1; fetch_pc = 32'h0000_0020; f_req_ready = 1'b1; dq_ready = 1'b1;",
            "fetch_start = 1'b0;",
            "fd_valid = 1'b1; fd_last = 1'b0;",
            "fd_last = 1'b1;",
            "fetch_start = 1'b1; fetch_pc = 32'h0000_001f;",
            "fetch_start = 1'b0;",
            "fetch_start = 1'b1; fetch_pc = 32'hffff_ffff;",
            "fetch_start = 1'b0; fetch_step = 1'b1;",
            "fetch_step = 1'b0; fetch_flush = 1'b1;",
            "fetch_flush = 1'b0;",
        ),
        shape="      rv = $random(seed); if (rv % 2) fetch_pc = {rv[27:0], 4'd0} ^ (rv >> 3);\n",
        # f_req_tag is TAG_FETCH and f_req_len is the beats per request; both
        # are constants of the configuration.
        constants=("f_req_tag", "f_req_len"),
    ),
    Case(
        name="seq_dispatch_wb64",
        top="qcore_seq_dispatch",
        sources=("rtl/qcore_pkg.sv", "rtl/qcore_seq_dispatch.sv"),
        params={"WB": 64, "B_MAX": 1},
        cycles=12000,
        directed=(
            "row_en_q = {B_MAX{1'b1}}; wr_idle = 1'b1; dq_valid = 1'b1; start = 1'b1;",
            "start = 1'b0; pc_q = 32'h0000_0040;",
            "dq_desc = 256'd0; dq_desc[7:0] = 8'd16; dq_desc[23:16] = 8'd1;"
            " dq_desc[119:96] = 24'd65; dq_desc[135:120] = 16'd33;",
            "dq_desc[26] = 1'b1; pos_q = 32'd65;",
            "done_gemv = 1'b1; stream_done = 1'b1;",
            "done_gemv = 1'b0; stream_done = 1'b0;",
            "dq_desc[27] = 1'b1; pos_q = 32'd63;",
            "pos_q = 32'd64;",
            "pos_q = 32'd127;",
        ),
        shape=DESC_SHAPE,
    ),
    Case(
        name="seq_dispatch_wb16",
        top="qcore_seq_dispatch",
        sources=("rtl/qcore_pkg.sv", "rtl/qcore_seq_dispatch.sv"),
        params={"WB": 16, "B_MAX": 2},
        cycles=12000,
        directed=(
            "row_en_q = {B_MAX{1'b1}}; wr_idle = 1'b1; dq_valid = 1'b1; start = 1'b1;",
            "start = 1'b0; pc_q = 32'h0000_0040;",
            "dq_desc = 256'd0; dq_desc[7:0] = 8'd16; dq_desc[23:16] = 8'd3;"
            " dq_desc[119:96] = 24'd17; dq_desc[135:120] = 16'd9;",
            "dq_desc[26] = 1'b1; pos_q = 32'd17;",
            "done_gemv = 1'b1; stream_done = 1'b1;",
            "done_gemv = 1'b0; stream_done = 1'b0;",
            "dq_desc[27] = 1'b1; pos_q = 32'd15;",
            "pos_q = 32'd16;",
            "pos_q = 32'd31;",
        ),
        shape=DESC_SHAPE,
    ),
    Case(
        name="requant_tiny",
        top="qcore_requant",
        sources=("rtl/qcore_pkg.sv", "rtl/qcore_requant.sv"),
        params={"WB": 16, "B_MAX": 2, "ACC_W": 40, "VSRAM_WORDS": 2048},
        cycles=4000,
        directed=(
            "cmd_rows = {B_MAX{1'b1}}; cmd_n = 24'd16; cmd_op = 8'd16;"
            " d_wr_ready = 1'b1; cmd_sx_m = {B_MAX{16'h8000}}; cmd_sx_e = {B_MAX{8'd0}};",
            "cmd_valid_gemv = 1'b1; cmd_sh0 = 8'd15; cmd_sh1 = 8'd0;",
            "cmd_valid_gemv = 1'b0; acc_valid = 1'b1; acc_nvalid = NVALID_MAX;"
            " acc_last = 1'b1; meta_valid = 1'b1; meta_data = 56'h00_8000_00000000;",
            "acc_flat = {(B_MAX*WB*ACC_W/8){8'h7f}};",
            "acc_flat = {(B_MAX*WB*ACC_W/8){8'h80}};",
        ),
        shape=(
            "      rv = $random(seed); cmd_n = (rv % 5) * 24'd16 + (rv >> 8) % 4;\n"
            "      rv = $random(seed); cmd_sh0 = rv % 70; rv = $random(seed); cmd_sh1 = rv % 70;\n"
            "      rv = $random(seed); acc_nvalid = rv % (WB + 1);\n"
            "      rv = $random(seed); cmd_out_mode = rv[1:0];\n"
            "      rv = $random(seed); cmd_op = (rv % 2) ? 8'd16 : 8'd17;\n"
        ),
    ),
    Case(
        name="csr",
        top="qcore_csr",
        sources=("rtl/qcore_pkg.sv", "rtl/qcore_csr.sv"),
        cycles=6000,
    ),
    Case(
        name="perf",
        top="qcore_perf",
        sources=("rtl/qcore_pkg.sv", "rtl/qcore_perf.sv"),
        params={"WB": 64},
        cycles=2000,
    ),
    Case(
        name="kv_writer_tiny",
        top="qcore_kv_writer",
        sources=("rtl/qcore_pkg.sv", "rtl/qcore_kv_writer.sv"),
        params={"WB": 16, "B_MAX": 2, "VSRAM_WORDS": 2048},
        cycles=6000,
    ),
    Case(
        name="row_tiny",
        top="qcore_row",
        sources=("rtl/qcore_pkg.sv", "rtl/qcore_mac_lane_group.sv", "rtl/qcore_row.sv"),
        params={"WB": 16, "ACC_W": 40, "VSRAM_WORDS": 2048},
        cycles=6000,
    ),
    Case(
        name="stream_ctrl_tiny",
        top="qcore_stream_ctrl",
        sources=("rtl/qcore_pkg.sv", "rtl/qcore_stream_ctrl.sv"),
        params={"WB": 16, "FIFO_BEATS": 32, "META_FIFO_BEATS": 8, "MAX_BURST": 8},
        cycles=6000,
    ),
    # The lookup ROMs carry their contents in the netlist, so the two front ends
    # have to read the same image out of the same $readmemh: Yosys through
    # `chparam -set ROM_FILE`, Icarus through the parameter on the instance.
    # ROM_FILE is listed first because each `chparam` re-elaborates the deferred
    # module, and the elaboration that loads the image has to have the path.
    Case(
        name="lut_rom_exp2",
        top="qcore_lut_rom",
        sources=("rtl/qcore_lut_rom.sv",),
        params={"ROM_FILE": rom_image("exp2"), "ENTRIES": 256},
        cycles=4000,
        reset=None,
        directed=(
            "en_a = 1'b1; en_b = 1'b1; idx_a = 0; idx_b = 0;",
            "idx_a = ENTRIES - 1; idx_b = ENTRIES - 1;",
            "idx_a = 1; idx_b = ENTRIES - 2;",
            "en_a = 1'b0;",
            "en_a = 1'b1; idx_a = ENTRIES / 2; idx_b = ENTRIES / 2 - 1;",
        ),
    ),
    Case(
        name="lut_rom_rsqrt",
        top="qcore_lut_rom",
        sources=("rtl/qcore_lut_rom.sv",),
        params={"ROM_FILE": rom_image("rsqrt"), "ENTRIES": 512},
        cycles=4000,
        reset=None,
        directed=(
            "en_a = 1'b1; en_b = 1'b1; idx_a = 0; idx_b = 0;",
            "idx_a = ENTRIES - 1; idx_b = ENTRIES - 1;",
            # the two sides of the segment boundary in one cycle
            "idx_a = 255; idx_b = 256;",
            "en_b = 1'b0;",
            "en_b = 1'b1; idx_a = ENTRIES / 2; idx_b = ENTRIES / 2 - 1;",
        ),
    ),
    Case(
        name="lut_interp",
        top="qcore_lut_interp",
        sources=("rtl/qcore_lut_interp.sv",),
        cycles=4000,
        directed=(
            "in_valid = 1'b1; v = 16'd32768; dv = 16'd128; frac8 = 8'd0;",
            "frac8 = 8'd255;",
            "dv = 16'hffff; frac8 = 8'd129;",
            "v = 16'hffff; dv = 16'h8000; frac8 = 8'd255;",
            "v = 16'd0; dv = 16'h7fff; frac8 = 8'd255;",
            "in_valid = 1'b0;",
        ),
    ),
    # The vector lane takes no parameter, so one case is every configuration of
    # it. The directed cycles walk the six ops through the operand extremes so
    # the products, the round-shift and both saturation signs are reached before
    # the random phases start.
    Case(
        name="vpu_lane",
        top="qcore_vpu_lane",
        sources=("rtl/qcore_pkg.sv", "rtl/qcore_vpu_lane.sv"),
        cycles=6000,
        directed=(
            "in_valid = 1'b1; op = 3'd0; a = 32'h7fff_ffff; b = 32'h7fff_ffff; sh = 6'd0;",
            "a = 32'h8000_0000; b = 32'h8000_0000; sh = 6'd63;",
            "sh = 6'd31;",
            "op = 3'd1; c = 16'hffff; sh = 6'd15;",
            "op = 3'd2; c = 16'h8000; c2 = 16'h7fff; sh = 6'd14;",
            "op = 3'd3;",
            "op = 3'd4; a = 32'h8000_0000; b = 32'h0000_0001;",
            "op = 3'd5; a = 32'hffff_ffff;",
            "op = 3'd6; in_valid = 1'b0;",
        ),
        shape=(
            "rv = $random(seed); in_valid = (rv % 4) != 0;\n"
            "rv = $random(seed); if (rv % 4) op = rv % 6;\n"
        ),
    ),
    # The scalar unit with the two tables it owns. ROM_FILE_* is listed first for
    # the same reason as the ROM cases above, and the directed cycles issue one
    # request of each op so both ports of both ROMs have been read before the
    # comparison window opens.
    Case(
        name="vpu_scalar",
        top="qcore_vpu_scalar",
        sources=(
            "rtl/qcore_pkg.sv",
            "rtl/qcore_lut_rom.sv",
            "rtl/qcore_lut_interp.sv",
            "rtl/qcore_vpu_scalar.sv",
        ),
        params={
            "ROM_FILE_RSQRT": rom_image("rsqrt"),
            "ROM_FILE_RECIP": rom_image("recip"),
        },
        cycles=6000,
        directed=(
            "req_valid = 1'b1; req_op = 2'd0; req_x = 64'h0000_0000_0002_0001;"
            " req_sh0 = 8'd16; req_sh = 6'd3; req_aux_m = 16'h8000; req_aux_e = 8'hf1;",
            "req_op = 2'd1; req_x = 64'h0000_0000_0000_ffff; req_w8 = 1'b1;",
            "req_op = 2'd2; req_x = 64'h0000_0010_0000_0000; req_w8 = 1'b0;",
            "req_op = 2'd1; req_mul_en = 1'b1; req_aux_m = 16'hffff; req_aux_e = 8'd7;",
            "req_x = 64'd0;",
            "req_op = 2'd0; req_x = 64'h0001_ffff_ffff_ffff; req_sh0 = 8'd255; req_sh = 6'd0;",
            "req_op = 2'd0; req_x = 64'h0000_0000_0003_ffff; req_sh0 = 8'd0; req_sh = 6'd63;",
            "req_valid = 1'b0;",
        ),
        shape=(
            "rv = $random(seed); req_valid = (rv % 4) != 0;\n"
            "rv = $random(seed); if (rv % 8) req_op = rv % 3;\n"
            "rv = $random(seed); if (rv % 2) req_x = {rv, $random(seed)} >> (rv % 40);\n"
        ),
    ),
    # The whole vector unit on the tiny configuration, with the lanes, the
    # scalar unit and the two curve tables it contains. ROM_FILE_* comes first
    # for the reason the ROM cases give, and the shape keeps the descriptors
    # short and mostly executable so a case reaches its passes, its SREG write
    # and its event strobes inside the window: a VROPE gets a whole number of
    # heads, a VSOFTMAX a length inside its n and the FRAC_S window docs/ISA.md
    # gives it. v_req_tag is TAG_VPU by construction.
    Case(
        name="vpu_top_tiny",
        top="qcore_vpu_top",
        sources=(
            "rtl/qcore_pkg.sv",
            "rtl/qcore_lut_rom.sv",
            "rtl/qcore_lut_interp.sv",
            "rtl/qcore_vpu_lane.sv",
            "rtl/qcore_vpu_scalar.sv",
            "rtl/qcore_vpu_top.sv",
        ),
        params={
            "ROM_FILE_SIGMOID": rom_image("sigmoid"),
            "ROM_FILE_EXP2": rom_image("exp2"),
            "ROM_FILE_RSQRT": rom_image("rsqrt"),
            "ROM_FILE_RECIP": rom_image("recip"),
            "WB": 16,
            "B_MAX": 2,
            "VL": 2,
            "VSRAM_WORDS": 2048,
            "VPU_FIFO_BEATS": 16,
            "MAX_BURST": 64,
        },
        cycles=8000,
        constants=("v_req_tag",),
        directed=(
            "cmd_op = 8'h21; cmd_n = 24'd12; cmd_vs_src = 16'd8; cmd_vs_dst = 16'd1024;"
            " cmd_vs_aux = 16'd2048; cmd_rows = 2'b11; cmd_sh0 = 8'd16; cmd_sh1 = 8'd16;"
            " cmd_sreg_dst = 8'd3; v_req_ready = 1'b1; cmd_valid_vpu = 1'b1;",
            "cmd_valid_vpu = 1'b0; vsa_rdata = {8{32'h0001_3579}}; vsb_rdata = {8{32'hfffe_0021}};",
            "cmd_op = 8'h22; cmd_n = 24'd64; cmd_pos = 32'd3; cmd_valid_vpu = 1'b1;",
            "cmd_valid_vpu = 1'b0;",
            "cmd_op = 8'h24; cmd_n = 24'd12; cmd_len = 24'd6; cmd_sh0 = 8'd20;"
            " cmd_valid_vpu = 1'b1;",
            "cmd_valid_vpu = 1'b0;",
        ),
        shape=(
            "rv = $random(seed);\n"
            "case (rv % 12)\n"
            "  0, 1: cmd_op = 8'h20;   // VRMSNORM\n"
            "  2, 3: cmd_op = 8'h21;   // VQUANT\n"
            "  4, 5: cmd_op = 8'h22;   // VROPE\n"
            "  6, 7: cmd_op = 8'h23;   // VSILUMUL\n"
            "  8, 9: cmd_op = 8'h24;   // VSOFTMAX\n"
            "  10:   cmd_op = 8'h25;   // VSUBC\n"
            "  default: cmd_op = rv[7:0];\n"
            "endcase\n"
            "rv = $random(seed); cmd_valid_vpu = (rv % 24) == 0;\n"
            "rv = $random(seed); cmd_n    = (rv % 4) ? (rv % 20) : 24'd0;\n"
            "if (cmd_op == 8'h22) cmd_n = 24'd64 * (1 + (rv % 2));\n"
            "rv = $random(seed); cmd_len = 24'd1 + (rv % 20);\n"
            "if (cmd_len > cmd_n) cmd_len = (cmd_n == 24'd0) ? 24'd1 : cmd_n;\n"
            "rv = $random(seed); cmd_pos = (rv % 4) ? (rv % 96) : rv;\n"
            "rv = $random(seed); cmd_vs_src = (rv % 8) ? (rv % 64) : rv[15:0];\n"
            "rv = $random(seed); cmd_vs_dst = (rv % 8) ? (16'd1024 + rv % 64) : rv[15:0];\n"
            "rv = $random(seed); cmd_vs_aux = (rv % 8) ? (16'd2048 + rv % 64) : rv[15:0];\n"
            "rv = $random(seed); cmd_rows = rv % 4;\n"
            "rv = $random(seed); cmd_sh0  = 8'd13 + (rv % 18);\n"
            "if (cmd_op == 8'h24) cmd_sh0 = 8'd16 + (rv % 15);\n"
            "rv = $random(seed); cmd_sh1  = rv % 96;\n"
            "rv = $random(seed); cmd_sreg_dst = rv % 40;\n"
            "rv = $random(seed); v_req_ready = (rv % 4) != 0;\n"
            "rv = $random(seed); rdv_valid   = (rv % 3) == 0;\n"
        ),
    ),
)

# The four stimulus phases the random body cycles through: (hold %, one %).
# `hold` is the chance an input keeps its value; `one` the chance a one-bit
# input is driven high. Together they sweep from free-running handshakes to
# heavy backpressure.
PHASES = ((0, 50), (60, 85), (25, 20), (85, 95))


def datdir() -> Path:
    """Where Yosys keeps its cell models: what yosys-config reports, and
    otherwise the share/yosys beside the yosys on PATH."""
    if shutil.which("yosys-config"):
        out = subprocess.run(["yosys-config", "--datdir"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip())
    exe = shutil.which("yosys")
    if exe is None:
        raise SystemExit("gatesim: yosys is not on PATH")
    return Path(exe).resolve().parent.parent / "share" / "yosys"


def modelled_cells(cells_sim: Path) -> set[str]:
    """Cell names the Yosys simulation library gives a behavioural body.

    The library declares the hard block-RAM primitives with ports alone, so a
    netlist that instantiates one cannot be simulated; naming them here turns
    that into an explicit failure instead of a netlist output stuck at Z.
    """
    text = cells_sim.read_text()
    have = set()
    for name, body in SIM_MODULE_RE.findall(text):
        if re.search(r"^\s*(always|assign|initial|specify)\b", body, re.M):
            have.add(name)
    return have


def yosys_script(case: Case, rtl_dir: Path, netlist: Path, ports: Path) -> str:
    files = " ".join(str(rewrite(s, rtl_dir)) for s in case.sources)
    chparam = "".join(f"chparam -set {k} {v} {case.top};\n" for k, v in case.params.items())
    return (
        f"read_verilog -sv -defer -I{rtl_dir} {files};\n"
        f"{chparam}"
        f"hierarchy -check -top {case.top};\n"
        f"synth_xilinx -family xc7 -top {case.top};\n"
        f"stat -tech xilinx;\n"
        f"flatten;\n"
        f"opt_clean;\n"
        f"rename {case.top} gate_dut;\n"
        f"write_json {ports};\n"
        f"write_verilog -noattr {netlist};\n"
    )


def rewrite(source: str, rtl_dir: Path) -> Path:
    """Map a repo-relative source path onto the RTL directory in use."""
    if source.startswith("rtl/"):
        return rtl_dir / source[len("rtl/") :]
    return REPO / source


def read_ports(path: Path) -> list[tuple[str, str, int]]:
    design = json.loads(path.read_text())
    mod = design["modules"]["gate_dut"]
    return [(n, p["direction"], len(p["bits"])) for n, p in mod["ports"].items()]


def decl(width: int) -> str:
    return "" if width == 1 else f"[{width - 1}:0] "


def indent(text: str, pad: str) -> str:
    return "".join(pad + ln + "\n" for ln in text.splitlines())


def random_assign(name: str, width: int, one_pct: int) -> str:
    """Verilog that draws one fresh value for an input."""
    if width == 1:
        return f"rv = $random(seed); {name} = (rv % 100) < {one_pct};"
    chunks = ", ".join(["$random(seed)"] * ((width + 31) // 32))
    top = f"{{1'b1, {{{max(width - 1, 1)}{{1'b0}}}}}}"
    return "\n".join(
        [
            "rv = $random(seed);",
            "case (rv % 8)",
            f"  0: {name} = 0;",
            f"  1: {name} = {{{width}{{1'b1}}}};",
            f"  2: begin rv = $random(seed); {name} = rv % 4; end",
            f"  3: begin rv = $random(seed); {name} = rv & 32'h3f; end",
            f"  4: begin rv = $random(seed); {name} = {top} ^ (rv % 4); end",
            f"  default: {name} = {{{chunks}}};",
            "endcase",
        ]
    )


def phase_body(ins: list[tuple[str, int]], hold: int, one: int) -> str:
    """One stimulus phase: every input either holds or takes a fresh draw."""
    out = []
    for n, w in ins:
        out.append("rv = $random(seed);")
        out.append(f"if ((rv % 100) >= {hold}) begin")
        out.append(indent(random_assign(n, w, one), "  ").rstrip("\n"))
        out.append("end")
    return "\n".join(out)


def gen_bench(case: Case, ports: list[tuple[str, str, int]]) -> str:
    ins = [(n, w) for n, d, w in ports if d == "input" and n not in ("clk", case.reset)]
    outs = [(n, w) for n, d, w in ports if d == "output"]
    if not outs:
        raise SystemExit(f"gatesim: {case.name}: the module has no outputs to compare")
    ow = sum(w for _, w in outs)
    n_dir = len(case.directed)
    warmup = RESET_CYCLES + n_dir + SETTLE_CYCLES

    # The integer parameters only: a directed or shape line names them
    # (`ENTRIES - 1`, `{WB{1'b1}}`), and a ROM image path is not an int.
    lp = [f"  localparam int {k} = {v};\n" for k, v in case.params.items() if isinstance(v, int)]
    if case.top == "qcore_requant":
        lp.append("  localparam int NVALID_MAX = WB;\n")

    src = ["`timescale 1ns/1ps\n", "// Generated by sim/gatesim/gatesim.py. Do not edit.\n"]
    src.append("module gatesim_tb;\n")
    src += lp
    src.append(f"  localparam int CYCLES = {case.cycles};\n")
    src.append(f"  localparam int WARMUP = {warmup};\n")
    src.append(f"  localparam int OW = {ow};\n")
    src.append("  reg clk = 1'b0;\n")
    if case.reset:
        src.append(f"  reg {case.reset} = 1'b1;\n")
    for n, w in ins:
        src.append(f"  reg {decl(w)}{n};\n")
    for n, w in outs:
        src.append(f"  wire {decl(w)}src_{n};\n")
        src.append(f"  wire {decl(w)}gate_{n};\n")
    src.append("  wire [OW-1:0] src_o = {" + ", ".join(f"src_{n}" for n, _ in outs) + "};\n")
    src.append("  wire [OW-1:0] gate_o = {" + ", ".join(f"gate_{n}" for n, _ in outs) + "};\n")
    src.append(
        f"  integer seed = {case.seed};\n"
        "  integer cyc; integer i;\n"
        "  reg [31:0] rv;\n"
        "  integer mism = 0; integer cmp = 0; integer xcyc = 0;\n"
        "  integer togg = 0; integer pt = 0;\n"
        "  reg [OW-1:0] seen0 = {OW{1'b0}};\n"
        "  reg [OW-1:0] seen1 = {OW{1'b0}};\n\n"
    )

    def inst(prefix: str, module: str, params: bool) -> str:
        pstr = ""
        if params and case.params:
            pstr = " #(" + ", ".join(f".{k}({v})" for k, v in case.params.items()) + ")"
        conns = ["    .clk(clk)"]
        if case.reset:
            conns.append(f"    .{case.reset}({case.reset})")
        conns += [f"    .{n}({n})" for n, _ in ins]
        conns += [f"    .{n}({prefix}_{n})" for n, _ in outs]
        return f"  {module}{pstr} u_{prefix} (\n" + ",\n".join(conns) + "\n  );\n"

    src.append(inst("src", case.top, True))
    src.append(inst("gate", "gate_dut", False))
    src.append("\n  always #5 clk = ~clk;\n\n")

    src.append("  initial begin\n")
    for n, w in ins:
        src.append(f"    {n} = {'1' if w == 1 else w}'d0;\n")
    src.append("    for (cyc = 0; cyc < CYCLES; cyc = cyc + 1) begin\n")
    src.append("      @(negedge clk);\n")
    src.append(f"      if (cyc < {RESET_CYCLES}) begin\n")
    if case.reset:
        src.append(f"        {case.reset} = 1'b1;\n")
    else:
        src.append("        ;\n")
    if n_dir:
        src.append(f"      end else if (cyc < {RESET_CYCLES + n_dir}) begin\n")
        if case.reset:
            src.append(f"        {case.reset} = 1'b0;\n")
        src.append(f"        case (cyc - {RESET_CYCLES})\n")
        for i, stmt in enumerate(case.directed):
            src.append(f"          {i}: begin {stmt} end\n")
        src.append("          default: ;\n")
        src.append("        endcase\n")
    src.append("      end else begin\n")
    if case.reset:
        # A rare mid-run reset, so the reset logic is compared as well.
        src.append(f"        rv = $random(seed); {case.reset} = (rv % 200) == 0;\n")
    src.append("        case ((cyc * 4) / CYCLES)\n")
    for ph, (hold, one) in enumerate(PHASES):
        src.append(f"          {ph}: begin\n")
        src.append(indent(phase_body(ins, hold, one), "            "))
        src.append("          end\n")
    src.append("          default: ;\n")
    src.append("        endcase\n")
    if case.shape:
        src.append(indent(textwrap.dedent(case.shape).strip("\n"), "        "))
    src.append("      end\n")
    src.append("      #1;\n")
    # A source output that is still X carries no information: the netlist's
    # flops come out of their INIT value where the source's are unwritten. Such
    # a cycle is counted and not compared, and too many of them fail the case.
    src.append("      if (cyc >= WARMUP) begin\n")
    src.append("        if ((^src_o) === 1'bx) begin\n")
    src.append("          xcyc = xcyc + 1;\n")
    src.append("        end else begin\n")
    src.append("          cmp = cmp + 1;\n")
    src.append("          seen0 = seen0 | ~src_o;\n")
    src.append("          seen1 = seen1 | src_o;\n")
    src.append("          if (src_o !== gate_o) begin\n")
    src.append("            mism = mism + 1;\n")
    src.append("            if (mism <= 3) begin\n")
    src.append('              $display("gatesim: MISMATCH cycle %0d", cyc);\n')
    for n, _ in outs:
        src.append(
            f"              if (src_{n} !== gate_{n})\n"
            f'                $display("gatesim:   {n} src=%h gate=%h", src_{n}, gate_{n});\n'
        )
    src.append("            end\n")
    src.append("          end\n")
    src.append("        end\n")
    src.append("      end\n")
    src.append("    end\n")
    src.append("    for (i = 0; i < OW; i = i + 1)\n")
    src.append("      if (seen0[i] && seen1[i]) togg = togg + 1;\n")
    # Per-port toggle counts land in the log, so an output the stimulus never
    # moves is visible when the coverage is tuned.
    off = 0
    for n, w in reversed(outs):
        src.append(f"    pt = 0;\n    for (i = {off}; i < {off + w}; i = i + 1)\n")
        src.append("      if (seen0[i] && seen1[i]) pt = pt + 1;\n")
        src.append(f'    $display("gatesim:   port {n} toggled %0d/{w}", pt);\n')
        off += w
    src.append(
        '    $display("GATESIM compared=%0d mismatches=%0d x_cycles=%0d toggled=%0d/%0d",\n'
        "             cmp, mism, xcyc, togg, OW);\n"
    )
    src.append("    $finish;\n  end\nendmodule\n")
    return "".join(src)


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    cells: int = 0
    compared: int = 0
    mismatches: int = 0
    toggled: int = 0
    out_bits: int = 0
    x_cycles: int = 0
    seconds: float = 0.0


CELL_RE = re.compile(r"^\s+(\d+) cells$", re.M)
MODULE_RE = re.compile(r"^module ", re.M)
CELL_INST_RE = re.compile(r"^  ([A-Z][A-Za-z0-9_]*) ", re.M)
SIM_MODULE_RE = re.compile(r"^module\s+(\w+)\b(.*?)^endmodule", re.M | re.S)
RESULT_RE = re.compile(
    r"GATESIM compared=(\d+) mismatches=(\d+) x_cycles=(\d+) toggled=(\d+)/(\d+)"
)
PORT_RE = re.compile(r"^gatesim:   port (\S+) toggled (\d+)/(\d+)$", re.M)


def config_text(case: Case) -> str:
    """The Configuration cell of one row: the parameters the case is elaborated with."""
    widths = ", ".join(f"{k}={v}" for k, v in case.params.items() if isinstance(v, int))
    roms = [
        Path(v.strip('"')).relative_to(REPO).as_posix()
        for v in case.params.values()
        if isinstance(v, str)
    ]
    cells = ([f"`{widths}`"] if widths else []) + [f"`{r}`" for r in roms]
    return ", ".join(cells) if cells else "defaults"


def case_table(cells: dict[str, int]) -> str:
    """The coverage table of README.md, with the cell count of each case from this run."""
    rows = ["| Case | Block | Configuration | Cells |", "|---|---|---|---|"]
    for case in CASES:
        rows.append(f"| `{case.name}` | `{case.top}` | {config_text(case)} | {cells[case.name]} |")
    return "\n".join(rows)


def sync_table(path: Path, table: str, write: bool) -> str | None:
    """Put `table` between the markers in `path`; the failure to report, or None."""
    text = path.read_text()
    head, sep, rest = text.partition(TABLE_BEGIN)
    _, end, tail = rest.partition(TABLE_END)
    if not sep or not end:
        return f"{path} carries no {TABLE_BEGIN} ... {TABLE_END} block for the table"
    fresh = f"{head}{TABLE_BEGIN}\n\n{table}\n\n{TABLE_END}{tail}"
    if fresh == text:
        return None
    if not write:
        rel = path.relative_to(REPO).as_posix()
        return f"{rel} does not carry this run's table; rewrite it with --write-table"
    path.write_text(fresh)
    return None


def run_case(
    case: Case, work: Path, rtl_dir: Path, cells_sim: Path, modelled: set[str], keep: bool
) -> Result:
    t0 = time.time()
    d = work / case.name
    d.mkdir(parents=True, exist_ok=True)
    netlist, ports, bench = d / "netlist.v", d / "ports.json", d / "bench.sv"
    script = yosys_script(case, rtl_dir, netlist, ports)
    (d / "synth.ys").write_text(script)
    y = subprocess.run(
        ["yosys", "-q", "-l", str(d / "synth.log"), "-s", str(d / "synth.ys")],
        capture_output=True,
        text=True,
    )
    if y.returncode != 0:
        return Result(case.name, False, f"yosys failed:\n{y.stdout}{y.stderr}")
    log = (d / "synth.log").read_text()
    m = CELL_RE.findall(log)
    cells = int(m[-1]) if m else 0
    text = netlist.read_text()
    if len(MODULE_RE.findall(text)) != 1:
        return Result(case.name, False, "the netlist holds more than one module")
    blind = sorted(set(CELL_INST_RE.findall(text)) - modelled)
    if blind:
        return Result(
            case.name,
            False,
            "the netlist instantiates cell(s) the Yosys simulation library only "
            f"declares: {' '.join(blind)}",
        )

    bench.write_text(gen_bench(case, read_ports(ports)))
    srcs = [str(rewrite(s, rtl_dir)) for s in case.sources]
    cc = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-DSYNTHESIS",
            f"-I{rtl_dir}",
            "-s",
            "gatesim_tb",
            "-o",
            str(d / "bench.vvp"),
            str(bench),
            *srcs,
            str(netlist),
            str(cells_sim),
        ],
        capture_output=True,
        text=True,
    )
    if cc.returncode != 0:
        return Result(case.name, False, f"iverilog failed:\n{cc.stdout}{cc.stderr}")
    sim = subprocess.run(["vvp", str(d / "bench.vvp")], capture_output=True, text=True, cwd=str(d))
    out = sim.stdout + sim.stderr
    (d / "sim.log").write_text(out)
    r = RESULT_RE.search(out)
    if sim.returncode != 0 or r is None:
        return Result(case.name, False, f"the bench did not finish:\n{out[-2000:]}")
    compared, mism, xcyc, togg, ow = (int(g) for g in r.groups())
    dead = [n for n, t, _ in PORT_RE.findall(out) if int(t) == 0 and n not in case.constants]
    window = compared + xcyc
    too_many_x = window == 0 or xcyc > MAX_X_SHARE * window
    ok = mism == 0 and not dead and not too_many_x
    detail = ""
    if mism:
        lines = [
            ln for ln in out.splitlines() if ln.startswith("gatesim:") and " toggled " not in ln
        ]
        detail = f"{mism} mismatching cycle(s)\n" + "\n".join(lines[:24])
    elif dead:
        detail = "the stimulus never moved these output(s): " + " ".join(dead)
    elif too_many_x:
        detail = (
            f"the source outputs were still undefined in {xcyc} of {window} cycles; "
            "the comparison window is too small to mean anything"
        )
    if not keep:
        (d / "bench.vvp").unlink(missing_ok=True)
    return Result(case.name, ok, detail, cells, compared, mism, togg, ow, xcyc, time.time() - t0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", default=None, help="run only these cases")
    ap.add_argument("--list", action="store_true", help="list the cases and exit")
    ap.add_argument("--rtl-dir", default=str(REPO / "rtl"), help="directory holding the RTL")
    ap.add_argument(
        "--work", default=str(REPO / "build" / "gatesim"), help="where the products land"
    )
    ap.add_argument(
        "--jobs", type=int, default=min(8, os.cpu_count() or 2), help="cases run in parallel"
    )
    ap.add_argument("--keep", action="store_true", help="keep the compiled benches")
    ap.add_argument(
        "--write-table",
        action="store_true",
        help="rewrite the coverage table of sim/gatesim/README.md from this run",
    )
    args = ap.parse_args()

    if args.list:
        for c in CASES:
            cfg = ", ".join(f"{k}={v}" for k, v in c.params.items()) or "defaults"
            print(f"{c.name:20s} {c.top:22s} {cfg}")
        return 0

    for tool in ("yosys", "iverilog", "vvp"):
        if shutil.which(tool) is None:
            print(f"gatesim: {tool} is not on PATH", file=sys.stderr)
            return 2
    cells_sim = datdir() / "xilinx" / "cells_sim.v"
    if not cells_sim.is_file():
        print(f"gatesim: no Xilinx cell models at {cells_sim}", file=sys.stderr)
        return 2

    cases = [c for c in CASES if args.only is None or c.name in args.only]
    missing = set(args.only or []) - {c.name for c in CASES}
    if missing:
        print(f"gatesim: unknown case(s): {' '.join(sorted(missing))}", file=sys.stderr)
        return 2

    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    rtl_dir = Path(args.rtl_dir).resolve()
    modelled = modelled_cells(cells_sim)
    print(f"gatesim: {len(cases)} case(s), RTL from {rtl_dir}, cells from {cells_sim}")

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(
            pool.map(
                lambda c: run_case(c, work, rtl_dir, cells_sim, modelled, args.keep),
                cases,
            )
        )

    bad = 0
    for r in results:
        if r.ok:
            skipped = f", {r.x_cycles} undefined" if r.x_cycles else ""
            print(
                f"gatesim: [{r.name}] {r.cells} cells, {r.compared} cycles compared"
                f"{skipped}, 0 mismatches, {r.toggled}/{r.out_bits} output bits "
                f"toggled ({r.seconds:.1f} s)"
            )
        else:
            bad += 1
            print(f"gatesim: [{r.name}] FAILED: {r.detail}")
    ok_n = len(results) - bad
    print(f"gatesim: {ok_n}/{len(results)} case(s) equivalent in {time.time() - t0:.1f} s")
    if bad:
        print("gatesim: FAILED")
        return 1
    # The whole set was run, so its cell counts are the coverage table: hold the
    # page to them, and rewrite it on request.
    if args.only is None:
        stale = sync_table(README, case_table({r.name: r.cells for r in results}), args.write_table)
        if stale is not None:
            print(f"gatesim: {stale}")
            print("gatesim: FAILED")
            return 1
        if args.write_table:
            print(f"gatesim: wrote the coverage table into {README.relative_to(REPO).as_posix()}")
    print("gatesim: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
