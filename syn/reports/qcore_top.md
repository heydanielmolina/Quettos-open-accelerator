# qcore_top synthesis (xc7)

Every number below is read back out of `build/synth/synth_top.log` by
`scripts/synth_report.py`, which `make synth` runs. The run comes first and this page is
written from it, so the two cannot disagree. `--check` regenerates the page and requires it
back byte for byte on the Yosys build named below; another build has its own LUT packing and
path lengths, so it is held to the parameters and the hard-block inventory.

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`

```sh
mkdir -p build/synth
yosys -q -l build/synth/synth_top.log -s syn/synth_top.ys
```

The cell counts are the `stat -tech xilinx` table after `synth_xilinx -family xc7`. A block
synthesized on its own has its ports on pads, so `IBUF` and `OBUF` are in its totals; inside
`qcore_top` they are internal wires.

The path depth is `ltp -noff` over the LUT fabric, cutting at the flops (`FDRE`, `FDSE`),
the clock buffer (`BUFG`), the I/O buffers (`IBUF`, `OBUF`) and the hard blocks (`DSP48E1`,
`RAMB36E1`, `RAMB18E1`, `RAM32M`, `RAM64M`, `RAM128X1D`), so it counts logic levels between
registers and means the same thing in every block. Each path below lists its two endpoints
and the named signals and source lines between them, with the position of each along the
path.

## Demo configuration

Parameters: `WB = 64`, `B_MAX = 1`, `VSRAM_WORDS = 4096`, `FIFO_BEATS = 128`, `ACC_W = 40`,
`META_FIFO_BEATS = 16`, `MAX_BURST = 64`, `DQ_DEPTH = 8`, `VL = 4`, `VPU_FIFO_BEATS = 16`.

| Cell | Count |
|---|---|
| `$scopeinfo` | 36 |
| `BUFG` | 1 |
| `CARRY4` | 1643 |
| `DSP48E1` | 109 |
| `FDRE` | 22371 |
| `FDSE` | 66 |
| `IBUF` | 563 |
| `INV` | 638 |
| `LUT1` | 28 |
| `LUT2` | 6468 |
| `LUT3` | 10948 |
| `LUT4` | 2724 |
| `LUT5` | 4364 |
| `LUT6` | 10978 |
| `MUXF7` | 644 |
| `MUXF8` | 135 |
| `OBUF` | 686 |
| `RAM32M` | 359 |
| `RAMB18E1` | 15 |
| `RAMB36E1` | 32 |
| `SRL16E` | 354 |

63162 cells in total (36 of them `$scopeinfo` hierarchy markers, which map to nothing):
35510 LUTs, 22437 flops, 1643 `CARRY4`, 1249 I/O pads. Yosys estimates 29014 LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `DSP48E1` | 64 | `g_row[*].u_row.g_grp[*].u_grp`, `$mul` at `rtl/qcore_mac_lane_group.sv:35` |
| `DSP48E1` | 16 | `u_vpu.g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:70` |
| `DSP48E1` | 8 | `u_vpu.g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:68` |
| `DSP48E1` | 8 | `u_vpu.g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:69` |
| `DSP48E1` | 4 | `u_vpu.g_interp[*].u_interp`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `DSP48E1` | 2 | `u_requant`, `$mul` at `rtl/qcore_requant.sv:519` |
| `DSP48E1` | 2 | `u_requant`, `$mul` at `rtl/qcore_requant.sv:572` |
| `DSP48E1` | 2 | `u_vpu.u_scalar`, `$mul` at `rtl/qcore_pkg.sv:151` |
| `DSP48E1` | 1 | `u_dispatch`, `$mul` at `rtl/qcore_seq_dispatch.sv:299` |
| `DSP48E1` | 1 | `u_vpu.u_scalar.u_rcp_i`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `DSP48E1` | 1 | `u_vpu.u_scalar.u_rsq_i`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `RAMB36E1` | 32 | `g_row[*].u_vsram.mem` |
| `RAMB18E1` | 15 | `u_stream.wmem` |
| `RAM32M` | 102 | `u_requant.dq_mem` |
| `RAM32M` | 86 | `u_vpu.fmem` |
| `RAM32M` | 85 | `u_fetch.gmem` |
| `RAM32M` | 80 | `u_stream.mmem` |
| `RAM32M` | 6 | `g_row[*].u_row.sreg` |

Memories, as `memory_libmap` mapped them:

- `qcore_top.g_row[0].u_row.sreg` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.g_row[0].u_vsram.mem` via `$__XILINX_BLOCKRAM_TDP_`
- `qcore_top.u_fetch.gmem` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.u_requant.dq_mem` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.u_stream.mmem` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.u_stream.wmem` via `$__XILINX_BLOCKRAM_SDP_`
- `qcore_top.u_vpu.fmem` via `$__XILINX_LUTRAM_SDP_`

Longest topological path through the LUT fabric: 41 cells.

```
 0  u_csr.pos_q [0]
10  u_dispatch.pos_p1 [32]
12  rtl/qcore_seq_dispatch.sv:243
13  rtl/qcore_seq_dispatch.sv:242
22  u_dispatch.n_ru [24]
26  rtl/qcore_seq_dispatch.sv:246
35  u_dispatch.n_pad [24]
41  u_dispatch.wt_c [37]
```

## Tiny configuration

Parameters: `WB = 16`, `B_MAX = 2`, `VSRAM_WORDS = 2048`, `FIFO_BEATS = 128`, `ACC_W = 40`,
`META_FIFO_BEATS = 16`, `MAX_BURST = 64`, `DQ_DEPTH = 8`, `VL = 2`, `VPU_FIFO_BEATS = 16`.

| Cell | Count |
|---|---|
| `$scopeinfo` | 28 |
| `BUFG` | 1 |
| `CARRY4` | 1380 |
| `DSP48E1` | 59 |
| `FDRE` | 18985 |
| `FDSE` | 60 |
| `IBUF` | 179 |
| `INV` | 529 |
| `LUT1` | 36 |
| `LUT2` | 4635 |
| `LUT3` | 6415 |
| `LUT4` | 1847 |
| `LUT5` | 2876 |
| `LUT6` | 8794 |
| `MUXF7` | 365 |
| `MUXF8` | 91 |
| `OBUF` | 254 |
| `RAM32M` | 126 |
| `RAMB36E1` | 34 |
| `SRL16E` | 182 |

46876 cells in total (28 of them `$scopeinfo` hierarchy markers, which map to nothing):
24603 LUTs, 19045 flops, 1380 `CARRY4`, 433 I/O pads. Yosys estimates 19932 LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `DSP48E1` | 32 | `g_row[*].u_row.g_grp[*].u_grp`, `$mul` at `rtl/qcore_mac_lane_group.sv:35` |
| `DSP48E1` | 8 | `u_vpu.g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:70` |
| `DSP48E1` | 4 | `u_vpu.g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:68` |
| `DSP48E1` | 4 | `u_vpu.g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:69` |
| `DSP48E1` | 2 | `u_requant`, `$mul` at `rtl/qcore_requant.sv:519` |
| `DSP48E1` | 2 | `u_requant`, `$mul` at `rtl/qcore_requant.sv:572` |
| `DSP48E1` | 2 | `u_vpu.g_interp[*].u_interp`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `DSP48E1` | 2 | `u_vpu.u_scalar`, `$mul` at `rtl/qcore_pkg.sv:151` |
| `DSP48E1` | 1 | `u_dispatch`, `$mul` at `rtl/qcore_seq_dispatch.sv:299` |
| `DSP48E1` | 1 | `u_vpu.u_scalar.u_rcp_i`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `DSP48E1` | 1 | `u_vpu.u_scalar.u_rsq_i`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `RAMB36E1` | 32 | `g_row[*].u_vsram.mem` |
| `RAMB36E1` | 2 | `u_stream.wmem` |
| `RAM32M` | 42 | `u_fetch.gmem` |
| `RAM32M` | 30 | `u_requant.dq_mem` |
| `RAM32M` | 22 | `u_vpu.fmem` |
| `RAM32M` | 20 | `u_stream.mmem` |
| `RAM32M` | 12 | `g_row[*].u_row.sreg` |

Memories, as `memory_libmap` mapped them:

- `qcore_top.g_row[0].u_row.sreg` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.g_row[0].u_vsram.mem` via `$__XILINX_BLOCKRAM_TDP_`
- `qcore_top.g_row[1].u_row.sreg` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.g_row[1].u_vsram.mem` via `$__XILINX_BLOCKRAM_TDP_`
- `qcore_top.u_fetch.gmem` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.u_requant.dq_mem` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.u_stream.mmem` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.u_stream.wmem` via `$__XILINX_BLOCKRAM_SDP_`
- `qcore_top.u_vpu.fmem` via `$__XILINX_LUTRAM_SDP_`

Longest topological path through the LUT fabric: 45 cells.

```
 0  u_csr.pos_q [0]
10  u_dispatch.pos_p1 [32]
12  rtl/qcore_seq_dispatch.sv:242
22  u_dispatch.n_ru [24]
25  rtl/qcore_seq_dispatch.sv:246
29  u_dispatch.n_dec [6]
35  u_dispatch.n_pad [24]
42  u_dispatch.wt_row [39]
45  u_dispatch.wt_c [36]
```

## Notes (hand-written)

The array and the vector unit split the fabric. The MAC lanes and both vector lanes are DSP
blocks and the VSRAMs are block RAM, so the LUTs that are left are the requant, the vector
unit's element pipeline and write staging, the KV writer and the stream controller. The
longest path is the POS derivation inside the dispatcher, the same chain
`syn/reports/qcore_seq_dispatch.md` shows for that block on its own; nothing the vector unit
adds is longer, and `syn/reports/qcore_vpu_top.md` gives its own deepest path, the softmax
exponential at the issue, as the shorter one.

`qcore_vpu_top` is instantiated here, so the vector unit is inside these numbers. `qcore_top`
takes no parameter that leaves it out, so the core before it carried one is a commit rather
than a configuration of `syn/synth_top.ys`: commit `26064b4`, whose own
`syn/reports/qcore_top.md` the Yosys build named above wrote, records 11484 estimated LCs in
the demo configuration (30835 cells, 13747 LUTs, 13945 flops, 69 `DSP48E1`) and 9407 in the
tiny one (25535 cells, 11379 LUTs, 12136 flops, 37 `DSP48E1`). That page is
`git show 26064b4:syn/reports/qcore_top.md`, and re-running `syn/synth_top.ys` over that
commit's `rtl/` reproduces both figures. Read against the estimated-LC lines of the two
tables above, that is what the vector unit costs, in each configuration, without a second
number being typed anywhere. The added DSPs are eight per `qcore_vpu_lane`, one per table
interpolator and four in `qcore_vpu_scalar`. The vector unit's QMEM operand FIFO is the one new
memory, in LUT RAM; its four lookup tables stay in the LUT fabric, which is why the block RAM
count does not move.

The VROPE and VSOFTMAX passes are the difference between this page and the one commit
`b9e3c5c` carries, whose vector unit executed the other four V opcodes: `git show
b9e3c5c:syn/reports/qcore_top.md` records 23026 estimated LCs in the demo configuration
(53177 cells, 28778 LUTs, 19545 flops, 101 `DSP48E1`) and 16317 in the tiny one (39669
cells, 20113 LUTs, 16612 flops, 55 `DSP48E1`). Read against the estimated-LC lines above,
the two operations cost 6017 LCs and eight `DSP48E1` at the demo width and 3675 LCs and four
`DSP48E1` at the tiny one. Every added DSP is the rotation's second 32 x 17 product, two per
`qcore_vpu_lane`; the table interpolators are unchanged, because the exp2 image shares the
instances the sigmoid already had. The longest path does not move to either pass: it is the
dispatcher's POS derivation at both widths, and the softmax exponential is the deeper of the
two new chains at a depth `syn/reports/qcore_vpu_top.md` gives.
