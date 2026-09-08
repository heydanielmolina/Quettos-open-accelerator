# qcore_vpu_top synthesis (xc7)

Every number below is read back out of `build/synth/synth_vpu_top.log` by
`scripts/synth_report.py`, which `make synth` runs. The run comes first and this page is
written from it, so the two cannot disagree. `--check` regenerates the page and requires it
back byte for byte on the Yosys build named below; another build has its own LUT packing and
path lengths, so it is held to the parameters and the hard-block inventory.

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`

```sh
mkdir -p build/synth
yosys -q -l build/synth/synth_vpu_top.log -s syn/synth_vpu_top.ys
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

Parameters: `WB = 64`, `B_MAX = 1`, `VL = 4`, `VSRAM_WORDS = 4096`, `VPU_FIFO_BEATS = 16`,
`MAX_BURST = 64`.

| Cell | Count |
|---|---|
| `$scopeinfo` | 17 |
| `BUFG` | 1 |
| `CARRY4` | 999 |
| `DSP48E1` | 40 |
| `FDRE` | 8305 |
| `FDSE` | 62 |
| `IBUF` | 1316 |
| `INV` | 444 |
| `LUT1` | 44 |
| `LUT2` | 4140 |
| `LUT3` | 5611 |
| `LUT4` | 1777 |
| `LUT5` | 3244 |
| `LUT6` | 6973 |
| `MUXF7` | 823 |
| `MUXF8` | 167 |
| `OBUF` | 407 |
| `RAM32M` | 86 |
| `SRL16E` | 354 |

34810 cells in total (17 of them `$scopeinfo` hierarchy markers, which map to nothing):
21789 LUTs, 8367 flops, 999 `CARRY4`, 1723 I/O pads. Yosys estimates 17605 LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `DSP48E1` | 16 | `g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:70` |
| `DSP48E1` | 8 | `g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:68` |
| `DSP48E1` | 8 | `g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:69` |
| `DSP48E1` | 4 | `g_interp[*].u_interp`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `DSP48E1` | 2 | `u_scalar`, `$mul` at `rtl/qcore_pkg.sv:151` |
| `DSP48E1` | 1 | `u_scalar.u_rcp_i`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `DSP48E1` | 1 | `u_scalar.u_rsq_i`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `RAM32M` | 86 | `fmem` |

Memories, as `memory_libmap` mapped them:

- `qcore_vpu_top.fmem` via `$__XILINX_LUTRAM_SDP_`

Longest topological path through the LUT fabric: 37 cells.

```
 0  wa0 [227]
12  rtl/qcore_vpu_top.sv:834
37  $\g_rom[0].u_exp.mem$rdreg[1]$d [17]
```

## Tiny configuration

Parameters: `WB = 16`, `B_MAX = 2`, `VL = 2`, `VSRAM_WORDS = 2048`, `VPU_FIFO_BEATS = 16`,
`MAX_BURST = 64`.

| Cell | Count |
|---|---|
| `$scopeinfo` | 11 |
| `BUFG` | 1 |
| `CARRY4` | 666 |
| `DSP48E1` | 22 |
| `FDRE` | 6705 |
| `FDSE` | 56 |
| `IBUF` | 965 |
| `INV` | 293 |
| `LUT1` | 14 |
| `LUT2` | 2347 |
| `LUT3` | 3358 |
| `LUT4` | 1031 |
| `LUT5` | 1375 |
| `LUT6` | 4725 |
| `MUXF7` | 501 |
| `MUXF8` | 129 |
| `OBUF` | 405 |
| `RAM32M` | 22 |
| `SRL16E` | 182 |

22808 cells in total (11 of them `$scopeinfo` hierarchy markers, which map to nothing):
12850 LUTs, 6761 flops, 666 `CARRY4`, 1370 I/O pads. Yosys estimates 10489 LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `DSP48E1` | 8 | `g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:70` |
| `DSP48E1` | 4 | `g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:68` |
| `DSP48E1` | 4 | `g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:69` |
| `DSP48E1` | 2 | `g_interp[*].u_interp`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `DSP48E1` | 2 | `u_scalar`, `$mul` at `rtl/qcore_pkg.sv:151` |
| `DSP48E1` | 1 | `u_scalar.u_rcp_i`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `DSP48E1` | 1 | `u_scalar.u_rsq_i`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `RAM32M` | 22 | `fmem` |

Memories, as `memory_libmap` mapped them:

- `qcore_vpu_top.fmem` via `$__XILINX_LUTRAM_SDP_`

Longest topological path through the LUT fabric: 35 cells.

```
 0  wa1 [3]
11  sm_sub [28]
12  rtl/qcore_vpu_top.sv:834
26  sm_f16 [12]
35  $auto$xilinx_dffopt.cc:347:execute$447609
```

## Notes (hand-written)

The multipliers are the hard blocks: eight `DSP48E1` per `qcore_vpu_lane` at both widths --
four for the 32 x 32 product of `L_MUL32` and two for each of its two 32 x 17 products, the
eight `syn/reports/qcore_vpu_lane.md` measures on the lane alone, since the rotation gives the
second 32 x 17 product a coefficient of its own -- one per table interpolator for
`dv * frac8`, and four in the scalar unit, one in each of its two interpolators and two for
its `sfloat_mul` products: 40 at `VL = 4` and 22 at `VL = 2`. The QMEM operand FIFO is the
only memory: `memory_libmap` puts its `VPU_FIFO_BEATS x WB*8` bits in distributed RAM
(`RAM32M`), and the `SRL16E` are the same FIFO's read-side pipelining, so no block RAM is
inferred anywhere and the four lookup tables -- sigmoid and exp2 in `ceil(VL/2)` instances
each here, rsqrt and recip inside the scalar unit -- stay in the LUT fabric exactly as
`syn/reports/qcore_lut_rom.md` records.

The flops are the nine-stage chunk pipeline: the operands travel with the chunk (`sopa` and
`sopb` six stages each, the per-element shift and the rotation's pair index six, the trip-A
result three), and the two 256-bit window registers of the operand streams, the two 512-bit
write staging registers a rotated head fills at once and the 1024-bit RoPE table row account
for most of the rest; they scale with `VL`, and with `WB` through the FIFO alone. The longest
path at both widths is the softmax exponential at the issue: an operand word out of the
port-A window, the 33-bit distance below the row's score maximum, its clamp, and the 16-bit
fraction that addresses the exp2 table, whose index register the sigmoid ROM shares. The two
tables above give its depth in each configuration; it is a few cells deeper than the lane's
own second stage (`syn/reports/qcore_vpu_lane.md`, 30 cells), and the one place where this
module is deeper than a block it contains.
