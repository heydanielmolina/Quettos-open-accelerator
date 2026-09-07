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
| `$scopeinfo` | 15 |
| `BUFG` | 1 |
| `CARRY4` | 710 |
| `DSP48E1` | 32 |
| `FDRE` | 5457 |
| `FDSE` | 11 |
| `IBUF` | 1260 |
| `INV` | 172 |
| `LUT1` | 49 |
| `LUT2` | 3120 |
| `LUT3` | 4289 |
| `LUT4` | 1185 |
| `LUT5` | 2348 |
| `LUT6` | 3831 |
| `MUXF7` | 690 |
| `MUXF8` | 98 |
| `OBUF` | 407 |
| `RAM32M` | 86 |
| `SRL16E` | 392 |

24153 cells in total (15 of them `$scopeinfo` hierarchy markers, which map to nothing):
14822 LUTs, 5468 flops, 710 `CARRY4`, 1667 I/O pads. Yosys estimates 11653 LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `DSP48E1` | 16 | `g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:70` |
| `DSP48E1` | 8 | `g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:68` |
| `DSP48E1` | 4 | `g_sig_interp[*].u_interp`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `DSP48E1` | 2 | `u_scalar`, `$mul` at `rtl/qcore_pkg.sv:151` |
| `DSP48E1` | 1 | `u_scalar.u_rcp_i`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `DSP48E1` | 1 | `u_scalar.u_rsq_i`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `RAM32M` | 86 | `fmem` |

Memories, as `memory_libmap` mapped them:

- `qcore_vpu_top.fmem` via `$__XILINX_LUTRAM_SDP_`

Longest topological path through the LUT fabric: 32 cells.

```
 0  p_ord [2]
 3  rtl/qcore_vpu_top.sv:317
 7  issue_mask [0]
 8  smask [16]
32  $auto$alumacc.cc:512:replace_alu$12686.CO [52]
```

## Tiny configuration

Parameters: `WB = 16`, `B_MAX = 2`, `VL = 2`, `VSRAM_WORDS = 2048`, `VPU_FIFO_BEATS = 16`,
`MAX_BURST = 64`.

| Cell | Count |
|---|---|
| `$scopeinfo` | 10 |
| `BUFG` | 1 |
| `CARRY4` | 489 |
| `DSP48E1` | 18 |
| `FDRE` | 4312 |
| `FDSE` | 9 |
| `IBUF` | 909 |
| `INV` | 137 |
| `LUT1` | 37 |
| `LUT2` | 1812 |
| `LUT3` | 2605 |
| `LUT4` | 569 |
| `LUT5` | 1343 |
| `LUT6` | 2850 |
| `MUXF7` | 762 |
| `MUXF8` | 96 |
| `OBUF` | 405 |
| `RAM32M` | 22 |
| `SRL16E` | 196 |

16582 cells in total (10 of them `$scopeinfo` hierarchy markers, which map to nothing): 9216
LUTs, 4321 flops, 489 `CARRY4`, 1314 I/O pads. Yosys estimates 7367 LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `DSP48E1` | 8 | `g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:70` |
| `DSP48E1` | 4 | `g_lane[*].u_lane`, `$mul` at `rtl/qcore_vpu_lane.sv:68` |
| `DSP48E1` | 2 | `g_sig_interp[*].u_interp`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `DSP48E1` | 2 | `u_scalar`, `$mul` at `rtl/qcore_pkg.sv:151` |
| `DSP48E1` | 1 | `u_scalar.u_rcp_i`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `DSP48E1` | 1 | `u_scalar.u_rsq_i`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `RAM32M` | 22 | `fmem` |

Memories, as `memory_libmap` mapped them:

- `qcore_vpu_top.fmem` via `$__XILINX_LUTRAM_SDP_`

Longest topological path through the LUT fabric: 33 cells.

```
 0  g_lane[1].u_lane.op1 [2]
25  g_lane[1].u_lane.rs64 [60]
33  g_lane[1].u_lane.y_n [0]
```

## Notes (hand-written)

The multipliers are the hard blocks: six `DSP48E1` per `qcore_vpu_lane` at both widths --
four for the 32 x 32 product of `L_MUL32` and two for one of its two 32 x 17 products, where
the lane synthesized on its own takes eight and puts the second 32 x 17 product in a DSP as
well (`syn/reports/qcore_vpu_lane.md`) -- one per sigmoid interpolator for `dv * frac8`, and
four in the scalar unit, one in each of its two interpolators and two for its `sfloat_mul`
products: 32 at `VL = 4` and 18 at `VL = 2`. The QMEM operand FIFO is the only memory:
`memory_libmap` puts its `VPU_FIFO_BEATS x WB*8` bits in distributed RAM (`RAM32M`), and the
`SRL16E` are the same FIFO's read-side pipelining, so no block RAM is inferred anywhere and
the three lookup tables -- sigmoid in `ceil(VL/2)` instances here, rsqrt and recip inside the
scalar unit -- stay in the LUT fabric exactly as `syn/reports/qcore_lut_rom.md` records.

The flops are the nine-stage chunk pipeline: the operands travel with the chunk (`sopa`
three stages, `sopb` six, the trip-A result three), the two 256-bit window registers of each
operand stream and the 512-bit write staging register account for most of the rest, and they
scale with `VL` and with `WB` through the FIFO alone. The longest path at both widths is
inside a lane: from a stage-1 register (`op1`, the held opcode) through `rs64`, the 64-bit
round-shift, to `y_n`, the saturating result select -- `qcore_vpu_lane`'s own second stage,
which `syn/reports/qcore_vpu_lane.md` measures at 30 cells on the block alone. The 33 cells
at `VL = 4` and 31 at `VL = 2` are that same stage as the flattened design packs it, so
nothing this module wraps around the lanes is deeper than the block it already contains.
