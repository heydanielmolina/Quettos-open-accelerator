# qcore_vpu_scalar synthesis (xc7)

Every number below is read back out of `build/synth/synth_vpu_scalar.log` by
`scripts/synth_report.py`, which `make synth` runs. The run comes first and this page is
written from it, so the two cannot disagree. `--check` regenerates the page and requires it
back byte for byte on the Yosys build named below; another build has its own LUT packing and
path lengths, so it is held to the parameters and the hard-block inventory.

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`

```sh
mkdir -p build/synth
yosys -q -l build/synth/synth_vpu_scalar.log -s syn/synth_vpu_scalar.ys
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

## Default configuration

`qcore_vpu_scalar` takes no parameters.

| Cell | Count |
|---|---|
| `$scopeinfo` | 4 |
| `BUFG` | 1 |
| `CARRY4` | 71 |
| `DSP48E1` | 4 |
| `FDRE` | 551 |
| `FDSE` | 3 |
| `IBUF` | 109 |
| `INV` | 31 |
| `LUT1` | 9 |
| `LUT2` | 207 |
| `LUT3` | 313 |
| `LUT4` | 133 |
| `LUT5` | 75 |
| `LUT6` | 487 |
| `MUXF7` | 169 |
| `MUXF8` | 44 |
| `OBUF` | 49 |

2260 cells in total (4 of them `$scopeinfo` hierarchy markers, which map to nothing): 1224
LUTs, 554 flops, 71 `CARRY4`, 158 I/O pads. Yosys estimates 1008 LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `DSP48E1` | 2 | `$mul` at `rtl/qcore_pkg.sv:151` |
| `DSP48E1` | 1 | `u_rcp_i`, `$mul` at `rtl/qcore_lut_interp.sv:25` |
| `DSP48E1` | 1 | `u_rsq_i`, `$mul` at `rtl/qcore_lut_interp.sv:25` |

`memory_libmap` mapped no memory: the block holds its state in flops.

Longest topological path through the LUT fabric: 19 cells.

```
 0  $auto$alumacc.cc:512:replace_alu$2962.A
14  s1_raw [9]
19  rtl/qcore_vpu_scalar.sv:309
```

## Notes (hand-written)

The two tables are the area: with their images resolved at read time a ROM has no writes, so
Yosys folds each one into the LUT fabric rather than a block RAM -- the same mapping
`syn/reports/qcore_lut_rom.md` records for the tables on their own -- and the `LUT6` and
`MUXF7` mass here is those 24576 bits of curve. The four DSP48E1 are the small multipliers:
one in each interpolator for `dv * frac8`, and one for each of the two `sfloat_mul` products,
the reciprocal-square-root scale and the VQUANT scale. The flops are the five pipeline stages
carrying the request's classes and constants alongside the table lookup, and the longest path
is the exponent arithmetic that forms `S1` and clamps it into `[0, 63]`.
