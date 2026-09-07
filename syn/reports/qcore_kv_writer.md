# qcore_kv_writer synthesis (xc7)

Every number below is read back out of `build/synth/synth_kv_writer.log` by
`scripts/synth_report.py`, which `make synth` runs. The run comes first and this page is
written from it, so the two cannot disagree. `--check` regenerates the page and requires it
back byte for byte on the Yosys build named below; another build has its own LUT packing and
path lengths, so it is held to the parameters and the hard-block inventory.

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`

```sh
mkdir -p build/synth
yosys -q -l build/synth/synth_kv_writer.log -s syn/synth_kv_writer.ys
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

Parameters: `WB = 64`, `B_MAX = 1`, `VSRAM_WORDS = 4096`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `CARRY4` | 46 |
| `FDRE` | 1869 |
| `FDSE` | 1 |
| `IBUF` | 414 |
| `INV` | 9 |
| `LUT1` | 2 |
| `LUT2` | 205 |
| `LUT3` | 163 |
| `LUT4` | 546 |
| `LUT5` | 63 |
| `LUT6` | 1169 |
| `MUXF7` | 56 |
| `MUXF8` | 15 |
| `OBUF` | 630 |

5189 cells in total: 2148 LUTs, 1870 flops, 46 `CARRY4`, 1044 I/O pads. Yosys estimates 1941
LCs.

No hard blocks: the design has no `DSP48E1`, `RAMB36E1`, `RAMB18E1`, `RAM32M`, `RAM64M` or
`RAM128X1D` cell.

`memory_libmap` mapped no memory: the block holds its state in flops.

Longest topological path through the LUT fabric: 10 cells.

```
 0  d_v_step [9]
10  $abc$24588$procmux$1344_Y[30]
```

## Tiny configuration

Parameters: `WB = 16`, `B_MAX = 2`, `VSRAM_WORDS = 2048`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `CARRY4` | 46 |
| `FDRE` | 1510 |
| `FDSE` | 1 |
| `IBUF` | 439 |
| `INV` | 9 |
| `LUT1` | 4 |
| `LUT2` | 253 |
| `LUT3` | 306 |
| `LUT4` | 100 |
| `LUT5` | 83 |
| `LUT6` | 1252 |
| `MUXF7` | 224 |
| `MUXF8` | 24 |
| `OBUF` | 197 |

4449 cells in total: 1998 LUTs, 1511 flops, 46 `CARRY4`, 636 I/O pads. Yosys estimates 1741
LCs.

No hard blocks: the design has no `DSP48E1`, `RAMB36E1`, `RAMB18E1`, `RAM32M`, `RAM64M` or
`RAM128X1D` cell.

`memory_libmap` mapped no memory: the block holds its state in flops.

Longest topological path through the LUT fabric: 10 cells.

```
 0  d_v_step [7]
10  $abc$55922$procmux$33259_Y[28]
```

## Notes (hand-written)

Almost none of this block is arithmetic. It is two byte shuffles -- the gathered read window
barrel-shifted to an element offset, and the transposed K beat rebuilt one byte at a time --
so the area is wide registered buffers and the LUTs that move bytes between them, with the
write-address step as the only carry chain. Both shuffles follow from the layout: the K cache
is transposed, and the source elements start at an arbitrary element index inside a VSRAM
word.
