# qcore_lut_rom synthesis (xc7)

Every number below is read back out of `build/synth/synth_lut.log` by
`scripts/synth_report.py`, which `make synth` runs. The run comes first and this page is
written from it, so the two cannot disagree. `--check` regenerates the page and requires it
back byte for byte on the Yosys build named below; another build has its own LUT packing and
path lengths, so it is held to the parameters and the hard-block inventory.

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`

```sh
mkdir -p build/synth
yosys -q -l build/synth/synth_lut.log -s syn/synth_lut.ys
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

## Exp2 configuration

Parameters: `ENTRIES = 256`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `FDRE` | 44 |
| `FDSE` | 2 |
| `IBUF` | 19 |
| `LUT1` | 4 |
| `LUT2` | 6 |
| `LUT3` | 4 |
| `LUT4` | 4 |
| `LUT5` | 7 |
| `LUT6` | 146 |
| `MUXF7` | 80 |
| `MUXF8` | 40 |
| `OBUF` | 64 |

421 cells in total: 171 LUTs, 46 flops, 83 I/O pads. Yosys estimates 161 LCs.

No hard blocks: the design has no `DSP48E1`, `RAMB36E1`, `RAMB18E1`, `RAM32M`, `RAM64M` or
`RAM128X1D` cell.

`memory_libmap` mapped no memory: the block holds its state in flops.

Longest topological path through the LUT fabric: 3 cells.

```
0  $abc$6192$auto$blifparse.cc:557:parse_blif$6194.A [7]
3  $\mem$rdreg[1]$d [0]
```

## Sigmoid configuration

Parameters: `ENTRIES = 512`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `FDRE` | 48 |
| `FDSE` | 2 |
| `IBUF` | 21 |
| `LUT1` | 2 |
| `LUT2` | 14 |
| `LUT3` | 15 |
| `LUT4` | 10 |
| `LUT5` | 15 |
| `LUT6` | 170 |
| `MUXF7` | 100 |
| `MUXF8` | 28 |
| `OBUF` | 64 |

490 cells in total: 226 LUTs, 50 flops, 85 I/O pads. Yosys estimates 210 LCs.

No hard blocks: the design has no `DSP48E1`, `RAMB36E1`, `RAMB18E1`, `RAM32M`, `RAM64M` or
`RAM128X1D` cell.

`memory_libmap` mapped no memory: the block holds its state in flops.

Longest topological path through the LUT fabric: 6 cells.

```
0  $abc$14015$auto$blifparse.cc:557:parse_blif$14017.A [6]
6  $\mem$rdreg[1]$d [9]
```

## Rsqrt configuration

Parameters: `ENTRIES = 512`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `FDRE` | 42 |
| `FDSE` | 2 |
| `IBUF` | 21 |
| `INV` | 2 |
| `LUT1` | 14 |
| `LUT2` | 12 |
| `LUT3` | 16 |
| `LUT4` | 16 |
| `LUT5` | 26 |
| `LUT6` | 269 |
| `MUXF7` | 178 |
| `MUXF8` | 67 |
| `OBUF` | 64 |

730 cells in total: 353 LUTs, 44 flops, 85 I/O pads. Yosys estimates 327 LCs.

No hard blocks: the design has no `DSP48E1`, `RAMB36E1`, `RAMB18E1`, `RAM32M`, `RAM64M` or
`RAM128X1D` cell.

`memory_libmap` mapped no memory: the block holds its state in flops.

Longest topological path through the LUT fabric: 6 cells.

```
0  $abc$23976$auto$blifparse.cc:557:parse_blif$24025.A [6]
6  $\mem$rdreg[1]$d [6]
```

## Recip configuration

Parameters: `ENTRIES = 256`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `FDRE` | 42 |
| `FDSE` | 4 |
| `IBUF` | 19 |
| `INV` | 2 |
| `LUT1` | 2 |
| `LUT2` | 4 |
| `LUT3` | 3 |
| `LUT5` | 8 |
| `LUT6` | 149 |
| `MUXF7` | 84 |
| `MUXF8` | 40 |
| `OBUF` | 64 |

422 cells in total: 166 LUTs, 46 flops, 83 I/O pads. Yosys estimates 161 LCs.

No hard blocks: the design has no `DSP48E1`, `RAMB36E1`, `RAMB18E1`, `RAM32M`, `RAM64M` or
`RAM128X1D` cell.

`memory_libmap` mapped no memory: the block holds its state in flops.

Longest topological path through the LUT fabric: 3 cells.

```
0  $abc$30692$auto$blifparse.cc:557:parse_blif$30693.A [3]
3  $\mem$rdreg[1]$d [0]
```

## Notes (hand-written)

The table is a constant, so nothing here stores it: `memory_libmap` hands the array to
`memory_map`, which folds the contents into a LUT and `MUXF7` / `MUXF8` mux tree per output
bit. No block RAM is inferred, and the only registers are the two read ports. A bit that
every entry of a table agrees on becomes a constant and loses its flop, which is where the
flop counts below 2 x 32 come from: 9 such bits per port in `exp2` (the leading one of every
value, and the top byte of every delta, since the deltas run 89 to 177), 7 in `sigmoid`, 10
in `rsqrt` and 9 in `recip`.

The mux tree is the area, and its shape follows the table rather than the RTL. `rsqrt` is
the widest at 353 LUTs because its two 256-entry segments share no decode; `sigmoid` costs
less than its 512 entries suggest because 157 of them hold the same saturated 32768; the two
256-entry tables are the smallest and read through half the depth of the 512-entry ones,
3 cells against 6. One instance of each table comes to 859 LCs together.
