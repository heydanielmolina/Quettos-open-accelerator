# qcore_csr synthesis (xc7)

Every number below is read back out of `build/synth/synth_csr.log` by
`scripts/synth_report.py`, which `make synth` runs. The run comes first and this page is
written from it, so the two cannot disagree. `--check` regenerates the page and requires it
back byte for byte on the Yosys build named below; another build has its own LUT packing and
path lengths, so it is held to the parameters and the hard-block inventory.

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`

```sh
mkdir -p build/synth
yosys -q -l build/synth/synth_csr.log -s syn/synth_csr.ys
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

`qcore_csr` takes no parameters.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `CARRY4` | 32 |
| `FDRE` | 370 |
| `IBUF` | 1212 |
| `LUT2` | 63 |
| `LUT3` | 62 |
| `LUT4` | 53 |
| `LUT5` | 68 |
| `LUT6` | 397 |
| `MUXF7` | 75 |
| `MUXF8` | 11 |
| `OBUF` | 163 |

2507 cells in total: 643 LUTs, 370 flops, 32 `CARRY4`, 1375 I/O pads. Yosys estimates 580
LCs.

No hard blocks: the design has no `DSP48E1`, `RAMB36E1`, `RAMB18E1`, `RAM32M`, `RAM64M` or
`RAM128X1D` cell.

`memory_libmap` mapped no memory: the block holds its state in flops.

Longest topological path through the LUT fabric: 9 cells.

```
0  $techmap16682$abc$15171$auto$blifparse.cc:557:parse_blif$15176.A [1]
9  $auto$alumacc.cc:512:replace_alu$1418.CO [28]
```

## Notes (hand-written)

The area is the read path, not the state. Every register the host can see fits in a few
hundred flops; what costs LUTs is `csr_rdata`, which selects a 32-bit slice out of the
1024-bit `perf_snap` input, so a kilobit-wide mux narrows to one word. Most of the cells are
`IBUF` pads carrying that snapshot in, and inside `qcore_top` it arrives on internal wires and
those pads disappear.
