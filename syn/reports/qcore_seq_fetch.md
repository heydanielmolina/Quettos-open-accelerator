# qcore_seq_fetch synthesis (xc7)

Every number below is read back out of `build/synth/synth_seq_fetch.log` by
`scripts/synth_report.py`, which `make synth` runs. The run comes first and this page is
written from it, so the two cannot disagree. `--check` regenerates the page and requires it
back byte for byte on the Yosys build named below; another build has its own LUT packing and
path lengths, so it is held to the parameters and the hard-block inventory.

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`

```sh
mkdir -p build/synth
yosys -q -l build/synth/synth_seq_fetch.log -s syn/synth_seq_fetch.ys
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

Parameters: `WB = 64`, `DQ_DEPTH = 8`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `CARRY4` | 14 |
| `FDRE` | 594 |
| `IBUF` | 554 |
| `INV` | 13 |
| `LUT2` | 21 |
| `LUT3` | 291 |
| `LUT4` | 5 |
| `LUT5` | 10 |
| `LUT6` | 6 |
| `OBUF` | 307 |
| `RAM32M` | 86 |

1902 cells in total: 333 LUTs, 594 flops, 14 `CARRY4`, 861 I/O pads. Yosys estimates 312
LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `RAM32M` | 86 | `gmem` |

Memories, as `memory_libmap` mapped them:

- `qcore_seq_fetch.gmem` via `$__XILINX_LUTRAM_SDP_`

Longest topological path through the LUT fabric: 9 cells.

```
0  req_addr [6]
9  $abc$5110$procmux$1360_Y[30]
```

## Tiny configuration

Parameters: `WB = 16`, `DQ_DEPTH = 8`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `CARRY4` | 18 |
| `FDRE` | 469 |
| `IBUF` | 170 |
| `INV` | 15 |
| `LUT2` | 25 |
| `LUT3` | 41 |
| `LUT4` | 4 |
| `LUT5` | 2 |
| `LUT6` | 7 |
| `MUXF7` | 1 |
| `OBUF` | 307 |
| `RAM32M` | 43 |

1103 cells in total: 79 LUTs, 469 flops, 18 `CARRY4`, 477 I/O pads. Yosys estimates 54 LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `RAM32M` | 43 | `gmem` |

Memories, as `memory_libmap` mapped them:

- `qcore_seq_fetch.gmem` via `$__XILINX_LUTRAM_SDP_`

Longest topological path through the LUT fabric: 9 cells.

```
0  req_addr [5]
9  $abc$10019$procmux$7788_Y[29]
```

## Notes (hand-written)

The descriptor queue is distributed RAM, and its cost follows the group width rather than the
depth: a `RAM32M` is 32 entries deep however few of those entries are used, so the same bits
of queue take twice the cells when a group carries two descriptors instead of one. The LUTs of
the wide build are the lane select that picks which descriptor of a group reaches the decode;
at the narrow width a group is one descriptor and that mux disappears.
