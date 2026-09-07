# qcore_vsram synthesis (xc7)

Every number below is read back out of `build/synth/synth_vsram.log` by
`scripts/synth_report.py`, which `make synth` runs. The run comes first and this page is
written from it, so the two cannot disagree. `--check` regenerates the page and requires it
back byte for byte on the Yosys build named below; another build has its own LUT packing and
path lengths, so it is held to the parameters and the hard-block inventory.

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`

```sh
mkdir -p build/synth
yosys -q -l build/synth/synth_vsram.log -s syn/synth_vsram.ys
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

Parameters: `WORDS = 4096`, `W = 256`, `NE = 8`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `FDRE` | 257 |
| `IBUF` | 291 |
| `LUT3` | 256 |
| `OBUF` | 512 |
| `RAMB36E1` | 32 |

1349 cells in total: 256 LUTs, 257 flops, 803 I/O pads. Yosys estimates 256 LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `RAMB36E1` | 32 | `mem` |

Memories, as `memory_libmap` mapped them:

- `qcore_vsram.mem` via `$__XILINX_BLOCKRAM_TDP_`

Longest topological path through the LUT fabric: 1 cell.

```
0  $techmap3879$abc$3107$auto$blifparse.cc:557:parse_blif$3108.A [1]
1  rd_b[254]
```

## Tiny configuration

Parameters: `WORDS = 2048`, `W = 256`, `NE = 8`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `FDRE` | 257 |
| `IBUF` | 289 |
| `LUT3` | 256 |
| `OBUF` | 512 |
| `RAMB36E1` | 16 |

1331 cells in total: 256 LUTs, 257 flops, 801 I/O pads. Yosys estimates 256 LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `RAMB36E1` | 16 | `mem` |

Memories, as `memory_libmap` mapped them:

- `qcore_vsram.mem` via `$__XILINX_BLOCKRAM_TDP_`

Longest topological path through the LUT fabric: 1 cell.

```
0  $techmap7712$abc$6940$auto$blifparse.cc:557:parse_blif$7196.A [0]
1  rd_b[255]
```

## Notes (hand-written)

The RAM maps to block RAM at both depths, and halving the depth halves the `RAMB36E1` count
while the fabric around it stays the same: one flop per output bit, one more for the
read-enable hold, and the LUTs that keep the read data when the read enable is low.
