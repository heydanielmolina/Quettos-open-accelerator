# qcore_vpu_lane synthesis (xc7)

Every number below is read back out of `build/synth/synth_vpu_lane.log` by
`scripts/synth_report.py`, which `make synth` runs. The run comes first and this page is
written from it, so the two cannot disagree. `--check` regenerates the page and requires it
back byte for byte on the Yosys build named below; another build has its own LUT packing and
path lengths, so it is held to the parameters and the hard-block inventory.

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`

```sh
mkdir -p build/synth
yosys -q -l build/synth/synth_vpu_lane.log -s syn/synth_vpu_lane.ys
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

`qcore_vpu_lane` takes no parameters.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `CARRY4` | 78 |
| `DSP48E1` | 8 |
| `FDRE` | 286 |
| `IBUF` | 108 |
| `INV` | 54 |
| `LUT1` | 93 |
| `LUT2` | 165 |
| `LUT3` | 222 |
| `LUT4` | 125 |
| `LUT5` | 100 |
| `LUT6` | 205 |
| `MUXF7` | 142 |
| `MUXF8` | 49 |
| `OBUF` | 98 |

1734 cells in total: 910 LUTs, 286 flops, 78 `CARRY4`, 206 I/O pads. Yosys estimates 652
LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `DSP48E1` | 4 | `$mul` at `rtl/qcore_vpu_lane.sv:70` |
| `DSP48E1` | 2 | `$mul` at `rtl/qcore_vpu_lane.sv:68` |
| `DSP48E1` | 2 | `$mul` at `rtl/qcore_vpu_lane.sv:69` |

`memory_libmap` mapped no memory: the block holds its state in flops.

Longest topological path through the LUT fabric: 30 cells.

```
 0  sh1 [3]
24  rs64 [56]
30  y_n [0]
```

## Notes (hand-written)

The eight DSP48E1 are the arithmetic: four carry the 32 x 32 product of `L_MUL32`, which is
also what `p64` hands the sum-of-squares pass, and two each carry the 32 x 17 products the
16-bit-coefficient ops multiply. Everything else is the second stage -- the 64-bit and
49-bit round-shift, whose rounding increment is the `CARRY4` chain, the saturation reduction
and the result select -- and the pipeline registers that hold a product and its shift for a
cycle. The path depth is that second stage: a barrel shift, a carry chain and a clamp between
two flops, the same shape as the requant's stage-2.
