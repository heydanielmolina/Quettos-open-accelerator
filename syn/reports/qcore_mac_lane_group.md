# qcore_mac_lane_group synthesis (xc7)

Every number below is read back out of `build/synth/synth_lane_group.log` by
`scripts/synth_report.py`, which `make synth` runs. The run comes first and this page is
written from it, so the two cannot disagree. `--check` regenerates the page and requires it
back byte for byte on the Yosys build named below; another build has its own LUT packing and
path lengths, so it is held to the parameters and the hard-block inventory.

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`

```sh
mkdir -p build/synth
yosys -q -l build/synth/synth_lane_group.log -s syn/synth_lane_group.ys
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

Parameters: `ACC_W = 40`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `DSP48E1` | 8 |
| `FDRE` | 320 |
| `IBUF` | 85 |
| `LUT2` | 81 |
| `LUT3` | 320 |
| `OBUF` | 320 |

1135 cells in total: 401 LUTs, 320 flops, 405 I/O pads. Yosys estimates 320 LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `DSP48E1` | 8 | `$mul` at `rtl/qcore_mac_lane_group.sv:35` |

`memory_libmap` mapped no memory: the block holds its state in flops.

Longest topological path through the LUT fabric: 1 cell.

```
0  g_lane[0].acc [39]
1  acc_drain[39]
```

## Notes (hand-written)

The DSP blocks are the block. Each lane's multiply, its accumulation, the tile restart and the
EMBED load through the `C` port sit inside one DSP48E1 -- `build/synth/qcore_mac_lane_group_dsp.txt`,
the `dump t:DSP48E1` of the script, shows `USE_MULT MULTIPLY` with `PREG` set and the input
registers bypassed -- so no carry chain is left in the fabric and what remains is the
accumulator hold and the drain select.
