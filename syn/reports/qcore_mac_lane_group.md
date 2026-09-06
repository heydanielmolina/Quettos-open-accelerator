# qcore_mac_lane_group synthesis (xc7)

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`.

Method: Yosys `synth_xilinx -family xc7` of the block on its own; the cell counts are the
post-synthesis `stat -tech xilinx` table of the top module (I/O buffers included: the block's
ports become pads when it is synthesized alone). `make synth` reruns every `syn/*.ys` script;
the exact command of each run is quoted with its table.

## ACC_W = 40 (the one parameter)

```sh
yosys -q -l build/synth/synth_lane_group.log -s syn/synth_lane_group.ys
```

| Cell | Count |
|---|---|
| `DSP48E1` | 8 |
| `FDRE` | 320 |
| `LUT2` | 81 |
| `LUT3` | 320 |
| `BUFG` | 1 |
| `IBUF` | 85 |
| `OBUF` | 320 |

1135 cells in total, 401 LUTs, estimated 320 LCs.

## Inference check

`build/synth/qcore_mac_lane_group_dsp.txt` (the `dump t:DSP48E1` of the script) lists eight DSP48E1 cells,
one per lane, each with `USE_MULT MULTIPLY`, `PREG 1`, `P` driving the lane's `acc` register,
`OPMODE {2'01, tile_start-derived bit, 4'0101}` (X/Y = the multiplier, Z = P or the C override) and
`CEP` from `en`: the accumulate adder, the tile restart and the EMBED load through `C` are absorbed by the
DSP block, so no CARRY4 remains. The 320 FDRE are the eight 40-bit hold sets; the LUTs form the
`acc_drain` select and the sign extension of the C input. No block RAM.
