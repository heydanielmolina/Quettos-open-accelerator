# qcore_perf synthesis (xc7)

Every number below is read back out of `build/synth/synth_perf.log` by
`scripts/synth_report.py`, which `make synth` runs. The run comes first and this page is
written from it, so the two cannot disagree. `--check` regenerates the page and requires it
back byte for byte on the Yosys build named below; another build has its own LUT packing and
path lengths, so it is held to the parameters and the hard-block inventory.

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`

```sh
mkdir -p build/synth
yosys -q -l build/synth/synth_perf.log -s syn/synth_perf.ys
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

Parameters: `WB = 64`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `CARRY4` | 255 |
| `FDRE` | 2036 |
| `IBUF` | 106 |
| `LUT2` | 102 |
| `LUT3` | 80 |
| `OBUF` | 1024 |

3604 cells in total: 182 LUTs, 2036 flops, 255 `CARRY4`, 1130 I/O pads. Yosys estimates 91
LCs.

No hard blocks: the design has no `DSP48E1`, `RAMB36E1`, `RAMB18E1`, `RAM32M`, `RAM64M` or
`RAM128X1D` cell.

`memory_libmap` mapped no memory: the block holds its state in flops.

Longest topological path through the LUT fabric: 17 cells.

```
 0  live [512]
17  $auto$alumacc.cc:512:replace_alu$1181.CO [60]
```

## Tiny configuration

Parameters: `WB = 16`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `CARRY4` | 255 |
| `FDRE` | 2040 |
| `IBUF` | 106 |
| `LUT2` | 102 |
| `LUT3` | 80 |
| `OBUF` | 1024 |

3608 cells in total: 182 LUTs, 2040 flops, 255 `CARRY4`, 1130 I/O pads. Yosys estimates 91
LCs.

No hard blocks: the design has no `DSP48E1`, `RAMB36E1`, `RAMB18E1`, `RAM32M`, `RAM64M` or
`RAM128X1D` cell.

`memory_libmap` mapped no memory: the block holds its state in flops.

Longest topological path through the LUT fabric: 17 cells.

```
 0  live [835]
17  $auto$alumacc.cc:512:replace_alu$10479.CO [60]
```

## Notes (hand-written)

This block is counters and nothing else: the live bank and its snapshot copy are almost all of
the flops, their wrapping adds are all of the `CARRY4`, and the LUTs are only the increment
selects. The depth is one counter's 64-bit add, so the timing is set by the counter width
alone and moves neither with `WB` nor with the number of events.
