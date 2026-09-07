# qcore_requant synthesis (xc7)

Every number below is read back out of `build/synth/synth_requant.log` by
`scripts/synth_report.py`, which `make synth` runs. The run comes first and this page is
written from it, so the two cannot disagree. `--check` regenerates the page and requires it
back byte for byte on the Yosys build named below; another build has its own LUT packing and
path lengths, so it is held to the parameters and the hard-block inventory.

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`

```sh
mkdir -p build/synth
yosys -q -l build/synth/synth_requant.log -s syn/synth_requant.ys
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

Parameters: `WB = 64`, `B_MAX = 1`, `ACC_W = 40`, `VSRAM_WORDS = 4096`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `CARRY4` | 99 |
| `DSP48E1` | 4 |
| `FDRE` | 3105 |
| `FDSE` | 8 |
| `IBUF` | 3040 |
| `INV` | 57 |
| `LUT2` | 693 |
| `LUT3` | 1064 |
| `LUT4` | 119 |
| `LUT5` | 236 |
| `LUT6` | 1836 |
| `MUXF7` | 724 |
| `MUXF8` | 122 |
| `OBUF` | 1011 |
| `RAM32M` | 102 |

12221 cells in total: 3948 LUTs, 3113 flops, 99 `CARRY4`, 4051 I/O pads. Yosys estimates
3255 LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `DSP48E1` | 2 | `$mul` at `rtl/qcore_requant.sv:519` |
| `DSP48E1` | 2 | `$mul` at `rtl/qcore_requant.sv:572` |
| `RAM32M` | 102 | `dq_mem` |

Memories, as `memory_libmap` mapped them:

- `qcore_requant.dq_mem` via `$__XILINX_LUTRAM_SDP_`

Longest topological path through the LUT fabric: 27 cells.

```
 0  d_s1 [1]
24  qcore_pkg::sat40_from57$func$rtl/qcore_requant.sv:541$467.$result [39]
27  qcore_pkg::sat40_from57$func$rtl/qcore_requant.sv:541$467.$result [0]
```

## Tiny configuration

Parameters: `WB = 16`, `B_MAX = 2`, `ACC_W = 40`, `VSRAM_WORDS = 2048`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `CARRY4` | 112 |
| `DSP48E1` | 4 |
| `FDRE` | 3309 |
| `FDSE` | 9 |
| `IBUF` | 1783 |
| `INV` | 55 |
| `LUT2` | 1128 |
| `LUT3` | 942 |
| `LUT4` | 239 |
| `LUT5` | 262 |
| `LUT6` | 1267 |
| `MUXF7` | 241 |
| `MUXF8` | 47 |
| `OBUF` | 578 |
| `RAM32M` | 30 |

10007 cells in total: 3838 LUTs, 3318 flops, 112 `CARRY4`, 2361 I/O pads. Yosys estimates
2710 LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `DSP48E1` | 2 | `$mul` at `rtl/qcore_requant.sv:519` |
| `DSP48E1` | 2 | `$mul` at `rtl/qcore_requant.sv:572` |
| `RAM32M` | 30 | `dq_mem` |

Memories, as `memory_libmap` mapped them:

- `qcore_requant.dq_mem` via `$__XILINX_LUTRAM_SDP_`

Longest topological path through the LUT fabric: 27 cells.

```
 0  d_s1 [1]
24  qcore_pkg::sat40_from57$func$rtl/qcore_requant.sv:541$36878.$result [39]
27  $auto$xilinx_dffopt.cc:347:execute$70524
```

## Notes (hand-written)

The two wide signed products are DSP blocks and the dump beat queue is LUTRAM; the rest is
fabric, and most of that is selection rather than arithmetic -- there are far more
`MUXF7` / `MUXF8` pairs than `CARRY4` slices. The longest path is the first rounding stage:
the descriptor's shift amount picks the round bit across the full 57-bit product and the
result saturates into 40 bits, which is where the wide muxes go.
