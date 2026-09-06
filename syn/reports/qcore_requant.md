# qcore_requant synthesis (xc7)

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`.

Method: Yosys `synth_xilinx -family xc7` of the block on its own; the cell counts are the
post-synthesis `stat -tech xilinx` table of the top module (I/O buffers included: the block's
ports become pads when it is synthesized alone). `make synth` reruns every `syn/*.ys` script;
the exact command of each run is quoted with its table.

## Default configuration (WB = 64, B_MAX = 1, ACC_W = 40, VSRAM_WORDS = 4096)

```sh
yosys -q -l build/synth/synth_requant.log -s syn/synth_requant.ys
```

| Cell | Count |
|---|---|
| `DSP48E1` | 4 |
| `RAM32M` | 102 |
| `FDRE` | 3105 |
| `FDSE` | 8 |
| `CARRY4` | 99 |
| `MUXF7` | 705 |
| `MUXF8` | 129 |
| `LUT2` | 711 |
| `LUT3` | 1060 |
| `LUT4` | 139 |
| `LUT5` | 258 |
| `LUT6` | 1759 |
| `INV` | 57 |
| `BUFG` | 1 |
| `IBUF` | 3040 |
| `OBUF` | 1011 |

12188 cells in total, 3927 LUTs, estimated 3216 LCs.

Longest topological path in cells: 25 over the LUT fabric alone, 62 through the DSP48E1 and RAM32M cells (the two `ltp` selections of the script).

## Tiny configuration (WB = 16, B_MAX = 2, VSRAM_WORDS = 2048)

```sh
yosys -q -l build/synth/qcore_requant_xc7_tiny.log -p "read_verilog -sv -defer -Irtl rtl/qcore_pkg.sv rtl/qcore_requant.sv; chparam -set WB 16 -set B_MAX 2 -set VSRAM_WORDS 2048 qcore_requant; hierarchy -check -top qcore_requant; synth_xilinx -family xc7 -top qcore_requant; stat -tech xilinx; ltp -noff t:FDRE t:FDSE t:BUFG t:IBUF t:OBUF t:DSP48E1 t:RAM32M %u %u %u %u %u %u %n; ltp -noff t:FDRE t:FDSE t:BUFG t:IBUF t:OBUF %u %u %u %u %n"
```

| Cell | Count |
|---|---|
| `DSP48E1` | 4 |
| `RAM32M` | 30 |
| `FDRE` | 3309 |
| `FDSE` | 9 |
| `CARRY4` | 112 |
| `MUXF7` | 235 |
| `MUXF8` | 47 |
| `LUT1` | 2 |
| `LUT2` | 1191 |
| `LUT3` | 925 |
| `LUT4` | 249 |
| `LUT5` | 270 |
| `LUT6` | 1235 |
| `INV` | 55 |
| `BUFG` | 1 |
| `IBUF` | 1783 |
| `OBUF` | 578 |

10035 cells in total, 3872 LUTs, estimated 2688 LCs.

Longest topological path in cells: 26 over the LUT fabric alone, 66 through the DSP48E1 and RAM32M cells.

## Inference check

The two 40 x 17 signed multiplies of the requant (stage 1 `acc * Sw_m`, stage 2 `t * Sx_m`) map to two
DSP48E1 each (4 in total); the 4-entry dump beat queue maps to RAM32M LUTRAM (the tile meta replay buffer,
generated only for `B_MAX > 1`, is a flop vector); the
rest is fabric. No block RAM. The longest paths (`ltp -noff` with the flops, I/O buffers, DSP48E1 and
RAM32M as cut points) run through the stage-2 `round_shift57` and `sat32` (the 64:1 round-bit select on
MUXF7/MUXF8 and the increment carry chain).
