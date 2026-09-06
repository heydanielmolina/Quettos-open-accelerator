# qcore_vsram synthesis (xc7)

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`.

Method: Yosys `synth_xilinx -family xc7` of the block on its own; the cell counts are the
post-synthesis `stat -tech xilinx` table of the top module (I/O buffers included: the block's
ports become pads when it is synthesized alone). `make synth` reruns every `syn/*.ys` script;
the exact command of each run is quoted with its table.

## Default configuration (WORDS = 4096, W = 256, NE = 8)

```sh
yosys -q -l build/synth/synth_vsram.log -s syn/synth_vsram.ys
```

| Cell | Count |
|---|---|
| `RAMB36E1` | 32 |
| `FDRE` | 257 |
| `LUT3` | 256 |
| `BUFG` | 1 |
| `IBUF` | 291 |
| `OBUF` | 512 |

1349 cells in total, 256 LUTs, estimated 256 LCs.

## Tiny configuration (WORDS = 2048)

```sh
yosys -q -l build/synth/qcore_vsram_xc7_tiny.log -p "read_verilog -sv -defer -Irtl rtl/qcore_vsram.sv; chparam -set WORDS 2048 qcore_vsram; hierarchy -check -top qcore_vsram; synth_xilinx -family xc7 -top qcore_vsram; stat -tech xilinx"
```

| Cell | Count |
|---|---|
| `RAMB36E1` | 16 |
| `FDRE` | 257 |
| `LUT3` | 256 |
| `BUFG` | 1 |
| `IBUF` | 289 |
| `OBUF` | 512 |

1331 cells in total, 256 LUTs, estimated 256 LCs.

## Inference check

`memory_libmap` reports `mapping memory qcore_vsram.mem via $__XILINX_BLOCKRAM_TDP_`: the 256-bit
true-dual-port RAM with element strobes maps to block RAM (32 RAMB36E1 at 4096 words, 16 at 2048), with
one FDRE per output bit plus one for the read-enable hold and 256 LUT3 for the read-data hold. Yosys
prints `Resizing cell port` warnings while it parametrizes the RAMB36E1 ports; the mapping is correct.
No DSP48E1 and no LUTRAM.
