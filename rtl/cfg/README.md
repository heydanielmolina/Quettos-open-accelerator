# RTL configuration sets

`qcore_top` is the parameter root. Every module below it receives its
parameters from `qcore_top`; nothing in the tree hardcodes a width. The same
RTL is built in three configurations. A change that makes one configuration
produce different tokens from another on the same image is a bug.

| Set        | WB  | B_MAX | VL | VSRAM_WORDS | FIFO_BEATS | Used for |
|------------|-----|-------|----|-------------|------------|----------|
| `fpga_w64` | 64  | 1     | 4  | 4096        | 128        | Demo default and the synthesized configuration (Yosys `synth_xilinx` xc7). |
| `sim_w128` | 128 | 1     | 4  | 4096        | 128        | Simulation fallback when the demo needs fewer Verilator cycles; identical tokens to `fpga_w64` by construction. |
| `tiny_w16` | 16  | 2     | 2  | 2048        | 128        | cocotb unit tests and random-tiny-shape RTL-vs-`isa_sim` tests; B_MAX=2 proves the batch dimension at block level. |

Parameter meaning:

- `WB` -- weight bytes per beat and number of output-stationary MAC lanes.
  Also the tile width of the `[N/WB][K][WB]` weight layout, so the image is
  built per `WB` by the compiler.
- `B_MAX` -- number of activation rows that ride one weight pass. v1 activates
  one row; the generate-for over rows is retained and tested at B=2.
- `VL` -- VPU vector lanes (elements processed per cycle by the V ops).
- `VSRAM_WORDS` -- depth of the 256-bit vector SRAM (4096 words = 128 KB =
  32,768 int32 elements; 2048 is enough for the truncated-model tests).
- `FIFO_BEATS` -- depth of the weight FIFO between `qcore_stream_ctrl` and the
  lanes.
- `ROM_FILE` (string, no default) -- absolute path to a `rtl/gen/*.hex` LUT
  image; set by the Makefile/.ys, never as a literal in RTL.

## Passing a configuration

Verilator (`-G` sets a top-level parameter; quote strings twice):

```sh
verilator --cc --exe --build -j 0 -O3 --top-module qcore_top \
  -GWB=64 -GB_MAX=1 -GVL=4 -GVSRAM_WORDS=4096 -GFIFO_BEATS=128 \
  -GROM_FILE='"'"$PWD"'/rtl/gen/exp2.hex"' \
  rtl/*.sv sim/verilator/main.cpp
```

Yosys (`-chparam` on `hierarchy`, before `proc`):

```sh
yosys -p "read_verilog -sv rtl/*.sv; \
  hierarchy -check -top qcore_top \
    -chparam WB 64 -chparam B_MAX 1 -chparam VL 4 \
    -chparam VSRAM_WORDS 4096 -chparam FIFO_BEATS 128 \
    -chparam ROM_FILE \"$PWD/rtl/gen/exp2.hex\"; \
  synth_xilinx -family xc7 -flatten -top qcore_top; stat -tech xilinx; ltp -noff"
```

Icarus (parse/lint only; `-P` sets a top parameter):

```sh
iverilog -g2012 -s qcore_top -Pqcore_top.WB=16 -Pqcore_top.B_MAX=2 -Pqcore_top.VL=2 \
  -Pqcore_top.VSRAM_WORDS=2048 -o /dev/null rtl/*.sv
```

The Makefile exposes these as `CFG=fpga_w64|sim_w128|tiny_w16` once the RTL
exists. Until then this file is the reference.
