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
- `ROM_FILE_EXP2`, `ROM_FILE_SIGMOID`, `ROM_FILE_RSQRT`, `ROM_FILE_RECIP`
  (untyped parameters, empty default) -- absolute paths to the `rtl/gen/*.hex`
  LUT images; set by the Makefile/.ys, never as literals in RTL.

## Passing a configuration

Verilator (`-G` sets a top-level parameter; quote strings twice):

```sh
verilator --cc --exe --build -j 0 -O3 --top-module qcore_top \
  -GWB=64 -GB_MAX=1 -GVL=4 -GVSRAM_WORDS=4096 -GFIFO_BEATS=128 \
  -GROM_FILE_EXP2='"'"$PWD"'/rtl/gen/exp2.hex"' \
  rtl/*.sv sim/verilator/main.cpp
```

Yosys (`read_verilog -defer` so that `$readmemh` runs after the parameters
are set with `chparam`; `hierarchy -chparam` cannot take a string value):

```sh
yosys -p "read_verilog -sv -defer -Irtl rtl/qcore_pkg.sv rtl/*.sv; \
  chparam -set WB 64 -set B_MAX 1 -set VL 4 \
    -set VSRAM_WORDS 4096 -set FIFO_BEATS 128 \
    -set ROM_FILE_EXP2 \"$PWD/rtl/gen/exp2.hex\" qcore_top; \
  hierarchy -check -top qcore_top; \
  synth_xilinx -family xc7 -flatten -top qcore_top; stat -tech xilinx; \
  ltp -noff t:FDRE t:FDSE t:BUFG t:IBUF t:OBUF t:DSP48E1 t:RAMB36E1 t:RAM32M %u %u %u %u %u %u %u %n"
```

The `ltp` selection uses the flops, I/O buffers, DSPs and RAMs as cut points;
`ltp -noff` alone reports paths through the `FDRE`/`FDSE` primitives.

Icarus (parse/lint only; `-P` sets a top parameter):

```sh
iverilog -g2012 -s qcore_top -Pqcore_top.WB=16 -Pqcore_top.B_MAX=2 -Pqcore_top.VL=2 \
  -Pqcore_top.VSRAM_WORDS=2048 -o /dev/null rtl/*.sv
```

## Who passes what

Each consumer sets the parameters its own way; the three sets above are what
they agree on.

- **The Verilator harness** takes them as make variables:
  `make harness HARNESS_CFG="WB=64 B_MAX=1 VL=4 VSRAM_WORDS=4096"`, which
  `make perf` passes on unchanged; `make bringup` takes its widths from
  `sw/quettos/compare.py`, which calls the same makefile once per configuration
  it compares. `sim/verilator/Makefile` turns each variable into a Verilator
  `-G` parameter and a matching `-D` define for the C++ side, and names the
  object directory after a hash of the RTL, the C++ and the configuration, so
  two configurations never share a build.
- **The synthesis scripts** set them with `chparam -set` after a deferred
  `read_verilog` and before `hierarchy`, one block per configuration in each
  `syn/synth_*.ys`. `scripts/synth_report.py` reads the `Parameter \X = Y` lines
  Yosys prints back, so every table in `syn/reports/` states the configuration
  the tool actually elaborated.
- **The cocotb benches** hold `tiny_w16` as `qc_runner.TINY` and hand a top the
  parameters it declares through `parameters=`; a bench that wants another width
  names it in its own `test_*.py` (`sim/cocotb/README.md`). The
  `*_elaborates` tests run all three parsers over a module in all three
  configurations.
- **`scripts/lint.sh`** elaborates every top at its declared defaults and passes
  only the `ROM_FILE*` image paths, which is the one parameter that must be a
  string: `-G` for Verilator, `-P` for Icarus, `chparam -set` for Yosys.

`sw/quettos/compare.py` keeps the same three sets in one place
(`Config(wb, b_max, vl, vsram_words)`) and drives both the compiler and the
harness from them, which is what makes an RTL-versus-simulator run use one
configuration on both sides.
