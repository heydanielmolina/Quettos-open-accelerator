# cocotb unit tests

Block-level tests of the `rtl/qcore_*.sv` modules on the tiny configuration
(`WB=16, B_MAX=2, VL=2, VSRAM_WORDS=2048`), run with cocotb 2.1 and
Verilator through cocotb's runner API. `make cocotb` (or
`uv run pytest -q sim/cocotb`) runs every test; a single one is
`uv run pytest -q sim/cocotb/test_vsram.py`. Build products go to
`build/cocotb/<top>/`.

| File | Role |
|---|---|
| `qc_runner.py` | `run(top, sources, test_module, parameters=...)`: verilates `rtl/qcore_pkg.sv` plus the listed `rtl/*.sv` files (or absolute paths), runs the cocotb module, asserts zero failures. `TINY` holds the configuration; `VERILATOR_ARGS` the flags added to cocotb's own (`--timing -Wall -Wpedantic --x-assign fast --x-initial unique --assert`, timescale `1ns/1ps`). ROM image parameters (`ROM_FILE`, `ROM_FILE_<TABLE>`) get the `rtl/gen/*.hex` paths automatically |
| `qc_qmem.py` | `QmemModel`: the QMEM bus model (fixed latency, one beat per cycle, in-order, in-flight window, strobed writes, acks); counters mirror `RD_BEATS` / `RD_BYTES` / `WR_BEATS` / `WR_BYTES` |
| `qc_stream.py` | valid/ready driver and monitor, a ready-pattern source, pulses, reset, value helpers; everything on falling edges |
| `qc_numerics.py` | the `sw/quettos/numerics.py` primitives and the packing of SREG words, meta records, descriptors and int8 beats as the RTL sees them |
| `tb_<module>.py` | the `@cocotb.test()` coroutines of one module |
| `test_<module>.py` | the pytest entry that builds the module and runs its `tb_` file |
| `wrappers/qcore_gemv_wrap.sv` | the GEMV path assembled as `qcore_top` wires it (`docs/RTL.md` 3.2): arbiter, stream controller, rows with their VSRAMs, requant, crossbar, SREG write mux, event adders |
| `test_gemv_wrap.py`, `tb_gemv_wrap.py` | elaborates the wrapper with the three parsers in the tiny, `fpga_w64` and `sim_w128` configurations, then runs GEMV / EMBED descriptors through it against the QMEM model and checks every output against `numerics.requant` over the exact integer matmul |

## Adding a test

1. Write `tb_<module>.py` with `@cocotb.test()` coroutines. Start the clock
   with `Clock(dut.clk, 10, unit="ns")`, drive inputs and sample outputs at
   `FallingEdge(dut.clk)`, and compare against `qc_numerics` (never against a
   re-implementation of the arithmetic).
2. Write `test_<module>.py`:

   ```python
   import qc_runner


   def test_requant() -> None:
       qc_runner.run(
           "qcore_requant",
           ["qcore_requant.sv"],
           "tb_requant",
           parameters={"WB": qc_runner.TINY["WB"], "ACC_W": qc_runner.TINY["ACC_W"]},
       )
   ```

   List every `rtl/*.sv` file the top instantiates; `qcore_pkg.sv` is added
   for you and listed first (Yosys and Icarus resolve `qcore_pkg::` only after
   the package). Pass only the parameters the top declares.
3. For a module with QMEM ports, instantiate `QmemModel(dut, wb=16, latency=32)`
   in the testbench, preload bytes with `write_bytes`, and
   `cocotb.start_soon(model.run())` before the first descriptor.
4. Keep each `tb_` module under a million cycles; `make cocotb` is the quick loop.

Waveforms: `qc_runner.run(..., waves=True)` writes `dump.vcd` into the build
directory for gtkwave.

Signal values: `qc_stream.value(sig)` reads any width (cocotb 2.1 hands a
`Logic` for 1-bit handles and a `LogicArray` otherwise); memories and
generate-block instances are reachable as `dut.g_row[r].u_vsram.mem[w]`
because the runner builds with `--public-flat-rw`.
