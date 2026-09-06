# cocotb unit tests

Block-level tests of the `rtl/qcore_*.sv` modules, run with cocotb 2.1 and
Verilator through cocotb's runner API. The default configuration is the tiny one
(`WB=16, B_MAX=2, VL=2, VSRAM_WORDS=2048`); the fetch unit, the dispatcher, the
KV writer and the GEMV wrapper also run or elaborate at `WB=64` and `WB=128`,
where one beat holds more than one descriptor and one K^T tile more than one row
of bytes. `make cocotb` (or `uv run pytest -q sim/cocotb`) runs every test; a
single one is `uv run pytest -q sim/cocotb/test_vsram.py`. Build products go to
`build/cocotb/<top>/<key>/`, where `key` is a digest of the sources and the
parameters, so two configurations of one top never share object files.

| File | Role |
|---|---|
| `qc_runner.py` | `build(top, sources, parameters=...)` verilates `rtl/qcore_pkg.sv` plus the listed `rtl/*.sv` files (or absolute paths) into a keyed build directory; `run(top, sources, test_module, ...)` builds, runs the cocotb module and asserts that tests ran and none failed. Its keyword arguments are `parameters`, `testcase`, `seed`, `waves` and `extra_env`. `TINY` holds the tiny configuration; `VERILATOR_ARGS` the flags added to cocotb's own (`--timing -Wall -Wpedantic --x-assign fast --x-initial unique --assert`, timescale `1ns/1ps`). ROM image parameters (`ROM_FILE`, `ROM_FILE_<TABLE>`) get the `rtl/gen/*.hex` paths automatically |
| `conftest.py` | puts this directory on `sys.path`, so a `test_*.py` imports `qc_runner` and a bench imports `qc_numerics` by name |
| `qc_qmem.py` | `QmemModel`: the QMEM bus model (fixed latency, one beat per cycle, in-order, in-flight window, strobed writes, acks). A burst takes its bytes when the request is accepted, so a write accepted while it is in flight cannot change what it returns -- the rule `sim/verilator/mem_model.hpp` follows, which is what makes the two buses the same bus. Counters mirror `RD_BEATS` / `RD_BYTES` / `WR_BEATS` / `WR_BYTES` |
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
   re-implementation of the arithmetic) or, for a block that sequences rather
   than computes, against the handshake and the cycle-level guarantee
   `docs/RTL.md` gives it.
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
4. A bench that runs at more than one width reads those widths from the
   environment (`extra_env={"QC_WB": "64", ...}` in the `test_*.py`, `WB =
   int(os.environ["QC_WB"])` in the bench), because the parameters live in the
   build and the cocotb module is imported per run.
5. Keep each `tb_` module under a million cycles; `make cocotb` is the quick
   loop.

Waveforms: `qc_runner.run(..., waves=True)` writes `dump.vcd` into the build
directory for gtkwave.

Signal values: `qc_stream.value(sig)` reads any width (cocotb 2.1 hands a
`Logic` for 1-bit handles and a `LogicArray` otherwise); memories and
generate-block instances are reachable as `dut.g_row[r].u_vsram.mem[w]`
because the runner builds with `--public-flat-rw`.
