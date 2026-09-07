# cocotb unit tests

Block-level tests of the `rtl/qcore_*.sv` modules, run with cocotb 2.1 and
Verilator through cocotb's runner API. The default configuration is the tiny one
(`WB=16, B_MAX=2, VL=2, VSRAM_WORDS=2048`); the fetch unit, the dispatcher, the
KV writer and the GEMV wrapper also run or elaborate at `WB=64` and `WB=128`,
where one beat holds more than one descriptor and one K^T tile more than one row
of bytes, and the vector unit runs at the demo widths as well. The lookup-table
ROM, its interpolator, the vector lane and the vector scalar unit take no width
parameter, so one build of each covers every configuration.

`make cocotb` (or `uv run pytest -q sim/cocotb`) runs every test; a single one is
`uv run pytest -q sim/cocotb/test_vsram.py`. Build products go to
`build/cocotb/<top>/<key>/`, where `key` is a digest of the sources and the
parameters, so two configurations of one top never share object files. From an
emptied `build/cocotb` the whole set takes 99.62 s, the median of five runs
(98.32 - 100.74, n=5, on the machine `docs/PERFORMANCE.md` names), Verilator
builds included; a rerun keeps the builds it can.

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
| `test_lut.py`, `tb_lut.py` | `qcore_lut_rom` built once per table (`QC_LUT_TABLE` names it, `ENTRIES` and `ROM_FILE` come from `numerics.TABLE_SPECS` and `rtl/gen/`) and read entry by entry through each port and both at once; `qcore_lut_interp` over every `(entry, frac8)` pair of all four tables against `numerics.Lut.interp`. Three further tests take the `ROM_FILE` flow through the other two parsers and compare what Yosys and Icarus loaded against the checked-in image |
| `test_vpu_lane.py`, `tb_vpu_lane.py` | `qcore_vpu_lane` on one build (it takes no width parameter): the six ops against `numerics.py` over `QCORE_VPU_LANE_VECTORS` random elements each, plus the rounding, saturation and coefficient-signedness boundaries and the two-cycle latency |
| `test_vpu_scalar.py`, `tb_vpu_scalar.py` | `qcore_vpu_scalar` with its rsqrt and recip ROMs: both table domains swept exhaustively, then `QCORE_VPU_SCALAR_REQUESTS` random requests per op against `numerics.py`, the leading-one search at every bit length, the `sfloat_mul` rounding overflow, the shift clamp and the fixed five-cycle latency |
| `test_vpu_top.py`, `tb_vpu_top.py` | `qcore_vpu_top` with its lanes, tables and scalar unit, on the tiny widths and on the demo widths (`WB=64, B_MAX=1, VL=4, VSRAM_WORDS=4096`), so both the two-lane and the four-lane operand windows run: `QCORE_VPU_TOP_CASES` descriptors per sweep against `isa_sim` over a model of the VSRAM banks, the SREG banks and the QMEM read port, every element, scale register and event count compared |

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
   loop. The exception is a block whose state space is small enough to sweep
   exhaustively -- the vector lane and the vector scalar unit run a few million
   -- and those put the count behind an environment variable
   (`QCORE_VPU_LANE_VECTORS`, `QCORE_VPU_SCALAR_REQUESTS`,
   `QCORE_VPU_TOP_CASES`) so a run can be shortened without editing the bench.

Waveforms: `qc_runner.run(..., waves=True)` writes `dump.vcd` into the build
directory for gtkwave.

Signal values: `qc_stream.value(sig)` reads any width (cocotb 2.1 hands a
`Logic` for 1-bit handles and a `LogicArray` otherwise); memories and
generate-block instances are reachable as `dut.g_row[r].u_vsram.mem[w]`
because the runner builds with `--public-flat-rw`.
