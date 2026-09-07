# Quettos Core documentation

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Thesis, v1 configuration, block diagram, decode-step dataflow, prefill/decode loop, host/RTL boundary, invariant, scope and measurement conventions |
| [ISA.md](ISA.md) | Programming model, descriptor bit layout, opcode table, binary formats, CSR table (generated from `isa.py`) |
| [RTL.md](RTL.md) | Module interface specification: every port, every handshake, the cycle-level guarantee each module makes, the resolved design decisions and the lint patterns |
| [NUMERICS.md](NUMERICS.md) | Integer numerics spec: sfloat, requant and the per-GEMV program constants, VQUANT, RMSNorm, RoPE, softmax, SiLU, LUTs; per-class formats with the calibration maxima and gate results for both models; the measured quality table (`uv run quettos check`) |
| [MEMORY_MAP.md](MEMORY_MAP.md) | QMEM image map, weight tiling, VSRAM element map, KV cache layout |
| [VERIFICATION.md](VERIFICATION.md) | Oracle chain (fp32 -> golden -> ISA simulator -> RTL), the eight test layers, what `models/<name>/` holds |
| [PERFORMANCE.md](PERFORMANCE.md) | The measurement protocol; the measured cycle, stall-bucket, wall-clock and simulation-rate tables of `qcore_top` over whole compiled programs, with the memory-latency and `WB=128` sweeps; the demo configuration and the rule it was chosen by; the C++ harness, the bring-up runs and the speed probe |
| [ROADMAP.md](ROADMAP.md) | What ships in v1, the demo configuration, and v1.1 .. v2 |
| [../rtl/cfg/README.md](../rtl/cfg/README.md) | The three parameter sets and how the harness, the synthesis scripts, the cocotb runner and the lint script each pass them |
| [../sim/cocotb/README.md](../sim/cocotb/README.md) | The block benches, the runner and the bus model, and how to add a bench |
| [../sim/probe/README.md](../sim/probe/README.md) | The Verilator speed probe: what the skeleton contains, what it deliberately leaves out, and how to build and run it |
| [../sim/gatesim/README.md](../sim/gatesim/README.md) | Gate-level equivalence: each block's Yosys netlist against the source it came from, what is covered and why the rest is not |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | SystemVerilog subset, lint recipe, hooks, contribution workflow, measured-numbers rule |

Conventions: anything analytical is marked **estimate**; anything measured
names the command that produced it.
