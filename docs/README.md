# Quettos Core documentation

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Thesis, v1 configuration, block diagram, decode-step dataflow, prefill/decode loop, host/RTL boundary, invariant, scope and measurement conventions |
| [ISA.md](ISA.md) | Programming model, descriptor bit layout, opcode table, binary formats, CSR table (generated from `isa.py`) |
| [NUMERICS.md](NUMERICS.md) | Integer numerics spec: sfloat, requant and the per-GEMV program constants, VQUANT, RMSNorm, RoPE, softmax, SiLU, LUTs; per-class formats with the calibration maxima and gate results for both models; the measured quality table (`uv run quettos check`) |
| [MEMORY_MAP.md](MEMORY_MAP.md) | QMEM image map, weight tiling, VSRAM element map, KV cache layout |
| [VERIFICATION.md](VERIFICATION.md) | Oracle chain (fp32 -> golden -> ISA simulator -> RTL), the eight test layers, what `models/<name>/` holds |
| [PERFORMANCE.md](PERFORMANCE.md) | Measured Verilator speed probe; `make perf` tables |
| [ROADMAP.md](ROADMAP.md) | What ships in v1, the demo configuration, and v1.1 .. v2 |
| [../rtl/cfg/README.md](../rtl/cfg/README.md) | The three parameter sets and how they are passed to Verilator / Yosys / Icarus |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | SystemVerilog subset, lint recipe, hooks, contribution workflow, measured-numbers rule |

Conventions: anything analytical is marked **estimate**; anything measured
names the command that produced it.
