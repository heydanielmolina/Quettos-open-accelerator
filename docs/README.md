# Quettos Core documentation

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Thesis, v1 configuration, block diagram, decode-step dataflow, prefill/decode loop, host/RTL boundary, invariant, scope and measurement conventions |
| [ISA.md](ISA.md) | Programming model, descriptor bit layout, opcode table, binary formats, CSR table (generated from `isa.py`) |
| [NUMERICS.md](NUMERICS.md) | Integer numerics spec: sfloat, requant, VQUANT, RMSNorm, RoPE, softmax, SiLU, LUTs; per-class formats (measured maxima TBD until calibration) |
| [MEMORY_MAP.md](MEMORY_MAP.md) | QMEM image map, weight tiling, VSRAM element map, KV cache layout |
| [VERIFICATION.md](VERIFICATION.md) | Oracle chain and the eight test layers |
| [PERFORMANCE.md](PERFORMANCE.md) | Measured Verilator speed probe and, later, `make perf` tables |
| [ROADMAP.md](ROADMAP.md) | What ships in v1, the demo configuration, and v1.1 .. v2 |
| [../rtl/cfg/README.md](../rtl/cfg/README.md) | The three parameter sets and how they are passed to Verilator / Yosys / Icarus |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | SystemVerilog subset, lint recipe, hooks, contribution workflow, measured-numbers rule |

Conventions: anything analytical is marked **estimate**; anything measured
names the command that produced it. `diagrams/` will hold the architecture SVG
and `demo.cast` the asciinema recording.
