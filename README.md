# Quettos Core

A decode-first LLM inference accelerator in synthesizable SystemVerilog, simulated
cycle-accurately with Verilator and synthesized with Yosys. Free tools only.

Quettos Core is an inference-only, weight-streaming LLM accelerator written in
synthesizable SystemVerilog and driven by a small descriptor ISA. The goal of
v1 is to show a real language model (Qwen2.5-0.5B-Instruct, with
SmolLM2-135M-Instruct as the fast regression model) generating text on the RTL
in cycle-accurate Verilator simulation, bit-exact against an integer golden
model, and to report Yosys synthesis numbers for an FPGA-class target. Every
weight byte streams through a `WB`-byte-per-cycle port once per token and is
consumed by `WB` output-stationary lanes the cycle it arrives; activations and
per-token scales stay in a 128 KB on-chip vector SRAM.
The design thesis (decode is the unit of work for agent inference) and the
roadmap are in [`docs/`](docs/README.md).

## What v1 delivers

- `qcore_top`, a buildable core in the SystemVerilog subset that Verilator 5,
  Yosys 0.65 and Icarus 13 all accept with zero rewrites;
- a compiler (`sw/quettos/`) that quantizes a Hugging Face checkpoint to int8
  weights / int16 activations, lays out the memory image, and emits
  straight-line descriptor programs;
- a layered oracle, HF fp32 -> integer golden -> ISA simulator -> RTL: the
  software layers agree bit for bit on both complete models, and `qcore_top`
  matches the ISA simulator element by element over the `EMBED` and matrix-vector
  path `make bringup-sweep` runs, at WB 16, 64 and 128;
- a C++ Verilator harness with a fixed-latency memory model that runs a whole
  compiled program on the RTL for its traffic and its hardware performance
  counters, and prints tokens as they leave the RTL once the six vector opcodes
  run on `qcore_vpu_top`;
- Yosys `synth_xilinx` cell counts for the demo configuration, with the exact
  command and tool version.

## Scope

v1 is a single-sequence decode engine, verified in cycle-accurate simulation.
Performance is reported as cycles per token, bytes per token and utilization
from the RTL's own counters; FPGA throughput is derived from those cycle counts
at a stated clock and memory bandwidth. Model quality is a measured perplexity,
KL and top-1 delta against fp32. The row dimension of the datapath, prefix reuse
through KV save/restore, and the tool-call demo are the foundations for batched
decode, a paged KV cache and constrained decoding on the
[roadmap](docs/ROADMAP.md).

## Results

Every row names the command that produced it. Simulation numbers are from an
Apple M5 Pro with Verilator 5.048 and Apple clang 17; synthesis is Yosys 0.65.
The workload is the two-layer Qwen2.5-0.5B-Instruct image
(`build/images/qwen2.5-0.5b-instruct-l2`) at the demo configuration
`WB=64, B_MAX=1, VSRAM_WORDS=4096`, run under `--traffic`: the real addresses,
strides, meta and partial tiles of the compiled program.

| Measurement | Value | Command |
|---|---|---|
| Decode token | 2,626,475 cycles, MAC array active in 98.8% of them, 63.8 read bytes per cycle | `make perf PERF_ARGS="--traffic --max-new 1 --prompt-ids 1"` |
| 36-token prompt + 1 decode token | 19,453,320 cycles, 97.6% MAC-active, 63.0 read bytes per cycle, 0.09% of cycles in sequencing | `make perf` |
| Verilator simulation rate, `--threads 1` | 3.70 Mcycles/s, median of 33 runs (3.59 - 3.76) | the decode-token run above |
| `qcore_top` on xc7 | 69 DSP48E1, 32 RAMB36E1, 15 RAMB18E1, 267 RAM32M, 13,747 LUTs, 13,945 flops; 30,835 cells, 11,484 estimated LCs | `make synth` (Yosys `synth_xilinx -family xc7 -flatten`); the table is the Demo configuration section of `syn/reports/qcore_top.md`, which that command regenerates from its own log |
| RTL against the ISA simulator | 20 of 20 runs match element by element at WB=64 and WB=128 | `make bringup-sweep` |
| Quality against fp32, W8A16 | Qwen KL 0.0124 nats and top-1 95.57%; SmolLM2 KL 0.0048 and top-1 95.66% | `uv run quettos check <alias>` |

The synthesis row is `qcore_top` as it stands: the GEMV, EMBED and KVWRITE
datapath and the whole control path, without `qcore_vpu_top`.
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) holds the full tables, the memory
model and the derivation of the demo configuration;
[`docs/NUMERICS.md`](docs/NUMERICS.md) holds the quality table with its
standard errors.

## Status

`qcore_top` runs a descriptor program end to end and matches the ISA simulator
element by element.

The software stack is complete: integer numerics, lookup tables, calibration
and the int8 quantizer in `sw/quettos/`, a bit-exact integer golden model that
generates text for both models (`models/*/expected_tokens.json`), and the
measured quality against fp32 in `models/*/quality.json`. The descriptor ISA
(`sw/quettos/isa.py`, generated into `rtl/qcore_csr_defs.svh` and
`sim/verilator/csr_defs.hpp`), the compiler (`image.bin`, `decode.prog`,
`prefill.prog`, `layout.json`) and the ISA simulator are in place, and the
simulator reproduces the golden model bit for bit on both complete models
(`uv run quettos isa-sim <alias> --compare`).

`rtl/` holds thirteen files, the package and twelve modules: the vector SRAM, the MAC lane
group, the row, the requant, the memory arbiter, the stream controller, the CSR
file, the descriptor fetch unit, the dispatcher, the KV writer, the performance
counters and `qcore_top` itself. `make lint` takes those and the GEMV-path
assembly under `sim/cocotb/wrappers/` through Verilator `-Wall -Wpedantic`,
Yosys `check -assert` and Icarus `-g2012`, each as its own top, with zero
warnings and no waivers. The twelve cocotb benches in `sim/cocotb/` cover every
block below `qcore_top`, each against the reference that fits it: the
requant and the assembled GEMV path bit for bit against
`sw/quettos/numerics.py`, the KV writer byte for byte against
`sw/quettos/isa_sim.py`, the MAC lane group and the row against the exact
integer matmul, and the CSR file, the performance counters, the descriptor
fetch unit, the dispatcher, the memory arbiter, the stream controller and the
vector SRAM against the handshakes and cycle-level guarantees of `docs/RTL.md`,
with the dispatcher's decoded fields taken from `isa_sim.py` as well.
`make gatesim` compares each block's Yosys netlist against the source it came
from, cycle by cycle. `make bringup` and `make bringup-sweep` run a program on
`qcore_top` through the C++ harness and on the ISA simulator and compare every
VSRAM element, SREG word, dumped logit, CSR and PERF counter after every
descriptor; `make perf` runs a whole compiled program on the RTL as a traffic
and cycle measurement; and `make synth` runs Yosys `synth_xilinx` over every
block and over the whole core, regenerating `syn/reports/*.md` from the run and
failing if a report no longer reproduces.

The six vector opcodes run on `qcore_vpu_top`, the module that completes v1 and
turns the traffic measurement into generated text. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) for the order and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for where it sits.

## Quick start

```sh
make lint           # three-parser RTL lint (Verilator, Yosys, Icarus), 14 tops
make test           # uv run pytest -q sw/tests
make cocotb         # cocotb + Verilator block benches (sim/cocotb)
make synth          # Yosys synth_xilinx per block and whole core; checks syn/reports/
make gatesim        # each Yosys netlist against the source it came from (sim/gatesim)
make harness        # build the C++ Verilator harness (sim/verilator)
make harness-csr    # the harness CSR driver against rtl/qcore_csr.sv
make bringup        # qcore_top vs the ISA simulator, tiny configuration
make bringup-sweep  # the same over random shapes at WB=64 and WB=128
make perf           # run a compiled program on qcore_top -> build/perf/perf.json
make probe          # Verilator speed probe (sim/probe)
```

The model pipeline, from a Hugging Face checkpoint to an image the RTL runs:

```sh
uv run quettos download qwen        # or smollm2; fetches the model from Hugging Face
uv run quettos calibrate qwen       # activation ranges -> models/<name>/calib.json
uv run quettos quantize qwen        # int8 weights + constants -> build/quant/<name>.npz
uv run quettos golden qwen          # greedy generation on the integer golden model
uv run quettos check qwen           # quality vs fp32 -> models/<name>/quality.json
uv run quettos compile qwen --layers 2  # image.bin, decode/prefill.prog, layout.json -> build/images/<name>-l2/
uv run quettos isa-sim qwen --compare   # run the programs on the ISA simulator, every descriptor vs golden
uv run quettos compare --image build/images/<name> --wb 64   # qcore_top vs the ISA simulator
uv run quettos csr-defs --check     # the generated ISA/CSR headers match sw/quettos/isa.py
```

`make perf` and `make bringup` build the harness themselves; `make perf` reads
the image named by `IMAGE` (default `build/images/qwen2.5-0.5b-instruct-l2`,
the two-layer compile above).

Requirements: Verilator 5.x, Yosys 0.65, Icarus Verilog 13, `uv` (Python 3.13
is pinned via `.python-version`). No vendor tools, no tokens, no secrets.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) -- thesis, block diagram, dataflow, host/RTL boundary
- [`docs/ISA.md`](docs/ISA.md) -- descriptor layout, opcodes, binary formats, CSR table
- [`docs/RTL.md`](docs/RTL.md) -- the module interface specification: every port, handshake and cycle-level guarantee
- [`docs/NUMERICS.md`](docs/NUMERICS.md) -- integer numerics specification and the measured quality table
- [`docs/MEMORY_MAP.md`](docs/MEMORY_MAP.md) -- image, VSRAM and KV layouts
- [`docs/VERIFICATION.md`](docs/VERIFICATION.md) -- oracle chain and test layers
- [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) -- the speed probe, the harness and the measured tables
- [`docs/ROADMAP.md`](docs/ROADMAP.md) -- what ships in v1, then v1.1 .. v2
- [`rtl/cfg/README.md`](rtl/cfg/README.md) -- parameter sets
- [`sim/cocotb/README.md`](sim/cocotb/README.md) -- the block benches and how to add one
- [`sim/gatesim/README.md`](sim/gatesim/README.md) -- gate-level equivalence: what it covers and what it cannot
- [`CONTRIBUTING.md`](CONTRIBUTING.md) -- SystemVerilog subset and process

## License

Apache-2.0 (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). Model weights are
downloaded at build time from the Hugging Face Hub and are not redistributed.
