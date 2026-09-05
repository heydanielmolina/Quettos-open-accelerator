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
- a layered oracle, HF fp32 -> integer golden -> ISA simulator -> RTL, in which
  the RTL matches the integer golden model bit for bit;
- a C++ Verilator harness with a fixed-latency memory model that prints tokens
  as they leave the RTL and reports the hardware performance counters;
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

## Status

**2026-09-04:** the integer numerics, lookup tables, calibration, int8
quantizer and the bit-exact integer golden model are in place; both models
generate text from integer arithmetic (`models/*/expected_tokens.json`), and
`models/*/quality.json` holds the measured quality against fp32. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) for what comes next and
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for the speed-probe result and
the locked demo configuration.

## Quick start

`make demo` lands with the end-to-end integration. Available now:

```sh
make lint    # three-parser RTL lint (Verilator, Yosys, Icarus)
make test    # uv run pytest -q sw/tests
make probe   # Verilator speed probe (sim/probe)

uv run quettos download qwen        # or smollm2; fetches the model from Hugging Face
uv run quettos calibrate qwen       # activation ranges -> models/<name>/calib.json
uv run quettos quantize qwen        # int8 weights + constants -> build/quant/<name>.npz
uv run quettos golden qwen          # greedy generation on the integer golden model
uv run quettos check qwen           # quality vs fp32 -> models/<name>/quality.json
```

Requirements: Verilator 5.x, Yosys 0.65, Icarus Verilog 13, `uv` (Python 3.13
is pinned via `.python-version`). No vendor tools, no tokens, no secrets.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) -- thesis, block diagram, dataflow, host/RTL boundary
- [`docs/ISA.md`](docs/ISA.md) -- descriptor layout, opcodes, binary formats
- [`docs/NUMERICS.md`](docs/NUMERICS.md) -- integer numerics specification
- [`docs/MEMORY_MAP.md`](docs/MEMORY_MAP.md) -- image, VSRAM and KV layouts
- [`docs/VERIFICATION.md`](docs/VERIFICATION.md) -- oracle chain and test layers
- [`docs/ROADMAP.md`](docs/ROADMAP.md) -- what ships in v1, then v1.1 .. v2
- [`rtl/cfg/README.md`](rtl/cfg/README.md) -- parameter sets
- [`CONTRIBUTING.md`](CONTRIBUTING.md) -- SystemVerilog subset and process

## License

Apache-2.0 (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). Model weights are
downloaded at build time from the Hugging Face Hub and are not redistributed.
