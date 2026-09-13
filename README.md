# Quettos Core

Quettos Core is an inference-only, weight-streaming LLM accelerator in
synthesizable SystemVerilog, driven by a descriptor ISA. Qwen2.5-0.5B-Instruct
generates text on it in cycle-accurate Verilator simulation, and Yosys maps it
to xc7 cells. Every weight byte crosses a `WB`-byte-per-cycle port once per
token and is consumed the cycle it arrives by `WB` output-stationary lanes;
activations and per-token scales stay in a 128 KB on-chip vector SRAM.

The design thesis -- decode is the unit of work for agent inference -- and the
roadmap are in [`docs/`](docs/README.md).

## What v1 delivers

- `qcore_top`, a buildable core in the SystemVerilog subset that Verilator 5,
  Yosys 0.65 and Icarus 13 all accept with zero rewrites and no waivers;
- a compiler (`sw/quettos/`) that quantizes a Hugging Face checkpoint to int8
  weights and int16 activations, lays out the memory image, and emits
  straight-line descriptor programs;
- a layered oracle, integer golden model -> ISA simulator -> RTL, whose three
  levels agree bit for bit: the golden model and the simulator over both
  complete models, and `qcore_top` against the simulator element by element over
  the bring-up, vector, attention and layer programs at WB 16, 64 and 128. Under
  it, a cocotb bench for every block below `qcore_top` and a gate-level run of
  the blocks' netlists against the source they came from
  (`sim/gatesim/README.md`); above it, the float32 forward of
  `sw/quettos/reference_np.py` as the quality reference, itself held to the
  `transformers` fp32 forward on the same weights, argmax for argmax at every
  position (`sw/tests/test_reference_np.py`). The layers are in
  [`docs/VERIFICATION.md`](docs/VERIFICATION.md);
- a C++ Verilator harness with a fixed-latency memory model that runs a whole
  compiled program on the RTL -- every descriptor, all six vector opcodes on
  the vector unit -- for its traffic and its hardware performance counters, and
  prints tokens as they leave the RTL;
- Yosys `synth_xilinx` cell counts for the demo configuration, with the exact
  command and tool version.

## Scope

v1 is a single-sequence decode engine, verified in cycle-accurate simulation.
Performance is reported as cycles per token, bytes per cycle and utilization
from the RTL's own counters; FPGA throughput follows from those cycle counts at
a clock a place-and-route timing result supplies and a stated memory bandwidth.
Model quality is a measured perplexity, KL and top-1 delta against fp32. Prefix
reuse through KV save/restore ships: `make demo-toolcall` computes an agent
turn's system-and-tools head once on the RTL, restores it and generates the tool
call after it. That and the row dimension of the datapath are the foundations
for the paged KV cache, the batched decode and the constrained decoding on the
[roadmap](docs/ROADMAP.md).

## Results

Every row names the command that produced it. Simulation numbers are from an
Apple M5 Pro with Verilator 5.048 and Apple clang 17; synthesis is Yosys 0.65.
The workload is a compiled image at the demo configuration
`WB=64, B_MAX=1, VL=4, VSRAM_WORDS=4096` -- Qwen2.5-0.5B-Instruct, the whole
model in `build/images/qwen2.5-0.5b-instruct` and its first two layers in
`build/images/qwen2.5-0.5b-instruct-l2`, except where a row names
SmolLM2-135M-Instruct: every descriptor at its real address, stride, meta and
partial tile, generating the tokens it generates. A wall clock is the median of
a set of runs, and [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) owns that set and
its extremes; cycle counts are the core's own counters and are identical run to
run.

| Measurement | Value | Command |
|---|---|---|
| The demo, checkpoint to text | the whole SmolLM2-135M-Instruct: 96,655,010 cycles at 80.2% MAC-active, 8 ids identical to the golden model's recorded continuation, **38.91 s** end to end | `make demo` |
| The demo on the headline model | the whole Qwen2.5-0.5B-Instruct: 283,190,917 cycles at 91.3% MAC-active, the same 8 ids as its own record, **108.47 s** end to end | `make demo-qwen` |
| Decode token, whole model | 8,321,833 cycles, MAC array active in 93.0% of them, 60.0 read bytes per cycle | `make perf IMAGE=build/images/qwen2.5-0.5b-instruct PERF_ARGS="--max-new 1 --prompt-ids 1"` |
| Decode token, two layers | 2,662,949 cycles, 97.4% MAC-active, 62.9 read bytes per cycle | `make perf PERF_ARGS="--max-new 1 --prompt-ids 1"` |
| 36-token prompt + 1 decode token, two layers | 20,736,018 cycles, 91.6% MAC-active, 59.2 read bytes per cycle, 6.2% in the vector unit and 0.10% in sequencing | `make perf` |
| 32-token prompt + 20 decode tokens, whole model | 358,570,293 cycles in 127.7 s, 91.8% MAC-active, 59.3 read bytes per cycle | `make perf IMAGE=build/images/qwen2.5-0.5b-instruct PERF_ARGS="--max-new 20 --prompt-ids <the first 32 ids of prompt.tokens> --eos 999999999"` |
| Verilator simulation rate, `--threads 1` | 2.754 Mcycles/s | the two-layer decode-token run above |
| `qcore_top` on xc7 | 109 DSP48E1, 32 RAMB36E1, 15 RAMB18E1, 359 RAM32M, 354 SRL16E, 35,510 LUTs, 22,437 flops; 63,162 cells, 29,014 estimated LCs | `make synth` (Yosys `synth_xilinx -family xc7 -flatten`); the table is the Demo configuration section of `syn/reports/qcore_top.md`, which that command regenerates from its own log |
| RTL against the ISA simulator, descriptor by descriptor | 80 of 80 runs match element by element at WB=64 and WB=128, over the bring-up, vector, attention and layer programs | `make bringup-sweep` |
| RTL against the ISA simulator, generated ids | the whole SmolLM2-135M-Instruct model, 37 prompt tokens and 4 generated: `[504, 3575, 282, 4649]` on both, in 86,361,436 clock cycles | `uv run quettos compare --image build/images/smollm2-135m-instruct --wb 64 --generate 4` |
| Quality against fp32, W8A16 (integer golden model) | on 32,704 held-out WikiText-2 positions: Qwen KL 0.0044 nats, top-1 96.65% and delta-NLL -0.0037 +/- 0.0006; SmolLM2 KL 0.0041, top-1 96.10% and +0.0023 +/- 0.0005 | `uv run quettos check <alias> --heldout` |

Every row but the last runs on the hardware: a cycle, byte or utilization figure
is `qcore_top`'s own counter, and the synthesis row is a Yosys run over the same
top. The ids are the hardware's, held to `models/<name>/expected_tokens.json` --
the integer golden model's own continuation of the same prompt, checked in at the
count and SHA-256 the image was compiled against, and the four ids of the SmolLM2
row are the first four of the same record. Quality is scored a layer up, on the
integer golden model against fp32 -- the rows above hold the RTL to that model,
so scoring those 32,704 positions in simulation would measure nothing new
(`docs/VERIFICATION.md`). `make provenance` names the text the quality rows were
scored on: it rebuilds the held-out windows from the WikiText-2 archive and the
calibration ids from `prompts/`, hashes both, and holds each hash to the one
stored with the rows.

The synthesis row is `qcore_top` as it stands: the GEMV, EMBED and KVWRITE
datapath, the whole control path and `qcore_vpu_top` with its lanes, its scalar
unit and its four lookup tables. The same core before it carried a vector unit
stood at 11,484 estimated LCs and 69 `DSP48E1` in the demo configuration, and the
one whose vector unit ran four of the six V opcodes at 23,026 LCs and 101
`DSP48E1` -- read against the row above, what the unit costs and what its
rotation and softmax passes cost; `syn/reports/qcore_top.md` names the commits
both were measured at. The cycle rows include the unit's work: 6.1% of the
32 + 20 run's cycles are `STALL_VPU`.

[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) holds the full tables, the
measurement protocol, the memory model and the demo configuration;
[`docs/NUMERICS.md`](docs/NUMERICS.md) holds the quality table with its
standard errors.

## Quick start

Requirements: `uv` (Python 3.13 is pinned via `.python-version`), Verilator 5.x
and the C++ compiler it drives, `git` and `make`; a first run also needs the
network, for the packages and the checkpoint. Yosys 0.65 and Icarus Verilog 13
are the other two parsers, for `make lint`, `make synth` and `make gatesim`.
The whole EDA flow is open source.

One command takes a clean clone to text generated by the RTL. It syncs the
Python environment, fetches the checkpoint from the Hugging Face Hub, quantizes
it to int8, compiles the image and the two descriptor programs, builds the
Verilator harness and runs the whole model on `qcore_top`, printing each token
as it leaves the hardware with the cycles it cost and the share of them the MAC
array was active in.

```sh
make demo           # SmolLM2-135M-Instruct, checkpoint to text in 38.9 s
make demo-qwen      # Qwen2.5-0.5B-Instruct, the same command, 108.5 s
```

Both seconds are with the checkpoint and the Python environment already on the
machine. A first run pays those too: it syncs the environment from the package
index and fetches `model.safetensors` from the Hub -- 269,060,552 B for SmolLM2,
988,097,824 B for Qwen -- into `build/models/<name>`, which every run after it
reads. From `git clone` to the same text, with nothing on the machine the clone
did not bring, is **62 s** for SmolLM2 and **160 s** for Qwen: `make clean-clone`
runs the quick start from a clone of the committed tree with a fresh uv cache and
a fresh managed Python, and holds the result to the demo's own verdict
([`docs/VERIFICATION.md`](docs/VERIFICATION.md), layer 9).

## What the demo prints

The tokens as they leave the core, then part of the summary underneath them.

```
[token 0 pos=36 id=504 cycles=2571646 mac=82.8%] The[token 1 pos=37 id=3575 cycles=2571916 mac=82.8%]  capital ...

  8 ids generated on qcore_top, one decode program each
    ids   [504, 3575, 282, 4649, 314, 7042, 30, 2]
    text  'The capital of France is Paris.<|im_end|>'
    reference  8/8 identical to models/smollm2-135m-instruct/expected_tokens.json,
               the integer golden model's own continuation of prompts/chat_short.json

  counters that a clean run leaves at zero
    SAT_REQ 0  SAT_VPU 0  ERR_SHIFT 0  ERR_BOUNDS 0

  cycles and utilization, from the core's own PERF counters
  pass      tokens          cycles   cycles/token   MAC_ACTIVE  read B/cycle
  prefill       36      76,071,582      2,113,099        79.6%         51.70
  decode         8      20,583,428      2,572,928        82.8%         53.80
  run           44      96,655,010                       80.2%         52.15

demo: OK -- smollm2-135m-instruct on qcore_top, 96,655,010 cycles, 80.2% MAC-active,
8 ids matching the recorded reference, no saturation or range events
```

The summary is the run's own verdict. Above the excerpt it names the model, the
image with its size and SHA-256, how many of the files `layout.json` records a
hash for still match it, the core configuration, the memory settings, the
descriptor count of each program and the prompt; below it, the six busy buckets,
the traffic totals and the seconds every stage took. A counter that should be
zero and is not fails the command, and so does an id the record does not carry.

## The tool call, over a prefix the hardware computed once

An agent turn repeats a long head: the system message and the tool descriptions
are the same on every call, and only the user's turn changes.

```sh
make demo-toolcall  # the tool call over a system-and-tools prefix, saved and restored, 940 s
```

`prompts/tool_call_weather.json` renders to 180 ids on Qwen2.5-0.5B-Instruct,
and the first 162 of them are the system turn that carries the tool
description. `make demo-toolcall` prefills those 162 positions on `qcore_top`
and writes the KV region out with the record of what it belongs to; restores
that file in a second run, which prefills the 17 positions of the user's turn
and generates
`<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>`;
and runs the same prompt a third time with no reuse at all. The restore leaves
1,011,148,686 of the 1,118,717,069 prefill cycles unspent, 90.4% of them, and
the 20 ids it generates are the 20 the run without reuse generates and the 20
the integer golden model recorded -- every position of both runs costing the
same cycles. A file whose record does not match the run offered it is refused
before the first cycle, which the demo shows by offering one.
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) holds the table and
[`docs/MEMORY_MAP.md`](docs/MEMORY_MAP.md) the prefix file format.

## The pipeline, command by command

`make demo` drives the download, the quantize and the compile; the rest
regenerate or check what is committed beside them.

```sh
uv run quettos download qwen        # or smollm2; fetches the checkpoint from Hugging Face
uv run quettos calibrate qwen       # activation ranges -> models/<name>/calib.json
uv run quettos quantize qwen        # int8 weights + constants -> build/quant/<name>.npz
uv run quettos golden qwen          # greedy generation on the integer golden model
uv run quettos corpus               # the held-out WikiText-2 archive -> build/corpus/, checked against its sha256
uv run quettos check qwen           # quality vs fp32 on the calibration set -> models/<name>/quality.json
uv run quettos check qwen --heldout # the held-out rows of the same file: 64 windows of 512 tokens
uv run quettos provenance qwen      # which text each published quality row was scored on, rebuilt and hashed
uv run quettos compile qwen --wb 64 --prompt prompts/chat_short.json   # image.bin, decode/prefill.prog, layout.json
uv run quettos isa-sim qwen --compare   # the programs on the ISA simulator, every descriptor against the golden model
uv run quettos compare --image build/images/qwen2.5-0.5b-instruct --wb 64 --generate 1   # the ids on qcore_top, the ISA simulator and the record
uv run quettos csr-defs --check     # the generated ISA/CSR headers match sw/quettos/isa.py

uv run quettos quantize qwen --no-qk-smoothing  # the ablation the quality table reports beside the shipped model
uv run quettos check qwen --no-qk-smoothing     # its rows in the same quality.json
uv run quettos compile qwen --layers 2          # the two-layer image make perf runs by default
```

`make demo`, `make perf` and `make bringup` build the harness themselves.
`make demo MODEL=qwen MAX_NEW=4` picks the model and the number of decode steps,
`make demo DEMO_ARGS="-- --lat 200"` passes flags to the harness run, and
`make perf IMAGE=<dir>` picks the image to run.

## Other targets

```sh
make lint           # three-parser RTL lint (Verilator, Yosys, Icarus)
make style          # ruff check and ruff format --check over every .py
make test           # uv run pytest -q sw/tests
make cocotb         # cocotb + Verilator block benches (sim/cocotb)
make synth          # Yosys synth_xilinx per block and whole core; checks syn/reports/
make gatesim        # each Yosys netlist against the source it came from (sim/gatesim)
make bringup-sweep  # qcore_top vs the ISA simulator over random shapes at WB=64 and WB=128
make stepcmp-model  # whole compiled programs, descriptor by descriptor, over both complete models
make determinism    # one program run every way the machine allows: the values do not move
make perf           # run a compiled program on qcore_top -> build/perf/perf.json
make provenance     # which text every published quality row was scored on, rebuilt and hashed
make clean-clone    # clone the committed tree elsewhere and run the quick start there
make ci             # every job of .github/workflows/ci.yml, and make determinism beside them
```

`make help` prints them all, with the demo above, from the Makefile itself.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) -- thesis, block diagram, module list, dataflow, host/RTL boundary
- [`docs/ISA.md`](docs/ISA.md) -- descriptor layout, opcodes, binary formats, CSR table
- [`docs/RTL.md`](docs/RTL.md) -- the module interface specification: every port, handshake and cycle-level guarantee
- [`docs/NUMERICS.md`](docs/NUMERICS.md) -- integer numerics specification and the measured quality table
- [`docs/MEMORY_MAP.md`](docs/MEMORY_MAP.md) -- image, VSRAM and KV layouts
- [`docs/VERIFICATION.md`](docs/VERIFICATION.md) -- oracle chain and test layers
- [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) -- the measurement protocol, the measured cycle, utilization and rate tables, the harness and the speed probe
- [`docs/ROADMAP.md`](docs/ROADMAP.md) -- what ships in v1, then v1.1 .. v2
- [`rtl/cfg/README.md`](rtl/cfg/README.md) -- parameter sets
- [`sim/cocotb/README.md`](sim/cocotb/README.md) -- the block benches and how to add one
- [`sim/gatesim/README.md`](sim/gatesim/README.md) -- gate-level equivalence: what it covers and what it cannot
- [`CONTRIBUTING.md`](CONTRIBUTING.md) -- SystemVerilog subset and process

## License

Apache-2.0 (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). Model weights are
downloaded at build time from the Hugging Face Hub and are not redistributed.
