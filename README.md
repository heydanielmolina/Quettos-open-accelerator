# Quettos Core

A decode-first LLM inference accelerator in synthesizable SystemVerilog,
simulated cycle-accurately with Verilator and synthesized with Yosys. Free
tools only.

Quettos Core is an inference-only, weight-streaming LLM accelerator written in
synthesizable SystemVerilog and driven by a small descriptor ISA. The goal of
v1 is to show a real language model (Qwen2.5-0.5B-Instruct, with
SmolLM2-135M-Instruct as the fast regression model) generating text on the RTL
in cycle-accurate Verilator simulation, bit-exact against an integer golden
model, and to report Yosys synthesis numbers for an FPGA-class target. Every
weight byte streams through a `WB`-byte-per-cycle port once per token and is
consumed by `WB` output-stationary lanes the cycle it arrives; activations and
per-token scales stay in a 128 KB on-chip vector SRAM. The design thesis
(decode is the unit of work for agent inference) and the roadmap are in
[`docs/`](docs/README.md).

## What v1 delivers

- `qcore_top`, a buildable core in the SystemVerilog subset that Verilator 5,
  Yosys 0.65 and Icarus 13 all accept with zero rewrites;
- a compiler (`sw/quettos/`) that quantizes a Hugging Face checkpoint to int8
  weights / int16 activations, lays out the memory image, and emits
  straight-line descriptor programs;
- a layered oracle, integer golden model -> ISA simulator -> RTL: those three
  agree bit for bit -- the golden model and the ISA simulator over both
  complete models, and `qcore_top` against the ISA simulator element by
  element over the `EMBED`, matrix-vector and vector programs, over the
  attention step of a compiled `decode.prog` at six positions and over a whole
  decoder layer at the same six, at WB 16 through `make bringup` and at WB 64
  and 128 through `make bringup-sweep`. The float32 forward of
  `sw/quettos/reference_np.py` sits above the three as the quality reference and
  is never one of them: it is a different arithmetic, measured as a top-1, KL
  and perplexity delta, and it is itself held to the `transformers` fp32 forward
  on the same weights, argmax for argmax at every position
  (`sw/tests/test_reference_np.py`);
- a C++ Verilator harness with a fixed-latency memory model that runs a whole
  compiled program on the RTL -- every descriptor, all six vector opcodes on
  the vector unit -- for its traffic and its hardware performance counters, and
  prints tokens as they leave the RTL;
- Yosys `synth_xilinx` cell counts for the demo configuration, with the exact
  command and tool version.

## Scope

v1 is a single-sequence decode engine, verified in cycle-accurate simulation.
Performance is reported as cycles per token, bytes per token and utilization
from the RTL's own counters; FPGA throughput follows from those cycle counts at
a clock a place-and-route timing result supplies and a stated memory bandwidth.
Model quality is a measured perplexity, KL and top-1 delta against fp32. The row
dimension of the datapath and prefix reuse through KV save/restore are the
foundations for batched decode and a paged KV cache. Those two, constrained
decoding, and the tool-call demo that shows it off are
[roadmap](docs/ROADMAP.md) items.

## Results

Every row names the command that produced it. Simulation numbers are from an
Apple M5 Pro with Verilator 5.048 and Apple clang 17; synthesis is Yosys 0.65.
The workload is a compiled image at the demo configuration
`WB=64, B_MAX=1, VL=4, VSRAM_WORDS=4096` -- Qwen2.5-0.5B-Instruct, the whole
model in `build/images/qwen2.5-0.5b-instruct` and its first two layers in
`build/images/qwen2.5-0.5b-instruct-l2`, except where a row names
SmolLM2-135M-Instruct: every descriptor of the compiled program at its real
address, stride, meta and partial tile, generating the tokens it generates. A
wall clock is a median of a set of runs; the set, its extremes and its size are
on [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md), which owns them. Cycle counts
are the core's own counters and are identical run to run.

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
| Quality against fp32, W8A16 (integer golden model) | Qwen KL 0.0124 nats and top-1 95.57%; SmolLM2 KL 0.0048 and top-1 95.66% | `uv run quettos check <alias>` |

Every row but the last runs on the hardware: a cycle, byte or utilization figure
is `qcore_top`'s own counter and the synthesis row is a Yosys run over the same
top. The generated ids close the oracle chain from the top: the ids of the two
demo rows are the integer golden model's own continuation of the same prompt,
recorded in `models/<name>/expected_tokens.json` and held to the count and
SHA-256 the image was compiled against, and the four of the SmolLM2 row are the
first four of the same record. Quality is scored one layer up, on the integer
golden model against fp32, because the RTL reproduces that model bit for bit and
scoring a thousand positions in simulation would measure nothing new
(`docs/VERIFICATION.md`).

The synthesis row is `qcore_top` as it stands: the GEMV, EMBED and KVWRITE
datapath, the whole control path and `qcore_vpu_top` with its lanes, its scalar
unit and its four lookup tables. The same core before it carried a vector unit
stood at 11,484 estimated LCs and 69 `DSP48E1` in the demo configuration, and
the one whose vector unit ran the other four V opcodes at 23,026 LCs and 101
`DSP48E1`; read against the row above, those two are what the unit costs and
what its rotation and softmax passes cost, and the note in
`syn/reports/qcore_top.md` names the commits both were measured at. The cycle
rows include the unit's work: 21,845,204 of the 32 + 20 run's cycles, 6.1%, are
`STALL_VPU`.
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) holds the full tables, the
measurement protocol, the memory model and the demo configuration;
[`docs/NUMERICS.md`](docs/NUMERICS.md) holds the quality table with its
standard errors.

## Status

`qcore_top` runs a descriptor program from `PC` to its `HALT` and matches the
ISA simulator element by element, the vector unit included. The bring-up program is the four
descriptors of `docs/ISA.md` -- `EMBED`, `VQUANT`, the tied LM head as a `GEMV`
in ARGMAX mode, `HALT` -- so the core writes its own activation scale and the
host loads no register.

The attention step runs the same way: the `EMBED` through the last value `GEMV`
of a compiled `decode.prog`, in the compiler's own order -- the rotation, the
`K^T` matrix-vector whose length comes from the position, the softmax, and the
value matrix-vector whose depth comes from it -- at six positions on one
machine, so the KV cache each position writes is what the next one reads. The
whole decoder layer runs the same prefix to its end. Both match the ISA
simulator on every VSRAM element, scale register, KV byte and dumped logit, and
on the registers and counters layer 4 of `docs/VERIFICATION.md` lists, at WB 16,
64 and 128. Run from the top, the prefill/decode loop on a compiled image emits
the ids the ISA simulator emits, and the ISA simulator reproduces the integer
golden model. `make demo` runs that whole path in one
command -- the checkpoint, the quantizer, the compiler, the harness build and
the generation -- and holds the ids the hardware emits to the golden model's
record.

The software stack is complete: integer numerics, lookup tables, calibration
and the int8 quantizer in `sw/quettos/`, a bit-exact integer golden model that
generates text for both models (`models/*/expected_tokens.json`), and the
measured quality against fp32 in `models/*/quality.json`. The descriptor ISA
(`sw/quettos/isa.py`, generated into `rtl/qcore_csr_defs.svh` and
`sim/verilator/csr_defs.hpp`), the compiler (`image.bin`, `decode.prog`,
`prefill.prog`, `layout.json`) and the ISA simulator are in place, and the
simulator reproduces the golden model bit for bit on both complete models
(`uv run quettos isa-sim <alias> --compare`).

`rtl/` holds eighteen `.sv` files, the package and seventeen modules: the vector
SRAM, the MAC lane group, the row, the requant, the memory arbiter, the stream
controller, the CSR file, the descriptor fetch unit, the dispatcher, the KV
writer, the performance counters, the lookup-table ROM and its interpolator,
the vector lane, the vector scalar unit, the vector unit itself and
`qcore_top`. `make lint` takes those and the GEMV-path assembly under
`sim/cocotb/wrappers/` through Verilator `-Wall -Wpedantic`, Yosys
`check -assert` and Icarus `-g2012`, each as its own top, with zero warnings
and no waivers. The sixteen cocotb benches in `sim/cocotb/` cover every block
below `qcore_top`, each against the reference that fits it:
the requant, the assembled GEMV path, the lookup tables, the vector lane and the
vector scalar unit bit for bit against `sw/quettos/numerics.py`, the KV writer
and the vector unit against `sw/quettos/isa_sim.py`, the MAC lane group and the
row against the exact integer matmul, and the CSR file, the performance
counters, the descriptor fetch unit, the dispatcher, the memory arbiter, the
stream controller and the vector SRAM against the handshakes and cycle-level
guarantees of `docs/RTL.md`, with the dispatcher's decoded fields taken from
`isa_sim.py` as well.
`make gatesim` compares fifteen of the seventeen modules' Yosys netlists
against the source they came from, cycle by cycle, in nineteen configurations;
`sim/gatesim/README.md` names the two it leaves out and why the Yosys cell
library cannot simulate them. `make bringup` and `make bringup-sweep` run four
programs on `qcore_top` through the C++ harness and on the ISA
simulator and compare, after every descriptor, every VSRAM element, every SREG
word, every dumped logit, `PC`, `STATUS`, the ARGMAX registers, the four event
counters and `DESCRIPTORS` / `MACS` / `WT_BYTES`, and over the attention step
and the whole decoder layer of a compiled `decode.prog` they add the KV region
as bytes at every one of six positions; `make perf` runs a whole compiled
program on the RTL and `make demo` builds one from the checkpoint and runs it;
and `make synth` runs Yosys `synth_xilinx` over every block and over the whole
core, regenerating `syn/reports/*.md` from the run and failing if a report no
longer reproduces.

`qcore_vpu_top` executes all six vector opcodes -- `VRMSNORM`, `VQUANT`,
`VROPE`, `VSILUMUL`, `VSOFTMAX` and `VSUBC` -- so every opcode the ISA defines
issues to a unit and `FAULT = OPCODE` names one thing: an opcode byte that is
none of the twelve, which stops the program with `STATUS.ERR`, that byte in
`FAULT_OP` and `PC` on the descriptor. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for where the unit sits and
[`docs/ROADMAP.md`](docs/ROADMAP.md) for what comes after v1.

## Quick start

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

Both seconds are with the checkpoint already on the machine. The first run on a
clean clone fetches it from the Hub as well -- 269,060,552 B of
`model.safetensors` for SmolLM2, 988,097,824 B for Qwen -- into
`build/models/<name>`, which every run after it reads.

An excerpt of what the command prints: the tokens as they leave the core, then
part of the summary underneath them.

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
descriptor count of each program and the prompt the run was given; below it,
the six busy buckets, the traffic totals and the seconds every stage took.
Between them is what the excerpt shows: the ids the hardware produced against
the integer golden model's recorded continuation of the same prompt, the four
counters a clean run leaves at zero, and the cycle and utilization table from
`qcore_top`'s counters. A counter that should be zero and is not fails the
command, and so does an id the record does not carry. The demo runs no
reference model of its own: the ids are the hardware's, and the file they are
held to is checked in.

The rest of the targets:

```sh
make lint           # three-parser RTL lint (Verilator, Yosys, Icarus), 19 tops
make style          # ruff check and ruff format --check over every .py
make test           # uv run pytest -q sw/tests
make cocotb         # cocotb + Verilator block benches (sim/cocotb)
make synth          # Yosys synth_xilinx per block and whole core; checks syn/reports/
make gatesim        # each Yosys netlist against the source it came from (sim/gatesim)
make harness        # build the C++ Verilator harness (sim/verilator)
make harness-csr    # the harness CSR driver against rtl/qcore_csr.sv
make bringup        # qcore_top vs the ISA simulator, tiny configuration
make bringup-sweep  # the same over random shapes at WB=64 and WB=128
make stepcmp        # whole compiled programs, descriptor by descriptor, vs the ISA simulator
make stepcmp-models # the same over truncated Qwen and SmolLM2 images
make stepcmp-model  # the same over the complete Qwen and SmolLM2 models
make determinism    # one program run every way the machine allows: the values do not move
make determinism-model # the same timing and width checks over a compiled image at both widths
make determinism-x  # the same programs from an undefined start (--x-initial unique)
make perf           # run a compiled program on qcore_top -> build/perf/perf.json
make regen-prefix   # prefill IMAGE's prompt and save the KV region to PREFIX_KV
make waves          # a VCD of the first WAVE_CYCLES cycles of a run -> build/waves
make probe          # Verilator speed probe (sim/probe)
make ci             # the job set of .github/workflows/ci.yml, run locally
make clean          # remove build/ and the Verilator object directories
```

`make help` prints them all, with the demo above, from the Makefile itself.

The pipeline behind `make demo`, step by step, from a Hugging Face checkpoint
to an image the RTL runs. It runs top to bottom as it stands; `make demo`
drives the download, the quantize and the compile, and the rest regenerate or
check what is committed beside them.

```sh
uv run quettos download qwen        # or smollm2; fetches the checkpoint from Hugging Face
uv run quettos calibrate qwen       # activation ranges -> models/<name>/calib.json
uv run quettos quantize qwen        # int8 weights + constants -> build/quant/<name>.npz
uv run quettos golden qwen          # greedy generation on the integer golden model
uv run quettos check qwen           # quality vs fp32 -> models/<name>/quality.json
uv run quettos compile qwen --wb 64 --prompt prompts/chat_short.json   # image.bin, decode/prefill.prog, layout.json -> build/images/qwen2.5-0.5b-instruct/
uv run quettos isa-sim qwen --compare   # run the programs on the ISA simulator, every descriptor vs golden
uv run quettos compare --image build/images/qwen2.5-0.5b-instruct --wb 64 --generate 1   # the ids on qcore_top, the ISA simulator and the record
uv run quettos csr-defs --check     # the generated ISA/CSR headers match sw/quettos/isa.py
```

Two builds off to the side of that line: the ablation the quality table reports
beside the shipped model, and the two-layer image `make perf` runs by default.

```sh
uv run quettos quantize qwen --no-qk-smoothing  # the same build without the Q/K smoothing fold -> build/quant/<name>-nosmooth.npz
uv run quettos check qwen --no-qk-smoothing     # its ablation rows in the same quality.json
uv run quettos compile qwen --layers 2          # the first two layers -> build/images/qwen2.5-0.5b-instruct-l2/
```

`make demo`, `make perf` and `make bringup` build the harness themselves;
`make perf` reads the image named by `IMAGE` (default
`build/images/qwen2.5-0.5b-instruct-l2`, the two-layer compile above), and
`make demo` builds `build/images/<name>` from the checkpoint. `make demo
MODEL=qwen MAX_NEW=4` picks the model and the number of decode steps, and
`make demo DEMO_ARGS="-- --lat 200"` passes flags to the harness run.

Requirements: Verilator 5.x, Yosys 0.65, Icarus Verilog 13, `uv` (Python 3.13
is pinned via `.python-version`). No vendor tools, no tokens, no secrets.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) -- thesis, block diagram, dataflow, host/RTL boundary
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
