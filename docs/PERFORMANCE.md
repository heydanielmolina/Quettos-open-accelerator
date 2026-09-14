# Performance

Every figure on this page comes from a run taken on the machine named below,
with the command that produced it beside it. Cycle counts are the core's own
PERF counters, read out of `qcore_top` through the CSR window; wall clocks and
rates are the harness clock loop over `std::chrono::steady_clock`. Every figure
in a table is a run of the command beside it; where the prose divides two of
them it names both and calls the result a ratio, and an analytical figure is
labelled **estimate**.

## Measurement protocol

- **Machine and tools.** Apple M5 Pro, 18 cores in the two performance levels
  macOS reports -- 6 `Super` (`hw.perflevel0`) and 12 `Performance`
  (`hw.perflevel1`) -- 64 GB, macOS 26.5.1 (Darwin 25.5.0), Apple clang 17.0.0,
  Verilator 5.048,
  Yosys 0.65. 2026-09-07. Threads were not pinned; `numactl` does not exist on
  macOS.
- **Cycles.** A cycle count is `PERF[CYCLES]`, which the core clears at `START`
  and snapshots at the `HALT` that retires the program, so it is the program's
  own cost. The harness `RESULT` line reports `clock_cycles` instead: the same
  count plus the cycles the host loop spends writing `TOK` / `POS`, pulsing
  `START` and polling `STATUS`, 48 per token here. Both are given where they
  differ. Cycle counts are exact and repeat run to run; every count on this page
  was identical across every run of its command.
- **Rates and wall clock.** Every rate and every wall clock is the median of a
  set of consecutive runs of one command on an otherwise idle machine, quoted
  with the extremes of that set and the size of the set. The long runs carry
  smaller sets than the short ones, and every table names its `n`. A wall clock
  is quoted to the place its own spread supports -- the coarsest digit that
  still divides the range in two -- and its extremes to that same place.
- **Machine load.** Every set here was taken on an otherwise idle machine,
  because work running beside a run costs it wall clock: on a working laptop or
  a shared runner a rate reads low and a wall clock reads high, and that is the
  direction to expect against the figures below. What the machine is doing
  never reaches a cycle count (**Cycles** above); it reaches the seconds and the
  rates. Quoting a median with its range and its run count is what makes the
  wall clocks reproducible: every set on this page carries all three, and the
  widest spread among them is 11.8% of its median -- the demo's sub-second
  checkpoint stage, where a hundredth of a second is 3%.
- **Workload.** `make perf` and `make demo` run a compiled program on the RTL:
  every descriptor of `decode.prog` and `prefill.prog` at its real address,
  stride, meta and partial tile, all six vector opcodes on `qcore_vpu_top`, and
  the tokens the run generates printed as they leave the core. The traffic, the
  cycles, the counters and the values are the compiled program's. `make perf`
  starts from a compiled image; `make demo` starts from the checkpoint and
  builds one (**The demo** below).
- **Memory model.** `image.bin` mapped copy-on-write behind the QMEM ports of
  `docs/RTL.md` 2.1: fixed read latency (`--lat`, default 32), one returned beat
  every `--bw-div` cycles (default 1), in-order returns across tags, a 64-beat
  in-flight window that throttles `rd_req_ready`, byte-strobed writes and their
  acks after the same latency.
- **Verilator flags.** `--cc --exe --build -j 0 -O3 --x-assign fast
  --x-initial fast --no-assert --no-timing -Wall --output-split 20000
  -CFLAGS -O2`, plus `--threads` and the `-G` width parameters. Zero Verilator
  warnings at build.

## The core

`rtl/qcore_top.sv` at the demo configuration -- `WB=64, B_MAX=1, VL=4,
VSRAM_WORDS=4096, FIFO_BEATS=128, META_FIFO_BEATS=16, ACC_W=40, MAX_BURST=64,
DQ_DEPTH=8, VPU_FIFO_BEATS=16` -- holds `qcore_csr`, `qcore_seq_fetch`,
`qcore_seq_dispatch`, `qcore_perf`, `qcore_mem_arb`, `qcore_stream_ctrl`, the
`qcore_row` and `qcore_vsram` pair, `qcore_requant`, `qcore_kv_writer`, and
`qcore_vpu_top` with its four vector lanes, its scalar unit and its four lookup
tables. Every measurement in this section is that top.

### Verilator build

Each row starts from an empty `build/verilator`, so every build is cold.

| Configuration | Cold build, median of 5 | Range |
|---|---|---|
| Demo, `WB=64 B_MAX=1 VL=4 VSRAM_WORDS=4096`, `--threads 1` | **1.59 s** | 1.58 - 1.66 |
| Tiny, `WB=16 B_MAX=2 VL=2 VSRAM_WORDS=2048` | 1.56 s | 1.53 - 1.56 |
| Demo, `--threads 4` | 1.62 s | 1.59 - 1.62 |
| `WB=128 B_MAX=1 VL=4 VSRAM_WORDS=4096` | 1.63 s | 1.62 - 1.64 |

The object directory is `build/verilator/<top>-<hash>` and the hash covers the
RTL, the lookup-table images, the C++ and the configuration, so an unchanged
tree relinks nothing: `make -C sim/verilator build` returns in 0.03 s.

```sh
make -C sim/verilator clean                                       # make the build cold
make -C sim/verilator build WB=64 B_MAX=1 VL=4 VSRAM_WORDS=4096   # prints build_seconds
```

### The cost of one token

Every count is `PERF[CYCLES]` for that token, at `POS = 0` so the rows compare.
A token costs more as `POS` grows, in steps rather than along a line: the scores
GEMV takes one more weight-port tile every `WB` positions, the value GEMV one
more token every position, and the softmax one more chunk every `VC`. The
36 + 1 run measures those steps directly -- its prefill token goes from 515,531
cycles at `POS = 0` to 517,155 at `POS = 34`, in 34 steps of 0 (eight of them),
28 (eighteen) and 140 (eight) -- and a run's total is the sum of its tokens'
counts, each of which `perf.json` records, so no total on this page is a fit.

| Model | Program | Cycles | `MAC_ACTIVE` | Read bytes per cycle |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct, 2 layers | decode | **2,662,949** | 97.4% | 62.93 |
| Qwen2.5-0.5B-Instruct, 2 layers | prefill | 515,531 | 90.7% | 58.63 |
| Qwen2.5-0.5B-Instruct, 24 layers | decode | **8,321,833** | 93.0% | 60.01 |
| Qwen2.5-0.5B-Instruct, 24 layers | prefill | 6,174,415 | 90.9% | 58.64 |
| SmolLM2-135M-Instruct, 30 layers | decode | 2,554,636 | 82.9% | 53.86 |
| SmolLM2-135M-Instruct, 30 layers | prefill | 2,105,202 | 79.6% | 51.72 |

Every row is a run of one token at `POS = 0`, so both the utilization and the
read-bytes-per-cycle columns are that token's own. A prefill row runs a
two-token prompt, whose first token is the one measured.

```sh
make perf IMAGE=build/images/qwen2.5-0.5b-instruct-l2 PERF_ARGS="--max-new 1 --prompt-ids 1"
make perf IMAGE=build/images/qwen2.5-0.5b-instruct    PERF_ARGS="--max-new 1 --prompt-ids 1"
make perf IMAGE=build/images/smollm2-135m-instruct    PERF_ARGS="--max-new 1 --prompt-ids 1"
make perf IMAGE=build/images/qwen2.5-0.5b-instruct    PERF_ARGS="--max-new 0 --prompt-ids 1,1"
```

The decode token streams the whole model once: 497,610,632 weight and meta bytes
for the 24-layer Qwen and 499,431,104 read bytes in all, in 7,803,611 returned
beats over 8,321,833 cycles. The weight port is therefore busy in 93.8% of the
token's cycles, which is what `MAC_ACTIVE` at 93.0% measures from the other side.

A prefill token costs less because it skips the LM head: 360,260,488 weight and
meta bytes against the decode token's 497,610,632, and the difference of
137,350,144 is exactly the tied 151,936 x 896 head plus the eight meta bytes per
output channel.

### Where the cycles go

`qcore_seq_dispatch` classifies every busy cycle into one of six exclusive
buckets, and the harness fails a run whose buckets do not sum to `BUSY`.

| Run | Cycles | `MAC_ACTIVE` | `STALL_MEM` | `STALL_VPU` | `STALL_KV` | `STALL_SEQ` | `STALL_DRAIN` |
|---|---|---|---|---|---|---|---|
| Qwen 2 layers, decode token | 2,662,949 | 97.44% | 0.96% | 1.37% | 0.021% | 0.022% | 0.193% |
| Qwen 2 layers, 36-token prompt + 1 decode | 20,736,018 | 91.55% | 1.21% | 6.17% | 0.095% | 0.099% | 0.881% |
| Qwen 24 layers, decode token | 8,321,833 | 93.01% | 1.03% | 5.09% | 0.078% | 0.077% | 0.722% |
| Qwen 24 layers, 32 prompt + 20 decode | 358,570,293 | 91.82% | 1.05% | 6.09% | 0.093% | 0.091% | 0.854% |
| SmolLM2 30 layers, 32 prompt + 20 decode | 116,937,992 | 80.97% | 2.00% | 14.03% | 0.463% | 0.276% | 2.250% |
| SmolLM2 30 layers, the demo (37 prompt + 8 decode) | 96,655,010 | 80.24% | 2.03% | 14.61% | 0.484% | 0.288% | 2.348% |
| Qwen 24 layers, the demo (36 prompt + 8 decode) | 283,190,917 | 91.34% | 1.07% | 6.48% | 0.099% | 0.097% | 0.911% |

`BUSY` equals `CYCLES` in every run here, so each share is a share of the whole
run. The vector unit's is the `STALL_VPU` column: 6.1% of the Qwen demo run and
14.0% of the SmolLM2 one, which has 30 narrower layers and so more vector passes
per weight byte. `STALL_SEQ`, the cost of fetching and decoding descriptors, is
between 0.02% and 0.28% across these runs.

### Whole compiled programs

| Run | Counted cycles | Clock cycles | Wall clock, median | Range, n |
|---|---|---|---|---|
| Qwen 2 layers, 36-token prompt + 1 decode (`make perf`) | 20,736,018 | 20,737,746 | **7.5 s** | 7.4 - 7.8, n=9 |
| Qwen 24 layers, 32 prompt + 20 decode | 358,570,293 | 358,572,741 | **128 s** | 126 - 129, n=3 |
| Qwen 24 layers, the demo (36 prompt + 8 decode) | 283,190,917 | 283,192,981 | **102.3 s** | 101.7 - 102.5, n=3 |
| SmolLM2 30 layers, the demo (37 prompt + 8 decode) | 96,655,010 | 96,657,122 | **34.8 s** | 34.8 - 35.1, n=3 |
| Qwen 24 layers, 4 prompt + 1 decode | 26,846,758 | 26,846,950 | 9.37 s | 9.30 - 9.42, n=9 |
| SmolLM2 30 layers, 32 prompt + 20 decode | 116,937,992 | 116,940,440 | 41.31 s | 41.22 - 41.35, n=3 |
| SmolLM2 30 layers, 8 prompt + 4 decode | 24,977,368 | 24,977,896 | 8.76 s | 8.72 - 8.81, n=9 |
| Qwen 2 layers, 35 prefill tokens (`make regen-prefix`) | 18,071,445 | 18,073,125 | 6.41 s | 6.37 - 6.56, n=9 |
| Qwen 24 layers, 35 prefill tokens | 216,438,845 | 216,440,525 | 77.7 s | 77.1 - 77.8, n=3 |

The two demo rows are what `make demo-qwen` and `make demo` run, and the wall
clock beside them is the harness clock loop inside the run; the section below
carries the whole command, from the checkpoint. Four more rows are reference
points rather than one-off measurements: the Qwen 32 + 20 shape the
configuration was chosen by with the SmolLM2 run of the same shape beside it,
and the Qwen 4 + 1 and SmolLM2 8 + 4 shapes, which are what a run sized to a CI
budget costs. The nightly end-to-end jobs run each image's own prompt instead,
plus one generated token on Qwen and four on SmolLM2
(`.github/workflows/nightly.yml`).

```sh
make perf                                                     # the first row
make perf PERF_ARGS="--max-new 0"                             # prefill tokens only

# The prompt an image was compiled with, cut to the length a row names.
ids() { tr '\n' ',' < "$1/prompt.tokens" | sed 's/,$//' | cut -d, -f1-"$2"; }
QWEN=build/images/qwen2.5-0.5b-instruct
SMOL=build/images/smollm2-135m-instruct

make perf IMAGE=$QWEN PERF_ARGS="--max-new 20 --prompt-ids $(ids $QWEN 32) --eos 999999999"
make perf IMAGE=$QWEN PERF_ARGS="--max-new 1 --prompt-ids $(ids $QWEN 4) --eos 999999999"
make perf IMAGE=$SMOL PERF_ARGS="--max-new 20 --prompt-ids $(ids $SMOL 32) --eos 999999999"
make perf IMAGE=$SMOL PERF_ARGS="--max-new 4 --prompt-ids $(ids $SMOL 8) --eos 999999999"
```

`--eos 999999999` is an id the vocabulary does not contain, so a loop runs its
full length whatever the model generates and every row above is the fixed shape
its command names.

### The demo

`make demo` is the whole pipeline in one command (`scripts/demo.sh`): it syncs
the Python environment, fetches the checkpoint from the Hugging Face Hub,
quantizes it to int8, compiles `image.bin` and the two descriptor programs,
builds the Verilator harness and runs the model on `qcore_top`, printing the
text as it leaves the hardware and, under it, what each token cost in cycles and
in the share of them the MAC array was active in. `quettos demo-report`
finishes it: every file `layout.json` carries a SHA-256 for, hashed again and
held to it; the prompt the
run was given, held to the ids the image was compiled with; the ids against the
golden model's recorded continuation; the four counters a clean run leaves at
zero; the cycle and utilization table; and the seconds every stage took. Any
counter that should be zero and is not fails the command, and so does an id the
record does not carry (`docs/VERIFICATION.md`, layer 5).

`make demo` runs SmolLM2-135M-Instruct and `make demo-qwen`
Qwen2.5-0.5B-Instruct. Both generate from `prompts/chat_short.json` until the
model's own end-of-sequence id, which is eight tokens on each, and both print
`The capital of France is Paris.<|im_end|>` -- the same text, id for id, that
the integer golden model recorded in `models/<name>/expected_tokens.json`.

| Stage | SmolLM2, median (range) | Qwen, median (range) |
|---|---|---|
| `uv sync --frozen --inexact` | 0.01 s | 0.01 s |
| `quettos download` | 0.34 s (0.31 - 0.35) | 0.33 s (0.33 - 0.34) |
| `quettos quantize` | 1.38 s (1.36 - 1.39) | 3.06 s (3.06 - 3.10) |
| `quettos compile` | 0.52 s (0.51 - 0.52) | 0.93 s (0.92 - 0.94) |
| harness build | 1.59 s (1.58 - 1.61) | 1.61 s (1.60 - 1.61) |
| the run on `qcore_top` | 35.1 s (35.0 - 35.4) | 102.5 s (101.9 - 102.8) |
| **end to end** | **38.9 s** (38.8 - 39.2) | **108.5 s** (107.9 - 108.8) |

n=3 each, and the run itself times every stage and prints the table. Each run
started cold: `build/quant/<name>.npz`, `build/images/<name>` and
`build/verilator` were removed before it, so the quantizer, the compiler and
Verilator all did their work again. The checkpoint and the uv cache were already
on the machine, which is what those two rows measure; a first clone pays the Hub
fetch once, 269,060,552 B of `model.safetensors` for SmolLM2 and 988,097,824 B
for Qwen.

The RTL is most of both runs -- 35.1 s of the 38.9 s and 102.5 s of the
108.5 s -- and what it produced is in the two demo rows above: 96,655,010
cycles at 80.24% `MAC_ACTIVE` and 52.15 read bytes per cycle for SmolLM2,
283,190,917 at 91.34% and 58.94 for Qwen, with `SAT_REQ`, `SAT_VPU`,
`ERR_SHIFT` and `ERR_BOUNDS` all zero. The prefill and decode halves are
separate rows of the summary the run prints:

| Run | Pass | Tokens | Cycles | Cycles per token | `MAC_ACTIVE` | Read bytes per cycle |
|---|---|---|---|---|---|---|
| SmolLM2 30 layers | prefill | 36 | 76,071,582 | 2,113,099 | 79.56% | 51.70 |
| SmolLM2 30 layers | decode | 8 | 20,583,428 | 2,572,928 | 82.76% | 53.80 |
| Qwen 24 layers | prefill | 35 | 216,438,845 | 6,183,967 | 90.86% | 58.62 |
| Qwen 24 layers | decode | 8 | 66,752,072 | 8,344,009 | 92.92% | 59.98 |

The cycles column is the sum of those tokens' own counts and the one beside it
is that sum over the token count; the per-token records `perf.json` keeps carry
the steps a position costs, which **The cost of one token** above measures.

```sh
make demo                     # SmolLM2-135M-Instruct, checkpoint to text
make demo-qwen                # Qwen2.5-0.5B-Instruct, the same command
make demo MODEL=qwen MAX_NEW=4        # fewer decode steps
make demo DEMO_ARGS="--fresh"         # quantize and compile again
make demo DEMO_ARGS="-- --lat 200"    # flags after -- go to the harness run
```

### Simulation rate

| Configuration | Workload | Mcycles/s, median | Range, n |
|---|---|---|---|
| Demo, `--threads 1` | decode token, 2 layers | **2.754** | 2.679 - 2.787, n=9 |
| Demo, `--threads 1` | decode token, 24 layers | 2.803 | 2.707 - 2.829, n=9 |
| Demo, `--threads 1` | 36-token prompt + 1 decode | 2.749 | 2.655 - 2.789, n=9 |
| Demo, `--threads 1` | 32 prompt + 20 decode, 24 layers | 2.807 | 2.778 - 2.857, n=3 |
| Demo, `--threads 1` | the demo run, 24-layer Qwen | 2.769 | 2.763 - 2.786, n=3 |
| Demo, `--threads 1` | the demo run, 30-layer SmolLM2 | 2.776 | 2.753 - 2.779, n=3 |
| Demo, `--threads 4` | decode token, 2 layers | 0.395 | 0.392 - 0.397, n=9 |
| `WB=128`, `--threads 1` | decode token, 2 layers | 0.971 | 0.892 - 0.976, n=9 |

Every set is consecutive runs of one command on an otherwise idle machine.

`--threads 4` is **7.0x slower than one thread**, and the cycle count is
identical to the bit (2,662,997 clock cycles either way), so the threaded build
is deterministic and functionally the same run. The design is far too small for Verilator's MTask partitioning; per-`eval`
thread synchronization dominates. **`--threads 1` is the default.**

### Memory latency

`--lat` sets the QMEM read latency and the write-ack latency. The decode token
of the two-layer image, five runs per setting.

| `--lat` | Counted cycles | `MAC_ACTIVE` | Read bytes per cycle | Mcycles/s, median (range) |
|---|---|---|---|---|
| 1 | 2,658,984 | 97.6% | 63.02 | 2.807 (2.785 - 2.844) |
| 8 | 2,659,853 | 97.6% | 63.00 | 2.806 (2.751 - 2.834) |
| 32 (default) | 2,662,949 | 97.4% | 62.93 | 2.808 (2.803 - 2.813) |
| 64 | 2,737,061 | 94.8% | 61.23 | 2.824 (2.805 - 2.829) |
| 128 | 5,271,271 | 49.2% | 31.79 | 3.084 (3.079 - 3.092) |
| 200 | 8,201,023 | 31.6% | 20.43 | 3.216 (3.197 - 3.238) |

`MAC_ACTIVE` is 2,594,844 cycles at every one of those settings: the arithmetic
is identical and the whole difference is `STALL_MEM`, which goes from 23,410
cycles at `--lat 1` to 5,547,251 at `--lat 200`. The knee is the memory model's
64-beat in-flight window. Up to 64 cycles of latency the window covers the round
trip and the port stays saturated; past it the window sets the bandwidth at
`64 * WB / lat` bytes per cycle, which is 32.0 at `--lat 128` against the 31.79
measured and 20.48 at `--lat 200` against 20.43. Cycles the core spends waiting
are cheap to simulate, which is why the Mcycles/s
figure *rises* as the design gets slower in cycles.

### The `WB=128` fallback

The same RTL runs at `WB=128`; partial last tiles are zero-padded and drained
per `docs/ISA.md`, so the two widths compute the same values, and
`make bringup-sweep` checks that against the ISA simulator at both.

| Width | Counted cycles, decode token | `MAC_ACTIVE` | Read bytes per cycle | Mcycles/s | Wall clock |
|---|---|---|---|---|---|
| `WB=64` | 2,662,949 | 97.4% | 62.93 | 2.754 | 0.967 s |
| `WB=128` | 1,357,780 | 95.6% | 123.56 | 0.971 | 1.398 s |

```sh
uv run quettos compile qwen --layers 2 --wb 128 --out build/images/qwen2.5-0.5b-instruct-l2-wb128
make -C sim/verilator run WB=128 B_MAX=1 VL=4 VSRAM_WORDS=4096 \
  IMAGE=build/images/qwen2.5-0.5b-instruct-l2-wb128 ARGS="--max-new 1 --prompt-ids 1"
```

`WB=128` cuts the cycles by 1.961x and costs 2.84x more per cycle to simulate,
so a token takes 1.45x longer in wall clock than at `WB=64`. The fallback is
there because the RTL supports it, not because it simulates faster.

## The demo configuration

**Locked: `WB=64, B_MAX=1, VL=4, VSRAM_WORDS=4096, FIFO_BEATS=128, ACC_W=40`,
Verilator `--threads 1`.** This is the configuration `make synth` reports and
the one `make perf` and `make demo` run.

The rule the configuration was chosen by: *if Qwen 32+20 takes 8 minutes or less
at `WB=64`, that is the demo configuration and the synthesized configuration.*
Measured again on the complete core with every descriptor executing:
**128 s, two minutes, against a budget of eight** (126 - 129, n=3,
358,570,293 counted cycles in every run). Eight minutes over 358,570,293 cycles
is 0.747 Mcycles/s; the measured 2.807 is 3.8x above it, so the design would
have to become nearly four times more expensive per cycle to change the answer.
The command the project ships is shorter still: `make demo-qwen` takes the
checkpoint to the text in 108.5 s, of which 102.3 s is the RTL.

`WB=128` stays the same-RTL fallback at both the width and the value level, and
the tiny configuration `WB=16, B_MAX=2, VL=2, VSRAM_WORDS=2048` is what the
cocotb benches and the CI bring-up job run.

## The prefix a run restores

An agent turn repeats a long head: the system message and the tool descriptions
are the same on every call, and only the user's turn changes. `--kv-save FILE`
writes the KV region of the positions a run has consumed, byte for byte at its
image layout, behind the record of what those bytes were computed under
(`docs/MEMORY_MAP.md`, the prefix file). `--kv-load FILE` holds every term of
that record to the run restoring it -- the ISA version, the width, `MAX_CTX`,
the model, the image SHA-256, the region, and the token id at every position
the file covers -- and starts the token loop at the first position the file
does not cover. The region is 1,179,648 B for the two-layer Qwen image and
14,155,776 B for the whole model at `max_ctx = 2048`, and the record is
`176 + 4 * positions` bytes ahead of it.

### The tool call over a restored prefix (`make demo-toolcall`)

One prompt, `prompts/tool_call_weather.json`, run three ways on the whole
Qwen2.5-0.5B-Instruct at the demo configuration. It renders to 180 ids: the
first 162 are its system turn, which carries the tool description, and the
other 18 are the user's question and the assistant header. Prefill runs
positions 0 to 178 and the first decode step consumes id 179.

| Prefill pass | Positions | Tokens | Counted cycles | Cycles/token |
|---|---|---|---|---|
| the prefix, computed once (`--prefix-len 162 --max-new 0 --kv-save`) | 0 - 161 | 162 | 1,011,148,686 | 6,241,658 |
| the user's turn, after the restore (`--kv-load`) | 162 - 178 | 17 | 107,568,383 | 6,327,551 |
| the same prompt, no reuse | 0 - 178 | 179 | 1,118,717,069 | 6,249,816 |

The restore leaves 1,011,148,686 of those 1,118,717,069 prefill cycles unspent,
90.4% of them, and the two passes above it add up to the third exactly. They add
up position by position as well: every one of the 179 prefill positions and all
20 decode steps cost the same cycles in the restored run as in the run that
recomputed everything, which is what the equality of the totals is made of.

The generation after the restore is the run the summary reports: 20 decode
tokens, 169,946,516 cycles, 277,514,899 for the run at 91.6% MAC-active and
59.3 read bytes per cycle. Its 20 ids are
`<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call><|im_end|>`,
identical to the 20 the same prompt generates with no reuse at all (that run is
1,288,663,585 cycles, the two above it summed) and to the integer golden model's
recorded continuation in `models/qwen2.5-0.5b-instruct/expected_tokens.json`.
The saved file is 14,156,600 B: the 14,155,776 B KV region behind 824 B of
record. The three passes are 2,577,327,170 simulated cycles in all, and the
command runs them back to back in **940 s** (920 - 950, n=3) with the
checkpoint, the quantized model, the compiled image and the harness build
already on the machine, so what those seconds measure is the three passes on
`qcore_top`. Every run of that set produced the cycle counts above, to the
cycle.

```sh
make demo-toolcall                                       # the three passes and the summary
make demo-toolcall TOOLCALL_ARGS="--fresh"               # quantize and compile again first
```

### The prefix of `IMAGE`'s own prompt

`make regen-prefix` prefills every prompt token of `IMAGE` but the last,
generates nothing, and writes the file to `PREFIX_KV`: on the 36-id prompt both
images are compiled with, 1,179,964 B for the two-layer image and 14,156,092 B
for the whole model, each the KV region behind the 316 B record of the 35
positions it covers.

| Run | Counted cycles | Wall clock, median | Range, n |
|---|---|---|---|
| Qwen 2 layers, 35 prefill tokens (`make regen-prefix`) | 18,071,445 | 6.41 s | 6.37 - 6.56, n=9 |
| Qwen 24 layers, 35 prefill tokens | 216,438,845 | 77.7 s | 77.1 - 77.8, n=3 |

```sh
make regen-prefix                                        # the first row
make regen-prefix IMAGE=build/images/qwen2.5-0.5b-instruct
```

## The harness (`sim/verilator/`)

`sim/verilator/` is the host side of the machine: the clock loop, the QMEM
memory model, the CSR driver, the prefill/decode loop, the bring-up run, token
printing and the counter output. It drives `rtl/qcore_top.sv` over the top's
two ports, plus the two row backdoors `device.hpp` opens for a bring-up run:
the VSRAM words a descriptor wrote and the SREG bank `--sreg` loads.
`make harness` builds it, `make perf` runs it on `IMAGE` and writes
`build/perf/perf.json`, `make demo` builds an image from the checkpoint and
then runs the same loop on it, and `make bringup` runs the RTL-vs-simulator
comparison.

- **Memory model** (`mem_model.hpp`): the ports and the timing described under
  the protocol above. A read takes its bytes when the request is accepted and
  holds them in the pending beat, so a write accepted while the burst is in
  flight leaves the returned data alone: memory the core has not fenced against
  returns what was there at the request. It follows the same drive-and-sample
  discipline as `sim/cocotb/qc_qmem.py`, which snapshots a beat the same way,
  so the bus the C++ presents and the bus the cocotb tests present are the same
  bus.
- **CSR driver** (`csr.hpp`): `CsrBus` performs every host operation over the
  `csr_*` port with the one-cycle read latency of `docs/RTL.md` 3.3, one clock
  per operation; `CsrFile` is the same register table in C++. `make harness-csr`
  runs both against `rtl/qcore_csr.sv` and compares every read.
- **Token loop** (`main.cpp`): `TOK`, `POS`, `ROW_EN`, then `START`, exactly as
  `sw/quettos/isa_sim.py` `run_token` and `generate` sequence it. Each token
  prints its id, its cycle count and its MAC utilization, and the tokens stream
  out as UTF-8 from `tokens.bin` with multi-byte characters held until they are
  complete. `--step` runs one descriptor per `CTRL.STEP` and `--dump-ops`
  writes, after each one, the VSRAM range, the scale registers, the memory
  regions and the CSRs that its `dump_plan.json` entry names, plus
  `DESCRIPTORS`, `MACS`, `WT_BYTES` and the four event counters since the
  token's first descriptor. A memory region travels as its FNV-1a hash, and as
  its bytes when `--dump-bytes N` covers it. A plan entry naming something that
  is no register of the CSR window stops the run.
- **Bring-up run** (`bringup.hpp`, `--program FILE --program-addr N`): a
  descriptor blob of the caller's own, loaded into the copy-on-write image and
  run from `PC = N`. `--sreg B:I=WORD` loads a scale register before the run,
  `--dump-vsram B:S:C` records a VSRAM range after every descriptor and
  `--dump-mem A:S` records a QMEM region at the end of every pass;
  `--bringup-json` writes the per-descriptor records `sw/quettos/compare.py`
  checks against the simulator.
- **Multi-pass bring-up** (`--at TOK:POS`, repeatable): the blob runs once per
  pass, in order and on one machine, so a program that writes the KV cache
  reads back at the next position what the pass before it wrote. The
  `--dump-mem` regions are read at the end of every pass and every record
  carries the pass, the token and the position it belongs to.
- **Prefix save and restore** (`prefix.hpp`; `--kv-save FILE`, `--kv-load
  FILE`, `--prefix-len N`): the KV region copied byte for byte at its image
  layout, behind the record of what it belongs to; a restore holds every term of
  that record to the run taking it and starts the token loop at the first
  position the file does not cover, so a file from another image, width or
  prompt is refused before the first cycle rather than restored into a machine
  it does not belong to (`docs/MEMORY_MAP.md`, the prefix file). `--prefix-len
  N` prefills the first `N` prompt ids and stops, which is the pass that
  computes a prefix; `make regen-prefix` and `make demo-toolcall` write one, and
  `sw/quettos/prefix.py` reads the header of one back.
- **What `perf.json` records.** The counters and events, the memory model's own
  counts, the per-token records, the build widths, and a `run` object that says
  how the run was taken: `lat`, `bw_div`, `max_new`, `step`, `status`,
  `clock_cycles`, `wall_seconds` and `mcycles_per_s`, the `image` the hardware
  executed with the `image_bytes` the memory model mapped of it, the `prompt`
  ids with the file they were read from, and on a run that stopped early
  `stop_reason`, `fault`, `fault_op` and `fault_pc`. A number is therefore read
  back beside the image, the prompt and the settings that produced it, which is
  what `quettos demo-report` holds a finished run to.
- **End-of-sequence ids.** `layout.json` carries the compiled model's own
  `model.eos_ids` and the harness stops on them; `--eos` overrides the list.
- **What it asserts.** `BUSY` equals the sum of the six exclusive buckets;
  every token retires its program's whole descriptor count, and `WT_BYTES` and
  `MACS` then equal `layout.json`'s traffic model, including the POS-derived
  attention terms; `RD_BYTES` equals `RD_BEATS * WB`; the write counters equal
  the memory model's own counts and `RD_BEATS` trails it by at most the fetch
  unit's two outstanding bursts per token; and a run fails on any `SAT_*` or
  `ERR_*` event unless `--allow-sat` is given, the events summed per token
  because `START` clears them. Every run on this page reported
  `SAT_REQ=0 SAT_VPU=0 ERR_SHIFT=0 ERR_BOUNDS=0`.
- **What it refuses.** Before the first cycle: a `layout.json` compiled for
  another `WB` or another `ISA_VERSION`, an `image.bin` whose size is not the
  one that `layout.json` describes, a model whose `ISA_VERSION` register is not
  the harness table's, and `--threads` or `--trace` the linked model was not
  built for. From the command line: `--program` without `--program-addr`,
  `--sreg` / `--dump-vsram` / `--dump-mem` / `--at` outside a `--program` run,
  and a bank, a scale-register index or a VSRAM range outside the built widths.
  Each is an error naming both values, so a run is the one its files describe or
  no run at all. `--max-cycles N` ends a run that outlives its budget, with
  `stop_reason` in the record.
- **Build caching.** The object directory is `build/verilator/<top>-<hash>`,
  where the hash covers `rtl/*.sv`, `rtl/*.svh`, the `rtl/gen/*.hex` lookup-table
  images that `$readmemh` loads into the ROMs, the C++ and the configuration, so
  an unchanged tree relinks nothing and an include-only or table-only edit still
  rebuilds. `TRACE=1` adds `--trace` and enables `--trace FILE`, and
  `--trace-cycles N` closes the file after `N` cycles and lets the run carry on
  to its `HALT`, so the window is what sets the file size.

## Measured against the ISA simulator

`make bringup` and `make bringup-sweep` run `sw/quettos/compare.py`: the
four-descriptor bring-up program of `docs/ISA.md`, the eight-descriptor
directed vector program, the attention step of the compiled `decode.prog` and
the whole decoder layer, the last two at six positions on one machine, over
random tiny models, on `qcore_top` and on `sw/quettos/isa_sim.py`, comparing,
after every descriptor, every VSRAM element, every SREG word, every dumped
logit, every KV byte, `PC`, `STATUS`, the ARGMAX registers, the four event
counters and `DESCRIPTORS` / `MACS` / `WT_BYTES`.

| Run | Result | Wall clock, median, cold | Range, n |
|---|---|---|---|
| `make bringup` -- 2 shapes at WB=16, 2 tokens, four programs | **16/16 match** | 11.8 s | 11.8 - 12.1, n=5 |
| `make bringup-sweep` -- 5 shapes at WB=64 and WB=128, 2 tokens, four programs | **80/80 match** | 47.8 s | 47.5 - 47.9, n=3 |
| `uv run pytest -q sw/tests/test_bringup.py` (WB 16 and 64) | 44 passed | 37 s | 36 - 39, n=3 |

Cold means `build/verilator` was removed before each run, so each figure
includes the Verilator builds the run needs -- one at `WB=16`, two more for the
sweep.

## The speed probe (`sim/probe/`)

`sim/probe/` is a full-width skeleton of the datapath, built to price a cycle of
the design before the RTL was assembled: the `WB`-lane int8 x int16 MAC array
with double-buffered 40-bit accumulators, the weight and meta FIFOs, the
activation buffer over a true-dual-port 4096 x 256-bit vsram, one-output-per-cycle
requant with argmax, `VL` vector lanes and 64-bit counters, with every datapath
folded into a checksum so nothing is eliminated. `sim/probe/README.md` says what
it does not contain. Its workload is a synthetic Qwen-shaped decode token,
8,217,614 cycles at `WB=64` and 4,336,246 at `WB=128`; the runs below are four
of those tokens.

| Config | `--threads` | Mcycles/s, median | Range, n | Cold build, median of 3 |
|---|---|---|---|---|
| WB=64 | 1 | **5.945** | 5.447 - 5.969, n=12 | 1.263 s |
| WB=64 | 2 | 1.190 | 1.190 - 1.193, n=3 | 1.250 s |
| WB=64 | 4 | 0.648 | 0.647 - 0.649, n=3 | 1.256 s |
| WB=128 | 1 | **3.577** | 3.554 - 3.594, n=12 | 1.329 s |
| WB=128 | 2 | 1.366 | 1.326 - 1.368, n=3 | 1.333 s |
| WB=128 | 4 | 0.665 | 0.665 - 0.666, n=3 | 1.333 s |

```sh
make -C sim/probe build WB=64 T=1
make -C sim/probe run   WB=64 T=1 TOKENS=4
```

Checksums are identical across thread counts for a given width (`a17f63bf` at
`WB=64`, `2601a93c` at `WB=128`), and threads cost 5.0x at two and 9.2x at four
on the skeleton -- the same direction and about the same size as the 7.0x the
complete core measures.

The skeleton's rate and the complete core's are both measured, so the distance
between them is a number rather than an allowance: `qcore_top` runs at 2.754
Mcycles/s against the skeleton's 5.945, so a cycle of the complete design costs
2.16x a cycle of the skeleton. In cycles the skeleton is close: its synthetic
decode token is 8,217,614 cycles against the 8,321,833 the core spends on the
real 24-layer one, 1.3% apart. In wall clock it is not: 1.38 s a token against
2.97 s. The decision the skeleton was built to
settle -- eight minutes for Qwen 32 + 20 at `WB=64` -- holds on the core's own
measurement with 3.8x to spare.

## Measurement notes

- The README's results table quotes the `qcore_top` rows above and the Demo
  configuration section of `syn/reports/qcore_top.md`, which `make synth`
  regenerates from `build/synth/synth_top.log` and checks. Its hard-block counts
  are what `synth_xilinx` infers from the source on any build; its `SRL16E`,
  LUT, flop, cell and LC figures are how the Yosys named under **Machine and
  tools** packed the fabric on this machine, and `make synth` prints what
  another build packs instead.
- The wall clock includes an ideal C++ memory model; `--lat` and `--bw-div` are
  how far it bends, and the memory-latency table above is that sweep.
- CI runs on `ubuntu-latest` (4 vCPU, no Apple silicon) with the same flags, so
  expect several times slower per cycle there. At five times slower the
  SmolLM2 8 + 4 shape above is an **estimate**d 44 s of simulation, well inside
  a 15-minute budget, and the whole SmolLM2 demo the `demo-smollm2` job runs is
  an **estimate**d three minutes of simulation inside a 30-minute timeout.
- `--threads` results were taken on macOS without core pinning; the direction
  is unambiguous at every run count.
- FPGA throughput is a separate derivation from these cycle counts, at a clock
  a place-and-route timing result supplies and a stated memory bandwidth, and
  belongs with the synthesis numbers in `syn/reports/`.
