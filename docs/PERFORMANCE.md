# Performance

Every figure on this page comes from a run taken on the machine named below,
with the command that produced it beside it. Cycle counts are the core's own
PERF counters, read out of `qcore_top` through the CSR window; wall clocks and
rates are the harness clock loop over `std::chrono::steady_clock`. The one
exception is the 512-token prefix, which is arithmetic over measured per-token
costs and is labelled an estimate in the table it appears in.

## Measurement protocol

- **Machine and tools.** Apple M5 Pro (6 performance + 12 efficiency cores),
  64 GB, macOS 26.5.1 (Darwin 25.5.0), Apple clang 17.0.0, Verilator 5.048,
  Yosys 0.65. 2026-09-07. Threads were not pinned; `numactl` does not exist on
  macOS.
- **Cycles.** A cycle count is `PERF[CYCLES]`, which the core clears at `START`
  and snapshots at the `HALT` that retires the program, so it is the program's
  own cost. The harness `RESULT` line reports `clock_cycles` instead: the same
  count plus the cycles the host loop spends writing `TOK` / `POS`, pulsing
  `START` and polling `STATUS`, 44 per token here. Both are given where they
  differ. Cycle counts are exact and repeat run to run; every count on this page
  was identical across every run of its command.
- **Rates and wall clock.** Every rate and every wall clock is the median of a
  set of runs of one command, quoted with the extremes of that set and the size
  of the set. Sets of nine or more were taken as three separated batches, with
  other work between them, so that one batch cannot carry the median.
- **Sustained load.** Rates fall a few percent as the machine stays busy: over
  the three batches of each `qcore_top` set the batch medians drop by 0.5% to
  2.5% from first to last, and the widest single run in a set sits 9.3% below
  its median. Quoting a median with its range and its run count is what
  makes those numbers reproducible; a single run on an idle machine reads high.
- **Workload.** `make perf` runs a compiled program on the RTL under
  `--traffic`, which rewrites the `VROPE` and `VSOFTMAX` descriptors of the
  program regions to `NOP` in the copy-on-write image before the run. Every
  other descriptor executes at its real address, stride, meta and partial tile,
  the four vector passes `qcore_vpu_top` carries included, so the traffic, the
  cycles and the counters are the compiled program's. The values such a run
  computes are not, and it is not asked for them: the run prints its rewrite
  count and records it in `perf.json`.
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
`qcore_vpu_top` with its four vector lanes, its scalar unit and its three lookup
tables. Every measurement in this section is that top.

### Verilator build

Each row starts from an empty `build/verilator`, so every build is cold.

| Configuration | Cold build, median of 5 | Range |
|---|---|---|
| Demo, `WB=64 B_MAX=1 VL=4 VSRAM_WORDS=4096`, `--threads 1` | **1.520 s** | 1.516 - 1.563 |
| Tiny, `WB=16 B_MAX=2 VL=2 VSRAM_WORDS=2048` | 1.485 s | 1.453 - 1.491 |
| Demo, `--threads 4` | 1.553 s | 1.509 - 1.554 |
| `WB=128 B_MAX=1 VL=4 VSRAM_WORDS=4096` | 1.541 s | 1.512 - 1.552 |

The object directory is `build/verilator/<top>-<hash>` and the hash covers the
RTL, the lookup-table images, the C++ and the configuration, so an unchanged
tree relinks nothing: `make -C sim/verilator build` returns in 0.03 s.

```sh
make -C sim/verilator clean                                       # make the build cold
make -C sim/verilator build WB=64 B_MAX=1 VL=4 VSRAM_WORDS=4096   # prints build_seconds
```

### The cost of one token

Every count is `PERF[CYCLES]` for that token, at `POS = 0` so the rows compare.
Each model's per-token cost then grows linearly with `POS` -- 28 cycles per
position on the two-layer image, 336 on the 24-layer Qwen, 270 on SmolLM2,
which is the attention term of the compiled traffic model -- and every whole-run
total below is that line summed over the run's positions, to the cycle. The
36 + 1 run, for instance: `35 * 496,077 + 28 * (0 + ... + 34)` for the prefill
tokens plus `2,643,493 + 28 * 35` for the decode token is 20,023,828, which is
the counted total of the run. The same arithmetic reproduces the 32 + 20, the
4 + 1 and both SmolLM2 totals exactly.

| Model | Program | Cycles | `MAC_ACTIVE` | Read bytes per cycle |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct, 2 layers | decode | **2,643,493** | 98.2% | 63.39 |
| Qwen2.5-0.5B-Instruct, 2 layers | prefill | 496,077 | 94.3% | 60.93 |
| Qwen2.5-0.5B-Instruct, 24 layers | decode | **8,088,383** | 95.7% | 61.74 |
| Qwen2.5-0.5B-Instruct, 24 layers | prefill | 5,940,967 | 94.5% | 60.94 |
| SmolLM2-135M-Instruct, 30 layers | decode | 2,364,674 | 89.6% | 58.17 |
| SmolLM2-135M-Instruct, 30 layers | prefill | 1,915,242 | 87.5% | 56.85 |

The read-bytes-per-cycle column of a decode row is that single token's run; of a
prefill row it is the whole prefill run the token comes from, 35 tokens for Qwen
and 36 for SmolLM2, because the harness reports that ratio per run.

```sh
make perf IMAGE=build/images/qwen2.5-0.5b-instruct-l2 PERF_ARGS="--traffic --max-new 1 --prompt-ids 1"
make perf IMAGE=build/images/qwen2.5-0.5b-instruct    PERF_ARGS="--traffic --max-new 1 --prompt-ids 1"
make perf IMAGE=build/images/smollm2-135m-instruct    PERF_ARGS="--traffic --max-new 1 --prompt-ids 1"
```

The decode token streams the whole model once: 497,610,632 weight and meta bytes
for the 24-layer Qwen and 499,385,024 read bytes in all, in 7,802,891 returned
beats over 8,088,383 cycles. The weight port is therefore busy in 96.5% of the
token's cycles, which is what `MAC_ACTIVE` at 95.7% measures from the other
side.

A prefill token costs less because it skips the LM head: 360,260,488 weight and
meta bytes against the decode token's 497,610,632, and the difference of
137,350,144 is exactly the tied 151,936 x 896 head plus the eight meta bytes per
output channel. It sits a little lower on `MAC_ACTIVE` because the vector and
requant passes are a larger share of the shorter stream.

### Where the cycles go

`qcore_seq_dispatch` classifies every busy cycle into one of six exclusive
buckets, and the harness fails a run whose buckets do not sum to `BUSY`.

| Run | Cycles | `MAC_ACTIVE` | `STALL_MEM` | `STALL_VPU` | `STALL_KV` | `STALL_SEQ` | `STALL_DRAIN` |
|---|---|---|---|---|---|---|---|
| Qwen 2 layers, decode token | 2,643,493 | 98.16% | 0.96% | 0.64% | 0.020% | 0.021% | 0.195% |
| Qwen 2 layers, 36-token prompt + 1 decode | 20,023,828 | 94.80% | 1.25% | 2.84% | 0.093% | 0.098% | 0.912% |
| Qwen 24 layers, decode token | 8,088,383 | 95.69% | 1.06% | 2.36% | 0.076% | 0.075% | 0.743% |
| Qwen 24 layers, 32 prompt + 20 decode | 346,366,037 | 95.05% | 1.09% | 2.79% | 0.091% | 0.089% | 0.884% |
| SmolLM2 30 layers, 32 prompt + 20 decode | 107,010,232 | 88.49% | 2.19% | 6.09% | 0.492% | 0.287% | 2.459% |

`BUSY` equals `CYCLES` in every run here, so each share is a share of the whole
run. The vector unit's is the `STALL_VPU` column: 2.8% of the Qwen demo run and
6.1% of the SmolLM2 one, which has 30 narrower layers and so more vector passes
per weight byte. `STALL_SEQ`, the cost of fetching and decoding descriptors, is
between 0.02% and 0.29% across these runs.

### Whole compiled programs

| Run | Counted cycles | Clock cycles | Wall clock, median | Range, n |
|---|---|---|---|---|
| Qwen 2 layers, 36-token prompt + 1 decode (`make perf`) | 20,023,828 | 20,025,412 | **6.840 s** | 6.771 - 7.276, n=12 |
| Qwen 24 layers, 32 prompt + 20 decode | 346,366,037 | 346,368,281 | **118.9 s** | 117.3 - 119.5, n=9 |
| Qwen 24 layers, 4 prompt + 1 decode | 25,913,300 | 25,913,476 | 8.756 s | 8.708 - 8.811, n=9 |
| SmolLM2 30 layers, 32 prompt + 20 decode | 107,010,232 | 107,012,476 | 37.18 s | 36.85 - 37.42, n=9 |
| SmolLM2 30 layers, 8 prompt + 4 decode | 22,880,240 | 22,880,724 | 7.821 s | 7.791 - 8.621, n=9 |
| Qwen 2 layers, 35 prefill tokens | 17,379,355 | 17,380,895 | 6.000 s | 5.963 - 6.419, n=9 |
| Qwen 24 layers, 35 prefill tokens | 208,133,765 | 208,135,305 | 70.80 s | 69.18 - 71.60, n=9 |

The named runs are in that table: the Qwen 32 + 20 demo, the Qwen 4 + 1 the
nightly job runs, and the SmolLM2 8 + 4 the PR matrix runs.

```sh
make perf                                                     # the first row
make perf PERF_ARGS="--traffic --max-new 0"                   # prefill tokens only
IDS=$(tr '\n' ',' < build/images/qwen2.5-0.5b-instruct/prompt.tokens | sed 's/,$//' | cut -d, -f1-32)
make perf IMAGE=build/images/qwen2.5-0.5b-instruct \
  PERF_ARGS="--traffic --max-new 20 --prompt-ids $IDS --eos 999999999"
```

`--eos 999999999` is an id the vocabulary does not contain, so the loop runs its
full twenty tokens: under `--traffic` the argmax a token produces is not the
model's, and the run is a traffic and cycle measurement of a fixed 32 + 20
shape.

### Simulation rate

| Configuration | Workload | Mcycles/s, median | Range, n |
|---|---|---|---|
| Demo, `--threads 1` | decode token, 2 layers | **3.022** | 2.742 - 3.053, n=15 |
| Demo, `--threads 1` | decode token, 24 layers | 3.032 | 2.793 - 3.085, n=9 |
| Demo, `--threads 1` | 36-token prompt + 1 decode | 2.928 | 2.752 - 2.958, n=12 |
| Demo, `--threads 1` | 32 prompt + 20 decode, 24 layers | 2.913 | 2.899 - 2.953, n=9 |
| Demo, `--threads 4` | decode token, 2 layers | 0.360 | 0.359 - 0.361, n=9 |
| `WB=128`, `--threads 1` | decode token, 2 layers | 1.014 | 1.003 - 1.021, n=9 |

Each set was taken as three separated batches: five, five and five runs for the
first row, four, four and four for the 36+1 row, three, three and three for the
rest.

`--threads 4` is **8.4x slower than one thread**, and the cycle count is
identical to the bit (2,643,537 clock cycles either way), so the threaded build
is deterministic and functionally the same run. The design is far too small for
Verilator's MTask partitioning; per-`eval` thread synchronization dominates.
**`--threads 1` is the default.**

### Memory latency

`--lat` sets the QMEM read latency and the write-ack latency. The decode token
of the two-layer image, five runs per setting.

| `--lat` | Counted cycles | `MAC_ACTIVE` | Read bytes per cycle | Mcycles/s, median (range) |
|---|---|---|---|---|
| 1 | 2,641,298 | 98.2% | 63.44 | 3.001 (2.899 - 3.033) |
| 8 | 2,641,789 | 98.2% | 63.43 | 2.996 (2.988 - 3.025) |
| 32 (default) | 2,643,493 | 98.2% | 63.39 | 2.996 (2.807 - 3.033) |
| 64 | 2,715,749 | 95.5% | 61.70 | 3.006 (2.977 - 3.026) |
| 128 | 5,246,247 | 49.5% | 31.94 | 3.271 (3.256 - 3.299) |
| 200 | 8,171,823 | 31.8% | 20.51 | 3.400 (3.372 - 3.412) |

`MAC_ACTIVE` is 2,594,844 cycles at every one of those settings: the arithmetic
is identical and the whole difference is `STALL_MEM`, which goes from 23,410
cycles at `--lat 1` to 5,547,251 at `--lat 200`. The knee is the memory model's
64-beat in-flight window. Up to 64 cycles of latency the window covers the round
trip and the port stays saturated; past it the window sets the bandwidth at
`64 * WB / lat` bytes per cycle, which is 32.0 at `--lat 128` against the 31.94
measured and 20.48 at `--lat 200` against 20.51. Cycles the core spends waiting
are cheap to simulate, which is why the Mcycles/s figure *rises* as the design
gets slower in cycles.

### The `WB=128` fallback

The same RTL runs at `WB=128`; partial last tiles are zero-padded and drained
per `docs/ISA.md`, so the two widths compute the same values, and
`make bringup-sweep` checks that against the ISA simulator at both.

| Width | Counted cycles, decode token | `MAC_ACTIVE` | Read bytes per cycle | Mcycles/s | Wall clock |
|---|---|---|---|---|---|
| `WB=64` | 2,643,493 | 98.2% | 63.39 | 3.022 | 0.875 s |
| `WB=128` | 1,338,326 | 97.0% | 125.35 | 1.014 | 1.320 s |

```sh
uv run quettos compile qwen --layers 2 --wb 128 --out build/images/qwen2.5-0.5b-instruct-l2-wb128
make -C sim/verilator run WB=128 B_MAX=1 VL=4 VSRAM_WORDS=4096 \
  IMAGE=build/images/qwen2.5-0.5b-instruct-l2-wb128 ARGS="--traffic --max-new 1 --prompt-ids 1"
```

`WB=128` cuts the cycles by 1.975x and costs 2.98x more per cycle to simulate,
so a token takes 1.51x longer in wall clock than at `WB=64`. The
fallback is there because the RTL supports it, not because it simulates faster.

## The demo configuration

**Locked: `WB=64, B_MAX=1, VL=4, VSRAM_WORDS=4096, FIFO_BEATS=128, ACC_W=40`,
Verilator `--threads 1`.** This is the configuration `make synth` reports and
the one `make perf` runs.

The rule the configuration was chosen by: *if Qwen 32+20 takes 8 minutes or less
at `WB=64`, that is `make demo` and the demo configuration is the synthesized
configuration.* Measured on the complete core: **118.9 s, two minutes, against
a budget of eight.** Eight minutes over 346,366,037 cycles is 0.722 Mcycles/s;
the measured 2.913 is 4.0x above it, so the design would have to become four
times more expensive per cycle to change the answer.

`WB=128` stays the same-RTL fallback at both the width and the value level, and
the tiny configuration `WB=16, B_MAX=2, VL=2, VSRAM_WORDS=2048` is what the
cocotb benches and the CI bring-up job run.

## The 512-token demo prefix

`make regen-prefix` prefills the 512-token demo prefix, and its cost is the one
figure on this page that is arithmetic rather than a run: the measured Qwen
prefill token, 5,940,967 cycles at `POS = 0` plus 336 per position, summed over
positions 0 to 511.

| Run | Cycles | Wall clock |
|---|---|---|
| 512-token prefix (`make regen-prefix`) | 3,085,729,280 (**estimate**) | 17.7 min (**estimate**) |

The wall clock divides those cycles by the 2.913 Mcycles/s measured over the
32 + 20 run. The same per-position arithmetic over the 32 + 20 shape gives
346,366,037 cycles, which is that run's counted total to the cycle, so the line
it extends is exact rather than fitted.

## The harness (`sim/verilator/`)

`sim/verilator/` is the host side of the machine: the clock loop, the QMEM
memory model, the CSR driver, the prefill/decode loop, the bring-up run, token
printing and the counter output. It drives `rtl/qcore_top.sv` over the top's two
ports and nothing else. `make harness` builds it, `make perf` runs it on `IMAGE`
and writes `build/perf/perf.json`, and `make bringup` runs the RTL-vs-simulator
comparison.

- **Memory model** (`mem_model.hpp`): the ports and the timing described under
  the protocol above. A read takes its bytes when the request is accepted and
  holds them in the pending beat, so a write accepted while the burst is in
  flight cannot change what the burst returns and a missing fence shows up as a
  value difference. It follows the same drive-and-sample discipline as
  `sim/cocotb/qc_qmem.py`, which snapshots a beat the same way, so the bus the
  C++ presents and the bus the cocotb tests present are the same bus.
- **CSR driver** (`csr.hpp`): `CsrBus` performs every host operation over the
  `csr_*` port with the one-cycle read latency of `docs/RTL.md` 3.3, one clock
  per operation; `CsrFile` is the same register table in C++. `make harness-csr`
  runs both against `rtl/qcore_csr.sv` and compares every read.
- **Token loop** (`main.cpp`): `TOK`, `POS`, `ROW_EN`, then `START`, exactly as
  `sw/quettos/isa_sim.py` `run_token` and `generate` sequence it. Each token
  prints its id, its cycle count and its MAC utilization, and the tokens stream
  out as UTF-8 from `tokens.bin` with multi-byte characters held until they are
  complete. `--step` runs one descriptor per `CTRL.STEP` and `--dump-ops`
  writes the VSRAM range, scale registers, memory regions and CSRs that each
  descriptor's `dump_plan.json` entry names.
- **Bring-up run** (`bringup.hpp`, `--program FILE --program-addr N`): a
  descriptor blob of the caller's own, loaded into the copy-on-write image and
  run from `PC = N`. `--sreg B:I=WORD` loads a scale register before the run,
  `--dump-vsram B:S:C` records a VSRAM range after every descriptor and
  `--dump-mem A:S` records a DUMP region after it; `--bringup-json` writes the
  per-descriptor records `sw/quettos/compare.py` checks against the simulator.
- **Traffic measurement** (`--traffic`): `VROPE` and `VSOFTMAX` name the two
  passes `qcore_vpu_top` does not carry, so `qcore_top` stops a program on one
  with `STATUS.ERR` and `FAULT = OPCODE`. `--traffic` rewrites those
  descriptors to `NOP` in the copy-on-write image first -- 60 of them in the
  two-layer image, 720 in the 24-layer one -- so every other descriptor of the
  compiled program runs at its real address, stride, meta and partial tile.
- **What `perf.json` records.** The counters and events, the memory model's own
  counts, the per-token records, the build widths, and a `run` object that says
  how the run was taken: `traffic` and `traffic_descriptors_rewritten` for the
  `--traffic` rewrite, `status`, and on a run that stopped early `stop_reason`,
  `fault`, `fault_op` and `fault_pc`. A number is therefore never separated from
  the conditions it was measured under.
- **End-of-sequence ids.** `layout.json` carries the compiled model's own
  `model.eos_ids` and the harness stops on them; `--eos` overrides the list.
- **What it asserts.** `BUSY` equals the sum of the six exclusive buckets;
  `WT_BYTES` and `MACS` equal `layout.json`'s traffic model for every token that
  runs a program to its `HALT`, including the POS-derived attention terms;
  `RD_BYTES` equals `RD_BEATS * WB`; the write counters equal the memory model's
  own counts and `RD_BEATS` trails it by at most the fetch unit's two
  outstanding bursts per token; and a run fails on any `SAT_*` or `ERR_*` event
  unless `--allow-sat` is given. Every run on this page reported
  `SAT_REQ=0 SAT_VPU=0 ERR_SHIFT=0 ERR_BOUNDS=0`.
- **Build caching.** The object directory is `build/verilator/<top>-<hash>`,
  where the hash covers `rtl/*.sv`, `rtl/*.svh`, the `rtl/gen/*.hex` lookup-table
  images that `$readmemh` loads into the ROMs, the C++ and the configuration, so
  an unchanged tree relinks nothing and an include-only or table-only edit still
  rebuilds. `TRACE=1` adds `--trace` and enables `--trace FILE`.

## Measured against the ISA simulator

`make bringup` and `make bringup-sweep` run `sw/quettos/compare.py`: the
four-descriptor bring-up program and the eight-descriptor directed vector
program of `docs/ISA.md`, over random tiny models, on `qcore_top` and on
`sw/quettos/isa_sim.py`, comparing every VSRAM element, SREG word, dumped logit,
CSR and PERF counter after every descriptor.

| Run | Result | Wall clock, median of 5, cold | Range |
|---|---|---|---|
| `make bringup` -- 2 shapes at WB=16, 2 tokens, both programs | **8/8 match** | 4.20 s | 4.19 - 4.34 |
| `make bringup-sweep` -- 5 shapes at WB=64 and WB=128, 2 tokens, both programs | **40/40 match** | 14.97 s | 14.91 - 15.05 |
| `uv run pytest -q sw/tests/test_bringup.py` (WB 16, 64 and 128) | 22 passed | 13.44 s | 13.39 - 13.50 |

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
on the skeleton -- the same direction and about the same size as the 8.4x the
complete core measures.

The skeleton's rate and the complete core's are both measured, so the distance
between them is a number rather than an allowance: `qcore_top` runs at 3.022
Mcycles/s against the skeleton's 5.945, so a cycle of the complete design costs
1.97x a cycle of the skeleton, and a skeleton rate takes a factor of 0.51 to
reach the design's. In cycles the skeleton is close: its synthetic decode token
is 8,217,614 cycles against the 8,088,383 the core spends on the real 24-layer
one, 1.6% apart. In wall clock it is not: 1.38 s a token against 2.667 s, and a
rate that reads about twice the design's. The decision the skeleton was built to
settle -- eight minutes for Qwen 32 + 20 at `WB=64` -- holds on the core's own
measurement with 4.0x to spare.

## Measurement notes

- The README's results table quotes the `qcore_top` rows above and the Demo
  configuration section of `syn/reports/qcore_top.md`, which `make synth`
  regenerates from `build/synth/synth_top.log` and checks.
- The wall clock includes an ideal C++ memory model; `--lat` and `--bw-div` are
  how far it bends, and the memory-latency table above is that sweep.
- CI runs on `ubuntu-latest` (4 vCPU, no Apple silicon) with the same flags, so
  expect several times slower per cycle there. Even at five times slower the
  SmolLM2 8 + 4 job is 39 s of simulation, well inside the 15-minute PR budget.
- `--threads` results were taken on macOS without core pinning; the direction
  is unambiguous at every run count.
- FPGA throughput is a separate derivation from these cycle counts at a stated
  clock and memory bandwidth, and belongs with the synthesis numbers in
  `syn/reports/`.
