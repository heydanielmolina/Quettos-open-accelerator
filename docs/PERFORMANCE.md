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
  `START` and polling `STATUS`, 44 per token here. Both are given where they
  differ. Cycle counts are exact and repeat run to run; every count on this page
  was identical across every run of its command.
- **Rates and wall clock.** Every rate and every wall clock is the median of a
  set of consecutive runs of one command on an otherwise idle machine, quoted
  with the extremes of that set and the size of the set. The long runs carry
  smaller sets than the short ones, and every table names its `n`.
- **Sustained load.** Rates fall a few percent as the machine stays busy, so a
  single run reads high. Quoting a median with its range and its run count is
  what makes these figures reproducible: the widest spread in any set on this
  page is 9.4% of its median and the median spread is 1.7%.
- **Workload.** `make perf` and `make demo` run a compiled program on the RTL:
  every descriptor of `decode.prog` and `prefill.prog` at its real address,
  stride, meta and partial tile, all six vector opcodes on `qcore_vpu_top`, and
  the tokens the run generates printed as they leave the core. The traffic, the
  cycles, the counters and the values are the compiled program's.
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

`BUSY` equals `CYCLES` in every run here, so each share is a share of the whole
run. The vector unit's is the `STALL_VPU` column: 6.1% of the Qwen demo run and
14.0% of the SmolLM2 one, which has 30 narrower layers and so more vector passes
per weight byte. `STALL_SEQ`, the cost of fetching and decoding descriptors, is
between 0.02% and 0.28% across these runs.

### Whole compiled programs

| Run | Counted cycles | Clock cycles | Wall clock, median | Range, n |
|---|---|---|---|---|
| Qwen 2 layers, 36-token prompt + 1 decode (`make perf`) | 20,736,018 | 20,737,602 | **7.544 s** | 7.434 - 7.811, n=9 |
| Qwen 24 layers, 32 prompt + 20 decode | 358,570,293 | 358,572,537 | **126.015 s** | 125.126 - 126.271, n=3 |
| Qwen 24 layers, 4 prompt + 1 decode | 26,846,758 | 26,846,934 | 9.365 s | 9.299 - 9.418, n=9 |
| SmolLM2 30 layers, 32 prompt + 20 decode | 116,937,992 | 116,940,236 | 41.313 s | 41.215 - 41.347, n=3 |
| SmolLM2 30 layers, 8 prompt + 4 decode | 24,977,368 | 24,977,852 | 8.756 s | 8.721 - 8.811, n=9 |
| Qwen 2 layers, 35 prefill tokens (`make regen-prefix`) | 18,071,445 | 18,072,985 | 6.401 s | 6.300 - 6.500, n=9 |
| Qwen 24 layers, 35 prefill tokens | 216,438,845 | 216,440,385 | 76.176 s | 75.901 - 76.329, n=3 |

Three of those rows are reference points rather than one-off measurements: the
Qwen 32 + 20 demo the configuration was chosen by, and the Qwen 4 + 1 and
SmolLM2 8 + 4 shapes, which are what a run sized to a CI budget costs. The
nightly end-to-end jobs run each image's own prompt instead, plus one generated
token on Qwen and four on SmolLM2 (`.github/workflows/nightly.yml`).

```sh
make perf                                                     # the first row
make perf PERF_ARGS="--max-new 0"                             # prefill tokens only
IDS=$(tr '\n' ',' < build/images/qwen2.5-0.5b-instruct/prompt.tokens | sed 's/,$//' | cut -d, -f1-32)
make perf IMAGE=build/images/qwen2.5-0.5b-instruct \
  PERF_ARGS="--max-new 20 --prompt-ids $IDS --eos 999999999"
```

`--eos 999999999` is an id the vocabulary does not contain, so the loop runs its
full twenty tokens whatever the model generates, and the run is a fixed 32 + 20
shape.

### Simulation rate

| Configuration | Workload | Mcycles/s, median | Range, n |
|---|---|---|---|
| Demo, `--threads 1` | decode token, 2 layers | **2.754** | 2.679 - 2.787, n=9 |
| Demo, `--threads 1` | decode token, 24 layers | 2.803 | 2.707 - 2.829, n=9 |
| Demo, `--threads 1` | 36-token prompt + 1 decode | 2.749 | 2.655 - 2.789, n=9 |
| Demo, `--threads 1` | 32 prompt + 20 decode, 24 layers | 2.845 | 2.840 - 2.866, n=3 |
| Demo, `--threads 4` | decode token, 2 layers | 0.395 | 0.392 - 0.397, n=9 |
| `WB=128`, `--threads 1` | decode token, 2 layers | 0.971 | 0.892 - 0.976, n=9 |

Every set is consecutive runs of one command on an otherwise idle machine.

`--threads 4` is **7.0x slower than one thread**, and the cycle count is
identical to the bit (2,662,993 clock cycles either way), so the threaded build
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
the one `make perf` runs.

The rule the configuration was chosen by: *if Qwen 32+20 takes 8 minutes or less
at `WB=64`, that is the demo configuration and the synthesized configuration.*
Measured on the complete core, every descriptor executing: **126.0 s, two
minutes, against a budget of eight.** Eight minutes over 358,570,293 cycles is
0.747 Mcycles/s; the measured 2.845 is 3.8x above it, so the design would have
to become nearly four times more expensive per cycle to change the answer.

`WB=128` stays the same-RTL fallback at both the width and the value level, and
the tiny configuration `WB=16, B_MAX=2, VL=2, VSRAM_WORDS=2048` is what the
cocotb benches and the CI bring-up job run.

## The prefix a run restores

`make regen-prefix` prefills every prompt token of `IMAGE`, generates nothing,
and writes the KV region to `PREFIX_KV`; `--kv-load` puts it back before a later
run. The file is the region at its image layout, so it belongs to the model and
the port width that produced it: 1,179,648 B for the two-layer Qwen image and
14,155,776 B for the whole model, both at `max_ctx = 2048`.

| Run | Counted cycles | Wall clock, median | Range, n |
|---|---|---|---|
| Qwen 2 layers, 35 prefill tokens (`make regen-prefix`) | 18,071,445 | 6.401 s | 6.300 - 6.500, n=9 |
| Qwen 24 layers, 35 prefill tokens | 216,438,845 | 76.176 s | 75.901 - 76.329, n=3 |

```sh
make regen-prefix                                        # the first row
make regen-prefix IMAGE=build/images/qwen2.5-0.5b-instruct
```

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
  `--dump-mem A:S` records a QMEM region at the end of every pass;
  `--bringup-json` writes the per-descriptor records `sw/quettos/compare.py`
  checks against the simulator.
- **Multi-pass bring-up** (`--at TOK:POS`, repeatable): the blob runs once per
  pass, in order and on one machine, so a program that writes the KV cache
  reads back at the next position what the pass before it wrote. The
  `--dump-mem` regions are read at the end of every pass and every record
  carries the pass, the token and the position it belongs to.
- **KV save and restore** (`--kv-save FILE`, `--kv-load FILE`): the KV region
  copied byte for byte at its image layout, so a saved file belongs to the
  model and the port width that produced it. `make regen-prefix` writes one.
- **What `perf.json` records.** The counters and events, the memory model's own
  counts, the per-token records, the build widths, and a `run` object that says
  how the run was taken: `lat`, `bw_div`, `max_new`, `step`, `status`, and on a
  run that stopped early `stop_reason`, `fault`, `fault_op` and `fault_pc`. A
  number is therefore never separated from the conditions it was measured
  under.
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
four-descriptor bring-up program of `docs/ISA.md`, the eight-descriptor
directed vector program, the attention step of the compiled `decode.prog` and
the whole decoder layer, the last two at six positions on one machine, over
random tiny models, on `qcore_top` and on `sw/quettos/isa_sim.py`, comparing
every VSRAM element, SREG word, dumped logit, KV byte, CSR and PERF counter
after every descriptor.

| Run | Result | Wall clock, median, cold | Range, n |
|---|---|---|---|
| `make bringup` -- 2 shapes at WB=16, 2 tokens, four programs | **16/16 match** | 11.83 s | 11.82 - 12.08, n=5 |
| `make bringup-sweep` -- 5 shapes at WB=64 and WB=128, 2 tokens, four programs | **80/80 match** | 47.83 s | 47.54 - 47.93, n=3 |
| `uv run pytest -q sw/tests/test_bringup.py` (WB 16 and 64) | 38 passed | 32.38 s | 32.25 - 32.55, n=3 |

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
2.968 s. The decision the skeleton was built to
settle -- eight minutes for Qwen 32 + 20 at `WB=64` -- holds on the core's own
measurement with 3.8x to spare.

## Measurement notes

- The README's results table quotes the `qcore_top` rows above and the Demo
  configuration section of `syn/reports/qcore_top.md`, which `make synth`
  regenerates from `build/synth/synth_top.log` and checks.
- The wall clock includes an ideal C++ memory model; `--lat` and `--bw-div` are
  how far it bends, and the memory-latency table above is that sweep.
- CI runs on `ubuntu-latest` (4 vCPU, no Apple silicon) with the same flags, so
  expect several times slower per cycle there. At five times slower the
  SmolLM2 8 + 4 shape above is an **estimate**d 44 s of simulation, well inside
  a 15-minute budget.
- `--threads` results were taken on macOS without core pinning; the direction
  is unambiguous at every run count.
- FPGA throughput is a separate derivation from these cycle counts at a stated
  clock and memory bandwidth, and belongs with the synthesis numbers in
  `syn/reports/`.
