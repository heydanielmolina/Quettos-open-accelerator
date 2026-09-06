# Performance: Verilator speed probe and the demo decision

The demo configuration was locked from `sim/probe/`, a full-width skeleton with
the final design's datapath widths and memories, derated by 0.65 as a safety
margin: the first six sections below. "The harness" holds the measurements
`make perf` takes on the RTL itself through `sim/verilator/`, and the README's
results table is filled from those.

## Methodology

- **Model:** `sim/probe/probe_top` at `B_MAX=1, VL=4, VSRAM_WORDS=4096,
  FIFO_BEATS=128, ACC_W=40`, at `WB=64` and `WB=128`. It contains, at full
  width, the structures that dominate the final design's simulation cost: the
  `WB`-lane int8 x int16 MAC array with double-buffered 40-bit accumulators,
  weight FIFO (128 x WB*8 bits), meta side-FIFO, activation word buffer over a
  true-dual-port 4096 x 256-bit vsram, one-output-per-cycle requant with argmax,
  `VL` vector lanes with 32x32 and 32x16 signed multipliers plus an RMSNorm-like
  read-modify-write pass, and 64-bit perf counters. Every datapath feeds a
  checksum the harness prints, so nothing is dead-code-eliminated; the
  generated C++ was checked to contain all 64 lanes and 128 accumulators.
  See `sim/probe/README.md` for what it does *not* contain.
- **Workload:** a synthetic Qwen2.5-0.5B-shaped decode token: per layer
  `GEMV N=1152 K=896`, `N=896 K=896`, `N=9728 K=896`, `N=896 K=4864` (x24
  layers) with a 900-word VPU pass between GEMVs, then the LM head
  `N=151936 K=896` in ARGMAX mode. 7,775,152 weight+meta beats per token at
  WB=64 (plan: 7,776,524 including 1,372 gamma beats the probe does not
  stream). Measured token cost: **8,217,614 cycles at WB=64** (MAC-active
  7,718,144 = 93.9%, weight-port busy 94.6%; the remaining 5.4% is the VPU
  stub, which is deliberately on the heavy side of the design's ~2%),
  **4,336,246 cycles at WB=128**.
- **Memory model:** C++ side, 64 MB xorshift buffer, fixed latency 32 cycles,
  one beat per cycle, 64-beat in-flight window, structured meta beats
  (`Sw_m` in [2^15, 2^16), `Sw_e` in [-8,-1], 20-bit bias) so the requant runs
  its non-saturating path (`sat=0`, `err=0` in all reported runs).
- **Verilator flags:** `--cc --exe --build -j 0 -O3 --x-assign fast
  --x-initial fast --no-assert --no-timing -Wall -Wno-fatal --output-split 20000
  -CFLAGS -O2`, plus `--threads N` and `-GWB=N`. Zero Verilator warnings at
  build; the probe RTL is clean under `verilator --lint-only -Wall -Wpedantic`,
  `yosys read_verilog -sv; hierarchy -check; proc; opt; check -assert` and
  `iverilog -g2012 -Wall`.
- **Timing:** `std::chrono::steady_clock` around the token loop (after reset),
  two `eval()` calls per cycle (rising and falling edge), wall-clock including
  the C++ memory model. Runs of 4 synthetic tokens (32.9M cycles at WB=64,
  17.3M at WB=128); every rate is the median of repeated runs of one command,
  quoted with its range and its run count: 12 runs of each single-thread
  configuration, 3 of each threaded one, 5 of each latency setting, and 3 builds
  per configuration.
- **Machine:** Apple M5 Pro (6 performance + 12 efficiency cores), 64 GB,
  macOS (Darwin 25.5), Apple clang 17.0.0, Verilator 5.048, 2026-09-06.
  No `numactl` on macOS; threads were not pinned.

## Raw measurements

Each row is `make -C sim/probe build WB=<w> T=<t>` followed by
`make -C sim/probe run WB=<w> T=<t> TOKENS=4`, the rate the run prints on its
`RESULT` line, and the build time the build prints.

| Config | `--threads` | Mcycles/s (median) | Mcycles/s (range, n) | Verilator build, median of 3 |
|---|---|---|---|---|
| WB=64  | 1 | **5.81** | 5.75 - 5.83 (n=12) | 1.33 s |
| WB=64  | 2 | 1.19 | 1.16 - 1.19 (n=3) | 1.33 s |
| WB=64  | 4 | 0.65 | 0.64 - 0.65 (n=3) | 1.33 s |
| WB=128 | 1 | **3.56** | 3.49 - 3.58 (n=12) | 1.36 s |
| WB=128 | 2 | 1.33 | 1.31 - 1.36 (n=3) | 1.40 s |
| WB=128 | 4 | 0.66 | 0.66 - 0.67 (n=3) | 1.39 s |

The spread inside one batch is under 1.5%. Rates fall by a few percent as the
machine stays under load, so the four-token rows above are sustained-load
medians taken across twelve consecutive runs, and a first run on an idle
machine sits about 4% higher. Every rate on this page is a median with its
range and run count for that reason.

Facts worth stating plainly:

- **`--threads` is slower than one thread on this design**, by 2.7-5.0x at 2
  threads and 5.5-9.4x at 4 threads. The skeleton is far too small for
  Verilator's MTask partitioning; per-eval thread synchronization dominates.
  This matches the rule ("default 1 unless >= 1.3x measured"): **`--threads 1`
  is the default.** `qcore_top` measures the same way (9.3x slower at 4
  threads).
- Checksums are identical across thread counts for a given WB
  (`a17f63bf` at WB=64, `2601a93c` at WB=128 for 4 tokens), i.e. the threaded
  builds are deterministic and functionally identical.
- Build times are 1.33-1.40 s because the skeleton generates only ~0.9 MB of
  C++ in 6 files; `qcore_top` (`--output-split 20000`) builds cold in 1.61 s,
  well inside the budget.
- Per-cycle cost scales sub-linearly with WB: WB=128 costs 1.63x more per
  cycle for 2x the work per cycle, so per *token* WB=128 is only 1.16x faster
  in wall-clock than WB=64 (4.34M cycles / 3.56 Mc/s = 1.22 s vs 8.22M / 5.81 =
  1.42 s). The fallback buys little.
- Latency sensitivity (WB=64, 1 thread, 2 tokens, n=5 each): `--lat 1` 6.05
  Mc/s over 16,429,214 cycles, the default `--lat 32` 6.05 Mc/s over 16,435,228,
  `--lat 200` 6.76 Mc/s but 3.0x the cycles (49,494,060), because the C++
  model's 64-beat in-flight window throttles bandwidth above LAT=64 (expected;
  idle cycles are cheaper to simulate).

## Derated numbers (x0.65 safety margin)

| Config | Raw Mcycles/s | Derated Mcycles/s |
|---|---|---|
| WB=64,  1 thread | 5.81 | **3.78** |
| WB=128, 1 thread | 3.56 | **2.31** |

The 0.65 derating is an allowance for everything the skeleton lacks
(sequencer/dispatch, memory arbiter, kv_writer, CSR, 16 perf counters, six VPU
op FSMs, LUT ROMs and their interpolators, real per-token meta strides). These
are predominantly control logic and narrow registers, cheap per cycle next to
the 64-lane datapath and the 512-bit FIFO copies the skeleton already pays for.
The measurement on `qcore_top` below lands at 3.70 Mcycles/s, 2% under the 3.78
this table projects.

## Projected wall-clock (derated; raw in parentheses)

Cycle counts here are analytical **estimates** for the complete model; the
probe's own token was 8.22M cycles at WB=64, 2% above the design estimate of
8.06M. Measured counters for the two-layer image are in "Measured on
`qcore_top`" below.

| Run | Cycles | WB=64 @ 3.78 Mc/s | WB=128 @ 2.31 Mc/s |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct 32+20 demo | 350M (WB=64) / 184M (WB=128) | **93 s = 1.5 min** (60 s) | 80 s = 1.3 min (52 s) |
| Qwen decode, one token | 8.06M | 2.1 s (1.4 s) | - |
| Qwen prefill, one prompt token | 5.91M | 1.6 s (1.0 s) | - |
| SmolLM2-135M-Instruct 32+20 | 107M | 27 s (18 s) | - |
| SmolLM2 CI 8+4 (this machine) | 24.5M | 6.3 s (4.1 s) | - |
| Qwen 4+1 nightly | ~31.7M | 8.1 s (5.3 s) | - |
| 512-token prefix regen (`make regen-prefix`) | ~3.03G | 12.9 min (8.4 min) | - |

CI runs on `ubuntu-latest` (4 vCPU, `-CFLAGS -O1`, no Apple silicon); expect
several times slower per cycle there. Even at 5x slower the SmolLM2 8+4 e2e
job is ~30 s of simulation, well inside the 15-minute PR budget.

## Decision rule applied

Decision rule (re-checked on the complete design): *if Qwen 32+20 <= 8 min on WB=64, that is
`make demo` (demo config == synthesized config); else WB=128; else SmolLM2 live.*

- Qwen 32+20 on WB=64, derated: **93 s (1.5 min) <= 8 min. PASS.**
- Margin: 8 min corresponds to 0.73 Mcycles/s; the derated measurement is 5.2x
  above that, the raw measurement 8.0x. The real design would have to be more
  than 5x slower per cycle than the derated skeleton to flip the decision.

**Locked demo configuration: `WB=64, B_MAX=1, VL=4, VSRAM_WORDS=4096,
FIFO_BEATS=128, ACC_W=40`, Verilator `--threads 1`.** This is also the
synthesized configuration. WB=128 stays a same-RTL fallback: partial last tiles
are zero-padded and drained per `docs/ISA.md`, so the two widths compute the
same values, and `make bringup-sweep` checks that at both widths against the
ISA simulator. The tiny CI config is unchanged.

## The harness (`sim/verilator/`)

`sim/verilator/` is the host side of the machine: the clock loop, the QMEM
memory model, the CSR driver, the prefill/decode loop, the bring-up run, token
printing and the counter output. It drives `rtl/qcore_top.sv` over the top's two
ports and nothing else. `make harness` builds it, `make perf` runs it on `IMAGE`
and writes `build/perf/perf.json`, and `make bringup` runs the RTL-vs-simulator
comparison.

- **Memory model** (`mem_model.hpp`): `image.bin` mapped copy-on-write behind
  the QMEM ports of `docs/RTL.md` 2.1 -- fixed read latency (`--lat`, default
  32), one returned beat every `--bw-div` cycles, in-order returns across tags,
  a 64-beat in-flight window that throttles `rd_req_ready`, byte-strobed writes
  and their acks after the same latency. A read takes its bytes when the request
  is accepted and holds them in the pending beat, so a write accepted while the
  burst is in flight cannot change what the burst returns and a missing fence
  shows up as a value difference. It follows the same drive-and-sample
  discipline as `sim/cocotb/qc_qmem.py`, which snapshots a beat the same way, so
  the bus the C++ presents and the bus the cocotb tests present are the same
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
  writes the VSRAM range, scale registers, memory regions and CSRs that each
  descriptor's `dump_plan.json` entry names.
- **Bring-up run** (`bringup.hpp`, `--program FILE --program-addr N`): a
  descriptor blob of the caller's own, loaded into the copy-on-write image and
  run from `PC = N`. `--sreg B:I=WORD` loads a scale register before the run,
  `--dump-vsram B:S:C` records a VSRAM range after every descriptor and
  `--dump-mem A:S` records a DUMP region after it; `--bringup-json` writes the
  per-descriptor records `sw/quettos/compare.py` checks against the simulator.
- **Traffic measurement** (`--traffic`): the six vector opcodes belong to
  `qcore_vpu_top`, which this top does not carry, so `qcore_top` stops a program
  on one with `STATUS.ERR` and `FAULT = OPCODE`. `--traffic` rewrites those
  descriptors to `NOP` in the copy-on-write image first, so the `GEMV`, `EMBED`
  and `KVWRITE` descriptors of the compiled program run at their real addresses,
  strides, meta and partial tiles. Traffic and cycles are the program's; the
  values it computes are not, and the run says so.
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
  unless `--allow-sat` is given.
- **Build caching.** The object directory is `build/verilator/<top>-<hash>`,
  where the hash covers `rtl/*.sv` and `rtl/*.svh` (the modules and the
  generated include they read), the C++ and the configuration, so an unchanged
  tree relinks nothing and an include-only edit still rebuilds. Flags are the
  probe's:
  `--cc --exe --build -j 0 -O3 --x-assign fast --x-initial fast --no-assert
  --no-timing -Wall --output-split 20000 -CFLAGS -O2`, plus `--threads` and the
  `-G` width parameters; `TRACE=1` adds `--trace` and enables `--trace FILE`.

### Measured on `qcore_top`

`rtl/qcore_top.sv` at the demo configuration: `qcore_csr`, `qcore_seq_fetch`,
`qcore_seq_dispatch`, `qcore_perf`, `qcore_mem_arb`, `qcore_stream_ctrl`, the
`qcore_row` and `qcore_vsram` pair, `qcore_requant` and `qcore_kv_writer`.
`qcore_vpu_top` is the one module it does not contain, and the six vector
opcodes are ~2% of the design's cycles (**estimate**, from the probe's workload
split). The workload is the two-layer Qwen2.5-0.5B image
(`build/images/qwen2.5-0.5b-instruct-l2`, WB=64) under `--traffic`: one decode
token on its own, and the compiled 36-token prompt followed by one decode token.
Apple M5 Pro, Verilator 5.048, Apple clang 17, 2026-09-06.

The single-token rows come from
`make perf PERF_ARGS="--traffic --max-new 1 --prompt-ids 1"`; the whole-run rows
from `make perf`.

| Measurement | Value |
|---|---|
| Verilator build, cold (median of 5) | 1.61 s (WB=64 demo, 1.60 - 1.65), 1.50 s (WB=16 tiny, 1.50 - 1.55), 1.52 s (`--threads 4`, 1.50 - 1.53) |
| Verilator build, unchanged tree | nothing to relink |
| Simulation rate, `--threads 1` | **3.70 Mcycles/s median** (3.59 - 3.76, n=33 in three batches) |
| Simulation rate, `--threads 4` | 0.408 Mcycles/s median (0.401 - 0.409, n=9), cycle for cycle the same run |
| Decode token (2 layers, POS=0) | 2,626,475 cycles, MAC_ACTIVE 98.8%, 63.8 read bytes/cycle |
| Prefill token (2 layers, POS=34) | 481,215 cycles, MAC_ACTIVE 97.4% |
| 36-token prompt + 1 decode token (`make perf`) | 19,453,320 counted cycles (19,454,904 clock cycles) in 5.48 s median (5.42 - 5.54, n=11), 3.55 Mcycles/s |
| Sequencing overhead of that run | `STALL_SEQ` 18,297 cycles, 0.09% |

A cycle count in that table is the `PERF` counter the run reports. A rate is the
harness clock loop over its own wall clock, which is the `clock_cycles` and
`seconds` of the `RESULT` line: 2,626,519 clock cycles around the decode token
and 19,454,904 around the whole run, the counter plus the cycles the loop spends
setting up and polling.

The probe measures 5.81 Mcycles/s on the skeleton and this page derates it by
0.65 to **3.78** as an allowance for the sequencer, the arbiter, the KV writer,
the CSR file and the sixteen counters. The complete top, with all of those
present, measures **3.70**, 2% under the projection: the complete design costs
0.64 of the skeleton's rate per cycle against the 0.65 the margin assumed. The
derating was the right size.

`--threads 4` is 9.3x slower than one thread here, the same direction and about
the same factor the probe measured, and the cycle count is identical, so
`--threads 1` stays the default.

### Measured against the ISA simulator

`make bringup` and `make bringup-sweep` run `sw/quettos/compare.py`: the
bring-up program of `docs/ISA.md` in the form this top runs it -- the `EMBED` of
the model's own `decode.prog` writing the LM head's activation slot, the tied LM
head as a `GEMV` in `ARGMAX_DUMP` mode, `HALT` -- over random tiny models, on
`qcore_top` and on `sw/quettos/isa_sim.py`, comparing every VSRAM element, SREG
word, dumped logit, CSR and PERF counter after every descriptor.

`qcore_vpu_top` owns the `VQUANT` that pairs an activation with its scale, so
the harness loads that one scale register with `--sreg` and raises the `EMBED`
output shift by the matching number of bits, which is what keeps the gathered
row inside the int16 window a GEMV activation is read through. Every other input
is the model's own image, both models are given the same scale, and the program
is self-checking: a v1 model ties the embedding and the LM head, so
`ARGMAX_TOK == TOK` for every token of every shape.

| Run | Result | Wall clock, median of 5 (cold, including the Verilator build) |
|---|---|---|
| `make bringup` -- 2 shapes at WB=16, 2 tokens each, 4 timing settings | 4/4 match | 3.05 s (3.03 - 3.19) |
| `make bringup-sweep` -- 5 shapes at WB=64 and WB=128, 2 tokens each | 20/20 match | 9.27 s (9.24 - 9.36) |
| `uv run pytest -q sw/tests/test_bringup.py` (WB 16, 64 and 128) | 12 passed | 9.36 s (9.34 - 9.43) |

## Measurement notes

- The README's results table quotes the `qcore_top` rows above and the Demo
  configuration section of `syn/reports/qcore_top.md`, which `make synth`
  regenerates from `build/synth/synth_top.log` and checks.
- The synthetic token matches Qwen decode at short context. It streams no gamma
  rows, K^T/V tiles, KV writes, embed gather or descriptors (together ~3% of
  the design's per-token traffic), and its VPU stub is heavier than the real
  VPU share; the two effects pull in opposite directions.
- Wall-clock includes an ideal C++ memory model; a DRAM timing model (v1.4)
  adds host-side cost.
- `--threads` results were taken on macOS without core pinning; the direction
  (slower) is unambiguous at every run count.
- The probe sections above are the record of how the demo configuration was
  chosen; the `qcore_top` sections are the measurements the README quotes.
