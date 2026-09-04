# Performance: Verilator speed probe and the demo decision

Status: **initial cost-model measurement.** Every number here comes from
`sim/probe/`, a full-width skeleton with the final design's datapath widths and
memories, derated by 0.65 as a safety margin. Its purpose is to lock the demo
configuration before RTL integration; `make perf` on the real `qcore_top`
supersedes it, and the README's Verilator column is filled from that run.

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
  17.3M at WB=128), single-thread configs repeated 4-5 times; the threaded
  configs were run once each (their slowdown is unambiguous).
- **Machine:** Apple M5 Pro (6 performance + 12 efficiency cores), 64 GB,
  macOS (Darwin 25.5), Apple clang 17.0.0, Verilator 5.048, 2026-09-02.
  No `numactl` on macOS; threads were not pinned.

## Raw measurements

| Config | `--threads` | Mcycles/s (median) | Mcycles/s (range, n) | Verilator build time |
|---|---|---|---|---|
| WB=64  | 1 | **5.84** | 5.81 - 5.93 (n=5) | 1.32 s |
| WB=64  | 2 | 1.18 | (n=1) | 1.33 s |
| WB=64  | 4 | 0.65 | (n=1) | 1.34 s |
| WB=128 | 1 | **3.49** | 3.44 - 3.51 (n=4) | 1.43 s |
| WB=128 | 2 | 1.29 | (n=1) | 1.43 s |
| WB=128 | 4 | 0.66 | (n=1) | 1.43 s |

Facts worth stating plainly:

- **`--threads` is slower than one thread on this design**, by 4.5-5x at 2
  threads and 9x at 4 threads. The skeleton is far too small for Verilator's
  MTask partitioning; per-eval thread synchronization dominates. This matches
  the rule ("default 1 unless >= 1.3x measured"): **`--threads 1` is the
  default.** The real top is re-measured with the same method.
- Checksums are identical across thread counts for a given WB
  (`a17f63bf` at WB=64, `2601a93c` at WB=128 for 4 tokens), i.e. the threaded
  builds are deterministic and functionally identical.
- Build times are ~1.3-1.4 s because the skeleton generates only ~0.9 MB of
  C++ in 6 files; the real top (~5,200 SV LOC, `--output-split 20000`) will
  take longer and stays well inside the budget.
- Per-cycle cost scales sub-linearly with WB: WB=128 costs 1.67x more per
  cycle for 2x the work per cycle, so per *token* WB=128 is only ~1.2x faster
  in wall-clock than WB=64 (4.34M cycles / 3.49 Mc/s = 1.24 s vs 8.22M / 5.84 =
  1.41 s). The fallback buys little.
- Latency sensitivity (WB=64, 1 thread, 2 tokens): `--lat 1` 5.79 Mc/s;
  `--lat 200` 6.51 Mc/s but 3x more cycles, because the C++ model's 64-beat
  in-flight window throttles bandwidth above LAT=64 (expected; idle cycles are
  cheaper to simulate).

## Derated numbers (x0.65 safety margin)

| Config | Raw Mcycles/s | Derated Mcycles/s |
|---|---|---|
| WB=64,  1 thread | 5.84 | **3.80** |
| WB=128, 1 thread | 3.49 | **2.27** |

The 0.65 derating is an allowance for everything the skeleton lacks
(sequencer/dispatch, memory arbiter, kv_writer, CSR, 16 perf counters, six VPU
op FSMs, LUT ROMs and their interpolators, real per-token meta strides). These
are predominantly control logic and narrow registers, cheap per cycle next to
the 64-lane datapath and the 512-bit FIFO copies the skeleton already pays for,
so 0.65 is believed conservative; the re-measurement on the real top will tell.

## Projected wall-clock (derated; raw in parentheses)

Cycle counts are analytical estimates (replaced by RTL counters once the
design runs end to end); the probe's own token was 8.22M cycles at WB=64, 2% above the
design estimate of 8.06M.

| Run | Cycles | WB=64 @ 3.80 Mc/s | WB=128 @ 2.27 Mc/s |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct 32+20 demo | 350M (WB=64) / 184M (WB=128) | **92 s = 1.5 min** (60 s) | 81 s = 1.4 min (53 s) |
| Qwen decode, one token | 8.06M | 2.1 s (1.4 s) | - |
| Qwen prefill, one prompt token | 5.91M | 1.6 s (1.0 s) | - |
| SmolLM2-135M-Instruct 32+20 | 107M | 28 s (18 s) | - |
| SmolLM2 CI 8+4 (this machine) | 24.5M | 6.4 s (4.2 s) | - |
| Qwen 4+1 nightly | ~31.7M | 8.3 s (5.4 s) | - |
| 512-token prefix regen (`make regen-prefix`) | ~3.03G | 13.3 min (8.6 min) | - |

CI runs on `ubuntu-latest` (4 vCPU, `-CFLAGS -O1`, no Apple silicon); expect
several times slower per cycle there. Even at 5x slower the SmolLM2 8+4 e2e
job is ~30 s of simulation, well inside the 15-minute PR budget.

## Decision rule applied

Decision rule (re-checked on the complete design): *if Qwen 32+20 <= 8 min on WB=64, that is
`make demo` (demo config == synthesized config); else WB=128; else SmolLM2 live.*

- Qwen 32+20 on WB=64, derated: **92 s (1.5 min) <= 8 min. PASS.**
- Margin: 8 min corresponds to 0.73 Mcycles/s; the derated measurement is 5.2x
  above that, the raw measurement 8x. The real design would have to be more
  than 5x slower per cycle than the derated skeleton to flip the decision.

**Locked demo configuration: `WB=64, B_MAX=1, VL=4, VSRAM_WORDS=4096,
FIFO_BEATS=128, ACC_W=40`, Verilator `--threads 1`.** This is also the
synthesized configuration. WB=128 stays a same-RTL fallback (identical tokens
by construction once partial-last-tile support exists); the tiny CI config
is unchanged.

## Measurement notes

- Cost-model measurement. The README's Verilator Mcycles/s column comes from
  `make perf` on the real design.
- The synthetic token matches Qwen decode at short context. It streams no gamma
  rows, K^T/V tiles, KV writes, embed gather or descriptors (together ~3% of
  the design's per-token traffic), and its VPU stub is heavier than the real
  VPU share; the two effects pull in opposite directions.
- Wall-clock includes an ideal C++ memory model; a DRAM timing model (v1.4)
  adds host-side cost.
- `--threads` results are single runs on macOS without core pinning; the
  direction (slower) is unambiguous.
- The same method is re-run on the real `qcore_top`, first with the VPU
  stubbed and then on the complete design with the real image (`make perf`);
  those measurements replace this page.
