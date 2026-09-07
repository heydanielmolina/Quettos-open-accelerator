# Verification

## Oracle chain

```
HF fp32 (transformers, optional torch)      quality reference only
   |  top-1 / KL / PPL deltas, never bit-exact
   v
golden.py         op-by-op integer forward, float64 BLAS for exact integer GEMV (partials < 2^53);
   |              forward_tokens (teacher-forced, all positions) == step (one program at one
   |  bit-exact   position) at every position; program.py supplies the requant constants
   v
quality.py        golden vs fp32 over the calibration set (uv run quettos check <alias>):
   |              top-1, KL, delta-NLL +/- SE, PPL -> models/<name>/quality.json
   |
isa_sim.py        executes decode.prog / prefill.prog on image.bin at value level, with
   |  bit-exact  numerics.py as its only arithmetic; == golden (every descriptor, KV byte and
   |             counter) on synthetic shapes at WB 16 / 64 / 128, two-layer real models and
   |             the complete models over prefill + decode (sw/tests/test_isa_sim.py)
   v
RTL (Verilator)   qcore_top, all three configurations; compare.py runs the bring-up
                  program on both and reports the first differing element
```

"Bit-exact" always means RTL == isa_sim == golden, our own integer model. It
never means "matches PyTorch". Quality relative to HF fp32 is a **measured
delta**, reported with standard errors at the calibration-set lengths (up to 520 tokens).

## The eight layers

1. **Static and structural, every push.** Three-parser lint (`scripts/lint.sh`:
   Verilator `--lint-only -Wall -Wpedantic`, Yosys `hierarchy -check; proc;
   opt; check -assert`, Icarus `-g2012`) over every `rtl/*.sv` module and every
   `sim/cocotb/wrappers/*.sv` assembly as its own top, zero warnings and no
   waivers; lutgen / RoPE regeneration diff report; `layout.json` sha256s;
   no-literal-`$readmemh` lint; forbidden construct greps.
   The lint proves the three tools accept the RTL; gate-level equivalence
   (`make gatesim`, `sim/gatesim/`) proves two of them read it the same way.
   Yosys maps a block to Xilinx 7-series cells and writes the netlist, and one
   Icarus bench drives the source module and that netlist from the same
   directed and random stimulus, comparing the concatenation of their outputs
   every cycle against Yosys's own cell models. The source side is read by the
   Icarus front end and the netlist side by the Yosys one, so a construct the
   two read differently is a mismatch instead of a silent difference between
   what simulates and what synthesizes. Thirteen configurations of ten blocks
   are covered, ten of the twelve modules under `rtl/`, and a case fails as well when an output port stops toggling,
   when the netlist instantiates a cell the Yosys library only declares, or
   when more than a tenth of the window is still undefined.
   `sim/gatesim/README.md` lists what is covered and the reason for each block
   that is not: `qcore_vsram`, the demo-width `qcore_stream_ctrl` FIFOs and
   `qcore_top` all infer `RAMB18E1` / `RAMB36E1`, which Yosys 0.65 declares
   without a simulation body.
2. **`numerics.py` pytest.** `round_shift` / `sat` / `sfloat` / `sfloat_mul`
   properties (m in range for all pairs), LUT error bounds vs float64,
   recip/rsqrt at `m in {1.0, 2.0, 4-eps}`, requant vs a big-int reference
   within 1 LSB including `S = 0 / 63` and the `m == 0` rule, the exponent
   equation, the zero-vector rule, the VQUANT clip rule (`|q| <= 32767`),
   softmax at `len = 1` / all-equal / extreme negatives, SiLU at
   `g in {-16, -8, 0, 8, 16}`, RoPE pairs, small-range exhaustive fuzz.
3. **cocotb 2.1 + Verilator block tests on the tiny config** (`make cocotb`,
   each <= 1M cycles): `vsram` on both ports with element strobes; lane group
   and row including the B=2 broadcast, which proves the batch parameter;
   requant fuzz including negative extremes, partial tiles, dumps and several
   rows; `stream_ctrl` at `LAT 1 / 32 / 200` with backpressure (weight beats per
   busy cycle >= 0.98), the EMBED gather and exact burst shapes; `mem_arb`
   routing, the reserved fetch slot and `wr_idle` timing; `kv_writer` for
   `POS in {0, 1, 63, 64, 2047}` compared byte for byte against `isa_sim`;
   `seq_fetch` restarts at every position inside a beat, stepping across beat
   boundaries and the `fetch_hold` write fence; `seq_dispatch` POS-derived
   fields over 2,160 decoded cases against `isa_sim.gemv_dims` and
   `softmax_len`, the auto-fence ordering, the six exclusive stall buckets, the
   zero-work rule, step mode, the write fence a step, a fault or an ABORT ends
   on, and the rule that a bounds event is counted only for a descriptor that
   commits;
   `csr` register semantics; `perf` bucket exclusivity and snapshot timing; and
   the assembled GEMV path (`wrappers/qcore_gemv_wrap.sv`) against
   `numerics.requant` over the exact integer matmul in all three
   configurations. The `lut_interp` and vector-op benches follow
   `qcore_vpu_top`.
4. **Op-level RTL vs isa_sim.** `sw/quettos/compare.py` (`make bringup`,
   `make bringup-sweep`, `sw/tests/test_bringup.py`) runs the three-descriptor
   bring-up program of `docs/ISA.md` (`EMBED`, the tied LM head as a `GEMV` in
   `ARGMAX_DUMP` mode, `HALT`, with the activation scale
   `qcore_vpu_top`'s `VQUANT` would write loaded by the host; see
   `docs/PERFORMANCE.md`, "Measured against the ISA simulator") on `qcore_top`
   through the Verilator harness and on `isa_sim`, one
   `CTRL.STEP` per descriptor, and compares the whole used VSRAM range of every
   bank, all 32 SREG words of every bank, the `DUMP` region holding every
   int32 logit, `PC`, `STATUS`, the ARGMAX registers, the four event counters
   and `DESCRIPTORS` / `MACS` / `WT_BYTES` after each descriptor. The first
   difference is reported as (descriptor index, opcode, field, element,
   expected, got). It runs over random tiny shapes (hidden 64-192, vocab
   128-256, kv_heads 1-3; at least 5 shapes) at WB 16 / 64 / 128, each at
   `--lat` 1 / 32 / 200 and `--bw-div` 2, and the RTL records must be identical
   across those four settings. The `--layers N` truncated models, the six
   vector opcodes and the KV region comparison follow `qcore_vpu_top` through
   `isa_sim.compare_sequence`.
5. **End-to-end.** A whole compiled program runs on `qcore_top` as a traffic
   and cycle measurement (`make perf`, `--traffic`): the real addresses,
   strides, meta and partial tiles of `decode.prog` and `prefill.prog` over a
   36-token prompt and one decode token, checked against `layout.json`'s
   traffic model and against the memory model's own counts, with the cycle
   counts, the six stall buckets and the conditions of the run recorded in
   `build/perf/perf.json`. Determinism is checked at `--lat 1 / 32 / 200` and
   `--bw-div 2` (layer 4 requires identical RTL records across the four
   settings), at `--threads 1` against `--threads 4` (identical cycle counts on
   `qcore_top`), and at WB=64 against WB=128 through `make bringup-sweep`.
   Token equality -- RTL tokens == isa_sim == golden on both models (CI:
   SmolLM2 8+4 on WB=64; Qwen 4+1 nightly; 32+20 recorded), one full
   151,936-logit dump compared, and the nightly `--x-initial unique` run --
   follows `qcore_vpu_top`, whose opcodes the compiled programs contain.
6. **Quality (golden vs fp32, not RTL).** `uv run quettos check <alias>`
   scores the calibration set (1181 / 1314 tokens, 1175 / 1308 scored positions) teacher-forced: paired
   delta-NLL +/- SE, KL, top-1 and PPL for W8A16 and W8A8 on both models,
   written to `models/<name>/quality.json` and tabulated in `NUMERICS.md`.
   CI gate on SmolLM2 W8A16 (`KL <= 0.02 nats`, `top-1 >= 93%`; measured
   0.0048 and 95.66%); Qwen is reported (W8A16 measured `KL 0.0124`, `top-1
   95.57%`, with the int8 K cache conditioned by K-centering and the
   pairwise Q/K smoothing fold, see the K-centering section of
   `NUMERICS.md`). Saturation and shift-error
   counters are zero on every reported run. The table covers the calibration set; `docs/ROADMAP.md` lists the
   longer WikiText-2 run.
7. **Perf counters.** The harness asserts `busy = mac_active + stall_mem +
   stall_vpu + stall_kv + stall_seq + stall_drain`; `RD_BYTES == RD_BEATS * WB`;
   the write counters equal the C++ memory model's own counts and `RD_BEATS`
   trails it by at most the fetch unit's two outstanding bursts per token, which
   is what a burst still in flight at a `HALT` snapshot costs; `WT_BYTES` ==
   `layout.json` `traffic.<program>.wt_bytes`; `MACS` == `traffic.<program>.macs`
   plus the attention term of `traffic.attention` at the token's position
   (`ISA.md`, PERF table); `isa_sim.py` counts the same three and is the
   reference for them, compared per descriptor by `compare.py`. A run fails on
   any `SAT_*` or `ERR_*` event unless `--allow-sat` is given.
8. **CI.** `.github/workflows/ci.yml` (ubuntu-latest, 4 vCPU) and
   `nightly.yml`: `YosysHQ/setup-oss-cad-suite@v4` pinned to a dated release
   (tool versions printed into `syn/reports`), `actions/cache` for the suite,
   the harness object directory (keyed on `rtl/*.sv`, `rtl/*.svh` and the C++)
   and `~/.ccache`, uv without torch, HF download
   cached. Five PR jobs with `timeout-minutes` each: lint + pytest, then the
   cocotb block tests, gate-level equivalence (`make gatesim`), the synthesis
   run (`make synth`: every block and the whole core in both configurations,
   with every `syn/reports/*.md` regenerated and checked against the run; a
   report written by a different Yosys build is held to its parameters and its
   hard-block inventory, since LUT packing and path length belong to the build)
   and
   the tiny-configuration bring-up comparison (`bringup-tiny`: `make
   harness-csr` then `make bringup`, 3.05 s locally from a cold Verilator
   build), the last four gated on the first. Nightly holds the long runs, each
   held by `if: false` until the piece it runs on is in the tree, with that
   condition named on the gate: the ECP5 stat and nextpnr fmax, Qwen 4+1 end to
   end, SmolLM2 8+4 (which sits there rather than in the PR matrix whenever the
   PR job would exceed 15 minutes), the `--x-initial unique` determinism run,
   and the quality regeneration over the wider set. The whole-core synthesis is
   a PR job, so nightly carries no synthesis of its own beyond ECP5.

## What is checked into `models/<model>/`

`calib.json`, `quality.json` and `expected_tokens.json`. `expected_tokens.json` holds the golden model's greedy
continuation of `prompts/chat_short.json` and `prompts/tool_call_weather.json`
(20 tokens or until an end-of-sequence id) with the SHA-256 of each id list,
written by `uv run quettos golden <alias> --write-expected` and reproduced by
`sw/tests/test_golden.py`; the RTL and the ISA simulator must emit exactly
these ids. `image.bin` stays in the gitignored `build/`. A clean clone must
reproduce the sha256s (checked on CI and on a second machine).
