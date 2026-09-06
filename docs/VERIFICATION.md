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
RTL (Verilator)   qcore_top, all three configurations
```

"Bit-exact" always means RTL == isa_sim == golden, our own integer model. It
never means "matches PyTorch". Quality relative to HF fp32 is a **measured
delta**, reported with standard errors at the calibration-set lengths (up to 520 tokens).

## The eight layers

1. **Static, every push.** Three-parser lint (`scripts/lint.sh`:
   Verilator `--lint-only -Wall -Wpedantic`, Yosys `hierarchy -check; proc;
   opt; check -assert`, Icarus `-g2012`); lutgen / RoPE regeneration diff
   report; `layout.json` sha256s; no-literal-`$readmemh` lint; forbidden
   construct greps.
2. **`numerics.py` pytest.** `round_shift` / `sat` / `sfloat` / `sfloat_mul`
   properties (m in range for all pairs), LUT error bounds vs float64,
   recip/rsqrt at `m in {1.0, 2.0, 4-eps}`, requant vs a big-int reference
   within 1 LSB including `S = 0 / 63` and the `m == 0` rule, the exponent
   equation, the zero-vector rule, the VQUANT clip rule (`|q| <= 32767`),
   softmax at `len = 1` / all-equal / extreme negatives, SiLU at
   `g in {-16, -8, 0, 8, 16}`, RoPE pairs, small-range exhaustive fuzz.
3. **cocotb 2.1 + Verilator unit tests on the tiny config** (each <= 1M
   cycles): lane group / row including B=2 broadcast (proves the batch
   parameter), requant fuzz including negative extremes and partial tiles,
   `lut_interp` at all indices, VPU ops including `len 1 / 64 / 2048` and
   group / scale_mul, `kv_writer` for `POS in {0, 1, 63, 64, 2047}`,
   `stream_ctrl` at `LAT 1 / 32 / 200` with backpressure (weight beats per busy
   cycle >= 0.98), dispatch POS-derived fields and auto-fence ordering,
   `ERR == 0` at `POS in {0, 1, 63, 64}`.
4. **Op-level RTL vs isa_sim.** `--layers N` truncated models (1-2 layers of
   both) and random tiny shapes (hidden 64-192, vocab 128-256, kv_heads 1-3;
   at least 5 shapes) -> step-mode per-op blobs (VSRAM range + SREG +
   memory range from `dump_plan.json`) -> compared bit-exact against
   `isa_sim.compare_sequence`, first
   mismatch reported at (op index, opcode, `.lst` line, element); the full KV
   region is compared after every token in truncated tests.
5. **End-to-end.** RTL tokens == isa_sim == golden on both models (CI: SmolLM2
   8+4 on WB=64; Qwen 4+1 nightly; 32+20 recorded); one full 151,936-logit
   dump compared; determinism at `--lat 1 / 200`, `--bw-div 2`, `--threads 1
   vs 4`; WB=64 vs WB=128 identical tokens on a tiny shape and on
   SmolLM2; nightly `--x-initial unique`.
6. **Quality (golden vs fp32, not RTL).** `uv run quettos check <alias>`
   scores the calibration set (1181 / 1314 tokens, 1175 / 1308 scored positions) teacher-forced: paired
   delta-NLL +/- SE, KL, top-1 and PPL for W8A16 and W8A8 on both models,
   written to `models/<name>/quality.json` and tabulated in `NUMERICS.md`.
   CI gate on SmolLM2 W8A16 (`KL <= 0.02 nats`, `top-1 >= 93%`; measured
   0.0049 and 95.66%); Qwen is reported (W8A16 measured `KL 0.0124`, `top-1
   95.57%`, with the int8 K cache conditioned by K-centering and the
   pairwise Q/K smoothing fold, see the K-centering section of
   `NUMERICS.md`). Saturation and shift-error
   counters are zero on every reported run. The table covers the calibration set; `docs/ROADMAP.md` lists the
   longer WikiText-2 run.
7. **Perf counters.** The harness asserts `busy = mac_active + stall_mem +
   stall_vpu + stall_kv + stall_seq + stall_drain`; byte counters are
   cross-checked against the C++ memory model; `WT_BYTES` ==
   `layout.json` `traffic.<program>.wt_bytes`; `MACS` == `traffic.<program>.macs`
   plus the attention term of `traffic.attention` at the token's position
   (`ISA.md`, PERF table); `isa_sim.py` counts the same three and is the
   reference for them.
8. **CI.** `.github/workflows/ci.yml` (ubuntu-latest, 4 vCPU) and
   `nightly.yml`: `YosysHQ/setup-oss-cad-suite@v4` pinned to a dated release
   (tool versions printed into `syn/reports`), `actions/cache` for the suite,
   `obj_dir` (by RTL hash) and `~/.ccache`, uv without torch, HF download
   cached. PR matrix with `timeout-minutes` per job: lint + pytest, cocotb
   units, tiny-shape RTL-vs-isa_sim e2e with `-CFLAGS -O1` + ccache, per-block
   `synth_xilinx` stat. SmolLM2 8+4 e2e moves to nightly if the PR job exceeds
   15 minutes on first measurement. Nightly: full-top synth + reports, ECP5
   (optional), Qwen 4+1, `--x-initial unique`, quality regeneration.

## What is checked into `models/<model>/`

`calib.json`, `quality.json` and `expected_tokens.json`. `expected_tokens.json` holds the golden model's greedy
continuation of `prompts/chat_short.json` and `prompts/tool_call_weather.json`
(20 tokens or until an end-of-sequence id) with the SHA-256 of each id list,
written by `uv run quettos golden <alias> --write-expected` and reproduced by
`sw/tests/test_golden.py`; the RTL and the ISA simulator must emit exactly
these ids. `image.bin` stays in the gitignored `build/`. A clean clone must
reproduce the sha256s (checked on CI and on a second machine).
