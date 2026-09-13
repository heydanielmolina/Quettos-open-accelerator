# Roadmap

## v1 (v0.1.0): what ships

- **Golden software stack.** `numerics.py` as the single source of integer
  primitives, LUT and RoPE table generation, a numpy-only int8 quantizer with
  calibration for both models, an op-by-op bit-exact golden model, and a
  batched quality model for perplexity / KL / top-1 against fp32 on two sets:
  the calibration set and 32,768 tokens of held-out WikiText-2 (`corpus.py`,
  fetched and hashed by `uv run quettos corpus`).
- **ISA and compiler.** Descriptor encoder and disassembler, tiled memory
  image, `decode.prog` / `prefill.prog`, `layout.json` with sha256s, and an
  ISA-level simulator that must match the golden model bit for bit.
- **RTL.** `qcore_pkg`, `qcore_vsram`, `qcore_mac_lane_group`, `qcore_row`,
  `qcore_requant`, `qcore_mem_arb`, `qcore_stream_ctrl`, `qcore_csr`,
  `qcore_seq_fetch`, `qcore_seq_dispatch`, `qcore_kv_writer`, `qcore_perf`,
  the vector unit (`qcore_vpu_*`, `qcore_lut_*`) and `qcore_top`, all clean
  under Verilator, Yosys and Icarus with zero waivers; cocotb block tests on the
  tiny configuration.
- **Integration.** Per-op bit-exact comparison of the RTL against the ISA
  simulator on random tiny shapes and truncated real models, then
  SmolLM2-135M-Instruct and Qwen2.5-0.5B-Instruct generating text end to end in
  Verilator with the C++ harness, `make demo` as one command, CI green.
- **Prefix reuse and the tool call.** The KV region saved with the record of
  what it belongs to (`docs/MEMORY_MAP.md`, the prefix file) and restored by a
  later run, which prefills only the positions after it; `make demo-toolcall`
  computes a prompt's system-and-tools turn once on `qcore_top`, restores it,
  prefills the user's turn and generates the tool call, beside the same prompt
  run with no reuse to hold the ids and the prefill cycles to.
- **Synthesis.** Yosys `synth_xilinx` (xc7) report for the demo configuration
  with the exact command and tool version; ECP5 and a nextpnr fmax
  follow in a later release.
- **Results.** Performance tables produced only by `make perf` from RTL
  counters, the two quality tables of `uv run quettos check` and
  `uv run quettos check --heldout`, the README and
  `docs/` with the block diagram of `docs/ARCHITECTURE.md` in it, and the run
  `make demo` prints: every token as it leaves `qcore_top`, and under it the
  summary `quettos demo-report` holds that run to.

### Demo configuration

`WB=64`, which is also the synthesized configuration. The rule it was chosen by:
`WB=64` stands as long as the Qwen 32-prompt + 20-token demo finishes in under
eight minutes in Verilator on a modern laptop, and it finishes in two
(`docs/PERFORMANCE.md`, the demo configuration). `WB=128` is the same-RTL
fallback and computes the same values. Headline numbers run the full vocabulary
with the LM head on the accelerator.

## After v1

- **v1.1**: rows-as-heads GQA attention and end-to-end multi-sequence decode
  with the measured utilization-vs-B curve; batched prefill; cross-op weight
  prefetch and per-row VPU slices; LOOP/JUMP descriptors; nextpnr-ecp5 fmax and
  a real tok/s line; WASM build of the same Verilator model for an in-browser
  demo.
- **v1.2**: paged KV with block tables (prefix sharing in hardware addressing),
  flash-style online softmax, wider attention weights for long contexts, on-chip
  K^T tile assembly buffer.
- **v1.3**: speculative-decode verify rows, grammar-mask port on the sampler,
  deterministic integer sampling in the bit-exact contract; the grammar mask
  under the v1 tool-call demo, so its output is well formed under sampling and
  not only at the argmax, over the paged prefix sharing of v1.2.
- **v1.4**: W4 with block-32 sub-scales (if 0.5B quality holds), SmoothQuant if
  measured to help, DRAM timing model.
- **v2**: multi-core weight sharding, on-chip weight residency, SmolLM2-360M /
  Granite-4.0-350M / Qwen3-0.6B targets, board bring-up with a DDR3 controller.
