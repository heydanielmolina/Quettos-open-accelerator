# Roadmap

## v1 (v0.1.0): what ships first

- **Golden software stack.** `numerics.py` as the single source of integer
  primitives, LUT and RoPE table generation, a numpy-only int8 quantizer with
  calibration for both models, an op-by-op bit-exact golden model, and a
  batched quality model for perplexity / KL / top-1 against fp32.
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
- **Synthesis.** Yosys `synth_xilinx` (xc7) report for the demo configuration
  with the exact command and tool version; ECP5 and a nextpnr fmax
  follow in a later release.
- **Results.** Performance tables produced only by `make perf` from RTL
  counters, the quality table of `uv run quettos check`, the tool-call demo
  with prefix-KV reuse, README, docs, architecture diagram and a recorded GIF.

### Demo configuration

WB=64 (which is also the synthesized configuration) as long as the Qwen
32-prompt + 20-token demo finishes in under 8 minutes in Verilator on a modern
laptop; otherwise the WB=128 build of the same RTL (shown to produce identical
tokens), and if needed SmolLM2-135M-Instruct as the live demo with Qwen as a
recorded run. Headline numbers always run the full vocabulary with the LM head
on the accelerator. See `docs/PERFORMANCE.md` for the measurement behind this
choice.

## After v1

- **v1.1**: rows-as-heads GQA attention and end-to-end multi-sequence decode
  with the measured utilization-vs-B curve; batched prefill; cross-op weight
  prefetch and per-row VPU slices; LOOP/JUMP descriptors; nextpnr-ecp5 fmax and
  a real tok/s line; quality scored over a wider set than the calibration corpus
  (WikiText-2, at least 32k tokens), the run the nightly `quality-regen` job is
  held for;
  WASM build of the same Verilator model for an in-browser demo.
- **v1.2**: paged KV with block tables (prefix sharing in hardware addressing),
  flash-style online softmax, wider attention weights for long contexts, on-chip
  K^T tile assembly buffer.
- **v1.3**: speculative-decode verify rows, grammar-mask port on the sampler,
  deterministic integer sampling in the bit-exact contract.
- **v1.4**: W4 with block-32 sub-scales (if 0.5B quality holds), SmoothQuant if
  measured to help, DRAM timing model.
- **v2**: multi-core weight sharding, on-chip weight residency, SmolLM2-360M /
  Granite-4.0-350M / Qwen3-0.6B targets, board bring-up with a DDR3 controller.
