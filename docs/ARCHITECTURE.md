# Quettos Core architecture

This document describes what the v1 hardware is and why it has this shape.
Numbers marked **estimate** are analytical projections; `make perf` and
`make synth` produce the measured values.

## Thesis

Agent inference is decode. Published traces of agent workloads put decode at
91-98.6% of LLM time, prefix-cache hit rates at 85-99%, and typical outputs at
~200 tokens per step (arXiv 2605.26297, 2606.30560, 2608.15127). The unit of
work is therefore one token for one sequence: a GEMV over ~498 MB of int8
weights (Qwen2.5-0.5B) with an arithmetic intensity of one MAC per weight byte.

A weight-stationary systolic array is a poor match for that: at batch 1 it is
roughly 3% utilized because it wants to reuse each weight against many
activations that do not exist. Quettos Core inverts the roles:

- every weight byte streams through a `WB`-byte-per-cycle port **exactly once
  per token** and is consumed the cycle it arrives;
- the `WB` output-stationary lanes each hold one 40-bit accumulator per output
  channel while a single activation scalar is broadcast per beat;
- activations, the residual stream, norm/RoPE/softmax/SiLU intermediates and
  per-token scales stay in a 128 KB on-chip vector SRAM.

So `cycles/token ~= bytes/token / WB + overhead` and `tokens/s = memory
bandwidth / bytes-per-token`. The design's job is to keep the weight port busy.

Attention is the same GEMV. K is stored transposed in 64-token tiles and
streams as "weights" with per-token scales; V streams row-major against int16
softmax weights. The row dimension `B_MAX` is the agent dimension (B activation
rows ride one weight pass); v1 keeps it as an RTL parameter proven by a
block-level B=2 test; end-to-end batched decode is the first roadmap item.

## v1 configuration

- `WB=64` lanes, `B_MAX=1` active row (generate-for over rows retained),
  `VL=4` vector lanes, heads processed sequentially, one descriptor in flight
  (fully serialized; cross-op weight prefetch is v1.1).
- Tiny CI config: `WB=16, B_MAX=2, VL=2, VSRAM_WORDS=2048`.
- Simulation fallback: `WB=128`, same RTL, identical tokens by construction
  once partial-last-tile support exists.

See `rtl/cfg/README.md` for how parameters are passed.

## Block diagram

The host/RTL boundary is the horizontal line above `qcore_top`. Above it is
C++ (or cocotb in unit tests). Below it is synthesizable SystemVerilog. The
external memory model is also C++ and sits outside the design boundary.

```
 host: C++ Verilator harness (tokenizer I/O, memory model, printing) / cocotb unit tests
       | CSR port (32b): CTRL, STATUS, PC, TOK, POS, ARGMAX_TOK/VAL, PERF[16]x64b, SAT/ERR counters
 +-----v---------------------------------------------------------------------------------+
 | qcore_top (WB, B_MAX, VL, VSRAM_WORDS, FIFO_BEATS)                                     |
 |  seq_fetch -> seq_dispatch --GEMV/EMBED--> stream_ctrl --64 int8 w/cycle + k tag--+    |
 |   (PC,       (decode, pos-derived      (bursts <=64 beats, weight FIFO,          v    |
 |    8-deep)    N/K/len/addr, in-order    meta side-FIFO, EMBED gather)   row[0] (x B_MAX)|
 |               issue, auto-fence,                                         8x lane_group |
 |               busy/retire)                                               (8w x 16a ->  |
 |                 |            |                                            40b acc x2)  |
 |                 | VPU cmd    | KVWRITE                                        |         |
 |                 v            v                                     A[k] ^    v         |
 |            vpu_top       kv_writer                            vsram 4096x256b  requant |
 |  (VRMSNORM VQUANT VROPE  (K^T byte    port A: MAC act reads  <----------------(acc*Sw  |
 |   VSILUMUL VSOFTMAX      scatter,     port B: VPU + requant writes             >>s1*Sx |
 |   VSUBC; VL lanes;       V row, meta) SREG[32] sfloat                          >>S,+b, |
 |   scalar LOD/sfloat;                                                           RMW,    |
 |   lut_interp exp2/sigmoid/rsqrt/recip)                                        argmax) |
 |                 |            |                                                  | dump  |
 |  mem_arb: 1 read port (stream > vpu tables > fetch, reserved fetch slot), 1 write port |
 |           (kv_writer, logit dump), write-ack counter for auto-fence; perf counters     |
 +--------------------------------------|------------------------------------------------+
        QMEM: rd_req{addr32,len8,tag4} v/r ; rd_data{WB*8b,tag,last} v ; wr{addr32,data,strobes} v/r ; wr_ack
 +--------------------------------------v------------------------------------------------+
 | C++ external memory model: fixed latency LAT (default 32), 1 beat/cycle, --bw-div N    |
 | image.bin: programs | RoPE table | k-center rows | tiled weights+meta | gammas |         |
 |            tied embedding/LM head | KV region (~498 MB Qwen + 14.2 MB KV @2048)          |
 +----------------------------------------------------------------------------------------+
```

Block labels inside the box are the `qcore_*` modules with the prefix dropped
for width (`vpu_top` = `qcore_vpu_top`, `mem_arb` = `qcore_mem_arb`, `requant`
= `qcore_requant`, `lane_group` = `qcore_mac_lane_group`, and so on).

Module list and responsibilities (all `rtl/qcore_*.sv`; LOC are **estimates**):

| Module | Purpose |
|---|---|
| `qcore_pkg` | parameters, opcode/flag localparams, descriptor field ranges, sfloat typedefs, `round_shift`/`sat` functions (explicit `qcore_pkg::` scoping only) |
| `qcore_top` | flat QMEM/CSR ports, instantiates everything, generate-for rows; parameter root |
| `qcore_csr` | CTRL/STATUS/PC/ROW_EN/TOK/POS/ARGMAX/PERF halves/SAT+ERR counters |
| `qcore_seq_fetch` | descriptor fetch, 8-deep queue, step-mode gating |
| `qcore_seq_dispatch` | decode, POS-derived N/K/len/addresses, in-order issue, auto-fence, busy/retire, stall classification |
| `qcore_mem_arb` | read arbiter with a reserved fetch slot, tag routing, write mux, ack counter, byte counters |
| `qcore_stream_ctrl` | bursts, weight FIFO, meta side-stream, tile/k counters, EMBED gather, partial last tile |
| `qcore_row` | activation word buffer, 8 lane groups, accumulator double buffer, tile handshake |
| `qcore_mac_lane_group` | 8 lanes of 8w x 16a -> 24-bit product, 40-bit accumulate (`acc <= prod + (tile_start ? 0 : acc)`) |
| `qcore_requant` | two-stage sfloat requant, S clamp + ERR, m==0 rule, bias, RMW, sat counters, absmax, argmax, dump, partial-tile drain |
| `qcore_vsram` | true-dual-port 256-bit RAM wrapper, `verilator public_flat_rd` for zero-cycle dumps |
| `qcore_vpu_top` / `_lane` / `_scalar` | the six V ops, VL lanes, LOD/sfloat/LUT scalar path |
| `qcore_lut_rom` / `qcore_lut_interp` | (v, dv) ROMs from `rtl/gen/*.hex` via `ROM_FILE`, linear interpolation |
| `qcore_kv_writer` | K^T byte scatter / V row / meta writes, issued-write tracking |
| `qcore_perf` | 16 x 64-bit counters with exclusive stall buckets |

## Decode step dataflow

The host writes `TOK`, `POS`, `ROW_EN`, pulses `START`; the program runs to
`HALT`. Everything position-dependent is derived in hardware from `POS`.

1. **EMBED**: gather 896 int8 bytes of row `TOK` from the tiled tied table
   (896 strided beats); `SREG <- row scale`; dequant through requant
   (`acc = q << 24`, the compiler subtracts 24 in `sbias`) -> residual `X`
   (int32, `FRAC_X`).
2. **Per layer**: `VRMSNORM(X, gamma_in)` -> `VQUANT` (int16 per-token) ->
   `GEMV(Wqkv, 18 tiles x 896 beats, meta {Sw, bias})` -> `QKV`. `VROPE` (14 q +
   2 k heads; cos/sin row at `rope_base + POS*128`). `VQUANT(q, group 64,
   scale_mul = log2e/8)` -> int16 q + 14 scales. `VSUBC(k, kcenter[layer][kvh])`
   then `VQUANT(k, group 64, int8)`; `VQUANT(v_raw, group 64, int8)` (V bias
   folded into the o_proj bias offline). `KVWRITE x4` (K^T byte scatter + meta;
   V row + meta). Hardware auto-fences memory-reading ops while writes are
   outstanding.
3. **Attention per q head** (14 sequential): `GEMV(K^T region of kv head h/7,
   n_from_pos, K=64, per-token K-scale meta)` -> scores (log2 domain);
   `VSOFTMAX(len = POS+1, V-scale array)` -> `w` int16 + `SREG_out`;
   `GEMV(V region, unit_meta, n=64, k_from_pos)` -> `CTX[64h..]`. Then
   `VQUANT(CTX)` -> `GEMV(Wo, accumulate: y += X)` -> `X`.
4. **MLP**: `VRMSNORM(X, gamma_post)` -> `VQUANT` -> `GEMV(Wgate|Wup, N=9728)`
   -> `GU`; `VSILUMUL` -> `VQUANT(int16, K=4864)` -> `GEMV(Wdown, accumulate)`
   -> `X`.
5. **Final**: `VRMSNORM` -> `VQUANT` -> `GEMV(LM head, N=151936,
   out_mode=ARGMAX)`; strict `>` so ties resolve to the lowest id; logits
   never touch VSRAM; optional `DUMP` for verification. `HALT`.

## Prefill/decode loop

Exact, and shared as one function by `golden.py`, `isa_sim.py` and the C++
harness:

```
for i in 0..P-2:          prefill.prog(TOK=prompt[i], POS=i)      # no final norm / LM head
decode.prog(TOK=prompt[P-1], POS=P-1)                    -> gen[0]
for j >= 1:               decode.prog(TOK=gen[j-1], POS=P-1+j) -> gen[j]
```

A 32-prompt + 20-generated run consumes 51 positions (labeled `T=51`). The
prefill program omits the final norm and LM head, which saves 27.6% of the
bytes per prompt token on Qwen (LM head share of linear MACs; see
`MEMORY_MAP.md`).

Prefix reuse (v1, `make demo-toolcall`): the **harness** saves and restores the
KV region plus `POS` in a WB-independent canonical format
(`[layer][kvh][token][64]` int8 + scales), re-laid-out on save/restore. This is
a harness save/restore of the RTL's KV state, not a prefix-caching system in
hardware.

## Host / RTL boundary

- Host side: tokenizer, chat template, the memory model (fixed latency `LAT`,
  1 beat/cycle, optional `--bw-div`), image loading, CSR driver, the
  prefill/decode loop above, token printing, perf collection, KV save/restore.
- RTL side: everything from descriptor fetch through argmax. The RTL never
  sees a token string, a float, or a fixed-point format; it sees descriptors,
  bytes, and shift fields.

## Invariant

**Nothing leaves the chip except K/V and the optional logit dump; the host
touches only CSRs inside a token.** Any change that needs the host to read or
write VSRAM mid-token is a design change, not a fix.

## Scope and measurement conventions

- **Performance** is reported as cycles per token, bytes per token, weight-port
  utilization and MAC utilization, read from the RTL's counters at a stated
  context length and batch. The memory model in simulation is fixed-latency,
  one beat per cycle; weight-port utilization counts read-beat busy cycles.
- **FPGA throughput** is derived from those cycle counts at a stated clock and
  memory bandwidth (100 MHz and 6.4 GB/s for the WB=64 configuration).
  Synthesis results come from Yosys `synth_xilinx`, reported post-synthesis
  with the exact command and tool version; fmax comes from nextpnr when run.
- **Bit-exact** means the RTL matches the integer golden model bit for bit,
  token by token and, on request, over the full logit vector. **Quality** is a
  measured perplexity, KL and top-1 delta against fp32 at the tested context
  lengths.
- **v1 runs a single sequence with heads sequential.** Every headline number
  runs the full vocabulary with the LM head on the accelerator. The row
  dimension (`B_MAX`), KV save/restore across requests and the tool-call demo
  are the foundations for batched decode, a paged KV cache and constrained
  decoding, which follow in v1.1 and later.
