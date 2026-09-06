# Memory map

Three address spaces: the external QMEM image (`image.bin`), the on-chip vector
SRAM (element-indexed), and the KV region inside QMEM. All QMEM regions are
64-byte aligned. `uv run quettos compile <alias>` (`sw/quettos/compiler.py`)
lays a quantized model out and writes every address and size to `layout.json`;
the numbers below are the compiler's output for Qwen2.5-0.5B-Instruct and
SmolLM2-135M-Instruct at `MAX_CTX = 2048`, `WB = 64`.

## QMEM image

Fixed bases for the programs, the RoPE table and the constants; the weights,
the tied embedding / LM head and the KV region follow, packed in that order.

| Base | Region | Qwen2.5-0.5B-Instruct | SmolLM2-135M-Instruct |
|---|---|---|---|
| `0x0000_0000` | `decode.prog` | 1,493 descriptors, 47,776 B | 1,475 descriptors, 47,200 B |
| next beat | `prefill.prog` | `0x0000_BAC0`: 1,490 descriptors, 47,680 B | `0x0000_B880`: 1,472 descriptors, 47,104 B |
| `0x0010_0000` | RoPE table, `MAX_CTX` x 128 B | 262,144 B | 262,144 B |
| `0x0014_0000` | K-centering rows, one per layer: `kv_heads` x 64 int32 | 24 x 512 B | 30 x 768 B |
| `0x0020_0000` | per layer: `Wqkv` + meta, `Wo` + meta, `Wgu` + meta, `Wdown` + meta, `gamma_in`, `gamma_post` | 15,014,400 B x 24 layers | 3,582,720 B x 30 layers |
| after the layers | tied embedding / LM head + meta + final gamma | `0x159A_7000`: 136,134,656 + 1,215,488 + 1,792 B | `0x0688_0A00`: 28,311,552 + 393,216 + 1,152 B |
| after that | KV region (one sequence) | `0x1DCA_4300`: 14,155,776 B | `0x083E_0E80`: 26,542,080 B |
| | **image** | **513,950,464 B** | **164,826,752 B** |

Per-layer region sizes (Qwen / SmolLM2): `Wqkv` 1,032,192 / 552,960 B with
9,216 / 7,680 B of meta; `Wo` 802,816 / 331,776 with 7,168 / 4,608; `Wgu`
8,716,288 / 1,769,472 with 77,824 / 24,576; `Wdown` 4,358,144 / 884,736 with
7,168 / 4,608; each gamma 1,792 / 1,152. The programs occupy 95,488 /
94,336 B of the 1 MB program window.

Bytes streamed per decode token (weights + meta + gammas; `traffic.<program>.total`
in `layout.json`): Qwen 497,697,536 B = 7,776,524 beats at 64 B; SmolLM2
136,187,520 B = 2,127,930 beats. A prefill token streams 360,345,600 /
107,481,600 B: the LM head is 136,134,656 of Qwen's 493,961,216 weight bytes
(27.6%), which is what `prefill.prog` skips. The hardware counter `WT_BYTES`
(weights + meta + the EMBED gather, no gammas) is `traffic.<program>.wt_bytes`:
Qwen decode 497,610,632 B, SmolLM2 136,117,832 B; `traffic.attention` holds the
per-head MAC rates that complete the `MACS` counter at a position (`ISA.md`,
PERF table).

Per-token traffic in addition to the weight stream (Qwen decode, T ~ 51,
**estimate**): K^T / V / K-meta reads ~2.67 MB (KV heads re-read 7x with heads
sequential), KV writes 3,216 beats (byte-scatter write amplification), embed
896 beats, program 747 beats. "Weight-port busy" is defined and labeled as
any-read-beat busy cycles.

## Compiler outputs

`uv run quettos compile <alias> [--layers N] [--out DIR] [--max-ctx 2048]
[--wb 64] [--a-bits 16] [--prompt FILE]` reads `build/quant/<name>.npz` and
writes to `build/images/<name>/` (`<name>-l<N>` when truncated to `N` layers):

| File | Contents |
|---|---|
| `image.bin` | the QMEM image above |
| `decode.prog`, `prefill.prog` | the descriptor programs, 32 B per descriptor, HALT-terminated; the same bytes sit at their image addresses |
| `program.lst`, `prefill.lst` | `isa.disassemble` listings of the two programs |
| `layout.json` | every region (`name`, `addr`, `size`, `kind`, plus the matrix shape and tile count, the meta record count, the gamma exponent or the KV layer / head), the KV tile counts, the VSRAM map, the SREG assignment, the CSR map, `FRAC` per class, the program constants, per-token stream bytes / beats / `wt_bytes` / MACs and the attention MAC rates, and the sha256 of the image, the programs, `dump_plan.json`, the RoPE table, `luts.json`, `tokens.bin`, the prompt ids and `expected_tokens.json` |
| `dump_plan.json` | per descriptor of each program: `op`, the dataflow `name` (the golden trace op), `layer`, `head`, `kv_head`, the VSRAM range written (`start`, `count`), the SREG ids written, the memory regions written (KVWRITE), the CSRs written (the ARGMAX GEMV) and `pos_dependent` (the listed extent is the capacity; POS sets the actual one) |
| `tokens.bin` | `u32 count`, then `u16 length` + bytes per id (`tokenizer_io.write_tokens_bin`) |
| `prompt.tokens` | one decimal token id per line for the `--prompt` file |

Two compiles of the same model and parameters produce byte-identical files.
The compiler rejects a requant shift window outside `[0, 63]`, a VSRAM or SREG
allocation that overflows, programs that reach the RoPE base, and a `MAX_CTX`
that is not a multiple of `WB` or exceeds 2048 (the RoPE table). The requant
constants of both programs are `program.build` at `MAX_CTX = 2048` for every
`--max-ctx`; a smaller `--max-ctx` sizes the KV region, the RoPE rows and the
capacity fields only, so the golden model's defaults reproduce the compiled
program at every image size.

## Weight tile layout

Each linear weight is int8 tiled `[N/WB][K][WB]`: tile `t` holds output
channels `t*WB .. t*WB+WB-1`; within a tile, beat `k` holds the `WB` weights of
input `k`. Byte address of `(tile, k)` is `addr_a + (tile*K + k) * WB`. Partial
last tiles are zero-padded and their meta has `m = 0`. Meta for channel `j` of
tile `t` is at `addr_m + (t*WB + j) * 8`; a meta region holds
`ceil(N/WB) * WB` records, so its size is `ceil(N/WB) * WB * 8` bytes (the
tables above are `WB = 64` figures; at `WB = 128` SmolLM2's `Wqkv` meta with
`N = 960` grows from 7,680 to 8,192 B). `compiler.tile_weights` /
`untile_weights` and `pack_meta` / `unpack_meta` are the two directions.

The tied embedding table uses the same tiling; EMBED gathers row `TOK` as 896
strided single-beat reads (one byte used per beat, 896 beats per token) and
reads its scale from the LM-head meta at `addr_m + TOK * 8`.

## VSRAM element map

The VSRAM is 4096 x 256-bit = 128 KB = 32,768 int32 elements. Element `e` is
word `e / 8`, bits `[32*(e%8)+31 : 32*(e%8)]`. Port A serves MAC activation
reads (one word per 8 `k`, prefetched); port B serves VPU and requant
writes/RMW (never concurrent in v1). READ_FIRST, registered reads.

The compiler allocates the regions below in this order, each start 8-aligned,
so every GEMV output (`X`, `QKV`, `CTX`, `GU`, `S`) begins on a word boundary.
q, K and V are quantized in place inside `QKV` (K after VSUBC); VSILUMUL writes
the int32 `silu(gate) * up` into `HQ` and the following VQUANT quantizes it in
place.

Qwen2.5-0.5B-Instruct, 24,320 elements:

| Elements | Name | Contents |
|---|---|---|
| 0 - 895 | `X` | residual stream (int32, `FRAC_X`) |
| 896 - 1791 | `XN` | normed residual |
| 1792 - 2687 | `A` | quantized activation (int16 in low half) |
| 2688 - 3839 | `QKV` | q (896) \| k (128) \| v (128) |
| 3840 - 4735 | `CTX` | attention context, 14 heads x 64 |
| 4736 - 5631 | `CTXQ` | quantized context |
| 5632 - 15359 | `GU` | gate \| up (9728) |
| 15360 - 20223 | `HQ` | `silu(gate) * up`, then its quantized form (4864) |
| 20224 - 22271 | `S` | attention scores, one head, up to 2048 positions |
| 22272 - 24319 | `W` | softmax weights int16, up to 2048 positions |

SmolLM2-135M-Instruct, 12,544 elements: `X` 0 - 575, `XN` 576 - 1151, `A`
1152 - 1727, `QKV` 1728 - 2687 (576 \| 192 \| 192), `CTX` 2688 - 3263, `CTXQ`
3264 - 3839, `GU` 3840 - 6911, `HQ` 6912 - 8447, `S` 8448 - 10495, `W`
10496 - 12543. The tiny config needs `VSRAM_WORDS >= 2048` for truncated-model
tests.

### Scale registers

`SREG` holds 32 x 32 b per row: an sfloat `{u16 m, i8 e}` in the low 24 bits,
or the absmax tracked by VRMSNORM / VSILUMUL as a non-negative int32. The
compiler assigns register 0 to the tracked absmax, 1 to the scale of the normed
activation (QKV, gate|up and LM-head GEMVs), 2 to the quantized context
(o_proj), 3 to the quantized `h` (down), 4 to the softmax `SREG_out` (PV), then
one per query head for the q scales, one per KV head for the K scales and one
per KV head for the V scales: 23 registers on Qwen, 20 on SmolLM2, at most
`5 + heads + 2 kv_heads`.

Other on-chip state (**estimates** of what synthesis will report): accumulators
64 x 40 b x 2; `SREG` 32 x 32 b; weight FIFO 128 x 64 B; meta side-FIFO
64 x 64 B; descriptor queue 8 x 32 B; ROMs ~6 RAMB36; total BRAM ~44 RAMB36.

## KV layout

Per (layer, kv head), in the order K^T, V, K meta, V meta, layer-major with
the KV head minor. Both caches are the weight tiling above, so the GEMV
address generator streams them unchanged: K^T has the tokens as output
channels and `K = 64` (`[MAX_CTX/WB][64][WB]`, the scores GEMV with
`n_from_pos`); V has the 64 dimensions as channels and the tokens as `K`
(`[ceil(64/WB)][MAX_CTX][WB]`, the PV GEMV with `k = MAX_CTX` as the tile
stride and `k_from_pos`). At `MAX_CTX = 2048`, `WB = 64`:

| Sub-region | Layout | Size |
|---|---|---|
| K^T tiles | `WB`-token tiles, transposed: tile `POS/WB` holds `[d=0..63][token%WB]` int8, i.e. byte `(POS/WB)*64*WB + d*WB + POS%WB` | 128 KB |
| V tiles | dimension `d` of token `POS` at byte `((d/WB)*MAX_CTX + POS)*WB + d%WB`: one zero-padded `WB`-byte row per token when `WB >= 64` (`v_raw`, bias folded into o_proj) | 128 KB |
| K meta | 8 B per token `{0, Sk}` at `POS*8` | 16 KB |
| V meta | 8 B per token `{0, Sv}` at `POS*8` | 16 KB |
| **total per (layer, kv head)** | | **288 KB**; x48 = 14,155,776 B for Qwen, x90 = 26,542,080 B for SmolLM2 |

Sizes are `kt = (MAX_CTX/WB) * 64 * WB` (always `64 * MAX_CTX`) and
`v = ceil(64/WB) * MAX_CTX * WB`: 256 KB per (layer, kv head) for V at
`WB = 128` (the KV region then totals 16,908,288 B on Qwen), 4 KB per 64
positions at `WB = 16`. `layout.json` records `kv.kt_tiles` and `kv.v_tiles`.
KVWRITE (`TRANSPOSED`) issues 64 single-byte-strobe writes per token; the V
variant issues `ceil(64/WB)` `WB`-byte writes, one per V tile, and both carry
`MAX_CTX` in the `k` field. The compiler writes zero data and zero meta for the
whole `MAX_CTX` range so that unwritten positions read as `m = 0`.

Canonical (WB-independent) format used by the harness for prefix save/restore:
`[layer][kvh][token][64]` int8 plus scales, re-laid-out to the tiled format on
restore.
