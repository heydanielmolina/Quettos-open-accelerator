# Memory map

Three address spaces: the external QMEM image (`image.bin`), the on-chip vector
SRAM (element-indexed), and the KV region inside QMEM. All QMEM regions are
64-byte aligned. Exact addresses for a given model and `WB` come from
`layout.json`; the numbers below are the Qwen2.5-0.5B-Instruct layout at
`MAX_CTX = 2048` as specified, and are confirmed by the compiler.

## QMEM image (Qwen2.5-0.5B-Instruct, MAX_CTX 2048)

| Base | Region | Size |
|---|---|---|
| `0x0000_0000` | programs (`decode.prog`, `prefill.prog`) | <= 1 MB |
| `0x0010_0000` | RoPE table, 2048 x 128 B | 256 KB |
| `0x0014_0000` | constants: k-center rows 24 layers x 2 kv heads x 256 B, zero pads | small |
| `0x0020_0000` | per layer (x24): `Wqkv` 1,032,192 + meta 9,216; `Wo` 802,816 + meta 7,168; `Wgu` 8,716,288 + meta 77,824; `Wdown` 4,358,144 + meta 7,168; `gamma_in` 1,792; `gamma_post` 1,792 | 15,014,400 B per layer + alignment padding, ~360.4 MB total |
| after layers | tied embedding / LM head 136,134,656 + meta 1,215,488 + final gamma 1,792 | ~137.4 MB |
| after that | KV region (per sequence) | 14.2 MB |
| | **total** | **~498 MB + KV** |

SmolLM2-135M-Instruct: ~136 MB of weights + 26.5 MB KV at MAX_CTX 2048 (30
layers x 3 kv heads).

Bytes streamed per decode token (weights + meta + gammas; from the model
shapes, not yet measured on RTL): Qwen 497,697,536 B = 7,776,524 beats at 64 B;
SmolLM2 136,187,520 B = 2,127,930 beats. The LM head is 136,134,656 of Qwen's
493,961,216 linear MACs (27.6%), which is what `prefill.prog` skips.

Per-token traffic in addition to weights (Qwen decode, T ~ 51, **estimate**
until `make perf`): K^T / V / K-meta reads ~2.67 MB (KV heads re-read 7x with
heads sequential), KV writes 3,216 beats (byte-scatter write amplification), embed 896 beats, program 759 beats. "Weight-port busy" is defined
and labeled as any-read-beat busy cycles.

## Weight tile layout

Each linear weight is int8 tiled `[N/WB][K][WB]`: tile `t` holds output
channels `t*WB .. t*WB+WB-1`; within a tile, beat `k` holds the `WB` weights of
input `k`. Byte address of `(tile, k)` is `addr_a + (tile*K + k) * WB`. Partial
last tiles are zero-padded and their meta has `m = 0`. Meta for channel `j` of
tile `t` is at `addr_m + (t*WB + j) * 8`.

The tied embedding table uses the same tiling; EMBED gathers row `TOK` as 896
strided single-beat reads (one byte used per beat, ~896 beats per token).

## VSRAM element map (Qwen, MAX_CTX 2048)

The VSRAM is 4096 x 256-bit = 128 KB = 32,768 int32 elements. Element `e` is
word `e / 8`, bits `[32*(e%8)+31 : 32*(e%8)]`. Port A serves MAC activation
reads (one word per 8 `k`, prefetched); port B serves VPU and requant
writes/RMW (never concurrent in v1). READ_FIRST, registered reads.

| Elements | Name | Contents |
|---|---|---|
| 0 - 895 | `X` | residual stream (int32, `FRAC_X`) |
| 896 - 1791 | `XN` | normed residual |
| 1792 - 2687 | `A` | quantized activation (int16 in low half) |
| 2688 - 3839 | `QKV` | q (896) \| k (128) \| v (128) |
| 3840 - 4735 | `CTX` | attention context, 14 heads x 64 |
| 4736 - 5631 | `CTXQ` | quantized context |
| 5632 - 15359 | `GU` | gate \| up (9728) |
| 15360 - 20223 | `HQ` | quantized SiLU(gate)*up (4864) |
| 20224 - 22271 | `S` | attention scores, one head, up to 2048 positions |
| 22272 - 24319 | `W` | softmax weights int16, up to 2048 positions |

SmolLM2 uses 12,544 elements; the tiny config needs `VSRAM_WORDS >= 2048` for
truncated-model tests.

Other on-chip state (**estimates** of what synthesis will report): accumulators
64 x 40 b x 2; `SREG` 32 x 24 b; weight FIFO 128 x 64 B; meta side-FIFO
64 x 64 B; descriptor queue 8 x 32 B; ROMs ~6 RAMB36; total BRAM ~44 RAMB36.

## KV layout

Per (layer, kv head) at `MAX_CTX = 2048`:

| Sub-region | Layout | Size |
|---|---|---|
| K^T tiles | 64-token tiles, transposed: tile `POS/WB` holds `[d=0..63][token%WB]` int8, i.e. byte `(POS/WB)*64*WB + d*WB + POS%WB` | 128 KB |
| V rows | row-major: token `POS` at byte `POS*WB`, 64 int8 (`v_raw`, bias folded into o_proj) | 128 KB |
| K meta | 8 B per token `{0, Sk}` at `POS*8` | 16 KB |
| V meta | 8 B per token `{0, Sv}` at `POS*8` | 16 KB |
| **total per (layer, kv head)** | | **288 KB**; x48 = 14.2 MB for Qwen |

KVWRITE (transposed) issues 64 single-byte-strobe writes per token; the
row-mode variant issues one WB-byte write. The compiler writes zero meta for
the whole `MAX_CTX` range so that unwritten positions read as `m = 0`.

Canonical (WB-independent) format used by the harness for prefix save/restore:
`[layer][kvh][token][64]` int8 plus scales, re-laid-out to the tiled format on
restore.
