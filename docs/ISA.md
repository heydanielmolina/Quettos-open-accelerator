# Quettos Core ISA

`ISA_VERSION = 1`. The constant lives in `qcore_pkg` and `sw/quettos/isa.py`,
is written to `layout.json`, and is asserted by the harness at load time.

## Programming model

The compiler emits, per model:

- `image.bin` -- programs, RoPE table, constants, tiled weights + meta, gammas,
  tied embedding / LM head, KV region (see `MEMORY_MAP.md`);
- two straight-line 32-byte descriptor programs, `decode.prog` and
  `prefill.prog` (prefill = decode minus the final norm and LM head);
- `layout.json` -- every address, the CSR map, the VSRAM map, `FRAC` per tensor
  class, sha256 of image / programs / tables / expected tokens;
- `program.lst` -- human-readable disassembly.

The sequencer fetches descriptors at `PC` (2 per 64-byte beat, 8-deep prefetch,
one fetch slot reserved per 64 stream beats when the queue is below 4), issues
**strictly in order** (the next descriptor issues when all units have retired;
GEMV retires when requant has drained) and stops at `HALT`. Per token the host
writes `TOK`, `POS`, `ROW_EN` and pulses `START`; all position-dependent values
(`n_from_pos`, `k_from_pos`, `len_from_pos`, KV addresses, RoPE row) derive in
hardware from `POS`.

Per Qwen decode token the program is ~1,518 descriptors = 48.6 KB (**estimate**,
< 0.01% of per-token traffic). Loops and branches are v1.1.

## Descriptor bit layout (256 bits)

| Bits | Field | Meaning |
|---|---|---|
| `[7:0]` | `opcode` | see table below |
| `[15:8]` | `flags` | per-opcode flags (VQUANT: width16/8, use_tracked, group, scale_mul) |
| `[23:16]` | `row_mask` | which of the `B_MAX` rows participate |
| `[24]` | `accumulate` | GEMV output `y = sat32(y + old)` (read-modify-write) |
| `[25]` | `unit_meta` | GEMV uses unit Sw and zero bias instead of reading meta (PV) |
| `[26]` | `n_from_pos` | N = POS+1 rounded up to the tile (scores GEMV) |
| `[27]` | `k_from_pos` | K = POS+1 (PV GEMV) |
| `[29:28]` | `out_mode` | 0 VSRAM, 1 ARGMAX, 2 ARGMAX+DUMP, 3 VSRAM+DUMP |
| `[30]` | `len_from_pos` | VSOFTMAX len = POS+1 (else imm32) |
| `[31]` | `track_absmax` | GEMV/V op tracks output absmax into `SREG[sreg_dst]` |
| `[63:32]` | `addr_a` | primary QMEM byte address (weights / table / gamma / K^T / V) |
| `[95:64]` | `addr_m` | meta base address (or the sqrt(d) sfloat constant for VRMSNORM, documented overload, low 24 bits) |
| `[119:96]` | `n` | output count (<= 2^24) |
| `[135:120]` | `k` | inner dimension (<= 65535) |
| `[151:136]` | `vs_src` | VSRAM element index of the source vector |
| `[167:152]` | `vs_dst` | VSRAM element index of the destination |
| `[183:168]` | `vs_aux` | auxiliary vector (VSILUMUL `u`; VQUANT group length) |
| `[191:184]` | `sreg_src` | scale register read |
| `[199:192]` | `sreg_dst` | scale register write |
| `[203:200]` | `src_row` | source activation row |
| `[207:204]` | `dst_row` | destination row |
| `[215:208]` | `sh0` (u8) | first shift or class: requant `s1`; VRMSNORM `FRAC_X` (the data-dependent `S1` is derived in hardware); VQUANT `FRAC_in`; VSILUMUL `FRAC_GU`; VSOFTMAX `FRAC_S` |
| `[223:216]` | `sh1` (i8) | second shift (requant `sbias`, VRMSNORM `G`, VSILUMUL `sh_h = 2 FRAC_GU - FRAC_H`) |
| `[255:224]` | `imm32` / `addr_c` | immediate (`eps_c`, sfloat constant, len) or dump address |

Compiler assertions: GEMV `vs_*` outputs are 8-aligned, `K <= 65535`,
`n <= 2^24`, everything fits in VSRAM, and the requant shift `S` is in `[0,63]`
for all reachable exponents.

## Opcode table

| Opcode | Name | Semantics |
|---|---|---|
| `0x00` / `0x01` | NOP / HALT | HALT sets `STATUS.done` and snapshots PERF |
| `0x10` | GEMV | for tile, for k: `beat = addr_a + (tile*K + k)*WB`; broadcast `A[k]` = low 16 bits of `vsram[vs_src+k]`; tile end swaps accumulators; requant with `meta[addr_m + (tile*WB+j)*8]` (or unit) and `SREG[sreg_src]`; out to `vsram[vs_dst + tile*WB + j]` (RMW if `accumulate`; absmax if `track_absmax`), the ARGMAX CSR, or DUMP at `addr_c`. Partial last tile: requant drains `min(WB, n - tile*WB)`. Used for QKV, o, gate\|up, down, LM head, scores (K^T, `n_from_pos`, `k=64`) and PV (V, `unit_meta`, `n=64`, `k_from_pos`) |
| `0x11` | EMBED | gather `K` bytes of row `TOK` from tiled table `addr_a`; the row scale from `meta[addr_m + TOK*8]` is `Sw`, `Sx = 1.0 = {2^15, -15}`, `acc = q << 24`, `sbias = -(FRAC_X + s1) + 24` with `8 <= s1 <= 24` |
| `0x20` | VRMSNORM | `vs_src -> vs_dst`; gamma int16 streamed from `addr_a`; `imm32 = eps_c`; `sh0 = FRAC_X`, `sh1 = G`; `addr_m` low 24 bits = sqrt(d) sfloat constant; absmax -> `SREG[sreg_dst]` |
| `0x21` | VQUANT | flags: width16/8, use_tracked, group (`vs_aux` = group length -> consecutive `SREG[sreg_dst..]`), scale_mul (`SREG *= sfloat imm32`); `sh0 = FRAC_in` |
| `0x22` | VROPE | in place at `vs_src`, `n = heads*64`, table row `addr_a + POS*128`, pairs `(i, i+32)`, shift 14 |
| `0x23` | VSILUMUL | `dst = silu(src) * aux`; `sh0 = FRAC_GU` (sigmoid index), `sh1 = 2 FRAC_GU - FRAC_H`; absmax -> SREG |
| `0x24` | VSOFTMAX | scores at `vs_src`, `len = POS+1` or `imm`; V-scale array at `addr_a`; `w` int16 to `vs_dst` (zeros beyond len); `SREG[sreg_dst] = sfloat(2^(1+e_max))`; `sh0 = FRAC_S` |
| `0x25` | VSUBC | `dst = sat32(src - const row streamed from addr_a)` |
| `0x30` | KVWRITE | transposed: 64 single-byte-strobe writes into `addr_a + (POS/WB)*64*WB + d*WB + POS%WB`; else one WB-byte row at `addr_a + POS*WB`; meta `{0, SREG[sreg_src]}` -> `addr_m + POS*8` |
| `0x31` | FENCE | wait for write-ack count == issued (auto-fence is implicit; explicit FENCE exists for step mode) |

Removed from v1 (v1.1): VCOPY, VADD/VSUB/VMOV, PERFMARK, last_row_only,
LOOP/JUMP.

## Binary formats

All multi-byte quantities are **little-endian**.

- **Descriptor**: 32 bytes. Bit `i` of the descriptor is bit `i % 8` of byte
  `i / 8`. So `opcode` is byte 0, `flags` byte 1, `addr_a` bytes 4..7, and
  `imm32`/`addr_c` bytes 28..31.
- **Program**: a contiguous array of descriptors starting at a 64-byte-aligned
  address; two descriptors per 64-byte beat; terminated by `HALT`.
- **VSRAM word**: 256 bits = 8 int32 elements. Element `j` of a word occupies
  bits `[32j+31 : 32j]`. Element index `e` lives in word `e / 8`, slot `e % 8`.
  GEMV activation reads take the low 16 bits of each element.
- **Per-channel / per-token meta** (8 bytes):
  `{ i32 bias_q, u16 m, i8 e, u8 pad = 0 }` -- `bias_q` at bytes 0..3 (signed,
  in the `FRAC_QKV` domain for QKV, the folded V bias for o_proj, else 0), the
  sfloat mantissa `m` at bytes 4..5 (in `[2^15, 2^16)`, or 0 for a padded
  channel/token), exponent `e` at byte 6 (signed), byte 7 zero. `m == 0` makes
  requant output 0 with no shift and no error count.
- **Weights**: int8, tiled `[N/WB][K][WB]`, zero-padded partial tiles allowed.
  The image is therefore built per `WB`.
- **Gamma**: int16 per element with one per-tensor exponent in `layout.json`.
- **RoPE table**: int16 Q1.14 `[pos][32 pairs]` = 128 bytes per position;
  checked in as `sw/quettos/tables/rope_theta1e6_2048.npy` and
  `rope_theta1e5_2048.npy`, sha256 in `layout.json`.
- **KV cache**: see `MEMORY_MAP.md`.

## CSR table

Generated from `sw/quettos/isa.py` into `rtl/qcore_csr_defs.svh` and
`sim/verilator/csr_defs.hpp`, with a pytest asserting the three agree. The register set is: `CTRL` (START/STEP/RESET), `STATUS`
(done/busy/step_halted), `PC`, `ROW_EN`, `TOK`, `POS`, `ARGMAX_TOK`,
`ARGMAX_VAL`, `PERF[16]` as 32-bit halves, `SAT_REQ`, `SAT_VPU`, `ERR_SHIFT`,
`ERR_BOUNDS`, `ISA_VERSION`.
