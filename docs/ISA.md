# Quettos Core ISA

`ISA_VERSION = 1`. The constant is defined in `sw/quettos/isa.py`, generated
into `rtl/qcore_csr_defs.svh` and `sim/verilator/csr_defs.hpp` by
`uv run quettos csr-defs`, written to `layout.json`, exposed in the
`ISA_VERSION` CSR and asserted by the harness at load time.

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
**strictly in order** (one descriptor is in flight at a time; a GEMV retires
when the requant has drained and the weight stream is idle) and stops at
`HALT`, or earlier on a fault (program end, below). `CTRL.STEP` runs exactly
one descriptor and reports `STEP_HALTED` once its writes are acknowledged, so
the host can read every register, and the memory the descriptor wrote, between
descriptors (`docs/RTL.md` 3.5, step mode). Per token the host writes `TOK`,
`POS`, `ROW_EN` and pulses `START`; all position-dependent values
(`n_from_pos`, `k_from_pos`, `len_from_pos`, KV addresses, RoPE row) derive in
hardware from `POS`. The programs carry no `FENCE`: the hardware fences every
memory-reading descriptor, and the descriptor prefetch with them, while KV or
dump writes are outstanding. The fence orders the bus; a descriptor already
prefetched is not read again, so a program keeps writes to its own descriptor
bytes outside the prefetch window (`docs/RTL.md` 3.4).

A decode program is `1 + L (16 + 2 KV + 3 H) + 4` descriptors (`L` layers,
`KV` KV heads, `H` query heads), the prefill program three fewer: Qwen2.5-0.5B
1,493 / 1,490 descriptors (47,776 / 47,680 B), SmolLM2-135M 1,475 / 1,472
(47,200 / 47,104 B), under 0.01% of the weight bytes a token streams. Loops
and branches are v1.1.

## Descriptor bit layout (256 bits)

| Bits | Field | Meaning |
|---|---|---|
| `[7:0]` | `opcode` | see table below |
| `[15:8]` | `flags` | per-opcode flags (VQUANT: `W8`, `USE_TRACKED`, `GROUP`, `SCALE_MUL`; KVWRITE: `TRANSPOSED`) |
| `[23:16]` | `row_mask` | which of the `B_MAX` rows participate |
| `[24]` | `accumulate` | GEMV output `y = sat32(y + old)` (read-modify-write) |
| `[25]` | `unit_meta` | GEMV uses unit Sw and zero bias instead of reading meta (PV) |
| `[26]` | `n_from_pos` | N = POS+1 rounded up to the tile (scores GEMV) |
| `[27]` | `k_from_pos` | K = POS+1 (PV GEMV) |
| `[29:28]` | `out_mode` | 0 VSRAM, 1 ARGMAX, 2 ARGMAX+DUMP, 3 VSRAM+DUMP |
| `[30]` | `len_from_pos` | VSOFTMAX len = POS+1 (else imm32) |
| `[31]` | `track_absmax` | GEMV/V op writes its output absmax into `SREG[sreg_dst]` |
| `[63:32]` | `addr_a` | primary QMEM byte address (weights / table / gamma / K^T / V) |
| `[95:64]` | `addr_m` | meta base address (or the sqrt(d) sfloat constant for VRMSNORM, documented overload, low 24 bits) |
| `[119:96]` | `n` | output count (`< 2^24`); the capacity with `n_from_pos` |
| `[135:120]` | `k` | inner dimension and tile stride (`< 2^16`); the capacity with `k_from_pos`; KVWRITE: the token capacity |
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

### Field conventions

Signedness and shifts:

- `sh1` is the one signed field (two's complement i8); every other field is
  unsigned.
- Shift amounts are GEMV / EMBED `sh0` (`s1`), VRMSNORM `sh1` (`G`) and
  VSILUMUL `sh1` (`sh_h`). Each lies in `[0, 63]`; a value outside is clamped
  into the window and counted once per output element in `ERR_SHIFT`, exactly
  like a requant stage-2 shift `S` out of range. The compiler's helpers reject
  such values (`sh0` on EMBED additionally lies in `[8, 24]`, the
  `embed_dequant` window).
- Class fields (`sh0` of VRMSNORM, VQUANT, VSILUMUL and VSOFTMAX) carry a
  `FRAC` in `[0, 30]` (`[13, 30]` on VSILUMUL, `[16, 30]` on VSOFTMAX). The
  compiler's helpers and the quantizer enforce these ranges; the hardware does
  not check them.
- `flags` bits: VQUANT bit 0 `W8` (int8 output; clear = int16), bit 1
  `USE_TRACKED`, bit 2 `GROUP`, bit 3 `SCALE_MUL`; KVWRITE bit 0 `TRANSPOSED`
  (K^T byte scatter; clear = V tiles). `GROUP` together with `USE_TRACKED` is
  reserved (the simulator rejects it).
- Fields an opcode does not use are written as zero and ignored by the
  hardware; unused `flags` bits likewise. EMBED reads `k` and ignores `n` (the
  helper writes `n = k`); VSOFTMAX with `len_from_pos` ignores `imm32`.

Rows (`B_MAX` activation rows share one weight stream):

- Every row `r < B_MAX` owns a VSRAM of `VSRAM_WORDS` words and an SREG bank
  of 32 x 32 b. Row `r` participates in a descriptor when
  `row_mask[r] & ROW_EN[r]` is set; bits at or above `B_MAX` are ignored. A
  participating row reads VSRAM / SREG row `src_row + r` and writes row
  `dst_row + r`; both lie below `B_MAX` (a compiler assertion; the simulator
  rejects a descriptor that breaks it, and the hardware halts it with
  `FAULT = ROW` rather than address a bank that does not exist).
- Rows execute in ascending order. A descriptor whose participating set is
  empty retires as a NOP: `DESCRIPTORS` counts it and nothing else changes (no
  memory traffic, no VSRAM, SREG, KV, ARGMAX or counter update).
- A GEMV or EMBED whose executed `N` or `K` is zero, and a V op with `n == 0`,
  retire the same way. The `ERR_BOUNDS` events of the POS derivation are still
  counted, since they describe the descriptor and not its work.
- `ARGMAX_TOK` / `ARGMAX_VAL` hold the result of the highest-numbered
  participating row; a DUMP writes row `r`'s outputs at `addr_c + r * 4 * N`.
  The `SAT_*` and `ERR_*` counters are shared by all rows.
- NOP, HALT and FENCE ignore `row_mask`. v1 programs write `row_mask = 1`,
  `src_row = dst_row = 0` and the host writes `ROW_EN = 1`; the row dimension
  is exercised at block level on the tiny configuration (`B_MAX = 2`).

Addresses and POS-derived values:

- GEMV beat `k` of tile `t` is at `addr_a + (t * k_field + k) * WB`: the `k`
  field is the tile stride and the number of beats streamed per tile, except
  with `k_from_pos`, where it is the stride (capacity) and `K = POS + 1` beats
  are streamed. One address generator therefore serves the weights, the K^T
  tiles (tokens as channels, `k = 64`) and the V tiles (dimensions as channels,
  `k = MAX_CTX`).
- `n_from_pos` (scores GEMV): `N = POS + 1` rounded up to a multiple of `WB`
  and capped at the capacity `n`; the rounding is silent, and only
  `POS + 1 > n` counts in `ERR_BOUNDS`. `k_from_pos` (PV GEMV): `K = POS + 1`
  capped at `k`, counted when above it. `len_from_pos` (VSOFTMAX):
  `len = POS + 1`; an immediate `len` or a derived one outside `[1, n]` is
  clamped into it and counted.
- KVWRITE writes nothing and counts in `ERR_BOUNDS` when `POS >= k` (the token
  capacity of the region).
- sfloat immediates: `m` in bits `[15:0]`, `e` as an i8 in bits `[23:16]`,
  bits `[31:24]` zero. VQUANT `scale_mul` travels in `imm32`, VRMSNORM
  `sqrt(d)` in `addr_m`.
- A DUMP (`out_mode` 2 or 3) writes the `N` int32 requant outputs
  little-endian at `addr_c + 4 * i` for `i < N`: exactly the elements the
  partial-tile drain produces (no padded channels), `N` the executed count
  (tile-rounded with `n_from_pos`, as in VSRAM mode). `addr_c` is a non-zero
  multiple of 64; the write is issued as full `WB`-byte beats with a
  byte-strobed final beat, so `WR_BEATS` counts `ceil(4N / WB)` per dump. The
  image reserves the dump region.

On-chip ranges:

- A VSRAM range that runs past `VSRAM_WORDS * 8` elements counts once in
  `ERR_BOUNDS` per read or write operand; the missing elements read as 0 and
  the writes to them are dropped, everything else executes.
- An SREG index at or above 32 counts once in `ERR_BOUNDS` per access; the
  read returns the zero scale (or absmax 0) and the write is dropped.
- SREG entries hold either an sfloat (`VQUANT` scales, `SREG_out`) or a tracked
  absmax (a non-negative int32). `track_absmax` overwrites `SREG[sreg_dst]`
  with the absmax of this descriptor's outputs. A VQUANT with `USE_TRACKED`
  reads the absmax from `SREG[sreg_src]` and produces exactly the values of a
  VQUANT that scans its data, so the two paths are interchangeable and
  bit-identical; the compiler uses `USE_TRACKED` after VRMSNORM and VSILUMUL
  (whose absmax is tracked into `SREG[0]`) and lets the q, K, V and context
  VQUANTs scan.

Program end:

- `HALT` sets `STATUS.DONE` and snapshots the PERF counters. It also waits for
  every issued write to be acknowledged, so `DONE` means the KV and dump bytes
  of the program have landed.
- A descriptor the hardware cannot execute stops the program the way `HALT`
  does, including that wait -- `STATUS.DONE` on the first cycle every issued
  write is acknowledged, PERF snapshotted, nothing further issued -- and in
  addition sets `STATUS.ERR` with the reason in `STATUS.FAULT`, the opcode byte
  in `STATUS.FAULT_OP` and `PC` left on the descriptor that faulted. Nothing of
  that descriptor executes and none of its work is counted. `DONE` therefore
  carries the same guarantee on the fault path as on the `HALT` path.
- `CTRL.ABORT` ends a run the same way: issue stops on the cycle the write is
  seen, the descriptor in flight retires, and `DONE` follows once every issued
  write is acknowledged.

| `FAULT` | Name | Raised when | `FAULT_OP` |
|---|---|---|---|
| 0 | `NONE` | no fault; the value while a program runs | 0 |
| 1 | `OPCODE` | the opcode byte is none of the twelve (`isa.opcode_of` / `isa.is_opcode` probe it before decoding), or it names a unit the build does not carry -- a `qcore_top` built without `qcore_vpu_top` refuses the six vector opcodes this way | that byte |
| 2 | `ROW` | a participating row's `src_row + r` or `dst_row + r` is at or above `B_MAX` | the opcode |
| 3 | `PC_ALIGN` | `START` or `STEP` with a `PC` that is not a multiple of 32 | 0 |

  `isa.status_word()` builds the word and `isa.status_fault()` reads the two
  fields back; `quettos.isa_sim` reports the `OPCODE` fault the same way.

Compiler assertions: GEMV / EMBED `vs_dst` is 8-aligned, `K < 2^16`,
`n < 2^24`, every VSRAM range fits the map, every SREG index is below 32, and
the requant shift `S` is in `[0, 63]` for all reachable exponents.

## Opcode table

| Opcode | Name | Semantics |
|---|---|---|
| `0x00` / `0x01` | NOP / HALT | HALT sets `STATUS.DONE` and snapshots PERF |
| `0x10` | GEMV | for tile, for k: `beat = addr_a + (tile*k_field + k)*WB`; broadcast `A[k]` = low 16 bits of `vsram[vs_src+k]`; tile end swaps accumulators; requant with `meta[addr_m + (tile*WB+j)*8]` (or unit) and `SREG[sreg_src]`; out to `vsram[vs_dst + tile*WB + j]` (RMW if `accumulate`; absmax if `track_absmax`), the ARGMAX CSR, or DUMP at `addr_c`. Partial last tile: requant drains `min(WB, n - tile*WB)`. Used for QKV, o, gate\|up, down, LM head, scores (K^T, `n_from_pos`, `k=64`) and PV (V, `unit_meta`, `n=64`, `k = MAX_CTX`, `k_from_pos`) |
| `0x11` | EMBED | gather `k` bytes of row `TOK` from tiled table `addr_a` (byte `(TOK/WB)*k*WB + i*WB + TOK%WB` for `i < k`); the row scale from `meta[addr_m + TOK*8]` is `Sw`, `Sx = 1.0 = {2^15, -15}`, `acc = q << 24`, `sbias = -(FRAC_X + s1) + 24` with `8 <= s1 <= 24`; `n` is written equal to `k` |
| `0x20` | VRMSNORM | `vs_src -> vs_dst`, `n` elements; gamma int16 streamed from `addr_a`; `imm32 = eps_c`; `sh0 = FRAC_X`, `sh1 = G` (shift); `addr_m` low 24 bits = sqrt(d) sfloat constant; absmax -> `SREG[sreg_dst]` |
| `0x21` | VQUANT | `vs_src -> vs_dst`, `n` elements; flags `W8` (int8, else int16), `USE_TRACKED` (absmax from `SREG[sreg_src]`), `GROUP` (`vs_aux` = group length -> consecutive `SREG[sreg_dst..]`), `SCALE_MUL` (`SREG *= sfloat imm32`); `sh0 = FRAC_in` |
| `0x22` | VROPE | in place at `vs_src` of row `src_row + r` (`dst_row` ignored), `n = heads*64` (the compiler covers the contiguous q and k heads with one VROPE), table row `addr_a + POS*128`, pairs `(i, i+32)`, shift 14 |
| `0x23` | VSILUMUL | `dst = silu(src) * aux`; `sh0 = FRAC_GU` (sigmoid index), `sh1 = 2 FRAC_GU - FRAC_H` (shift); absmax -> SREG |
| `0x24` | VSOFTMAX | scores at `vs_src`, `len = POS+1` or `imm`, clamped into `[1, n]`; V-scale meta at `addr_a`; `w` int16 to `vs_dst` (zeros from `len` to `n`); `SREG[sreg_dst] = sfloat(2^(1+e_max))`; `sh0 = FRAC_S` |
| `0x25` | VSUBC | `dst = sat32(src - const row streamed from addr_a)`, `n` elements |
| `0x30` | KVWRITE | the 64 int8 values at `vs_src` (low byte of each element), `k` = token capacity; flag `TRANSPOSED`: 64 single-byte-strobe writes into `addr_a + (POS/WB)*64*WB + d*WB + POS%WB`; else `ceil(64/WB)` `WB`-byte beats, tile `t` at `addr_a + (t*k + POS)*WB` holding dims `t*WB ..` zero-padded; meta `{0, SREG[sreg_src]}` -> `addr_m + POS*8`; `POS >= k` writes nothing and counts in `ERR_BOUNDS` |
| `0x31` | FENCE | wait for write-ack count == issued (the auto-fence ahead of every memory-reading descriptor is implicit; an explicit FENCE parks a program at a point where every issued write is acknowledged) |

Removed from v1 (v1.1): VCOPY, VADD/VSUB/VMOV, PERFMARK, last_row_only,
LOOP/JUMP.

## Binary formats

All multi-byte quantities are **little-endian**.

- **Descriptor**: 32 bytes. Bit `i` of the descriptor is bit `i % 8` of byte
  `i / 8`. So `opcode` is byte 0, `flags` byte 1, `addr_a` bytes 4..7, and
  `imm32`/`addr_c` bytes 28..31.
- **Program**: a contiguous array of descriptors starting at a 64-byte-aligned
  address; two descriptors per 64-byte beat; terminated by `HALT`.
- **Listing** (`program.lst`, `isa.disassemble`): one line per descriptor,
  `index  OPCODE field=value ...`, showing only the fields that differ from
  the all-zero descriptor with `row_mask = 1`: addresses in hex, one-bit
  fields by name, `flags` by name, `sh1` signed, and `imm32` by its meaning
  (`eps_c`, `len`, `scale_mul={m,e}`, `addr_c`; VRMSNORM's `addr_m` as
  `sqrt_d={m,e}`). An immediate that is not a well-formed sfloat is listed as
  the raw field in hex, so every decodable program has a listing.
- **VSRAM word**: 256 bits = 8 int32 elements. Element `j` of a word occupies
  bits `[32j+31 : 32j]`. Element index `e` lives in word `e / 8`, slot `e % 8`.
  GEMV activation reads take the low 16 bits of each element; KVWRITE takes
  the low 8.
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
- **Dump**: `N` int32 little-endian at `addr_c` (field conventions above).
- **KV cache**: see `MEMORY_MAP.md`.

## Bring-up programs

Two short programs over a compiled image bring a machine up: one for a machine
with the vector unit, one for the GEMV and EMBED units alone.
Both are small enough to read end to end in a waveform and both are
self-checking on a tied-embedding model: v1 ties the embedding and the LM head,
so the largest logit of the embedding row of token `t` against the embedding
matrix is row `t` itself. `ARGMAX_TOK == TOK` for every token of the model, and
`SAT_REQ`, `SAT_VPU`, `ERR_SHIFT` and `ERR_BOUNDS` all read 0.

### Four descriptors: the whole decode path

`bringup_program` in `sw/tests/test_isa.py` assembles it from the model's own
`decode.prog` and runs it on `quettos.isa_sim` over every token of a random
tiny model from `quettos.synthetic`, so `ARGMAX_TOK`, `ARGMAX_VAL`,
`DESCRIPTORS`, `MACS` and `WT_BYTES` have a reference.

| # | Descriptor | Exercises |
|---|---|---|
| 0 | `EMBED` -- descriptor 0 of `decode.prog` | fetch, dispatch, the TOK-gathered table read, the meta record, the requant, a VSRAM write |
| 1 | `VQUANT` of the EMBED output into the activation slot | the vector unit, an SREG scale write |
| 2 | `GEMV` with `out_mode = ARGMAX` -- the LM-head descriptor of `decode.prog` with its `out_mode` replaced | the weight stream, the tiles, the MAC rows, the requant, the ARGMAX CSRs |
| 3 | `HALT` | the auto-fence, the PERF snapshot, `STATUS.DONE` |

The VQUANT earns its place: a GEMV takes its `Sx` from
`SREG[src_row + r][sreg_src]`, the SREG banks have no reset, and a VQUANT is
what writes a scale there, so a program that reaches a GEMV without one has no
defined activation scale.

### Three descriptors: the GEMV and EMBED units

`quettos.compiler.build_bringup` assembles it and `sw/quettos/compare.py` runs
it on `qcore_top` through the Verilator harness and on `quettos.isa_sim` at the
same time, comparing every VSRAM element, SREG word, dumped logit, CSR and PERF
counter after each descriptor (`make bringup`, `make bringup-sweep`,
`sw/tests/test_bringup.py`).

| # | Descriptor | Exercises |
|---|---|---|
| 0 | `EMBED` -- descriptor 0 of `decode.prog`, writing the LM head's activation slot, with `sh1` raised by `q` | fetch, dispatch, the TOK-gathered table read, the meta record, the requant, a VSRAM write |
| 1 | `GEMV` in `ARGMAX_DUMP` mode -- the LM-head descriptor of `decode.prog` | the weight stream, the tiles, the MAC rows, the requant, the ARGMAX CSRs and a dump of all `N` int32 logits |
| 2 | `HALT` | the auto-fence, the PERF snapshot, `STATUS.DONE` |

The activation scale is the host's here: `SREG[0][sreg_src] = 2^-(FRAC_X + q)`,
written before `START`, paired with the EMBED output shift raised by the same
`q = max(0, bitlen(absmax) - 15)`. That is the scale a `VQUANT` would have
written, and it keeps the gathered row inside the int16 window a GEMV
activation is read through. Both models are given the same scale; every other
input is the compiled image. `ARGMAX_DUMP` puts every logit in memory, so the
comparison covers the whole output vector and not only its argmax.

Format, both programs: the same 32-byte descriptors as any program, assembled
with `isa.assemble` and placed in the 1 MB program window of `image.bin` after
`prefill.prog`, at `programs.prefill.addr + programs.prefill.size` rounded up
to 64 from `layout.json`; nothing else in the image moves. The three-descriptor
program's dump region follows the descriptors at the next 64-byte boundary,
inside the same window. The host writes the program address to `PC`, the token
under test to `TOK`, `POS = 0` and `ROW_EN = 1`, then pulses `START` and waits
for `STATUS.DONE`.

## CSR table

The host sees 64 32-bit words (256 bytes). `sw/quettos/isa.py` is the source;
`uv run quettos csr-defs` generates `rtl/qcore_csr_defs.svh` (one
`` `define QCORE_<NAME> <value> `` per constant, include-guarded; a module or
package restates the constants it uses as `localparam`, which keeps the file
clean under `verilator -Wall`) and `sim/verilator/csr_defs.hpp`
(`constexpr uint32_t <NAME>` in namespace `qcore`) from it; `--check` diffs
them and `sw/tests/test_isa.py` asserts the three agree and runs the include
through the three lint parsers. The same table carries the opcodes, output
modes, flag masks and descriptor field positions (`QCORE_DESC_<FIELD>_LSB` /
`_W`). Access: `rw` host read/write, `ro` read-only, `w1p` write-one-to-pulse (a 1
written to a bit acts once; reads return 0), `w1c` read plus write-one-to-clear
(a 1 written to a bit clears it).

| Word | Name | Access | Contents |
|---|---|---|---|
| 0 | `CTRL` | w1p | bit 0 `START`: clear the counters and run from `PC` to HALT; bit 1 `STEP`: execute the descriptor at `PC`, then set `STEP_HALTED` once every write it issued is acknowledged (`DONE` if it was HALT); bit 2 `ABORT`: stop issuing, let the descriptor in flight retire, set `DONE` once every issued write is acknowledged. A descriptor already popped but not issued does not run, and `PC` is left on it. `START` and `STEP` act only while `BUSY` is low and `ABORT` only while it is high, and `START` wins over `STEP` in one write, so a write produces at most one pulse. Reads return 0 |
| 1 | `STATUS` | w1c | bit 0 `DONE` (set by HALT, by ABORT and by a fault), bit 1 `BUSY` (a descriptor is in flight or queued), bit 2 `STEP_HALTED` (a stepped descriptor finished and its writes are acknowledged), bit 3 `ERR` (a fault stopped the program), bits `[7:4]` `FAULT` (the code below), bits `[15:8]` `FAULT_OP` (the opcode byte that faulted). `START` and `STEP` clear `DONE`, `STEP_HALTED`, `ERR` and both fault fields, and so does writing a one to a bit -- a one in `ERR` clears `FAULT` and `FAULT_OP` with it. `BUSY` and the reserved bits ignore writes |
| 2 | `PC` | rw | byte address of the next descriptor; written 32-byte aligned before `START` (64-byte aligned at the start of a program), advanced by 32 per retired descriptor, left where it is by a fault |
| 3 | `ROW_EN` | rw | bit `r` enables activation row `r` for the token |
| 4 | `TOK` | rw | token id gathered by EMBED |
| 5 | `POS` | rw | position of the token; source of every POS-derived value |
| 6 | `ARGMAX_TOK` | ro | index of the largest output of the most recent ARGMAX-mode GEMV (strict `>`, so ties resolve to the lowest id) |
| 7 | `ARGMAX_VAL` | ro | that output, int32 |
| 8 | `SAT_REQ` | ro | requant `sat40` / `sat32` events since `START` |
| 9 | `SAT_VPU` | ro | vector-unit `sat32` events since `START` (clips are not saturations) |
| 10 | `ERR_SHIFT` | ro | shift amounts clamped into `[0, 63]` since `START`, one per output element: requant stage-2 `S`, VRMSNORM `S1`, and the descriptor shift fields `s1`, `G`, `sh_h` |
| 11 | `ERR_BOUNDS` | ro | since `START`: POS-derived values above their capacity field, VSOFTMAX `len` outside `[1, n]`, KVWRITE at `POS >= k`, VSRAM operand ranges past the end, SREG indices at or above 32 |
| 12 | `ISA_VERSION` | ro | the constant `ISA_VERSION` |
| 13-15 | | | zero |
| 16 + 2i | `PERF<i>_LO` | ro | bits `[31:0]` of `PERF[i]`, `i` in `0..15` |
| 17 + 2i | `PERF<i>_HI` | ro | bits `[63:32]` of `PERF[i]` |
| 48-63 | | | zero |

The four `rw` registers take a host write only while `BUSY` is low, so the
values a descriptor sees cannot change under it; the host reads a register back
to confirm. `START` clears the PERF, SAT and ERR counters; `STEP` clears none of
them, so a stepped descriptor's contribution is the difference between two
consecutive steps. The PERF counters are snapshotted into the halves above on a
HALT retire and at the end of a STEP, an ABORT or a fault, so they read stable
while the host works between descriptors or prepares the next token. `PERF` indices:

| `i` | Name | Counts |
|---|---|---|
| 0 | `CYCLES` | cycles from `START` to HALT |
| 1 | `BUSY` | cycles with a descriptor in flight; equals the sum of indices 2-7, which are exclusive |
| 2 | `MAC_ACTIVE` | cycles a weight beat entered the MAC array |
| 3 | `STALL_MEM` | GEMV / EMBED cycles waiting for read data |
| 4 | `STALL_VPU` | cycles a vector op occupied the unit |
| 5 | `STALL_KV` | cycles in KVWRITE or the auto-fence on outstanding writes |
| 6 | `STALL_SEQ` | cycles waiting for descriptor fetch or dispatch |
| 7 | `STALL_DRAIN` | cycles draining requant after the last beat of a GEMV |
| 8 | `RD_BEATS` | read beats returned by QMEM (weight-port busy cycles) |
| 9 | `RD_BYTES` | bytes in those beats |
| 10 | `WT_BYTES` | weight bytes consumed, **per participating row**: the tile bytes and the padded meta records (`ceil(N/WB)*WB*8` B) of every GEMV without POS-derived dimensions, plus the `k` table bytes and the 8 meta bytes an EMBED gathers, each multiplied by `popcount(rows)`; gammas, constants and the RoPE row are not counted. With one row this equals `traffic.<program>.wt_bytes` in `layout.json` |
| 11 | `WR_BEATS` | write beats issued (KVWRITE, logit dump) |
| 12 | `WR_BYTES` | bytes in those beats |
| 13 | `MACS` | multiply-accumulates issued, **per participating row**: `popcount(rows) * ceil(N/WB) * WB * K` for every GEMV, padded lanes and the attention GEMVs included. With one row, at position `POS`, this is `traffic.<program>.macs + head_layers * (ceil((POS+1)/WB) * scores_macs_per_tile + (POS+1) * pv_macs_per_token)` with the three constants from `traffic.attention` in `layout.json` |
| 14 | `DESCRIPTORS` | descriptors retired |
| 15 | `FETCH_BEATS` | descriptor fetch beats |

Buckets 2 to 7 are exclusive: every cycle with `BUSY` high belongs to exactly
one of them, by the priority in `docs/RTL.md` 3.5. The harness asserts
`BUSY = MAC_ACTIVE + STALL_MEM + STALL_VPU + STALL_KV + STALL_SEQ +
STALL_DRAIN` and cross-checks the byte counters against its memory model. The
ISA simulator counts `DESCRIPTORS`, `MACS` and `WT_BYTES`; the cycle and beat
counters are the RTL's.
