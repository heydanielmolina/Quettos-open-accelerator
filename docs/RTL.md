# RTL interface specification

Module-level contract of the v1 hardware: every port, the handshake on every
interface, and the cycle-level timing each module guarantees. The value-level
semantics are `docs/ISA.md` and `sw/quettos/isa_sim.py`; the arithmetic is
`sw/quettos/numerics.py`; layouts are `docs/MEMORY_MAP.md`. Where this
document names a width it is the width the RTL uses.

## 1. Conventions

| Item | Rule |
|---|---|
| Clock, reset | one `clk`; `rst` synchronous, active high, held at least 2 cycles; every register with a reset value is reset, memories and datapath pipelines are not |
| Parameters | `qcore_top` is the root; every module takes what it needs from it. `WB` bytes per beat and MAC lanes (16 / 64 / 128), `B_MAX` activation rows (1 / 2), `VL` vector lanes (2 / 4), `VSRAM_WORDS` (2048 / 4096), `FIFO_BEATS` (128), `ACC_W` (40), `META_FIFO_BEATS` (16), `VPU_FIFO_BEATS` (16), `MAX_BURST` (64), `DQ_DEPTH` (8), the `ROM_FILE_<TABLE>` image paths (untyped parameters with an empty default, set by the build; `qcore_top` carries `SIGMOID`, `RSQRT` and `RECIP`, and `EXP2` arrives with VSOFTMAX) |
| Derived widths | `DW = WB*8` beat bits; `NG = WB/8` lane groups; `AW = $clog2(VSRAM_WORDS)` word address; `TW = 20` tile index (`n < 2^24`, `WB >= 16`); `NVW = $clog2(WB)+1` channels in a tile (`1..WB`); `RW = 4` physical row index; element index arithmetic is 17 bits |
| valid/ready | a transfer happens at a rising edge where `valid` and `ready` are both high. `valid` and the payload hold until the transfer. `valid` never depends combinationally on `ready`; every `ready` is a function of the receiver's registers and, where a module's section says so, of one input of the same handshake chain (the arbiter's requester readies follow `rd_req_ready` / `wr_ready`, section 3.6; a row's `ws_ready` on a tile-end beat follows `acc_ready`, 3.8; the requant's `acc_ready` in a tile's final-read cycle follows `meta_valid`, 3.10). No interface has a combinational path from an input valid to an output valid, and no ready chain closes a loop |
| Pulses | one cycle wide, driven from a register; the exception is `ev_beat` / `gemv_beat`, the beat accept itself (3.8) |
| Command bundle | held stable by the dispatcher from the issue pulse until the unit's `done` pulse; units latch what they need on the issue pulse |
| Beat data | byte `j` of a beat is bits `[8j+7:8j]`; int8 lane `j` of a weight beat is output channel `tile*WB + j`; a QMEM beat address is a byte address |
| sfloat on wires | `m[15:0]` and `e[7:0]` (i8) as two signals; `m == 0` is the zero scale. Exponent arithmetic is i8 two's complement |
| SREG word | 32 bits: `{8'd0, e[7:0], m[15:0]}` for a scale (same encoding as an sfloat descriptor immediate), a tracked absmax as a u32 (`2^31` is representable) |
| Meta record on wires | 56 bits `{e[7:0], m[15:0], bias_q[31:0]}`; the QMEM pad byte is dropped by `qcore_stream_ctrl` |
| Rows | physical MAC row `r` reads VSRAM / SREG bank `src_row + r` and writes bank `dst_row + r` through the crossbar in `qcore_top`; rows execute ascending; `cmd_rows[r]` marks participation |

## 2. Buses

### 2.1 QMEM (external memory)

| Channel | Signals (direction from `qcore_top`) | Protocol |
|---|---|---|
| `rd_req` | `rd_req_valid` o, `rd_req_ready` i, `rd_req_addr[31:0]` o, `rd_req_len[7:0]` o, `rd_req_tag[3:0]` o | valid/ready. `len` = beats in the burst, `1..255`; v1 masters issue at most `MAX_BURST` = 64. Beat `i` of a burst is the `WB` bytes at `addr + i*WB` |
| `rd_data` | `rd_data_valid` i, `rd_data[DW-1:0]` i, `rd_data_tag[3:0]` i, `rd_data_last` i | valid only: the core accepts every beat the cycle it is presented (each sink reserves buffer space before its request is issued). `tag` mirrors the request; `last` marks the final beat of its burst |
| `wr` | `wr_valid` o, `wr_ready` i, `wr_addr[31:0]` o, `wr_data[DW-1:0]` o, `wr_strb[WB-1:0]` o | valid/ready; byte `j` is written to `addr + j` when `strb[j]` is set |
| `wr_ack` | `wr_ack` i | one pulse per accepted write, in order of acceptance |

Memory-model contract (C++ harness, `sim/cocotb/qc_qmem.py`): beats return in
request order across all tags, one beat per cycle (`--bw-div N`: one per `N`
cycles); the first beat of a burst arrives `LAT` cycles after the request
transfer (default 32); `rd_req_ready` drops while 64 beats are in flight;
`wr_ready` is high; the ack of a write comes `LAT` cycles after its transfer.
No alignment is required of a beat address: the byte-scatter, meta and dump
writes and the gamma / centering-row / V-scale reads address exact bytes.

Read tags (`qcore_pkg`): `TAG_FETCH` 0 (descriptor beats), `TAG_WEIGHT` 1
(weight / EMBED tile beats), `TAG_META` 2 (per-channel meta), `TAG_VPU` 3
(gamma, RoPE row, centering row, V-scale meta). `qcore_pkg::rd_route(tag)`
is the one-hot sink select.

### 2.2 Descriptor issue (dispatcher to units)

The dispatcher decodes the raw descriptor once and holds a decoded bundle;
each unit connects the subset listed in its port table. All `cmd_*` signals
are outputs of `qcore_seq_dispatch`.

| Signal | Width | Meaning |
|---|---|---|
| `cmd_valid_gemv`, `cmd_valid_vpu`, `cmd_valid_kv` | 1 | issue pulse to the GEMV group (`stream_ctrl`, rows, `requant`), the VPU, the KV writer |
| `done_gemv`, `done_vpu`, `done_kv` | 1 (inputs) | completion pulse from `requant`, `vpu_top`, `kv_writer` |
| `cmd_op` | 8 | opcode |
| `cmd_out_mode` | 2 | GEMV / EMBED output mode |
| `cmd_accumulate`, `cmd_unit_meta`, `cmd_track_absmax` | 1 | the descriptor bits |
| `cmd_vq_w8`, `cmd_vq_use_tracked`, `cmd_vq_group`, `cmd_vq_scale_mul`, `cmd_kv_transposed` | 1 | the defined `flags` bits; undefined bits are never read |
| `cmd_addr_a`, `cmd_addr_m`, `cmd_imm32` | 32 | descriptor fields (`imm32` is `addr_c`, `eps_c`, `len` or `scale_mul` by opcode) |
| `cmd_n` | 24 | executed `N`: `n_from_pos` applied and capped; EMBED: the `k` field |
| `cmd_k` | 16 | executed `K`: `k_from_pos` applied and capped |
| `cmd_k_stride` | 16 | the `k` field (tile stride; KVWRITE capacity) |
| `cmd_len` | 24 | VSOFTMAX `len` clamped into `[1, n]` |
| `cmd_vs_src`, `cmd_vs_dst`, `cmd_vs_aux` | 16 | element indices |
| `cmd_sreg_dst` | 8 | SREG write index |
| `cmd_src_row`, `cmd_dst_row` | 4 | row bases |
| `cmd_sh0` | 8 | `s1` / `FRAC` class |
| `cmd_sh1` | 8 | i8: `sbias` / `G` / `sh_h` |
| `cmd_sqrt_m`, `cmd_sqrt_e` | 16, 8 | VRMSNORM `sqrt(d)` from `addr_m[23:0]` |
| `cmd_rows` | `B_MAX` | participating rows `row_mask & ROW_EN`, bits at or above `B_MAX` dropped |
| `cmd_pos`, `cmd_tok` | 32 | the CSRs at issue |
| `cmd_sx_m`, `cmd_sx_e` | `B_MAX*16`, `B_MAX*8` | per row `r`: `SREG[src_row+r][sreg_src]` as an sfloat, read at issue (GEMV, KVWRITE); `{2^15, -15}` for EMBED |
| `cmd_sreg_u32` | `B_MAX*32` | the same word as a u32 (VQUANT `USE_TRACKED` absmax) |

Issue timing: the dispatcher pops a descriptor, decodes it (1 cycle), waits
for the auto-fence when the opcode reads QMEM, reads one SREG word per
participating row (2 cycles each), then pulses `cmd_valid_*`. Exactly one
descriptor is in flight: the next descriptor is popped only after the current
one has retired, and its issue pulse follows at least 3 cycles later. Every
`cmd_*` field holds from the issue pulse to the retire.

| Opcode | Waits for before the issue | Issue pulse | Retires on |
|---|---|---|---|
| NOP | nothing | none | the decode cycle |
| HALT | `wr_idle` | none | the cycle `wr_idle` is high |
| FENCE | `wr_idle` | none | the cycle `wr_idle` is high |
| GEMV | `wr_idle`, then `SREG[src_row + r][sreg_src]` of every participating row | `cmd_valid_gemv` | `done_gemv` **and** `qcore_stream_ctrl.busy` low |
| EMBED | `wr_idle` | `cmd_valid_gemv` | `done_gemv` **and** `qcore_stream_ctrl.busy` low |
| VRMSNORM, VROPE, VSOFTMAX, VSUBC | `wr_idle` | `cmd_valid_vpu` | `done_vpu` |
| VQUANT | `SREG[src_row + r][sreg_src]` of every participating row, with `USE_TRACKED` and not `GROUP` | `cmd_valid_vpu` | `done_vpu` |
| VSILUMUL | nothing | `cmd_valid_vpu` | `done_vpu` |
| KVWRITE | `SREG[src_row + r][sreg_src]` of every participating row | `cmd_valid_kv` | `done_kv` |

Auto-fence: GEMV, EMBED, VRMSNORM, VROPE, VSOFTMAX and VSUBC read QMEM and wait
for `wr_idle`, and so do FENCE and HALT. The descriptor prefetch takes the same
fence through `fetch_hold` (3.4). No read of any kind therefore passes a KV or
dump write of an earlier descriptor, and `STATUS.DONE` means every write the
program issued has been acknowledged -- on the HALT, the fault and the ABORT
path alike (3.5). NOP, VQUANT, VSILUMUL and KVWRITE read no QMEM and never
wait.

A GEMV or EMBED retires only once `done_gemv` has pulsed and
`qcore_stream_ctrl.busy` is low: the padded meta beats of a partial last tile
can still be in flight after the requant's `done`, and the next descriptor may
be a V op that shares the arbiter. Those cycles count as `STALL_DRAIN`.

Zero work: the dispatcher retires a descriptor without an issue pulse when the
participating set `row_mask & ROW_EN` (bits at or above `B_MAX` dropped) is
empty, when a GEMV or EMBED has `N == 0` or `K == 0`, and when a V op has
`n == 0`. The rows, the requant and the VPU stay idle; `DESCRIPTORS` counts the
descriptor and nothing else changes, except that the decode cycle still raises
the `err_bounds_inc` of the POS derivation, which describes the descriptor
rather than its work. `sw/quettos/isa_sim.py` follows the same rule.

POS-derived fields: the dispatcher derives three scalar extents from `POS` and
hands the units `cmd_pos` and `cmd_tok`, the CSRs as they stood at the issue.

| Value | Rule | Reference in `sw/quettos/isa_sim.py` |
|---|---|---|
| `cmd_n` (GEMV) | `n_from_pos`: `min(roundup(POS + 1, WB), n)`; otherwise the `n` field | `gemv_dims(d, POS, WB)[0]` |
| `cmd_k` (GEMV) | `k_from_pos`: `min(POS + 1, k)`; otherwise the `k` field | `gemv_dims(d, POS, WB)[1]` |
| `cmd_n`, `cmd_k` (EMBED) | both the `k` field | `_exec_embed` |
| `cmd_len` (VSOFTMAX) | `len_from_pos ? POS + 1 : imm32`, clamped into `[1, n]` | `softmax_len(d, POS)` |
| `cmd_k_stride` | the raw `k` field: the GEMV tile stride, the KVWRITE token capacity | `_stream_gemv` (`k_cap`), `_exec_kvwrite` |
| `err_bounds_inc` | once per participating row per event: `POS + 1 > n` with `n_from_pos`, `POS + 1 > k` with `k_from_pos`, a clamped VSOFTMAX `len` | `gemv_dims(...)[2]`, `softmax_len(...)[1]` |

Every position-dependent address is formed in the unit that issues the request,
not in the dispatcher: `qcore_stream_ctrl` builds the EMBED table base
`addr_a + (TOK/WB)*K*WB` and record address `addr_m + TOK*8` (3.7),
`qcore_vpu_top` the RoPE row `addr_a + POS*128` (3.12), and
`qcore_kv_writer` the K^T, V and meta addresses (3.17).

### 2.3 Weight stream (`qcore_stream_ctrl` to every `qcore_row`)

| Signal | Width | Meaning |
|---|---|---|
| `ws_valid`, `ws_ready` | 1 | valid/ready; `ws_ready` is the AND of the rows' readies (`qcore_top`) |
| `ws_data` | `DW` | `WB` int8 weights of input `k` (GEMV) or `WB` gathered embedding bytes (EMBED) |
| `ws_k` | 16 | `k` within the tile, `0..K-1` (0 for an EMBED beat) |
| `ws_tile` | `TW` | tile index |
| `ws_tile_start`, `ws_tile_end` | 1 | `k == 0`, `k == K-1` (both set on an EMBED beat) |
| `ws_nvalid` | `NVW` | channels of this tile: `min(WB, N - tile*WB)` |
| `ws_last` | 1 | last beat of the descriptor |
| `ws_embed` | 1 | EMBED beat: the rows load `q << 24` instead of multiplying |

One beat per cycle whenever the weight FIFO holds data and the rows are ready.
`ws_valid` comes from the FIFO output register; a beat that arrived on
`rd_data` is presented at most 3 cycles later when the FIFO is empty, and the
first beat of a descriptor at least one cycle after the issue pulse. A row
learns of a transfer only through its own `ws_ready`, so the participating rows
of a descriptor run in lockstep (same beats, same `acc_ready`, same VSRAM
timing).

### 2.4 Accumulator handoff (`qcore_row` to `qcore_requant`)

| Signal | Width | Meaning |
|---|---|---|
| `acc_valid` | 1 | a finished tile sits on `acc_flat` (OR over rows; participating rows raise it together) |
| `acc_ready` | 1 | from `requant`: high when its reader is idle or reading the final channel of the previous tile this cycle (in that cycle it also follows `meta_valid` for a GEMV row fed from the meta stream) |
| `acc_flat` | `B_MAX*WB*ACC_W` | row `r`, channel `j` at bits `[(r*WB+j)*ACC_W +: ACC_W]`, the finished tile of every row |
| `acc_tile`, `acc_nvalid`, `acc_last` | `TW`, `NVW`, 1 | the tile's index, channel count, last-tile flag |

Timing: the beat with `ws_tile_end` is accepted only when `!acc_valid &&
acc_ready`. The cycle after it, the finished tile is on `acc_flat` and
`acc_valid` rises. The transfer
`acc_valid && acc_ready` starts the drain: `requant` reads channel `r*WB+j`
(participating rows ascending, `j < acc_nvalid`) at `handshake + 1 + i` for
the `i`-th element, `acc_ready` low from `handshake + 1` until the cycle of
the final read. `acc_flat` is stable from the cycle after a tile-end beat
until the cycle after the next one. Minimum tile period is therefore
`max(K, popcount(rows) * nvalid + 1)` cycles.

### 2.5 Meta side-stream (`qcore_stream_ctrl` to `qcore_requant`)

`meta_valid` / `meta_ready` / `meta_data[55:0]`, one record per transfer,
records in channel order (`tile*WB + j`), `nvalid` per tile (the stream
controller drops the records of a partial tile's padded channels), only for
descriptors without `unit_meta`. An EMBED pushes exactly one record (the
row's), requested before the tile's weight bursts so that the requant holds
`Sw` before the first pseudo-tile arrives. `requant` consumes one record per
drained channel of the first participating row and replays the tile's records
from a `WB`-record buffer for the other rows (GEMV), or latches the single
record from the issue on (EMBED); its reader stalls while `meta_valid` is low.

### 2.6 VSRAM ports and the crossbar

Every VSRAM instance `v` (one per row) has port A (read) and port B (read and
strobed write). During one descriptor each port has one owner; `qcore_top`
routes by the descriptor's row bases:

| Descriptor class | Port A of VSRAM `v` | Port B of VSRAM `v` |
|---|---|---|
| GEMV / EMBED | row `r = v - src_row` (activation words), when `cmd_rows[r]` | `requant` while `vsb_row + dst_row == v` (old-word reads for `accumulate`, output writes) |
| V ops | `vpu_top` port A while `cur_row + src_row == v` | `vpu_top` port B while `cur_row + (vsb_sel_dst ? dst_row : src_row) == v` |
| KVWRITE | idle | `kv_writer` while `vsb_row + src_row == v` (64-element read) |

Unrouted ports are idle (`en = 0`, `we = 0`). A port-A read and a port-B
write of the same word in the same cycle is a design error (asserted in
simulation); the units' read-ahead rules and the rule that a V op's destination
either coincides with its source or misses it entirely (3.12) keep it from
happening. A written word is readable on
either port from the next cycle. The requant never changes `vsb_row` in the
cycle after a read and never issues a read and a write in the same cycle, so
`qcore_top` may select `vsb_rdata` combinationally on the current `vsb_row`.

Bounds: each accessor compares element indices in 17 bits before truncating
to `AW`; a word at or past `VSRAM_WORDS` reads as 0 and is not written, and
the accessor raises one `err_bounds` pulse per operand range per row that
runs past `VSRAM_WORDS*8` elements.

### 2.7 SREG banks

One bank of 32 x 32-bit registers per row inside `qcore_row`; one read port
(dispatcher) and one write port (`requant` or `vpu_top`, muxed in
`qcore_top` by the descriptor class). Index ports are 8 bits: an index at or
above 32 reads 0, drops the write and raises the bank's `sreg_err` pulse
(once per access). Reads are registered (data the cycle after `sreg_rd_en`).
Writes land the cycle after `sreg_wr_en`; the dispatcher reads only between
descriptors, so no bypass exists. The banks are memories and carry no reset, so
a register holds nothing until a descriptor writes it: every program writes the
scale a later GEMV, KVWRITE or VQUANT reads, which is what the VQUANT before
each GEMV does.

### 2.8 Event strobes

| Counter | Source pulses (summed in `qcore_top` into the `qcore_csr` increment) |
|---|---|
| `SAT_REQ` | `requant.sat_inc[2:0]`: a per-cycle count, `0..4`, one per saturating `sat40` / `sat32` stage of the element leaving the pipeline (stage-1, stage-2, bias add, accumulate add) |
| `SAT_VPU` | `vpu_top.sat_inc[7:0]`: a per-cycle count, `0..VL`, one per element of a writing pass whose `sat32` saturated. A two-product pass reports both of its `sat32`, the intermediate as well as the result -- VRMSNORM's `xhat` and its `y`, VSILUMUL's `silu` and its `h` -- so an intermediate that leaves int32 raises the counter instead of quietly changing the result. The intermediate's flag travels with the chunk and is ORed with the result's before the count, so such an element is one event whether one of the two trips saturated or both. VSUBC contributes its `y`, and VROPE its two outputs with that pass. VQUANT contributes nothing: its clip is the defined result and is counted as a clip (3.12) |
| `ERR_SHIFT` | `requant.err_shift_inc[1:0]`: a per-cycle count, `0..2`, per element `sh0 > 63` (every element) and stage-2 `S` outside `[0, 63]` (elements with both scales non-zero); `vpu_top.err_shift_inc[7:0]`: a per-cycle count, `0..2 VL`, one per element per clamped shift -- VRMSNORM `S1` outside `[0, 63]` and its `G` outside `[0, 63]` can both count for one element, VSILUMUL `sh_h` outside `[0, 63]` counts once (`sh1` is i8: negative values clamp to 0) |
| `ERR_BOUNDS` | `seq_dispatch.err_bounds_inc[3:0]`: `n_from_pos` / `k_from_pos` above capacity and VSOFTMAX `len` outside `[1, n]`, each once per participating row, counted when the descriptor commits (3.5); `kv_writer.err_bounds_inc[1:0]`: `POS >= k` and a `vs_src` range past the end, once each per row; `requant.err_bounds_inc[1:0]` and `vpu_top.err_bounds_inc[3:0]`: the VSRAM ranges each leaves, once per row per range; the `err_bounds` pulse of each row and the `sreg_err` pulse of each bank |

Every port above is a per-cycle count, not a flag: the requant's three, the
vector unit's three and the dispatcher's and the KV writer's are added
arithmetically into the four CSR increments, and only the rows' `err_bounds`
and the banks' `sreg_err` are one-bit pulses, added as 0 or 1.

PERF events: `qcore_seq_dispatch` classifies every busy cycle into one of six
exclusive buckets (section 3.5) and adds `MACS`, `WT_BYTES` at issue and
`DESCRIPTORS` at retire; `qcore_mem_arb` raises `RD_BEATS` / `RD_BYTES` per
returned beat and `WR_BEATS` / `WR_BYTES` per accepted write (registered, one
cycle after the beat; `ev_wr_bytes` is 0 in cycles without `ev_wr_beat`);
`qcore_seq_fetch` raises `FETCH_BEATS` per returned `TAG_FETCH` beat.

## 3. Modules

Port tables list name, direction (from the module), width and meaning.
Parameters are named as in section 1. Sections 3.1 to 3.18 specify `qcore_pkg`
and the seventeen modules `rtl/` holds, and `make cocotb`, `make gatesim` and
`make bringup-sweep` exercise them. Within 3.12, the VROPE and VSOFTMAX passes,
the two bundle fields they read and the exp2 table image arrive with those two
opcodes (`docs/ROADMAP.md`); the rest of the section is built.

### 3.1 `qcore_pkg`

Pure functions and the read tags; no state. `desc_<field>(d)` extracts every
descriptor field from the 256-bit word at the generated position (`sh1`
signed). `sreg_pack(m, e)` builds an SREG word. `sfloat_mul(ma, ea, mb, eb)`
returns `{e[7:0], m[15:0]}` of the rounded product, `sfloat_from_int16(a, e0)`
the exact sfloat of a 16-bit integer, `bitlen64(x)` the significant-bit count
and `norm_hi16(x, len)` the truncating 16-bit normalization. `round_shift64 /
57 / 49(x, s)` are `numerics.round_shift` at those widths, computed as
`(x >>> s) + x[s-1]`; `sat40_from57`, `sat32_from64 / 57 / 49 / 33` return
`{overflow, value}`; `clip16_from49`, `clip8_from49` return `{clipped,
value}`; `abs32` gives a u32 magnitude. Every function keeps every input bit
in use so a module that calls it stays clean under `-Wall`.

### 3.2 `qcore_top`

Parameters: `WB`, `B_MAX`, `VSRAM_WORDS`, `FIFO_BEATS`, `ACC_W`,
`META_FIFO_BEATS`, `MAX_BURST`, `DQ_DEPTH`, `VL`, `VPU_FIFO_BEATS` and the
`ROM_FILE_SIGMOID`, `ROM_FILE_RSQRT` and `ROM_FILE_RECIP` image paths (each
`parameter ROM_FILE_<TABLE> = ""`, forwarded to `qcore_vpu_top`) -- what the
modules it contains take. `ROM_FILE_EXP2` arrives with VSOFTMAX, the one
opcode that reads that table.

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `clk`, `rst` | i | 1 | clock, synchronous reset |
| `csr_we`, `csr_addr`, `csr_wdata` | i | 1, 6, 32 | CSR write (word index) |
| `csr_re`, `csr_rdata` | i, o | 1, 32 | CSR read; `csr_rdata` valid the cycle after `csr_re` |
| QMEM | | | section 2.1 |

Contains one `qcore_vsram` and one `qcore_row` per row (generate loop
`g_row[r]`), the crossbar of section 2.6, the SREG read select and write mux,
the event adders of section 2.8, and single instances of `qcore_csr`,
`qcore_seq_fetch`, `qcore_seq_dispatch`, `qcore_perf`, `qcore_mem_arb`,
`qcore_stream_ctrl`, `qcore_requant`, `qcore_vpu_top` and `qcore_kv_writer`.
Its own logic is the muxes and adders below and the opcode stop at the end of
this section.
`sim/cocotb/wrappers/qcore_gemv_wrap.sv` assembles the GEMV path (arbiter,
stream controller, rows with their VSRAMs, requant, crossbar, SREG write mux,
event adders) exactly as this section wires it; the block tests elaborate it in
the three configurations and run descriptors through it against the QMEM bus
model.

| Mux | Rule |
|---|---|
| VSRAM port A | the vector unit owns the port while its `busy` is high (bank `src_row + cur_row`); otherwise bank `src_row + r` takes row `r`'s `vsa_en` / `vsa_addr` while `cmd_rows[r]`, and the row reads that bank's `rd_a` |
| VSRAM port B | one owner at a time, since one descriptor is in flight: the vector unit while its `busy` is high (bank `(vsb_sel_dst ? dst_row : src_row) + cur_row`), else the KV writer while its `busy` is high (bank `src_row + vsb_row`, reads only), else the requant (bank `dst_row + vsb_row`); `wd_b` is the owner's and the addressed bank's `rd_b` returns to it |
| SREG read | bank `sreg_rd_row` takes the dispatcher's enable and returns its word one cycle later |
| SREG write | bank `dst_row + sreg_wr_row` takes the vector unit's write while its `busy` is high, the requant's otherwise |
| Returned beat | `rdd_data` fans out to the fetch unit, the stream controller and the vector unit; `rdf_valid` / `rdw_valid` / `rdm_valid` / `rdv_valid` say which sink the beat belongs to. `rdd_last` belongs to whichever tag returned, so the vector unit is handed `rdv_valid & rdd_last`, the last beat of its own burst |
| `ROW_EN` | the dispatcher receives bits `[B_MAX-1:0]`; the rest name rows the core does not have |
| `acc_tile`, `acc_nvalid`, `acc_last` | from the lowest participating row, which the lockstep of 2.3 makes the whole handoff |
| Event counts | `SAT_REQ` the requant's `sat_inc`, `SAT_VPU` the vector unit's, `ERR_SHIFT` the requant's plus the vector unit's, `ERR_BOUNDS` the dispatcher's, the requant's, the KV writer's and the vector unit's counts plus each row's `err_bounds` and `sreg_err` (2.8) |

The control path is one loop: the host port reaches `qcore_csr`, whose
`start` / `step` / `abort_run` pulses and `pc_q`, `row_en_q`, `tok_q`,
`pos_q` registers drive `qcore_seq_dispatch`; the dispatcher drives
`qcore_seq_fetch`, the three `cmd_valid_*` groups, and back into `qcore_csr`
the `pc_set` update, the three status pulses and the fault fields; `qcore_perf`
takes `perf_clear` / `perf_snapshot` and the event strobes and returns
`perf_snap`. `qcore_top` adds the per-cycle event counts of the units, the rows
and the SREG banks into the four `*_inc` inputs of `qcore_csr` (section 2.8). `STATUS.BUSY` is the dispatcher's `busy` alone; the
units' own `busy` outputs go to the dispatcher, which folds them into the
retire conditions of section 2.2.

Opcodes with no unit: `qcore_vpu_top` executes VRMSNORM, VQUANT, VSILUMUL and
VSUBC, and VROPE and VSOFTMAX name passes it does not carry. `qcore_top` refuses
a descriptor with either of those two opcodes at the queue head: it holds the
`dq_valid` / `dq_ready` handshake so the descriptor is never popped, raises the
dispatcher's `abort_run`, and drives `qcore_csr`'s `err_set` with
`FAULT = OPCODE` and `FAULT_OP` = the opcode byte in the cycle the dispatcher
ends the run. The descriptor in flight retires first, then the run ends on the
write fence of 3.5: the fetch queue is flushed, the PERF counters are
snapshotted, `DONE` and `ERR` are set together, `busy` drops and `PC` names the
refused descriptor. Nothing is issued and nothing is counted for it. This is the
`OPCODE` fault of `docs/ISA.md` in its second form -- a defined opcode whose
pass the build does not carry.

Simulation checks (`` `ifndef SYNTHESIS ``): no VROPE or VSOFTMAX reaches an
issue pulse (the message carries `cmd_len` and `cmd_pos`, the bundle fields
those two read), at most one of the requant, the KV writer and the vector unit
claims port B in a cycle, and no VSRAM word is read on port A and written on
port B in one cycle (2.6).

### 3.3 `qcore_csr`

| Port | Dir | Width | Meaning |
|---|---|---|---|
| host `csr_*` | | | as `qcore_top` |
| `start`, `step`, `abort_run` | o | 1 | w1p pulses, the cycle after the CTRL write |
| `pc_q` | o | 32 | `PC` |
| `pc_set`, `pc_set_val` | i | 1, 32 | dispatcher update (retire: `+32`); takes precedence over a host write in the same cycle |
| `row_en_q`, `tok_q`, `pos_q` | o | 32 | the rw registers |
| `busy_i` | i | 1 | `STATUS.BUSY` |
| `done_set`, `step_halted_set`, `err_set` | i | 1 | set pulses for the three sticky bits |
| `fault_code`, `fault_op` | i | 4, 8 | latched into `STATUS.FAULT` and `STATUS.FAULT_OP` in the cycle of `err_set` |
| `argmax_we`, `argmax_tok`, `argmax_val` | i | 1, 32, 32 | ARGMAX CSR write |
| `sat_req_inc`, `sat_vpu_inc`, `err_shift_inc`, `err_bounds_inc` | i | 8 each | per-cycle counts (2.8); the four counters wrap at 32 bits and clear on `start` |
| `perf_snap` | i | 1024 | `PERF[i]` at bits `[64i +: 64]` |

The port name is `abort_run` because Verilator reserves `abort`.

Register behaviour, one row per access class of the CSR table in
`docs/ISA.md`:

| Register | Behaviour |
|---|---|
| `CTRL` (w1p) | a one in bit `START`, `STEP` or `ABORT` produces that pulse the cycle after the write. `START` and `STEP` act only while `busy_i` is low, `ABORT` only while it is high, and `START` wins over `STEP` in one write, so a write produces at most one pulse. Reads return 0 |
| `STATUS` (w1c) | `DONE`, `STEP_HALTED` and `ERR` are set by their pulses and stay set; `BUSY` is the live `busy_i`; `FAULT` and `FAULT_OP` latch on `err_set`. `start` and `step` clear the three bits and both fault fields, and so does writing a one to a bit (a one in `ERR` clears `FAULT` and `FAULT_OP` with it). A set pulse in the cycle of such a write wins |
| `PC`, `ROW_EN`, `TOK`, `POS` (rw) | a host write lands only while `busy_i` is low, so the values a descriptor sees cannot change under it; the host reads a register back to confirm. `pc_set` overrides a host write of `PC` in the same cycle and is the only writer while the core runs |
| `ARGMAX_TOK`, `ARGMAX_VAL` (ro) | written together by `argmax_we`; kept across `start` |
| `SAT_REQ`, `SAT_VPU`, `ERR_SHIFT`, `ERR_BOUNDS` (ro) | `+= inc` every cycle, wrapping at 32 bits; cleared by `start`, kept by `step` |
| `ISA_VERSION` (ro) | the generated constant |
| `PERF<i>_LO`, `PERF<i>_HI` (ro) | the two halves of `perf_snap` counter `i` |
| unmapped words | read 0; writes are ignored, as they are to every ro word |

Reads are registered: `csr_rdata` carries the addressed word the cycle after
`csr_re` and holds it until the next read. A read and a write of the same word
in one cycle return the value before the write.

### 3.4 `qcore_seq_fetch`

Parameters: `WB`, `DQ_DEPTH`.

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `fetch_start`, `fetch_pc` | i | 1, 32 | restart at `fetch_pc` (32-byte aligned) |
| `fetch_step` | i | 1 | level: fetch exactly one descriptor |
| `fetch_flush` | i | 1 | drop the queue; beats of requests still in flight are discarded on return |
| `fetch_hold` | i | 1 | level: issue no new request while it is high (the dispatcher drives `!wr_idle`) |
| `f_req_valid`, `f_req_ready`, `f_req_addr`, `f_req_len`, `f_req_tag` | o, i, o, o, o | 1, 1, 32, 8, 4 | read requests, `tag = TAG_FETCH` |
| `fd_valid`, `fd_data`, `fd_last` | i | 1, `DW`, 1 | routed `TAG_FETCH` beats |
| `dq_valid`, `dq_desc`, `dq_ready` | o, o, i | 1, 256, 1 | queue head; pop on `dq_valid && dq_ready` |
| `dq_count` | o | 4 | descriptors queued |
| `ev_fetch_beat` | o | 1 | a `TAG_FETCH` beat returned |

`fetch_pc` is always a multiple of 32: the dispatcher faults a `START` or
`STEP` with a misaligned `PC` before any request is issued (3.5).

Requests: `WB >= 32`: one beat at `ptr & ~(WB-1)` holding `WB/32`
descriptors (those before `ptr` in the beat are dropped, so a `PC` left at a
32-byte boundary by STEP resumes correctly); `WB < 32`: `32/WB` consecutive
beats per descriptor, assembled least-significant beat first. A request is
issued only when the queue has room for the whole burst, fewer than 2 bursts
are outstanding and `fetch_hold` is low; in step mode only one descriptor is
fetched. A request already on `f_req_valid` keeps its valid asserted until it
is granted, as section 1 requires. Queue depth `DQ_DEPTH` = 8 descriptors.
Descriptor bit `i` is bit `i` of the assembled 256-bit word (byte `i/8`, bit
`i%8` of the beat bytes).

`fetch_hold` puts the prefetch under the same fence every QMEM-reading
descriptor takes, so no descriptor read is issued while a KV or dump write is
unacknowledged. That orders the bus, not the program: descriptors already in the
queue or in a burst in flight are not re-read, so a program that writes into its
own descriptor stream must keep those writes outside the prefetch window, which
is up to `DQ_DEPTH` queued descriptors plus the two bursts the unit keeps
outstanding (`max(WB, 32)` bytes each). A write that lands ahead of that window
is read back by the fetch that follows it, because the fence holds every new
fetch request behind the acknowledgement.

### 3.5 `qcore_seq_dispatch`

Parameters: `WB`, `B_MAX`.

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `start`, `step`, `abort_run` | i | 1 | from `qcore_csr` |
| `pc_q`, `tok_q`, `pos_q` | i | 32 | CSRs |
| `row_en_q` | i | `B_MAX` | `ROW_EN` bits `[B_MAX-1:0]`; `qcore_top` drops the rest, which name rows the core does not have |
| `pc_set`, `pc_set_val` | o | 1, 32 | retire update |
| `busy` | o | 1 | `STATUS.BUSY`: high from the cycle after `start` / `step` until `DONE`, `STEP_HALTED` or `ERR` is set |
| `done_set`, `step_halted_set`, `err_set` | o | 1 | status pulses |
| `fault_code`, `fault_op` | o | 4, 8 | the fault and the opcode byte, valid with `err_set` |
| `perf_clear`, `perf_snapshot` | o | 1 | `perf_clear` on `start`; `perf_snapshot` on a HALT retire, and at the end of a STEP, an ABORT or a fault, which is the first cycle `wr_idle` is high |
| `fetch_start`, `fetch_pc`, `fetch_step`, `fetch_flush`, `fetch_hold` | o | 1, 32, 1, 1, 1 | to `qcore_seq_fetch`; `fetch_hold` is `!wr_idle` |
| `dq_valid`, `dq_desc`, `dq_ready` | i, i, o | 1, 256, 1 | queue head |
| `sreg_rd_en`, `sreg_rd_row`, `sreg_rd_idx`, `sreg_rd_data` | o, o, o, i | 1, `RW`, 8, 32 | SREG read of bank `sreg_rd_row` (the physical bank, `src_row + r`) |
| `cmd_*`, `cmd_valid_*`, `done_*` | | | section 2.2 |
| `wr_idle` | i | 1 | every issued QMEM write acknowledged |
| `gemv_beat` | i | 1 | a GEMV weight beat was accepted by the rows this cycle (the OR of the rows' `ev_beat`, combinational) |
| `stream_done` | i | 1 | `qcore_stream_ctrl` delivered its last beat |
| `stream_busy` | i | 1 | `qcore_stream_ctrl.busy`: requests outstanding or beats undelivered, the second half of the GEMV / EMBED retire condition |
| `ev_bucket` | o | 6 | one-hot per busy cycle: `{STALL_DRAIN, STALL_SEQ, STALL_KV, STALL_VPU, STALL_MEM, MAC_ACTIVE}` |
| `ev_desc` | o | 1 | descriptor retired |
| `ev_macs_valid`, `ev_macs` | o | 1, 40 | at issue of a GEMV: `popcount(rows) * ceil(N/WB) * WB * K` |
| `ev_wt_valid`, `ev_wt_bytes` | o | 1, 40 | at issue: GEMV without `n_from_pos` / `k_from_pos`: `popcount(rows) * (ceil(N/WB)*K*WB + (unit_meta ? 0 : ceil(N/WB)*WB*8))`; EMBED: `popcount(rows) * (K + 8)` |
| `err_bounds_inc` | o | 4 | section 2.8; pulsed at the descriptor's commit -- its issue, or the retire of one that does no work -- so a descriptor an ABORT stops before its issue counts nothing |

Sequencing per descriptor: pop, decode (1 cycle), check the faults below, wait
for the auto-fence, read one SREG word per participating row, derive the
POS-dependent extents, pulse `cmd_valid_*`, retire. Section 2.2 holds the
per-opcode table of what is waited for and what retires, the zero-work rule and
the POS derivations; this section holds what the dispatcher does around them.

Retire: `pc_set` with `pc_q + 32`, `ev_desc`, and the bucket of that cycle.
HALT additionally pulses `perf_snapshot`, `done_set` and `fetch_flush` and
drops `busy`; it comes out of the fence, so it needs no further wait.

ABORT stops further issue on the cycle the pulse is seen. It is acted on in the
fetch, decode, prepare, fence and SREG-read states, so a descriptor already
popped but not issued does not run: it is not retired, not counted -- nothing of
it reaches `ERR_BOUNDS` either -- and `PC` is left on it. A descriptor already
in flight retires normally. The run then ends on the write fence below.

Faults: a descriptor the hardware cannot execute stops the program instead of
running on. `fault_code`, `fault_op` and `PC` are fixed at the faulting cycle,
so `STATUS` names the fault and the opcode and `PC` names the address. Nothing
is issued and nothing is counted for it. The run ends on the write fence below.

The end of a run: a fault, an ABORT and a STEP retire declare the run over only
once every issued write has been acknowledged, which is the guarantee
`docs/ISA.md` attaches to `STATUS.DONE` and to `STATUS.STEP_HALTED`, and the one
a HALT retire already carries out of the fence. State `S_STOP` holds the
dispatcher while `wr_idle` is low, `busy` stays high and those cycles count as
`STALL_KV`; on the first cycle `wr_idle` is high the dispatcher pulses
`done_set` -- `step_halted_set` instead when the run ends on a step, `err_set`
with `fault_code` and `fault_op` as well on the fault path -- then `fetch_flush`
and `perf_snapshot`, and drops `busy`. A stepped HALT ends on `DONE` at its
retire, out of the auto-fence it has just left, and an `ABORT` written during a
step ends on `DONE` too.

| `FAULT` | Name | Raised when | `FAULT_OP` |
|---|---|---|---|
| 0 | `NONE` | no fault | 0 |
| 1 | `OPCODE` | the opcode byte is none of the twelve, or it names a pass the build does not carry (3.2, opcodes with no unit) | that byte |
| 2 | `ROW` | a participating row's `src_row + r` or `dst_row + r` is at or above `B_MAX` | the opcode |
| 3 | `PC_ALIGN` | `start` or `step` with `PC` not a multiple of 32 | 0 |

`PC_ALIGN` is checked in the cycle after the pulse, before the first fetch
request; `OPCODE` and `ROW` in the decode cycle, before the auto-fence and the
SREG reads. The compiler cannot emit a `ROW` descriptor (`docs/ISA.md`, rows)
and `sw/quettos/isa_sim.py` rejects one; the fault is what the hardware does
when it meets one anyway.

Step mode: the host single-steps a program through `CTRL` and `STATUS`.

1. While `BUSY` is low, write `PC` (a multiple of 32), `TOK`, `POS` and
   `ROW_EN`; they are ignored while `BUSY`.
2. Write `CTRL.STEP`. The pulse clears `DONE`, `STEP_HALTED`, `ERR` and the
   fault fields, pulses `fetch_flush` and `fetch_start` at `PC` with
   `fetch_step` high so exactly one descriptor is fetched, and raises `busy`.
3. That descriptor is issued and retired like any other, and `PC` advances by
   32. On the first cycle `wr_idle` is high after the retire -- the same write
   fence a fault and an `ABORT` end on -- `perf_snapshot` copies the live
   counters into the PERF halves, the core sets `STEP_HALTED` and `busy` drops.
   A stepped `KVWRITE` or dump is therefore acknowledged in memory before the
   host reads it back. A HALT sets `DONE` at its retire instead, and a fault
   `ERR` with its fault code at the fence.
4. The host polls `STATUS` until one of `STEP_HALTED`, `DONE` and `ERR` is set,
   then reads `PC`, `ARGMAX_TOK` / `ARGMAX_VAL`, the four event counters and
   the PERF halves, and dumps VSRAM and QMEM through the harness.
5. Writing the observed bits back clears them, so the next poll is unambiguous.
   Repeat from 2 until `DONE`.

A fault in a stepped descriptor ends the run on the write fence above. An
`ABORT` written before the stepped descriptor retires ends the run the same way
and sets `DONE`; one written after it retires finds the step complete, so the
core reports `STEP_HALTED` and the abort takes effect at the next `STEP`.

`STEP` clears no counter, so the event and PERF counters run from the last
`START` (or from reset) and the value of one descriptor is the difference
between two consecutive steps. `START` is the free run: it clears the same
status bits, pulses `perf_clear`, and executes from `PC` to HALT. Both restart
the fetch at `PC`, so the host may move `PC` between steps.

PERF buckets: every cycle with `busy` high belongs to exactly one bucket and no
cycle with `busy` low belongs to any, so `BUSY = MAC_ACTIVE + STALL_MEM +
STALL_VPU + STALL_KV + STALL_SEQ + STALL_DRAIN` holds cycle by cycle. A
descriptor is *in flight* between its issue pulse and its retire. The first
matching row owns the cycle:

| Priority | Bucket | Condition | Owner |
|---|---|---|---|
| 1 | `MAC_ACTIVE` | a GEMV is in flight and `gemv_beat` | the rows (`qcore_row.ev_beat`) |
| 2 | `STALL_DRAIN` | a GEMV or EMBED is in flight, `stream_done`, and this is not the issue cycle | the requant, still draining after the last beat |
| 3 | `STALL_MEM` | a GEMV or EMBED is in flight | `qcore_stream_ctrl`, waiting for beats |
| 4 | `STALL_VPU` | a V op is in flight | `qcore_vpu_top` |
| 5 | `STALL_KV` | a KVWRITE is in flight, or the dispatcher waits for `wr_idle` -- the auto-fence, a FENCE, or the fence a fault, an ABORT or a step ends on | `qcore_kv_writer`, `qcore_mem_arb` |
| 6 | `STALL_SEQ` | every other busy cycle | the dispatcher: fetch wait, decode, SREG reads, issue setup, retire |

Only the first three rows can be true together, since one descriptor is in
flight at a time: the last beat of a GEMV is `MAC_ACTIVE`, not `STALL_DRAIN`.
An EMBED never reaches `MAC_ACTIVE` -- its rows accept pseudo-beats but multiply
nothing -- so its cycles are `STALL_MEM` and then `STALL_DRAIN`. The auto-fence
and FENCE cycles are `STALL_KV` even when the waiting descriptor is a GEMV,
because nothing has been issued yet. `ev_cycle` is `busy`, so `CYCLES` counts
the same cycles as `BUSY`.

The issue cycle is excluded from `STALL_DRAIN` because `qcore_stream_ctrl`
re-evaluates `stream_done` at the registered command pulse: in that one cycle
the level still describes the previous descriptor while the dispatcher already
counts the new one as in flight. The cycle belongs to `STALL_MEM`, which is what
it is -- the descriptor waiting for its first beat.

### 3.6 `qcore_mem_arb`

Parameters: `WB`, `MAX_BURST`.

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `s_req_*`, `v_req_*`, `f_req_*` | | | three read requesters (`valid` i, `ready` o, `addr` 32 i, `len` 8 i, `tag` 4 i): stream, VPU, fetch |
| `dq_count` | i | 4 | fetch queue occupancy |
| `rd_req_*` | | | QMEM read request, section 2.1 |
| `rd_data_valid`, `rd_data`, `rd_data_tag`, `rd_data_last` | i | | QMEM read data |
| `rdf_valid`, `rdw_valid`, `rdm_valid`, `rdv_valid` | o | 1 | `rd_data_valid & rd_route(tag)` for fetch, weight, meta, VPU |
| `rdd_data`, `rdd_last` | o | `DW`, 1 | the payload of every returned beat, fanned out unchanged to the sinks' `rd_data` / `rd_data_last` |
| `k_wr_*`, `d_wr_*` | | | two write requesters (`valid` i, `ready` o, `addr` 32, `data` `DW`, `strb` `WB` i): KV writer, dump |
| `wr_*`, `wr_ack` | | | QMEM write, section 2.1 |
| `wr_idle` | o | 1 | no write presented, held in the stage, or unacknowledged |
| `ev_rd_beat`, `ev_wr_beat`, `ev_wr_bytes` | o | 1, 1, 8 | registered, the cycle after a returned beat; after an accepted write, with `popcount(strb)` |

One request per cycle through a registered output stage (requester
`ready` = the stage is free or draining this cycle, so it follows
`rd_req_ready`; a granted request appears on `rd_req_*` the next cycle). Priority stream, VPU, fetch; the fetch
requester wins instead when at least `MAX_BURST` stream beats have been
granted since the last fetch grant and `dq_count < 4`. Writes: KV writer over
dump (never concurrent), same registered stage. `wr_idle` is the two 32-bit
counters `issued` and `acked` agreeing, and the stage and both write valids
low with them: a write a requester presents this cycle holds `wr_idle` low
from that cycle, so a descriptor fetch the fence releases cannot be granted
ahead of it.

### 3.7 `qcore_stream_ctrl`

Parameters: `WB`, `FIFO_BEATS`, `META_FIFO_BEATS`, `MAX_BURST`.

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `cmd_valid_gemv`, `cmd_op`, `cmd_addr_a`, `cmd_addr_m`, `cmd_n`, `cmd_k`, `cmd_k_stride`, `cmd_unit_meta`, `cmd_tok` | i | | section 2.2 |
| `s_req_*` | | | read requests to `qcore_mem_arb` (`tag` = `TAG_WEIGHT` or `TAG_META`) |
| `rdw_valid`, `rdm_valid`, `rd_data`, `rd_data_last` | i | 1, 1, `DW`, 1 | routed beats |
| `ws_*` | | | weight stream, section 2.3 |
| `meta_valid`, `meta_ready`, `meta_data` | o, i, o | 1, 1, 56 | meta side-stream, section 2.5 |
| `stream_done` | o | 1 | level: the descriptor's last stream beat was accepted |
| `busy` | o | 1 | requests outstanding or beats undelivered |

Request order per GEMV tile `t`: weight bursts covering `K` beats from
`addr_a + t*k_stride*WB` (burst length `min(MAX_BURST, remaining)`), then, if
not `unit_meta`, one 8-beat burst from `addr_m + t*WB*8` (the tile's `WB`
records). Tiles ascending, `ceil(N/WB)` of them. EMBED: one beat at
`addr_m + TOK*8` whose lanes `0..7` are the row's record, then weight bursts
covering `K` beats from `addr_a + (TOK/WB)*K*WB` (the record leads, so the
requant holds `Sw` before the first pseudo-tile). A burst is issued only when the
target FIFO has `len` free beats after the beats already outstanding for it
(`FIFO_BEATS` weight beats, `META_FIFO_BEATS` meta beats). Weight beats leave
the FIFO in order as `ws_*` with `k`, tile, `tile_start`, `tile_end`,
`nvalid = min(WB, N - tile*WB)`, `last` on the final beat. Meta beats are
serialized into records (`WB/8` per beat, record `i` at beat bits
`[64i +: 56]`), so the meta stream runs one record per cycle, `nvalid` records
per tile: the records of a partial tile's padded channels are dropped inside
the controller (a beat whose remaining records are all padding is popped in one
cycle). A descriptor with `N == 0` or `K == 0` issues no request and no beat;
`stream_done` rises the cycle after the pulse and `busy` stays low.

EMBED gather: from weight beat `i` the byte at lane `TOK % WB` is placed in
lane `i % WB` of pseudo-beat `i / WB`; a pseudo-beat is emitted when full or
at byte `K-1` with `ws_embed = 1`, `ws_k = 0`, `ws_tile = i / WB`,
`ws_tile_start = ws_tile_end = 1`, `ws_nvalid = min(WB, K - tile*WB)`,
unfilled lanes 0. `ws_last` on the final pseudo-beat.

### 3.8 `qcore_row`

Parameters: `WB`, `ACC_W`, `VSRAM_WORDS`.

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `cmd_valid_gemv`, `row_active`, `cmd_vs_src`, `cmd_k` | i | 1, 1, 16, 16 | issue; `row_active = cmd_rows[r]` |
| `ws_*` | i (`ws_ready` o) | | section 2.3 |
| `vsa_en`, `vsa_addr`, `vsa_rdata` | o, o, i | 1, `AW`, 256 | VSRAM port A of bank `src_row + r` |
| `acc_valid`, `acc_ready`, `acc_flat`, `acc_tile`, `acc_nvalid`, `acc_last` | o, i, o, o, o, o | 1, 1, `WB*ACC_W`, `TW`, `NVW`, 1 | section 2.4 (this row's slice of `acc_flat`) |
| `sreg_rd_en`, `sreg_rd_idx`, `sreg_rd_data` | i, i, o | 1, 8, 32 | bank read, data the next cycle |
| `sreg_wr_en`, `sreg_wr_idx`, `sreg_wr_data` | i | 1, 8, 32 | bank write |
| `sreg_err` | o | 1 | index at or above 32 |
| `err_bounds` | o | 1 | `vs_src + K > VSRAM_WORDS*8`, once per descriptor (active rows) |
| `ev_beat` | o | 1 | a beat accepted this cycle |

Activation: `A[k]` is the low 16 bits of element `vs_src + k` of the bank.
From the issue on the row reads port A one word ahead of the stream into a
four-word ring (the fetcher wraps to the word of `vs_src` after the last word
of a tile and is flushed on the `ws_last` beat), so an aligned or unaligned
`vs_src` streams one beat per cycle with no bubble at tile boundaries; the
first beat of a descriptor is accepted in cycle `issue + 3`. `ws_ready` is low
until the word for the beat's `k` is present, and low on a `ws_tile_end` beat
while `acc_valid || !acc_ready` (an EMBED beat needs no activation word).
Elements past `VSRAM_WORDS*8` read as 0. A non-participating row accepts every
beat, reads nothing and never raises `acc_valid`. Each of the `NG` lane groups
multiplies its 8 lanes by the broadcast activation and accumulates into its
live set (section 3.9); the finished tile stays in the live set until the next
tile's first beat moves it into the hold set, `buf_sel` telling the groups
which set `acc_flat` shows. `ev_beat` is the accept itself (`ws_valid &&
ws_ready` of a participating row). The handoff follows section 2.4.

### 3.9 `qcore_mac_lane_group`

Parameter: `ACC_W`.

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `en` | i | 1 | a beat is consumed this cycle |
| `tile_start`, `embed`, `buf_sel` | i | 1 | first beat of a tile; EMBED load; 1: `acc_drain` shows the live set, 0: the hold set |
| `w` | i | 64 | 8 int8 weights, lane `i` at `[8i +: 8]` |
| `a` | i | 16 | broadcast int16 activation |
| `acc_drain` | o | `8*ACC_W` | the finished tile, lane `i` at `[i*ACC_W +: ACC_W]` |

`prod[23:0] = $signed(w[8i +: 8]) * $signed(a_eff)` with `a_eff = embed ? 0 :
a`; on `en`: `acc <= sext(prod) + (tile_start ? c : acc)` where `c` is
`{sext(w[8i +: 8]), 24'b0}` on an EMBED beat and 0 otherwise, so one adder
serves the accumulate, the tile restart and the EMBED load (Yosys maps it to a
DSP48E1 accumulator with the P feedback and the C override; a second
accumulator set fed by the same multiplier would stay in fabric). On `en &&
tile_start` the hold set takes the finished `acc`. `acc_drain` is the live set
while `buf_sel` is high and the hold set otherwise, combinationally. One cycle
per beat; the sets carry no reset.

### 3.10 `qcore_requant`

Parameters: `WB`, `B_MAX`, `ACC_W`, `VSRAM_WORDS`.

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `cmd_valid_gemv`, `cmd_op`, `cmd_out_mode`, `cmd_accumulate`, `cmd_unit_meta`, `cmd_track_absmax`, `cmd_n`, `cmd_vs_dst`, `cmd_sreg_dst`, `cmd_sh0`, `cmd_sh1`, `cmd_imm32`, `cmd_rows`, `cmd_sx_m`, `cmd_sx_e` | i | | section 2.2 (`cmd_imm32` is `addr_c`) |
| `acc_*` | i (`acc_ready` o) | | section 2.4 |
| `meta_valid`, `meta_ready`, `meta_data` | i, o, i | 1, 1, 56 | section 2.5 |
| `vsb_row` | o | `RW` | physical row offset `r` of the current element; `qcore_top` routes port B to bank `dst_row + r` |
| `vsb_en`, `vsb_we`, `vsb_addr`, `vsb_wdata`, `vsb_rdata` | o, o, o, o, i | 1, 8, `AW`, 256, 256 | port B: old-word reads (`accumulate`), strobed output writes |
| `d_wr_valid`, `d_wr_ready`, `d_wr_addr`, `d_wr_data`, `d_wr_strb` | o, i, o, o, o | 1, 1, 32, `DW`, `WB` | dump beats |
| `sreg_wr_en`, `sreg_wr_row`, `sreg_wr_idx`, `sreg_wr_data` | o | 1, `RW`, 8, 32 | tracked absmax to bank `dst_row + sreg_wr_row` |
| `argmax_we`, `argmax_tok`, `argmax_val` | o | 1, 32, 32 | ARGMAX CSR write |
| `sat_inc`, `err_shift_inc`, `err_bounds_inc` | o | 3, 2, 2 | section 2.8 |
| `done`, `busy` | o | 1 | `done` after the last VSRAM write, dump beat, SREG and ARGMAX write of the descriptor |

Per element (row `r`, channel `c = tile*WB + j`, `j < nvalid`), one per
cycle, pipelined over seven register stages (operands, stage-1 product, `t`,
stage-2 product, `y`, bias add, accumulate add): stage 0 reads `acc_flat` and
the meta record (GEMV: one record per element from the stream for the first
participating row, from the tile buffer for the others; EMBED: the record
latched from the issue on, in any state; `unit_meta`:
`Sw = {2^15, -15}`, `bias = 0`); stage 1 `t = sat40(round_shift57(acc * Sw_m,
s1))` with `s1 = min(sh0, 63)` (`sh0 > 63` counts `ERR_SHIFT` for every
element); stage 2 `S = sbias - (Sw_e + Sx_e)` in 10-bit signed arithmetic
clamped into `[0, 63]` (`ERR_SHIFT` when clamped and both mantissas are
non-zero), `y = sat32(round_shift57(t * Sx_m, S))`, `y = 0` with no events when
`Sw_m == 0` or `Sx_m == 0`; stage 3 `y = sat32(y + bias_q)` when `bias_q != 0`;
stage 4 `y = sat32(y + old)` with `accumulate`; stage 5 absmax (per row, u32),
argmax (per row, strict `>`, initial value `-2^31` at index 0, so ties
resolve to the lowest `c`), word assembly and output. `Sx` is the row's
`cmd_sx_m / cmd_sx_e`. Every `sat` overflow is a `SAT_REQ` event.

VSRAM output (`out_mode` 0 or 3): element `c` of row `r` goes to element
`vs_dst + c` of bank `dst_row + r` (`vs_dst` is 8-aligned by the compiler;
the hardware uses `vs_dst[15:3]` as the word base). A word is written with
the strobes of the elements it received when its last element leaves the
pipeline (`c % 8 == 7` or the row's final element): the element read from
`acc_flat` in cycle `T` is in its word's port-B write in cycle `T + 8`. With
`accumulate`, the old word is read on port B in cycle `T + 1` of the word's
first element and captured two cycles later; the read is deferred (stalling
stage 0) while a write uses the port in the next cycle or while the row
select would have to change during the return of an earlier read, so
`vsb_row` holds for the cycle after every read and a read and a write never
coincide. Bounds: one `err_bounds` pulse per row for
the write range and one for the old-read range when `vs_dst + N` exceeds
`VSRAM_WORDS*8`; words past the end are neither read (0) nor written.

Dump (`out_mode` 2 or 3): outputs of row `r` are packed little-endian int32,
`WB/4` per beat, into beats at `addr_c + r*4*N + b*WB`; a beat is issued when
full or at the row's final element, with `4 * (elements in the beat)` strobes,
`ceil(4N/WB)` beats per row (offered in cycle `T + 9` of its last element).
The beats queue in a 4-entry buffer behind the registered `d_wr_*` word; the
pipeline freezes only while the queue is full, so no beat is lost and the
drain runs at one element per cycle whenever the memory accepts writes. With
several participating rows the beats of a tile's rows interleave in drain
order; within a row the addresses ascend.

Descriptor end: for each participating row ascending, the SREG absmax write
(`track_absmax`) and the ARGMAX write (`out_mode` 1 or 2), one cycle each;
the last participating row's values remain in the CSRs. `done` follows. A
descriptor with `N == 0` or an empty participating set pulses `done` without
a handoff. The drain of a tile takes `popcount(rows) * nvalid` cycles and the
next handshake follows one cycle later; the first VSRAM write of a tile lands
16 cycles after its handshake.

### 3.11 `qcore_vsram`

Parameters: `WORDS`, `W = 256`, `NE = 8`. Ports: `clk`; port A `en_a`,
`addr_a[$clog2(WORDS)-1:0]`, `rd_a[W-1:0]`; port B `en_b`, `we_b[NE-1:0]`,
`addr_b`, `wd_b[W-1:0]`, `rd_b[W-1:0]`. `rd_a` / `rd_b` hold the word
addressed in the previous cycle with the enable set and keep their value
otherwise; `we_b[i]` writes element `i` (bits `[32i +: 32]`) at the clock edge
regardless of `en_b`; a read and a write on port B in the same cycle return
the old word (READ_FIRST). One instance per row; `mem` is `verilator
public_flat_rd` for zero-cycle dumps. Yosys `synth_xilinx` maps the 4096 x 256
default to 32 RAMB36E1 (the 2048-word tiny configuration to 16).

### 3.12 `qcore_vpu_top`

Parameters: `WB`, `B_MAX`, `VL`, `VSRAM_WORDS`, `VPU_FIFO_BEATS`, `MAX_BURST`,
and the three image paths of the tables its passes read: `ROM_FILE_SIGMOID`,
`ROM_FILE_RSQRT` and `ROM_FILE_RECIP` (the last two forwarded to
`qcore_vpu_scalar`). `ROM_FILE_EXP2` joins them with VSOFTMAX, the one pass that
reads that table.

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `cmd_valid_vpu`, `cmd_op`, `cmd_vq_w8`, `cmd_vq_use_tracked`, `cmd_vq_group`, `cmd_vq_scale_mul`, `cmd_track_absmax`, `cmd_addr_a`, `cmd_n`, `cmd_vs_src`, `cmd_vs_dst`, `cmd_vs_aux`, `cmd_sreg_dst`, `cmd_sh0`, `cmd_sh1`, `cmd_imm32`, `cmd_sqrt_m`, `cmd_sqrt_e`, `cmd_rows`, `cmd_sreg_u32` | i | | section 2.2 |
| `cmd_len`, `cmd_pos`, `ROM_FILE_EXP2` | | | the VSOFTMAX length, the VROPE table row and the softmax table: they arrive with those two opcodes. An input a module does not read is an UNUSEDSIGNAL error and an unused parameter an UNUSEDPARAM error (section 5), so they join the port list with the passes that read them |
| `v_req_*` | | | read requests to `qcore_mem_arb`, `tag = TAG_VPU` |
| `rdv_valid`, `rd_data`, `rd_data_last` | i | 1, `DW`, 1 | routed beats |
| `cur_row` | o | `RW` | physical row offset `r` being processed |
| `vsa_en`, `vsa_addr`, `vsa_rdata` | o, o, i | 1, `AW`, 256 | port A of bank `src_row + r` (reads) |
| `vsb_sel_dst`, `vsb_en`, `vsb_we`, `vsb_addr`, `vsb_wdata`, `vsb_rdata` | o, o, o, o, o, i | 1, 1, 8, `AW`, 256, 256 | port B of bank `src_row + r` (`vsb_sel_dst = 0`, reads) or `dst_row + r` (`vsb_sel_dst = 1`, writes) |
| `sreg_wr_en`, `sreg_wr_row`, `sreg_wr_idx`, `sreg_wr_data` | o | 1, `RW`, 8, 32 | to bank `dst_row + sreg_wr_row` (VROPE writes no SREG) |
| `sat_inc`, `err_shift_inc`, `err_bounds_inc` | o | 8, 8, 4 | section 2.8 |
| `done`, `busy` | o | 1 | `done` after the last write of the last row |

Rows execute ascending and to completion (SREG writes included) before the
next row starts. Operand fetch presents `VL` consecutive elements from any
element offset (two-word window), reads on port A (first operand) and port B
(second operand, `vsb_sel_dst = 0`), writes on port B with element strobes
(`vsb_sel_dst = 1`; VROPE writes bank `src_row + r`). QMEM operands stream
through a `VPU_FIFO_BEATS`-deep FIFO in bursts of at most
`min(MAX_BURST, VPU_FIFO_BEATS)`, re-read for every row: gamma `2n` bytes at
`addr_a` (int16), the RoPE row 128 bytes at `addr_a + POS*128`, the centering
row `4n` bytes at `addr_a` (int32), the V-scale meta `8*len` bytes at `addr_a`
(records; only `m`, `e` are used). Throughput per pass is `VL` elements per
cycle for the passes that need one product per element, and `VL` elements every
two cycles for the two that need two -- VRMSNORM pass 3 and VSILUMUL -- since
`qcore_vpu_lane` performs one operation per cycle (3.13) and a chunk makes two
trips through it. The lane pipeline is 2 cycles, a table lookup 2 cycles (ROM 1,
interpolation 1), `qcore_vpu_scalar` a fixed 5 (3.14).

Port B carries both the second operand and the output writes, and the port-B
bank the 2.6 crossbar selects follows `vsb_sel_dst` combinationally, so the unit
never presents a write in the cycle after a port-B read -- the analogue of the
requant's `vsb_row` rule in 3.10.

| Op | Passes (each `n/VL` cycles plus pipeline fill) | Result placement |
|---|---|---|
| VRMSNORM | 1: absmax; 2: `sum((x >> sh)^2)` with `sh = max(0, bitlen(amax) - 15)`, accumulated in 56 bits; scalar `RMS_SCALE`; 3: `xhat = sat32(round_shift49(x * Rc_m, S1))`, `y = sat32(round_shift49(xhat * gamma, G))` | `vs_dst`; absmax to `SREG[sreg_dst]` |
| VQUANT | 1: absmax (skipped with `USE_TRACKED`: `cmd_sreg_u32` of the row); scalar `QUANT_SCALE`; 2: `q = clip(round_shift49(x * inv, shift))`; with `GROUP` the three steps repeat per `vs_aux`-element group (`USE_TRACKED` ignored) | int8 / int16 values sign-extended to int32 at `vs_dst`; scale(s) to `SREG[sreg_dst + g]` |
| VROPE | 1 per 64-element head: `a' = sat32(round_shift49(a*cos - b*sin, 14))`, `b' = sat32(round_shift49(b*cos + a*sin, 14))` on pairs `(i, i+32)`; table row from QMEM | in place at `vs_src` of bank `src_row + r` |
| VSILUMUL | 1: `sig = sigmoid table`, `silu = round_shift49(g * sig, 15)`, `h = sat32(round_shift64(silu * u, sh_h))` | `vs_dst`; absmax to `SREG[sreg_dst]` |
| VSOFTMAX | 1: `m = max s[0..len)`; 2: `e_t` (exp2 table, Q1.23) written as int32 into `vs_dst[0..len)`, `sum`; scalar `SOFTMAX_NORM`; 3: `p_t`, `w_t` from the stored `e_t` and the streamed V scales, `e_max` over non-zero scales, weights clipped at 32767, rewritten over `vs_dst[0..len)`, zeros written to `vs_dst[len..n)` | `vs_dst`; `SREG[sreg_dst] = {2^15, e_max - 14}` or 0 |
| VSUBC | 1: `sat32(x - c)` with the streamed row | `vs_dst` |

`ERR_BOUNDS`: one pulse per row per operand range past `VSRAM_WORDS*8`
(VRMSNORM, VQUANT, VSUBC, VROPE: read and write; VSILUMUL: `src`, `aux`,
write; VSOFTMAX: the `len` read and the `n` write); SREG indices through the
banks' `sreg_err`. `ERR_SHIFT` and `SAT_VPU` as in section 2.8; VQUANT and
softmax clips are not events.

Destination and source: a pass streams its operands ahead of its writes, so it
reproduces `sw/quettos/isa_sim.py` -- which takes the whole source before it
writes anything -- only where a source and the destination cannot interleave.
The rule holds per source range the opcode reads, `vs_src` and, on VSILUMUL,
`vs_aux`: `vs_dst` starts either at exactly that source, the in-place form the
compiler uses for VQUANT and VSUBC, or at least `n` elements away from it.
Either source may be the one written in place, so a VSILUMUL may write over
`vs_aux` while `vs_src` lies elsewhere. A destination that starts inside a
source range is written before the rest of that source is read; far enough into
it, the same word is read on port A and written on port B in one cycle, the
design error of section 2.6.

The rule binds within one bank. A pass reads bank `src_row + r` and writes bank
`dst_row + r`, so a descriptor with `dst_row != src_row` addresses a different
row's VSRAM for its writes and no element index can alias; VROPE writes its own
source and is exempt by definition. `compiler.vector_overlap` is the rule in
software, the simulation check in `rtl/qcore_top.sv` is the same rule in
hardware (that level owns the bank crossbar, so it is the level that knows
whether a source and a destination share a bank), and `docs/ISA.md` states it
on the opcodes.

Two bounds the VRMSNORM passes rest on. Pass 2 keeps 15 magnitude bits of the
largest element, so each `(x >> sh)^2` is under `2^30` and the 56-bit
accumulator is exact while `n <= 2^26`; the descriptor's `n < 2^24` sits inside
that. Pass 3 carries `xhat` through the lane, whose result is an int32, so the
unit reproduces `numerics.rmsnorm` while `sqrt(d) * 2^FRAC_X * (1 + 2^-13)` fits
a signed 32-bit value: `quantize.xhat_bits` puts that bound at 22 bits at
`d = 896` with `FRAC_X = 16`, signed bits as every width in these documents is
unless it says magnitude (`docs/NUMERICS.md`, Primitives).
`quantize.check_rmsnorm_domain` refuses a model that breaks it, an `xhat` that
did leave int32 would saturate in the lane and count in `SAT_VPU` (2.8) rather
than pass silently, and `sim/cocotb/tb_vpu_top.py` asserts the domain on every
VRMSNORM it drives.

### 3.13 `qcore_vpu_lane`

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `in_valid`, `op` | i | 1, 3 | operation (below) |
| `a`, `b` | i | 32 | int32 operands |
| `c`, `c2` | i | 16 | int16 coefficients (mantissa, table value, cos / sin) |
| `sh` | i | 6 | shift |
| `out_valid`, `y` | o | 1, 32 | result, 2 cycles after `in_valid` |
| `p64` | o | 64 | the raw `a * b` product of `L_MUL32` (sum of squares) |
| `sat` | o | 1 | the result saturated |

Ops: `L_MUL32` `y = sat32(round_shift64(a * b, sh))`; `L_MUL16` `y =
sat32(round_shift49(a * c, sh))`; `L_ROPE_A` `y = sat32(round_shift49(a*c -
b*c2, sh))`; `L_ROPE_B` `y = sat32(round_shift49(a*c + b*c2, sh))`; `L_SUB`
`y = sat32(a - b)`; `L_PASS` `y = a`. Coefficients are handled as 17-bit
signed: `L_MUL16` widens `c` as u16, which is what VRMSNORM's `Rc_m`, VQUANT's
`inv` and VSILUMUL's sigmoid need, and `L_ROPE_A` / `L_ROPE_B` widen `c` and
`c2` as int16, which is what `numerics.rope`'s Q1.14 row needs. A signed
coefficient outside a rotation reaches the same product through `L_ROPE_A` with
`b = 0`. Clips, maxima and absmax are computed in `qcore_vpu_top` from `y`.
One element per cycle; the encodings are `L_MUL32` 0 through `L_PASS` 5, the
order they are listed in.

### 3.14 `qcore_vpu_scalar`

Parameters: `ROM_FILE_RSQRT`, `ROM_FILE_RECIP` (owns those two ROMs and
their interpolators).

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `req_valid`, `req_op` | i | 1, 2 | `RMS_SCALE` 0, `QUANT_SCALE` 1, `SOFTMAX_NORM` 2 |
| `req_x` | i | 64 | `ss'` (49 bits used) / `a_eff` (33 bits) / `total` (37 bits) |
| `req_sh0` | i | 8 | `FRAC_X` / `FRAC_in` |
| `req_sh` | i | 6 | VRMSNORM `sh` |
| `req_w8`, `req_mul_en` | i | 1 | VQUANT width and `SCALE_MUL` |
| `req_aux_m`, `req_aux_e` | i | 16, 8 | `sqrt_d` / `scale_mul` |
| `rsp_valid` | o | 1 | five cycles after `req_valid` |
| `rsp_m` | o | 16 | `Rc_m` / `inv` / `inv` |
| `rsp_shift`, `rsp_shift_err` | o | 6, 1 | `S1` clamped to `[0, 63]` / `31 + e_a - w` / `7 + e_s`; `err` on a clamp in either direction, which is the `ERR_SHIFT` definition of `docs/ISA.md`, and low when `rsp_zero` is set |
| `rsp_sx_m`, `rsp_sx_e` | o | 16, 8 | VQUANT `Sx` after `scale_mul` |
| `rsp_zero` | o | 1 | `req_x == 0`, for all three ops; `qcore_vpu_top` drives `req_x = 0` for a zero VQUANT vector, since `a_eff = a + (a >> (w-1)) + 1` is at least 1 and has no encoding for it. A zero magnitude returns the canonical zero scale `{0, 0}` with `rsp_shift_err` low |

`RMS_SCALE`: `L = bitlen64(ss')`, `2e = L-1` if `L` odd else `L-2`, `m_q16 =
norm` of `ss'` to `[2^16, 2^18)`, `R = rsqrt(m_q16)`, `Rc = sfloat_mul(sfloat_from_int16(R, -15),
sqrt_d)`, `S1 = -(Rc_e + FRAC_X - sh - e)`. `QUANT_SCALE`: `e_a = bitlen(a_eff) -
16`, `a_hi = norm_hi16(a_eff)`, `inv = recip(a_hi)`, `Sx = {a_hi, e_a - (w-1) -
FRAC_in}`, then `sfloat_mul` with `scale_mul`. `SOFTMAX_NORM`: `e_s =
bitlen(total) - 16`, `sum_hi = norm_hi16(total)`, `inv = recip(sum_hi)`,
`shift = 7 + e_s`. The pipeline is a fixed five stages and nothing stalls, so a
request may enter every cycle.

### 3.15 `qcore_lut_rom`

Parameters: `ENTRIES` (256 or 512), `ROM_FILE` (`parameter ROM_FILE = ""`,
the absolute image path set by the build; Yosys 0.65 rejects `parameter
string`). Ports:
`clk`; `en_a`, `idx_a[$clog2(ENTRIES)-1:0]`, `v_a[15:0]`, `dv_a[15:0]`;
`en_b`, `idx_b`, `v_b`, `dv_b`. Each line of the hex file is one 32-bit word
`{v[15:0], dv[15:0]}`; outputs are registered (valid the cycle after the
enable). The `initial $readmemh(ROM_FILE, mem)` that loads it is the one
initial block in the synthesizable RTL. `qcore_vpu_top` instantiates
`ceil(VL/2)` sigmoid ROMs -- two ports each, one per lane -- and
`qcore_vpu_scalar` one rsqrt and one recip ROM; the exp2 ROMs join them with
VSOFTMAX.

### 3.16 `qcore_lut_interp`

Ports: `clk`, `rst`, `in_valid`, `v[15:0]`, `dv[15:0]`, `frac8[7:0]`,
`out_valid`, `y[15:0]`. `v` is the unsigned Q1.15 sample and `dv` the i16
forward difference, which is what `numerics.Lut` requires: exp2's entries run to
65359, so a signed `v` would be negative over most of that table.
`y = v + ((dv * frac8 + 128) >>> 8)` in 25-bit signed arithmetic; the result
lies in `[0, 65535]` by table construction. One cycle.

### 3.17 `qcore_kv_writer`

Parameters: `WB`, `B_MAX`, `VSRAM_WORDS`.

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `cmd_valid_kv`, `cmd_kv_transposed`, `cmd_addr_a`, `cmd_addr_m`, `cmd_k_stride`, `cmd_vs_src`, `cmd_rows`, `cmd_sx_m`, `cmd_sx_e`, `cmd_pos` | i | | section 2.2 (`cmd_k_stride` is the token capacity) |
| `vsb_row`, `vsb_en`, `vsb_addr`, `vsb_rdata` | o, o, o, i | `RW`, 1, `AW`, 256 | port B read of bank `src_row + r` |
| `k_wr_valid`, `k_wr_ready`, `k_wr_addr`, `k_wr_data`, `k_wr_strb` | o, i, o, o, o | 1, 1, 32, `DW`, `WB` | write beats |
| `err_bounds_inc` | o | 2 | per row: `POS >= k`; `vs_src + 64` past the end |
| `done`, `busy` | o | 1 | `done` after the meta beat is accepted |

Per participating row ascending: when `POS >= k` count and skip; else read
the `ceil((vs_src % 8 + 64) / 8)` words holding elements `vs_src .. vs_src+63`
(one per cycle, pipelined), take the low byte of each element, then issue,
one beat per cycle when `k_wr_ready`: `TRANSPOSED`: 64 beats, beat `d` at
`addr_a + (POS/WB)*64*WB + d*WB + POS%WB` with byte `d` in lane 0 and `strb =
1`; otherwise `ceil(64/WB)` beats, beat `t` at `addr_a + (t*k + POS)*WB` with
bytes `t*WB ..` in lanes `0 ..` (zero beyond 64) and every strobe set; then
the meta beat at `addr_m + POS*8`: lanes `0..7` = `{bias 0, m, e, 0}`, `strb =
0xFF`. `WR_BEATS` per row is 65 (`TRANSPOSED`) or `ceil(64/WB) + 1`.

### 3.18 `qcore_perf`

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `clear`, `snapshot` | i | 1 | zero everything; copy the live counters into the snapshot |
| `ev_cycle`, `ev_busy` | i | 1 | level: `CYCLES`, `BUSY` count while high |
| `ev_bucket` | i | 6 | one-hot, `MAC_ACTIVE` .. `STALL_DRAIN` |
| `ev_rd_beat`, `ev_wr_beat`, `ev_wr_bytes` | i | 1, 1, 8 | `RD_BEATS`, `RD_BYTES += WB`, `WR_BEATS`, `WR_BYTES += ev_wr_bytes` |
| `ev_wt_valid`, `ev_wt_bytes`, `ev_macs_valid`, `ev_macs` | i | 1, 40, 1, 40 | `WT_BYTES`, `MACS` bulk adds |
| `ev_desc`, `ev_fetch_beat` | i | 1 | `DESCRIPTORS`, `FETCH_BEATS` |
| `perf_snap` | o | 1024 | the snapshot, `PERF[i]` at `[64i +: 64]` |

Sixteen 64-bit live counters, wrapping, and sixteen snapshot registers behind
them. `clear` zeroes both sets, `snapshot` copies live into snapshot, and
`perf_snap` presents the snapshot; the dispatcher pulses them as section 3.5
says, so the halves the host reads are stable while it works between a HALT or
a step and the next `START`. `ev_cycle` is `busy` from the dispatcher, so
`CYCLES == BUSY` in v1.

`ev_bucket` is one-hot while `ev_busy` is high and zero otherwise (3.5);
bit `i` increments counter `2 + i`, so `BUSY` equals the sum of indices 2 to 7.
`ev_wt_bytes` and `ev_macs` are bulk adds at the issue of a GEMV or EMBED and
count **per participating row**, exactly as `sw/quettos/isa_sim.py` counts
`WT_BYTES` and `MACS`. `ev_wr_bytes` is a count of strobed bytes and is 0 in a
cycle without `ev_wr_beat`; `sat_*_inc`, `err_*_inc` and the requant's event
ports are per-cycle counts too, added arithmetically rather than OR-ed
(section 2.8). The `SAT_*` and `ERR_*` counters themselves live in `qcore_csr`,
not here. A `` `ifndef SYNTHESIS `` check asserts the one-hot rule and that
the six bucket counters sum to `BUSY` at every `snapshot`.

## 4. Resolved decisions

| Question | Rule |
|---|---|
| EMBED on the stream interface | one tile (`TOK/WB`) of `K` beats is streamed; `qcore_stream_ctrl` extracts lane `TOK % WB` and packs `WB` bytes per pseudo-beat with `ws_embed`; rows load `q << 24`; the requant drains it as ordinary tiles with the single latched meta record |
| Where `Sx` comes from | the dispatcher reads `SREG[src_row + r][sreg_src]` per participating row at issue and hands `cmd_sx_m / cmd_sx_e` to the requant (`{2^15, -15}` for EMBED); KVWRITE and VQUANT `USE_TRACKED` receive the same word |
| `accumulate` old values | requant reads the old word on VSRAM port B of bank `dst_row + r` when the word's first element enters its pipeline; writes have priority on the port |
| Partial words and unaligned ranges | VSRAM port B has element strobes; GEMV / EMBED `vs_dst` is 8-aligned (compiler assertion); the VPU handles any element offset |
| Partial last tile | `ws_nvalid = min(WB, N - tile*WB)` from the stream, forwarded with the handoff; the requant drains `popcount(rows) * nvalid` elements |
| DUMP beats | packed at `addr_c + r*4*N`, `WB/4` int32 per beat, `ceil(4N/WB)` beats per row, strobes on the final beat; a mid-drain stall of the write port pauses the reader |
| QMEM alignment | none; byte-scatter, meta, dump and VPU operand beats address exact bytes |
| MAC_ACTIVE | GEMV weight beats accepted by the rows; EMBED cycles are `STALL_MEM` then `STALL_DRAIN` |
| `WR_BYTES` | strobed bytes |
| `MACS`, `WT_BYTES` with rows | per participating row, as `isa_sim.py` counts them |
| Rows in the requant | rows drained ascending within each tile; per-row absmax and argmax; CSR and SREG writes at the descriptor end, last row last |
| Fetch and beat width | `WB >= 32`: one beat holds `WB/32` descriptors; `WB = 16`: two beats per descriptor |
| Fetch reservation | fetch wins the arbiter after `MAX_BURST` stream beats when fewer than 4 descriptors are queued |
| Auto-fence | every QMEM-reading opcode, FENCE and HALT wait for `wr_idle` (KV and dump writes alike); NOP, VQUANT, VSILUMUL and KVWRITE do not. The descriptor prefetch takes the same fence through `fetch_hold`, so no read of any kind passes an unacknowledged write |
| End of a run | HALT retires out of the fence; a fault, an ABORT and a step retire wait for `wr_idle` in `S_STOP` first, so `STATUS.DONE` and `STATUS.STEP_HALTED` both imply every write was acknowledged, on every path |
| Descriptors in flight | one; the next descriptor is popped only after the current one retires |
| GEMV / EMBED retire | `done_gemv` **and** `qcore_stream_ctrl.busy` low; the cycles between them are `STALL_DRAIN` |
| Zero work | an empty participating set, a GEMV / EMBED with `N == 0` or `K == 0`, and a V op with `n == 0` retire without an issue pulse |
| Fault stop | `STATUS.ERR` with `FAULT` and `FAULT_OP`, `PC` left on the descriptor that faulted; codes `OPCODE`, `ROW`, `PC_ALIGN` (3.5) |
| `PERF` snapshot | on a HALT retire, and the cycle a STEP, an ABORT or a fault closes its write fence; `perf_clear` only on `start` |
| Step-mode counters | `STEP` clears no counter; a descriptor's contribution is the difference between two consecutive steps |
| `STATUS` clearing | `start`, `step`, or writing a one to the bit; a set pulse in the same cycle wins |
| Returned beat payload | `qcore_mem_arb` presents it as `rdd_data` / `rdd_last`; `qcore_top` fans both out to the sinks' `rd_data` / `rd_data_last` |
| Exponents | i8 everywhere; `S` is formed in 10 bits and clamped |
| SREG absmax | a u32 (`2^31` for an output of `-2^31`) |
| VSOFTMAX intermediate | `e_t` parked in `vs_dst[0..len)` between pass 2 and pass 3 |
| `USE_TRACKED` with `GROUP` | `USE_TRACKED` ignored |
| Two products per element | VRMSNORM pass 3 and VSILUMUL send each chunk through the lane twice, so those passes run `VL` elements every two cycles; the one-product passes run `VL` per cycle |
| VSRAM port B during a V op | the vector unit never presents a write in the cycle after a port-B read, so the bank the crossbar selects holds while the read returns |
| A V op's destination | per source range the opcode reads (`vs_src`, plus `vs_aux` on VSILUMUL), `vs_dst` starts at that source or at least `n` elements away from it; with `dst_row != src_row` the two lie in different banks and the rule does not bind. The passes read ahead of their writes (3.12). The simulation check in `qcore_top` is qualified on the opcodes whose `vs_dst` names a destination (`compiler.VECTOR_SOURCES`: VRMSNORM, VQUANT, VSILUMUL, VSOFTMAX, VSUBC); VROPE rewrites `vs_src` in place and its `vs_dst` field is not a range |
| GROUP scale index | `sreg_dst + g` with `g` saturating at 256, so a group form with more groups than the 32-register file holds addresses an index at or above the file: dropped and counted in `ERR_BOUNDS` like any other out-of-range SREG access, never wrapped onto a register the same descriptor already wrote |
| Opcodes with no unit | VROPE and VSOFTMAX are refused at the queue head with `FAULT = OPCODE` (3.2); the other four V opcodes issue to `qcore_vpu_top` |
| SREG bank location | inside `qcore_row`, one read port and one write port |
| CSR read latency | one cycle |
| rw CSRs while BUSY | `PC`, `ROW_EN`, `TOK` and `POS` ignore host writes while `BUSY`; `PC` is the hardware's, the other three are written before START |
| Padded meta records | `qcore_stream_ctrl` drops them; the meta stream carries `nvalid` records per tile |
| Meta records with several rows | the first participating row consumes the stream; the requant replays the tile's records from its `WB`-record buffer |
| EMBED request order | the meta beat before the weight bursts |
| Accumulator sets | one DSP accumulator per lane plus a fabric hold set loaded by the next tile's first beat; `buf_sel` marks which set holds the finished tile |

## 5. Lint patterns

Every file in `rtl/` passes `verilator --lint-only -Wall -Wpedantic`, the Yosys
check and Icarus on its own (`make lint` runs each `rtl/*.sv` as its own top
with all files on the command line). The patterns that keep a module clean:

- Extract descriptor fields only through `qcore_pkg::desc_<field>(d)`; a
  register holding the raw descriptor stays fully used. Test flag bits with a
  mask and compare on the whole field, never by storing a partial slice.
- Interfaces carry exactly the bits their consumer reads: sfloats as
  `m` + `e`, meta records as 56 bits, tracked absmax as 32, `S` as 6.
- Saturation and clip helpers return `{flag, value}`; consume both.
- Memories are `logic [W-1:0] mem [0:N-1]` with separate read and write
  `always_ff` blocks and registered reads; strobed writes are a `for` loop
  over element enables (maps to RAMB36 byte enables).
- Simulation-only checks live under `` `ifndef SYNTHESIS `` in a plain
  `always @(posedge clk)` with `$error`; Yosys defines `SYNTHESIS`, Icarus
  warns on system tasks inside `always_ff`.
- A package holds only the localparams its own functions use (`-Wall`
  reports unused localparams in packages); bus constants live in
  `qcore_pkg` because `rd_route` uses them, sizes are `qcore_top` parameters.
- ROM images come through an untyped `ROM_FILE*` parameter with an empty
  default (`parameter string` is a Yosys 0.65 syntax error); the value is
  passed by parameter name: `-G` for Verilator, `-P` for Icarus, and for Yosys
  `read_verilog -defer` followed by `chparam -set ROM_FILE "<path>" <top>`
  before `hierarchy`, the one form that reaches `$readmemh`. Each `chparam
  -set` re-elaborates the deferred module immediately, so the image paths come
  first in the `chparam` that touches it -- setting any other parameter first
  runs `$readmemh` on the empty default and hard-errors. `scripts/lint.sh`,
  `sim/cocotb/qc_runner.py` and `sim/verilator/Makefile` compute absolute
  paths; a checked-in `syn/*.ys` script carries `rtl/gen/<table>.hex` relative
  to the repo root, which is where those scripts run.
- `qcore_pkg.sv` is listed first on every Yosys and Icarus command line: both
  resolve `qcore_pkg::` references only after the package has been parsed
  (`scripts/lint.sh` and `sim/cocotb/qc_runner.py` order it so).
- An input a module does not read is an UNUSEDSIGNAL error, so a payload
  that only passes through becomes an output (`qcore_mem_arb.rdd_*`).
- A unary operator on a size cast is parenthesized: `~(25'(WB - 1))`, never
  `~25'(WB - 1)`. Yosys 0.65 binds the operator to the size literal and reads
  the second form as the mask itself, where Verilator and Icarus read the
  complement; `scripts/lint.sh` greps for `~ & | ^ - + !` in front of a bare
  size cast and fails on it, and `make gatesim` compares the netlist against
  the source that produced it.
