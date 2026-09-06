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
| Parameters | `qcore_top` is the root; every module takes what it needs from it. `WB` bytes per beat and MAC lanes (16 / 64 / 128), `B_MAX` activation rows (1 / 2), `VL` vector lanes (2 / 4), `VSRAM_WORDS` (2048 / 4096), `FIFO_BEATS` (128), `ACC_W` (40), `META_FIFO_BEATS` (16), `VPU_FIFO_BEATS` (16), `MAX_BURST` (64), `DQ_DEPTH` (8), the four `ROM_FILE_<TABLE>` image paths (untyped parameters with an empty default, set by the build) |
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

Issue timing: the dispatcher pops a descriptor, decodes it (1 cycle), waits for
the auto-fence when the opcode reads QMEM, reads one SREG word per
participating row (2 cycles each), then pulses `cmd_valid_*`. The unit's
`done` pulse retires the descriptor the following cycle; the next issue is at
least 3 cycles after `done` and, for GEMV and EMBED, waits for
`qcore_stream_ctrl.busy` to fall (the padded meta beats of a partial last
tile may still be draining after the requant's `done`). A GEMV or EMBED with
`N == 0` or `K == 0` retires in the dispatcher without an issue pulse.

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
simulation); the compiler's disjoint source / destination ranges and the
units' read-ahead rules keep it from happening. A written word is readable on
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
descriptors, so no bypass exists.

### 2.8 Event strobes

| Counter | Source pulses (summed in `qcore_top` into the `qcore_csr` increment) |
|---|---|
| `SAT_REQ` | `requant.sat_inc[2:0]`: a per-cycle count, `0..4`, one per saturating `sat40` / `sat32` stage of the element leaving the pipeline (stage-1, stage-2, bias add, accumulate add) |
| `SAT_VPU` | `vpu_top.sat_inc[7:0]`: per lane per cycle, one per saturating `sat32` (VRMSNORM output, VROPE outputs, VSILUMUL output, VSUBC) |
| `ERR_SHIFT` | `requant.err_shift_inc[1:0]`: a per-cycle count, `0..2`, per element `sh0 > 63` (every element) and stage-2 `S` outside `[0, 63]` (elements with both scales non-zero); `vpu_top.err_shift_inc[7:0]`: per element, VRMSNORM `S1 < 0`, VRMSNORM `G` and VSILUMUL `sh_h` outside `[0, 63]` (`sh1` is i8: negative values clamp to 0) |
| `ERR_BOUNDS` | `seq_dispatch.err_bounds_inc[3:0]`: `n_from_pos` / `k_from_pos` above capacity and VSOFTMAX `len` outside `[1, n]`, each once per participating row; `kv_writer.err_bounds_inc[1:0]`: `POS >= k` once per row; the VSRAM range pulses of `row`, `requant`, `vpu_top`, `kv_writer`; the `sreg_err` pulses of the banks |

The requant's three ports (and `err_bounds_inc[1:0]`, `0..2`) are counts,
added arithmetically into the CSR increments; the other sources are one-bit
pulses.

PERF events: `qcore_seq_dispatch` classifies every busy cycle into one of six
exclusive buckets (section 3.5) and adds `MACS`, `WT_BYTES` at issue and
`DESCRIPTORS` at retire; `qcore_mem_arb` raises `RD_BEATS` / `RD_BYTES` per
returned beat and `WR_BEATS` / `WR_BYTES` per accepted write (registered, one
cycle after the beat; `ev_wr_bytes` is 0 in cycles without `ev_wr_beat`);
`qcore_seq_fetch` raises `FETCH_BEATS` per returned `TAG_FETCH` beat.

## 3. Modules

Port tables list name, direction (from the module), width and meaning.
Parameters are named as in section 1.

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

Parameters: `WB`, `B_MAX`, `VL`, `VSRAM_WORDS`, `FIFO_BEATS`, `ACC_W`,
`META_FIFO_BEATS`, `VPU_FIFO_BEATS`, `MAX_BURST`, `DQ_DEPTH`,
`ROM_FILE_EXP2`, `ROM_FILE_SIGMOID`, `ROM_FILE_RSQRT`, `ROM_FILE_RECIP`
(each `parameter ROM_FILE_<TABLE> = ""`, forwarded to the ROMs).

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `clk`, `rst` | i | 1 | clock, synchronous reset |
| `csr_we`, `csr_addr`, `csr_wdata` | i | 1, 6, 32 | CSR write (word index) |
| `csr_re`, `csr_rdata` | i, o | 1, 32 | CSR read; `csr_rdata` valid the cycle after `csr_re` |
| QMEM | | | section 2.1 |

Contains one `qcore_vsram` and one `qcore_row` per row (generate loop
`g_row[r]`), the crossbar of section 2.6, the SREG write mux, the event
adders of section 2.8, and single instances of every other module. No logic
of its own beyond muxes and adders. `sim/cocotb/wrappers/qcore_gemv_wrap.sv`
assembles the GEMV path (arbiter, stream controller, rows with their VSRAMs,
requant, crossbar, SREG write mux, event adders) exactly as this section wires
it; the block tests elaborate it in the three configurations and run
descriptors through it against the QMEM bus model.

### 3.3 `qcore_csr`

| Port | Dir | Width | Meaning |
|---|---|---|---|
| host `csr_*` | | | as `qcore_top` |
| `start`, `step`, `abort` | o | 1 | w1p pulses, the cycle after the CTRL write; `start` and `step` are suppressed while `busy_i` |
| `pc_q` | o | 32 | `PC` |
| `pc_set`, `pc_set_val` | i | 1, 32 | dispatcher update (retire: `+32`); takes precedence over a host write in the same cycle |
| `row_en_q`, `tok_q`, `pos_q` | o | 32 | the rw registers |
| `busy_i` | i | 1 | `STATUS.BUSY` |
| `done_set`, `step_halted_set`, `err_set` | i | 1 | set pulses; `start` and `step` clear the three bits |
| `argmax_we`, `argmax_tok`, `argmax_val` | i | 1, 32, 32 | ARGMAX CSR write |
| `sat_req_inc`, `sat_vpu_inc`, `err_shift_inc`, `err_bounds_inc` | i | 8 each | per-cycle increments; the four counters wrap at 32 bits and clear on `start` |
| `perf_snap` | i | 1024 | `PERF[i]` at bits `[64i +: 64]` |

Reads return the word the cycle after `csr_re`; unmapped words read 0; writes
to ro words are ignored. `ISA_VERSION` is the generated constant.

### 3.4 `qcore_seq_fetch`

Parameters: `WB`, `DQ_DEPTH`.

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `fetch_start`, `fetch_pc` | i | 1, 32 | restart at `fetch_pc` (32-byte aligned) |
| `fetch_step` | i | 1 | level: fetch exactly one descriptor |
| `fetch_flush` | i | 1 | drop the queue; beats of requests still in flight are discarded on return |
| `f_req_valid`, `f_req_ready`, `f_req_addr`, `f_req_len`, `f_req_tag` | o, i, o, o, o | 1, 1, 32, 8, 4 | read requests, `tag = TAG_FETCH` |
| `fd_valid`, `fd_data`, `fd_last` | i | 1, `DW`, 1 | routed `TAG_FETCH` beats |
| `dq_valid`, `dq_desc`, `dq_ready` | o, o, i | 1, 256, 1 | queue head; pop on `dq_valid && dq_ready` |
| `dq_count` | o | 4 | descriptors queued |
| `ev_fetch_beat` | o | 1 | a `TAG_FETCH` beat returned |

Requests: `WB >= 32`: one beat at `ptr & ~(WB-1)` holding `WB/32`
descriptors (those before `ptr` in the beat are dropped, so a `PC` left at a
32-byte boundary by STEP resumes correctly); `WB < 32`: `32/WB` consecutive
beats per descriptor, assembled least-significant beat first. A request is
issued only when the queue has room for the whole burst and fewer than 2
bursts are outstanding; in step mode only one descriptor is fetched. Queue
depth `DQ_DEPTH` = 8 descriptors. Descriptor bit `i` is bit `i` of the
assembled 256-bit word (byte `i/8`, bit `i%8` of the beat bytes).

### 3.5 `qcore_seq_dispatch`

Parameters: `WB`, `B_MAX`.

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `start`, `step`, `abort` | i | 1 | from `qcore_csr` |
| `pc_q`, `row_en_q`, `tok_q`, `pos_q` | i | 32 | CSRs |
| `pc_set`, `pc_set_val` | o | 1, 32 | retire update |
| `busy` | o | 1 | `STATUS.BUSY`: high from the cycle after `start` / `step` until DONE or STEP_HALTED is set |
| `done_set`, `step_halted_set`, `err_set` | o | 1 | status pulses |
| `perf_clear`, `perf_snapshot` | o | 1 | on `start`; on HALT retire and on an unknown opcode |
| `fetch_start`, `fetch_pc`, `fetch_step`, `fetch_flush` | o | 1, 32, 1, 1 | to `qcore_seq_fetch` |
| `dq_valid`, `dq_desc`, `dq_ready` | i, i, o | 1, 256, 1 | queue head |
| `sreg_rd_en`, `sreg_rd_row`, `sreg_rd_idx`, `sreg_rd_data` | o, o, o, i | 1, `RW`, 8, 32 | SREG read of bank `sreg_rd_row` (the physical bank, `src_row + r`) |
| `cmd_*`, `cmd_valid_*`, `done_*` | | | section 2.2 |
| `wr_idle` | i | 1 | every issued QMEM write acknowledged |
| `gemv_beat` | i | 1 | a GEMV weight beat was accepted by the rows this cycle (the OR of the rows' `ev_beat`, combinational) |
| `stream_done` | i | 1 | `qcore_stream_ctrl` delivered its last beat |
| `ev_bucket` | o | 6 | one-hot per busy cycle: `{STALL_DRAIN, STALL_SEQ, STALL_KV, STALL_VPU, STALL_MEM, MAC_ACTIVE}` |
| `ev_desc` | o | 1 | descriptor retired |
| `ev_macs_valid`, `ev_macs` | o | 1, 40 | at issue of a GEMV: `popcount(rows) * ceil(N/WB) * WB * K` |
| `ev_wt_valid`, `ev_wt_bytes` | o | 1, 40 | at issue: GEMV without `n_from_pos` / `k_from_pos`: `popcount(rows) * (ceil(N/WB)*K*WB + (unit_meta ? 0 : ceil(N/WB)*WB*8))`; EMBED: `popcount(rows) * (K + 8)` |
| `err_bounds_inc` | o | 4 | section 2.8 |

Sequencing per descriptor: pop, decode (1 cycle; an unknown opcode raises
`err_set` and `done_set`, flushes, snapshots PERF, leaves `PC`; nothing is
counted), NOP / HALT / FENCE / empty participating set retire without an
issue (FENCE and every QMEM-reading opcode — GEMV, EMBED, VRMSNORM, VROPE,
VSOFTMAX, VSUBC — first wait for `wr_idle`), otherwise the SREG reads
(GEMV: `sreg_src`; VQUANT with `USE_TRACKED` and not `GROUP`: `sreg_src`;
KVWRITE: `sreg_src`; one read per participating row), POS-derived values
(`gemv_dims`, `softmax_len` of `isa_sim.py`; EMBED `cmd_n = k`), then the
issue pulse. Retire on `done`: `pc_set (+32)`, `ev_desc`; HALT: `perf_snapshot`,
`done_set`, `fetch_flush`, `busy` low. STEP: one descriptor, then
`step_halted_set` (`done_set` for HALT). ABORT: no further issue; the in-flight
descriptor retires; then `fetch_flush`, `done_set`, `busy` low.

Bucket per busy cycle, first match wins: `MAC_ACTIVE` when `gemv_beat`
(GEMV only); `STALL_DRAIN` when a GEMV / EMBED is in flight and `stream_done`;
`STALL_MEM` when a GEMV / EMBED is in flight otherwise; `STALL_VPU` when a V
op is in flight; `STALL_KV` when a KVWRITE is in flight or the dispatcher waits
for `wr_idle` (auto-fence or FENCE); `STALL_SEQ` for every other busy cycle
(fetch wait, decode, SREG reads, retire). `BUSY` therefore equals the bucket
sum by construction; `CYCLES` counts the same cycles.

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
| `wr_idle` | o | 1 | writes issued == acks received |
| `ev_rd_beat`, `ev_wr_beat`, `ev_wr_bytes` | o | 1, 1, 8 | registered, the cycle after a returned beat; after an accepted write, with `popcount(strb)` |

One request per cycle through a registered output stage (requester
`ready` = the stage is free or draining this cycle, so it follows
`rd_req_ready`; a granted request appears on `rd_req_*` the next cycle). Priority stream, VPU, fetch; the fetch
requester wins instead when at least `MAX_BURST` stream beats have been
granted since the last fetch grant and `dq_count < 4`. Writes: KV writer over
dump (never concurrent), same registered stage. The two 32-bit counters
`issued` and `acked` give `wr_idle`.

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
the four `ROM_FILE_*`.

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `cmd_valid_vpu`, `cmd_op`, `cmd_vq_w8`, `cmd_vq_use_tracked`, `cmd_vq_group`, `cmd_vq_scale_mul`, `cmd_track_absmax`, `cmd_addr_a`, `cmd_n`, `cmd_len`, `cmd_vs_src`, `cmd_vs_dst`, `cmd_vs_aux`, `cmd_sreg_dst`, `cmd_sh0`, `cmd_sh1`, `cmd_imm32`, `cmd_sqrt_m`, `cmd_sqrt_e`, `cmd_rows`, `cmd_sreg_u32`, `cmd_pos` | i | | section 2.2 |
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
cycle; the lane pipeline is 2 cycles, a table lookup 2 cycles (ROM 1,
interpolation 1), `qcore_vpu_scalar` at most 8 cycles per request.

| Op | Passes (each `n/VL` cycles plus pipeline fill) | Result placement |
|---|---|---|
| VRMSNORM | 1: absmax; 2: `sum((x >> sh)^2)` (49-bit, `sh = max(0, bitlen(amax) - 15)`); scalar `RMS_SCALE`; 3: `xhat = round_shift49(x * Rc_m, S1)`, `y = sat32(round_shift49(xhat * gamma, G))` | `vs_dst`; absmax to `SREG[sreg_dst]` |
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
`y = sat32(a - b)`; `L_PASS` `y = a`. Coefficients are u16 handled as 17-bit
signed. Clips, maxima and absmax are computed in `qcore_vpu_top` from `y`.
One element per cycle.

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
| `rsp_valid` | o | 1 | at most 8 cycles after `req_valid` |
| `rsp_m` | o | 16 | `Rc_m` / `inv` / `inv` |
| `rsp_shift`, `rsp_shift_err` | o | 6, 1 | `S1` clamped to `[0, 63]` (`err` when it was negative) / `31 + e_a - w` / `7 + e_s` |
| `rsp_sx_m`, `rsp_sx_e` | o | 16, 8 | VQUANT `Sx` after `scale_mul` |
| `rsp_zero` | o | 1 | `ss' == 0` / `a == 0` |

`RMS_SCALE`: `L = bitlen64(ss')`, `2e = L-1` if `L` odd else `L-2`, `m_q16 =
norm` of `ss'` to `[2^16, 2^18)`, `R = rsqrt(m_q16)`, `Rc = sfloat_mul(sfloat_from_int16(R, -15),
sqrt_d)`, `S1 = -(Rc_e + FRAC_X - sh - e)`. `QUANT_SCALE`: `e_a = bitlen(a_eff) -
16`, `a_hi = norm_hi16(a_eff)`, `inv = recip(a_hi)`, `Sx = {a_hi, e_a - (w-1) -
FRAC_in}`, then `sfloat_mul` with `scale_mul`. `SOFTMAX_NORM`: `e_s =
bitlen(total) - 16`, `sum_hi = norm_hi16(total)`, `inv = recip(sum_hi)`,
`shift = 7 + e_s`. One request in flight.

### 3.15 `qcore_lut_rom`

Parameters: `ENTRIES` (256 or 512), `ROM_FILE` (`parameter ROM_FILE = ""`,
the absolute image path set by the build; Yosys 0.65 rejects `parameter
string`). Ports:
`clk`; `en_a`, `idx_a[$clog2(ENTRIES)-1:0]`, `v_a[15:0]`, `dv_a[15:0]`;
`en_b`, `idx_b`, `v_b`, `dv_b`. Each line of the hex file is one 32-bit word
`{v[15:0], dv[15:0]}`; outputs are registered (valid the cycle after the
enable). The `initial $readmemh(ROM_FILE, mem)` that loads it is the one
initial block in the synthesizable RTL. `qcore_vpu_top` instantiates
`ceil(VL/2)` exp2 and `ceil(VL/2)` sigmoid ROMs; `qcore_vpu_scalar` one rsqrt
and one recip ROM.

### 3.16 `qcore_lut_interp`

Ports: `clk`, `rst`, `in_valid`, `v[15:0]`, `dv[15:0]` (i16), `frac8[7:0]`,
`out_valid`, `y[15:0]`. `y = v + ((dv * frac8 + 128) >>> 8)` in 25-bit signed
arithmetic; the result lies in `[0, 65535]` by table construction. One cycle.

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

Sixteen 64-bit live counters, wrapping; `ev_cycle` is `busy` from the
dispatcher, so `CYCLES == BUSY` in v1 and `BUSY` equals the bucket sum.

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
| Auto-fence | every QMEM-reading opcode and FENCE wait for `wr_idle` (KV and dump writes alike); KVWRITE does not |
| Exponents | i8 everywhere; `S` is formed in 10 bits and clamped |
| SREG absmax | a u32 (`2^31` for an output of `-2^31`) |
| VSOFTMAX intermediate | `e_t` parked in `vs_dst[0..len)` between pass 2 and pass 3 |
| `USE_TRACKED` with `GROUP` | `USE_TRACKED` ignored |
| SREG bank location | inside `qcore_row`, one read port and one write port |
| CSR read latency | one cycle |
| `PC` while BUSY | written by the hardware only; the host writes it before START |
| Padded meta records | `qcore_stream_ctrl` drops them; the meta stream carries `nvalid` records per tile |
| Meta records with several rows | the first participating row consumes the stream; the requant replays the tile's records from its `WB`-record buffer |
| EMBED request order | the meta beat before the weight bursts |
| Accumulator sets | one DSP accumulator per lane plus a fabric hold set loaded by the next tile's first beat; `buf_sel` marks which set holds the finished tile |

## 5. Lint patterns

Every file passes `verilator --lint-only -Wall -Wpedantic`, the Yosys check
and Icarus on its own (`make lint` runs each `rtl/*.sv` as its own top with
all files on the command line). The patterns that keep a module clean:

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
  default (`parameter string` is a Yosys 0.65 syntax error); the runner and
  the lint script pass absolute `rtl/gen/*.hex` paths by parameter name:
  `-G` for Verilator, `-P` for Icarus, and for Yosys `read_verilog -defer`
  followed by `chparam -set ROM_FILE "<path>" <top>` before `hierarchy`,
  the one form that reaches `$readmemh`.
- `qcore_pkg.sv` is listed first on every Yosys and Icarus command line: both
  resolve `qcore_pkg::` references only after the package has been parsed
  (`scripts/lint.sh` and `sim/cocotb/qc_runner.py` order it so).
- An input a module does not read is an UNUSEDSIGNAL error, so a payload
  that only passes through becomes an output (`qcore_mem_arb.rdd_*`).
