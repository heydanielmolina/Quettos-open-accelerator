# qcore_top synthesis (xc7)

Every number below is read back out of `build/synth/synth_top.log` by
`scripts/synth_report.py`, which `make synth` runs. The run comes first and this page is
written from it, so the two cannot disagree. `--check` regenerates the page and requires it
back byte for byte on the Yosys build named below; another build has its own LUT packing and
path lengths, so it is held to the parameters and the hard-block inventory.

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`

```sh
mkdir -p build/synth
yosys -q -l build/synth/synth_top.log -s syn/synth_top.ys
```

The cell counts are the `stat -tech xilinx` table after `synth_xilinx -family xc7`. A block
synthesized on its own has its ports on pads, so `IBUF` and `OBUF` are in its totals; inside
`qcore_top` they are internal wires.

The path depth is `ltp -noff` over the LUT fabric, cutting at the flops (`FDRE`, `FDSE`),
the clock buffer (`BUFG`), the I/O buffers (`IBUF`, `OBUF`) and the hard blocks (`DSP48E1`,
`RAMB36E1`, `RAMB18E1`, `RAM32M`, `RAM64M`, `RAM128X1D`), so it counts logic levels between
registers and means the same thing in every block. Each path below lists its two endpoints
and the named signals and source lines between them, with the position of each along the
path.

## Demo configuration

Parameters: `WB = 64`, `B_MAX = 1`, `VSRAM_WORDS = 4096`, `FIFO_BEATS = 128`, `ACC_W = 40`,
`META_FIFO_BEATS = 16`, `MAX_BURST = 64`, `DQ_DEPTH = 8`.

| Cell | Count |
|---|---|
| `$scopeinfo` | 18 |
| `BUFG` | 1 |
| `CARRY4` | 650 |
| `DSP48E1` | 69 |
| `FDRE` | 13935 |
| `FDSE` | 10 |
| `IBUF` | 563 |
| `INV` | 213 |
| `LUT1` | 211 |
| `LUT2` | 2052 |
| `LUT3` | 4639 |
| `LUT4` | 649 |
| `LUT5` | 1463 |
| `LUT6` | 4733 |
| `MUXF7` | 520 |
| `MUXF8` | 109 |
| `OBUF` | 686 |
| `RAM32M` | 267 |
| `RAMB18E1` | 15 |
| `RAMB36E1` | 32 |

30835 cells in total (18 of them `$scopeinfo` hierarchy markers, which map to nothing):
13747 LUTs, 13945 flops, 650 `CARRY4`, 1249 I/O pads. Yosys estimates 11484 LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `DSP48E1` | 64 | `g_row[*].u_row.g_grp[*].u_grp`, `$mul` at `rtl/qcore_mac_lane_group.sv:35` |
| `DSP48E1` | 2 | `u_requant`, `$mul` at `rtl/qcore_requant.sv:519` |
| `DSP48E1` | 2 | `u_requant`, `$mul` at `rtl/qcore_requant.sv:572` |
| `DSP48E1` | 1 | `u_dispatch`, `$mul` at `rtl/qcore_seq_dispatch.sv:285` |
| `RAMB36E1` | 32 | `g_row[*].u_vsram.mem` |
| `RAMB18E1` | 15 | `u_stream.wmem` |
| `RAM32M` | 102 | `u_requant.dq_mem` |
| `RAM32M` | 81 | `u_fetch.gmem` |
| `RAM32M` | 80 | `u_stream.mmem` |
| `RAM32M` | 4 | `g_row[*].u_row.sreg` |

Memories, as `memory_libmap` mapped them:

- `qcore_top.g_row[0].u_row.sreg` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.g_row[0].u_vsram.mem` via `$__XILINX_BLOCKRAM_TDP_`
- `qcore_top.u_fetch.gmem` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.u_requant.dq_mem` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.u_stream.mmem` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.u_stream.wmem` via `$__XILINX_BLOCKRAM_SDP_`

Longest topological path through the LUT fabric: 42 cells.

```
 0  u_csr.pos_q [0]
10  u_dispatch.pos_p1 [32]
12  rtl/qcore_seq_dispatch.sv:229
13  rtl/qcore_seq_dispatch.sv:228
22  u_dispatch.n_ru [24]
26  rtl/qcore_seq_dispatch.sv:232
30  u_dispatch.n_dec [6]
36  u_dispatch.n_pad [24]
42  u_dispatch.wt_c [37]
```

## Tiny configuration

Parameters: `WB = 16`, `B_MAX = 2`, `VSRAM_WORDS = 2048`, `FIFO_BEATS = 128`, `ACC_W = 40`,
`META_FIFO_BEATS = 16`, `MAX_BURST = 64`, `DQ_DEPTH = 8`.

| Cell | Count |
|---|---|
| `$scopeinfo` | 16 |
| `BUFG` | 1 |
| `CARRY4` | 703 |
| `DSP48E1` | 37 |
| `FDRE` | 12126 |
| `FDSE` | 10 |
| `IBUF` | 179 |
| `INV` | 230 |
| `LUT1` | 33 |
| `LUT2` | 1939 |
| `LUT3` | 2899 |
| `LUT4` | 774 |
| `LUT5` | 1531 |
| `LUT6` | 4203 |
| `MUXF7` | 427 |
| `MUXF8` | 41 |
| `OBUF` | 254 |
| `RAM32M` | 98 |
| `RAMB36E1` | 34 |

25535 cells in total (16 of them `$scopeinfo` hierarchy markers, which map to nothing):
11379 LUTs, 12136 flops, 703 `CARRY4`, 433 I/O pads. Yosys estimates 9407 LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `DSP48E1` | 32 | `g_row[*].u_row.g_grp[*].u_grp`, `$mul` at `rtl/qcore_mac_lane_group.sv:35` |
| `DSP48E1` | 2 | `u_requant`, `$mul` at `rtl/qcore_requant.sv:519` |
| `DSP48E1` | 2 | `u_requant`, `$mul` at `rtl/qcore_requant.sv:572` |
| `DSP48E1` | 1 | `u_dispatch`, `$mul` at `rtl/qcore_seq_dispatch.sv:285` |
| `RAMB36E1` | 32 | `g_row[*].u_vsram.mem` |
| `RAMB36E1` | 2 | `u_stream.wmem` |
| `RAM32M` | 40 | `u_fetch.gmem` |
| `RAM32M` | 30 | `u_requant.dq_mem` |
| `RAM32M` | 20 | `u_stream.mmem` |
| `RAM32M` | 8 | `g_row[*].u_row.sreg` |

Memories, as `memory_libmap` mapped them:

- `qcore_top.g_row[0].u_row.sreg` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.g_row[0].u_vsram.mem` via `$__XILINX_BLOCKRAM_TDP_`
- `qcore_top.g_row[1].u_row.sreg` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.g_row[1].u_vsram.mem` via `$__XILINX_BLOCKRAM_TDP_`
- `qcore_top.u_fetch.gmem` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.u_requant.dq_mem` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.u_stream.mmem` via `$__XILINX_LUTRAM_SDP_`
- `qcore_top.u_stream.wmem` via `$__XILINX_BLOCKRAM_SDP_`

Longest topological path through the LUT fabric: 48 cells.

```
 0  u_csr.pos_q [0]
10  u_dispatch.pos_p1 [32]
12  rtl/qcore_seq_dispatch.sv:228
22  u_dispatch.n_ru [24]
27  rtl/qcore_seq_dispatch.sv:232
30  u_dispatch.n_dec [3]
38  u_dispatch.n_pad [24]
45  u_dispatch.wt_row [39]
48  u_dispatch.wt_c [36]
```

## Notes (hand-written)

The array dominates. The MAC lanes are DSP blocks and the VSRAMs are block RAM, so the fabric
that is left is mostly the requant, the KV writer and the stream controller. The longest path
is the POS derivation inside the dispatcher, the same chain `syn/reports/qcore_seq_dispatch.md`
shows for that block on its own. `qcore_vpu_top` is not instantiated here, so the vector unit
is outside these numbers; the GEMV, EMBED and KVWRITE datapath and the whole control path are
inside them.
