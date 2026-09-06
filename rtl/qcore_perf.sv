// Quettos Core performance counters: the sixteen 64-bit counters of docs/ISA.md
// behind sixteen snapshot registers. Every counter takes one wrapping add per
// cycle from its event port: the level counters CYCLES and BUSY, the six
// exclusive bucket bits, the beat and byte strobes of the arbiter and the fetch
// unit, and the bulk WT_BYTES / MACS adds a GEMV or EMBED issue carries. clear
// zeroes both sets; snapshot copies the live counters including the events of
// the cycle it is pulsed in, so the descriptor and the bucket of a HALT or a
// STEP retire are in the snapshot the host reads. perf_snap presents the
// snapshot with counter i at bits [64i +: 64] and changes only on a snapshot or
// a clear. Simulation checks the one-hot bucket rule and that the six bucket
// counters sum to BUSY at every snapshot.
`include "qcore_csr_defs.svh"
module qcore_perf #(
  parameter int WB = 64
) (
  input  logic          clk,
  input  logic          rst,
  input  logic          clear,
  input  logic          snapshot,
  // level counters
  input  logic          ev_cycle,
  input  logic          ev_busy,
  // one-hot per busy cycle: {STALL_DRAIN, STALL_SEQ, STALL_KV, STALL_VPU, STALL_MEM, MAC_ACTIVE}
  input  logic [5:0]    ev_bucket,
  // memory traffic
  input  logic          ev_rd_beat,
  input  logic          ev_wr_beat,
  input  logic [7:0]    ev_wr_bytes,
  // bulk adds at the issue of a GEMV or an EMBED
  input  logic          ev_wt_valid,
  input  logic [39:0]   ev_wt_bytes,
  input  logic          ev_macs_valid,
  input  logic [39:0]   ev_macs,
  // sequencer
  input  logic          ev_desc,
  input  logic          ev_fetch_beat,
  output logic [1023:0] perf_snap
);
  localparam int CW    = 64;                    // one counter
  localparam int NC    = `QCORE_PERF_COUNT;     // counters
  localparam int TOTW  = NC * CW;

  localparam int I_CYCLES      = `QCORE_PERF_CYCLES;
  localparam int I_BUSY        = `QCORE_PERF_BUSY;
  localparam int I_MAC_ACTIVE  = `QCORE_PERF_MAC_ACTIVE;
  localparam int I_STALL_MEM   = `QCORE_PERF_STALL_MEM;
  localparam int I_STALL_VPU   = `QCORE_PERF_STALL_VPU;
  localparam int I_STALL_KV    = `QCORE_PERF_STALL_KV;
  localparam int I_STALL_SEQ   = `QCORE_PERF_STALL_SEQ;
  localparam int I_STALL_DRAIN = `QCORE_PERF_STALL_DRAIN;
  localparam int I_RD_BEATS    = `QCORE_PERF_RD_BEATS;
  localparam int I_RD_BYTES    = `QCORE_PERF_RD_BYTES;
  localparam int I_WT_BYTES    = `QCORE_PERF_WT_BYTES;
  localparam int I_WR_BEATS    = `QCORE_PERF_WR_BEATS;
  localparam int I_WR_BYTES    = `QCORE_PERF_WR_BYTES;
  localparam int I_MACS        = `QCORE_PERF_MACS;
  localparam int I_DESCRIPTORS = `QCORE_PERF_DESCRIPTORS;
  localparam int I_FETCH_BEATS = `QCORE_PERF_FETCH_BEATS;

  logic [TOTW-1:0] live;
  logic [TOTW-1:0] snap;
  logic [TOTW-1:0] nxt;

  // One add per counter, as a continuous assign per slice so that every bit of
  // nxt has exactly one driver and no process carries a constant select.
  assign nxt[I_CYCLES*CW      +: CW] = live[I_CYCLES*CW      +: CW] + {63'd0, ev_cycle};
  assign nxt[I_BUSY*CW        +: CW] = live[I_BUSY*CW        +: CW] + {63'd0, ev_busy};
  assign nxt[I_MAC_ACTIVE*CW  +: CW] = live[I_MAC_ACTIVE*CW  +: CW] + {63'd0, ev_bucket[0]};
  assign nxt[I_STALL_MEM*CW   +: CW] = live[I_STALL_MEM*CW   +: CW] + {63'd0, ev_bucket[1]};
  assign nxt[I_STALL_VPU*CW   +: CW] = live[I_STALL_VPU*CW   +: CW] + {63'd0, ev_bucket[2]};
  assign nxt[I_STALL_KV*CW    +: CW] = live[I_STALL_KV*CW    +: CW] + {63'd0, ev_bucket[3]};
  assign nxt[I_STALL_SEQ*CW   +: CW] = live[I_STALL_SEQ*CW   +: CW] + {63'd0, ev_bucket[4]};
  assign nxt[I_STALL_DRAIN*CW +: CW] = live[I_STALL_DRAIN*CW +: CW] + {63'd0, ev_bucket[5]};
  assign nxt[I_RD_BEATS*CW    +: CW] = live[I_RD_BEATS*CW    +: CW] + {63'd0, ev_rd_beat};
  assign nxt[I_RD_BYTES*CW    +: CW] = live[I_RD_BYTES*CW    +: CW] +
                                       (ev_rd_beat ? 64'(WB) : 64'd0);
  assign nxt[I_WT_BYTES*CW    +: CW] = live[I_WT_BYTES*CW    +: CW] +
                                       (ev_wt_valid ? {24'd0, ev_wt_bytes} : 64'd0);
  assign nxt[I_WR_BEATS*CW    +: CW] = live[I_WR_BEATS*CW    +: CW] + {63'd0, ev_wr_beat};
  assign nxt[I_WR_BYTES*CW    +: CW] = live[I_WR_BYTES*CW    +: CW] + {56'd0, ev_wr_bytes};
  assign nxt[I_MACS*CW        +: CW] = live[I_MACS*CW        +: CW] +
                                       (ev_macs_valid ? {24'd0, ev_macs} : 64'd0);
  assign nxt[I_DESCRIPTORS*CW +: CW] = live[I_DESCRIPTORS*CW +: CW] + {63'd0, ev_desc};
  assign nxt[I_FETCH_BEATS*CW +: CW] = live[I_FETCH_BEATS*CW +: CW] + {63'd0, ev_fetch_beat};

  always_ff @(posedge clk) begin
    if (rst || clear) begin
      live <= {TOTW{1'b0}};
      snap <= {TOTW{1'b0}};
    end else begin
      live <= nxt;
      if (snapshot) snap <= nxt;
    end
  end

  assign perf_snap = snap;

`ifndef SYNTHESIS
  // Simulation-only checks of the bucket contract (docs/RTL.md 3.5, 3.18).
  logic [2:0]  bucket_ones;
  logic [63:0] bucket_sum;

  assign bucket_ones = {2'd0, ev_bucket[0]} + {2'd0, ev_bucket[1]} + {2'd0, ev_bucket[2]} +
                       {2'd0, ev_bucket[3]} + {2'd0, ev_bucket[4]} + {2'd0, ev_bucket[5]};

  assign bucket_sum = nxt[I_MAC_ACTIVE*CW +: CW] + nxt[I_STALL_MEM*CW   +: CW] +
                      nxt[I_STALL_VPU*CW  +: CW] + nxt[I_STALL_KV*CW    +: CW] +
                      nxt[I_STALL_SEQ*CW  +: CW] + nxt[I_STALL_DRAIN*CW +: CW];

  always @(posedge clk) begin
    if (!rst) begin
      if (ev_busy && (bucket_ones != 3'd1)) begin
        $error("qcore_perf: %0d bucket bits set in a busy cycle", bucket_ones);
      end
      if (!ev_busy && (ev_bucket != 6'd0)) begin
        $error("qcore_perf: bucket %0h set outside a busy cycle", ev_bucket);
      end
      if (snapshot && !clear && (bucket_sum != nxt[I_BUSY*CW +: CW])) begin
        $error("qcore_perf: buckets sum to %0d, BUSY is %0d",
               bucket_sum, nxt[I_BUSY*CW +: CW]);
      end
    end
  end
`endif
endmodule
