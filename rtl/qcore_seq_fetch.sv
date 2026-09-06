// Quettos Core descriptor fetch: the program counter's read side. fetch_start
// restarts the stream at a 32-byte-aligned fetch_pc, fetch_flush drops the
// queue and discards the beats of requests still in flight, and fetch_step
// limits a run to one request. One request covers max(WB, 32) bytes: with
// WB >= 32 a single beat carrying WB/32 descriptors (those before fetch_pc in
// the first beat are dropped), with WB < 32 the 32/WB beats of one descriptor,
// least-significant beat first. A request is issued only into reserved queue
// space, with fewer than two bursts outstanding and with fetch_hold low, so
// every returned beat is accepted the cycle it arrives and no fetch read passes
// an unacknowledged KV or dump write. The queue holds DQ_DEPTH descriptors and
// presents its head on dq_valid / dq_desc; ev_fetch_beat is a registered pulse
// one cycle after every returned beat.
`include "qcore_csr_defs.svh"
module qcore_seq_fetch #(
  parameter int WB       = 64,
  parameter int DQ_DEPTH = 8
) (
  input  logic            clk,
  input  logic            rst,
  // restart, step and flush from the dispatcher
  input  logic            fetch_start,
  input  logic [31:0]     fetch_pc,
  input  logic            fetch_step,
  input  logic            fetch_flush,
  input  logic            fetch_hold,
  // read requests to the arbiter
  output logic            f_req_valid,
  input  logic            f_req_ready,
  output logic [31:0]     f_req_addr,
  output logic [7:0]      f_req_len,
  output logic [3:0]      f_req_tag,
  // routed TAG_FETCH beats
  input  logic            fd_valid,
  input  logic [WB*8-1:0] fd_data,
  input  logic            fd_last,
  // descriptor queue head
  output logic            dq_valid,
  output logic [255:0]    dq_desc,
  input  logic            dq_ready,
  output logic [3:0]      dq_count,
  // PERF
  output logic            ev_fetch_beat
);
  localparam int DW     = WB * 8;
  localparam int DESC_B = `QCORE_DESC_BYTES;              // 32 bytes per descriptor
  localparam int DESC_W = DESC_B * 8;                     // 256 bits
  localparam int GB     = (WB > DESC_B) ? WB : DESC_B;    // bytes per request
  localparam int BPR    = GB / WB;                        // beats per request
  localparam int DPR    = GB / DESC_B;                    // descriptors per request
  localparam int ASM_W  = DESC_W * DPR;                   // == DW * BPR
  localparam int GD     = ((DQ_DEPTH / DPR) > 2) ? (DQ_DEPTH / DPR) : 2;  // queue entries
  localparam int GAW    = $clog2(GD);
  localparam int LGD    = (DPR > 1) ? $clog2(DPR) : 1;    // descriptor index in a group
  localparam int BCW    = (BPR > 1) ? $clog2(BPR) : 1;    // beat index in a group

  // ---------------------------------------------------------------- queue occupancy
  logic [GAW-1:0]   g_wptr;
  logic [GAW-1:0]   g_rptr;
  logic [GAW:0]     g_cnt;
  logic             o_valid;
  logic [ASM_W-1:0] o_grp;
  logic [LGD-1:0]   o_lane;
  logic [1:0]       bursts;     // granted bursts whose last beat has not returned
  logic [7:0]       g_used;     // groups queued, in flight or in the output register
  logic             room;
  logic             burst_room;

  assign g_used     = 8'(g_cnt) + 8'(o_valid) + 8'(bursts) + 8'(f_req_valid);
  assign room       = g_used < 8'(GD);
  assign burst_room = (8'(bursts) + 8'(f_req_valid)) < 8'd2;

  // ---------------------------------------------------------------- request generator
  logic [31:0] req_addr;
  logic        req_en;    // this run still wants descriptors
  logic        stale;     // outstanding beats belong to a flushed run
  logic        req_slot;
  logic        load_req;
  logic        grant;

  assign req_slot = !f_req_valid || f_req_ready;
  assign grant    = f_req_valid && f_req_ready;
  assign load_req = req_slot && req_en && !stale && room && burst_room &&
                    !fetch_hold && !fetch_flush && !fetch_start;

  always_ff @(posedge clk) begin
    if (rst) begin
      f_req_valid <= 1'b0;
      f_req_addr  <= 32'd0;
      req_addr    <= 32'd0;
      req_en      <= 1'b0;
    end else begin
      if (load_req) begin
        f_req_valid <= 1'b1;
        f_req_addr  <= req_addr;
        req_addr    <= req_addr + 32'(GB);
        if (fetch_step) req_en <= 1'b0;  // step mode fetches one descriptor
      end else if (req_slot) begin
        f_req_valid <= 1'b0;
      end
      if (fetch_flush) req_en <= 1'b0;
      if (fetch_start) begin
        req_addr <= fetch_pc & ~(32'(GB - 1));
        req_en   <= 1'b1;
      end
    end
  end

  assign f_req_len = 8'(BPR);
  assign f_req_tag = qcore_pkg::TAG_FETCH;

  // ---------------------------------------------------------------- returned beats
  logic [7:0] out_beats;   // beats granted and still outstanding
  logic [7:0] out_next;
  logic       req_pending_next;
  logic       drop_beat;
  logic       take_beat;

  assign out_next         = out_beats + (grant ? 8'(BPR) : 8'd0) - {7'd0, fd_valid};
  assign req_pending_next = load_req || (f_req_valid && !f_req_ready);
  assign drop_beat        = fd_valid && (stale || fetch_flush || fetch_start);
  assign take_beat        = fd_valid && !drop_beat;

  always_ff @(posedge clk) begin
    if (rst) begin
      out_beats <= 8'd0;
      bursts    <= 2'd0;
      stale     <= 1'b0;
    end else begin
      out_beats <= out_next;
      bursts    <= bursts + {1'b0, grant} - {1'b0, fd_valid & fd_last};
      stale     <= (fetch_flush || fetch_start || stale) &&
                   ((out_next != 8'd0) || req_pending_next);
    end
  end

  // ---------------------------------------------------------------- group assembly
  logic [ASM_W-1:0] asm_n;
  logic             grp_full;

  generate
    if (BPR > 1) begin : g_asm
      // WB < 32: BPR beats make one descriptor, least-significant beat first.
      logic [ASM_W-DW-1:0] hist;   // the beats of this group that already arrived
      logic [BCW-1:0]      bcnt;
      assign asm_n    = {fd_data, hist};
      assign grp_full = take_beat && (bcnt == BCW'(BPR - 1));
      always_ff @(posedge clk) begin
        if (take_beat) hist <= asm_n[ASM_W-1:DW];
      end
      always_ff @(posedge clk) begin
        if (rst || fetch_flush || fetch_start) bcnt <= {BCW{1'b0}};
        else if (take_beat)                    bcnt <= grp_full ? {BCW{1'b0}} : bcnt + BCW'(1);
      end
    end else begin : g_asm
      // WB >= 32: one beat is one group of DPR descriptors.
      assign asm_n    = fd_data;
      assign grp_full = take_beat;
    end
  endgenerate

  // ---------------------------------------------------------------- descriptor queue
  logic [ASM_W-1:0] gmem [0:GD-1];
  logic             g_rd;
  logic             g_pop;
  logic             dq_pop;
  logic [3:0]       dcount;
  logic [3:0]       add_n;

  // fetch_pc may name a descriptor inside the first beat; the descriptors
  // before it are dropped from the count and from the read lane.
  logic [LGD-1:0] skip;
  logic           skip_w;
  logic           skip_r;

  always_ff @(posedge clk) begin
    if (rst) begin
      skip   <= {LGD{1'b0}};
      skip_w <= 1'b0;
      skip_r <= 1'b0;
    end else begin
      if (grp_full) skip_w <= 1'b0;
      if (g_rd)     skip_r <= 1'b0;
      if (fetch_start) begin
        skip   <= LGD'(fetch_pc >> 5) & LGD'(DPR - 1);
        skip_w <= 1'b1;
        skip_r <= 1'b1;
      end
    end
  end

  assign dq_pop   = o_valid && dq_ready;
  assign g_pop    = dq_pop && (o_lane == LGD'(DPR - 1));
  assign g_rd     = (g_cnt != {(GAW + 1){1'b0}}) && (!o_valid || g_pop);
  assign dq_valid = o_valid;
  assign dq_count = dcount;
  assign add_n    = 4'(DPR) - (skip_w ? 4'(skip) : 4'd0);

  always_ff @(posedge clk) begin
    if (grp_full) gmem[g_wptr] <= asm_n;
  end

  always_ff @(posedge clk) begin
    if (g_rd) o_grp <= gmem[g_rptr];
  end

  always_ff @(posedge clk) begin
    if (rst || fetch_flush || fetch_start) begin
      g_wptr  <= {GAW{1'b0}};
      g_rptr  <= {GAW{1'b0}};
      g_cnt   <= {(GAW + 1){1'b0}};
      o_valid <= 1'b0;
      o_lane  <= {LGD{1'b0}};
      dcount  <= 4'd0;
    end else begin
      if (grp_full) g_wptr <= (g_wptr == GAW'(GD - 1)) ? {GAW{1'b0}} : g_wptr + GAW'(1);
      if (g_rd)     g_rptr <= (g_rptr == GAW'(GD - 1)) ? {GAW{1'b0}} : g_rptr + GAW'(1);
      g_cnt  <= g_cnt + {{GAW{1'b0}}, grp_full} - {{GAW{1'b0}}, g_rd};
      dcount <= dcount + (grp_full ? add_n : 4'd0) - {3'd0, dq_pop};
      if (g_rd)       o_valid <= 1'b1;
      else if (g_pop) o_valid <= 1'b0;
      if (g_rd)                    o_lane <= skip_r ? skip : {LGD{1'b0}};
      else if (dq_pop && !g_pop)   o_lane <= o_lane + LGD'(1);
    end
  end

  // The head descriptor: lane o_lane of the group in the output register.
  // A continuous assign keeps the select out of an always_* process.
  generate
    if (DPR > 1) begin : g_head
      assign dq_desc = o_grp[{o_lane, 8'd0} +: DESC_W];
    end else begin : g_head
      assign dq_desc = o_grp;
    end
  endgenerate

  // ---------------------------------------------------------------- PERF
  always_ff @(posedge clk) begin
    if (rst) ev_fetch_beat <= 1'b0;
    else     ev_fetch_beat <= fd_valid;
  end

`ifndef SYNTHESIS
  // Simulation-only protocol checks: the queue never overflows and no beat
  // arrives without an outstanding request.
  always @(posedge clk) begin
    if (!rst) begin
      if (grp_full && (g_cnt == (GAW + 1)'(GD)))
        $error("qcore_seq_fetch: descriptor queue overflow");
      if (fd_valid && (out_beats == 8'd0))
        $error("qcore_seq_fetch: beat without an outstanding request");
    end
  end
`endif
endmodule
