// Quettos Core activation row: WB output-stationary MAC lanes in WB/8 lane
// groups, the activation word ring that feeds them, the tile handoff to the
// requant and the row's SREG bank. From the issue on, VSRAM port A is read one
// word ahead of the stream into a four-word ring (wrapping to the word of
// vs_src after the last word of a tile), so an aligned or unaligned vs_src
// streams one beat per cycle; ws_ready is low until the word of the beat's k
// is present and, on a ws_tile_end beat, while acc_valid || !acc_ready. The
// cycle after a tile-end beat is accepted acc_flat shows the finished tile and
// acc_valid rises until the acc_valid && acc_ready transfer; acc_flat then
// holds until the next tile end. A non-participating row accepts every beat
// and does nothing else. SREG reads return data the cycle after sreg_rd_en.
module qcore_row #(
  parameter int WB          = 64,
  parameter int ACC_W       = 40,
  parameter int VSRAM_WORDS = 4096
) (
  input  logic                    clk,
  input  logic                    rst,
  // issue
  input  logic                    cmd_valid_gemv,
  input  logic                    row_active,
  input  logic [15:0]             cmd_vs_src,
  input  logic [15:0]             cmd_k,
  // weight stream
  input  logic                    ws_valid,
  output logic                    ws_ready,
  input  logic [WB*8-1:0]         ws_data,
  input  logic [15:0]             ws_k,
  input  logic [19:0]             ws_tile,
  input  logic                    ws_tile_start,
  input  logic                    ws_tile_end,
  input  logic [$clog2(WB):0]     ws_nvalid,
  input  logic                    ws_last,
  input  logic                    ws_embed,
  // VSRAM port A of the row's bank
  output logic                    vsa_en,
  output logic [$clog2(VSRAM_WORDS)-1:0] vsa_addr,
  input  logic [255:0]            vsa_rdata,
  // accumulator handoff
  output logic                    acc_valid,
  input  logic                    acc_ready,
  output logic [WB*ACC_W-1:0]     acc_flat,
  output logic [19:0]             acc_tile,
  output logic [$clog2(WB):0]     acc_nvalid,
  output logic                    acc_last,
  // SREG bank
  input  logic                    sreg_rd_en,
  input  logic [7:0]              sreg_rd_idx,
  output logic [31:0]             sreg_rd_data,
  input  logic                    sreg_wr_en,
  input  logic [7:0]              sreg_wr_idx,
  input  logic [31:0]             sreg_wr_data,
  output logic                    sreg_err,
  // events
  output logic                    err_bounds,
  output logic                    ev_beat
);
  localparam int NG  = WB / 8;               // lane groups
  localparam int AW  = $clog2(VSRAM_WORDS);  // word address
  localparam int NW  = 4;                    // ring words
  localparam int NWB = 2;                    // ring index bits
  localparam int WW  = 14;                   // word index bits (17-bit elements)

  // ------------------------------------------------------------------ descriptor state
  logic          active;      // participating and streaming
  logic [15:0]   vs_src_q;
  logic [WW-1:0] w0;          // word of vs_src
  logic [WW-1:0] wlast;       // word of vs_src + K - 1
  logic [WW-1:0] fword;       // next word to fetch
  logic [16:0]   e_end;
  logic [16:0]   e_last;
  logic [15:0]   k_m1;

  assign k_m1   = (cmd_k == 16'd0) ? 16'd0 : cmd_k - 16'd1;
  assign e_end  = {1'b0, cmd_vs_src} + {1'b0, cmd_k};
  assign e_last = {1'b0, cmd_vs_src} + {1'b0, k_m1};

  // ------------------------------------------------------------------ activation ring
  logic [NW*256-1:0] ring;
  logic [NWB-1:0]    head;
  logic [NWB-1:0]    tail;
  logic [NWB:0]      count;       // landed, unconsumed words
  logic              pend;        // a read lands this cycle
  logic [NWB-1:0]    pslot;
  logic              pzero;       // the landing word lies past VSRAM_WORDS
  logic              do_issue;
  logic              in_range;
  logic [2:0]        slot;
  logic [16*NW-1:0]  ent_a;
  logic [15:0]       a_bcast;
  logic              head_valid;
  logic              accept;
  logic              en;
  logic              pop;
  logic              flush;

  assign in_range   = (fword < WW'(VSRAM_WORDS));
  assign do_issue   = active && ({1'b0, count} + {{NWB{1'b0}}, 1'b0, pend} < (NWB+2)'(NW));
  assign vsa_en     = do_issue && in_range;
  assign vsa_addr   = fword[AW-1:0];
  assign head_valid = (count != {(NWB+1){1'b0}});
  assign slot       = vs_src_q[2:0] + ws_k[2:0];

  genvar e;
  generate
    for (e = 0; e < NW; e++) begin : g_ent
      assign ent_a[e*16 +: 16] = ring[{NWB'(e), slot, 5'd0} +: 16];
    end
  endgenerate
  assign a_bcast = ent_a[{head, 4'd0} +: 16];

  // ------------------------------------------------------------------ stream handshake
  assign ws_ready = !active
                  || ((ws_embed || head_valid) && !(ws_tile_end && (acc_valid || !acc_ready)));
  assign accept   = ws_valid && ws_ready;
  assign en       = accept && active;
  assign pop      = en && !ws_embed && (slot == 3'd7 || ws_tile_end);
  assign flush    = en && ws_last;
  assign ev_beat  = en;

  always_ff @(posedge clk) begin
    if (rst) begin
      active     <= 1'b0;
      head       <= {NWB{1'b0}};
      tail       <= {NWB{1'b0}};
      count      <= {(NWB+1){1'b0}};
      pend       <= 1'b0;
      err_bounds <= 1'b0;
    end else if (cmd_valid_gemv) begin
      active     <= row_active && (cmd_k != 16'd0);
      head       <= {NWB{1'b0}};
      tail       <= {NWB{1'b0}};
      count      <= {(NWB+1){1'b0}};
      pend       <= 1'b0;
      err_bounds <= row_active && (e_end > 17'(VSRAM_WORDS * 8));
    end else if (flush) begin
      active     <= 1'b0;
      head       <= {NWB{1'b0}};
      tail       <= {NWB{1'b0}};
      count      <= {(NWB+1){1'b0}};
      pend       <= 1'b0;
      err_bounds <= 1'b0;
    end else begin
      head       <= head + {{(NWB-1){1'b0}}, pop};
      tail       <= tail + {{(NWB-1){1'b0}}, do_issue};
      count      <= count + {{NWB{1'b0}}, pend} - {{NWB{1'b0}}, pop};
      pend       <= do_issue;
      err_bounds <= 1'b0;
    end
  end

  always_ff @(posedge clk) begin
    if (cmd_valid_gemv) begin
      vs_src_q <= cmd_vs_src;
      w0       <= {1'b0, cmd_vs_src[15:3]};
      wlast    <= WW'(e_last >> 3);
      fword    <= {1'b0, cmd_vs_src[15:3]};
    end else if (do_issue) begin
      fword    <= (fword == wlast) ? w0 : fword + {{(WW-1){1'b0}}, 1'b1};
    end
    if (do_issue) begin
      pslot <= tail;
      pzero <= !in_range;
    end
  end

  always_ff @(posedge clk) begin
    for (int i = 0; i < NW; i++) begin
      if (pend && pslot == NWB'(i)) ring[i*256 +: 256] <= pzero ? 256'd0 : vsa_rdata;
    end
  end

  // ------------------------------------------------------------------ lane groups
  logic buf_sel;   // the live accumulators hold a finished tile

  always_ff @(posedge clk) begin
    if (rst)     buf_sel <= 1'b0;
    else if (en) buf_sel <= ws_tile_end;
  end

  genvar g;
  generate
    for (g = 0; g < NG; g++) begin : g_grp
      qcore_mac_lane_group #(.ACC_W(ACC_W)) u_grp (
        .clk       (clk),
        .en        (en),
        .tile_start(ws_tile_start),
        .embed     (ws_embed),
        .buf_sel   (buf_sel),
        .w         (ws_data[g*64 +: 64]),
        .a         (a_bcast),
        .acc_drain (acc_flat[g*8*ACC_W +: 8*ACC_W])
      );
    end
  endgenerate

  // ------------------------------------------------------------------ tile handoff
  always_ff @(posedge clk) begin
    if (rst) begin
      acc_valid <= 1'b0;
    end else if (en && ws_tile_end) begin
      acc_valid <= 1'b1;
    end else if (acc_valid && acc_ready) begin
      acc_valid <= 1'b0;
    end
  end

  always_ff @(posedge clk) begin
    if (en && ws_tile_end) begin
      acc_tile   <= ws_tile;
      acc_nvalid <= ws_nvalid;
      acc_last   <= ws_last;
    end
  end

  // ------------------------------------------------------------------ SREG bank
  logic [31:0] sreg [0:31];
  logic [31:0] sreg_rd_word;
  logic        sreg_rd_zero;
  logic        rd_bad;
  logic        wr_bad;

  assign rd_bad       = (sreg_rd_idx[7:5] != 3'd0);
  assign wr_bad       = (sreg_wr_idx[7:5] != 3'd0);
  assign sreg_rd_data = sreg_rd_zero ? 32'd0 : sreg_rd_word;

  always_ff @(posedge clk) begin
    if (sreg_rd_en) sreg_rd_word <= sreg[sreg_rd_idx[4:0]];
  end

  always_ff @(posedge clk) begin
    if (sreg_wr_en && !wr_bad) sreg[sreg_wr_idx[4:0]] <= sreg_wr_data;
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      sreg_rd_zero <= 1'b0;
      sreg_err     <= 1'b0;
    end else begin
      if (sreg_rd_en) sreg_rd_zero <= rd_bad;
      sreg_err <= (sreg_rd_en && rd_bad) || (sreg_wr_en && wr_bad);
    end
  end

`ifndef SYNTHESIS
  // Simulation-only checks: the stream's k sequence must match the ring's
  // word sequence, and a descriptor is never issued to a streaming row. The
  // block is held in reset like every register above it, so what it reads is
  // state the reset has defined and the checks hold from any start value.
  logic [WW-1:0] hword;
  logic [16:0]   e_cur;
  assign e_cur = {1'b0, vs_src_q} + {1'b0, ws_k};
  always @(posedge clk) begin
    if (rst) begin
      hword <= {WW{1'b0}};
    end else begin
      if (cmd_valid_gemv) begin
        hword <= {1'b0, cmd_vs_src[15:3]};
        if (active) $error("qcore_row: issue while streaming");
      end else if (pop) begin
        hword <= (hword == wlast) ? w0 : hword + {{(WW-1){1'b0}}, 1'b1};
      end
      if (en && !ws_embed) begin
        if (!head_valid) $error("qcore_row: beat accepted without its activation word");
        if (WW'(e_cur >> 3) != hword)
          $error("qcore_row: beat k=%0d needs word %0d, ring head is word %0d",
                 ws_k, WW'(e_cur >> 3), hword);
      end
    end
  end
`endif
endmodule
