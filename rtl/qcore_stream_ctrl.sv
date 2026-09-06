// Quettos Core stream controller: turns one GEMV / EMBED descriptor into QMEM
// read bursts and delivers the beats as the weight stream and the meta
// side-stream. Bursts of up to MAX_BURST beats are issued only into reserved
// FIFO space (FIFO_BEATS weight beats, META_FIFO_BEATS meta beats), so every
// returned beat is accepted the cycle it arrives. Weight beats leave the FIFO
// output register in order with k, tile, tile_start / tile_end, nvalid and
// last; an EMBED requests its meta beat before the tile's weight bursts and
// gathers lane TOK % WB of every beat into WB-byte pseudo-beats. Meta beats are
// serialized one 56-bit record per cycle, nvalid records per tile (the padding
// records of a partial tile are dropped). stream_done is the level after the
// last weight beat; busy covers every outstanding request and every
// undelivered beat or record. FIFO_BEATS >= MAX_BURST and META_FIFO_BEATS >= 8;
// both are powers of two; WB is 16, 64 or 128.
`include "qcore_csr_defs.svh"
module qcore_stream_ctrl #(
  parameter int WB              = 64,
  parameter int FIFO_BEATS      = 128,
  parameter int META_FIFO_BEATS = 16,
  parameter int MAX_BURST       = 64
) (
  input  logic                clk,
  input  logic                rst,
  // issue (held by the dispatcher until done)
  input  logic                cmd_valid_gemv,
  input  logic [7:0]          cmd_op,
  input  logic [31:0]         cmd_addr_a,
  input  logic [31:0]         cmd_addr_m,
  input  logic [23:0]         cmd_n,
  input  logic [15:0]         cmd_k,
  input  logic [15:0]         cmd_k_stride,
  input  logic                cmd_unit_meta,
  input  logic [31:0]         cmd_tok,
  // read requests to the arbiter
  output logic                s_req_valid,
  input  logic                s_req_ready,
  output logic [31:0]         s_req_addr,
  output logic [7:0]          s_req_len,
  output logic [3:0]          s_req_tag,
  // routed beats
  input  logic                rdw_valid,
  input  logic                rdm_valid,
  input  logic [WB*8-1:0]     rd_data,
  input  logic                rd_data_last,
  // weight stream
  output logic                ws_valid,
  input  logic                ws_ready,
  output logic [WB*8-1:0]     ws_data,
  output logic [15:0]         ws_k,
  output logic [19:0]         ws_tile,
  output logic                ws_tile_start,
  output logic                ws_tile_end,
  output logic [$clog2(WB):0] ws_nvalid,
  output logic                ws_last,
  output logic                ws_embed,
  // meta side-stream
  output logic                meta_valid,
  input  logic                meta_ready,
  output logic [55:0]         meta_data,
  // status
  output logic                stream_done,
  output logic                busy
);
  localparam int DW  = WB * 8;
  localparam int NG  = WB / 8;          // meta records per beat
  localparam int LWB = $clog2(WB);      // lane index bits
  localparam int LNG = $clog2(NG);      // record-in-beat index bits
  localparam int TW  = 20;              // tile index bits
  localparam int NVW = $clog2(WB) + 1;  // channels-in-tile bits
  localparam int WAW = $clog2(FIFO_BEATS);
  localparam int MAW = $clog2(META_FIFO_BEATS);

  localparam logic [7:0] OP_EMBED = 8'(`QCORE_OP_EMBED);

  localparam logic [1:0] S_IDLE = 2'd0;  // no descriptor
  localparam logic [1:0] S_MUL  = 2'd1;  // EMBED: forming the tile base
  localparam logic [1:0] S_W    = 2'd2;  // weight bursts of the current tile
  localparam logic [1:0] S_M    = 2'd3;  // meta burst of the current tile (first for an EMBED)

  // ---------------------------------------------------------------- descriptor
  logic           d_embed;     // EMBED descriptor
  logic           d_meta;      // a meta burst follows every tile
  logic [15:0]    d_k;         // weight beats per tile
  logic [15:0]    d_k_m1;      // K - 1
  logic [15:0]    d_kend;      // ws_k of a tile-end beat: K-1, or 0 for EMBED
  logic [31:0]    d_stride;    // bytes between tile bases: k_stride * WB
  logic [TW-1:0]  d_tiles_m1;  // tiles - 1
  logic [NVW-1:0] d_last_nv;   // channels of the last tile
  logic [LWB-1:0] d_lane;      // EMBED: the gathered lane, TOK % WB
  logic [23:0]    n_eff;       // N, or K for an EMBED
  logic [23:0]    n_m1;
  logic           issue_empty; // nothing to stream

  assign n_eff       = (cmd_op == OP_EMBED) ? {8'd0, cmd_k} : cmd_n;
  assign n_m1        = n_eff - 24'd1;
  assign issue_empty = (n_eff == 24'd0) || (cmd_k == 16'd0);
  assign d_kend      = d_embed ? 16'd0 : d_k_m1;

  always_ff @(posedge clk) begin
    if (rst) begin
      d_embed    <= 1'b0;
      d_meta     <= 1'b0;
      d_k        <= 16'd0;
      d_k_m1     <= 16'd0;
      d_stride   <= 32'd0;
      d_tiles_m1 <= {TW{1'b0}};
      d_last_nv  <= {NVW{1'b0}};
      d_lane     <= {LWB{1'b0}};
    end else if (cmd_valid_gemv) begin
      d_embed    <= (cmd_op == OP_EMBED);
      d_meta     <= (cmd_op == OP_EMBED) || !cmd_unit_meta;
      d_k        <= cmd_k;
      d_k_m1     <= cmd_k - 16'd1;
      d_stride   <= {{(16 - LWB){1'b0}}, cmd_k_stride, {LWB{1'b0}}};
      d_tiles_m1 <= TW'(n_m1 >> LWB);
      d_last_nv  <= (n_eff[LWB-1:0] == {LWB{1'b0}}) ? NVW'(WB) : {1'b0, n_eff[LWB-1:0]};
      d_lane     <= cmd_tok[LWB-1:0];
    end
  end

  // ---------------------------------------------------------------- FIFO occupancy
  logic [WAW:0] w_cnt;         // weight beats held in wmem
  logic         w_ovalid;      // weight output register holds a beat
  logic [15:0]  w_outstanding; // weight beats requested and still to return
  logic [15:0]  w_free;        // weight beats a new burst may reserve
  logic [MAW:0] m_cnt;
  logic         m_ovalid;
  logic [15:0]  m_outstanding;
  logic [15:0]  m_free;

  assign w_free = 16'(FIFO_BEATS) - 16'(w_cnt) - {15'd0, w_ovalid} - w_outstanding;
  assign m_free = 16'(META_FIFO_BEATS) - 16'(m_cnt) - {15'd0, m_ovalid} - m_outstanding;

  // ---------------------------------------------------------------- request generator
  logic [1:0]    st;
  logic [TW-1:0] r_tile;    // tile being requested
  logic [31:0]   r_addr;    // next weight burst address
  logic [31:0]   r_base;    // base of the current tile
  logic [15:0]   r_rem;     // weight beats still to request in this tile
  logic [31:0]   r_maddr;   // meta address of the current tile
  logic [31:0]   mul_m;     // EMBED: (TOK / WB) * WB, shifted per step
  logic [15:0]   mul_b;     // EMBED: K, shifted per step
  logic [3:0]    mul_i;
  logic [31:0]   base_n;    // r_base after this multiply step
  logic [7:0]    w_len;     // length of the next weight burst
  logic [7:0]    m_len;     // length of the meta burst
  logic          req_slot;  // the request register can take a new request
  logic          load_w;
  logic          load_m;
  logic          w_tile_done;
  logic          tile_last;

  assign base_n      = r_base + (mul_b[0] ? mul_m : 32'd0);
  assign w_len       = (r_rem > 16'(MAX_BURST)) ? 8'(MAX_BURST) : r_rem[7:0];
  assign m_len       = d_embed ? 8'd1 : 8'd8;
  assign req_slot    = !s_req_valid || s_req_ready;
  assign load_w      = req_slot && (st == S_W) && (w_free >= {8'd0, w_len});
  assign load_m      = req_slot && (st == S_M) && (m_free >= {8'd0, m_len});
  assign w_tile_done = (r_rem == {8'd0, w_len});
  assign tile_last   = d_embed || (r_tile == d_tiles_m1);  // an EMBED requests one tile

  always_ff @(posedge clk) begin
    if (rst) begin
      st          <= S_IDLE;
      s_req_valid <= 1'b0;
      s_req_addr  <= 32'd0;
      s_req_len   <= 8'd0;
      s_req_tag   <= 4'd0;
      r_tile      <= {TW{1'b0}};
      r_addr      <= 32'd0;
      r_base      <= 32'd0;
      r_rem       <= 16'd0;
      r_maddr     <= 32'd0;
      mul_m       <= 32'd0;
      mul_b       <= 16'd0;
      mul_i       <= 4'd0;
    end else if (cmd_valid_gemv) begin
      r_tile <= {TW{1'b0}};
      r_rem  <= cmd_k;
      r_base <= cmd_addr_a;
      r_addr <= cmd_addr_a;
      mul_i  <= 4'd0;
      if (cmd_op == OP_EMBED) begin
        r_maddr <= cmd_addr_m + {cmd_tok[28:0], 3'b000};
        mul_m   <= {cmd_tok[31:LWB], {LWB{1'b0}}};
        mul_b   <= cmd_k;
        st      <= issue_empty ? S_IDLE : S_MUL;
      end else begin
        r_maddr <= cmd_addr_m;
        mul_m   <= 32'd0;
        mul_b   <= 16'd0;
        st      <= issue_empty ? S_IDLE : S_W;
      end
    end else begin
      case (st)
        S_MUL: begin
          r_base <= base_n;
          mul_m  <= {mul_m[30:0], 1'b0};
          mul_b  <= {1'b0, mul_b[15:1]};
          mul_i  <= mul_i + 4'd1;
          if (mul_i == 4'd15) begin
            r_addr <= base_n;
            st     <= S_M;
          end
        end
        S_W: begin
          if (load_w) begin
            r_addr <= r_addr + {{(24 - LWB){1'b0}}, w_len, {LWB{1'b0}}};
            r_rem  <= r_rem - {8'd0, w_len};
            if (w_tile_done) begin
              if (d_meta && !d_embed) begin
                st <= S_M;
              end else if (tile_last) begin
                st <= S_IDLE;
              end else begin
                r_tile  <= r_tile + {{(TW - 1){1'b0}}, 1'b1};
                r_base  <= r_base + d_stride;
                r_addr  <= r_base + d_stride;
                r_rem   <= d_k;
                r_maddr <= r_maddr + 32'(WB * 8);
              end
            end
          end
        end
        S_M: begin
          if (load_m) begin
            if (d_embed) begin
              st <= S_W;
            end else if (tile_last) begin
              st <= S_IDLE;
            end else begin
              r_tile  <= r_tile + {{(TW - 1){1'b0}}, 1'b1};
              r_base  <= r_base + d_stride;
              r_addr  <= r_base + d_stride;
              r_rem   <= d_k;
              r_maddr <= r_maddr + 32'(WB * 8);
              st      <= S_W;
            end
          end
        end
        default: begin
        end
      endcase
      if (load_w) begin
        s_req_valid <= 1'b1;
        s_req_addr  <= r_addr;
        s_req_len   <= w_len;
        s_req_tag   <= qcore_pkg::TAG_WEIGHT;
      end else if (load_m) begin
        s_req_valid <= 1'b1;
        s_req_addr  <= r_maddr;
        s_req_len   <= m_len;
        s_req_tag   <= qcore_pkg::TAG_META;
      end else if (req_slot) begin
        s_req_valid <= 1'b0;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      w_outstanding <= 16'd0;
      m_outstanding <= 16'd0;
    end else begin
      w_outstanding <= w_outstanding + (load_w ? {8'd0, w_len} : 16'd0) - {15'd0, rdw_valid};
      m_outstanding <= m_outstanding + (load_m ? {8'd0, m_len} : 16'd0) - {15'd0, rdm_valid};
    end
  end

  // ---------------------------------------------------------------- EMBED gather
  logic [DW-1:0]  g_acc;   // pseudo-beat under assembly
  logic [DW-1:0]  g_next;  // g_acc with this beat's byte inserted
  logic [LWB-1:0] g_lane;  // lane the next byte fills
  logic [15:0]    g_i;     // index of the next byte
  logic [7:0]     g_byte;  // this beat's byte at lane TOK % WB
  logic           g_done;  // this beat completes a pseudo-beat

  assign g_byte = rd_data[{d_lane, 3'b000} +: 8];
  assign g_done = (&g_lane) || (g_i == d_k_m1);

  always_comb begin
    for (int j = 0; j < WB; j++) begin
      g_next[8*j +: 8] = (g_lane == LWB'(j)) ? g_byte : g_acc[8*j +: 8];
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      g_acc  <= {DW{1'b0}};
      g_lane <= {LWB{1'b0}};
      g_i    <= 16'd0;
    end else if (cmd_valid_gemv) begin
      g_acc  <= {DW{1'b0}};
      g_lane <= {LWB{1'b0}};
      g_i    <= 16'd0;
    end else if (rdw_valid && d_embed) begin
      g_acc  <= g_done ? {DW{1'b0}} : g_next;
      g_lane <= g_lane + {{(LWB - 1){1'b0}}, 1'b1};
      g_i    <= g_i + 16'd1;
    end
  end

  // ---------------------------------------------------------------- weight FIFO
  logic [DW-1:0]  wmem [0:FIFO_BEATS-1];
  logic [WAW-1:0] w_wptr;
  logic [WAW-1:0] w_rptr;
  logic [DW-1:0]  w_odata;
  logic [DW-1:0]  w_wdata;
  logic           w_wr;
  logic           w_rd;
  logic           w_pop;

  assign w_wdata = d_embed ? g_next : rd_data;
  assign w_wr    = rdw_valid && (!d_embed || g_done);
  assign w_pop   = w_ovalid && ws_ready;
  assign w_rd    = (w_cnt != {(WAW + 1){1'b0}}) && (!w_ovalid || w_pop);

  always_ff @(posedge clk) begin
    if (w_wr) wmem[w_wptr] <= w_wdata;
  end

  always_ff @(posedge clk) begin
    if (w_rd) w_odata <= wmem[w_rptr];
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      w_wptr   <= {WAW{1'b0}};
      w_rptr   <= {WAW{1'b0}};
      w_cnt    <= {(WAW + 1){1'b0}};
      w_ovalid <= 1'b0;
    end else begin
      if (w_wr) w_wptr <= (w_wptr == WAW'(FIFO_BEATS - 1)) ? {WAW{1'b0}} : w_wptr + WAW'(1);
      if (w_rd) w_rptr <= (w_rptr == WAW'(FIFO_BEATS - 1)) ? {WAW{1'b0}} : w_rptr + WAW'(1);
      w_cnt <= w_cnt + {{WAW{1'b0}}, w_wr} - {{WAW{1'b0}}, w_rd};
      if (w_rd)       w_ovalid <= 1'b1;
      else if (w_pop) w_ovalid <= 1'b0;
    end
  end

  // ---------------------------------------------------------------- weight stream
  logic [15:0]   o_k;
  logic [TW-1:0] o_tile;
  logic          o_tile_last;

  assign o_tile_last   = (o_tile == d_tiles_m1);
  assign ws_valid      = w_ovalid;
  assign ws_data       = w_odata;
  assign ws_k          = o_k;
  assign ws_tile       = o_tile;
  assign ws_tile_start = (o_k == 16'd0);
  assign ws_tile_end   = (o_k == d_kend);
  assign ws_nvalid     = o_tile_last ? d_last_nv : NVW'(WB);
  assign ws_last       = ws_tile_end && o_tile_last;
  assign ws_embed      = d_embed;

  always_ff @(posedge clk) begin
    if (rst) begin
      o_k         <= 16'd0;
      o_tile      <= {TW{1'b0}};
      stream_done <= 1'b0;
    end else if (cmd_valid_gemv) begin
      o_k         <= 16'd0;
      o_tile      <= {TW{1'b0}};
      stream_done <= issue_empty;
    end else if (w_pop) begin
      if (ws_tile_end) begin
        o_k    <= 16'd0;
        o_tile <= o_tile + {{(TW - 1){1'b0}}, 1'b1};
      end else begin
        o_k <= o_k + 16'd1;
      end
      if (ws_last) stream_done <= 1'b1;
    end
  end

  // ---------------------------------------------------------------- meta FIFO
  logic [DW-1:0]  mmem [0:META_FIFO_BEATS-1];
  logic [MAW-1:0] m_wptr;
  logic [MAW-1:0] m_rptr;
  logic [DW-1:0]  m_odata;
  logic           m_rd;
  logic           m_pop;

  always_ff @(posedge clk) begin
    if (rdm_valid) mmem[m_wptr] <= rd_data;
  end

  always_ff @(posedge clk) begin
    if (m_rd) m_odata <= mmem[m_rptr];
  end

  assign m_rd = (m_cnt != {(MAW + 1){1'b0}}) && (!m_ovalid || m_pop);

  always_ff @(posedge clk) begin
    if (rst) begin
      m_wptr   <= {MAW{1'b0}};
      m_rptr   <= {MAW{1'b0}};
      m_cnt    <= {(MAW + 1){1'b0}};
      m_ovalid <= 1'b0;
    end else begin
      if (rdm_valid) m_wptr <= (m_wptr == MAW'(META_FIFO_BEATS - 1)) ? {MAW{1'b0}} : m_wptr + MAW'(1);
      if (m_rd)      m_rptr <= (m_rptr == MAW'(META_FIFO_BEATS - 1)) ? {MAW{1'b0}} : m_rptr + MAW'(1);
      m_cnt <= m_cnt + {{MAW{1'b0}}, rdm_valid} - {{MAW{1'b0}}, m_rd};
      if (m_rd)       m_ovalid <= 1'b1;
      else if (m_pop) m_ovalid <= 1'b0;
    end
  end

  // ---------------------------------------------------------------- meta serializer
  // Record j = {beat, lane} of tile ms_tile; records at or past the tile's
  // channel count are padding and their beats are dropped.
  logic [TW-1:0]  ms_tile;
  logic [2:0]     ms_beat;
  logic [LNG-1:0] ms_lane;
  logic [2:0]     ms_beats_m1;  // beats per tile - 1
  logic [NVW-1:0] ms_recs;      // records of this tile
  logic           ms_present;
  logic           ms_skip;
  logic           ms_take;
  logic           ms_next_beat;

  assign ms_beats_m1  = d_embed ? 3'd0 : 3'd7;
  assign ms_recs      = d_embed ? NVW'(1) : ((ms_tile == d_tiles_m1) ? d_last_nv : NVW'(WB));
  assign ms_present   = m_ovalid && ({1'b0, ms_beat, ms_lane} < ms_recs);
  assign ms_skip      = m_ovalid && !({1'b0, ms_beat, ms_lane} < ms_recs);
  assign ms_take      = ms_present && meta_ready;
  assign ms_next_beat = (ms_take && (&ms_lane)) || ms_skip;
  assign m_pop        = ms_next_beat;
  assign meta_valid   = ms_present;
  assign meta_data    = m_odata[{ms_lane, 6'b000000} +: 56];

  always_ff @(posedge clk) begin
    if (rst) begin
      ms_tile <= {TW{1'b0}};
      ms_beat <= 3'd0;
      ms_lane <= {LNG{1'b0}};
    end else if (cmd_valid_gemv) begin
      ms_tile <= {TW{1'b0}};
      ms_beat <= 3'd0;
      ms_lane <= {LNG{1'b0}};
    end else if (ms_next_beat) begin
      ms_lane <= {LNG{1'b0}};
      if (ms_beat == ms_beats_m1) begin
        ms_beat <= 3'd0;
        ms_tile <= ms_tile + {{(TW - 1){1'b0}}, 1'b1};
      end else begin
        ms_beat <= ms_beat + 3'd1;
      end
    end else if (ms_take) begin
      ms_lane <= ms_lane + {{(LNG - 1){1'b0}}, 1'b1};
    end
  end

  // ---------------------------------------------------------------- status
  assign busy = (st != S_IDLE) || s_req_valid ||
                (w_outstanding != 16'd0) || (m_outstanding != 16'd0) ||
                w_ovalid || (w_cnt != {(WAW + 1){1'b0}}) ||
                m_ovalid || (m_cnt != {(MAW + 1){1'b0}});

`ifndef SYNTHESIS
  // Simulation-only protocol checks: reserved space is never exceeded, beats
  // arrive only for outstanding requests, and every burst ends with last.
  int chk_bursts;
  always @(posedge clk) begin
    if (rst) begin
      chk_bursts <= 0;
    end else begin
      chk_bursts <= chk_bursts + ((load_w || load_m) ? 1 : 0)
                    - (((rdw_valid || rdm_valid) && rd_data_last) ? 1 : 0);
      if (w_wr && (w_cnt == (WAW + 1)'(FIFO_BEATS)))
        $error("qcore_stream_ctrl: weight FIFO overflow");
      if (rdm_valid && (m_cnt == (MAW + 1)'(META_FIFO_BEATS)))
        $error("qcore_stream_ctrl: meta FIFO overflow");
      if (rdw_valid && (w_outstanding == 16'd0))
        $error("qcore_stream_ctrl: weight beat without an outstanding request");
      if (rdm_valid && (m_outstanding == 16'd0))
        $error("qcore_stream_ctrl: meta beat without an outstanding request");
      if ((rdw_valid || rdm_valid) && rd_data_last && (chk_bursts == 0))
        $error("qcore_stream_ctrl: burst end without an open burst");
    end
  end
`endif
endmodule
