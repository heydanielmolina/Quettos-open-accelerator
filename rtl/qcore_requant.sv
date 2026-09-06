// Quettos Core requant: drains the finished accumulator tiles of the MAC rows
// one channel per cycle through the two-stage sfloat requantization of
// numerics.requant (stage 1: acc * Sw_m round-shifted by s1 into 40 bits;
// stage 2: * Sx_m round-shifted by S = sbias - (Sw_e + Sx_e), clamped to
// [0, 63], into 32 bits; a zero scale gives 0; then + bias_q and the
// accumulate add of the old VSRAM value), assembles the outputs into VSRAM
// words, dump beats, the per-row running argmax and absmax, and counts every
// saturation, shift clamp and range error as per-cycle increments.
// Timing: one element per cycle while the meta stream, VSRAM port B and the
// dump port keep up, so a tile of n channels drains in n cycles per
// participating row and the next handshake follows one cycle later. The
// element read from acc_flat in cycle T leaves the arithmetic in cycle T+7,
// the word it completes is written on port B in cycle T+8 and the dump beat
// it completes is offered in cycle T+9; the accumulate old-word read is on
// port B in cycle T+1. Port B outputs, event counts, SREG / ARGMAX writes and
// done are registered; acc_ready and meta_ready are functions of internal
// registers, except that acc_ready in the final-read cycle of a tile also
// follows meta_valid. The meta stream carries nvalid records per GEMV tile
// (the stream controller drops the records of the padded channels).
`include "qcore_csr_defs.svh"
module qcore_requant #(
  parameter int WB          = 64,
  parameter int B_MAX       = 1,
  parameter int ACC_W       = 40,
  parameter int VSRAM_WORDS = 4096
) (
  input  logic                           clk,
  input  logic                           rst,
  // descriptor issue; the bundle holds from the pulse to done
  input  logic                           cmd_valid_gemv,
  input  logic [7:0]                     cmd_op,
  input  logic [1:0]                     cmd_out_mode,
  input  logic                           cmd_accumulate,
  input  logic                           cmd_unit_meta,
  input  logic                           cmd_track_absmax,
  input  logic [23:0]                    cmd_n,
  input  logic [15:0]                    cmd_vs_dst,
  input  logic [7:0]                     cmd_sreg_dst,
  input  logic [7:0]                     cmd_sh0,
  input  logic signed [7:0]              cmd_sh1,
  input  logic [31:0]                    cmd_imm32,
  input  logic [B_MAX-1:0]               cmd_rows,
  input  logic [B_MAX*16-1:0]            cmd_sx_m,
  input  logic [B_MAX*8-1:0]             cmd_sx_e,
  // accumulator handoff from the rows
  input  logic                           acc_valid,
  output logic                           acc_ready,
  input  logic [B_MAX*WB*ACC_W-1:0]      acc_flat,
  input  logic [19:0]                    acc_tile,
  input  logic [$clog2(WB):0]            acc_nvalid,
  input  logic                           acc_last,
  // meta side-stream: {e[7:0], m[15:0], bias_q[31:0]}
  input  logic                           meta_valid,
  output logic                           meta_ready,
  input  logic [55:0]                    meta_data,
  // VSRAM port B of bank dst_row + vsb_row; vsb_row holds for the cycle after a read
  output logic [3:0]                     vsb_row,
  output logic                           vsb_en,
  output logic [7:0]                     vsb_we,
  output logic [$clog2(VSRAM_WORDS)-1:0] vsb_addr,
  output logic [255:0]                   vsb_wdata,
  input  logic [255:0]                   vsb_rdata,
  // dump beats
  output logic                           d_wr_valid,
  input  logic                           d_wr_ready,
  output logic [31:0]                    d_wr_addr,
  output logic [WB*8-1:0]                d_wr_data,
  output logic [WB-1:0]                  d_wr_strb,
  // tracked absmax to SREG bank dst_row + sreg_wr_row
  output logic                           sreg_wr_en,
  output logic [3:0]                     sreg_wr_row,
  output logic [7:0]                     sreg_wr_idx,
  output logic [31:0]                    sreg_wr_data,
  // ARGMAX CSR write
  output logic                           argmax_we,
  output logic [31:0]                    argmax_tok,
  output logic [31:0]                    argmax_val,
  // per-cycle event increments: saturations (0..4), shift clamps (0..2), range errors (0..2)
  output logic [2:0]                     sat_inc,
  output logic [1:0]                     err_shift_inc,
  output logic [1:0]                     err_bounds_inc,
  output logic                           done,
  output logic                           busy
);
  localparam int LWB = $clog2(WB);            // channel index within a tile
  localparam int NVW = LWB + 1;               // channel count of a tile
  localparam int TW  = 20;                    // tile index
  localparam int CW  = TW + LWB;              // channel index c = tile * WB + j
  localparam int RW  = 4;                     // physical row index on the ports
  localparam int RIW = (B_MAX > 1) ? $clog2(B_MAX) : 1;
  localparam int AW  = $clog2(VSRAM_WORDS);
  localparam int DW  = WB * 8;
  localparam int EPB = WB / 4;                // int32 elements per dump beat
  localparam int LEB = LWB - 2;
  localparam int NEL = B_MAX * WB;            // accumulators on acc_flat
  localparam int AIW = $clog2(NEL);
  localparam int WWD = CW - 2;                // VSRAM word arithmetic
  localparam int DQD = 4;                     // dump beats queued
  localparam int DQP = $clog2(DQD);
  localparam int DEW = 32 + WB + DW;          // one dump beat: addr, strb, data

  localparam logic [7:0] OP_EMBED        = 8'(`QCORE_OP_EMBED);
  localparam logic [1:0] OUT_VSRAM       = 2'(`QCORE_OUT_VSRAM);
  localparam logic [1:0] OUT_ARGMAX      = 2'(`QCORE_OUT_ARGMAX);
  localparam logic [1:0] OUT_ARGMAX_DUMP = 2'(`QCORE_OUT_ARGMAX_DUMP);
  localparam logic [1:0] OUT_VSRAM_DUMP  = 2'(`QCORE_OUT_VSRAM_DUMP);

  typedef enum logic [2:0] {
    S_IDLE   = 3'd0,
    S_WAIT   = 3'd1,
    S_DRAIN  = 3'd2,
    S_EMPTY  = 3'd3,
    S_REPORT = 3'd4,
    S_DONE   = 3'd5
  } state_t;

  // ------------------------------------------------------------------ descriptor
  state_t             state;
  logic               d_embed, d_vsram, d_argmax, d_dump, d_acc, d_unit, d_track;
  logic [12:0]        d_vsw;          // vs_dst[15:3]: word base
  logic [7:0]         d_sreg;
  logic [5:0]         d_s1;
  logic               d_s1_err;       // sh0 > 63
  logic signed [7:0]  d_sbias;
  logic [B_MAX-1:0]   d_rows;
  logic [B_MAX*16-1:0] d_sx_m;
  logic [B_MAX*8-1:0]  d_sx_e;
  logic [RIW-1:0]     d_first_row;
  logic               d_oob;          // vs_dst + N past the VSRAM
  logic [B_MAX*32-1:0] d_dbase;       // dump base per row: addr_c + r * 4 * N
  logic [55:0]        emb_rec;
  logic               emb_ok;

  logic [RIW-1:0]     first_row_c;
  logic [B_MAX*32-1:0] dbase_c;
  logic [31:0]        n4;
  logic [31:0]        base_tmp;

  assign n4 = {6'd0, cmd_n, 2'd0};

  always_comb begin
    first_row_c = {RIW{1'b0}};
    for (int r = B_MAX - 1; r >= 0; r--) begin
      if (cmd_rows[r]) first_row_c = RIW'(r);
    end
    base_tmp = cmd_imm32;
    dbase_c  = {(B_MAX*32){1'b0}};
    for (int r = 0; r < B_MAX; r++) begin
      dbase_c[r*32 +: 32] = base_tmp;
      base_tmp = base_tmp + n4;
    end
  end

  // ------------------------------------------------------------------ sequencer state
  logic [RIW-1:0] cur_row;
  logic [LWB-1:0] j;
  logic [TW-1:0]  t_tile;
  logic [NVW-1:0] t_nvalid;
  logic           t_last;
  logic [RIW-1:0] rep_row;
  logic           rep_step;

  logic [RIW-1:0] nxt_row;
  logic           nxt_any;

  always_comb begin
    nxt_row = {RIW{1'b0}};
    nxt_any = 1'b0;
    for (int r = B_MAX - 1; r >= 0; r--) begin
      if (d_rows[r] && (RIW'(r) > cur_row)) begin
        nxt_row = RIW'(r);
        nxt_any = 1'b1;
      end
    end
  end

  // ------------------------------------------------------------------ pipeline registers
  // P1: operands selected; P2: stage-1 product; P3: t; P4: stage-2 product;
  // P5: y after stage 2; P6: + bias; P7: + old (final value).
  logic               p1_v, p2_v, p3_v, p4_v, p5_v, p6_v, p7_v;
  logic signed [39:0] p1_acc;
  logic [15:0]        p1_swm;
  logic [7:0]         p1_swe;
  logic [31:0]        p1_bias, p2_bias, p3_bias, p4_bias, p5_bias;
  logic [CW-1:0]      p1_c, p2_c, p3_c, p4_c, p5_c, p6_c, p7_c;
  logic [RIW-1:0]     p1_row, p2_row, p3_row, p4_row, p5_row, p6_row, p7_row;
  logic [AW-1:0]      p1_waddr, p2_waddr, p3_waddr, p4_waddr, p5_waddr, p6_waddr, p7_waddr;
  logic               p1_oob, p2_oob, p3_oob, p4_oob, p5_oob, p6_oob, p7_oob;
  logic               p1_wfirst, p2_wfirst;
  logic               p1_wlast, p2_wlast, p3_wlast, p4_wlast, p5_wlast, p6_wlast, p7_wlast;
  logic               p1_blast, p2_blast, p3_blast, p4_blast, p5_blast, p6_blast, p7_blast;
  logic signed [56:0] p2_p;
  logic               p2_nz, p3_nz, p4_nz, p5_nz, p6_nz, p7_nz;
  logic [5:0]         p2_s, p3_s, p4_s;
  logic               p2_serr, p3_serr, p4_serr, p5_serr, p6_serr, p7_serr;
  logic signed [39:0] p3_t;
  logic [31:0]        p3_old, p4_old, p5_old, p6_old;
  logic [2:0]         p3_satc, p4_satc, p5_satc, p6_satc, p7_satc;
  logic signed [56:0] p4_p;
  logic [31:0]        p5_y, p6_y, p7_y;
  logic [255:0]       old_word_q;

  // ------------------------------------------------------------------ dump beat queue
  // DQD beats in a small memory behind a registered output word (d_wr_*);
  // the arithmetic pipeline freezes while the memory is full.
  logic [DEW-1:0] dq_mem [0:DQD-1];
  logic [DQP-1:0] dq_wp, dq_rp;
  logic [2:0]     dq_cnt;
  logic           dq_full, dq_push, dq_pop, dq_rd, dq_out_v;
  logic [DEW-1:0] dq_out;
  logic           adv;

  assign dq_full = (dq_cnt == 3'(DQD));
  assign adv     = !dq_full;

  // ------------------------------------------------------------------ port B registers
  logic          vsb_en_q;
  logic [AW-1:0] rd_addr_q;
  logic [RIW-1:0] rd_row_q;
  logic [7:0]    wr_we_q;
  logic [AW-1:0] wr_addr_q;
  logic [255:0]  wr_wdata_q;
  logic [RIW-1:0] wr_row_q;
  logic [RW-1:0] hold_row_q;
  logic          wr_act;

  assign wr_act    = |wr_we_q;
  assign vsb_en    = vsb_en_q;
  assign vsb_we    = wr_we_q;
  assign vsb_wdata = wr_wdata_q;
  assign vsb_addr  = wr_act ? wr_addr_q : rd_addr_q;
  assign vsb_row   = wr_act ? RW'(wr_row_q) : (vsb_en_q ? RW'(rd_row_q) : hold_row_q);

  // ------------------------------------------------------------------ stage 0: element select
  logic [CW-1:0]  s0_c;
  logic           j_last, s0_rfinal, s0_wfirst, s0_wlast, s0_blast, s0_oob, tile_final, row_start;
  logic [WWD-1:0] s0_wsum;
  logic [AW-1:0]  s0_waddr;
  logic [AIW-1:0] acc_idx;
  logic [ACC_W-1:0] acc_sel;
  logic [55:0]    mbuf_rd, m_rec;
  logic           row_first, need_stream, need_emb, need_rd, rd_block, s0_ok, fire, emb_wait;

  assign s0_c       = {t_tile, j};
  assign j_last     = ({1'b0, j} == (t_nvalid - NVW'(1)));
  assign s0_rfinal  = t_last && j_last;
  assign s0_wfirst  = (j[2:0] == 3'd0);
  assign s0_wlast   = (&j[2:0]) || s0_rfinal;
  assign s0_blast   = (&j[LEB-1:0]) || s0_rfinal;
  assign s0_wsum    = {{(WWD-13){1'b0}}, d_vsw} + {1'b0, s0_c[CW-1:3]};
  assign s0_oob     = (s0_wsum >= WWD'(VSRAM_WORDS));
  assign s0_waddr   = s0_wsum[AW-1:0];
  assign tile_final = j_last && !nxt_any;
  assign row_start  = (t_tile == {TW{1'b0}}) && (j == {LWB{1'b0}});

  generate
    if (B_MAX > 1) begin : g_idx_rows
      assign acc_idx = {cur_row, j};
    end else begin : g_idx_one
      assign acc_idx = j;
    end
  endgenerate

  always_comb begin
    acc_sel = {ACC_W{1'b0}};
    for (int i = 0; i < NEL; i++) begin
      if (acc_idx == AIW'(i)) acc_sel = acc_flat[i*ACC_W +: ACC_W];
    end
  end

  assign row_first   = (cur_row == d_first_row);
  assign need_stream = !d_unit && !d_embed && row_first;
  assign need_emb    = !d_unit && d_embed;
  assign need_rd     = d_acc && s0_wfirst && !s0_oob;

  // Meta records of a tile are consumed from the stream by the first
  // participating row and replayed from the tile buffer for the others.
  generate
    if (B_MAX > 1) begin : g_mbuf
      logic [WB*56-1:0] mbuf;
      always_ff @(posedge clk) begin
        for (int i = 0; i < WB; i++) begin
          if (fire && need_stream && (j == LWB'(i))) mbuf[i*56 +: 56] <= meta_data;
        end
      end
      always_comb begin
        mbuf_rd = 56'd0;
        for (int i = 0; i < WB; i++) begin
          if (j == LWB'(i)) mbuf_rd = mbuf[i*56 +: 56];
        end
      end
    end else begin : g_no_mbuf
      assign mbuf_rd = 56'd0;
    end
  endgenerate

  always_comb begin
    if (d_unit)         m_rec = {8'hF1, 16'h8000, 32'd0};
    else if (d_embed)   m_rec = emb_rec;
    else if (row_first) m_rec = meta_data;
    else                m_rec = mbuf_rd;
  end

  // A read is issued only when no write uses port B next cycle and the row
  // select stays on the read's bank while its data returns.
  assign rd_block = (p7_v && p7_wlast && d_vsram && !p7_oob)
                 || (p6_v && p6_wlast && d_vsram && !p6_oob && (p6_row != cur_row))
                 || (vsb_en_q && (rd_row_q != cur_row));
  assign s0_ok    = (state == S_DRAIN) && adv && !(need_rd && rd_block) && (!need_emb || emb_ok);
  assign fire     = s0_ok && (!need_stream || meta_valid);
  assign emb_wait = need_emb && !emb_ok && ((state == S_WAIT) || (state == S_DRAIN));

  assign meta_ready = (s0_ok && need_stream) || emb_wait;
  assign acc_ready  = (state == S_WAIT) || (fire && tile_final);
  assign busy       = (state != S_IDLE);

  // ------------------------------------------------------------------ descriptor latch
  always_ff @(posedge clk) begin
    if (rst) begin
      d_embed  <= 1'b0;
      d_vsram  <= 1'b0;
      d_argmax <= 1'b0;
      d_dump   <= 1'b0;
      d_acc    <= 1'b0;
      d_unit   <= 1'b0;
      d_track  <= 1'b0;
      d_vsw    <= 13'd0;
      d_sreg   <= 8'd0;
      d_s1     <= 6'd0;
      d_s1_err <= 1'b0;
      d_sbias  <= 8'sd0;
      d_rows   <= {B_MAX{1'b0}};
      d_sx_m   <= {(B_MAX*16){1'b0}};
      d_sx_e   <= {(B_MAX*8){1'b0}};
      d_first_row <= {RIW{1'b0}};
      d_oob    <= 1'b0;
      d_dbase  <= {(B_MAX*32){1'b0}};
      emb_rec  <= 56'd0;
      emb_ok   <= 1'b0;
    end else begin
      if ((state == S_IDLE) && cmd_valid_gemv) begin
        d_embed  <= (cmd_op == OP_EMBED);
        d_vsram  <= (cmd_out_mode == OUT_VSRAM) || (cmd_out_mode == OUT_VSRAM_DUMP);
        d_argmax <= (cmd_out_mode == OUT_ARGMAX) || (cmd_out_mode == OUT_ARGMAX_DUMP);
        d_dump   <= (cmd_out_mode == OUT_ARGMAX_DUMP) || (cmd_out_mode == OUT_VSRAM_DUMP);
        d_acc    <= cmd_accumulate;
        d_unit   <= cmd_unit_meta && (cmd_op != OP_EMBED);
        d_track  <= cmd_track_absmax;
        d_vsw    <= cmd_vs_dst[15:3];
        d_sreg   <= cmd_sreg_dst;
        d_s1     <= (cmd_sh0 > 8'd63) ? 6'd63 : cmd_sh0[5:0];
        d_s1_err <= (cmd_sh0 > 8'd63);
        d_sbias  <= cmd_sh1;
        d_rows   <= cmd_rows;
        d_sx_m   <= cmd_sx_m;
        d_sx_e   <= cmd_sx_e;
        d_first_row <= first_row_c;
        d_oob    <= (({9'd0, cmd_vs_dst} + {1'b0, cmd_n}) > 25'(VSRAM_WORDS * 8));
        d_dbase  <= dbase_c;
        emb_ok   <= 1'b0;
      end else if (meta_valid && emb_wait) begin
        emb_rec <= meta_data;
        emb_ok  <= 1'b1;
      end
    end
  end

  // ------------------------------------------------------------------ sequencer
  logic pipe_empty;
  assign pipe_empty = !(p1_v || p2_v || p3_v || p4_v || p5_v || p6_v || p7_v)
                   && (dq_cnt == 3'd0) && !dq_out_v && !wr_act;

  always_ff @(posedge clk) begin
    if (rst) begin
      state      <= S_IDLE;
      cur_row    <= {RIW{1'b0}};
      j          <= {LWB{1'b0}};
      t_tile     <= {TW{1'b0}};
      t_nvalid   <= {NVW{1'b0}};
      t_last     <= 1'b0;
      rep_row    <= {RIW{1'b0}};
      rep_step   <= 1'b0;
    end else begin
      case (state)
        S_IDLE: begin
          if (cmd_valid_gemv) begin
            if ((cmd_n == 24'd0) || (cmd_rows == {B_MAX{1'b0}})) state <= S_DONE;
            else state <= S_WAIT;
          end
        end
        S_WAIT: begin
          if (acc_valid) begin
            t_tile   <= acc_tile;
            t_nvalid <= acc_nvalid;
            t_last   <= acc_last;
            cur_row  <= d_first_row;
            j        <= {LWB{1'b0}};
            state    <= S_DRAIN;
          end
        end
        S_DRAIN: begin
          if (fire) begin
            if (j_last) begin
              j <= {LWB{1'b0}};
              if (nxt_any) begin
                cur_row <= nxt_row;
              end else if (t_last) begin
                state <= S_EMPTY;
              end else if (acc_valid) begin
                t_tile   <= acc_tile;
                t_nvalid <= acc_nvalid;
                t_last   <= acc_last;
                cur_row  <= d_first_row;
              end else begin
                state <= S_WAIT;
              end
            end else begin
              j <= j + LWB'(1);
            end
          end
        end
        S_EMPTY: begin
          if (pipe_empty) begin
            state    <= S_REPORT;
            rep_row  <= {RIW{1'b0}};
            rep_step <= 1'b0;
          end
        end
        S_REPORT: begin
          rep_step <= !rep_step;
          if (rep_step) begin
            if (rep_row == RIW'(B_MAX - 1)) state <= S_DONE;
            else rep_row <= rep_row + RIW'(1);
          end
        end
        S_DONE: state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end

  // ------------------------------------------------------------------ P1: operand select
  logic [15:0] sxm_p1, sxm_p3;
  logic [7:0]  sxe_p1;

  always_comb begin
    sxm_p1 = 16'd0;
    sxe_p1 = 8'd0;
    sxm_p3 = 16'd0;
    for (int r = 0; r < B_MAX; r++) begin
      if (p1_row == RIW'(r)) begin
        sxm_p1 = d_sx_m[r*16 +: 16];
        sxe_p1 = d_sx_e[r*8 +: 8];
      end
      if (p3_row == RIW'(r)) sxm_p3 = d_sx_m[r*16 +: 16];
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      p1_v <= 1'b0;
      p2_v <= 1'b0;
      p3_v <= 1'b0;
      p4_v <= 1'b0;
      p5_v <= 1'b0;
      p6_v <= 1'b0;
      p7_v <= 1'b0;
    end else if (adv) begin
      p1_v <= fire;
      p2_v <= p1_v;
      p3_v <= p2_v;
      p4_v <= p3_v;
      p5_v <= p4_v;
      p6_v <= p5_v;
      p7_v <= p6_v;
    end
  end

  always_ff @(posedge clk) begin
    if (adv && fire) begin
      p1_acc    <= acc_sel[39:0];
      p1_swm    <= m_rec[47:32];
      p1_swe    <= m_rec[55:48];
      p1_bias   <= d_embed ? 32'd0 : m_rec[31:0];
      p1_c      <= s0_c;
      p1_row    <= cur_row;
      p1_waddr  <= s0_waddr;
      p1_oob    <= s0_oob;
      p1_wfirst <= s0_wfirst;
      p1_wlast  <= s0_wlast;
      p1_blast  <= s0_blast;
    end
  end

  // ------------------------------------------------------------------ P2: stage-1 product, S
  logic signed [9:0] s_raw;
  logic [5:0]        s_raw_lo, s_cl;
  logic              s_err;

  assign s_raw = $signed({{2{d_sbias[7]}}, d_sbias})
               - ($signed({{2{p1_swe[7]}}, p1_swe}) + $signed({{2{sxe_p1[7]}}, sxe_p1}));
  assign s_raw_lo = s_raw[5:0];

  always_comb begin
    if (s_raw < 10'sd0) begin
      s_cl  = 6'd0;
      s_err = 1'b1;
    end else if (s_raw > 10'sd63) begin
      s_cl  = 6'd63;
      s_err = 1'b1;
    end else begin
      s_cl  = s_raw_lo;
      s_err = 1'b0;
    end
  end

  always_ff @(posedge clk) begin
    if (adv && p1_v) begin
      p2_p      <= $signed({{17{p1_acc[39]}}, p1_acc}) * $signed({41'd0, p1_swm});
      p2_nz     <= (p1_swm != 16'd0) && (sxm_p1 != 16'd0);
      p2_s      <= s_cl;
      p2_serr   <= s_err;
      p2_bias   <= p1_bias;
      p2_c      <= p1_c;
      p2_row    <= p1_row;
      p2_waddr  <= p1_waddr;
      p2_oob    <= p1_oob;
      p2_wfirst <= p1_wfirst;
      p2_wlast  <= p1_wlast;
      p2_blast  <= p1_blast;
    end
  end

  // ------------------------------------------------------------------ P3: t = sat40(round_shift(p, s1)), old value
  logic signed [56:0] r1;
  logic [40:0]        sat1_t;
  logic [31:0]        old_sel;
  logic [2:0]         p2_slot;

  assign r1      = qcore_pkg::round_shift57(p2_p, d_s1);
  assign sat1_t  = qcore_pkg::sat40_from57(r1);
  assign p2_slot = p2_c[2:0];

  always_comb begin
    if (!d_acc)          old_sel = 32'd0;
    else if (p2_wfirst)  old_sel = p2_oob ? 32'd0 : vsb_rdata[{p2_slot, 5'd0} +: 32];
    else                 old_sel = old_word_q[{p2_slot, 5'd0} +: 32];
  end

  always_ff @(posedge clk) begin
    if (adv && p2_v) begin
      p3_t     <= sat1_t[39:0];
      p3_satc  <= {2'd0, sat1_t[40] & p2_nz};
      p3_nz    <= p2_nz;
      p3_s     <= p2_s;
      p3_serr  <= p2_serr;
      p3_bias  <= p2_bias;
      p3_old   <= old_sel;
      p3_c     <= p2_c;
      p3_row   <= p2_row;
      p3_waddr <= p2_waddr;
      p3_oob   <= p2_oob;
      p3_wlast <= p2_wlast;
      p3_blast <= p2_blast;
      if (p2_wfirst) old_word_q <= p2_oob ? 256'd0 : vsb_rdata;
    end
  end

  // ------------------------------------------------------------------ P4: stage-2 product
  always_ff @(posedge clk) begin
    if (adv && p3_v) begin
      p4_p     <= $signed({{17{p3_t[39]}}, p3_t}) * $signed({41'd0, sxm_p3});
      p4_satc  <= p3_satc;
      p4_nz    <= p3_nz;
      p4_s     <= p3_s;
      p4_serr  <= p3_serr;
      p4_bias  <= p3_bias;
      p4_old   <= p3_old;
      p4_c     <= p3_c;
      p4_row   <= p3_row;
      p4_waddr <= p3_waddr;
      p4_oob   <= p3_oob;
      p4_wlast <= p3_wlast;
      p4_blast <= p3_blast;
    end
  end

  // ------------------------------------------------------------------ P5: y = sat32(round_shift(p, S)), zero scales
  logic signed [56:0] r2;
  logic [32:0]        sat2_y;

  assign r2     = qcore_pkg::round_shift57(p4_p, p4_s);
  assign sat2_y = qcore_pkg::sat32_from57(r2);

  always_ff @(posedge clk) begin
    if (adv && p4_v) begin
      p5_y     <= p4_nz ? sat2_y[31:0] : 32'd0;
      p5_satc  <= p4_satc + {2'd0, sat2_y[32] & p4_nz};
      p5_nz    <= p4_nz;
      p5_serr  <= p4_serr;
      p5_bias  <= p4_bias;
      p5_old   <= p4_old;
      p5_c     <= p4_c;
      p5_row   <= p4_row;
      p5_waddr <= p4_waddr;
      p5_oob   <= p4_oob;
      p5_wlast <= p4_wlast;
      p5_blast <= p4_blast;
    end
  end

  // ------------------------------------------------------------------ P6: + bias_q
  logic signed [32:0] sum3;
  logic [32:0]        sat3_y;

  assign sum3   = $signed({p5_y[31], p5_y}) + $signed({p5_bias[31], p5_bias});
  assign sat3_y = qcore_pkg::sat32_from33(sum3);

  always_ff @(posedge clk) begin
    if (adv && p5_v) begin
      p6_y     <= sat3_y[31:0];
      p6_satc  <= p5_satc + {2'd0, sat3_y[32]};
      p6_nz    <= p5_nz;
      p6_serr  <= p5_serr;
      p6_old   <= p5_old;
      p6_c     <= p5_c;
      p6_row   <= p5_row;
      p6_waddr <= p5_waddr;
      p6_oob   <= p5_oob;
      p6_wlast <= p5_wlast;
      p6_blast <= p5_blast;
    end
  end

  // ------------------------------------------------------------------ P7: + old (accumulate)
  logic signed [32:0] sum4;
  logic [32:0]        sat4_y;

  assign sum4   = $signed({p6_y[31], p6_y}) + $signed({p6_old[31], p6_old});
  assign sat4_y = qcore_pkg::sat32_from33(sum4);

  always_ff @(posedge clk) begin
    if (adv && p6_v) begin
      p7_y     <= sat4_y[31:0];
      p7_satc  <= p6_satc + {2'd0, sat4_y[32]};
      p7_nz    <= p6_nz;
      p7_serr  <= p6_serr;
      p7_c     <= p6_c;
      p7_row   <= p6_row;
      p7_waddr <= p6_waddr;
      p7_oob   <= p6_oob;
      p7_wlast <= p6_wlast;
      p7_blast <= p6_blast;
    end
  end

  // ------------------------------------------------------------------ output stage
  logic         o_fire;
  logic [31:0]  abs_y;
  logic [2:0]   p7_slot;
  logic [LEB-1:0] p7_bslot;
  logic [255:0] wbuf, word_next;
  logic [7:0]   wstrb, strb_next;
  logic [DW-1:0] dbuf, beat_next;
  logic [WB-1:0] beat_strb;
  logic [31:0]  beat_addr, beat_off, dbase_row;
  logic [DEW-1:0] dq_new;

  assign o_fire   = adv && p7_v;
  assign abs_y    = qcore_pkg::abs32(p7_y);
  assign p7_slot  = p7_c[2:0];
  assign p7_bslot = p7_c[LEB-1:0];
  assign beat_off = {{(30-CW){1'b0}}, p7_c[CW-1:LEB], {LWB{1'b0}}};

  always_comb begin
    word_next = wbuf;
    strb_next = wstrb;
    for (int i = 0; i < 8; i++) begin
      if (p7_slot == 3'(i)) begin
        word_next[i*32 +: 32] = p7_y;
        strb_next[i]          = 1'b1;
      end
    end
    beat_next = dbuf;
    beat_strb = {WB{1'b0}};
    for (int i = 0; i < EPB; i++) begin
      if (p7_bslot == LEB'(i)) beat_next[i*32 +: 32] = p7_y;
      if (LEB'(i) <= p7_bslot) beat_strb[i*4 +: 4] = 4'hF;
    end
    dbase_row = 32'd0;
    for (int r = 0; r < B_MAX; r++) begin
      if (p7_row == RIW'(r)) dbase_row = d_dbase[r*32 +: 32];
    end
    beat_addr = dbase_row + beat_off;
    dq_new    = {beat_addr, beat_strb, beat_next};
  end

  // VSRAM word assembly and the port B write / read registers
  always_ff @(posedge clk) begin
    if (rst) begin
      wstrb      <= 8'd0;
      wr_we_q    <= 8'd0;
      wr_addr_q  <= {AW{1'b0}};
      wr_row_q   <= {RIW{1'b0}};
      vsb_en_q   <= 1'b0;
      rd_addr_q  <= {AW{1'b0}};
      rd_row_q   <= {RIW{1'b0}};
      hold_row_q <= {RW{1'b0}};
    end else begin
      wr_we_q <= 8'd0;
      if (o_fire && d_vsram) begin
        wstrb <= p7_wlast ? 8'd0 : strb_next;
        if (p7_wlast && !p7_oob) begin
          wr_we_q   <= strb_next;
          wr_addr_q <= p7_waddr;
          wr_row_q  <= p7_row;
        end
      end
      vsb_en_q <= fire && need_rd;
      if (fire && need_rd) begin
        rd_addr_q <= s0_waddr;
        rd_row_q  <= cur_row;
      end
      if (vsb_en_q || wr_act) hold_row_q <= vsb_row;
    end
  end

  always_ff @(posedge clk) begin
    if (o_fire && d_vsram) begin
      wbuf <= word_next;
      if (p7_wlast) wr_wdata_q <= word_next;
    end
  end

  // dump beat assembly and the beat queue
  assign dq_push    = o_fire && d_dump && p7_blast;
  assign dq_pop     = dq_out_v && d_wr_ready;
  assign dq_rd      = (dq_cnt != 3'd0) && (!dq_out_v || dq_pop);
  assign d_wr_valid = dq_out_v;
  assign {d_wr_addr, d_wr_strb, d_wr_data} = dq_out;

  always_ff @(posedge clk) begin
    if (o_fire && d_dump) dbuf <= beat_next;
  end

  always_ff @(posedge clk) begin
    if (dq_push) dq_mem[dq_wp] <= dq_new;
  end

  always_ff @(posedge clk) begin
    if (dq_rd) dq_out <= dq_mem[dq_rp];
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      dq_wp    <= {DQP{1'b0}};
      dq_rp    <= {DQP{1'b0}};
      dq_cnt   <= 3'd0;
      dq_out_v <= 1'b0;
    end else begin
      if (dq_push) dq_wp <= dq_wp + DQP'(1);
      if (dq_rd)   dq_rp <= dq_rp + DQP'(1);
      dq_cnt <= dq_cnt + {2'd0, dq_push} - {2'd0, dq_rd};
      if (dq_rd)       dq_out_v <= 1'b1;
      else if (dq_pop) dq_out_v <= 1'b0;
    end
  end

  // per-row absmax and argmax
  logic [B_MAX*32-1:0] amax_q, amx_val_q;
  logic [B_MAX*CW-1:0] amx_idx_q;

  always_ff @(posedge clk) begin
    if (rst) begin
      amax_q    <= {(B_MAX*32){1'b0}};
      amx_val_q <= {B_MAX{32'h8000_0000}};
      amx_idx_q <= {(B_MAX*CW){1'b0}};
    end else if ((state == S_IDLE) && cmd_valid_gemv) begin
      amax_q    <= {(B_MAX*32){1'b0}};
      amx_val_q <= {B_MAX{32'h8000_0000}};
      amx_idx_q <= {(B_MAX*CW){1'b0}};
    end else if (o_fire) begin
      for (int r = 0; r < B_MAX; r++) begin
        if (p7_row == RIW'(r)) begin
          if (abs_y > amax_q[r*32 +: 32]) amax_q[r*32 +: 32] <= abs_y;
          if ($signed(p7_y) > $signed(amx_val_q[r*32 +: 32])) begin
            amx_val_q[r*32 +: 32] <= p7_y;
            amx_idx_q[r*CW +: CW] <= p7_c;
          end
        end
      end
    end
  end

  // descriptor end: SREG absmax and ARGMAX writes per participating row, ascending
  logic          rep_part;
  logic [31:0]   rep_amax, rep_val;
  logic [CW-1:0] rep_idx;

  always_comb begin
    rep_part = 1'b0;
    rep_amax = 32'd0;
    rep_val  = 32'd0;
    rep_idx  = {CW{1'b0}};
    for (int r = 0; r < B_MAX; r++) begin
      if (rep_row == RIW'(r)) begin
        rep_part = d_rows[r];
        rep_amax = amax_q[r*32 +: 32];
        rep_val  = amx_val_q[r*32 +: 32];
        rep_idx  = amx_idx_q[r*CW +: CW];
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      sreg_wr_en     <= 1'b0;
      sreg_wr_row    <= {RW{1'b0}};
      sreg_wr_idx    <= 8'd0;
      sreg_wr_data   <= 32'd0;
      argmax_we      <= 1'b0;
      argmax_tok     <= 32'd0;
      argmax_val     <= 32'd0;
      done           <= 1'b0;
      sat_inc        <= 3'd0;
      err_shift_inc  <= 2'd0;
      err_bounds_inc <= 2'd0;
    end else begin
      sreg_wr_en <= (state == S_REPORT) && !rep_step && rep_part && d_track;
      argmax_we  <= (state == S_REPORT) && rep_step && rep_part && d_argmax;
      if (state == S_REPORT) begin
        sreg_wr_row  <= RW'(rep_row);
        sreg_wr_idx  <= d_sreg;
        sreg_wr_data <= rep_amax;
        if (rep_step && rep_part && d_argmax) begin
          argmax_tok <= {{(32-CW){1'b0}}, rep_idx};
          argmax_val <= rep_val;
        end
      end
      done           <= (state == S_DONE);
      sat_inc        <= o_fire ? p7_satc : 3'd0;
      err_shift_inc  <= o_fire ? ({1'b0, d_s1_err} + {1'b0, p7_serr & p7_nz}) : 2'd0;
      err_bounds_inc <= (fire && row_start) ? ({1'b0, d_vsram & d_oob} + {1'b0, d_acc & d_oob}) : 2'd0;
    end
  end
endmodule
