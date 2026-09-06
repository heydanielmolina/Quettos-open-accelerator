// GEMV path of the core for block-level checks: qcore_mem_arb, qcore_stream_ctrl,
// B_MAX qcore_row / qcore_vsram pairs and qcore_requant wired as docs/RTL.md
// sections 2.3 to 2.8 describe, with the VSRAM crossbar, the SREG write mux
// and the event adders of qcore_top. The issue bundle, the QMEM bus, the
// fetch / VPU / KV-writer clients of the arbiter and the SREG read ports are the
// wrapper's ports; the wrapper itself holds no state.
module qcore_gemv_wrap #(
  parameter int WB              = 16,
  parameter int B_MAX           = 2,
  parameter int VSRAM_WORDS     = 2048,
  parameter int FIFO_BEATS      = 128,
  parameter int ACC_W           = 40,
  parameter int META_FIFO_BEATS = 16,
  parameter int MAX_BURST       = 64
) (
  input  logic                  clk,
  input  logic                  rst,
  // issue bundle (held from the pulse until done_gemv)
  input  logic                  cmd_valid_gemv,
  input  logic [7:0]            cmd_op,
  input  logic [1:0]            cmd_out_mode,
  input  logic                  cmd_accumulate,
  input  logic                  cmd_unit_meta,
  input  logic                  cmd_track_absmax,
  input  logic [31:0]           cmd_addr_a,
  input  logic [31:0]           cmd_addr_m,
  input  logic [31:0]           cmd_imm32,
  input  logic [23:0]           cmd_n,
  input  logic [15:0]           cmd_k,
  input  logic [15:0]           cmd_k_stride,
  input  logic [15:0]           cmd_vs_src,
  input  logic [15:0]           cmd_vs_dst,
  input  logic [7:0]            cmd_sreg_dst,
  input  logic [3:0]            cmd_src_row,
  input  logic [3:0]            cmd_dst_row,
  input  logic [7:0]            cmd_sh0,
  input  logic signed [7:0]     cmd_sh1,
  input  logic [B_MAX-1:0]      cmd_rows,
  input  logic [31:0]           cmd_tok,
  input  logic [B_MAX*16-1:0]   cmd_sx_m,
  input  logic [B_MAX*8-1:0]    cmd_sx_e,
  output logic                  done_gemv,
  output logic                  gemv_beat,
  output logic                  stream_done,
  output logic                  stream_busy,
  output logic                  requant_busy,
  // QMEM
  output logic                  rd_req_valid,
  input  logic                  rd_req_ready,
  output logic [31:0]           rd_req_addr,
  output logic [7:0]            rd_req_len,
  output logic [3:0]            rd_req_tag,
  input  logic                  rd_data_valid,
  input  logic [WB*8-1:0]       rd_data,
  input  logic [3:0]            rd_data_tag,
  input  logic                  rd_data_last,
  output logic                  wr_valid,
  input  logic                  wr_ready,
  output logic [31:0]           wr_addr,
  output logic [WB*8-1:0]       wr_data,
  output logic [WB-1:0]         wr_strb,
  input  logic                  wr_ack,
  output logic                  wr_idle,
  // the arbiter's other clients: fetch, VPU, KV writer
  input  logic                  f_req_valid,
  output logic                  f_req_ready,
  input  logic [31:0]           f_req_addr,
  input  logic [7:0]            f_req_len,
  input  logic [3:0]            f_req_tag,
  input  logic [3:0]            dq_count,
  input  logic                  v_req_valid,
  output logic                  v_req_ready,
  input  logic [31:0]           v_req_addr,
  input  logic [7:0]            v_req_len,
  input  logic [3:0]            v_req_tag,
  output logic                  rdf_valid,
  output logic                  rdv_valid,
  output logic [WB*8-1:0]       rdd_data,
  output logic                  rdd_last,
  input  logic                  k_wr_valid,
  output logic                  k_wr_ready,
  input  logic [31:0]           k_wr_addr,
  input  logic [WB*8-1:0]       k_wr_data,
  input  logic [WB-1:0]         k_wr_strb,
  // SREG bank reads (one enable per bank, data the next cycle)
  input  logic [B_MAX-1:0]      sreg_rd_en,
  input  logic [7:0]            sreg_rd_idx,
  output logic [B_MAX*32-1:0]   sreg_rd_data,
  // ARGMAX CSR write and event increments
  output logic                  argmax_we,
  output logic [31:0]           argmax_tok,
  output logic [31:0]           argmax_val,
  output logic [2:0]            sat_inc,
  output logic [1:0]            err_shift_inc,
  output logic [3:0]            err_bounds_inc,
  output logic                  ev_rd_beat,
  output logic                  ev_wr_beat,
  output logic [7:0]            ev_wr_bytes
);
  localparam int DW  = WB * 8;
  localparam int AW  = $clog2(VSRAM_WORDS);
  localparam int NVW = $clog2(WB) + 1;
  localparam int TW  = 20;

  // ---------------------------------------------------------------- arbiter <-> stream
  logic          s_req_valid, s_req_ready;
  logic [31:0]   s_req_addr;
  logic [7:0]    s_req_len;
  logic [3:0]    s_req_tag;
  logic          rdw_valid, rdm_valid;
  logic          d_wr_valid, d_wr_ready;
  logic [31:0]   d_wr_addr;
  logic [DW-1:0] d_wr_data;
  logic [WB-1:0] d_wr_strb;

  qcore_mem_arb #(.WB(WB), .MAX_BURST(MAX_BURST)) u_arb (
    .clk(clk), .rst(rst),
    .s_req_valid(s_req_valid), .s_req_ready(s_req_ready), .s_req_addr(s_req_addr),
    .s_req_len(s_req_len), .s_req_tag(s_req_tag),
    .v_req_valid(v_req_valid), .v_req_ready(v_req_ready), .v_req_addr(v_req_addr),
    .v_req_len(v_req_len), .v_req_tag(v_req_tag),
    .f_req_valid(f_req_valid), .f_req_ready(f_req_ready), .f_req_addr(f_req_addr),
    .f_req_len(f_req_len), .f_req_tag(f_req_tag),
    .dq_count(dq_count),
    .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready), .rd_req_addr(rd_req_addr),
    .rd_req_len(rd_req_len), .rd_req_tag(rd_req_tag),
    .rd_data_valid(rd_data_valid), .rd_data(rd_data), .rd_data_tag(rd_data_tag),
    .rd_data_last(rd_data_last),
    .rdf_valid(rdf_valid), .rdw_valid(rdw_valid), .rdm_valid(rdm_valid), .rdv_valid(rdv_valid),
    .rdd_data(rdd_data), .rdd_last(rdd_last),
    .k_wr_valid(k_wr_valid), .k_wr_ready(k_wr_ready), .k_wr_addr(k_wr_addr),
    .k_wr_data(k_wr_data), .k_wr_strb(k_wr_strb),
    .d_wr_valid(d_wr_valid), .d_wr_ready(d_wr_ready), .d_wr_addr(d_wr_addr),
    .d_wr_data(d_wr_data), .d_wr_strb(d_wr_strb),
    .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_addr(wr_addr), .wr_data(wr_data),
    .wr_strb(wr_strb), .wr_ack(wr_ack), .wr_idle(wr_idle),
    .ev_rd_beat(ev_rd_beat), .ev_wr_beat(ev_wr_beat), .ev_wr_bytes(ev_wr_bytes)
  );

  // ---------------------------------------------------------------- weight stream
  logic           ws_valid, ws_ready;
  logic [DW-1:0]  ws_data;
  logic [15:0]    ws_k;
  logic [TW-1:0]  ws_tile;
  logic           ws_tile_start, ws_tile_end, ws_last, ws_embed;
  logic [NVW-1:0] ws_nvalid;
  logic           meta_valid, meta_ready;
  logic [55:0]    meta_data;

  qcore_stream_ctrl #(
    .WB(WB), .FIFO_BEATS(FIFO_BEATS), .META_FIFO_BEATS(META_FIFO_BEATS), .MAX_BURST(MAX_BURST)
  ) u_stream (
    .clk(clk), .rst(rst),
    .cmd_valid_gemv(cmd_valid_gemv), .cmd_op(cmd_op), .cmd_addr_a(cmd_addr_a),
    .cmd_addr_m(cmd_addr_m), .cmd_n(cmd_n), .cmd_k(cmd_k), .cmd_k_stride(cmd_k_stride),
    .cmd_unit_meta(cmd_unit_meta), .cmd_tok(cmd_tok),
    .s_req_valid(s_req_valid), .s_req_ready(s_req_ready), .s_req_addr(s_req_addr),
    .s_req_len(s_req_len), .s_req_tag(s_req_tag),
    .rdw_valid(rdw_valid), .rdm_valid(rdm_valid), .rd_data(rdd_data), .rd_data_last(rdd_last),
    .ws_valid(ws_valid), .ws_ready(ws_ready), .ws_data(ws_data), .ws_k(ws_k),
    .ws_tile(ws_tile), .ws_tile_start(ws_tile_start), .ws_tile_end(ws_tile_end),
    .ws_nvalid(ws_nvalid), .ws_last(ws_last), .ws_embed(ws_embed),
    .meta_valid(meta_valid), .meta_ready(meta_ready), .meta_data(meta_data),
    .stream_done(stream_done), .busy(stream_busy)
  );

  // ---------------------------------------------------------------- rows and VSRAMs
  logic [B_MAX-1:0]        row_ws_ready, row_acc_valid, row_acc_last;
  logic [B_MAX-1:0]        row_err_bounds, row_ev_beat, row_sreg_err, row_sreg_wr_en;
  logic [B_MAX-1:0]        row_vsa_en;
  logic [B_MAX*AW-1:0]     row_vsa_addr;
  logic [B_MAX*256-1:0]    row_vsa_rdata;
  logic [B_MAX*TW-1:0]     row_acc_tile;
  logic [B_MAX*NVW-1:0]    row_acc_nvalid;
  logic [B_MAX*WB*ACC_W-1:0] acc_flat;
  logic                    acc_valid, acc_ready, acc_last;
  logic [TW-1:0]           acc_tile;
  logic [NVW-1:0]          acc_nvalid;
  logic [B_MAX-1:0]        vs_en_a, vs_en_b;
  logic [B_MAX*AW-1:0]     vs_addr_a;
  logic [B_MAX*8-1:0]      vs_we_b;
  logic [B_MAX*256-1:0]    vs_rd_a, vs_rd_b;
  logic [3:0]              rq_vsb_row, rq_sreg_wr_row;
  logic                    rq_vsb_en, rq_sreg_wr_en;
  logic [7:0]              rq_vsb_we, rq_sreg_wr_idx;
  logic [AW-1:0]           rq_vsb_addr;
  logic [255:0]            rq_vsb_wdata, rq_vsb_rdata;
  logic [31:0]             rq_sreg_wr_data;
  logic [1:0]              rq_err_bounds_inc;
  logic [3:0]              vsb_bank, sreg_bank;

  assign ws_ready  = &row_ws_ready;
  assign acc_valid = |row_acc_valid;
  assign gemv_beat = |row_ev_beat;
  assign vsb_bank  = cmd_dst_row + rq_vsb_row;
  assign sreg_bank = cmd_dst_row + rq_sreg_wr_row;

  // The handoff descriptors come from the lowest participating row.
  always_comb begin
    acc_tile   = {TW{1'b0}};
    acc_nvalid = {NVW{1'b0}};
    acc_last   = 1'b0;
    for (int r = B_MAX - 1; r >= 0; r--) begin
      if (cmd_rows[r]) begin
        acc_tile   = row_acc_tile[r*TW +: TW];
        acc_nvalid = row_acc_nvalid[r*NVW +: NVW];
        acc_last   = row_acc_last[r];
      end
    end
  end

  // Crossbar: row r reads bank src_row + r on port A; the requant owns port B
  // of bank dst_row + vsb_row and writes SREG bank dst_row + sreg_wr_row.
  always_comb begin
    vs_en_a       = {B_MAX{1'b0}};
    vs_addr_a     = {(B_MAX*AW){1'b0}};
    row_vsa_rdata = {(B_MAX*256){1'b0}};
    for (int v = 0; v < B_MAX; v++) begin
      for (int r = 0; r < B_MAX; r++) begin
        if (cmd_rows[r] && (4'(cmd_src_row + 4'(r)) == 4'(v))) begin
          vs_en_a[v]                   = row_vsa_en[r];
          vs_addr_a[v*AW +: AW]        = row_vsa_addr[r*AW +: AW];
          row_vsa_rdata[r*256 +: 256]  = vs_rd_a[v*256 +: 256];
        end
      end
    end
    vs_en_b        = {B_MAX{1'b0}};
    vs_we_b        = {(B_MAX*8){1'b0}};
    rq_vsb_rdata   = 256'd0;
    row_sreg_wr_en = {B_MAX{1'b0}};
    for (int v = 0; v < B_MAX; v++) begin
      if (vsb_bank == 4'(v)) begin
        vs_en_b[v]        = rq_vsb_en;
        vs_we_b[v*8 +: 8] = rq_vsb_we;
        rq_vsb_rdata      = vs_rd_b[v*256 +: 256];
      end
      if (sreg_bank == 4'(v)) row_sreg_wr_en[v] = rq_sreg_wr_en;
    end
  end

  genvar r;
  generate
    for (r = 0; r < B_MAX; r++) begin : g_row
      qcore_row #(.WB(WB), .ACC_W(ACC_W), .VSRAM_WORDS(VSRAM_WORDS)) u_row (
        .clk(clk), .rst(rst),
        .cmd_valid_gemv(cmd_valid_gemv), .row_active(cmd_rows[r]),
        .cmd_vs_src(cmd_vs_src), .cmd_k(cmd_k),
        .ws_valid(ws_valid), .ws_ready(row_ws_ready[r]), .ws_data(ws_data), .ws_k(ws_k),
        .ws_tile(ws_tile), .ws_tile_start(ws_tile_start), .ws_tile_end(ws_tile_end),
        .ws_nvalid(ws_nvalid), .ws_last(ws_last), .ws_embed(ws_embed),
        .vsa_en(row_vsa_en[r]), .vsa_addr(row_vsa_addr[r*AW +: AW]),
        .vsa_rdata(row_vsa_rdata[r*256 +: 256]),
        .acc_valid(row_acc_valid[r]), .acc_ready(acc_ready),
        .acc_flat(acc_flat[r*WB*ACC_W +: WB*ACC_W]),
        .acc_tile(row_acc_tile[r*TW +: TW]), .acc_nvalid(row_acc_nvalid[r*NVW +: NVW]),
        .acc_last(row_acc_last[r]),
        .sreg_rd_en(sreg_rd_en[r]), .sreg_rd_idx(sreg_rd_idx),
        .sreg_rd_data(sreg_rd_data[r*32 +: 32]),
        .sreg_wr_en(row_sreg_wr_en[r]), .sreg_wr_idx(rq_sreg_wr_idx),
        .sreg_wr_data(rq_sreg_wr_data), .sreg_err(row_sreg_err[r]),
        .err_bounds(row_err_bounds[r]), .ev_beat(row_ev_beat[r])
      );

      qcore_vsram #(.WORDS(VSRAM_WORDS), .W(256), .NE(8)) u_vsram (
        .clk(clk),
        .en_a(vs_en_a[r]), .addr_a(vs_addr_a[r*AW +: AW]), .rd_a(vs_rd_a[r*256 +: 256]),
        .en_b(vs_en_b[r]), .we_b(vs_we_b[r*8 +: 8]), .addr_b(rq_vsb_addr),
        .wd_b(rq_vsb_wdata), .rd_b(vs_rd_b[r*256 +: 256])
      );
    end
  endgenerate

  // ---------------------------------------------------------------- requant
  qcore_requant #(.WB(WB), .B_MAX(B_MAX), .ACC_W(ACC_W), .VSRAM_WORDS(VSRAM_WORDS)) u_requant (
    .clk(clk), .rst(rst),
    .cmd_valid_gemv(cmd_valid_gemv), .cmd_op(cmd_op), .cmd_out_mode(cmd_out_mode),
    .cmd_accumulate(cmd_accumulate), .cmd_unit_meta(cmd_unit_meta),
    .cmd_track_absmax(cmd_track_absmax), .cmd_n(cmd_n), .cmd_vs_dst(cmd_vs_dst),
    .cmd_sreg_dst(cmd_sreg_dst), .cmd_sh0(cmd_sh0), .cmd_sh1(cmd_sh1), .cmd_imm32(cmd_imm32),
    .cmd_rows(cmd_rows), .cmd_sx_m(cmd_sx_m), .cmd_sx_e(cmd_sx_e),
    .acc_valid(acc_valid), .acc_ready(acc_ready), .acc_flat(acc_flat), .acc_tile(acc_tile),
    .acc_nvalid(acc_nvalid), .acc_last(acc_last),
    .meta_valid(meta_valid), .meta_ready(meta_ready), .meta_data(meta_data),
    .vsb_row(rq_vsb_row), .vsb_en(rq_vsb_en), .vsb_we(rq_vsb_we), .vsb_addr(rq_vsb_addr),
    .vsb_wdata(rq_vsb_wdata), .vsb_rdata(rq_vsb_rdata),
    .d_wr_valid(d_wr_valid), .d_wr_ready(d_wr_ready), .d_wr_addr(d_wr_addr),
    .d_wr_data(d_wr_data), .d_wr_strb(d_wr_strb),
    .sreg_wr_en(rq_sreg_wr_en), .sreg_wr_row(rq_sreg_wr_row), .sreg_wr_idx(rq_sreg_wr_idx),
    .sreg_wr_data(rq_sreg_wr_data),
    .argmax_we(argmax_we), .argmax_tok(argmax_tok), .argmax_val(argmax_val),
    .sat_inc(sat_inc), .err_shift_inc(err_shift_inc), .err_bounds_inc(rq_err_bounds_inc),
    .done(done_gemv), .busy(requant_busy)
  );

  // ---------------------------------------------------------------- event adders
  always_comb begin
    err_bounds_inc = {2'd0, rq_err_bounds_inc};
    for (int i = 0; i < B_MAX; i++) begin
      err_bounds_inc = err_bounds_inc + {3'd0, row_err_bounds[i]} + {3'd0, row_sreg_err[i]};
    end
  end
endmodule
