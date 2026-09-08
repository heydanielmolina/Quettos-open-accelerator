// Quettos Core top level: the flat QMEM and CSR ports, the parameter root, and
// the wiring of every unit. It holds one qcore_vsram and one qcore_row per
// activation row, the VSRAM port-A and port-B crossbars, the SREG read select
// and write mux, the fan-out of the arbiter's returned beat, and the arithmetic
// sum of the units' per-cycle event counts into the four CSR counters.
// qcore_vpu_top holds the passes of every V opcode, and this level hands it
// cmd_len, cmd_pos and the exp2 table image the VSOFTMAX and VROPE passes read.
// Every opcode of the ISA issues to a unit from here; a byte that is none of
// them is decoded as unknown by qcore_seq_dispatch, which ends the run with
// STATUS.ERR, FAULT = OPCODE, FAULT_OP = the byte and PC on the descriptor.
`include "qcore_csr_defs.svh"
module qcore_top #(
  parameter int WB              = 64,
  parameter int B_MAX           = 1,
  parameter int VSRAM_WORDS     = 4096,
  parameter int FIFO_BEATS      = 128,
  parameter int ACC_W           = 40,
  parameter int META_FIFO_BEATS = 16,
  parameter int MAX_BURST       = 64,
  parameter int DQ_DEPTH        = 8,
  parameter int VL              = 4,
  parameter int VPU_FIFO_BEATS  = 16,
  // Lookup-table images, forwarded to the ROMs qcore_vpu_top holds
  parameter     ROM_FILE_SIGMOID = "",
  parameter     ROM_FILE_EXP2    = "",
  parameter     ROM_FILE_RSQRT   = "",
  parameter     ROM_FILE_RECIP   = ""
) (
  input  logic            clk,
  input  logic            rst,
  // host CSR port; csr_rdata carries the addressed word the cycle after csr_re
  input  logic            csr_we,
  input  logic [5:0]      csr_addr,
  input  logic [31:0]     csr_wdata,
  input  logic            csr_re,
  output logic [31:0]     csr_rdata,
  // QMEM read request
  output logic            rd_req_valid,
  input  logic            rd_req_ready,
  output logic [31:0]     rd_req_addr,
  output logic [7:0]      rd_req_len,
  output logic [3:0]      rd_req_tag,
  // QMEM read data
  input  logic            rd_data_valid,
  input  logic [WB*8-1:0] rd_data,
  input  logic [3:0]      rd_data_tag,
  input  logic            rd_data_last,
  // QMEM write
  output logic            wr_valid,
  input  logic            wr_ready,
  output logic [31:0]     wr_addr,
  output logic [WB*8-1:0] wr_data,
  output logic [WB-1:0]   wr_strb,
  input  logic            wr_ack
);
  localparam int DW  = WB * 8;
  localparam int AW  = $clog2(VSRAM_WORDS);
  localparam int NVW = $clog2(WB) + 1;
  localparam int TW  = 20;

  localparam logic [31:0] ROW_EN_MASK  = 32'((32'd1 << B_MAX) - 32'd1);

  // ---------------------------------------------------------------- control path
  logic            start, step, abort_run;
  logic [31:0]     pc_q, row_en_q, tok_q, pos_q;
  logic            pc_set;
  logic [31:0]     pc_set_val;
  logic            busy, done_set, step_halted_set, err_set;
  logic [3:0]      fault_code;
  logic [7:0]      fault_op;
  logic            csr_err_set;
  logic [3:0]      csr_fault_code;
  logic [7:0]      csr_fault_op;
  logic            disp_abort_run;
  logic [B_MAX-1:0] disp_row_en;
  logic            perf_clear, perf_snapshot;
  logic [1023:0]   perf_snap;
  logic [7:0]      sat_req_inc, sat_vpu_inc, err_shift_inc, err_bounds_inc;

  // ---------------------------------------------------------------- descriptor fetch
  logic            fetch_start, fetch_step, fetch_flush, fetch_hold;
  logic [31:0]     fetch_pc;
  logic            fq_valid, fq_ready;
  logic            dq_valid, dq_ready;
  logic [255:0]    dq_desc;
  logic [3:0]      dq_count;
  logic            ev_fetch_beat;

  // ---------------------------------------------------------------- issue bundle
  logic            cmd_valid_gemv, cmd_valid_vpu, cmd_valid_kv;
  logic            done_gemv, done_kv;
  logic [7:0]      cmd_op;
  logic [1:0]      cmd_out_mode;
  logic            cmd_accumulate, cmd_unit_meta, cmd_track_absmax;
  logic            cmd_vq_w8, cmd_vq_use_tracked, cmd_vq_group, cmd_vq_scale_mul;
  logic            cmd_kv_transposed;
  logic [31:0]     cmd_addr_a, cmd_addr_m, cmd_imm32;
  logic [23:0]     cmd_n, cmd_len;
  logic [15:0]     cmd_k, cmd_k_stride, cmd_vs_src, cmd_vs_dst, cmd_vs_aux;
  logic [7:0]      cmd_sreg_dst, cmd_sh0, cmd_sh1, cmd_sqrt_e;
  logic [15:0]     cmd_sqrt_m;
  logic [3:0]      cmd_src_row, cmd_dst_row;
  logic [B_MAX-1:0] cmd_rows;
  logic [31:0]     cmd_pos, cmd_tok;
  logic [B_MAX*16-1:0] cmd_sx_m;
  logic [B_MAX*8-1:0]  cmd_sx_e;
  logic [B_MAX*32-1:0] cmd_sreg_u32;
  logic [5:0]      ev_bucket;
  logic            ev_desc, ev_macs_valid, ev_wt_valid;
  logic [39:0]     ev_macs, ev_wt_bytes;
  logic [3:0]      disp_err_bounds_inc;

  // ---------------------------------------------------------------- memory arbiter
  logic            s_req_valid, s_req_ready;
  logic [31:0]     s_req_addr;
  logic [7:0]      s_req_len;
  logic [3:0]      s_req_tag;
  logic            v_req_valid, v_req_ready;
  logic [31:0]     v_req_addr;
  logic [7:0]      v_req_len;
  logic [3:0]      v_req_tag;
  logic            f_req_valid, f_req_ready;
  logic [31:0]     f_req_addr;
  logic [7:0]      f_req_len;
  logic [3:0]      f_req_tag;
  logic            rdf_valid, rdw_valid, rdm_valid, rdv_valid;
  logic [DW-1:0]   rdd_data;
  logic            rdd_last;
  logic            k_wr_valid, k_wr_ready;
  logic [31:0]     k_wr_addr;
  logic [DW-1:0]   k_wr_data;
  logic [WB-1:0]   k_wr_strb;
  logic            d_wr_valid, d_wr_ready;
  logic [31:0]     d_wr_addr;
  logic [DW-1:0]   d_wr_data;
  logic [WB-1:0]   d_wr_strb;
  logic            wr_idle;
  logic            ev_rd_beat, ev_wr_beat;
  logic [7:0]      ev_wr_bytes;

  // ---------------------------------------------------------------- weight stream
  logic            ws_valid, ws_ready;
  logic [DW-1:0]   ws_data;
  logic [15:0]     ws_k;
  logic [TW-1:0]   ws_tile;
  logic            ws_tile_start, ws_tile_end, ws_last, ws_embed;
  logic [NVW-1:0]  ws_nvalid;
  logic            meta_valid, meta_ready;
  logic [55:0]     meta_data;
  logic            stream_done, stream_busy;

  // ---------------------------------------------------------------- rows and VSRAMs
  logic [B_MAX-1:0]          row_ws_ready, row_acc_valid, row_acc_last;
  logic [B_MAX-1:0]          row_err_bounds, row_ev_beat, row_sreg_err;
  logic [B_MAX-1:0]          row_sreg_rd_en, row_sreg_wr_en;
  logic [B_MAX*32-1:0]       row_sreg_rd_data;
  logic [B_MAX-1:0]          row_vsa_en;
  logic [B_MAX*AW-1:0]       row_vsa_addr;
  logic [B_MAX*256-1:0]      row_vsa_rdata;
  logic [B_MAX*TW-1:0]       row_acc_tile;
  logic [B_MAX*NVW-1:0]      row_acc_nvalid;
  logic [B_MAX*WB*ACC_W-1:0] acc_flat;
  logic                      acc_valid, acc_ready, acc_last;
  logic [TW-1:0]             acc_tile;
  logic [NVW-1:0]            acc_nvalid;
  logic [B_MAX-1:0]          vs_en_a, vs_en_b;
  logic [B_MAX*AW-1:0]       vs_addr_a;
  logic [B_MAX*8-1:0]        vs_we_b;
  logic [B_MAX*256-1:0]      vs_rd_a, vs_rd_b;
  logic                      gemv_beat;
  logic                      sreg_rd_en;
  logic [3:0]                sreg_rd_row;
  logic [7:0]                sreg_rd_idx;
  logic [31:0]               sreg_rd_data;

  // ---------------------------------------------------------------- port B owners
  logic [3:0]      rq_vsb_row, rq_sreg_wr_row;
  logic            rq_vsb_en, rq_sreg_wr_en, requant_busy;
  logic [7:0]      rq_vsb_we, rq_sreg_wr_idx;
  logic [AW-1:0]   rq_vsb_addr;
  logic [255:0]    rq_vsb_wdata;
  logic [31:0]     rq_sreg_wr_data;
  logic [2:0]      rq_sat_inc;
  logic [1:0]      rq_err_shift_inc, rq_err_bounds_inc;
  logic            argmax_we;
  logic [31:0]     argmax_tok, argmax_val;
  logic [3:0]      kv_vsb_row;
  logic            kv_vsb_en, kv_busy;
  logic [AW-1:0]   kv_vsb_addr;
  logic [1:0]      kv_err_bounds_inc;
  logic            vsb_kv, vsb_vpu, vsb_en_sel;
  logic [3:0]      vsb_bank, sreg_wr_bank;
  logic [7:0]      vsb_we_sel;
  logic [AW-1:0]   vsb_addr;
  logic [255:0]    vsb_wdata, vsb_rdata;
  logic            sreg_wr_en_sel;
  logic [3:0]      sreg_wr_row_sel;
  logic [7:0]      sreg_wr_idx_sel;
  logic [31:0]     sreg_wr_data_sel;

  // ---------------------------------------------------------------- vector processor
  logic            done_vpu, vpu_busy;
  logic [3:0]      vpu_cur_row;
  logic            vpu_vsa_en;
  logic [AW-1:0]   vpu_vsa_addr;
  logic [255:0]    vpu_vsa_rdata;
  logic            vpu_vsb_sel_dst, vpu_vsb_en;
  logic [7:0]      vpu_vsb_we;
  logic [AW-1:0]   vpu_vsb_addr;
  logic [255:0]    vpu_vsb_wdata;
  logic            vpu_sreg_wr_en;
  logic [3:0]      vpu_sreg_wr_row;
  logic [7:0]      vpu_sreg_wr_idx;
  logic [31:0]     vpu_sreg_wr_data;
  logic [7:0]      vpu_sat_inc, vpu_err_shift_inc;
  logic [3:0]      vpu_err_bounds_inc;
  logic [3:0]      vpu_vsa_bank;

  // ================================================================ control loop
  qcore_csr u_csr (
    .clk(clk), .rst(rst),
    .csr_we(csr_we), .csr_addr(csr_addr), .csr_wdata(csr_wdata),
    .csr_re(csr_re), .csr_rdata(csr_rdata),
    .start(start), .step(step), .abort_run(abort_run),
    .pc_q(pc_q), .pc_set(pc_set), .pc_set_val(pc_set_val),
    .row_en_q(row_en_q), .tok_q(tok_q), .pos_q(pos_q),
    .busy_i(busy), .done_set(done_set), .step_halted_set(step_halted_set),
    .err_set(csr_err_set), .fault_code(csr_fault_code), .fault_op(csr_fault_op),
    .argmax_we(argmax_we), .argmax_tok(argmax_tok), .argmax_val(argmax_val),
    .sat_req_inc(sat_req_inc), .sat_vpu_inc(sat_vpu_inc),
    .err_shift_inc(err_shift_inc), .err_bounds_inc(err_bounds_inc),
    .perf_snap(perf_snap)
  );

  qcore_seq_fetch #(.WB(WB), .DQ_DEPTH(DQ_DEPTH)) u_fetch (
    .clk(clk), .rst(rst),
    .fetch_start(fetch_start), .fetch_pc(fetch_pc), .fetch_step(fetch_step),
    .fetch_flush(fetch_flush), .fetch_hold(fetch_hold),
    .f_req_valid(f_req_valid), .f_req_ready(f_req_ready), .f_req_addr(f_req_addr),
    .f_req_len(f_req_len), .f_req_tag(f_req_tag),
    .fd_valid(rdf_valid), .fd_data(rdd_data), .fd_last(rdd_last),
    .dq_valid(fq_valid), .dq_desc(dq_desc), .dq_ready(fq_ready), .dq_count(dq_count),
    .ev_fetch_beat(ev_fetch_beat)
  );

  qcore_seq_dispatch #(.WB(WB), .B_MAX(B_MAX)) u_dispatch (
    .clk(clk), .rst(rst),
    .start(start), .step(step), .abort_run(disp_abort_run),
    .pc_q(pc_q), .row_en_q(disp_row_en), .tok_q(tok_q), .pos_q(pos_q),
    .pc_set(pc_set), .pc_set_val(pc_set_val),
    .busy(busy), .done_set(done_set), .step_halted_set(step_halted_set),
    .err_set(err_set), .fault_code(fault_code), .fault_op(fault_op),
    .perf_clear(perf_clear), .perf_snapshot(perf_snapshot),
    .fetch_start(fetch_start), .fetch_pc(fetch_pc), .fetch_step(fetch_step),
    .fetch_flush(fetch_flush), .fetch_hold(fetch_hold),
    .dq_valid(dq_valid), .dq_desc(dq_desc), .dq_ready(dq_ready),
    .sreg_rd_en(sreg_rd_en), .sreg_rd_row(sreg_rd_row), .sreg_rd_idx(sreg_rd_idx),
    .sreg_rd_data(sreg_rd_data),
    .cmd_valid_gemv(cmd_valid_gemv), .cmd_valid_vpu(cmd_valid_vpu),
    .cmd_valid_kv(cmd_valid_kv),
    .done_gemv(done_gemv), .done_vpu(done_vpu), .done_kv(done_kv),
    .cmd_op(cmd_op), .cmd_out_mode(cmd_out_mode), .cmd_accumulate(cmd_accumulate),
    .cmd_unit_meta(cmd_unit_meta), .cmd_track_absmax(cmd_track_absmax),
    .cmd_vq_w8(cmd_vq_w8), .cmd_vq_use_tracked(cmd_vq_use_tracked),
    .cmd_vq_group(cmd_vq_group), .cmd_vq_scale_mul(cmd_vq_scale_mul),
    .cmd_kv_transposed(cmd_kv_transposed),
    .cmd_addr_a(cmd_addr_a), .cmd_addr_m(cmd_addr_m), .cmd_imm32(cmd_imm32),
    .cmd_n(cmd_n), .cmd_k(cmd_k), .cmd_k_stride(cmd_k_stride), .cmd_len(cmd_len),
    .cmd_vs_src(cmd_vs_src), .cmd_vs_dst(cmd_vs_dst), .cmd_vs_aux(cmd_vs_aux),
    .cmd_sreg_dst(cmd_sreg_dst), .cmd_src_row(cmd_src_row), .cmd_dst_row(cmd_dst_row),
    .cmd_sh0(cmd_sh0), .cmd_sh1(cmd_sh1),
    .cmd_sqrt_m(cmd_sqrt_m), .cmd_sqrt_e(cmd_sqrt_e),
    .cmd_rows(cmd_rows), .cmd_pos(cmd_pos), .cmd_tok(cmd_tok),
    .cmd_sx_m(cmd_sx_m), .cmd_sx_e(cmd_sx_e), .cmd_sreg_u32(cmd_sreg_u32),
    .wr_idle(wr_idle), .gemv_beat(gemv_beat), .stream_done(stream_done),
    .stream_busy(stream_busy),
    .ev_bucket(ev_bucket), .ev_desc(ev_desc),
    .ev_macs_valid(ev_macs_valid), .ev_macs(ev_macs),
    .ev_wt_valid(ev_wt_valid), .ev_wt_bytes(ev_wt_bytes),
    .err_bounds_inc(disp_err_bounds_inc)
  );

  qcore_perf #(.WB(WB)) u_perf (
    .clk(clk), .rst(rst),
    .clear(perf_clear), .snapshot(perf_snapshot),
    .ev_cycle(busy), .ev_busy(busy), .ev_bucket(ev_bucket),
    .ev_rd_beat(ev_rd_beat), .ev_wr_beat(ev_wr_beat), .ev_wr_bytes(ev_wr_bytes),
    .ev_wt_valid(ev_wt_valid), .ev_wt_bytes(ev_wt_bytes),
    .ev_macs_valid(ev_macs_valid), .ev_macs(ev_macs),
    .ev_desc(ev_desc), .ev_fetch_beat(ev_fetch_beat),
    .perf_snap(perf_snap)
  );

  // ================================================================ memory
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
    .rdf_valid(rdf_valid), .rdw_valid(rdw_valid), .rdm_valid(rdm_valid),
    .rdv_valid(rdv_valid), .rdd_data(rdd_data), .rdd_last(rdd_last),
    .k_wr_valid(k_wr_valid), .k_wr_ready(k_wr_ready), .k_wr_addr(k_wr_addr),
    .k_wr_data(k_wr_data), .k_wr_strb(k_wr_strb),
    .d_wr_valid(d_wr_valid), .d_wr_ready(d_wr_ready), .d_wr_addr(d_wr_addr),
    .d_wr_data(d_wr_data), .d_wr_strb(d_wr_strb),
    .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_addr(wr_addr), .wr_data(wr_data),
    .wr_strb(wr_strb), .wr_ack(wr_ack), .wr_idle(wr_idle),
    .ev_rd_beat(ev_rd_beat), .ev_wr_beat(ev_wr_beat), .ev_wr_bytes(ev_wr_bytes)
  );

  qcore_stream_ctrl #(
    .WB(WB), .FIFO_BEATS(FIFO_BEATS), .META_FIFO_BEATS(META_FIFO_BEATS),
    .MAX_BURST(MAX_BURST)
  ) u_stream (
    .clk(clk), .rst(rst),
    .cmd_valid_gemv(cmd_valid_gemv), .cmd_op(cmd_op), .cmd_addr_a(cmd_addr_a),
    .cmd_addr_m(cmd_addr_m), .cmd_n(cmd_n), .cmd_k(cmd_k),
    .cmd_k_stride(cmd_k_stride), .cmd_unit_meta(cmd_unit_meta), .cmd_tok(cmd_tok),
    .s_req_valid(s_req_valid), .s_req_ready(s_req_ready), .s_req_addr(s_req_addr),
    .s_req_len(s_req_len), .s_req_tag(s_req_tag),
    .rdw_valid(rdw_valid), .rdm_valid(rdm_valid), .rd_data(rdd_data),
    .rd_data_last(rdd_last),
    .ws_valid(ws_valid), .ws_ready(ws_ready), .ws_data(ws_data), .ws_k(ws_k),
    .ws_tile(ws_tile), .ws_tile_start(ws_tile_start), .ws_tile_end(ws_tile_end),
    .ws_nvalid(ws_nvalid), .ws_last(ws_last), .ws_embed(ws_embed),
    .meta_valid(meta_valid), .meta_ready(meta_ready), .meta_data(meta_data),
    .stream_done(stream_done), .busy(stream_busy)
  );

  // ================================================================ rows
  genvar r;
  generate
    for (r = 0; r < B_MAX; r++) begin : g_row
      qcore_row #(.WB(WB), .ACC_W(ACC_W), .VSRAM_WORDS(VSRAM_WORDS)) u_row (
        .clk(clk), .rst(rst),
        .cmd_valid_gemv(cmd_valid_gemv), .row_active(cmd_rows[r]),
        .cmd_vs_src(cmd_vs_src), .cmd_k(cmd_k),
        .ws_valid(ws_valid), .ws_ready(row_ws_ready[r]), .ws_data(ws_data),
        .ws_k(ws_k), .ws_tile(ws_tile), .ws_tile_start(ws_tile_start),
        .ws_tile_end(ws_tile_end), .ws_nvalid(ws_nvalid), .ws_last(ws_last),
        .ws_embed(ws_embed),
        .vsa_en(row_vsa_en[r]), .vsa_addr(row_vsa_addr[r*AW +: AW]),
        .vsa_rdata(row_vsa_rdata[r*256 +: 256]),
        .acc_valid(row_acc_valid[r]), .acc_ready(acc_ready),
        .acc_flat(acc_flat[r*WB*ACC_W +: WB*ACC_W]),
        .acc_tile(row_acc_tile[r*TW +: TW]),
        .acc_nvalid(row_acc_nvalid[r*NVW +: NVW]), .acc_last(row_acc_last[r]),
        .sreg_rd_en(row_sreg_rd_en[r]), .sreg_rd_idx(sreg_rd_idx),
        .sreg_rd_data(row_sreg_rd_data[r*32 +: 32]),
        .sreg_wr_en(row_sreg_wr_en[r]), .sreg_wr_idx(sreg_wr_idx_sel),
        .sreg_wr_data(sreg_wr_data_sel), .sreg_err(row_sreg_err[r]),
        .err_bounds(row_err_bounds[r]), .ev_beat(row_ev_beat[r])
      );

      qcore_vsram #(.WORDS(VSRAM_WORDS), .W(256), .NE(8)) u_vsram (
        .clk(clk),
        .en_a(vs_en_a[r]), .addr_a(vs_addr_a[r*AW +: AW]),
        .rd_a(vs_rd_a[r*256 +: 256]),
        .en_b(vs_en_b[r]), .we_b(vs_we_b[r*8 +: 8]), .addr_b(vsb_addr),
        .wd_b(vsb_wdata), .rd_b(vs_rd_b[r*256 +: 256])
      );
    end
  endgenerate

  qcore_requant #(
    .WB(WB), .B_MAX(B_MAX), .ACC_W(ACC_W), .VSRAM_WORDS(VSRAM_WORDS)
  ) u_requant (
    .clk(clk), .rst(rst),
    .cmd_valid_gemv(cmd_valid_gemv), .cmd_op(cmd_op), .cmd_out_mode(cmd_out_mode),
    .cmd_accumulate(cmd_accumulate), .cmd_unit_meta(cmd_unit_meta),
    .cmd_track_absmax(cmd_track_absmax), .cmd_n(cmd_n), .cmd_vs_dst(cmd_vs_dst),
    .cmd_sreg_dst(cmd_sreg_dst), .cmd_sh0(cmd_sh0), .cmd_sh1(cmd_sh1),
    .cmd_imm32(cmd_imm32), .cmd_rows(cmd_rows), .cmd_sx_m(cmd_sx_m),
    .cmd_sx_e(cmd_sx_e),
    .acc_valid(acc_valid), .acc_ready(acc_ready), .acc_flat(acc_flat),
    .acc_tile(acc_tile), .acc_nvalid(acc_nvalid), .acc_last(acc_last),
    .meta_valid(meta_valid), .meta_ready(meta_ready), .meta_data(meta_data),
    .vsb_row(rq_vsb_row), .vsb_en(rq_vsb_en), .vsb_we(rq_vsb_we),
    .vsb_addr(rq_vsb_addr), .vsb_wdata(rq_vsb_wdata), .vsb_rdata(vsb_rdata),
    .d_wr_valid(d_wr_valid), .d_wr_ready(d_wr_ready), .d_wr_addr(d_wr_addr),
    .d_wr_data(d_wr_data), .d_wr_strb(d_wr_strb),
    .sreg_wr_en(rq_sreg_wr_en), .sreg_wr_row(rq_sreg_wr_row),
    .sreg_wr_idx(rq_sreg_wr_idx), .sreg_wr_data(rq_sreg_wr_data),
    .argmax_we(argmax_we), .argmax_tok(argmax_tok), .argmax_val(argmax_val),
    .sat_inc(rq_sat_inc), .err_shift_inc(rq_err_shift_inc),
    .err_bounds_inc(rq_err_bounds_inc),
    .done(done_gemv), .busy(requant_busy)
  );

  qcore_kv_writer #(
    .WB(WB), .B_MAX(B_MAX), .VSRAM_WORDS(VSRAM_WORDS)
  ) u_kv_writer (
    .clk(clk), .rst(rst),
    .cmd_valid_kv(cmd_valid_kv), .cmd_kv_transposed(cmd_kv_transposed),
    .cmd_addr_a(cmd_addr_a), .cmd_addr_m(cmd_addr_m),
    .cmd_k_stride(cmd_k_stride), .cmd_vs_src(cmd_vs_src), .cmd_rows(cmd_rows),
    .cmd_sx_m(cmd_sx_m), .cmd_sx_e(cmd_sx_e), .cmd_pos(cmd_pos),
    .vsb_row(kv_vsb_row), .vsb_en(kv_vsb_en), .vsb_addr(kv_vsb_addr),
    .vsb_rdata(vsb_rdata),
    .k_wr_valid(k_wr_valid), .k_wr_ready(k_wr_ready), .k_wr_addr(k_wr_addr),
    .k_wr_data(k_wr_data), .k_wr_strb(k_wr_strb),
    .err_bounds_inc(kv_err_bounds_inc), .done(done_kv), .busy(kv_busy)
  );

  // The vector unit owns both VSRAM ports of the row it is on while it is busy,
  // and the arbiter's TAG_VPU read port. `rdd_last` is the shared bus flag of
  // whichever tag returned this cycle, so every sink qualifies it with its own
  // valid; the vector unit does that itself, as the stream controller and the
  // fetch unit do. cmd_len and cmd_pos are the fields of the bundle VSOFTMAX and
  // VROPE read, so they arrive with those two opcodes.
  qcore_vpu_top #(
    .WB(WB), .B_MAX(B_MAX), .VL(VL), .VSRAM_WORDS(VSRAM_WORDS),
    .VPU_FIFO_BEATS(VPU_FIFO_BEATS), .MAX_BURST(MAX_BURST),
    .ROM_FILE_SIGMOID(ROM_FILE_SIGMOID), .ROM_FILE_EXP2(ROM_FILE_EXP2),
    .ROM_FILE_RSQRT(ROM_FILE_RSQRT), .ROM_FILE_RECIP(ROM_FILE_RECIP)
  ) u_vpu (
    .clk(clk), .rst(rst),
    .cmd_valid_vpu(cmd_valid_vpu), .cmd_op(cmd_op), .cmd_vq_w8(cmd_vq_w8),
    .cmd_vq_use_tracked(cmd_vq_use_tracked), .cmd_vq_group(cmd_vq_group),
    .cmd_vq_scale_mul(cmd_vq_scale_mul), .cmd_track_absmax(cmd_track_absmax),
    .cmd_addr_a(cmd_addr_a), .cmd_n(cmd_n), .cmd_len(cmd_len), .cmd_pos(cmd_pos),
    .cmd_vs_src(cmd_vs_src),
    .cmd_vs_dst(cmd_vs_dst), .cmd_vs_aux(cmd_vs_aux), .cmd_sreg_dst(cmd_sreg_dst),
    .cmd_sh0(cmd_sh0), .cmd_sh1(cmd_sh1), .cmd_imm32(cmd_imm32),
    .cmd_sqrt_m(cmd_sqrt_m), .cmd_sqrt_e(cmd_sqrt_e), .cmd_rows(cmd_rows),
    .cmd_sreg_u32(cmd_sreg_u32),
    .v_req_valid(v_req_valid), .v_req_ready(v_req_ready), .v_req_addr(v_req_addr),
    .v_req_len(v_req_len), .v_req_tag(v_req_tag),
    .rdv_valid(rdv_valid), .rd_data(rdd_data), .rd_data_last(rdd_last),
    .cur_row(vpu_cur_row),
    .vsa_en(vpu_vsa_en), .vsa_addr(vpu_vsa_addr), .vsa_rdata(vpu_vsa_rdata),
    .vsb_sel_dst(vpu_vsb_sel_dst), .vsb_en(vpu_vsb_en), .vsb_we(vpu_vsb_we),
    .vsb_addr(vpu_vsb_addr), .vsb_wdata(vpu_vsb_wdata), .vsb_rdata(vsb_rdata),
    .sreg_wr_en(vpu_sreg_wr_en), .sreg_wr_row(vpu_sreg_wr_row),
    .sreg_wr_idx(vpu_sreg_wr_idx), .sreg_wr_data(vpu_sreg_wr_data),
    .sat_inc(vpu_sat_inc), .err_shift_inc(vpu_err_shift_inc),
    .err_bounds_inc(vpu_err_bounds_inc),
    .done(done_vpu), .busy(vpu_busy)
  );

  // ================================================================ crossbar
  assign ws_ready  = &row_ws_ready;
  assign acc_valid = |row_acc_valid;
  assign gemv_beat = |row_ev_beat;

  // The tile descriptors of the handoff come from the lowest participating row;
  // the participating rows run in lockstep (docs/RTL.md 2.3).
  always_comb begin
    acc_tile   = {TW{1'b0}};
    acc_nvalid = {NVW{1'b0}};
    acc_last   = 1'b0;
    for (int i = B_MAX - 1; i >= 0; i--) begin
      if (cmd_rows[i]) begin
        acc_tile   = row_acc_tile[i*TW +: TW];
        acc_nvalid = row_acc_nvalid[i*NVW +: NVW];
        acc_last   = row_acc_last[i];
      end
    end
  end

  // Port A has one owner per descriptor: the rows during a GEMV or EMBED, each
  // reading bank src_row + r while it participates; the vector unit during a
  // V op, reading bank src_row + cur_row.
  assign vpu_vsa_bank = 4'(cmd_src_row + vpu_cur_row);

  always_comb begin
    vs_en_a       = {B_MAX{1'b0}};
    vs_addr_a     = {(B_MAX*AW){1'b0}};
    row_vsa_rdata = {(B_MAX*256){1'b0}};
    vpu_vsa_rdata = 256'd0;
    for (int v = 0; v < B_MAX; v++) begin
      if (vpu_busy) begin
        if (vpu_vsa_bank == 4'(v)) begin
          vs_en_a[v]            = vpu_vsa_en;
          vs_addr_a[v*AW +: AW] = vpu_vsa_addr;
          vpu_vsa_rdata         = vs_rd_a[v*256 +: 256];
        end
      end else begin
        for (int i = 0; i < B_MAX; i++) begin
          if (cmd_rows[i] && (4'(cmd_src_row + 4'(i)) == 4'(v))) begin
            vs_en_a[v]                  = row_vsa_en[i];
            vs_addr_a[v*AW +: AW]       = row_vsa_addr[i*AW +: AW];
            row_vsa_rdata[i*256 +: 256] = vs_rd_a[v*256 +: 256];
          end
        end
      end
    end
  end

  // Port B has one owner per descriptor: the requant during a GEMV or EMBED
  // (old-word reads and output writes of bank dst_row + r), the KV writer
  // during a KVWRITE (64-element reads of bank src_row + r), the vector unit
  // during a V op (second-operand reads of bank src_row + cur_row with
  // vsb_sel_dst low, output writes of bank dst_row + cur_row with it high).
  // Only one of the three is busy at a time: one descriptor is in flight.
  assign vsb_vpu    = vpu_busy;
  assign vsb_kv     = kv_busy;
  assign vsb_bank   = vsb_vpu
                      ? 4'((vpu_vsb_sel_dst ? cmd_dst_row : cmd_src_row) + vpu_cur_row)
                      : (vsb_kv ? 4'(cmd_src_row + kv_vsb_row)
                                : 4'(cmd_dst_row + rq_vsb_row));
  assign vsb_en_sel = vsb_vpu ? vpu_vsb_en    : (vsb_kv ? kv_vsb_en   : rq_vsb_en);
  assign vsb_we_sel = vsb_vpu ? vpu_vsb_we    : (vsb_kv ? 8'd0        : rq_vsb_we);
  assign vsb_addr   = vsb_vpu ? vpu_vsb_addr  : (vsb_kv ? kv_vsb_addr : rq_vsb_addr);
  assign vsb_wdata  = vsb_vpu ? vpu_vsb_wdata : rq_vsb_wdata;

  // The SREG write port of bank dst_row + r: the requant's, or the vector
  // unit's while it is busy (the scale or tracked absmax a V op leaves behind).
  assign sreg_wr_en_sel   = vsb_vpu ? vpu_sreg_wr_en   : rq_sreg_wr_en;
  assign sreg_wr_row_sel  = vsb_vpu ? vpu_sreg_wr_row  : rq_sreg_wr_row;
  assign sreg_wr_idx_sel  = vsb_vpu ? vpu_sreg_wr_idx  : rq_sreg_wr_idx;
  assign sreg_wr_data_sel = vsb_vpu ? vpu_sreg_wr_data : rq_sreg_wr_data;
  assign sreg_wr_bank     = 4'(cmd_dst_row + sreg_wr_row_sel);

  // ROW_EN bits at or above B_MAX name rows this core does not have.
  assign disp_row_en  = B_MAX'(row_en_q & ROW_EN_MASK);

  always_comb begin
    vs_en_b        = {B_MAX{1'b0}};
    vs_we_b        = {(B_MAX*8){1'b0}};
    vsb_rdata      = 256'd0;
    row_sreg_wr_en = {B_MAX{1'b0}};
    row_sreg_rd_en = {B_MAX{1'b0}};
    sreg_rd_data   = 32'd0;
    for (int v = 0; v < B_MAX; v++) begin
      if (vsb_bank == 4'(v)) begin
        vs_en_b[v]        = vsb_en_sel;
        vs_we_b[v*8 +: 8] = vsb_we_sel;
        vsb_rdata         = vs_rd_b[v*256 +: 256];
      end
      if (sreg_wr_bank == 4'(v)) row_sreg_wr_en[v] = sreg_wr_en_sel;
      if (sreg_rd_row == 4'(v)) begin
        row_sreg_rd_en[v] = sreg_rd_en;
        sreg_rd_data      = row_sreg_rd_data[v*32 +: 32];
      end
    end
  end

  // ================================================================ event counts
  // Per-cycle counts, added arithmetically (docs/RTL.md 2.8).
  assign sat_req_inc   = {5'd0, rq_sat_inc};
  assign sat_vpu_inc   = vpu_sat_inc;
  assign err_shift_inc = {6'd0, rq_err_shift_inc} + vpu_err_shift_inc;

  always_comb begin
    err_bounds_inc = {4'd0, disp_err_bounds_inc} + {6'd0, rq_err_bounds_inc} +
                     {6'd0, kv_err_bounds_inc} + {4'd0, vpu_err_bounds_inc};
    for (int i = 0; i < B_MAX; i++) begin
      err_bounds_inc = err_bounds_inc + {7'd0, row_err_bounds[i]} +
                       {7'd0, row_sreg_err[i]};
    end
  end

  // ================================================================ descriptor queue
  // Every opcode of the ISA has a unit at this level, so the fetch queue's head
  // goes straight to the dispatcher and the fault path is the dispatcher's
  // alone: an opcode byte that is none of the twelve ends the run with
  // FAULT = OPCODE and that byte in FAULT_OP (3.5), as do a row above B_MAX and
  // a misaligned PC.
  assign dq_valid       = fq_valid;
  assign fq_ready       = dq_ready;
  assign disp_abort_run = abort_run;
  assign csr_err_set    = err_set;
  assign csr_fault_code = fault_code;
  assign csr_fault_op   = fault_op;

`ifndef SYNTHESIS
  // The V-op range rule of docs/ISA.md, checked here because this is where the
  // element ranges and the row bases meet: the vector unit reads its operands
  // ahead of its writes, so a destination range is either exactly a source
  // range or disjoint from it -- but only within one bank. Row r reads bank
  // src_row + r and writes bank dst_row + r, so different bases are different
  // memories and no pair of element indices can alias. The two ranges have the
  // same length, so they are equal when the distance is zero, disjoint when it
  // is at least n, and partially overlapping in between. Element indices are 17
  // bits and n is 24, so the comparison is 25 bits wide.
  //
  // v_has_dst is the opcode set the rule applies to, and it mirrors
  // compiler.VECTOR_SOURCES: the vector opcodes that name a destination in
  // vs_dst. VROPE is not one of them -- it rewrites vs_src in place and leaves
  // vs_dst unused -- so its vs_dst field carries no range and comparing it
  // against vs_src would report an overlap that does not exist.
  localparam logic [7:0] OP_VRMSNORM = 8'(`QCORE_OP_VRMSNORM);
  localparam logic [7:0] OP_VQUANT   = 8'(`QCORE_OP_VQUANT);
  localparam logic [7:0] OP_VSILUMUL = 8'(`QCORE_OP_VSILUMUL);
  localparam logic [7:0] OP_VSOFTMAX = 8'(`QCORE_OP_VSOFTMAX);
  localparam logic [7:0] OP_VSUBC    = 8'(`QCORE_OP_VSUBC);

  logic [24:0] v_src_e, v_dst_e, v_aux_e, v_n_e, v_gap_sd, v_gap_ad;
  logic        v_one_bank, v_has_dst;

  assign v_src_e    = {9'd0, cmd_vs_src};
  assign v_dst_e    = {9'd0, cmd_vs_dst};
  assign v_aux_e    = {9'd0, cmd_vs_aux};
  assign v_n_e      = {1'b0, cmd_n};
  assign v_one_bank = (cmd_src_row == cmd_dst_row);
  assign v_has_dst  = (cmd_op == OP_VRMSNORM) || (cmd_op == OP_VQUANT)
                      || (cmd_op == OP_VSILUMUL) || (cmd_op == OP_VSOFTMAX)
                      || (cmd_op == OP_VSUBC);
  assign v_gap_sd   = (v_src_e > v_dst_e) ? (v_src_e - v_dst_e) : (v_dst_e - v_src_e);
  assign v_gap_ad   = (v_aux_e > v_dst_e) ? (v_aux_e - v_dst_e) : (v_dst_e - v_aux_e);

  always @(posedge clk) begin
    if (!rst) begin
      if (cmd_valid_vpu && v_has_dst && v_one_bank && (v_gap_sd != 25'd0)
          && (v_gap_sd < v_n_e)) begin
        $error("qcore_top: vs_dst %0d partially overlaps vs_src %0d over %0d elements of bank %0d",
               cmd_vs_dst, cmd_vs_src, cmd_n, cmd_src_row);
      end
      if (cmd_valid_vpu && v_one_bank && (cmd_op == OP_VSILUMUL) && (v_gap_ad != 25'd0)
          && (v_gap_ad < v_n_e)) begin
        $error("qcore_top: vs_dst %0d partially overlaps vs_aux %0d over %0d elements of bank %0d",
               cmd_vs_dst, cmd_vs_aux, cmd_n, cmd_src_row);
      end
      if (({2'd0, requant_busy} + {2'd0, kv_busy} + {2'd0, vpu_busy}) > 3'd1) begin
        $error("qcore_top: %0d units claim VSRAM port B in one cycle",
               {2'd0, requant_busy} + {2'd0, kv_busy} + {2'd0, vpu_busy});
      end
      for (int v = 0; v < B_MAX; v++) begin
        if (vs_en_a[v] && (vs_we_b[v*8 +: 8] != 8'd0) &&
            (vs_addr_a[v*AW +: AW] == vsb_addr)) begin
          $error("qcore_top: VSRAM %0d word %0d is read on port A and written on port B",
                 v, vsb_addr);
        end
      end
    end
  end
`endif
endmodule
