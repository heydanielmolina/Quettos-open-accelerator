// Quettos Core -- speed probe: a simulation cost model. The accelerator RTL lives in rtl/.
//
// Full-width skeleton of the Quettos Core datapath, built only to measure
// simulator cycles-per-second on a design whose per-cycle simulation cost is
// representative of the final qcore_top: WB output-stationary int8x16 lanes
// with double-buffered 40-bit accumulators, weight FIFO, meta side-FIFO,
// activation word buffer over a true-dual-port 4096x256 vsram, one-per-cycle
// requant drain with argmax, VL vector lanes with 32x32 and 32x16 multipliers,
// and 64-bit perf counters. There is no sequencer, no memory arbiter, no
// KV writer, no LUT ROMs and no descriptor decode: the C++ harness issues
// commands directly. Every datapath feeds the checksum output.
module probe_top #(
  parameter int WB          = 64,
  parameter int B_MAX       = 1,
  parameter int VL          = 4,
  parameter int VSRAM_WORDS = 4096,
  parameter int FIFO_BEATS  = 128,
  parameter int ACC_W       = 40
) (
  input  logic                           clk,
  input  logic                           rst,
  // QMEM-like read-data return (tag 0 = weight beat, 1 = per-channel meta beat)
  input  logic                           rd_valid,
  input  logic                           rd_tag,
  input  logic [WB*8-1:0]                rd_data,
  output logic                           rd_ready,
  // command interface (one command in flight)
  input  logic                           start,
  input  logic                           cmd_vpu,     // 0: GEMV, 1: VPU pass
  input  logic                           cmd_argmax,  // GEMV out_mode ARGMAX (no vsram writes)
  input  logic [15:0]                    n_tiles,
  input  logic [15:0]                    k_len,       // GEMV K (multiple of 8) / VPU word count
  input  logic [$clog2(VSRAM_WORDS)-1:0] vs_src,
  input  logic [$clog2(VSRAM_WORDS)-1:0] vs_dst,      // GEMV destination / VPU base
  input  logic [15:0]                    sx_m,
  input  logic [7:0]                     sx_e,
  input  logic [7:0]                     sbias,
  input  logic [5:0]                     vsh1,
  input  logic [5:0]                     vsh2,
  output logic                           done,
  output logic                           busy,
  // CSR-like status
  output logic [31:0]                    checksum,
  output logic [31:0]                    absmax_req,
  output logic [31:0]                    absmax_vpu,
  output logic [31:0]                    argmax_idx,
  output logic [31:0]                    argmax_val,
  output logic [31:0]                    sat_count,
  output logic [31:0]                    err_count,
  output logic [63:0]                    perf_cycles,
  output logic [63:0]                    perf_busy,
  output logic [63:0]                    perf_beats,
  output logic [63:0]                    perf_mac
);
  localparam int DW    = WB * 8;
  localparam int AW    = $clog2(VSRAM_WORDS);
  localparam int NG    = WB / 8;
  localparam int MFIFO = 16;

  // ------------------------------------------------------------- FIFOs
  logic          wf_full, wf_out_valid, wf_pop;
  logic [DW-1:0] wf_out;
  logic          mf_full, mf_out_valid, mf_pop;
  logic [DW-1:0] mf_out;

  assign rd_ready = !wf_full && !mf_full;

  probe_fifo #(.W(DW), .DEPTH(FIFO_BEATS)) u_wfifo (
    .clk(clk), .rst(rst),
    .wr_valid(rd_valid && !rd_tag), .wr_data(rd_data), .full(wf_full),
    .out_valid(wf_out_valid), .out_data(wf_out), .out_pop(wf_pop)
  );

  probe_fifo #(.W(DW), .DEPTH(MFIFO)) u_mfifo (
    .clk(clk), .rst(rst),
    .wr_valid(rd_valid && rd_tag), .wr_data(rd_data), .full(mf_full),
    .out_valid(mf_out_valid), .out_data(mf_out), .out_pop(mf_pop)
  );

  // ------------------------------------------------------------- vsram
  logic          va_en;
  logic [AW-1:0] va_addr;
  logic [255:0]  va_rd;
  logic          vb_re, vb_we;
  logic [AW-1:0] vb_addr;
  logic [255:0]  vb_wd, vb_rd;

  probe_vsram #(.WORDS(VSRAM_WORDS), .W(256)) u_vsram (
    .clk(clk),
    .en_a(va_en), .addr_a(va_addr), .rd_a(va_rd),
    .re_b(vb_re), .we_b(vb_we), .addr_b(vb_addr), .wd_b(vb_wd), .rd_b(vb_rd)
  );

  // ------------------------------------------------------------- GEMV control
  localparam logic [2:0] G_IDLE  = 3'd0;
  localparam logic [2:0] G_LOAD0 = 3'd1;
  localparam logic [2:0] G_LOAD1 = 3'd2;
  localparam logic [2:0] G_RUN   = 3'd3;
  localparam logic [2:0] G_WAIT  = 3'd4;

  logic [2:0]    gst;
  logic [15:0]   tile, k, n_tiles_r, k_len_r;
  logic          argmax_r;
  logic [AW-1:0] vs_src_r, vs_dst_r;
  logic [15:0]   sx_m_r;
  logic [7:0]    sx_e_r, sbias_r;
  logic          buf_sel;
  logic [255:0]  act_cur, act_next, act_w0;
  logic          pf_pending;
  logic          mac_en, k_last, tile_last, k_lo0;
  logic          rq_start, rq_busy;
  logic          gemv_start, gemv_done;
  logic          vpu_start, vpu_done, vpu_busy;
  logic [AW-1:0] k_word;

  assign k_last     = (k == (k_len_r - 16'd1));
  assign tile_last  = (tile == (n_tiles_r - 16'd1));
  assign mac_en     = (gst == G_RUN) && wf_out_valid && !(k_last && rq_busy);
  assign wf_pop     = mac_en;
  assign rq_start   = mac_en && k_last;
  assign gemv_start = start && !cmd_vpu && (gst == G_IDLE) && !vpu_busy;
  assign vpu_start  = start &&  cmd_vpu && (gst == G_IDLE) && !vpu_busy;
  assign k_word     = k[AW+2:3];
  assign k_lo0      = (k[2:0] == 3'd0);

  // port A: activation word fetch (word 0 at load, word (k>>3)+1 prefetched at k%8==0)
  always_comb begin
    va_en   = 1'b0;
    va_addr = vs_src_r;
    if (gst == G_LOAD0) begin
      va_en = 1'b1;
    end else if (mac_en && k_lo0) begin
      va_en   = 1'b1;
      va_addr = vs_src_r + k_word + {{(AW-1){1'b0}}, 1'b1};
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      gst        <= G_IDLE;
      tile       <= 16'd0;
      k          <= 16'd0;
      n_tiles_r  <= 16'd1;
      k_len_r    <= 16'd8;
      argmax_r   <= 1'b0;
      vs_src_r   <= {AW{1'b0}};
      vs_dst_r   <= {AW{1'b0}};
      sx_m_r     <= 16'd0;
      sx_e_r     <= 8'd0;
      sbias_r    <= 8'd0;
      buf_sel    <= 1'b0;
      act_cur    <= 256'd0;
      act_next   <= 256'd0;
      act_w0     <= 256'd0;
      pf_pending <= 1'b0;
      gemv_done  <= 1'b0;
    end else begin
      gemv_done <= 1'b0;
      if (pf_pending) begin
        act_next   <= va_rd;
        pf_pending <= 1'b0;
      end
      case (gst)
        G_IDLE: begin
          if (gemv_start) begin
            n_tiles_r <= n_tiles;
            k_len_r   <= k_len;
            argmax_r  <= cmd_argmax;
            vs_src_r  <= vs_src;
            vs_dst_r  <= vs_dst;
            sx_m_r    <= sx_m;
            sx_e_r    <= sx_e;
            sbias_r   <= sbias;
            tile      <= 16'd0;
            k         <= 16'd0;
            gst       <= G_LOAD0;
          end
        end
        G_LOAD0: gst <= G_LOAD1;
        G_LOAD1: begin
          act_w0  <= va_rd;
          act_cur <= va_rd;
          gst     <= G_RUN;
        end
        G_RUN: begin
          if (mac_en) begin
            if (k[2:0] == 3'd0) pf_pending <= 1'b1;
            if (k_last) begin
              k       <= 16'd0;
              tile    <= tile + 16'd1;
              buf_sel <= ~buf_sel;
              act_cur <= act_w0;
              if (tile_last) gst <= G_WAIT;
            end else begin
              k <= k + 16'd1;
              if (k[2:0] == 3'd7) act_cur <= act_next;
            end
          end
        end
        G_WAIT: begin
          if (!rq_busy) begin
            gemv_done <= 1'b1;
            gst       <= G_IDLE;
          end
        end
        default: gst <= G_IDLE;
      endcase
    end
  end

  // ------------------------------------------------------------- MAC rows
  logic [15:0] a_bcast;
  logic [7:0]  a_off;
  logic [B_MAX*WB*ACC_W-1:0] acc_drain_all;

  assign a_off   = {k[2:0], 5'b00000};
  assign a_bcast = act_cur[a_off +: 16];

  genvar r, g;
  generate
    for (r = 0; r < B_MAX; r++) begin : g_row
      for (g = 0; g < NG; g++) begin : g_grp
        probe_mac_group #(.ACC_W(ACC_W)) u_grp (
          .clk        (clk),
          .rst        (rst),
          .en         (mac_en),
          .tile_start (k == 16'd0),
          .buf_sel    (buf_sel),
          .w          (wf_out[g*64 +: 64]),
          .a          (a_bcast),
          .acc_drain  (acc_drain_all[(r*WB + g*8)*ACC_W +: 8*ACC_W])
        );
      end
    end
  endgenerate

  // rows > 0 (B_MAX > 1 only) are folded into the checksum at tile end so they
  // are never dead; row 0 goes through the requant.
  logic [31:0] rows_fold;
  always_comb begin
    rows_fold = 32'd0;
    for (int rr = 1; rr < B_MAX; rr++) begin
      for (int q = 0; q < (WB * ACC_W) / 32; q++) begin
        rows_fold = rows_fold ^ acc_drain_all[rr*WB*ACC_W + q*32 +: 32];
      end
    end
  end

  // ------------------------------------------------------------- requant
  logic          rq_wr_en;
  logic [AW-1:0] rq_wr_addr;
  logic [255:0]  rq_wr_data;
  logic          rq_y_valid;
  logic [31:0]   rq_y;

  probe_requant #(.WB(WB), .ACC_W(ACC_W), .AW(AW)) u_rq (
    .clk(clk), .rst(rst),
    .gemv_start(gemv_start),
    .start(rq_start),
    .tile(tile),
    .acc_flat(acc_drain_all[WB*ACC_W-1:0]),
    .meta_valid(mf_out_valid), .meta_data(mf_out), .meta_pop(mf_pop),
    .sx_m(sx_m_r), .sx_e(sx_e_r), .sbias(sbias_r),
    .argmax_mode(argmax_r),
    .vs_dst(vs_dst_r),
    .busy(rq_busy),
    .wr_en(rq_wr_en), .wr_addr(rq_wr_addr), .wr_data(rq_wr_data),
    .y_valid(rq_y_valid), .y_out(rq_y),
    .absmax(absmax_req), .argmax_idx(argmax_idx), .argmax_val(argmax_val),
    .sat_count(sat_count), .err_count(err_count)
  );

  // ------------------------------------------------------------- VPU
  logic          vp_re, vp_we;
  logic [AW-1:0] vp_addr;
  logic [255:0]  vp_wd;
  logic          vp_chk_valid;
  logic [31:0]   vp_chk, vp_sat;

  probe_vpu #(.VL(VL), .AW(AW)) u_vpu (
    .clk(clk), .rst(rst),
    .start(vpu_start), .base(vs_dst), .n_words(k_len), .sh1(vsh1), .sh2(vsh2),
    .done(vpu_done), .busy(vpu_busy),
    .re_b(vp_re), .we_b(vp_we), .addr_b(vp_addr), .wd_b(vp_wd), .rd_b(vb_rd),
    .chk_valid(vp_chk_valid), .chk_data(vp_chk), .absmax(absmax_vpu), .sat_cnt(vp_sat)
  );

  // vsram port B: VPU RMW and requant writes are never concurrent (serialized ops)
  assign vb_re   = vp_re;
  assign vb_we   = vp_we | rq_wr_en;
  assign vb_addr = vpu_busy ? vp_addr : rq_wr_addr;
  assign vb_wd   = vpu_busy ? vp_wd   : rq_wr_data;

  // ------------------------------------------------------------- status / perf
  assign done = gemv_done | vpu_done;
  assign busy = (gst != G_IDLE) | vpu_busy;

  always_ff @(posedge clk) begin
    if (rst) begin
      checksum    <= 32'd0;
      perf_cycles <= 64'd0;
      perf_busy   <= 64'd0;
      perf_beats  <= 64'd0;
      perf_mac    <= 64'd0;
    end else begin
      perf_cycles <= perf_cycles + 64'd1;
      if (busy)                 perf_busy  <= perf_busy  + 64'd1;
      if (rd_valid && rd_ready) perf_beats <= perf_beats + 64'd1;
      if (mac_en)               perf_mac   <= perf_mac   + 64'd1;
      checksum <= {checksum[26:0], checksum[31:27]}
                ^ (rq_y_valid   ? rq_y   : 32'd0)
                ^ (vp_chk_valid ? vp_chk : 32'd0)
                ^ (rq_start     ? rows_fold : 32'd0)
                ^ (gemv_done    ? (argmax_idx ^ argmax_val ^ absmax_req ^ sat_count ^ err_count) : 32'd0)
                ^ (vpu_done     ? (absmax_vpu ^ vp_sat) : 32'd0);
    end
  end
endmodule
