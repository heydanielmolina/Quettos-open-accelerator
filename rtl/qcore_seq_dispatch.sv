// Quettos Core descriptor dispatcher: the sequencer's control loop. It pops one
// descriptor at a time from qcore_seq_fetch, decodes it, stops the program with
// a fault code on an opcode or a row the hardware cannot execute, waits for the
// auto-fence on every QMEM-reading opcode, reads one SREG scale per
// participating row, derives the POS-dependent extents, pulses the issue of the
// GEMV group, the VPU or the KV writer, and retires on the unit's done (a GEMV
// or EMBED also on the weight stream falling idle). One descriptor is in flight
// at a time and the next is popped only after the current one retires, so the
// issue pulse follows a pop by at least three cycles. Every program end waits
// for the write fence before it declares DONE or STEP_HALTED, so a fault, an
// ABORT and a step carry the same guarantee a HALT retire does: the host that
// reads memory between descriptors sees every write the descriptor made. Every
// busy cycle is classified into exactly one PERF bucket, MACS, WT_BYTES and
// DESCRIPTORS are counted per participating row as sw/quettos/isa_sim.py counts
// them, and ERR_BOUNDS is counted only for a descriptor the core commits to.
`include "qcore_csr_defs.svh"
module qcore_seq_dispatch #(
  parameter int WB    = 64,
  parameter int B_MAX = 1
) (
  input  logic                  clk,
  input  logic                  rst,
  // control and status registers
  input  logic                  start,
  input  logic                  step,
  input  logic                  abort_run,
  input  logic [31:0]           pc_q,
  input  logic [B_MAX-1:0]      row_en_q,
  input  logic [31:0]           tok_q,
  input  logic [31:0]           pos_q,
  output logic                  pc_set,
  output logic [31:0]           pc_set_val,
  output logic                  busy,
  output logic                  done_set,
  output logic                  step_halted_set,
  output logic                  err_set,
  output logic [3:0]            fault_code,
  output logic [7:0]            fault_op,
  output logic                  perf_clear,
  output logic                  perf_snapshot,
  // descriptor fetch
  output logic                  fetch_start,
  output logic [31:0]           fetch_pc,
  output logic                  fetch_step,
  output logic                  fetch_flush,
  output logic                  fetch_hold,
  input  logic                  dq_valid,
  input  logic [255:0]          dq_desc,
  output logic                  dq_ready,
  // SREG read port (the physical bank src_row + r)
  output logic                  sreg_rd_en,
  output logic [3:0]            sreg_rd_row,
  output logic [7:0]            sreg_rd_idx,
  input  logic [31:0]           sreg_rd_data,
  // issue
  output logic                  cmd_valid_gemv,
  output logic                  cmd_valid_vpu,
  output logic                  cmd_valid_kv,
  input  logic                  done_gemv,
  input  logic                  done_vpu,
  input  logic                  done_kv,
  output logic [7:0]            cmd_op,
  output logic [1:0]            cmd_out_mode,
  output logic                  cmd_accumulate,
  output logic                  cmd_unit_meta,
  output logic                  cmd_track_absmax,
  output logic                  cmd_vq_w8,
  output logic                  cmd_vq_use_tracked,
  output logic                  cmd_vq_group,
  output logic                  cmd_vq_scale_mul,
  output logic                  cmd_kv_transposed,
  output logic [31:0]           cmd_addr_a,
  output logic [31:0]           cmd_addr_m,
  output logic [31:0]           cmd_imm32,
  output logic [23:0]           cmd_n,
  output logic [15:0]           cmd_k,
  output logic [15:0]           cmd_k_stride,
  output logic [23:0]           cmd_len,
  output logic [15:0]           cmd_vs_src,
  output logic [15:0]           cmd_vs_dst,
  output logic [15:0]           cmd_vs_aux,
  output logic [7:0]            cmd_sreg_dst,
  output logic [3:0]            cmd_src_row,
  output logic [3:0]            cmd_dst_row,
  output logic [7:0]            cmd_sh0,
  output logic [7:0]            cmd_sh1,
  output logic [15:0]           cmd_sqrt_m,
  output logic [7:0]            cmd_sqrt_e,
  output logic [B_MAX-1:0]      cmd_rows,
  output logic [31:0]           cmd_pos,
  output logic [31:0]           cmd_tok,
  output logic [B_MAX*16-1:0]   cmd_sx_m,
  output logic [B_MAX*8-1:0]    cmd_sx_e,
  output logic [B_MAX*32-1:0]   cmd_sreg_u32,
  // unit status
  input  logic                  wr_idle,
  input  logic                  gemv_beat,
  input  logic                  stream_done,
  input  logic                  stream_busy,
  // PERF
  output logic [5:0]            ev_bucket,
  output logic                  ev_desc,
  output logic                  ev_macs_valid,
  output logic [39:0]           ev_macs,
  output logic                  ev_wt_valid,
  output logic [39:0]           ev_wt_bytes,
  output logic [3:0]            err_bounds_inc
);
  localparam int DESC_B = `QCORE_DESC_BYTES;
  localparam int RIW    = (B_MAX > 1) ? $clog2(B_MAX) : 1;

  localparam logic [7:0] OP_NOP      = 8'(`QCORE_OP_NOP);
  localparam logic [7:0] OP_HALT     = 8'(`QCORE_OP_HALT);
  localparam logic [7:0] OP_GEMV     = 8'(`QCORE_OP_GEMV);
  localparam logic [7:0] OP_EMBED    = 8'(`QCORE_OP_EMBED);
  localparam logic [7:0] OP_VRMSNORM = 8'(`QCORE_OP_VRMSNORM);
  localparam logic [7:0] OP_VQUANT   = 8'(`QCORE_OP_VQUANT);
  localparam logic [7:0] OP_VROPE    = 8'(`QCORE_OP_VROPE);
  localparam logic [7:0] OP_VSILUMUL = 8'(`QCORE_OP_VSILUMUL);
  localparam logic [7:0] OP_VSOFTMAX = 8'(`QCORE_OP_VSOFTMAX);
  localparam logic [7:0] OP_VSUBC    = 8'(`QCORE_OP_VSUBC);
  localparam logic [7:0] OP_KVWRITE  = 8'(`QCORE_OP_KVWRITE);
  localparam logic [7:0] OP_FENCE    = 8'(`QCORE_OP_FENCE);

  localparam logic [7:0] FLAG_VQ_W8          = 8'(`QCORE_VQ_W8);
  localparam logic [7:0] FLAG_VQ_USE_TRACKED = 8'(`QCORE_VQ_USE_TRACKED);
  localparam logic [7:0] FLAG_VQ_GROUP       = 8'(`QCORE_VQ_GROUP);
  localparam logic [7:0] FLAG_VQ_SCALE_MUL   = 8'(`QCORE_VQ_SCALE_MUL);
  localparam logic [7:0] FLAG_KVW_TRANSPOSED = 8'(`QCORE_KVW_TRANSPOSED);

  localparam logic [3:0] FAULT_OPCODE   = 4'(`QCORE_FAULT_OPCODE);
  localparam logic [3:0] FAULT_ROW      = 4'(`QCORE_FAULT_ROW);
  localparam logic [3:0] FAULT_PC_ALIGN = 4'(`QCORE_FAULT_PC_ALIGN);

  localparam logic [3:0] S_IDLE  = 4'd0;  // no program running
  localparam logic [3:0] S_CHK   = 4'd1;  // PC alignment, the cycle after start / step
  localparam logic [3:0] S_FETCH = 4'd2;  // waiting for the queue head
  localparam logic [3:0] S_DEC   = 4'd3;  // decode, faults, zero work
  localparam logic [3:0] S_PREP  = 4'd4;  // MACS / WT_BYTES, the EMBED unit scale
  localparam logic [3:0] S_FENCE = 4'd5;  // auto-fence on outstanding writes
  localparam logic [3:0] S_SREG  = 4'd6;  // one scale per participating row
  localparam logic [3:0] S_RUN   = 4'd7;  // issued, waiting for the unit
  localparam logic [3:0] S_STOP  = 4'd8;  // the write fence a fault, an ABORT or a step ends on

  logic [3:0]   state;
  logic [255:0] d;           // the descriptor in flight
  logic         step_mode;
  logic         abort_q;
  logic         stop_err;    // the stop waiting on the fence is a fault, not an ABORT
  logic         stop_step;   // ... and ends a step, so it sets STEP_HALTED, not DONE
  logic         gemv_done_q;
  logic [RIW-1:0] sr_r;
  logic         sr_ph;
  logic [39:0]  macs_q;
  logic [39:0]  wt_q;
  logic [3:0]   errb_q;      // the bounds events of the descriptor being decoded

  // ---------------------------------------------------------------- descriptor fields
  logic [7:0]  op;
  logic [7:0]  flags;
  logic [23:0] n_field;
  logic [15:0] k_field;
  logic [3:0]  src_row;
  logic [3:0]  dst_row;
  logic        f_n_from_pos;
  logic        f_k_from_pos;
  logic        f_len_from_pos;

  assign op             = qcore_pkg::desc_opcode(d);
  assign flags          = qcore_pkg::desc_flags(d);
  assign n_field        = qcore_pkg::desc_n(d);
  assign k_field        = qcore_pkg::desc_k(d);
  assign src_row        = qcore_pkg::desc_src_row(d);
  assign dst_row        = qcore_pkg::desc_dst_row(d);
  assign f_n_from_pos   = qcore_pkg::desc_n_from_pos(d);
  assign f_k_from_pos   = qcore_pkg::desc_k_from_pos(d);
  assign f_len_from_pos = qcore_pkg::desc_len_from_pos(d);

  logic op_nop;
  logic op_halt;
  logic op_gemv;
  logic op_embed;
  logic op_vquant;
  logic op_vsoftmax;
  logic op_vpu;
  logic op_kv;
  logic op_fence;
  logic op_known;
  logic uses_rows;
  logic needs_fence;
  logic needs_sreg;

  assign op_nop      = (op == OP_NOP);
  assign op_halt     = (op == OP_HALT);
  assign op_gemv     = (op == OP_GEMV);
  assign op_embed    = (op == OP_EMBED);
  assign op_vquant   = (op == OP_VQUANT);
  assign op_vsoftmax = (op == OP_VSOFTMAX);
  assign op_kv       = (op == OP_KVWRITE);
  assign op_fence    = (op == OP_FENCE);
  assign op_vpu      = (op == OP_VRMSNORM) || op_vquant || (op == OP_VROPE) ||
                       (op == OP_VSILUMUL) || op_vsoftmax || (op == OP_VSUBC);
  assign op_known    = op_nop || op_halt || op_gemv || op_embed || op_vpu || op_kv || op_fence;
  assign uses_rows   = !(op_nop || op_halt || op_fence);
  assign needs_fence = op_gemv || op_embed || op_fence || op_halt ||
                       (op == OP_VRMSNORM) || (op == OP_VROPE) || op_vsoftmax || (op == OP_VSUBC);
  assign needs_sreg  = op_gemv || op_kv ||
                       (op_vquant && cmd_vq_use_tracked && !cmd_vq_group);

  // ---------------------------------------------------------------- POS-derived extents
  // gemv_dims and softmax_len of sw/quettos/isa_sim.py, at the RTL widths.
  logic [32:0] pos_p1;
  logic        n_over;
  logic        k_over;
  logic [23:0] want_n;
  logic [24:0] n_ru;
  logic [23:0] n_pos;
  logic [15:0] k_pos;
  logic [23:0] n_dec;
  logic [15:0] k_dec;
  logic [32:0] len_want;
  logic        len_hi;
  logic        len_lo;
  logic        len_err;
  logic [23:0] len_dec;

  assign pos_p1   = {1'b0, pos_q} + 33'd1;
  assign n_over   = op_gemv && f_n_from_pos && (pos_p1 > {9'd0, n_field});
  assign k_over   = op_gemv && f_k_from_pos && (pos_p1 > {17'd0, k_field});
  assign want_n   = n_over ? n_field : pos_p1[23:0];
  assign n_ru     = ({1'b0, want_n} + 25'(WB - 1)) & ~(25'(WB - 1));
  assign n_pos    = (n_ru > {1'b0, n_field}) ? n_field : n_ru[23:0];
  assign k_pos    = k_over ? k_field : pos_p1[15:0];
  assign n_dec    = op_embed ? {8'd0, k_field}
                             : ((op_gemv && f_n_from_pos) ? n_pos : n_field);
  assign k_dec    = (op_gemv && f_k_from_pos) ? k_pos : k_field;
  assign len_want = f_len_from_pos ? pos_p1 : {1'b0, qcore_pkg::desc_imm32(d)};
  assign len_hi   = len_want > {9'd0, n_field};
  assign len_lo   = len_want < 33'd1;
  assign len_err  = op_vsoftmax && (len_hi || len_lo);
  assign len_dec  = len_hi ? n_field : (len_lo ? 24'd1 : len_want[23:0]);

  // ---------------------------------------------------------------- rows
  logic [B_MAX-1:0] rows_c;
  logic [B_MAX-1:0] row_bad;
  logic [B_MAX-1:0] sr_sel;
  logic             row_fault;
  logic             rows_empty;
  logic             sr_active;

  assign rows_c     = B_MAX'(qcore_pkg::desc_row_mask(d)) & row_en_q;
  assign row_fault  = |row_bad;
  assign rows_empty = (rows_c == {B_MAX{1'b0}});
  assign sr_active  = |(sr_sel & rows_c);

  generate
    for (genvar r = 0; r < B_MAX; r++) begin : g_row
      assign row_bad[r] = rows_c[r] &&
                          (((5'(src_row) + 5'(r)) >= 5'(B_MAX)) ||
                           ((5'(dst_row) + 5'(r)) >= 5'(B_MAX)));
      assign sr_sel[r]  = (sr_r == RIW'(r));
    end
  endgenerate

  // ---------------------------------------------------------------- zero work
  logic zero_work;

  assign zero_work = uses_rows &&
                     (rows_empty ||
                      ((op_gemv || op_embed) && ((n_dec == 24'd0) || (k_dec == 16'd0))) ||
                      (op_vpu && (n_field == 24'd0)));

  // ---------------------------------------------------------------- per-row event sums
  // ERR_BOUNDS counts one event per participating row; MACS and WT_BYTES are
  // the per-row work summed over the participating rows.
  logic [24:0] n_pad;
  logic [39:0] base40;
  logic [39:0] wt_row;
  logic [1:0]  ev_per_row;
  logic [3:0]  errb_c;
  logic [39:0] macs_c;
  logic [39:0] wt_c;

  assign n_pad      = ({1'b0, n_dec} + 25'(WB - 1)) & ~(25'(WB - 1));
  assign base40     = {15'd0, n_pad} * {24'd0, k_dec};
  assign wt_row     = op_embed ? (40'(k_dec) + 40'd8)
                               : (base40 + (cmd_unit_meta ? 40'd0 : {12'd0, n_pad, 3'd0}));
  assign ev_per_row = {1'b0, n_over} + {1'b0, k_over} + {1'b0, len_err};

  always_comb begin
    errb_c = 4'd0;
    macs_c = 40'd0;
    wt_c   = 40'd0;
    for (int r = 0; r < B_MAX; r++) begin
      if (rows_c[r]) begin
        errb_c = errb_c + {2'd0, ev_per_row};
        macs_c = macs_c + base40;
        wt_c   = wt_c + wt_row;
      end
    end
  end

  // ---------------------------------------------------------------- next state
  logic       pc_misaligned;
  logic       pop_ok;
  logic       sr_step;
  logic       sr_last;
  logic       sr_done;
  logic       retire_run;
  logic [3:0] nstate;
  logic       a_pop;
  logic       a_retire;
  logic       a_issue;
  logic       a_fault;
  logic [3:0] a_fcode;
  logic [7:0] a_fop;
  logic       a_abort_end;
  logic       a_fstart;
  logic       a_stop;
  logic       a_step_stop;
  logic       a_stopping;
  logic       a_end;
  logic       end_is_stop;
  logic       end_step;

  assign pc_misaligned = (pc_q[4:0] != 5'd0);
  assign pop_ok        = (state == S_FETCH) && !abort_q && !fetch_start && !fetch_flush;
  assign dq_ready      = pop_ok;
  assign sr_step       = !sr_active || sr_ph;
  assign sr_last       = (sr_r == RIW'(B_MAX - 1));
  assign sr_done       = sr_step && sr_last;
  assign sreg_rd_en    = (state == S_SREG) && sr_active && !sr_ph;
  assign sreg_rd_row   = src_row + 4'(sr_r);
  assign sreg_rd_idx   = qcore_pkg::desc_sreg_src(d);
  assign retire_run    = (op_gemv || op_embed) ? ((gemv_done_q || done_gemv) && !stream_busy)
                       : (op_kv                ? done_kv : done_vpu);

  always_comb begin
    nstate      = state;
    a_pop       = 1'b0;
    a_retire    = 1'b0;
    a_issue     = 1'b0;
    a_fault     = 1'b0;
    a_fcode     = 4'd0;
    a_fop       = 8'd0;
    a_abort_end = 1'b0;
    a_fstart    = 1'b0;
    case (state)
      S_IDLE: begin
        if (start || step) nstate = S_CHK;
      end
      S_CHK: begin
        if (pc_misaligned) begin
          a_fault = 1'b1;
          a_fcode = FAULT_PC_ALIGN;
        end else begin
          a_fstart = 1'b1;
          nstate   = S_FETCH;
        end
      end
      S_FETCH: begin
        if (abort_q) begin
          a_abort_end = 1'b1;
        end else if (pop_ok && dq_valid) begin
          a_pop  = 1'b1;
          nstate = S_DEC;
        end
      end
      S_DEC: begin
        // ABORT is acted on wherever nothing has been issued: the descriptor
        // popped here is left unexecuted and uncounted, with PC on it.
        if (abort_q) begin
          a_abort_end = 1'b1;
        end else if (!op_known) begin
          a_fault = 1'b1;
          a_fcode = FAULT_OPCODE;
          a_fop   = op;
        end else if (uses_rows && row_fault) begin
          a_fault = 1'b1;
          a_fcode = FAULT_ROW;
          a_fop   = op;
        end else if (op_nop || zero_work) begin
          a_retire = 1'b1;
        end else begin
          nstate = S_PREP;
        end
      end
      S_PREP: begin
        if (abort_q) begin
          a_abort_end = 1'b1;
        end else if (needs_fence) begin
          nstate = S_FENCE;
        end else if (needs_sreg) begin
          nstate = S_SREG;
        end else begin
          a_issue = 1'b1;
          nstate  = S_RUN;
        end
      end
      S_FENCE: begin
        if (abort_q) begin
          a_abort_end = 1'b1;
        end else if (wr_idle) begin
          if (op_halt || op_fence) begin
            a_retire = 1'b1;
          end else if (needs_sreg) begin
            nstate = S_SREG;
          end else begin
            a_issue = 1'b1;
            nstate  = S_RUN;
          end
        end
      end
      S_SREG: begin
        if (abort_q) begin
          a_abort_end = 1'b1;
        end else if (sr_done) begin
          a_issue = 1'b1;
          nstate  = S_RUN;
        end
      end
      S_RUN: begin
        if (retire_run) a_retire = 1'b1;
      end
      S_STOP: begin
        // the write fence a fault or an ABORT ends on; a_end below closes it
      end
      default: nstate = S_IDLE;
    endcase
    // Program end. A fault, an ABORT and a step retire all end the run only
    // once every issued write has been acknowledged, the guarantee docs/ISA.md
    // attaches to DONE and the one a HALT retire already carries out of
    // S_FENCE. The step takes the same fence because the host reads VSRAM and
    // QMEM between descriptors, and a KVWRITE or a dump retires before its
    // writes are acked. A HALT retire needs no further wait: it has just left
    // the fence. A stepped descriptor ends on STEP_HALTED, a stepped HALT and
    // a step an ABORT cut short on DONE.
    a_step_stop = a_retire && step_mode && !abort_q && !op_halt;
    a_stop      = a_fault || a_abort_end || (a_retire && abort_q && !op_halt) || a_step_stop;
    a_stopping  = a_stop || (state == S_STOP);
    end_is_stop = a_stopping && wr_idle;
    end_step    = end_is_stop && ((state == S_STOP) ? stop_step : a_step_stop);
    a_end       = end_is_stop || (a_retire && op_halt);
    if (a_retire)   nstate = (op_halt || step_mode || abort_q) ? S_IDLE : S_FETCH;
    if (a_stopping) nstate = wr_idle ? S_IDLE : S_STOP;
  end

  // ---------------------------------------------------------------- sequencing
  always_ff @(posedge clk) begin
    if (rst) begin
      state           <= S_IDLE;
      busy            <= 1'b0;
      step_mode       <= 1'b0;
      abort_q         <= 1'b0;
      stop_err        <= 1'b0;
      stop_step       <= 1'b0;
      gemv_done_q     <= 1'b0;
      sr_r            <= {RIW{1'b0}};
      sr_ph           <= 1'b0;
      d               <= 256'd0;
      macs_q          <= 40'd0;
      wt_q            <= 40'd0;
      errb_q          <= 4'd0;
      cmd_pos         <= 32'd0;
      cmd_tok         <= 32'd0;
      cmd_sx_m        <= {(B_MAX*16){1'b0}};
      cmd_sx_e        <= {(B_MAX*8){1'b0}};
      cmd_sreg_u32    <= {(B_MAX*32){1'b0}};
      pc_set          <= 1'b0;
      pc_set_val      <= 32'd0;
      done_set        <= 1'b0;
      step_halted_set <= 1'b0;
      err_set         <= 1'b0;
      fault_code      <= 4'd0;
      fault_op        <= 8'd0;
      perf_snapshot   <= 1'b0;
      fetch_start     <= 1'b0;
      fetch_flush     <= 1'b0;
      cmd_valid_gemv  <= 1'b0;
      cmd_valid_vpu   <= 1'b0;
      cmd_valid_kv    <= 1'b0;
      ev_desc         <= 1'b0;
      ev_macs_valid   <= 1'b0;
      ev_wt_valid     <= 1'b0;
      err_bounds_inc  <= 4'd0;
    end else begin
      state           <= nstate;
      pc_set          <= 1'b0;
      done_set        <= 1'b0;
      step_halted_set <= 1'b0;
      err_set         <= 1'b0;
      perf_snapshot   <= 1'b0;
      fetch_start     <= 1'b0;
      fetch_flush     <= 1'b0;
      cmd_valid_gemv  <= 1'b0;
      cmd_valid_vpu   <= 1'b0;
      cmd_valid_kv    <= 1'b0;
      ev_desc         <= 1'b0;
      ev_macs_valid   <= 1'b0;
      ev_wt_valid     <= 1'b0;
      err_bounds_inc  <= 4'd0;

      if (abort_run) abort_q <= 1'b1;
      if (done_gemv) gemv_done_q <= 1'b1;

      if ((state == S_IDLE) && (start || step)) begin
        busy      <= 1'b1;
        step_mode <= step && !start;
        abort_q   <= 1'b0;
      end

      if (a_fstart) begin
        fetch_start <= 1'b1;
        fetch_flush <= 1'b1;
      end

      if (a_pop) begin
        d           <= dq_desc;
        cmd_pos     <= pos_q;
        cmd_tok     <= tok_q;
        gemv_done_q <= 1'b0;
      end

      // ERR_BOUNDS describes a descriptor the core commits to. The decode
      // latches the count of the descriptor in hand; the counter pulses at the
      // issue, and at the retire of a descriptor whose work is empty, which is
      // the one commit that never issues. A descriptor an ABORT stops between
      // the decode and the issue is never executed and contributes nothing.
      if (state == S_DEC) errb_q <= errb_c;
      if (a_issue)                           err_bounds_inc <= errb_q;
      else if (a_retire && (state == S_DEC)) err_bounds_inc <= errb_c;

      if (state == S_PREP) begin
        macs_q <= macs_c;
        wt_q   <= wt_c;
        sr_r   <= {RIW{1'b0}};
        sr_ph  <= 1'b0;
        if (op_embed) begin
          for (int r = 0; r < B_MAX; r++) begin
            cmd_sx_m[r*16 +: 16] <= 16'h8000;   // Sx = 1.0 = {2^15, -15}
            cmd_sx_e[r*8  +: 8]  <= 8'hF1;
          end
        end
      end

      if (state == S_SREG) begin
        if (sr_active && !sr_ph) begin
          sr_ph <= 1'b1;
        end else begin
          sr_ph <= 1'b0;
          sr_r  <= sr_r + RIW'(1);
        end
        if (sr_active && sr_ph) begin
          for (int r = 0; r < B_MAX; r++) begin
            if (sr_sel[r]) begin
              cmd_sx_m[r*16 +: 16]     <= sreg_rd_data[15:0];
              cmd_sx_e[r*8  +: 8]      <= sreg_rd_data[23:16];
              cmd_sreg_u32[r*32 +: 32] <= sreg_rd_data;
            end
          end
        end
      end

      if (a_issue) begin
        cmd_valid_gemv <= op_gemv || op_embed;
        cmd_valid_vpu  <= op_vpu;
        cmd_valid_kv   <= op_kv;
        ev_macs_valid  <= op_gemv;
        ev_wt_valid    <= op_embed || (op_gemv && !f_n_from_pos && !f_k_from_pos);
        gemv_done_q    <= 1'b0;
      end

      if (a_retire) begin
        pc_set     <= 1'b1;
        pc_set_val <= pc_q + 32'(DESC_B);
        ev_desc    <= 1'b1;
      end

      if (a_fault) begin
        fault_code <= a_fcode;
        fault_op   <= a_fop;
      end

      if (a_stop) begin
        stop_err  <= a_fault;
        stop_step <= a_step_stop;
      end

      if (a_end) begin
        perf_snapshot <= 1'b1;
        fetch_flush   <= 1'b1;
        busy          <= 1'b0;
        abort_q       <= 1'b0;
        if (end_step) begin
          step_halted_set <= 1'b1;
        end else begin
          done_set <= 1'b1;
          if (end_is_stop) err_set <= (state == S_STOP) ? stop_err : a_fault;
        end
      end
    end
  end

  // ---------------------------------------------------------------- issue bundle
  assign cmd_op             = op;
  assign cmd_out_mode       = qcore_pkg::desc_out_mode(d);
  assign cmd_accumulate     = qcore_pkg::desc_accumulate(d);
  assign cmd_unit_meta      = qcore_pkg::desc_unit_meta(d);
  assign cmd_track_absmax   = qcore_pkg::desc_track_absmax(d);
  assign cmd_vq_w8          = (flags & FLAG_VQ_W8) != 8'd0;
  assign cmd_vq_use_tracked = (flags & FLAG_VQ_USE_TRACKED) != 8'd0;
  assign cmd_vq_group       = (flags & FLAG_VQ_GROUP) != 8'd0;
  assign cmd_vq_scale_mul   = (flags & FLAG_VQ_SCALE_MUL) != 8'd0;
  assign cmd_kv_transposed  = (flags & FLAG_KVW_TRANSPOSED) != 8'd0;
  assign cmd_addr_a         = qcore_pkg::desc_addr_a(d);
  assign cmd_addr_m         = qcore_pkg::desc_addr_m(d);
  assign cmd_imm32          = qcore_pkg::desc_imm32(d);
  assign cmd_n              = n_dec;
  assign cmd_k              = k_dec;
  assign cmd_k_stride       = k_field;
  assign cmd_len            = len_dec;
  assign cmd_vs_src         = qcore_pkg::desc_vs_src(d);
  assign cmd_vs_dst         = qcore_pkg::desc_vs_dst(d);
  assign cmd_vs_aux         = qcore_pkg::desc_vs_aux(d);
  assign cmd_sreg_dst       = qcore_pkg::desc_sreg_dst(d);
  assign cmd_src_row        = src_row;
  assign cmd_dst_row        = dst_row;
  assign cmd_sh0            = qcore_pkg::desc_sh0(d);
  assign cmd_sh1            = qcore_pkg::desc_sh1(d);
  assign cmd_sqrt_m         = cmd_addr_m[15:0];
  assign cmd_sqrt_e         = cmd_addr_m[23:16];
  assign cmd_rows           = rows_c;
  assign ev_macs            = macs_q;
  assign ev_wt_bytes        = wt_q;

  assign perf_clear = start;
  assign fetch_pc   = pc_q;
  assign fetch_step = step_mode;

  // The descriptor prefetch is a QMEM read and takes the same auto-fence every
  // reading opcode takes: no fetch request is issued while a KV or dump write
  // is unacknowledged, so a fetch beat never passes a write of an earlier
  // descriptor.
  assign fetch_hold = !wr_idle;

  // ---------------------------------------------------------------- PERF buckets
  // First match wins: MAC_ACTIVE, STALL_DRAIN, STALL_MEM, STALL_VPU, STALL_KV,
  // STALL_SEQ. One descriptor is in flight, so rows 4 to 6 cannot collide with
  // the first three. The auto-fence and the fence a fault or an ABORT ends on
  // are both STALL_KV.
  logic in_flight;
  logic fl_gemv;
  logic b_mac;
  logic b_drain;
  logic b_mem;
  logic b_vpu;
  logic b_kv;
  logic b_seq;

  assign in_flight = (state == S_RUN);
  assign fl_gemv   = in_flight && (op_gemv || op_embed);
  assign b_mac     = fl_gemv && op_gemv && gemv_beat;
  // qcore_stream_ctrl re-evaluates stream_done at the registered command pulse,
  // so on the issue cycle the level still belongs to the previous descriptor;
  // that cycle is the new descriptor waiting for its first beat.
  assign b_drain   = !b_mac && fl_gemv && stream_done && !cmd_valid_gemv;
  assign b_mem     = !b_mac && !b_drain && fl_gemv;
  assign b_vpu     = in_flight && op_vpu;
  assign b_kv      = (in_flight && op_kv) || (state == S_FENCE) || (state == S_STOP);
  assign b_seq     = !(b_mac || b_drain || b_mem || b_vpu || b_kv);
  assign ev_bucket = busy ? {b_drain, b_seq, b_kv, b_vpu, b_mem, b_mac} : 6'd0;

`ifndef SYNTHESIS
  // Simulation-only checks: exactly one bucket per busy cycle, none otherwise,
  // and at most one issue pulse per cycle.
  always @(posedge clk) begin
    if (!rst) begin
      if (busy && ((ev_bucket == 6'd0) || ((ev_bucket & (ev_bucket - 6'd1)) != 6'd0)))
        $error("qcore_seq_dispatch: ev_bucket is not one-hot while busy");
      if (!busy && (ev_bucket != 6'd0))
        $error("qcore_seq_dispatch: ev_bucket set while idle");
      if (({2'd0, cmd_valid_gemv} + {2'd0, cmd_valid_vpu} + {2'd0, cmd_valid_kv}) > 3'd1)
        $error("qcore_seq_dispatch: more than one issue pulse");
    end
  end
`endif
endmodule
