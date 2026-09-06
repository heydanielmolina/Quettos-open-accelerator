// Quettos Core control and status registers: the host's 64-word window onto the
// sequencer. CTRL is write-one-to-pulse (START and STEP only while the core is
// idle, ABORT only while it is busy, START over STEP); every pulse leaves this
// module one cycle after the write. STATUS holds the three sticky bits with the
// fault code and the faulting opcode byte, set by the dispatcher's pulses,
// cleared by START, by STEP and by writing a one to the bit; a set pulse in the
// cycle of such a write wins. PC follows the dispatcher's retire update and
// takes a host write only while the core is idle, as ROW_EN, TOK and POS do.
// The four event counters add their per-cycle increments and clear on START;
// the PERF halves read the snapshot the perf module presents. Reads are
// registered: csr_rdata carries the addressed word one cycle after csr_re and
// holds it until the next read; unmapped words read zero.
`include "qcore_csr_defs.svh"
module qcore_csr (
  input  logic          clk,
  input  logic          rst,
  // host port
  input  logic          csr_we,
  input  logic [5:0]    csr_addr,
  input  logic [31:0]   csr_wdata,
  input  logic          csr_re,
  output logic [31:0]   csr_rdata,
  // control pulses to the dispatcher
  output logic          start,
  output logic          step,
  output logic          abort_run,
  // program counter
  output logic [31:0]   pc_q,
  input  logic          pc_set,
  input  logic [31:0]   pc_set_val,
  // per-token registers
  output logic [31:0]   row_en_q,
  output logic [31:0]   tok_q,
  output logic [31:0]   pos_q,
  // status
  input  logic          busy_i,
  input  logic          done_set,
  input  logic          step_halted_set,
  input  logic          err_set,
  input  logic [3:0]    fault_code,
  input  logic [7:0]    fault_op,
  // argmax result
  input  logic          argmax_we,
  input  logic [31:0]   argmax_tok,
  input  logic [31:0]   argmax_val,
  // per-cycle event counts
  input  logic [7:0]    sat_req_inc,
  input  logic [7:0]    sat_vpu_inc,
  input  logic [7:0]    err_shift_inc,
  input  logic [7:0]    err_bounds_inc,
  // PERF snapshot, counter i at bits [64 i +: 64]
  input  logic [1023:0] perf_snap
);
  localparam logic [5:0] W_CTRL        = 6'(`QCORE_CSR_CTRL);
  localparam logic [5:0] W_STATUS      = 6'(`QCORE_CSR_STATUS);
  localparam logic [5:0] W_PC          = 6'(`QCORE_CSR_PC);
  localparam logic [5:0] W_ROW_EN      = 6'(`QCORE_CSR_ROW_EN);
  localparam logic [5:0] W_TOK         = 6'(`QCORE_CSR_TOK);
  localparam logic [5:0] W_POS         = 6'(`QCORE_CSR_POS);
  localparam logic [5:0] W_ARGMAX_TOK  = 6'(`QCORE_CSR_ARGMAX_TOK);
  localparam logic [5:0] W_ARGMAX_VAL  = 6'(`QCORE_CSR_ARGMAX_VAL);
  localparam logic [5:0] W_SAT_REQ     = 6'(`QCORE_CSR_SAT_REQ);
  localparam logic [5:0] W_SAT_VPU     = 6'(`QCORE_CSR_SAT_VPU);
  localparam logic [5:0] W_ERR_SHIFT   = 6'(`QCORE_CSR_ERR_SHIFT);
  localparam logic [5:0] W_ERR_BOUNDS  = 6'(`QCORE_CSR_ERR_BOUNDS);
  localparam logic [5:0] W_ISA_VERSION = 6'(`QCORE_CSR_ISA_VERSION);
  localparam logic [5:0] W_PERF_FIRST  = 6'(`QCORE_PERF_BASE);
  localparam logic [5:0] W_PERF_LAST   = 6'(`QCORE_PERF_BASE + 2 * `QCORE_PERF_COUNT - 1);

  // ---------------------------------------------------------------- write decode
  logic sel_ctrl;    // CTRL write this cycle
  logic sel_status;  // STATUS write this cycle (write one to clear)
  logic host_wr;     // a host write that may reach a rw register

  assign sel_ctrl   = csr_we && (csr_addr == W_CTRL);
  assign sel_status = csr_we && (csr_addr == W_STATUS);
  assign host_wr    = csr_we && !busy_i;  // rw registers belong to the core while it runs

  logic req_start;
  logic req_step;
  logic req_abort;

  assign req_abort = sel_ctrl &&  busy_i && csr_wdata[`QCORE_CTRL_ABORT];
  assign req_start = sel_ctrl && !busy_i && csr_wdata[`QCORE_CTRL_START];
  assign req_step  = sel_ctrl && !busy_i && csr_wdata[`QCORE_CTRL_STEP] &&
                     !csr_wdata[`QCORE_CTRL_START];

  always_ff @(posedge clk) begin
    if (rst) begin
      start     <= 1'b0;
      step      <= 1'b0;
      abort_run <= 1'b0;
    end else begin
      start     <= req_start;
      step      <= req_step;
      abort_run <= req_abort;
    end
  end

  // ---------------------------------------------------------------- rw registers
  always_ff @(posedge clk) begin
    if (rst) begin
      pc_q     <= 32'd0;
      row_en_q <= 32'd0;
      tok_q    <= 32'd0;
      pos_q    <= 32'd0;
    end else begin
      if (host_wr && (csr_addr == W_ROW_EN)) row_en_q <= csr_wdata;
      if (host_wr && (csr_addr == W_TOK))    tok_q    <= csr_wdata;
      if (host_wr && (csr_addr == W_POS))    pos_q    <= csr_wdata;
      if (host_wr && (csr_addr == W_PC))     pc_q     <= csr_wdata;
      if (pc_set)                            pc_q     <= pc_set_val;  // the core wins
    end
  end

  // ---------------------------------------------------------------- status
  logic       done_q;
  logic       step_halted_q;
  logic       err_q;
  logic [3:0] fault_q;
  logic [7:0] fault_op_q;

  always_ff @(posedge clk) begin
    if (rst || start || step) begin
      done_q        <= 1'b0;
      step_halted_q <= 1'b0;
      err_q         <= 1'b0;
      fault_q       <= 4'd0;
      fault_op_q    <= 8'd0;
    end else begin
      if (sel_status && csr_wdata[`QCORE_STATUS_DONE])        done_q        <= 1'b0;
      if (sel_status && csr_wdata[`QCORE_STATUS_STEP_HALTED]) step_halted_q <= 1'b0;
      if (sel_status && csr_wdata[`QCORE_STATUS_ERR]) begin
        err_q      <= 1'b0;
        fault_q    <= 4'd0;
        fault_op_q <= 8'd0;
      end
      if (done_set)        done_q        <= 1'b1;  // a set pulse wins over the clear
      if (step_halted_set) step_halted_q <= 1'b1;
      if (err_set) begin
        err_q      <= 1'b1;
        fault_q    <= fault_code;
        fault_op_q <= fault_op;
      end
    end
  end

  logic [31:0] status_word;

  always_comb begin
    status_word = 32'd0;
    status_word[`QCORE_STATUS_DONE]        = done_q;
    status_word[`QCORE_STATUS_BUSY]        = busy_i;
    status_word[`QCORE_STATUS_STEP_HALTED] = step_halted_q;
    status_word[`QCORE_STATUS_ERR]         = err_q;
    status_word[`QCORE_STATUS_FAULT_LSB    +: `QCORE_STATUS_FAULT_W]    = fault_q;
    status_word[`QCORE_STATUS_FAULT_OP_LSB +: `QCORE_STATUS_FAULT_OP_W] = fault_op_q;
  end

  // ---------------------------------------------------------------- argmax
  logic [31:0] argmax_tok_q;
  logic [31:0] argmax_val_q;

  always_ff @(posedge clk) begin
    if (rst) begin
      argmax_tok_q <= 32'd0;
      argmax_val_q <= 32'd0;
    end else if (argmax_we) begin
      argmax_tok_q <= argmax_tok;
      argmax_val_q <= argmax_val;
    end
  end

  // ---------------------------------------------------------------- event counters
  logic [31:0] sat_req_q;
  logic [31:0] sat_vpu_q;
  logic [31:0] err_shift_q;
  logic [31:0] err_bounds_q;

  always_ff @(posedge clk) begin
    if (rst || start) begin
      sat_req_q    <= 32'd0;
      sat_vpu_q    <= 32'd0;
      err_shift_q  <= 32'd0;
      err_bounds_q <= 32'd0;
    end else begin
      sat_req_q    <= sat_req_q    + 32'(sat_req_inc);
      sat_vpu_q    <= sat_vpu_q    + 32'(sat_vpu_inc);
      err_shift_q  <= err_shift_q  + 32'(err_shift_inc);
      err_bounds_q <= err_bounds_q + 32'(err_bounds_inc);
    end
  end

  // ---------------------------------------------------------------- read port
  logic       perf_sel;
  logic [4:0] perf_idx;
  logic [9:0] perf_lsb;

  assign perf_sel = (csr_addr >= W_PERF_FIRST) && (csr_addr <= W_PERF_LAST);
  assign perf_idx = 5'(csr_addr - W_PERF_FIRST);
  assign perf_lsb = {perf_idx, 5'd0};

  logic [31:0] rd_mux;

  always_comb begin
    case (csr_addr)
      W_STATUS:      rd_mux = status_word;
      W_PC:          rd_mux = pc_q;
      W_ROW_EN:      rd_mux = row_en_q;
      W_TOK:         rd_mux = tok_q;
      W_POS:         rd_mux = pos_q;
      W_ARGMAX_TOK:  rd_mux = argmax_tok_q;
      W_ARGMAX_VAL:  rd_mux = argmax_val_q;
      W_SAT_REQ:     rd_mux = sat_req_q;
      W_SAT_VPU:     rd_mux = sat_vpu_q;
      W_ERR_SHIFT:   rd_mux = err_shift_q;
      W_ERR_BOUNDS:  rd_mux = err_bounds_q;
      W_ISA_VERSION: rd_mux = 32'(`QCORE_ISA_VERSION);
      default:       rd_mux = perf_sel ? perf_snap[perf_lsb +: 32] : 32'd0;
    endcase
  end

  always_ff @(posedge clk) begin
    if (rst) csr_rdata <= 32'd0;
    else if (csr_re) csr_rdata <= rd_mux;
  end
endmodule
