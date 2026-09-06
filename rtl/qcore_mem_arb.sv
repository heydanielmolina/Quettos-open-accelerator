// Quettos Core memory arbiter: the one QMEM read port and the one QMEM write
// port, shared by the stream controller, the VPU, the fetch unit, the KV writer
// and the dump path. Read requests pass through one registered stage with the
// priority stream > VPU > fetch; the fetch wins instead once MAX_BURST stream
// beats were granted since its last grant while the descriptor queue holds
// fewer than 4. A requester's ready is high while the stage is free or drains
// this cycle. Returned beats are routed by tag as rdf / rdw / rdm / rdv valids
// in the cycle they arrive; the payload fans out unchanged on rdd_*. Writes:
// KV writer over dump through one registered stage; wr_idle is high while every
// write accepted has been acknowledged. The PERF strobes lag their event by one
// cycle and come from registers.
module qcore_mem_arb #(
  parameter int WB        = 64,
  parameter int MAX_BURST = 64
) (
  input  logic            clk,
  input  logic            rst,
  // read requesters: stream, VPU, fetch
  input  logic            s_req_valid,
  output logic            s_req_ready,
  input  logic [31:0]     s_req_addr,
  input  logic [7:0]      s_req_len,
  input  logic [3:0]      s_req_tag,
  input  logic            v_req_valid,
  output logic            v_req_ready,
  input  logic [31:0]     v_req_addr,
  input  logic [7:0]      v_req_len,
  input  logic [3:0]      v_req_tag,
  input  logic            f_req_valid,
  output logic            f_req_ready,
  input  logic [31:0]     f_req_addr,
  input  logic [7:0]      f_req_len,
  input  logic [3:0]      f_req_tag,
  input  logic [3:0]      dq_count,
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
  // routed beats: fetch, weight, meta, VPU; the payload fans out on rdd_*
  output logic            rdf_valid,
  output logic            rdw_valid,
  output logic            rdm_valid,
  output logic            rdv_valid,
  output logic [WB*8-1:0] rdd_data,
  output logic            rdd_last,
  // write requesters: KV writer, dump
  input  logic            k_wr_valid,
  output logic            k_wr_ready,
  input  logic [31:0]     k_wr_addr,
  input  logic [WB*8-1:0] k_wr_data,
  input  logic [WB-1:0]   k_wr_strb,
  input  logic            d_wr_valid,
  output logic            d_wr_ready,
  input  logic [31:0]     d_wr_addr,
  input  logic [WB*8-1:0] d_wr_data,
  input  logic [WB-1:0]   d_wr_strb,
  // QMEM write
  output logic            wr_valid,
  input  logic            wr_ready,
  output logic [31:0]     wr_addr,
  output logic [WB*8-1:0] wr_data,
  output logic [WB-1:0]   wr_strb,
  input  logic            wr_ack,
  output logic            wr_idle,
  // PERF strobes
  output logic            ev_rd_beat,
  output logic            ev_wr_beat,
  output logic [7:0]      ev_wr_bytes
);
  localparam int DW = WB * 8;

  // ---------------------------------------------------------------- read requests
  logic        rq_valid;
  logic [31:0] rq_addr;
  logic [7:0]  rq_len;
  logic [3:0]  rq_tag;
  logic [15:0] s_beats;      // stream beats granted since the last fetch grant, saturating
  logic        stage_free;
  logic        f_reserved;   // the fetch slot rule holds this cycle
  logic        take_s;
  logic        take_v;
  logic        take_f;

  assign stage_free  = !rq_valid || rd_req_ready;
  assign f_reserved  = f_req_valid && (s_beats >= 16'(MAX_BURST)) && (dq_count < 4'd4);
  assign s_req_ready = stage_free && !f_reserved;
  assign v_req_ready = stage_free && !f_reserved && !s_req_valid;
  assign f_req_ready = stage_free && (f_reserved || (!s_req_valid && !v_req_valid));
  assign take_s      = s_req_valid && s_req_ready;
  assign take_v      = v_req_valid && v_req_ready;
  assign take_f      = f_req_valid && f_req_ready;

  always_ff @(posedge clk) begin
    if (rst) begin
      rq_valid <= 1'b0;
      rq_addr  <= 32'd0;
      rq_len   <= 8'd0;
      rq_tag   <= 4'd0;
      s_beats  <= 16'd0;
    end else begin
      if (stage_free) begin
        rq_valid <= take_s | take_v | take_f;
        if (take_s) begin
          rq_addr <= s_req_addr;
          rq_len  <= s_req_len;
          rq_tag  <= s_req_tag;
        end else if (take_v) begin
          rq_addr <= v_req_addr;
          rq_len  <= v_req_len;
          rq_tag  <= v_req_tag;
        end else if (take_f) begin
          rq_addr <= f_req_addr;
          rq_len  <= f_req_len;
          rq_tag  <= f_req_tag;
        end
      end
      if (take_f) begin
        s_beats <= 16'd0;
      end else if (take_s && !(&s_beats[15:8])) begin
        s_beats <= s_beats + {8'd0, s_req_len};
      end
    end
  end

  assign rd_req_valid = rq_valid;
  assign rd_req_addr  = rq_addr;
  assign rd_req_len   = rq_len;
  assign rd_req_tag   = rq_tag;

  // ---------------------------------------------------------------- read data routing
  logic [3:0] route;

  assign route     = qcore_pkg::rd_route(rd_data_tag);
  assign rdf_valid = rd_data_valid & route[0];
  assign rdw_valid = rd_data_valid & route[1];
  assign rdm_valid = rd_data_valid & route[2];
  assign rdv_valid = rd_data_valid & route[3];
  assign rdd_data  = rd_data;
  assign rdd_last  = rd_data_last;

  // ---------------------------------------------------------------- writes
  logic          wq_valid;
  logic [31:0]   wq_addr;
  logic [DW-1:0] wq_data;
  logic [WB-1:0] wq_strb;
  logic          wstage_free;
  logic          take_k;
  logic          take_d;
  logic          wr_take;
  logic [31:0]   issued;
  logic [31:0]   acked;
  logic [7:0]    strb_cnt;

  assign wstage_free = !wq_valid || wr_ready;
  assign k_wr_ready  = wstage_free;
  assign d_wr_ready  = wstage_free && !k_wr_valid;
  assign take_k      = k_wr_valid && k_wr_ready;
  assign take_d      = d_wr_valid && d_wr_ready;
  assign wr_take     = wq_valid && wr_ready;

  always_ff @(posedge clk) begin
    if (rst) begin
      wq_valid <= 1'b0;
      wq_addr  <= 32'd0;
      wq_strb  <= {WB{1'b0}};
    end else if (wstage_free) begin
      wq_valid <= take_k | take_d;
      if (take_k) begin
        wq_addr <= k_wr_addr;
        wq_strb <= k_wr_strb;
      end else if (take_d) begin
        wq_addr <= d_wr_addr;
        wq_strb <= d_wr_strb;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (wstage_free) begin
      if (take_k)      wq_data <= k_wr_data;
      else if (take_d) wq_data <= d_wr_data;
    end
  end

  assign wr_valid = wq_valid;
  assign wr_addr  = wq_addr;
  assign wr_data  = wq_data;
  assign wr_strb  = wq_strb;

  always_ff @(posedge clk) begin
    if (rst) begin
      issued <= 32'd0;
      acked  <= 32'd0;
    end else begin
      if (wr_take) issued <= issued + 32'd1;
      if (wr_ack)  acked  <= acked + 32'd1;
    end
  end

  assign wr_idle = !wq_valid && (issued == acked);

  // ---------------------------------------------------------------- PERF strobes
  always_comb begin
    strb_cnt = 8'd0;
    for (int i = 0; i < WB; i++) begin
      strb_cnt = strb_cnt + {7'd0, wq_strb[i]};
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      ev_rd_beat  <= 1'b0;
      ev_wr_beat  <= 1'b0;
      ev_wr_bytes <= 8'd0;
    end else begin
      ev_rd_beat  <= rd_data_valid;
      ev_wr_beat  <= wr_take;
      ev_wr_bytes <= wr_take ? strb_cnt : 8'd0;
    end
  end
endmodule
