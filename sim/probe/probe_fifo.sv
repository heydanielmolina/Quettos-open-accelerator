// Quettos Core -- speed probe: a simulation cost model. The accelerator RTL lives in rtl/.
// Simple synchronous FIFO with a registered output stage (1-cycle read latency
// into an output register, so the storage maps to a BRAM with registered reads).
// SV subset: ANSI ports, logic, always_ff with synchronous reset, always_comb,
// 1-D unpacked array for storage, no unpacked-array ports.
module probe_fifo #(
  parameter int W     = 512,
  parameter int DEPTH = 128
) (
  input  logic          clk,
  input  logic          rst,
  input  logic          wr_valid,
  input  logic [W-1:0]  wr_data,
  output logic          full,
  output logic          out_valid,
  output logic [W-1:0]  out_data,
  input  logic          out_pop
);
  localparam int AW = $clog2(DEPTH);

  logic [W-1:0]  mem [0:DEPTH-1];
  logic [AW-1:0] wr_ptr;
  logic [AW-1:0] rd_ptr;
  logic [AW:0]   cnt;       // entries held in mem (excludes the output register)
  logic          do_rd;

  assign do_rd = (cnt != {(AW+1){1'b0}}) && (!out_valid || out_pop);
  assign full  = cnt[AW];   // cnt == DEPTH (DEPTH is a power of two)

  // storage write
  always_ff @(posedge clk) begin
    if (wr_valid) mem[wr_ptr] <= wr_data;
  end

  // registered read into the output stage
  always_ff @(posedge clk) begin
    if (do_rd) out_data <= mem[rd_ptr];
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      wr_ptr    <= {AW{1'b0}};
      rd_ptr    <= {AW{1'b0}};
      cnt       <= {(AW+1){1'b0}};
      out_valid <= 1'b0;
    end else begin
      if (wr_valid) wr_ptr <= wr_ptr + {{(AW-1){1'b0}}, 1'b1};
      if (do_rd)    rd_ptr <= rd_ptr + {{(AW-1){1'b0}}, 1'b1};
      cnt <= cnt + {{AW{1'b0}}, wr_valid} - {{AW{1'b0}}, do_rd};
      if (do_rd)        out_valid <= 1'b1;
      else if (out_pop) out_valid <= 1'b0;
    end
  end
endmodule
