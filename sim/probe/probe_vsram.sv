// Quettos Core -- speed probe: a simulation cost model. The accelerator RTL lives in rtl/.
// True-dual-port vector SRAM model: port A read-only (MAC activation words),
// port B read or write (VPU RMW and requant writes). Separate always_ff
// processes for reads and writes, registered read data (BRAM-style).
module probe_vsram #(
  parameter int WORDS = 4096,
  parameter int W     = 256
) (
  input  logic                     clk,
  input  logic                     en_a,
  input  logic [$clog2(WORDS)-1:0] addr_a,
  output logic [W-1:0]             rd_a,
  input  logic                     re_b,
  input  logic                     we_b,
  input  logic [$clog2(WORDS)-1:0] addr_b,
  input  logic [W-1:0]             wd_b,
  output logic [W-1:0]             rd_b
);
  logic [W-1:0] mem [0:WORDS-1];

  always_ff @(posedge clk) begin
    if (en_a) rd_a <= mem[addr_a];
  end

  always_ff @(posedge clk) begin
    if (re_b) rd_b <= mem[addr_b];
  end

  always_ff @(posedge clk) begin
    if (we_b) mem[addr_b] <= wd_b;
  end
endmodule
