// Quettos Core vector SRAM: one true-dual-port WORDS x W RAM per activation row.
// Port A reads (MAC activation words); port B reads or writes with one write
// strobe per NE-th of the word (requant, VPU and KV-writer traffic). Reads are
// registered: rd_a / rd_b hold the word addressed in the previous cycle with
// en_a / en_b set. Port B is READ_FIRST: a read and a write on port B in the
// same cycle return the old word. A port-A read of the address port B writes
// in the same cycle is a design error and is asserted in simulation.
module qcore_vsram #(
  parameter int WORDS = 4096,
  parameter int W     = 256,
  parameter int NE    = 8
) (
  input  logic                     clk,
  // port A: read only
  input  logic                     en_a,
  input  logic [$clog2(WORDS)-1:0] addr_a,
  output logic [W-1:0]             rd_a,
  // port B: read (en_b) and strobed write (we_b), independently or together
  input  logic                     en_b,
  input  logic [NE-1:0]            we_b,
  input  logic [$clog2(WORDS)-1:0] addr_b,
  input  logic [W-1:0]             wd_b,
  output logic [W-1:0]             rd_b
);
  localparam int EW = W / NE;

  logic [W-1:0] mem [0:WORDS-1] /* verilator public_flat_rd */;

  always_ff @(posedge clk) begin
    if (en_a) rd_a <= mem[addr_a];
  end

  always_ff @(posedge clk) begin
    if (en_b) rd_b <= mem[addr_b];
  end

  always_ff @(posedge clk) begin
    for (int i = 0; i < NE; i++) begin
      if (we_b[i]) mem[addr_b][i*EW +: EW] <= wd_b[i*EW +: EW];
    end
  end

`ifndef SYNTHESIS
  // Simulation-only check of the cross-port collision rule.
  always @(posedge clk) begin
    if (en_a && (|we_b) && (addr_a == addr_b)) begin
      $error("qcore_vsram: port A reads word %0d while port B writes it", addr_a);
    end
  end
`endif
endmodule
