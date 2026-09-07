// Quettos Core lookup-table ROM: one ENTRIES x 32 image of a (value, delta)
// table, loaded from the checked-in rtl/gen/<table>.hex through the untyped
// ROM_FILE parameter. Word i is {v[15:0], dv[15:0]} of entry i, exactly the
// line lutgen.hex_lines writes; v is the Q1.15 sample and dv the forward
// difference to the next sample. Two independent read ports serve two vector
// lanes in the same cycle: with en set, the entry addressed this cycle is on
// v/dv the next cycle, and the outputs hold while the enable is low. The ROM
// holds no writes, no reset and no arithmetic; the interpolation between two
// samples is qcore_lut_interp.
module qcore_lut_rom #(
  parameter int ENTRIES  = 256,
  parameter     ROM_FILE = ""
) (
  input  logic                       clk,
  // port A
  input  logic                       en_a,
  input  logic [$clog2(ENTRIES)-1:0] idx_a,
  output logic [15:0]                v_a,
  output logic [15:0]                dv_a,
  // port B
  input  logic                       en_b,
  input  logic [$clog2(ENTRIES)-1:0] idx_b,
  output logic [15:0]                v_b,
  output logic [15:0]                dv_b
);
  logic [31:0] mem [0:ENTRIES-1];

  // The one initial block in the synthesizable RTL (docs/RTL.md 3.15): the
  // build passes the rtl/gen/*.hex path by parameter name, so no file name is
  // spelled in the source.
  initial begin
    $readmemh(ROM_FILE, mem);
  end

  always_ff @(posedge clk) begin
    if (en_a) begin
      v_a  <= mem[idx_a][31:16];
      dv_a <= mem[idx_a][15:0];
    end
  end

  always_ff @(posedge clk) begin
    if (en_b) begin
      v_b  <= mem[idx_b][31:16];
      dv_b <= mem[idx_b][15:0];
    end
  end
endmodule
