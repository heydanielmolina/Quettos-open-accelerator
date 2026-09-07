// Quettos Core lookup-table interpolator: the linear step between two samples
// of a qcore_lut_rom entry, y = v + ((dv * frac8 + 128) >>> 8), which is
// numerics.Lut.interp. v is the unsigned Q1.15 sample of the entry the index
// selected, dv its signed forward difference and frac8 the eight index bits
// below it. The product is 25-bit signed (a 16-bit signed delta by a 9-bit
// signed fraction), the +128 rounds half toward +inf and the shift is
// arithmetic; the sum lies in [0, 65535] for every entry of every checked-in
// table, so the result is the low 16 bits. One cycle: y and out_valid are
// registered, so a request on in_valid leaves on the next edge. The module
// holds no table and no index arithmetic.
module qcore_lut_interp (
  input  logic        clk,
  input  logic        rst,
  input  logic        in_valid,
  input  logic [15:0] v,      // Q1.15 sample, unsigned
  input  logic [15:0] dv,     // forward difference, i16
  input  logic [7:0]  frac8,
  output logic        out_valid,
  output logic [15:0] y
);
  logic signed [24:0] prod;
  logic signed [16:0] step;
  logic signed [17:0] sum;

  assign prod = $signed(dv) * $signed({1'b0, frac8});
  // (prod + 128) >>> 8 stays inside 17 signed bits for every 16-bit delta.
  assign step = 17'(($signed(prod) + 25'sd128) >>> 8);
  assign sum  = $signed({2'b00, v}) + $signed({step[16], step});

  always_ff @(posedge clk) begin
    if (rst) out_valid <= 1'b0;
    else     out_valid <= in_valid;
  end

  always_ff @(posedge clk) begin
    if (in_valid) y <= sum[15:0];
  end

`ifndef SYNTHESIS
  // The table construction keeps every interpolated value inside [0, 65535].
  always @(posedge clk) begin
    if (!rst && in_valid && (sum[17:16] != 2'b00)) begin
      $error("qcore_lut_interp: v=%0d dv=%0d frac8=%0d leaves [0, 65535]", v, $signed(dv), frac8);
    end
  end
`endif
endmodule
