// Quettos Core MAC lane group: 8 output-stationary lanes, each an int8 weight
// times the broadcast int16 activation into an ACC_W-bit accumulator. On en the
// live accumulator takes prod + (tile_start ? 0 : acc), or sext(w) << 24 on an
// EMBED beat; the first beat of the next tile moves the finished tile into the
// hold set. acc_drain shows the live set while buf_sel is high (a finished tile
// that no beat has overwritten yet) and the hold set otherwise, combinationally.
// One beat per cycle, no pipeline stage; the sets carry no reset.
module qcore_mac_lane_group #(
  parameter int ACC_W = 40
) (
  input  logic               clk,
  input  logic               en,          // a beat is consumed this cycle
  input  logic               tile_start,  // first beat of a tile (k == 0)
  input  logic               embed,       // EMBED beat: load sext(w) << 24
  input  logic               buf_sel,     // 1: acc_drain is the live set, 0: the hold set
  input  logic [63:0]        w,           // 8 int8 weights, lane i at [8i +: 8]
  input  logic [15:0]        a,           // broadcast int16 activation
  output logic [8*ACC_W-1:0] acc_drain    // lane i at [i*ACC_W +: ACC_W]
);
  logic [15:0] a_eff;

  // An EMBED beat multiplies by zero and enters its value through the
  // accumulator's override input, so one adder serves both cases.
  assign a_eff = embed ? 16'd0 : a;

  genvar i;
  generate
    for (i = 0; i < 8; i++) begin : g_lane
      logic signed [23:0]      prod;
      logic signed [ACC_W-1:0] prod_ext;
      logic signed [ACC_W-1:0] c_val;
      logic signed [ACC_W-1:0] acc;
      logic signed [ACC_W-1:0] hold;

      assign prod     = $signed({{16{w[8*i+7]}}, w[8*i +: 8]}) * $signed({{8{a_eff[15]}}, a_eff});
      assign prod_ext = {{(ACC_W-24){prod[23]}}, prod};
      assign c_val    = embed ? {{(ACC_W-32){w[8*i+7]}}, w[8*i +: 8], 24'd0} : {ACC_W{1'b0}};

      always_ff @(posedge clk) begin
        if (en) acc <= prod_ext + (tile_start ? c_val : acc);
      end

      always_ff @(posedge clk) begin
        if (en && tile_start) hold <= acc;
      end

      assign acc_drain[i*ACC_W +: ACC_W] = buf_sel ? acc : hold;
    end
  endgenerate
endmodule
