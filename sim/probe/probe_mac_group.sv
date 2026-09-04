// Quettos Core -- speed probe: a simulation cost model. The accelerator RTL lives in rtl/.
// 8-lane MAC group: int8 weight x int16 broadcast activation -> explicitly
// sized 24-bit signed product -> ACC_W-bit accumulator with
// acc <= prod + (tile_start ? 0 : acc). Two accumulator sets, swapped by
// buf_sel at tile end; the inactive set is exposed for the requant drain.
module probe_mac_group #(
  parameter int ACC_W = 40
) (
  input  logic               clk,
  input  logic               rst,
  input  logic               en,          // a weight beat is consumed this cycle
  input  logic               tile_start,  // k == 0
  input  logic               buf_sel,     // active accumulator set
  input  logic [63:0]        w,           // 8 x int8 weights
  input  logic [15:0]        a,           // broadcast int16 activation
  output logic [8*ACC_W-1:0] acc_drain    // inactive set, flattened
);
  genvar i;
  generate
    for (i = 0; i < 8; i++) begin : g_lane
      logic signed [23:0]      prod;
      logic signed [ACC_W-1:0] prod_ext;
      logic signed [ACC_W-1:0] acc0;
      logic signed [ACC_W-1:0] acc1;

      assign prod     = $signed({{16{w[8*i+7]}}, w[8*i +: 8]}) * $signed({{8{a[15]}}, a});
      assign prod_ext = {{(ACC_W-24){prod[23]}}, prod};

      always_ff @(posedge clk) begin
        if (rst) begin
          acc0 <= {ACC_W{1'b0}};
          acc1 <= {ACC_W{1'b0}};
        end else if (en) begin
          if (!buf_sel) acc0 <= prod_ext + (tile_start ? {ACC_W{1'b0}} : acc0);
          else          acc1 <= prod_ext + (tile_start ? {ACC_W{1'b0}} : acc1);
        end
      end

      assign acc_drain[i*ACC_W +: ACC_W] = buf_sel ? acc0 : acc1;
    end
  endgenerate
endmodule
