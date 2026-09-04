// Quettos Core -- speed probe: a simulation cost model. The accelerator RTL lives in rtl/.
// One vector lane: stage 1 uses the 32x16 -> 48-bit signed multiplier
// (x * c16, round-shift by sh1, sat32); stage 2 uses the 32x32 -> 64-bit
// signed multiplier (xh * c32, round-shift by sh2, sat32). Tracks absmax of
// the output and a saturation counter. p64 (the raw stage-2 product) is
// exposed so a sum-of-squares pass can use the lane with c16 = 1, c32 = x.
module probe_vpu_lane (
  input  logic               clk,
  input  logic               rst,
  input  logic               clr,        // clear absmax
  input  logic               in_valid,
  input  logic signed [31:0] x,
  input  logic signed [31:0] c32,
  input  logic        [15:0] c16,        // unsigned 16-bit mantissa
  input  logic        [5:0]  sh1,
  input  logic        [5:0]  sh2,
  output logic               out_valid,
  output logic signed [31:0] y,
  output logic signed [63:0] p64,
  output logic        [31:0] absmax,
  output logic        [31:0] sat_cnt
);
  function automatic logic signed [31:0] sat32_48(input logic signed [47:0] v);
    logic ovf;
    ovf      = ~((&v[47:31]) | ~(|v[47:31]));
    sat32_48 = ovf ? {v[47], {31{~v[47]}}} : v[31:0];
  endfunction

  function automatic logic signed [31:0] sat32_64(input logic signed [63:0] v);
    logic ovf;
    ovf      = ~((&v[63:31]) | ~(|v[63:31]));
    sat32_64 = ovf ? {v[63], {31{~v[63]}}} : v[31:0];
  endfunction

  // ---------------------------------------------------------------- stage 1
  logic signed [47:0] pa;
  logic [63:0]        pa_bits;
  logic [5:0]         sh1_m1;
  logic               rb1;
  logic signed [47:0] pa_sh;
  logic signed [31:0] xh;
  logic               xh_sat;

  assign pa      = $signed({{16{x[31]}}, x}) * $signed({{32{1'b0}}, c16});
  assign pa_bits = {{16{pa[47]}}, pa};
  assign sh1_m1  = sh1 - 6'd1;
  assign rb1     = (sh1 != 6'd0) & pa_bits[sh1_m1];
  assign pa_sh   = (pa >>> sh1) + $signed({47'd0, rb1});
  assign xh      = sat32_48(pa_sh);
  assign xh_sat  = ~((&pa_sh[47:31]) | ~(|pa_sh[47:31]));

  logic               v1;
  logic signed [31:0] xh_r;
  logic signed [31:0] c32_r;
  logic               sat1_r;

  always_ff @(posedge clk) begin
    if (rst) begin
      v1     <= 1'b0;
      xh_r   <= 32'sd0;
      c32_r  <= 32'sd0;
      sat1_r <= 1'b0;
    end else begin
      v1 <= in_valid;
      if (in_valid) begin
        xh_r   <= xh;
        c32_r  <= c32;
        sat1_r <= xh_sat;
      end
    end
  end

  // ---------------------------------------------------------------- stage 2
  logic signed [63:0] pb;
  logic [5:0]         sh2_m1;
  logic               rb2;
  logic signed [63:0] pb_sh;
  logic signed [31:0] y2;
  logic               y2_sat;
  logic [31:0]        y2_abs;

  assign pb     = $signed({{32{xh_r[31]}}, xh_r}) * $signed({{32{c32_r[31]}}, c32_r});
  assign sh2_m1 = sh2 - 6'd1;
  assign rb2    = (sh2 != 6'd0) & pb[sh2_m1];
  assign pb_sh  = (pb >>> sh2) + $signed({63'd0, rb2});
  assign y2     = sat32_64(pb_sh);
  assign y2_sat = ~((&pb_sh[63:31]) | ~(|pb_sh[63:31]));
  assign y2_abs = y2[31] ? (~y2 + 32'd1) : y2;

  always_ff @(posedge clk) begin
    if (rst) begin
      out_valid <= 1'b0;
      y         <= 32'sd0;
      p64       <= 64'sd0;
      absmax    <= 32'd0;
      sat_cnt   <= 32'd0;
    end else begin
      out_valid <= v1;
      if (clr) absmax <= 32'd0;
      if (v1) begin
        y   <= y2;
        p64 <= pb;
        if (!clr && (y2_abs > absmax)) absmax <= y2_abs;
        if (sat1_r || y2_sat) sat_cnt <= sat_cnt + 32'd1;
      end
    end
  end
endmodule
