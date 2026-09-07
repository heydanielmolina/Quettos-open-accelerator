// Quettos Core vector lane: the elementwise arithmetic of the vector unit, one
// element per cycle. Stage 1 forms the products and the difference, stage 2
// rounds, shifts and saturates, so y and sat are valid two cycles after
// in_valid and p64 carries the raw 64-bit a * b of the same element for the
// sum-of-squares pass. Every value equals sw/quettos/numerics.py bit for bit:
// round_shift is round half toward +inf and sat clamps to the signed range.
//
//   L_MUL32  0  y = sat32(round_shift64(a * b, sh)),   p64 = a * b
//   L_MUL16  1  y = sat32(round_shift49(a * c, sh)),   c unsigned
//   L_ROPE_A 2  y = sat32(round_shift49(a * c - b * c2, sh)), c / c2 signed
//   L_ROPE_B 3  y = sat32(round_shift49(b * c + a * c2, sh)), c / c2 signed
//   L_SUB    4  y = sat32(a - b)
//   L_PASS   5  y = a                          (6 and 7 pass a through as well)
//
// a and b are int32. A coefficient is 16 bits widened to the 17-bit signed
// operand the products use: L_MUL16 widens it as the u16 of a mantissa, a
// table value or an sfloat scale (Rc_m and Sv_m reach 65535, sigmoid reaches
// 32768), while L_ROPE_A and L_ROPE_B widen it as the int16 of the Q1.14
// rotation row, whose cosines and sines are negative over half the positions.
// A signed coefficient outside a rotation reaches the same 32 x 17 product
// through L_ROPE_A with b = 0, and through L_MUL32 with the coefficient
// sign-extended into b.
//
// Widths: a * b is 64 bits, a * c is 49 bits and so is the rotation sum
// (|a * c| <= 2^46 and the sum of two is below 2^48); a - b is 33 bits. The
// saturation of every op but L_PASS is reported on sat, whether y is the final
// value of an element or an intermediate a second trip through the lane
// consumes -- VRMSNORM's xhat is the second case, so its int32 range is a
// property of the arithmetic and not a silent narrowing (docs/NUMERICS.md).
// Clips, maxima and the absmax are formed in qcore_vpu_top from y, and the sum
// of squares from p64. Latency is fixed: nothing stalls and no input is held
// past its cycle.
module qcore_vpu_lane (
  input  logic        clk,
  input  logic        rst,
  input  logic        in_valid,   // an element enters the pipeline this cycle
  input  logic [2:0]  op,
  input  logic [31:0] a,          // int32 first operand
  input  logic [31:0] b,          // int32 second operand
  input  logic [15:0] c,          // coefficient: mantissa, table value or cos
  input  logic [15:0] c2,         // second coefficient: sin
  input  logic [5:0]  sh,         // round-shift amount
  output logic        out_valid,  // two cycles after in_valid
  output logic [31:0] y,          // int32 result
  output logic [63:0] p64,        // the raw a * b product of the same element
  output logic        sat         // the result saturated
);
  localparam logic [2:0] L_MUL32  = 3'd0;
  localparam logic [2:0] L_MUL16  = 3'd1;
  localparam logic [2:0] L_ROPE_A = 3'd2;
  localparam logic [2:0] L_ROPE_B = 3'd3;
  localparam logic [2:0] L_SUB    = 3'd4;
  localparam logic [2:0] L_PASS   = 3'd5;

  // ------------------------------------------------------------------ stage 1: products
  logic               rope;
  logic signed [16:0] c_ext, c2_ext;
  logic signed [31:0] m1, m2;
  logic signed [48:0] p1, p2, s49;
  logic signed [63:0] p_ab;
  logic signed [32:0] d33;

  assign rope   = (op == L_ROPE_A) || (op == L_ROPE_B);
  assign c_ext  = rope ? $signed({c[15], c})   : $signed({1'b0, c});
  assign c2_ext = rope ? $signed({c2[15], c2}) : $signed({1'b0, c2});
  assign m1     = (op == L_ROPE_B) ? $signed(b) : $signed(a);
  assign m2     = (op == L_ROPE_B) ? $signed(a) : $signed(b);
  assign p1     = $signed({{17{m1[31]}}, m1}) * $signed({{32{c_ext[16]}}, c_ext});
  assign p2     = $signed({{17{m2[31]}}, m2}) * $signed({{32{c2_ext[16]}}, c2_ext});
  assign p_ab   = $signed({{32{a[31]}}, a}) * $signed({{32{b[31]}}, b});
  assign d33    = $signed({a[31], a}) - $signed({b[31], b});

  always_comb begin
    case (op)
      L_ROPE_A: s49 = p1 - p2;
      L_ROPE_B: s49 = p1 + p2;
      default:  s49 = p1;
    endcase
  end

  logic               v1;
  logic [2:0]         op1;
  logic [5:0]         sh1;
  logic signed [48:0] q49;
  logic signed [63:0] q64;
  logic signed [32:0] q33;
  logic [31:0]        qa;

  // ------------------------------------------------------------------ stage 2: round, shift, saturate
  logic signed [63:0] rs64;
  logic signed [48:0] rs49;
  logic [32:0]        y64, y49, y33, y_n;

  assign rs64 = qcore_pkg::round_shift64(q64, sh1);
  assign rs49 = qcore_pkg::round_shift49(q49, sh1);
  assign y64  = qcore_pkg::sat32_from64(rs64);
  assign y49  = qcore_pkg::sat32_from49(rs49);
  assign y33  = qcore_pkg::sat32_from33(q33);

  always_comb begin
    case (op1)
      L_MUL32:                     y_n = y64;
      L_MUL16, L_ROPE_A, L_ROPE_B: y_n = y49;
      L_SUB:                       y_n = y33;
      L_PASS:                      y_n = {1'b0, qa};
      default:                     y_n = {1'b0, qa};
    endcase
  end

  // ------------------------------------------------------------------ pipeline
  always_ff @(posedge clk) begin
    if (rst) begin
      v1        <= 1'b0;
      out_valid <= 1'b0;
    end else begin
      v1        <= in_valid;
      out_valid <= v1;
    end
  end

  always_ff @(posedge clk) begin
    if (in_valid) begin
      op1 <= op;
      sh1 <= sh;
      q49 <= s49;
      q64 <= p_ab;
      q33 <= d33;
      qa  <= a;
    end
  end

  always_ff @(posedge clk) begin
    if (v1) begin
      y   <= y_n[31:0];
      sat <= y_n[32];
      p64 <= q64;
    end
  end
endmodule
