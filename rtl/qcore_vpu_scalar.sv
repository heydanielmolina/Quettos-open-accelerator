// Quettos Core vector scalar unit: the one per-pass reduction the vector unit
// cannot do a lane at a time. It normalises a wide non-negative magnitude,
// looks up a reciprocal or a reciprocal square root, folds the result into an
// sfloat and hands back the mantissa and the shift the following pass applies.
// No divider and no square root: the leading-one search is a bit-length, the
// normalisation a shift, and the curve comes from qcore_lut_rom through
// qcore_lut_interp. Every value equals sw/quettos/numerics.py bit for bit.
//
//   RMS_SCALE   (req_op 0, req_x = ss')     numerics.rmsnorm
//     L = bitlen64(ss'), 2e = L-1 for odd L else L-2, m_q16 = ss' * 2^(16-2e)
//     truncated into [2^16, 2^18), R = rsqrt_q15(m_q16),
//     Rc = sfloat_mul(sfloat_from_int(R, -15), sqrt_d),
//     rsp_m = Rc_m, rsp_shift = S1 = -(Rc_e + FRAC_X - sh - e)
//   QUANT_SCALE (req_op 1, req_x = a_eff)   numerics.quant
//     e_a = bitlen(a_eff) - 16, a_hi = the top 16 bits, inv = recip_q15(a_hi),
//     rsp_m = inv, rsp_shift = 31 + e_a - w,
//     Sx = {a_hi, e_a - (w-1) - FRAC_in}, multiplied by scale_mul on req_mul_en
//   SOFTMAX_NORM (req_op 2, req_x = total)  numerics.softmax
//     e_s = bitlen(total) - 16, sum_hi = the top 16 bits,
//     rsp_m = inv = recip_q15(sum_hi), rsp_shift = 7 + e_s. total arrives on the
//     whole width from the vector unit's 56-bit accumulator: at most 2^24
//     exponentials of at most 2^23 each, so 47 bits, which bitlen64 and the
//     normalisation cover as they do the 49 bits of ss'
//
// req_x carries the magnitude as an unsigned integer and is zero exactly on the
// zero cases numerics takes early (ss' == 0, a == 0, an empty sum), which
// rsp_zero reports; the mantissa of a zero input is the canonical zero {0, 0}.
// rsp_shift is clamped into [0, 63] and rsp_shift_err marks the clamp, which a
// compiled program reaches only as the negative S1 of docs/RTL.md 2.8.
//
// The rsqrt ROM's two ports read the two segments of its domain, so the segment
// bit of m_q16 selects a result rather than an address; the recip ROM's port A
// serves QUANT_SCALE and its port B SOFTMAX_NORM. Latency is a fixed five
// cycles, one request may enter every cycle, and nothing stalls.
module qcore_vpu_scalar #(
  parameter ROM_FILE_RSQRT = "",
  parameter ROM_FILE_RECIP = ""
) (
  input  logic               clk,
  input  logic               rst,
  input  logic               req_valid,
  input  logic [1:0]         req_op,
  input  logic [63:0]        req_x,      // ss' (49 bits) / a_eff (33) / total (47)
  input  logic [7:0]         req_sh0,    // FRAC_X / FRAC_in
  input  logic [5:0]         req_sh,     // VRMSNORM sh
  input  logic               req_w8,     // VQUANT output width: 1 = int8
  input  logic               req_mul_en, // VQUANT SCALE_MUL
  input  logic [15:0]        req_aux_m,  // sqrt(d) / scale_mul mantissa
  input  logic signed [7:0]  req_aux_e,  // sqrt(d) / scale_mul exponent
  output logic               rsp_valid,  // five cycles after req_valid
  output logic [15:0]        rsp_m,      // Rc_m / inv
  output logic [5:0]         rsp_shift,  // S1 / 31 + e_a - w / 7 + e_s
  output logic               rsp_shift_err,
  output logic [15:0]        rsp_sx_m,   // VQUANT Sx after scale_mul
  output logic signed [7:0]  rsp_sx_e,
  output logic               rsp_zero
);
  localparam logic [1:0] OP_RMS   = 2'd0;
  localparam logic [1:0] OP_QUANT = 2'd1;
  localparam logic [1:0] OP_SOFT  = 2'd2;

  // ------------------------------------------------------------------ stage 1: the leading one
  logic               v1;
  logic [63:0]        x1;
  logic [6:0]         len1;
  logic               zero1;
  logic [1:0]         op1;
  logic [7:0]         sh0_1;
  logic [5:0]         sh_1;
  logic               w8_1, mul_1;
  logic [15:0]        auxm_1;
  logic signed [7:0]  auxe_1;

  always_ff @(posedge clk) begin
    if (rst) v1 <= 1'b0;
    else     v1 <= req_valid;
  end

  always_ff @(posedge clk) begin
    if (req_valid) begin
      x1     <= req_x;
      len1   <= qcore_pkg::bitlen64(req_x);
      zero1  <= (req_x == 64'd0);
      op1    <= req_op;
      sh0_1  <= req_sh0;
      sh_1   <= req_sh;
      w8_1   <= req_w8;
      mul_1  <= req_mul_en;
      auxm_1 <= req_aux_m;
      auxe_1 <= req_aux_e;
    end
  end

  // ------------------------------------------------------------------ stage 2: normalise, address the tables
  // 2e is the even exponent below the leading one, so m_q16 = ss' * 2^(16 - 2e)
  // lands in [2^16, 2^18); hi16 is the top 16 significant bits of req_x.
  logic [5:0]  e_c;
  logic [6:0]  e2;
  logic [17:0] m_q16;
  logic [15:0] hi16;
  logic [8:0]  rsq_idx_a, rsq_idx_b;
  logic [7:0]  rsq_fr_a, rsq_fr_b, rcp_idx, rcp_frac;
  logic        rsq_en, rcp_en_a, rcp_en_b;

  assign e_c       = (len1 == 7'd0) ? 6'd0 : 6'((len1 - 7'd1) >> 1);
  assign e2        = {e_c, 1'b0};
  assign m_q16     = (e2 >= 7'd16) ? 18'(x1 >> (e2 - 7'd16)) : 18'(x1 << (7'd16 - e2));
  assign hi16      = qcore_pkg::norm_hi16(x1, len1);
  assign rsq_idx_a = {1'b0, m_q16[15:8]};   // segment 0: m_q16 below 2^17
  assign rsq_idx_b = {1'b1, m_q16[16:9]};   // segment 1: m_q16 at or above 2^17
  assign rsq_fr_a  = m_q16[7:0];
  assign rsq_fr_b  = m_q16[8:1];
  assign rcp_idx   = hi16[14:7];
  assign rcp_frac  = {hi16[6:0], 1'b0};
  assign rsq_en    = v1 && (op1 == OP_RMS);
  assign rcp_en_a  = v1 && (op1 == OP_QUANT);
  assign rcp_en_b  = v1 && (op1 == OP_SOFT);

  logic              v2;
  logic [1:0]        op2;
  logic              seg2, zero2, w8_2, mul_2;
  logic [7:0]        fr_rsq2, fr_rcp2, sh0_2;
  logic [5:0]        sh_2, e_2;
  logic signed [7:0] e_a2, auxe_2;
  logic [15:0]       hi16_2, auxm_2;

  always_ff @(posedge clk) begin
    if (rst) v2 <= 1'b0;
    else     v2 <= v1;
  end

  always_ff @(posedge clk) begin
    if (v1) begin
      op2     <= op1;
      seg2    <= m_q16[17];
      fr_rsq2 <= m_q16[17] ? rsq_fr_b : rsq_fr_a;
      fr_rcp2 <= rcp_frac;
      hi16_2  <= hi16;
      e_2     <= e_c;
      e_a2    <= $signed({1'b0, len1}) - 8'sd16;
      zero2   <= zero1;
      sh0_2   <= sh0_1;
      sh_2    <= sh_1;
      w8_2    <= w8_1;
      mul_2   <= mul_1;
      auxm_2  <= auxm_1;
      auxe_2  <= auxe_1;
    end
  end

  // ------------------------------------------------------------------ the tables
  logic [15:0] rsq_v_a, rsq_dv_a, rsq_v_b, rsq_dv_b;
  logic [15:0] rcp_v_a, rcp_dv_a, rcp_v_b, rcp_dv_b;
  logic [15:0] rsq_v, rsq_dv, rcp_v, rcp_dv;
  logic [15:0] rsq_y, rcp_y;
  logic        rsq_ov, rcp_ov;

  qcore_lut_rom #(
    .ENTRIES (512),
    .ROM_FILE(ROM_FILE_RSQRT)
  ) u_rsqrt (
    .clk  (clk),
    .en_a (rsq_en),
    .idx_a(rsq_idx_a),
    .v_a  (rsq_v_a),
    .dv_a (rsq_dv_a),
    .en_b (rsq_en),
    .idx_b(rsq_idx_b),
    .v_b  (rsq_v_b),
    .dv_b (rsq_dv_b)
  );

  qcore_lut_rom #(
    .ENTRIES (256),
    .ROM_FILE(ROM_FILE_RECIP)
  ) u_recip (
    .clk  (clk),
    .en_a (rcp_en_a),
    .idx_a(rcp_idx),
    .v_a  (rcp_v_a),
    .dv_a (rcp_dv_a),
    .en_b (rcp_en_b),
    .idx_b(rcp_idx),
    .v_b  (rcp_v_b),
    .dv_b (rcp_dv_b)
  );

  assign rsq_v  = seg2 ? rsq_v_b  : rsq_v_a;
  assign rsq_dv = seg2 ? rsq_dv_b : rsq_dv_a;
  assign rcp_v  = (op2 == OP_SOFT) ? rcp_v_b  : rcp_v_a;
  assign rcp_dv = (op2 == OP_SOFT) ? rcp_dv_b : rcp_dv_a;

  qcore_lut_interp u_rsq_i (
    .clk      (clk),
    .rst      (rst),
    .in_valid (v2),
    .v        (rsq_v),
    .dv       (rsq_dv),
    .frac8    (fr_rsq2),
    .out_valid(rsq_ov),
    .y        (rsq_y)
  );

  qcore_lut_interp u_rcp_i (
    .clk      (clk),
    .rst      (rst),
    .in_valid (v2),
    .v        (rcp_v),
    .dv       (rcp_dv),
    .frac8    (fr_rcp2),
    .out_valid(rcp_ov),
    .y        (rcp_y)
  );

  // ------------------------------------------------------------------ stage 3: the table result
  logic              v3;
  logic [1:0]        op3;
  logic              zero3, w8_3, mul_3;
  logic [7:0]        sh0_3;
  logic [5:0]        sh_3, e_3;
  logic signed [7:0] e_a3, auxe_3;
  logic [15:0]       hi16_3, auxm_3;

  assign v3 = rsq_ov & rcp_ov;

  always_ff @(posedge clk) begin
    if (v2) begin
      op3    <= op2;
      hi16_3 <= hi16_2;
      e_3    <= e_2;
      e_a3   <= e_a2;
      zero3  <= zero2;
      sh0_3  <= sh0_2;
      sh_3   <= sh_2;
      w8_3   <= w8_2;
      mul_3  <= mul_2;
      auxm_3 <= auxm_2;
      auxe_3 <= auxe_2;
    end
  end

  // ------------------------------------------------------------------ stage 4: sfloat forms
  // R normalised into an sfloat (numerics.sfloat_from_int with 16 significant
  // bits) and the VQUANT scale before the optional scale_mul.
  logic [23:0]       rsf;
  logic signed [7:0] sxe_c;

  assign rsf   = qcore_pkg::sfloat_from_int16(rsq_y, -8'sd15);
  // The i8 exponent arithmetic of docs/RTL.md 4: e_a - (w - 1) - FRAC_in.
  assign sxe_c = e_a3 - (w8_3 ? 8'sd7 : 8'sd15) - $signed(sh0_3);

  logic               v4;
  logic [1:0]         op4;
  logic               zero4, w8_4, mul_4;
  logic [23:0]        rsf4;
  logic [15:0]        inv4, sxm4, auxm_4;
  logic signed [7:0]  sxe4, e_a4, auxe_4;
  logic [7:0]         sh0_4;
  logic [5:0]         sh_4, e_4;

  always_ff @(posedge clk) begin
    if (rst) v4 <= 1'b0;
    else     v4 <= v3;
  end

  always_ff @(posedge clk) begin
    if (v3) begin
      op4    <= op3;
      rsf4   <= rsf;
      inv4   <= rcp_y;
      sxm4   <= hi16_3;
      sxe4   <= sxe_c;
      e_a4   <= e_a3;
      e_4    <= e_3;
      sh0_4  <= sh0_3;
      sh_4   <= sh_3;
      w8_4   <= w8_3;
      mul_4  <= mul_3;
      auxm_4 <= auxm_3;
      auxe_4 <= auxe_3;
      zero4  <= zero3;
    end
  end

  // ------------------------------------------------------------------ stage 5: the scale product and the shift
  logic [23:0]        rc, sx_mul, sx_out;
  logic signed [11:0] s1_raw, q_shift, s_shift, sel_shift;
  logic [5:0]         sel_lo, shift_cl;
  logic               shift_neg, shift_big, shift_err;

  assign rc     = qcore_pkg::sfloat_mul(rsf4[15:0], $signed(rsf4[23:16]), auxm_4, auxe_4);
  assign sx_mul = qcore_pkg::sfloat_mul(sxm4, sxe4, auxm_4, auxe_4);
  assign sx_out = (sxm4 == 16'd0) ? 24'd0 : (mul_4 ? sx_mul : {sxe4, sxm4});

  assign s1_raw  = 12'sd0 - ($signed({{4{rc[23]}}, rc[23:16]}) + $signed({4'd0, sh0_4})
                             - $signed({6'd0, sh_4}) - $signed({6'd0, e_4}));
  assign q_shift = 12'sd31 + $signed({{4{e_a4[7]}}, e_a4}) - (w8_4 ? 12'sd8 : 12'sd16);
  assign s_shift = 12'sd7 + $signed({{4{e_a4[7]}}, e_a4});

  always_comb begin
    case (op4)
      OP_RMS:   sel_shift = s1_raw;
      OP_QUANT: sel_shift = q_shift;
      OP_SOFT:  sel_shift = s_shift;
      default:  sel_shift = s_shift;
    endcase
  end

  assign sel_lo    = sel_shift[5:0];
  assign shift_neg = (sel_shift < 12'sd0);
  assign shift_big = (sel_shift > 12'sd63);
  assign shift_cl  = shift_neg ? 6'd0 : (shift_big ? 6'd63 : sel_lo);
  assign shift_err = shift_neg | shift_big;

  always_ff @(posedge clk) begin
    if (rst) rsp_valid <= 1'b0;
    else     rsp_valid <= v4;
  end

  always_ff @(posedge clk) begin
    if (v4) begin
      rsp_m         <= (op4 == OP_RMS) ? rc[15:0] : inv4;
      rsp_shift     <= shift_cl;
      rsp_shift_err <= shift_err & ~zero4;
      rsp_sx_m      <= sx_out[15:0];
      rsp_sx_e      <= $signed(sx_out[23:16]);
      rsp_zero      <= zero4;
    end
  end
endmodule
