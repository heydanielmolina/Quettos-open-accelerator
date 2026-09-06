// Quettos Core shared package: descriptor field extraction, meta / SREG / sfloat
// packing, the QMEM read tags, and the rounding, saturation and clip primitives
// every datapath module uses. Pure functions only; no state, no clocks. Every
// function mirrors the same-named operation of sw/quettos/numerics.py bit for
// bit at the widths stated on it. Modules reference members with explicit
// qcore_pkg:: scoping.
`include "qcore_csr_defs.svh"
package qcore_pkg;

  // Widths used by the functions below. ISA constants are the generated macros.
  localparam int DESC_W = `QCORE_DESC_BYTES * 8;  // 256-bit descriptor
  localparam int SREG_W = 32;                     // one scale register

  // QMEM read tags: which sink a returned beat belongs to.
  localparam logic [3:0] TAG_FETCH  = 4'd0;  // descriptor fetch queue
  localparam logic [3:0] TAG_WEIGHT = 4'd1;  // weight / EMBED tile beats
  localparam logic [3:0] TAG_META   = 4'd2;  // per-channel meta beats
  localparam logic [3:0] TAG_VPU    = 4'd3;  // gamma / RoPE / centering / V-scale beats

  // One-hot sink select of a returned beat: {vpu, meta, weight, fetch}.
  function logic [3:0] rd_route(input logic [3:0] tag);
    rd_route = {tag == TAG_VPU, tag == TAG_META, tag == TAG_WEIGHT, tag == TAG_FETCH};
  endfunction

  // ------------------------------------------------------------------ descriptor fields
  // desc_<field>(d) returns field <field> of the raw 256-bit descriptor at the
  // generated bit position; sh1 is the one signed field.

  function logic [`QCORE_DESC_OPCODE_W-1:0] desc_opcode(input logic [DESC_W-1:0] d);
    desc_opcode = (`QCORE_DESC_OPCODE_W)'(d >> `QCORE_DESC_OPCODE_LSB);
  endfunction

  function logic [`QCORE_DESC_FLAGS_W-1:0] desc_flags(input logic [DESC_W-1:0] d);
    desc_flags = (`QCORE_DESC_FLAGS_W)'(d >> `QCORE_DESC_FLAGS_LSB);
  endfunction

  function logic [`QCORE_DESC_ROW_MASK_W-1:0] desc_row_mask(input logic [DESC_W-1:0] d);
    desc_row_mask = (`QCORE_DESC_ROW_MASK_W)'(d >> `QCORE_DESC_ROW_MASK_LSB);
  endfunction

  function logic desc_accumulate(input logic [DESC_W-1:0] d);
    desc_accumulate = (`QCORE_DESC_ACCUMULATE_W)'(d >> `QCORE_DESC_ACCUMULATE_LSB);
  endfunction

  function logic desc_unit_meta(input logic [DESC_W-1:0] d);
    desc_unit_meta = (`QCORE_DESC_UNIT_META_W)'(d >> `QCORE_DESC_UNIT_META_LSB);
  endfunction

  function logic desc_n_from_pos(input logic [DESC_W-1:0] d);
    desc_n_from_pos = (`QCORE_DESC_N_FROM_POS_W)'(d >> `QCORE_DESC_N_FROM_POS_LSB);
  endfunction

  function logic desc_k_from_pos(input logic [DESC_W-1:0] d);
    desc_k_from_pos = (`QCORE_DESC_K_FROM_POS_W)'(d >> `QCORE_DESC_K_FROM_POS_LSB);
  endfunction

  function logic [`QCORE_DESC_OUT_MODE_W-1:0] desc_out_mode(input logic [DESC_W-1:0] d);
    desc_out_mode = (`QCORE_DESC_OUT_MODE_W)'(d >> `QCORE_DESC_OUT_MODE_LSB);
  endfunction

  function logic desc_len_from_pos(input logic [DESC_W-1:0] d);
    desc_len_from_pos = (`QCORE_DESC_LEN_FROM_POS_W)'(d >> `QCORE_DESC_LEN_FROM_POS_LSB);
  endfunction

  function logic desc_track_absmax(input logic [DESC_W-1:0] d);
    desc_track_absmax = (`QCORE_DESC_TRACK_ABSMAX_W)'(d >> `QCORE_DESC_TRACK_ABSMAX_LSB);
  endfunction

  function logic [`QCORE_DESC_ADDR_A_W-1:0] desc_addr_a(input logic [DESC_W-1:0] d);
    desc_addr_a = (`QCORE_DESC_ADDR_A_W)'(d >> `QCORE_DESC_ADDR_A_LSB);
  endfunction

  function logic [`QCORE_DESC_ADDR_M_W-1:0] desc_addr_m(input logic [DESC_W-1:0] d);
    desc_addr_m = (`QCORE_DESC_ADDR_M_W)'(d >> `QCORE_DESC_ADDR_M_LSB);
  endfunction

  function logic [`QCORE_DESC_N_W-1:0] desc_n(input logic [DESC_W-1:0] d);
    desc_n = (`QCORE_DESC_N_W)'(d >> `QCORE_DESC_N_LSB);
  endfunction

  function logic [`QCORE_DESC_K_W-1:0] desc_k(input logic [DESC_W-1:0] d);
    desc_k = (`QCORE_DESC_K_W)'(d >> `QCORE_DESC_K_LSB);
  endfunction

  function logic [`QCORE_DESC_VS_SRC_W-1:0] desc_vs_src(input logic [DESC_W-1:0] d);
    desc_vs_src = (`QCORE_DESC_VS_SRC_W)'(d >> `QCORE_DESC_VS_SRC_LSB);
  endfunction

  function logic [`QCORE_DESC_VS_DST_W-1:0] desc_vs_dst(input logic [DESC_W-1:0] d);
    desc_vs_dst = (`QCORE_DESC_VS_DST_W)'(d >> `QCORE_DESC_VS_DST_LSB);
  endfunction

  function logic [`QCORE_DESC_VS_AUX_W-1:0] desc_vs_aux(input logic [DESC_W-1:0] d);
    desc_vs_aux = (`QCORE_DESC_VS_AUX_W)'(d >> `QCORE_DESC_VS_AUX_LSB);
  endfunction

  function logic [`QCORE_DESC_SREG_SRC_W-1:0] desc_sreg_src(input logic [DESC_W-1:0] d);
    desc_sreg_src = (`QCORE_DESC_SREG_SRC_W)'(d >> `QCORE_DESC_SREG_SRC_LSB);
  endfunction

  function logic [`QCORE_DESC_SREG_DST_W-1:0] desc_sreg_dst(input logic [DESC_W-1:0] d);
    desc_sreg_dst = (`QCORE_DESC_SREG_DST_W)'(d >> `QCORE_DESC_SREG_DST_LSB);
  endfunction

  function logic [`QCORE_DESC_SRC_ROW_W-1:0] desc_src_row(input logic [DESC_W-1:0] d);
    desc_src_row = (`QCORE_DESC_SRC_ROW_W)'(d >> `QCORE_DESC_SRC_ROW_LSB);
  endfunction

  function logic [`QCORE_DESC_DST_ROW_W-1:0] desc_dst_row(input logic [DESC_W-1:0] d);
    desc_dst_row = (`QCORE_DESC_DST_ROW_W)'(d >> `QCORE_DESC_DST_ROW_LSB);
  endfunction

  function logic [`QCORE_DESC_SH0_W-1:0] desc_sh0(input logic [DESC_W-1:0] d);
    desc_sh0 = (`QCORE_DESC_SH0_W)'(d >> `QCORE_DESC_SH0_LSB);
  endfunction

  function logic signed [`QCORE_DESC_SH1_W-1:0] desc_sh1(input logic [DESC_W-1:0] d);
    desc_sh1 = (`QCORE_DESC_SH1_W)'(d >> `QCORE_DESC_SH1_LSB);
  endfunction

  function logic [`QCORE_DESC_IMM32_W-1:0] desc_imm32(input logic [DESC_W-1:0] d);
    desc_imm32 = (`QCORE_DESC_IMM32_W)'(d >> `QCORE_DESC_IMM32_LSB);
  endfunction

  // ------------------------------------------------------------------ meta records
  // A meta side-stream record is the 56-bit {e[7:0], m[15:0], bias_q[31:0]} of
  // the 8-byte QMEM record (pad byte dropped): consumers slice it directly.

  // ------------------------------------------------------------------ SREG / sfloat words
  // A scale register and an sfloat descriptor immediate share one encoding:
  // m in bits [15:0], e as an i8 in [23:16], [31:24] zero. A tracked absmax
  // occupies the whole word as a u32. The zero scale is m == 0 (word 0 when
  // written by hardware). Readers slice w[15:0] / w[23:16] / w[31:0].

  function logic [SREG_W-1:0] sreg_pack(input logic [15:0] m, input logic signed [7:0] e);
    sreg_pack = {8'd0, e, m};
  endfunction

  // Exponent arithmetic is i8 throughout (two's complement wrap); every
  // exponent a compiled program reaches lies inside the range.

  // sfloat_mul: {e[7:0], m[15:0]} = a * b rounded once (numerics.sfloat_mul).
  // p = m_a * m_b in [2^30, 2^32); p >= 2^31: m = round_shift(p, 16), e = e_a + e_b + 16;
  // otherwise m = round_shift(p, 15), e = e_a + e_b + 15; m == 2^16 folds to {2^15, e + 1};
  // a zero operand gives the zero word.
  function logic [23:0] sfloat_mul(input logic [15:0] ma, input logic signed [7:0] ea,
                                   input logic [15:0] mb, input logic signed [7:0] eb);
    logic signed [33:0] p;
    logic [16:0]        m;
    logic signed [7:0]  e;
    p = $signed({{18{1'b0}}, ma}) * $signed({{18{1'b0}}, mb});
    if (p[31]) begin
      m = 17'(p >> 16) + {16'd0, p[15]};
      e = ea + eb + 8'sd16;
    end else begin
      m = 17'(p >> 15) + {16'd0, p[14]};
      e = ea + eb + 8'sd15;
    end
    if (m[16]) begin
      m = 17'h08000;
      e = e + 8'sd1;
    end
    sfloat_mul = (ma == 16'd0 || mb == 16'd0) ? 24'd0 : {e, m[15:0]};
  endfunction

  // bitlen64: number of significant bits of a non-negative 64-bit value (0 -> 0).
  function logic [6:0] bitlen64(input logic [63:0] x);
    bitlen64 = 7'd0;
    for (int i = 0; i < 64; i++) begin
      if (x[i]) bitlen64 = 7'(i + 1);
    end
  endfunction

  // norm_hi16: the top 16 significant bits of x given len = bitlen64(x):
  // x << (16 - len) for len <= 16, x >> (len - 16) otherwise (truncation, the
  // a_hi / sum_hi / m_q16 normalization of numerics.quant, softmax and rmsnorm).
  function logic [15:0] norm_hi16(input logic [63:0] x, input logic [6:0] len);
    if (len > 7'd16) norm_hi16 = 16'(x >> (len - 7'd16));
    else             norm_hi16 = 16'(x << (7'd16 - len));
  endfunction

  // sfloat_from_int16: {e[7:0], m[15:0]} of a * 2^e0 for a below 2^16 (exact;
  // numerics.sfloat_from_int with at most 16 significant bits): m = a << (16 - bitlen(a)),
  // e = e0 + bitlen(a) - 16; a == 0 gives the zero word.
  function logic [23:0] sfloat_from_int16(input logic [15:0] a, input logic signed [7:0] e0);
    logic [6:0]        len;
    logic [15:0]       m;
    logic signed [7:0] e;
    len = bitlen64({48'd0, a});
    m   = a << (5'd16 - 5'(len));
    e   = e0 + $signed({1'b0, len}) - 8'sd16;
    sfloat_from_int16 = (a == 16'd0) ? 24'd0 : {e, m};
  endfunction

  // ------------------------------------------------------------------ round_shift
  // round_shift(x, s) = (x + 2^(s-1)) >>> s for s in [1, 63], x for s = 0
  // (numerics.round_shift), computed as (x >>> s) + x[s-1], which is the same
  // value and never overflows the input width.

  function logic signed [63:0] round_shift64(input logic signed [63:0] x, input logic [5:0] s);
    logic signed [63:0] sh;
    logic [5:0]         sm1;
    logic               rb;
    sh  = x >>> s;
    sm1 = s - 6'd1;
    rb  = (s != 6'd0) & x[sm1];
    round_shift64 = sh + $signed({{63{1'b0}}, rb});
  endfunction

  function logic signed [56:0] round_shift57(input logic signed [56:0] x, input logic [5:0] s);
    logic signed [56:0] sh;
    logic [63:0]        xb;
    logic [5:0]         sm1;
    logic               rb;
    sh  = x >>> s;
    xb  = {{7{x[56]}}, x};
    sm1 = s - 6'd1;
    rb  = (s != 6'd0) & xb[sm1];
    round_shift57 = sh + $signed({{56{1'b0}}, rb});
  endfunction

  function logic signed [48:0] round_shift49(input logic signed [48:0] x, input logic [5:0] s);
    logic signed [48:0] sh;
    logic [63:0]        xb;
    logic [5:0]         sm1;
    logic               rb;
    sh  = x >>> s;
    xb  = {{15{x[48]}}, x};
    sm1 = s - 6'd1;
    rb  = (s != 6'd0) & xb[sm1];
    round_shift49 = sh + $signed({{48{1'b0}}, rb});
  endfunction

  // ------------------------------------------------------------------ saturation
  // sat<N>_from<W>(x) = {overflow, x saturated to signed N bits} (numerics.sat);
  // the overflow bit is the SAT_* event.

  function logic [40:0] sat40_from57(input logic signed [56:0] x);
    logic ovf;
    ovf = ~((&x[56:39]) | ~(|x[56:39]));
    sat40_from57 = {ovf, ovf ? {x[56], {39{~x[56]}}} : x[39:0]};
  endfunction

  function logic [32:0] sat32_from64(input logic signed [63:0] x);
    logic ovf;
    ovf = ~((&x[63:31]) | ~(|x[63:31]));
    sat32_from64 = {ovf, ovf ? {x[63], {31{~x[63]}}} : x[31:0]};
  endfunction

  function logic [32:0] sat32_from57(input logic signed [56:0] x);
    logic ovf;
    ovf = ~((&x[56:31]) | ~(|x[56:31]));
    sat32_from57 = {ovf, ovf ? {x[56], {31{~x[56]}}} : x[31:0]};
  endfunction

  function logic [32:0] sat32_from49(input logic signed [48:0] x);
    logic ovf;
    ovf = ~((&x[48:31]) | ~(|x[48:31]));
    sat32_from49 = {ovf, ovf ? {x[48], {31{~x[48]}}} : x[31:0]};
  endfunction

  function logic [32:0] sat32_from33(input logic signed [32:0] x);
    logic ovf;
    ovf = x[32] ^ x[31];
    sat32_from33 = {ovf, ovf ? {x[32], {31{~x[32]}}} : x[31:0]};
  endfunction

  // ------------------------------------------------------------------ clips
  // clip<N>_from49(x) = {clipped, x clipped to [-(2^(N-1) - 1), 2^(N-1) - 1]}:
  // the VQUANT and softmax-weight clips, counted apart from saturations.

  function logic [16:0] clip16_from49(input logic signed [48:0] x);
    logic hi;
    logic lo;
    hi = (x > 49'sd32767);
    lo = (x < -49'sd32767);
    clip16_from49 = {hi | lo, hi ? 16'h7FFF : (lo ? 16'h8001 : x[15:0])};
  endfunction

  function logic [8:0] clip8_from49(input logic signed [48:0] x);
    logic hi;
    logic lo;
    hi = (x > 49'sd127);
    lo = (x < -49'sd127);
    clip8_from49 = {hi | lo, hi ? 8'h7F : (lo ? 8'h81 : x[7:0])};
  endfunction

  // abs32: magnitude of an int32 as a u32 (2^31 for the most negative input).
  function logic [31:0] abs32(input logic signed [31:0] x);
    abs32 = x[31] ? (~x + 32'd1) : x;
  endfunction

endpackage
