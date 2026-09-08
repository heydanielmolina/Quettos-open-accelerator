// Quettos Core vector unit: the six V descriptors of docs/ISA.md, one
// descriptor at a time, participating rows ascending and each row finished
// (SREG write included) before the next starts. Every value equals
// sw/quettos/isa_sim.py and sw/quettos/numerics.py bit for bit; the arithmetic
// lives in qcore_vpu_lane (elementwise), qcore_vpu_scalar (the per-pass
// reciprocal, reciprocal square root and softmax normalisation) and
// qcore_lut_rom / qcore_lut_interp (the sigmoid and exp2 curves).
//
//   VRMSNORM  pass 1 absmax(x); pass 2 ss = sum((x >> sh)^2) with sh = max(0,
//             bitlen(absmax) - 15); RMS_SCALE on ss + (eps_c >> 2 sh); pass 3
//             xhat = round_shift49(x * Rc_m, S1) then y = sat32(round_shift49(
//             xhat * gamma, G)), gamma the int16 row streamed from QMEM
//   VQUANT    pass 1 absmax(x), skipped with USE_TRACKED (the row's SREG word);
//             QUANT_SCALE on a_eff = a + (a >> (w-1)) + 1; pass 2
//             q = clip(round_shift49(x * inv, shift)) sign-extended to int32.
//             GROUP repeats the three steps every vs_aux elements and writes
//             SREG[sreg_dst + g]; otherwise one scale to SREG[sreg_dst]. The
//             group index saturates at 256, so a group form with more groups
//             than the register file holds addresses an index above the file
//             -- dropped and counted, as sw/quettos/isa_sim.py drops it --
//             rather than wrapping onto a register the descriptor owns
//   VROPE     one two-trip pass per 64-element head, n >> 6 heads ascending and
//             32 pairs each: a' = sat32(round_shift49(a*cos - b*sin, 14)) and
//             b' = sat32(round_shift49(b*cos + a*sin, 14)) over the pair
//             (x[64h+i], x[64h+32+i]), rewritten in place. The 128-byte table
//             row at addr_a + POS*128 is latched whole once per row
//   VSILUMUL  one pass: sigmoid(g) from the table with the odd symmetry and the
//             |g| >= 16 clamp, silu = round_shift49(g * sig, 15),
//             h = sat32(round_shift64(silu * u, sh_h))
//   VSOFTMAX  four passes and one scalar request. Pass 1 reduces the score
//             maximum m and e_max, the largest V-scale exponent of the row over
//             the records whose mantissa is non-zero; pass 2 writes e_t =
//             round_shift49(exp2(f16) << 8, nn) over the destination and sums
//             it, nn the whole log2 units a score lies below m and f16 the
//             fraction left over; SOFTMAX_NORM turns that total into inv and
//             sh_p; pass 3 reads the parked e_t back and makes
//             p_t = round_shift49(e_t * inv, sh_p) then
//             w_t = clip16(round_shift49(p_t * Sv_m, 24 + e_max - Sv_e));
//             pass 4 zeroes vs_dst[len..n). e_max needs the whole row before
//             the first weight, so the maximum and the weights cannot share a
//             pass and the records are streamed twice
//   VSUBC     one pass: sat32(x - c), c the int32 row streamed from QMEM
//
// A pass walks its elements in chunks of VC and every chunk takes the same
// nine-stage path: operand fetch, table address, interpolation, lane trip A,
// lane trip B, write. VC is VL, except on a pass that streams 8-byte V-scale
// records, where it is min(VL, WB/8) so a chunk's records lie inside one beat,
// as a chunk's gamma or centering elements always do. The passes that need two
// products per element (VRMSNORM pass 3, VSILUMUL, VROPE and the VSOFTMAX
// weight pass) issue a chunk every other cycle, so trip A of one chunk and trip
// B of another never meet on the lane; the others issue every cycle and carry
// their single result through the trip-B slots. Nothing downstream of the issue
// stalls, so a missing operand word or beat costs an issue cycle and nothing
// else.
//
// VSRAM port A reads the vs_src stream, port B the second element stream (the
// VSILUMUL vs_aux row, the VROPE upper half of each head, the VSOFTMAX weight
// pass's own e_t) and writes with element strobes, one strobed word per eight
// elements out of a two-word staging register. VROPE carries a second staging
// register, because a chunk produces the two halves of its pairs at once and
// they lie 32 elements apart; the two words leave on consecutive cycles.
// vsb_sel_dst names the bank the crossbar gives port B: the destination on
// every write but VROPE's, which rewrites its own source, and the destination
// for the whole weight pass, whose reads and writes are both there. A port-B
// read is never followed by a write, so the bank the crossbar selects holds
// while a read returns. QMEM operands stream through a VPU_FIFO_BEATS-deep FIFO
// in bursts of at most min(MAX_BURST, VPU_FIFO_BEATS), re-read for every row
// and, on the two record passes, for every pass; space is reserved before a
// burst is issued so every returned beat is accepted. Words at or past
// VSRAM_WORDS read as zero and are not written; each operand range that leaves
// the VSRAM raises one err_bounds count for the row. done pulses after the last
// write of the last row and busy is high from the issue pulse until then.
//
// Reads run ahead of the writes that follow them by the depth of that path, so
// a destination range that overlaps a source range without matching it element
// for element would read back values the same descriptor has already written.
// docs/ISA.md makes an equal or disjoint range the rule and the compiler
// enforces it. The rule binds within one bank -- row r reads bank src_row + r
// and writes bank dst_row + r, two different memories whenever the bases differ
// -- and the bases reach the crossbar rather than this unit, so the simulation
// check that reports a partial overlap sits in qcore_top, where both the ranges
// and the bases are. VROPE and the VSOFTMAX weight pass are the equal-range
// form of that rule applied to one bank.
//
// File size: this module is over the ~1000-line ceiling of CONTRIBUTING
// section 3, and the reason is that its passes are one timeline. Operand fetch,
// the table lookup, the two lane trips and the write are stages of a single
// shift register, and the issue decision is a combinational function of the
// operand windows, the FIFO and the pass state, while those windows advance on
// that same issue. A boundary drawn at any of those seams carries the stage
// indices across it and makes the issue decision a combinational round trip
// through it, which is the shape the handshake rules of docs/RTL.md section 1
// exist to keep out of the interfaces. The file is kept readable instead by
// section headers that follow the timeline, one per stage group.
`include "qcore_csr_defs.svh"
module qcore_vpu_top #(
  parameter int WB               = 64,
  parameter int B_MAX            = 1,
  parameter int VL               = 4,
  parameter int VSRAM_WORDS      = 4096,
  parameter int VPU_FIFO_BEATS   = 16,
  parameter int MAX_BURST        = 64,
  parameter     ROM_FILE_SIGMOID = "",
  parameter     ROM_FILE_EXP2    = "",
  parameter     ROM_FILE_RSQRT   = "",
  parameter     ROM_FILE_RECIP   = ""
) (
  input  logic                           clk,
  input  logic                           rst,
  // descriptor issue (docs/RTL.md 2.2); the bundle holds from the pulse to done
  input  logic                           cmd_valid_vpu,
  input  logic [7:0]                     cmd_op,
  input  logic                           cmd_vq_w8,
  input  logic                           cmd_vq_use_tracked,
  input  logic                           cmd_vq_group,
  input  logic                           cmd_vq_scale_mul,
  input  logic                           cmd_track_absmax,
  input  logic [31:0]                    cmd_addr_a,
  input  logic [23:0]                    cmd_n,
  input  logic [23:0]                    cmd_len,
  input  logic [31:0]                    cmd_pos,
  input  logic [15:0]                    cmd_vs_src,
  input  logic [15:0]                    cmd_vs_dst,
  input  logic [15:0]                    cmd_vs_aux,
  input  logic [7:0]                     cmd_sreg_dst,
  input  logic [7:0]                     cmd_sh0,
  input  logic signed [7:0]              cmd_sh1,
  input  logic [31:0]                    cmd_imm32,
  input  logic [15:0]                    cmd_sqrt_m,
  input  logic signed [7:0]              cmd_sqrt_e,
  input  logic [B_MAX-1:0]               cmd_rows,
  input  logic [B_MAX*32-1:0]            cmd_sreg_u32,
  // QMEM read requests through qcore_mem_arb, tag = TAG_VPU
  output logic                           v_req_valid,
  input  logic                           v_req_ready,
  output logic [31:0]                    v_req_addr,
  output logic [7:0]                     v_req_len,
  output logic [3:0]                     v_req_tag,
  input  logic                           rdv_valid,
  input  logic [WB*8-1:0]                rd_data,
  input  logic                           rd_data_last,
  // VSRAM: port A of bank src_row + cur_row, port B of src_row or dst_row
  output logic [3:0]                     cur_row,
  output logic                           vsa_en,
  output logic [$clog2(VSRAM_WORDS)-1:0] vsa_addr,
  input  logic [255:0]                   vsa_rdata,
  output logic                           vsb_sel_dst,
  output logic                           vsb_en,
  output logic [7:0]                     vsb_we,
  output logic [$clog2(VSRAM_WORDS)-1:0] vsb_addr,
  output logic [255:0]                   vsb_wdata,
  input  logic [255:0]                   vsb_rdata,
  // SREG write into bank dst_row + sreg_wr_row
  output logic                           sreg_wr_en,
  output logic [3:0]                     sreg_wr_row,
  output logic [7:0]                     sreg_wr_idx,
  output logic [31:0]                    sreg_wr_data,
  // event strobes (docs/RTL.md 2.8)
  output logic [7:0]                     sat_inc,
  output logic [7:0]                     err_shift_inc,
  output logic [3:0]                     err_bounds_inc,
  output logic                           done,
  output logic                           busy
);
  localparam int DW     = WB * 8;
  localparam int AW     = $clog2(VSRAM_WORDS);
  localparam int NE     = `QCORE_VSRAM_WORD_ELEMS;   // 8 int32 elements per word
  localparam int EL     = 25;                        // element index arithmetic
  localparam int WIX    = EL - 3;                    // VSRAM word index
  localparam int NROM   = (VL + 1) / 2;              // table ROMs, two ports each
  localparam int EPBG   = WB / 2;                    // int16 operands per beat
  localparam int EPBC   = WB / 4;                    // int32 operands per beat
  localparam int EPBR   = WB / 8;                    // 8-byte meta records per beat
  localparam int VCR    = (VL < EPBR) ? VL : EPBR;   // chunk of a record pass
  localparam int LGG    = $clog2(WB / 2);
  localparam int LGC    = $clog2(WB / 4);
  localparam int LGR    = $clog2(WB / 8);
  localparam int BOW    = LGG + 1;                   // beat element offset
  localparam int FAW    = $clog2(VPU_FIFO_BEATS);
  localparam int BURST  = (MAX_BURST < VPU_FIFO_BEATS) ? MAX_BURST : VPU_FIFO_BEATS;
  localparam int SIGE   = 512;                       // sigmoid table entries
  localparam int EXPE   = 256;                       // exp2 table entries
  localparam int VH     = (VL + 1) / 2;              // lanes after the first max level
  localparam int RBEATS = (1024 + DW - 1) / DW;      // beats in one 128-byte RoPE row
  localparam int RROW   = RBEATS * DW;               // bits the latched row occupies
  localparam int HALF   = 32;                        // elements between the halves of a head

  localparam logic [7:0] OP_VRMSNORM = `QCORE_OP_VRMSNORM;
  localparam logic [7:0] OP_VQUANT   = `QCORE_OP_VQUANT;
  localparam logic [7:0] OP_VROPE    = `QCORE_OP_VROPE;
  localparam logic [7:0] OP_VSILUMUL = `QCORE_OP_VSILUMUL;
  localparam logic [7:0] OP_VSOFTMAX = `QCORE_OP_VSOFTMAX;
  localparam logic [7:0] OP_VSUBC    = `QCORE_OP_VSUBC;

  // qcore_vpu_lane ops, in the order docs/RTL.md 3.13 lists them
  localparam logic [2:0] L_MUL32  = 3'd0;
  localparam logic [2:0] L_MUL16  = 3'd1;
  localparam logic [2:0] L_ROPE_A = 3'd2;
  localparam logic [2:0] L_ROPE_B = 3'd3;
  localparam logic [2:0] L_SUB    = 3'd4;
  localparam logic [2:0] L_PASS   = 3'd5;

  // qcore_vpu_scalar ops
  localparam logic [1:0] SC_RMS   = 2'd0;
  localparam logic [1:0] SC_QUANT = 2'd1;
  localparam logic [1:0] SC_SOFT  = 2'd2;

  // pass kinds
  localparam logic [2:0] P_ABSMAX = 3'd0;   // x through the lane, absmax of x
  localparam logic [2:0] P_SUMSQ  = 3'd1;   // p64 = (x >> sh)^2, accumulated
  localparam logic [2:0] P_OUT    = 3'd2;   // the op's single output pass
  localparam logic [2:0] P_ROPE   = 3'd3;   // one head: both halves of 32 pairs
  localparam logic [2:0] P_SMAX   = 3'd4;   // softmax 1: the score maximum and e_max
  localparam logic [2:0] P_SEXP   = 3'd5;   // softmax 2: e_t into the destination
  localparam logic [2:0] P_SW     = 3'd6;   // softmax 3: p_t then the weight
  localparam logic [2:0] P_SFILL  = 3'd7;   // softmax 4: zeros past the length

  // The softmax constants of numerics.softmax that are not descriptor fields:
  // 25 log2 units is the distance beyond which a weight rounds to zero, the
  // exponential carries 8 fraction bits beyond Q1.15, and the weight shift is
  // 16 + 8 + e_max - Sv_e capped at 40.
  localparam int SM_CLAMP = 25;
  localparam int SM_BASE  = 24;
  localparam int SM_SHMAX = 40;

  typedef enum logic [3:0] {
    S_IDLE   = 4'd0,
    S_ROW    = 4'd1,
    S_TABLE  = 4'd2,
    S_GRP    = 4'd3,
    S_PASS   = 4'd4,
    S_DRAIN  = 4'd5,
    S_SCALAR = 4'd6,
    S_SREG   = 4'd7,
    S_NEXT   = 4'd8,
    S_END    = 4'd9
  } state_t;

  state_t state;

  // ================================================================== descriptor
  logic                    d_rms, d_quant, d_rope, d_silu, d_soft, d_subc, d_known;
  logic                    d_w8, d_tracked, d_group, d_track;
  logic [31:0]             d_addr_a, d_eps, cur_u32, d_dmax, d_qmask;
  logic [EL-1:0]           d_n, d_nw, d_len, d_src, d_dst, d_aux, d_grp, d_beats;
  logic [7:0]              d_sreg_dst, d_sh0;
  logic [5:0]              d_g, d_frs, d_frs16;
  logic                    d_g_err, d_smul, d_mem, d_e16;
  logic [15:0]             d_aux_m;
  logic signed [7:0]       d_aux_e;
  logic [BOW:0]            d_epb;
  logic [1:0]              d_err_row;
  logic [B_MAX:0]          rows_left;
  logic [(B_MAX+1)*32-1:0] u32_left;
  logic [3:0]              row_i;

  logic [EL-1:0] cmd_n_e, cmd_len_e, cmd_src_e, cmd_dst_e, cmd_aux_e;
  logic          cmd_rd_err, cmd_rdl_err, cmd_wr_err, cmd_aux_err, cmd_g_neg, cmd_g_big;
  logic [1:0]    cmd_err_row;

  assign cmd_n_e     = {1'b0, cmd_n};
  assign cmd_len_e   = {1'b0, cmd_len};
  assign cmd_src_e   = {9'd0, cmd_vs_src};
  assign cmd_dst_e   = {9'd0, cmd_vs_dst};
  assign cmd_aux_e   = {9'd0, cmd_vs_aux};
  assign cmd_rd_err  = ({1'b0, cmd_src_e} + {1'b0, cmd_n_e}) > (EL+1)'(VSRAM_WORDS * NE);
  assign cmd_rdl_err = ({1'b0, cmd_src_e} + {1'b0, cmd_len_e}) > (EL+1)'(VSRAM_WORDS * NE);
  assign cmd_wr_err  = ({1'b0, cmd_dst_e} + {1'b0, cmd_n_e}) > (EL+1)'(VSRAM_WORDS * NE);
  assign cmd_aux_err = ({1'b0, cmd_aux_e} + {1'b0, cmd_n_e}) > (EL+1)'(VSRAM_WORDS * NE);
  assign cmd_g_neg   = cmd_sh1 < 8'sd0;
  assign cmd_g_big   = cmd_sh1 > 8'sd63;

  // VROPE reads and writes one range and counts it twice, as isa_sim does;
  // VSOFTMAX reads vs_src + len and writes vs_dst + n.
  always_comb begin
    if (cmd_op == OP_VROPE) begin
      cmd_err_row = {cmd_rd_err, 1'b0};
    end else if (cmd_op == OP_VSOFTMAX) begin
      cmd_err_row = {1'b0, cmd_rdl_err} + {1'b0, cmd_wr_err};
    end else if (cmd_op == OP_VSILUMUL) begin
      cmd_err_row = {1'b0, cmd_rd_err} + {1'b0, cmd_wr_err} + {1'b0, cmd_aux_err};
    end else begin
      cmd_err_row = {1'b0, cmd_rd_err} + {1'b0, cmd_wr_err};
    end
  end

  // ================================================================== group and pass
  logic [EL-1:0] grp_start, grp_rem, pb_a, pb_b, pb_d, p_len, p_ord;
  logic [8:0]    grp_idx;
  logic [2:0]    p_kind, p_vc;
  logic          pass_init, p_two, p_writes, p_mem, p_amax, p_sumsq, p_bstream;
  logic          p_rec, p_sig, p_rope, p_smax, p_sexp, p_sw, p_areads, p_dstb;
  logic [5:0]    sh_ss, p_sha, p_shb;
  logic [6:0]    amax_len;
  logic [2:0]    p_opa, p_opb;

  assign p_rope    = (p_kind == P_ROPE);
  assign p_smax    = (p_kind == P_SMAX);
  assign p_sexp    = (p_kind == P_SEXP);
  assign p_sw      = (p_kind == P_SW);
  assign p_rec     = p_smax || p_sw;
  assign p_sig     = (p_kind == P_OUT) && d_silu;
  assign p_two     = ((p_kind == P_OUT) && (d_rms || d_silu)) || p_rope || p_sw;
  assign p_writes  = (p_kind == P_OUT) || p_rope || p_sexp || p_sw || (p_kind == P_SFILL);
  assign p_mem     = ((p_kind == P_OUT) && d_mem) || p_rec;
  assign p_bstream = p_sig || p_rope || p_sw;
  assign p_areads  = !p_sw && (p_kind != P_SFILL);
  assign p_dstb    = p_sw;
  assign p_amax    = (p_kind == P_ABSMAX)
                     || ((p_kind == P_OUT) && d_track && (d_rms || d_silu));
  assign p_sumsq   = (p_kind == P_SUMSQ);
  assign p_vc      = p_rec ? 3'(VCR) : 3'(VL);

  always_comb begin
    case (p_kind)
      P_ABSMAX: p_opa = L_PASS;
      P_SUMSQ:  p_opa = L_MUL32;
      P_ROPE:   p_opa = L_ROPE_A;
      P_SMAX:   p_opa = L_PASS;
      P_SEXP:   p_opa = L_MUL16;
      P_SW:     p_opa = L_MUL16;
      P_SFILL:  p_opa = L_PASS;
      default:  p_opa = d_subc ? L_SUB : L_MUL16;
    endcase
  end

  always_comb begin
    if (p_rope)     p_opb = L_ROPE_B;
    else if (p_sw)  p_opb = L_MUL16;
    else if (d_rms) p_opb = L_ROPE_A;
    else            p_opb = L_MUL32;
  end

  // scalar response, held for the output pass
  logic [15:0]       r_m, r_sxm;
  logic [5:0]        r_sh;
  logic              r_sherr;
  logic signed [7:0] r_sxe;   // the VQUANT scale exponent

  always_comb begin
    case (p_kind)
      P_OUT:   p_sha = d_silu ? 6'd15 : ((d_rms || d_quant) ? r_sh : 6'd0);
      P_ROPE:  p_sha = 6'd14;
      P_SW:    p_sha = r_sh;
      default: p_sha = 6'd0;
    endcase
  end

  assign p_shb = p_rope ? 6'd14 : d_g;

  // accumulators: the absmax or signed maximum of a reduction pass, the 56-bit
  // sum of squares or of the exponentials, and the row's largest V-scale exponent
  logic [31:0]       acc_amax;
  logic [55:0]       acc_ss;
  logic signed [7:0] acc_emax;
  logic              emax_seen;

  assign amax_len = qcore_pkg::bitlen64({32'd0, acc_amax});

  // ================================================================== VSRAM windows
  // Two identical two-word windows: A on port A (vs_src), B on port B (the
  // second element stream). cnt words are held and res marks a fetch whose word
  // arrives this cycle; a chunk waits until the window holds the one or two
  // words its VC elements straddle.
  logic [2:0]     n_chunk;   // elements this chunk actually carries, 1 .. VC
  logic [255:0]   wa0, wa1, wb0, wb1, wa0_n, wa1_n, wb0_n, wb1_n, data_a, data_b;
  logic [1:0]     cnt_a, cnt_b, slot_a, slot_b;
  logic           res_a, res_b, res_z_a, res_z_b, push_a, push_b, pop_a, pop_b;
  logic           fetch_a, fetch_b, oob_a, oob_b, rdy_a, rdy_b, need2_a, need2_b;
  logic [2:0]     off_a, off_b;
  logic [3:0]     off_a_end, off_b_end;
  logic [WIX-1:0] nxt_a, nxt_b, end_a, end_b;

  assign off_a_end = {1'b0, off_a} + {1'b0, n_chunk};
  assign off_b_end = {1'b0, off_b} + {1'b0, n_chunk};
  assign need2_a   = off_a_end > 4'(NE);
  assign need2_b   = off_b_end > 4'(NE);
  assign rdy_a     = need2_a ? (cnt_a == 2'd2) : (cnt_a != 2'd0);
  assign rdy_b     = need2_b ? (cnt_b == 2'd2) : (cnt_b != 2'd0);
  assign oob_a     = nxt_a >= WIX'(VSRAM_WORDS);
  assign oob_b     = nxt_b >= WIX'(VSRAM_WORDS);
  assign push_a    = res_a;
  assign push_b    = res_b;
  assign data_a    = res_z_a ? 256'd0 : vsa_rdata;
  assign data_b    = res_z_b ? 256'd0 : vsb_rdata;

  // ================================================================== QMEM FIFO
  logic [DW-1:0]  fmem [0:VPU_FIFO_BEATS-1];
  logic [DW-1:0]  f_odata;
  logic [FAW-1:0] f_wptr, f_rptr;
  logic [FAW:0]   f_cnt;
  logic           f_ovalid, f_rd, f_pop, m_can, f_take;
  logic [15:0]    m_out, m_free;
  logic [EL-1:0]  m_left;
  logic [31:0]    m_addr;
  logic [7:0]     m_len;
  logic [BOW-1:0] mb_off;
  logic [BOW:0]   mb_end;

  assign m_free = 16'(VPU_FIFO_BEATS) - {{(15-FAW){1'b0}}, f_cnt} - {15'd0, f_ovalid} - m_out;
  assign m_len  = (m_left > EL'(BURST)) ? 8'(BURST) : 8'(m_left);
  assign m_can  = (m_left != {EL{1'b0}}) && (m_free >= {8'd0, m_len});
  assign f_rd   = (f_cnt != {(FAW+1){1'b0}}) && (!f_ovalid || f_pop);
  assign mb_end = {1'b0, mb_off} + {{(BOW-2){1'b0}}, p_vc};
  assign v_req_tag = qcore_pkg::TAG_VPU;

  // ================================================================== chunk issue
  logic          issue, phase, chunks_left, last_chunk;
  logic [VL-1:0] issue_mask;
  logic [EL-1:0] ord_end;

  assign ord_end     = p_ord + {{(EL-3){1'b0}}, p_vc};
  assign n_chunk     = ((p_len - p_ord) >= {{(EL-3){1'b0}}, p_vc}) ? p_vc : 3'(p_len - p_ord);
  assign chunks_left = (state == S_PASS) && !pass_init && (p_ord < p_len);
  assign last_chunk  = ord_end >= p_len;
  assign issue       = chunks_left && (!p_two || !phase) && (!p_areads || rdy_a)
                       && (!p_bstream || rdy_b) && (!p_mem || f_ovalid);

  always_comb begin
    for (int j = 0; j < VL; j++) begin
      issue_mask[j] = (3'(j) < p_vc) && ((p_ord + EL'(j)) < p_len);
    end
  end

  // ================================================================== stage pipeline
  // Stage i of a chunk is i + 1 cycles after its issue cycle; slot i of each
  // shift register below holds stage i of the chunk that occupies it.
  logic [8:0]         sv;
  logic [9*VL-1:0]    smask;
  logic [7:0]         slast;
  logic [6*32*VL-1:0] sopa;    // stages 0..5: x, (x >> sh), or a rotation's a half
  logic [6*32*VL-1:0] sopb;    // stages 0..5: u, gamma, a V-scale record, or the b half
  logic [6*6*VL-1:0]  ssh;     // stages 0..5: a per-element shift
  logic [6*5-1:0]     sidx;    // stages 0..5: the chunk's first RoPE pair index
  logic [8*VL-1:0]    sfrc;    // stage 0: the table fraction
  logic [2*VL-1:0]    ssgn;    // stages 0..1: the sigmoid argument was negative
  logic [2*VL-1:0]    sinr;    // stages 0..1: the sigmoid index is inside the table
  logic [3*32*VL-1:0] sya;     // stages 5..7: the trip-A result (slot i is stage 5 + i)
  logic [3*VL-1:0]    ssat;    // stages 5..7: its saturation

  logic [32*VL-1:0] issue_opa, issue_opb;
  logic [8*VL-1:0]  issue_frc;
  logic [6*VL-1:0]  issue_sh;
  logic [VL-1:0]    issue_sgn, issue_inr;
  logic [9*VL-1:0]  issue_idx;

  // ================================================================== lanes and tables
  logic             lane_iv, sig_en, exp_en, tab_iv;
  logic [2:0]       lane_op;
  logic [6*VL-1:0]  lane_sh;
  logic [32*VL-1:0] lane_a, lane_b, lane_y;
  logic [64*VL-1:0] lane_p64;
  logic [16*VL-1:0] lane_c, lane_c2, sig_v, sig_dv, exp_v, exp_dv;
  logic [16*VL-1:0] tab_v, tab_dv, tab_y, sig_val, sig_q;
  logic [VL-1:0]    lane_sat, lane_ov, sig_ov;

  // ================================================================== write staging
  // Two staging registers: the primary one every writing pass fills, and the
  // second half of a VROPE chunk, whose destination is HALF elements further on.
  logic [511:0]     wbuf, wbuf_p, wbuf_n, wbuf2, wbuf2_p, wbuf2_n;
  logic [15:0]      wstrb, wstrb_p, wstrb_n, wstrb2, wstrb2_p, wstrb2_n;
  logic [2:0]       wr_off;
  logic [WIX-1:0]   wr_word, wr_word2;
  logic [3:0]       wr_off_end;
  logic             emit, flush_req, flush_now, wr_ok, wr_ok2;
  logic             wp_v0, wp_v1;
  logic [AW-1:0]    wp_a0, wp_a1;
  logic [255:0]     wp_d0, wp_d1;
  logic [7:0]       wp_s0, wp_s1;
  logic [32*VL-1:0] y_fin, y_fin2, y_out, y_out2;
  logic [VL-1:0]    sat_fin, sat_fin2;
  logic [2:0]       n_act;     // elements of the stage-7 chunk, 1 .. VC

  assign wr_off_end = {1'b0, wr_off} + {1'b0, n_act};
  assign emit       = sv[7] && p_writes && (wr_off_end >= 4'(NE));
  assign wr_word2   = wr_word + WIX'(HALF / NE);
  assign wr_ok      = wr_word  < WIX'(VSRAM_WORDS);
  assign wr_ok2     = wr_word2 < WIX'(VSRAM_WORDS);
  assign flush_now  = flush_req && !emit && !wp_v0 && !wp_v1;

  // ================================================================== sequencer
  logic [EL-1:0]     grp_this;
  logic              last_group, pipe_idle, sc_valid, sc_rsp, sc_sherr, sc_zero;
  logic [63:0]       sc_x;
  logic [32:0]       a_eff;
  logic [15:0]       sc_m, sc_sxm;
  logic [31:0]       eps_sh;
  logic [55:0]       ss_eps;
  logic [5:0]        sc_shift;
  logic [1:0]        sc_op;
  logic signed [7:0] sc_sxe;
  logic [8:0]        sreg_idx_sum;
  logic [3:0]        rope_beat;
  logic [RROW-1:0]   rope_row;

  assign grp_this     = (grp_rem < d_grp) ? grp_rem : d_grp;
  assign last_group   = grp_rem <= d_grp;
  assign pipe_idle    = (sv == 9'd0) && !flush_req && !wp_v0 && !wp_v1;
  assign a_eff        = {1'b0, acc_amax}
                        + (d_w8 ? {8'd0, acc_amax[31:7]} : {16'd0, acc_amax[31:15]}) + 33'd1;
  assign eps_sh       = d_eps >> {sh_ss, 1'b0};
  assign ss_eps       = acc_ss + {24'd0, eps_sh};
  assign sc_op        = d_rms ? SC_RMS : (d_soft ? SC_SOFT : SC_QUANT);

  always_comb begin
    if (d_rms)       sc_x = {8'd0, ss_eps};
    else if (d_soft) sc_x = {8'd0, acc_ss};
    else             sc_x = (acc_amax == 32'd0) ? 64'd0 : {31'd0, a_eff};
  end

  assign sreg_idx_sum = {1'b0, d_sreg_dst} + grp_idx;

  always_ff @(posedge clk) begin
    if (rst) begin
      state          <= S_IDLE;
      rows_left      <= {(B_MAX+1){1'b0}};
      row_i          <= 4'd0;
      done           <= 1'b0;
      sc_valid       <= 1'b0;
      cur_row        <= 4'd0;
      err_bounds_inc <= 4'd0;
      sreg_wr_en     <= 1'b0;
      pass_init      <= 1'b0;
      rope_beat      <= 4'd0;
    end else begin
      done           <= 1'b0;
      sc_valid       <= 1'b0;
      err_bounds_inc <= 4'd0;
      sreg_wr_en     <= 1'b0;
      case (state)
        S_IDLE: begin
          if (cmd_valid_vpu) begin
            d_rms      <= cmd_op == OP_VRMSNORM;
            d_quant    <= cmd_op == OP_VQUANT;
            d_rope     <= cmd_op == OP_VROPE;
            d_silu     <= cmd_op == OP_VSILUMUL;
            d_soft     <= cmd_op == OP_VSOFTMAX;
            d_subc     <= cmd_op == OP_VSUBC;
            d_known    <= (cmd_op == OP_VRMSNORM) || (cmd_op == OP_VQUANT)
                          || (cmd_op == OP_VROPE) || (cmd_op == OP_VSILUMUL)
                          || (cmd_op == OP_VSOFTMAX) || (cmd_op == OP_VSUBC);
            d_w8       <= cmd_vq_w8;
            d_tracked  <= cmd_vq_use_tracked && !cmd_vq_group;
            d_group    <= cmd_vq_group;
            d_smul     <= cmd_vq_scale_mul;
            d_track    <= cmd_track_absmax;
            // the RoPE table row is the one operand whose address carries POS
            d_addr_a   <= (cmd_op == OP_VROPE) ? (cmd_addr_a + (cmd_pos << 7))
                                               : cmd_addr_a;
            d_n        <= cmd_n_e;
            d_nw       <= (cmd_op == OP_VROPE) ? {cmd_n_e[EL-1:6], 6'd0} : cmd_n_e;
            d_len      <= cmd_len_e;
            d_src      <= cmd_src_e;
            d_dst      <= cmd_dst_e;
            d_aux      <= cmd_aux_e;
            d_grp      <= (cmd_op == OP_VROPE) ? EL'(2 * HALF)
                          : (((cmd_op == OP_VQUANT) && cmd_vq_group && (cmd_vs_aux != 16'd0))
                             ? cmd_aux_e : cmd_n_e);
            d_sreg_dst <= cmd_sreg_dst;
            d_sh0      <= cmd_sh0;
            // qcore_seq_dispatch faults a VSOFTMAX whose class leaves
            // [QCORE_SOFTMAX_FRAC_MIN, QCORE_SOFTMAX_FRAC_MAX] (3.5), so the
            // three softmax constants below are exact over every class this
            // block is issued; the width guards keep them defined for a block
            // bench that drives one the sequencer would have refused.
            d_frs      <= (cmd_sh0 > 8'd63) ? 6'd63 : cmd_sh0[5:0];
            d_frs16    <= (cmd_sh0 > 8'd63) ? 6'd47
                          : ((cmd_sh0 < 8'd16) ? 6'd0 : 6'(cmd_sh0 - 8'd16));
            // 25 << FRAC_S saturated: above FRAC_S = 27 the clamp cannot bind,
            // since the largest score difference is a 33-bit signed subtraction
            d_dmax     <= (cmd_sh0 > 8'd27) ? {32{1'b1}}
                                            : (32'(SM_CLAMP) << cmd_sh0[4:0]);
            // the FRAC_S - 16 bits below the Q16 fraction: their OR is the
            // rounding bit of both f16 and n
            d_qmask    <= (cmd_sh0 < 8'd17) ? 32'd0
                          : ((32'd1 << ((cmd_sh0 > 8'd47) ? 5'd31 : 5'(cmd_sh0 - 8'd16)))
                             - 32'd1);
            d_g        <= cmd_g_neg ? 6'd0 : (cmd_g_big ? 6'd63 : cmd_sh1[5:0]);
            d_g_err    <= cmd_g_neg || cmd_g_big;
            d_eps      <= cmd_imm32;
            d_aux_m    <= (cmd_op == OP_VRMSNORM) ? cmd_sqrt_m : cmd_imm32[15:0];
            d_aux_e    <= (cmd_op == OP_VRMSNORM) ? cmd_sqrt_e : $signed(cmd_imm32[23:16]);
            d_mem      <= (cmd_op == OP_VRMSNORM) || (cmd_op == OP_VSUBC);
            d_e16      <= cmd_op == OP_VRMSNORM;
            d_epb      <= (cmd_op == OP_VRMSNORM) ? (BOW+1)'(EPBG)
                          : ((cmd_op == OP_VSOFTMAX) ? (BOW+1)'(EPBR) : (BOW+1)'(EPBC));
            if (cmd_op == OP_VRMSNORM) begin
              d_beats <= EL'((cmd_n_e + EL'(EPBG - 1)) >> LGG);
            end else if (cmd_op == OP_VSOFTMAX) begin
              d_beats <= EL'((cmd_len_e + EL'(EPBR - 1)) >> LGR);
            end else if (cmd_op == OP_VROPE) begin
              d_beats <= EL'(RBEATS);
            end else begin
              d_beats <= EL'((cmd_n_e + EL'(EPBC - 1)) >> LGC);
            end
            d_err_row  <= cmd_err_row;
            rows_left  <= {1'b0, cmd_rows};
            u32_left   <= {32'd0, cmd_sreg_u32};
            row_i      <= 4'd0;
            state      <= S_ROW;
          end
        end

        S_ROW: begin
          if (row_i == 4'(B_MAX)) begin
            state <= S_END;
          end else begin
            row_i     <= row_i + 4'd1;
            rows_left <= {1'b0, rows_left[B_MAX:1]};
            u32_left  <= {32'd0, u32_left[(B_MAX+1)*32-1:32]};
            if (rows_left[0] && (d_nw != {EL{1'b0}}) && d_known) begin
              cur_row        <= row_i;
              cur_u32        <= u32_left[31:0];
              err_bounds_inc <= {2'd0, d_err_row};
              grp_start      <= {EL{1'b0}};
              grp_rem        <= d_nw;
              grp_idx        <= 9'd0;
              rope_beat      <= 4'd0;
              if (d_rope) state <= S_TABLE;
              else        state <= S_GRP;
            end
          end
        end

        // The 128-byte RoPE row, latched whole before the row's first chunk: a
        // pair needs cos_i and sin_i from bytes 2i and 64 + 2i, which lie in
        // different beats at every WB below 128.
        S_TABLE: begin
          if (f_ovalid) begin
            if (rope_beat == 4'(RBEATS - 1)) begin
              rope_beat <= 4'd0;
              state     <= S_GRP;
            end else begin
              rope_beat <= rope_beat + 4'd1;
            end
          end
        end

        S_GRP: begin
          pb_a  <= d_soft ? d_src : (d_src + grp_start);
          pb_b  <= d_rope ? (d_src + grp_start + EL'(HALF)) : (d_aux + grp_start);
          pb_d  <= d_rope ? (d_src + grp_start) : (d_dst + grp_start);
          p_len <= d_rope ? EL'(HALF) : (d_soft ? d_len : grp_this);
          if (d_quant && d_tracked) begin
            sc_valid <= 1'b1;
            state    <= S_SCALAR;
          end else begin
            if (d_rope)       p_kind <= P_ROPE;
            else if (d_soft)  p_kind <= P_SMAX;
            else              p_kind <= (d_rms || d_quant) ? P_ABSMAX : P_OUT;
            pass_init <= 1'b1;
            state     <= S_PASS;
          end
        end

        S_PASS: begin
          pass_init <= 1'b0;
          if (!pass_init && (p_ord >= p_len)) state <= S_DRAIN;
        end

        S_DRAIN: begin
          if (pipe_idle) begin
            if (p_kind == P_ABSMAX) begin
              if (d_rms) begin
                p_kind    <= P_SUMSQ;
                sh_ss     <= (amax_len > 7'd15) ? 6'(amax_len - 7'd15) : 6'd0;
                pass_init <= 1'b1;
                state     <= S_PASS;
              end else begin
                sc_valid <= 1'b1;
                state    <= S_SCALAR;
              end
            end else if (p_kind == P_SUMSQ) begin
              sc_valid <= 1'b1;
              state    <= S_SCALAR;
            end else if (p_smax) begin
              // a row whose V scales are all the canonical zero is the zero
              // vector and the zero scale: the exponential, the normalisation
              // and the weight pass have nothing to compute
              pass_init <= 1'b1;
              state     <= S_PASS;
              if (emax_seen) begin
                p_kind <= P_SEXP;
                pb_a   <= d_src;
                pb_d   <= d_dst;
                p_len  <= d_len;
              end else begin
                p_kind <= P_SFILL;
                pb_d   <= d_dst;
                p_len  <= d_n;
              end
            end else if (p_sexp) begin
              sc_valid <= 1'b1;
              state    <= S_SCALAR;
            end else if (p_sw) begin
              if (d_len < d_n) begin
                p_kind    <= P_SFILL;
                pb_d      <= d_dst + d_len;
                p_len     <= d_n - d_len;
                pass_init <= 1'b1;
                state     <= S_PASS;
              end else begin
                state <= S_SREG;
              end
            end else begin
              state <= S_SREG;
            end
          end
        end

        S_SCALAR: begin
          if (sc_rsp) begin
            r_m       <= sc_zero ? 16'd0 : sc_m;
            r_sh      <= sc_zero ? 6'd0 : sc_shift;
            r_sherr   <= sc_sherr;
            r_sxm     <= sc_sxm;
            r_sxe     <= sc_sxe;
            p_kind    <= d_soft ? P_SW : P_OUT;
            if (d_soft) begin
              pb_b  <= d_dst;
              pb_d  <= d_dst;
              p_len <= d_len;
            end
            pass_init <= 1'b1;
            state     <= S_PASS;
          end
        end

        S_SREG: begin
          sreg_wr_row <= cur_row;
          if (d_quant) begin
            sreg_wr_en   <= 1'b1;
            sreg_wr_idx  <= sreg_idx_sum[8] ? 8'hFF : sreg_idx_sum[7:0];
            sreg_wr_data <= {8'd0, r_sxe, r_sxm};
          end else if (d_soft) begin
            // the descriptor's output scale, written whether or not
            // track_absmax is set: 2^(1 + e_max) as {2^15, e_max - 14}
            sreg_wr_en   <= 1'b1;
            sreg_wr_idx  <= d_sreg_dst;
            sreg_wr_data <= emax_seen ? qcore_pkg::sreg_pack(16'h8000, acc_emax - 8'sd14)
                                      : 32'd0;
          end else if (d_track && !d_rope) begin
            sreg_wr_en   <= 1'b1;
            sreg_wr_idx  <= d_sreg_dst;
            sreg_wr_data <= acc_amax;
          end
          state <= S_NEXT;
        end

        S_NEXT: begin
          if (((d_quant && d_group) || d_rope) && !last_group) begin
            grp_start <= grp_start + d_grp;
            grp_rem   <= grp_rem - d_grp;
            grp_idx   <= (grp_idx == 9'd256) ? grp_idx : (grp_idx + 9'd1);
            state     <= S_GRP;
          end else begin
            state <= S_ROW;
          end
        end

        S_END: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  assign busy = (state != S_IDLE) || done;

  // ================================================================== operand windows
  logic [511:0]     win_a, win_b;
  logic [32*VL-1:0] sel_a, sel_b, sel_c, sel_r;
  logic [16*VL-1:0] sel_g;
  logic [7:0]       sh_idx, sh_frc;

  assign win_a  = {wa1, wa0};
  assign win_b  = {wb1, wb0};
  assign sel_a  = win_a[{1'b0, off_a, 5'd0} +: (32*VL)];
  assign sel_b  = win_b[{1'b0, off_b, 5'd0} +: (32*VL)];
  assign sh_idx = d_sh0 - 8'd5;
  assign sh_frc = d_sh0 - 8'd13;

  // The streamed beat holds EPB operands and a chunk starts at a multiple of
  // its own width, so every slice below is constant and inside the beat. A
  // record chunk is VCR wide, which is what keeps its records in one beat.
  always_comb begin
    sel_g = {(16*VL){1'b0}};
    for (int b = 0; b < EPBG; b += VL) begin
      if (mb_off == BOW'(b)) sel_g = f_odata[b*16 +: (16*VL)];
    end
    sel_c = {(32*VL){1'b0}};
    for (int b = 0; b < EPBC; b += VL) begin
      if (mb_off == BOW'(b)) sel_c = f_odata[b*32 +: (32*VL)];
    end
    sel_r = {(32*VL){1'b0}};
    for (int b = 0; b < EPBR; b += VCR) begin
      if (mb_off == BOW'(b)) begin
        for (int j = 0; j < VCR; j++) begin
          sel_r[j*32 +: 32] = {8'd0, f_odata[(b + j)*64 + 32 +: 24]};
        end
      end
    end
  end

  // ---- the softmax at the issue: numerics.softmax's d, n and f16, and the
  // per-token weight shift of its third pass
  logic [32:0]       sm_sub;
  logic [31:0]       sm_d, sm_dhi;
  logic [15:0]       sm_hi16, sm_f16;
  logic              sm_bnz, sm_lnz;
  logic [32*VL-1:0]  xw, xsh, axw, idx32;
  logic [5:0]        sm_nn, sw_sh;
  logic signed [9:0] sw_raw;

  always_comb begin
    for (int j = 0; j < VL; j++) begin
      xw[j*32 +: 32]    = sel_a[j*32 +: 32];
      xsh[j*32 +: 32]   = 32'($signed(sel_a[j*32 +: 32]) >>> sh_ss);
      axw[j*32 +: 32]   = qcore_pkg::abs32($signed(sel_a[j*32 +: 32]));
      idx32[j*32 +: 32] = axw[j*32 +: 32] >> sh_idx;

      // softmax pass 2: the distance below the maximum, clamped, split into the
      // integer part n and the Q16 fraction f16 of 2^-d = 2^-n * 2^(f16/2^16).
      // Writing d as dhi * 2^FRAC_S + A * 2^(FRAC_S-16) + B, the reference's
      // g = (n << FRAC_S) - d gives f16 = 2^16 - A - [B != 0] in 16 bits, which
      // is the same value with the three chained shifts of that expression
      // replaced by two the element takes in parallel and a mask the descriptor
      // fixed. n is dhi plus one whenever any of those low bits is set.
      sm_sub = $signed({acc_amax[31], acc_amax}) - $signed({xw[j*32+31], xw[j*32 +: 32]});
      if (sm_sub[32])                       sm_d = 32'd0;
      else if (sm_sub[31:0] > d_dmax)       sm_d = d_dmax;
      else                                  sm_d = sm_sub[31:0];
      sm_dhi  = sm_d >> d_frs;
      sm_hi16 = 16'(sm_d >> d_frs16);
      sm_bnz  = (sm_d & d_qmask) != 32'd0;
      sm_lnz  = (sm_hi16 != 16'd0) || sm_bnz;
      sm_f16  = 16'd0 - sm_hi16 - {15'd0, sm_bnz};
      sm_nn   = (sm_dhi > 32'd62) ? 6'd63 : (6'(sm_dhi) + {5'd0, sm_lnz});

      // softmax pass 3: the per-token weight shift, 24 + e_max - Sv_e capped at
      // 40. A record with a zero mantissa gives a zero product at any shift.
      sw_raw = $signed(10'(SM_BASE)) + $signed({{2{acc_emax[7]}}, acc_emax})
               - $signed({{2{sel_r[j*32 + 23]}}, sel_r[j*32 + 16 +: 8]});
      if (sw_raw < 10'sd0)                sw_sh = 6'd0;
      else if (sw_raw > 10'sd40)          sw_sh = 6'(SM_SHMAX);
      else                                sw_sh = sw_raw[5:0];

      if (p_sexp)                     issue_opa[j*32 +: 32] = 32'd256;
      else if (p_sw)                  issue_opa[j*32 +: 32] = sel_b[j*32 +: 32];
      else if (p_kind == P_SFILL)     issue_opa[j*32 +: 32] = 32'd0;
      else if (p_sumsq)               issue_opa[j*32 +: 32] = xsh[j*32 +: 32];
      else                            issue_opa[j*32 +: 32] = xw[j*32 +: 32];

      if (p_sumsq)                    issue_opb[j*32 +: 32] = xsh[j*32 +: 32];
      else if (p_rec)                 issue_opb[j*32 +: 32] = sel_r[j*32 +: 32];
      else if (p_sig || p_rope)       issue_opb[j*32 +: 32] = sel_b[j*32 +: 32];
      else if (p_mem && d_e16)        issue_opb[j*32 +: 32] = {16'd0, sel_g[j*16 +: 16]};
      else if (p_mem)                 issue_opb[j*32 +: 32] = sel_c[j*32 +: 32];
      else                            issue_opb[j*32 +: 32] = 32'd0;

      issue_frc[j*8 +: 8] = p_sexp ? sm_f16[7:0] : 8'((axw[j*32 +: 32] >> sh_frc));
      issue_sgn[j]        = !p_sexp && xw[j*32 + 31];
      issue_inr[j]        = p_sexp || (idx32[j*32 + 9 +: 23] == 23'd0);
      issue_idx[j*9 +: 9] = p_sexp ? {1'b0, sm_f16[15:8]} : idx32[j*32 +: 9];
      issue_sh[j*6 +: 6]  = p_sexp ? sm_nn : (p_sw ? sw_sh : 6'd0);
    end
  end

  // window A: port A reads of bank src_row + cur_row
  assign fetch_a = (state == S_PASS) && !pass_init && p_areads
                   && (({1'b0, cnt_a} + 3'(res_a)) < 3'd2) && (nxt_a <= end_a);
  assign vsa_en  = fetch_a && !oob_a;
  assign vsa_addr = nxt_a[AW-1:0];
  assign pop_a   = issue && p_areads && (off_a_end >= 4'(NE));
  assign slot_a  = cnt_a - {1'b0, pop_a};

  always_comb begin
    wa0_n = pop_a ? wa1 : wa0;
    wa1_n = wa1;
    if (push_a) begin
      if (slot_a == 2'd0) wa0_n = data_a;
      else                wa1_n = data_a;
    end
  end

  // window B: port B reads, never in the cycle before a write, so the bank the
  // crossbar selects holds while the read returns
  assign fetch_b = (state == S_PASS) && !pass_init && p_bstream
                   && (({1'b0, cnt_b} + 3'(res_b)) < 3'd2) && (nxt_b <= end_b)
                   && (oob_b || (!wp_v0 && !wp_v1 && !emit && !flush_req));
  assign vsb_en  = fetch_b && !oob_b;
  assign pop_b   = issue && p_bstream && (off_b_end >= 4'(NE));
  assign slot_b  = cnt_b - {1'b0, pop_b};

  always_comb begin
    wb0_n = pop_b ? wb1 : wb0;
    wb1_n = wb1;
    if (push_b) begin
      if (slot_b == 2'd0) wb0_n = data_b;
      else                wb1_n = data_b;
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      cnt_a   <= 2'd0;
      cnt_b   <= 2'd0;
      res_a   <= 1'b0;
      res_b   <= 1'b0;
      res_z_a <= 1'b0;
      res_z_b <= 1'b0;
      off_a   <= 3'd0;
      off_b   <= 3'd0;
      p_ord   <= {EL{1'b0}};
      phase   <= 1'b0;
    end else if (pass_init) begin
      cnt_a   <= 2'd0;
      cnt_b   <= 2'd0;
      res_a   <= 1'b0;
      res_b   <= 1'b0;
      res_z_a <= 1'b0;
      res_z_b <= 1'b0;
      off_a   <= pb_a[2:0];
      off_b   <= pb_b[2:0];
      nxt_a   <= pb_a[EL-1:3];
      nxt_b   <= pb_b[EL-1:3];
      end_a   <= WIX'((pb_a + p_len - EL'(1)) >> 3);
      end_b   <= WIX'((pb_b + p_len - EL'(1)) >> 3);
      p_ord   <= {EL{1'b0}};
      phase   <= 1'b0;
    end else begin
      phase   <= (state == S_PASS) ? !phase : 1'b0;
      res_a   <= fetch_a;
      res_b   <= fetch_b;
      res_z_a <= fetch_a && oob_a;
      res_z_b <= fetch_b && oob_b;
      if (fetch_a) nxt_a <= nxt_a + WIX'(1);
      if (fetch_b) nxt_b <= nxt_b + WIX'(1);
      cnt_a <= cnt_a + {1'b0, push_a} - {1'b0, pop_a};
      cnt_b <= cnt_b + {1'b0, push_b} - {1'b0, pop_b};
      if (issue) begin
        p_ord <= ord_end;
        off_a <= off_a_end[2:0];
        off_b <= off_b_end[2:0];
      end
    end
  end

  always_ff @(posedge clk) begin
    wa0 <= wa0_n;
    wa1 <= wa1_n;
    wb0 <= wb0_n;
    wb1 <= wb1_n;
  end

  // ================================================================== QMEM stream
  assign f_take = issue && p_mem && ((mb_end >= d_epb) || last_chunk);
  assign f_pop  = f_take || ((state == S_TABLE) && f_ovalid);

  always_ff @(posedge clk) begin
    if (rdv_valid) fmem[f_wptr] <= rd_data;
  end

  always_ff @(posedge clk) begin
    if (f_rd) f_odata <= fmem[f_rptr];
  end

  always_ff @(posedge clk) begin
    if ((state == S_TABLE) && f_ovalid) begin
      for (int b = 0; b < RBEATS; b++) begin
        if (rope_beat == 4'(b)) rope_row[b*DW +: DW] <= f_odata;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      f_wptr   <= {FAW{1'b0}};
      f_rptr   <= {FAW{1'b0}};
      f_cnt    <= {(FAW+1){1'b0}};
      f_ovalid <= 1'b0;
      m_out    <= 16'd0;
      m_left   <= {EL{1'b0}};
      v_req_valid <= 1'b0;
      mb_off   <= {BOW{1'b0}};
    end else begin
      if (rdv_valid) f_wptr <= (f_wptr == FAW'(VPU_FIFO_BEATS-1)) ? {FAW{1'b0}}
                                                                  : f_wptr + FAW'(1);
      if (f_rd)      f_rptr <= (f_rptr == FAW'(VPU_FIFO_BEATS-1)) ? {FAW{1'b0}}
                                                                  : f_rptr + FAW'(1);
      f_cnt <= f_cnt + {{FAW{1'b0}}, rdv_valid} - {{FAW{1'b0}}, f_rd};
      if (f_rd)       f_ovalid <= 1'b1;
      else if (f_pop) f_ovalid <= 1'b0;
      m_out <= m_out - {15'd0, rdv_valid}
               + ((v_req_valid && v_req_ready) ? {8'd0, v_req_len} : 16'd0);

      // one burst request at a time, held until the arbiter takes it
      if (v_req_valid) begin
        if (v_req_ready) begin
          v_req_valid <= 1'b0;
          m_addr      <= m_addr + (32'(v_req_len) << $clog2(WB));
          m_left      <= m_left - {{(EL-8){1'b0}}, v_req_len};
        end
      end else if ((state != S_IDLE) && m_can) begin
        v_req_valid <= 1'b1;
        v_req_addr  <= m_addr;
        v_req_len   <= m_len;
      end

      // the stream restarts for every row, and for every pass that streams the
      // V-scale records, since passes 1 and 3 each read the whole run
      if ((state == S_ROW) && rows_left[0] && (d_nw != {EL{1'b0}}) && d_known
          && (d_mem || d_rope)) begin
        m_left <= d_beats;
        m_addr <= d_addr_a;
      end else if (pass_init && p_rec) begin
        m_left <= d_beats;
        m_addr <= d_addr_a;
      end
      if (pass_init)   mb_off <= {BOW{1'b0}};
      else if (f_take) mb_off <= (mb_end >= d_epb) ? {BOW{1'b0}} : mb_end[BOW-1:0];
      else if (issue && p_mem) mb_off <= mb_end[BOW-1:0];
    end
  end

  // ================================================================== stage registers
  always_ff @(posedge clk) begin
    if (rst) begin
      sv    <= 9'd0;
      slast <= 8'd0;
    end else begin
      sv    <= {sv[7:0], issue};
      slast <= {slast[6:0], issue && last_chunk};
    end
  end

  always_ff @(posedge clk) begin
    smask <= {smask[8*VL-1:0], issue_mask};
    sopa  <= {sopa[5*32*VL-1:0], issue_opa};
    sopb  <= {sopb[5*32*VL-1:0], issue_opb};
    ssh   <= {ssh[5*6*VL-1:0], issue_sh};
    sidx  <= {sidx[5*5-1:0], p_ord[4:0]};
    sfrc  <= issue_frc;
    ssgn  <= {ssgn[VL-1:0], issue_sgn};
    sinr  <= {sinr[VL-1:0], issue_inr};
    sig_q <= sig_val;
    sya   <= {sya[2*32*VL-1:0], lane_y};
    ssat  <= {ssat[2*VL-1:0], lane_sat};
  end

  // ================================================================== the tables
  // Two ROM banks, two lanes each; one interpolator per lane, fed the sigmoid
  // pair on a VSILUMUL and the exp2 pair on a softmax exponential pass, since
  // no descriptor reads both curves.
  assign sig_en = issue && p_sig;
  assign exp_en = issue && p_sexp;
  assign tab_iv = sv[0] && (p_sig || p_sexp);

  genvar gr;
  generate
    for (gr = 0; gr < NROM; gr++) begin : g_rom
      localparam int LA = 2 * gr;
      localparam int LB = (2 * gr + 1 < VL) ? (2 * gr + 1) : (VL - 1);
      qcore_lut_rom #(
        .ENTRIES (SIGE),
        .ROM_FILE(ROM_FILE_SIGMOID)
      ) u_sig (
        .clk  (clk),
        .en_a (sig_en),
        .idx_a(issue_idx[LA*9 +: 9]),
        .v_a  (sig_v[LA*16 +: 16]),
        .dv_a (sig_dv[LA*16 +: 16]),
        .en_b (sig_en),
        .idx_b(issue_idx[LB*9 +: 9]),
        .v_b  (sig_v[LB*16 +: 16]),
        .dv_b (sig_dv[LB*16 +: 16])
      );
      qcore_lut_rom #(
        .ENTRIES (EXPE),
        .ROM_FILE(ROM_FILE_EXP2)
      ) u_exp (
        .clk  (clk),
        .en_a (exp_en),
        .idx_a(issue_idx[LA*9 +: 8]),
        .v_a  (exp_v[LA*16 +: 16]),
        .dv_a (exp_dv[LA*16 +: 16]),
        .en_b (exp_en),
        .idx_b(issue_idx[LB*9 +: 8]),
        .v_b  (exp_v[LB*16 +: 16]),
        .dv_b (exp_dv[LB*16 +: 16])
      );
    end
  endgenerate

  always_comb begin
    for (int j = 0; j < VL; j++) begin
      tab_v[j*16 +: 16]  = p_sexp ? exp_v[j*16 +: 16]  : sig_v[j*16 +: 16];
      tab_dv[j*16 +: 16] = p_sexp ? exp_dv[j*16 +: 16] : sig_dv[j*16 +: 16];
    end
  end

  genvar gi;
  generate
    for (gi = 0; gi < VL; gi++) begin : g_interp
      qcore_lut_interp u_interp (
        .clk      (clk),
        .rst      (rst),
        .in_valid (tab_iv),
        .v        (tab_v[gi*16 +: 16]),
        .dv       (tab_dv[gi*16 +: 16]),
        .frac8    (sfrc[gi*8 +: 8]),
        .out_valid(sig_ov[gi]),
        .y        (tab_y[gi*16 +: 16])
      );
    end
  endgenerate

  // sigmoid(-x) = 1 - sigmoid(x); |x| past the table saturates to 1.0. The
  // exponential takes the interpolated value as it stands.
  always_comb begin
    for (int j = 0; j < VL; j++) begin
      logic [15:0] pos;
      pos = sinr[VL + j] ? tab_y[j*16 +: 16] : 16'h8000;
      sig_val[j*16 +: 16] = ssgn[VL + j] ? (16'h8000 - pos) : pos;
    end
  end

  // ================================================================== the lanes
  logic [4:0] rot2, rot5;

  assign lane_iv = sv[2] || (p_two && sv[5]);
  assign lane_op = sv[2] ? p_opa : p_opb;
  assign rot2    = sidx[2*5 +: 5];
  assign rot5    = sidx[5*5 +: 5];

  always_comb begin
    for (int j = 0; j < VL; j++) begin
      logic [4:0] i2;
      logic [4:0] i5;
      i2 = rot2 + 5'(j);
      i5 = rot5 + 5'(j);
      if (sv[2]) begin
        lane_a[j*32 +: 32] = sopa[2*32*VL + j*32 +: 32];
        lane_b[j*32 +: 32] = sopb[2*32*VL + j*32 +: 32];
        if (p_rope)                  lane_c[j*16 +: 16] = rope_row[{1'b0, i2, 4'd0} +: 16];
        else if (d_silu || p_sexp)   lane_c[j*16 +: 16] = sig_q[j*16 +: 16];
        else                         lane_c[j*16 +: 16] = r_m;
        lane_c2[j*16 +: 16] = p_rope ? rope_row[{1'b1, i2, 4'd0} +: 16] : 16'd0;
        lane_sh[j*6 +: 6]   = p_sexp ? ssh[2*6*VL + j*6 +: 6] : p_sha;
      end else begin
        lane_a[j*32 +: 32] = p_rope ? sopa[5*32*VL + j*32 +: 32] : sya[j*32 +: 32];
        lane_b[j*32 +: 32] = (d_rms || p_sw) ? 32'd0 : sopb[5*32*VL + j*32 +: 32];
        lane_c[j*16 +: 16] = p_rope ? rope_row[{1'b0, i5, 4'd0} +: 16]
                                    : sopb[5*32*VL + j*32 +: 16];
        lane_c2[j*16 +: 16] = p_rope ? rope_row[{1'b1, i5, 4'd0} +: 16] : 16'd0;
        lane_sh[j*6 +: 6]   = p_sw ? ssh[5*6*VL + j*6 +: 6] : p_shb;
      end
    end
  end

  genvar gl;
  generate
    for (gl = 0; gl < VL; gl++) begin : g_lane
      qcore_vpu_lane u_lane (
        .clk      (clk),
        .rst      (rst),
        .in_valid (lane_iv),
        .op       (lane_op),
        .a        (lane_a[gl*32 +: 32]),
        .b        (lane_b[gl*32 +: 32]),
        .c        (lane_c[gl*16 +: 16]),
        .c2       (lane_c2[gl*16 +: 16]),
        .sh       (lane_sh[gl*6 +: 6]),
        .out_valid(lane_ov[gl]),
        .y        (lane_y[gl*32 +: 32]),
        .p64      (lane_p64[gl*64 +: 64]),
        .sat      (lane_sat[gl])
      );
    end
  endgenerate

  // ================================================================== the scalar unit
  qcore_vpu_scalar #(
    .ROM_FILE_RSQRT(ROM_FILE_RSQRT),
    .ROM_FILE_RECIP(ROM_FILE_RECIP)
  ) u_scalar (
    .clk          (clk),
    .rst          (rst),
    .req_valid    (sc_valid),
    .req_op       (sc_op),
    .req_x        (sc_x),
    .req_sh0      (d_sh0),
    .req_sh       (sh_ss),
    .req_w8       (d_w8),
    .req_mul_en   (d_smul),
    .req_aux_m    (d_aux_m),
    .req_aux_e    (d_aux_e),
    .rsp_valid    (sc_rsp),
    .rsp_m        (sc_m),
    .rsp_shift    (sc_shift),
    .rsp_shift_err(sc_sherr),
    .rsp_sx_m     (sc_sxm),
    .rsp_sx_e     (sc_sxe),
    .rsp_zero     (sc_zero)
  );

  // ================================================================== the result
  logic [16:0]      clip16;
  logic [8:0]       clip8;
  logic [VL-1:0]    clip_hit;
  logic [32*VL-1:0] clip_ext, red;
  logic [32*VH-1:0] pair;
  logic [63:0]      sq_sum;
  logic [31:0]      amax_chunk, amax_q;
  logic             use_clip, use8;
  logic [3:0]       n_sat;
  logic signed [7:0] emax_chunk;
  logic              emax_hit;

  // A two-product pass carries its intermediate -- VRMSNORM's xhat, VSILUMUL's
  // silu, the softmax's p_t -- through the lane, so that value is an int32 and
  // saturates there. Both trips are reported: slot 2 of ssat holds trip A's
  // saturation of the chunk whose trip B lands this cycle, so an intermediate
  // that leaves int32 raises SAT_VPU instead of quietly changing the result.
  // VROPE is the exception: its two trips are two results, not an intermediate
  // and a result, so each is reported on its own.
  always_comb begin
    for (int j = 0; j < VL; j++) begin
      y_fin[j*32 +: 32]  = (p_two && !p_rope) ? lane_y[j*32 +: 32]
                                              : sya[2*32*VL + j*32 +: 32];
      y_fin2[j*32 +: 32] = lane_y[j*32 +: 32];
      sat_fin[j]         = (p_two && !p_rope) ? (lane_sat[j] || ssat[2*VL + j])
                                              : ssat[2*VL + j];
      sat_fin2[j]        = lane_sat[j];
    end
  end

  assign use_clip = (d_quant && (p_kind == P_OUT)) || p_sw;
  assign use8     = d_quant && d_w8;

  always_comb begin
    clip_ext = {(32*VL){1'b0}};
    clip_hit = {VL{1'b0}};
    for (int j = 0; j < VL; j++) begin
      clip16 = qcore_pkg::clip16_from49($signed({{17{y_fin[j*32+31]}}, y_fin[j*32 +: 32]}));
      clip8  = qcore_pkg::clip8_from49($signed({{17{y_fin[j*32+31]}}, y_fin[j*32 +: 32]}));
      clip_ext[j*32 +: 32] = use8 ? {{24{clip8[7]}}, clip8[7:0]}
                                  : {{16{clip16[15]}}, clip16[15:0]};
      clip_hit[j] = use8 ? clip8[8] : clip16[16];
    end
  end

  always_comb begin
    for (int j = 0; j < VL; j++) begin
      y_out[j*32 +: 32]  = use_clip ? clip_ext[j*32 +: 32] : y_fin[j*32 +: 32];
      y_out2[j*32 +: 32] = y_fin2[j*32 +: 32];
    end
  end

  // The chunk's contribution to the accumulators and the event counts. The
  // reduction is a balanced tree, so the stage-7 path is one absolute value and
  // ceil(log2(VL)) + 1 comparisons rather than a chain of VL of them; a softmax
  // maximum pass compares signed and starts from the most negative int32.
  always_comb begin
    sq_sum = 64'd0;
    n_act  = 3'd0;
    n_sat  = 4'd0;
    for (int j = 0; j < VL; j++) begin
      if (smask[4*VL + j]) begin
        sq_sum = sq_sum + (p_sexp ? {32'd0, lane_y[j*32 +: 32]} : lane_p64[j*64 +: 64]);
      end
      if (smask[7*VL + j]) begin
        red[j*32 +: 32] = p_smax ? y_fin[j*32 +: 32]
                                 : qcore_pkg::abs32($signed(y_fin[j*32 +: 32]));
      end else begin
        red[j*32 +: 32] = p_smax ? 32'h8000_0000 : 32'd0;
      end
      if (smask[7*VL + j]) begin
        n_act = n_act + 3'd1;
        if (sat_fin[j])               n_sat = n_sat + 4'd1;
        if (p_rope && sat_fin2[j])    n_sat = n_sat + 4'd1;
      end
    end
  end

  always_comb begin
    for (int j = 0; j < VH; j++) begin
      logic [31:0] lo;
      logic [31:0] hi;
      lo = red[(2*j)*32 +: 32];
      hi = (2*j + 1 < VL) ? red[(2*j + 1)*32 +: 32] : (p_smax ? 32'h8000_0000 : 32'd0);
      pair[j*32 +: 32] = (p_smax ? ($signed(lo) > $signed(hi)) : (lo > hi)) ? lo : hi;
    end
    amax_chunk = pair[0 +: 32];
    for (int j = 1; j < VH; j++) begin
      if (p_smax ? ($signed(pair[j*32 +: 32]) > $signed(amax_chunk))
                 : (pair[j*32 +: 32] > amax_chunk)) begin
        amax_chunk = pair[j*32 +: 32];
      end
    end
  end

  // e_max is reduced in the pass that reduces the score maximum, over the
  // records whose mantissa is non-zero; a chunk's records reach stage 0 with it.
  always_comb begin
    emax_chunk = 8'sd0;
    emax_hit   = 1'b0;
    for (int j = 0; j < VL; j++) begin
      if (smask[j] && (sopb[j*32 +: 16] != 16'd0)) begin
        if (!emax_hit || ($signed(sopb[j*32 + 16 +: 8]) > emax_chunk)) begin
          emax_chunk = $signed(sopb[j*32 + 16 +: 8]);
        end
        emax_hit = 1'b1;
      end
    end
  end

  always_ff @(posedge clk) begin
    if ((state == S_GRP) && d_quant && d_tracked) begin
      acc_amax <= cur_u32;                    // USE_TRACKED: the row's SREG word
    end else if (pass_init) begin
      if (p_amax)  acc_amax <= 32'd0;
      if (p_smax)  acc_amax <= 32'h8000_0000;
      if (p_sumsq || p_sexp) acc_ss <= 56'd0;
    end else begin
      if (sv[4] && (p_sumsq || p_sexp)) acc_ss <= acc_ss + sq_sum[55:0];
      if (sv[8] && (p_amax || p_smax)
          && (p_smax ? ($signed(amax_q) > $signed(acc_amax)) : (amax_q > acc_amax))) begin
        acc_amax <= amax_q;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (pass_init && p_smax) begin
      acc_emax  <= 8'sd0;
      emax_seen <= 1'b0;
    end else if (sv[0] && p_smax && emax_hit) begin
      if (!emax_seen || (emax_chunk > acc_emax)) acc_emax <= emax_chunk;
      emax_seen <= 1'b1;
    end
  end

  // The chunk's maximum is registered before it meets the running one, so the
  // stage-7 path is the absolute value and the VL-way reduction alone.
  always_ff @(posedge clk) begin
    if (sv[7]) amax_q <= amax_chunk;
  end

  // ================================================================== events
  logic [1:0] err_per;

  assign err_per = d_rms ? ({1'b0, r_sherr} + {1'b0, d_g_err})
                         : (d_silu ? {1'b0, d_g_err} : 2'd0);

  always_ff @(posedge clk) begin
    if (rst) begin
      sat_inc       <= 8'd0;
      err_shift_inc <= 8'd0;
    end else begin
      sat_inc       <= (sv[7] && p_writes && !d_quant) ? {4'd0, n_sat} : 8'd0;
      if (sv[7] && (p_kind == P_OUT)) begin
        case (err_per)
          2'd1:    err_shift_inc <= {5'd0, n_act};
          2'd2:    err_shift_inc <= {4'd0, n_act, 1'b0};
          default: err_shift_inc <= 8'd0;
        endcase
      end else begin
        err_shift_inc <= 8'd0;
      end
    end
  end

  // ================================================================== the writes
  always_comb begin
    wbuf_p   = wbuf;
    wstrb_p  = wstrb;
    wbuf2_p  = wbuf2;
    wstrb2_p = wstrb2;
    if (sv[7] && p_writes) begin
      for (int e = 0; e < 2*NE; e++) begin
        for (int j = 0; j < VL; j++) begin
          if (smask[7*VL + j] && (({1'b0, wr_off} + 4'(j)) == 4'(e))) begin
            wbuf_p[e*32 +: 32] = y_out[j*32 +: 32];
            wstrb_p[e]         = 1'b1;
            if (p_rope) begin
              wbuf2_p[e*32 +: 32] = y_out2[j*32 +: 32];
              wstrb2_p[e]         = 1'b1;
            end
          end
        end
      end
    end
    wbuf_n   = emit ? {256'd0, wbuf_p[511:256]} : wbuf_p;
    wstrb_n  = emit ? {8'd0, wstrb_p[15:8]} : wstrb_p;
    wbuf2_n  = emit ? {256'd0, wbuf2_p[511:256]} : wbuf2_p;
    wstrb2_n = emit ? {8'd0, wstrb2_p[15:8]} : wstrb2_p;
  end

  // The pending write is a two-slot queue: slot 0 drives port B this cycle and
  // slot 1, which only VROPE fills, follows it on the next. Emits are at least
  // NE/VC issue slots apart, so a push never overtakes a slot still waiting.
  always_ff @(posedge clk) begin
    if (rst) begin
      wstrb     <= 16'd0;
      wstrb2    <= 16'd0;
      wp_v0     <= 1'b0;
      wp_v1     <= 1'b0;
      flush_req <= 1'b0;
      wr_off    <= 3'd0;
    end else if (pass_init) begin
      wstrb     <= 16'd0;
      wstrb2    <= 16'd0;
      wp_v0     <= 1'b0;
      wp_v1     <= 1'b0;
      flush_req <= 1'b0;
      wr_off    <= pb_d[2:0];
      wr_word   <= pb_d[EL-1:3];
    end else begin
      wbuf   <= wbuf_n;
      wstrb  <= wstrb_n;
      wbuf2  <= wbuf2_n;
      wstrb2 <= wstrb2_n;
      if (sv[7] && p_writes) wr_off <= wr_off_end[2:0];
      if (emit) wr_word <= wr_word + WIX'(1);
      if (sv[7] && slast[7] && p_writes) flush_req <= 1'b1;

      wp_v0 <= wp_v1;
      wp_a0 <= wp_a1;
      wp_d0 <= wp_d1;
      wp_s0 <= wp_s1;
      wp_v1 <= 1'b0;
      if (emit) begin
        wp_v0 <= wr_ok;
        wp_a0 <= wr_word[AW-1:0];
        wp_d0 <= wbuf_p[255:0];
        wp_s0 <= wstrb_p[7:0];
        wp_v1 <= p_rope && wr_ok2;
        wp_a1 <= wr_word2[AW-1:0];
        wp_d1 <= wbuf2_p[255:0];
        wp_s1 <= wstrb2_p[7:0];
      end else if (flush_now) begin
        flush_req <= 1'b0;
        wp_v0 <= wr_ok && (wstrb[7:0] != 8'd0);
        wp_a0 <= wr_word[AW-1:0];
        wp_d0 <= wbuf[255:0];
        wp_s0 <= wstrb[7:0];
        wp_v1 <= p_rope && wr_ok2 && (wstrb2[7:0] != 8'd0);
        wp_a1 <= wr_word2[AW-1:0];
        wp_d1 <= wbuf2[255:0];
        wp_s1 <= wstrb2[7:0];
      end
    end
  end

  // VROPE rewrites its own source, so its writes address bank src_row + r; the
  // softmax weight pass reads and writes the destination bank throughout.
  assign vsb_sel_dst = p_dstb ? 1'b1 : (d_rope ? 1'b0 : wp_v0);
  assign vsb_we      = wp_v0 ? wp_s0 : 8'd0;
  assign vsb_wdata   = wp_d0;
  assign vsb_addr    = wp_v0 ? wp_a0 : nxt_b[AW-1:0];

`ifndef SYNTHESIS
  // A VQUANT or softmax clip is an expected value and not an event
  // (docs/RTL.md 3.12), so no counter port carries it; this simulation-only
  // count of the clips of one descriptor is what a bench compares against
  // numerics.Stats.clip.
  logic [31:0] clip_count;
  logic [2:0]  clip_now;

  always_comb begin
    clip_now = 3'd0;
    for (int j = 0; j < VL; j++) begin
      if (smask[7*VL + j] && clip_hit[j]) clip_now = clip_now + 3'd1;
    end
  end

  always_ff @(posedge clk) begin
    if (rst || cmd_valid_vpu)              clip_count <= 32'd0;
    else if (sv[7] && p_writes && use_clip) clip_count <= clip_count + {29'd0, clip_now};
  end

  // The pipeline's own timing rules, and the lane and interpolator latencies
  // the stage shift register assumes.
  always @(posedge clk) begin
    if (!rst) begin
      if (sv[2] && p_two && sv[5]) begin
        $error("qcore_vpu_top: lane trips A and B collided");
      end
      if (lane_ov != {VL{1'b1}} && lane_ov != {VL{1'b0}}) begin
        $error("qcore_vpu_top: the lanes disagree on out_valid");
      end
      if (lane_ov[0] != sv[4] && !p_two) begin
        $error("qcore_vpu_top: lane out_valid %0d does not match stage 4", lane_ov[0]);
      end
      if ((sig_ov != {VL{1'b0}}) && !sv[1]) begin
        $error("qcore_vpu_top: an interpolation landed outside stage 1");
      end
      if (rdv_valid && rd_data_last && (m_out == 16'd0)) begin
        $error("qcore_vpu_top: a last beat returned with no request outstanding");
      end
      if (rdv_valid && (f_cnt == (FAW+1)'(VPU_FIFO_BEATS))) begin
        $error("qcore_vpu_top: operand FIFO overflow");
      end
      if (cmd_valid_vpu && (state != S_IDLE)) begin
        $error("qcore_vpu_top: a descriptor was issued while the unit was busy");
      end
      if (cmd_valid_vpu && (cmd_op != OP_VRMSNORM) && (cmd_op != OP_VQUANT)
          && (cmd_op != OP_VROPE) && (cmd_op != OP_VSILUMUL)
          && (cmd_op != OP_VSOFTMAX) && (cmd_op != OP_VSUBC)) begin
        $error("qcore_vpu_top: opcode 0x%0h is not one this build executes", cmd_op);
      end
      if (emit && wp_v1) begin
        $error("qcore_vpu_top: a staged word was pushed over one still waiting");
      end
    end
  end
`endif
endmodule
