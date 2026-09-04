// Quettos Core -- speed probe: a simulation cost model. The accelerator RTL lives in rtl/.
// VPU stub: VL lanes plus a small FSM that streams n_words vsram words through
// the lanes as an RMSNorm-like pass on vsram port B:
//   pass 1: sum of squares (read-only, 8/VL cycles per word),
//   Rc:     leading-one detect + fake rsqrt mantissa (the real LUT ROMs come with the vector unit),
//   pass 2: y = sat32(round_shift(sat32(round_shift(x*Rc_m, sh1)) * gamma, sh2)),
//           read-modify-write, 8/VL + 1 cycles per word (one write bubble).
// gamma is a pseudo-random int16 stream (the real design streams it from QMEM).
module probe_vpu #(
  parameter int VL = 4,
  parameter int AW = 12
) (
  input  logic          clk,
  input  logic          rst,
  input  logic          start,
  input  logic [AW-1:0] base,
  input  logic [15:0]   n_words,
  input  logic [5:0]    sh1,
  input  logic [5:0]    sh2,
  output logic          done,
  output logic          busy,
  // vsram port B
  output logic          re_b,
  output logic          we_b,
  output logic [AW-1:0] addr_b,
  output logic [255:0]  wd_b,
  input  logic [255:0]  rd_b,
  // observability
  output logic          chk_valid,
  output logic [31:0]   chk_data,
  output logic [31:0]   absmax,
  output logic [31:0]   sat_cnt
);
  localparam int SUB = 8 / VL;          // lane-input sub-blocks per 256-bit word (>= 2)
  localparam int SW  = $clog2(SUB) + 1; // phase counter width (must hold SUB)

  localparam logic [3:0] V_IDLE   = 4'd0;
  localparam logic [3:0] V_P1_RD0 = 4'd1;
  localparam logic [3:0] V_P1     = 4'd2;
  localparam logic [3:0] V_P1_FL  = 4'd3;
  localparam logic [3:0] V_RC     = 4'd4;
  localparam logic [3:0] V_P2_RD0 = 4'd5;
  localparam logic [3:0] V_P2     = 4'd6;
  localparam logic [3:0] V_P2_FL  = 4'd7;
  localparam logic [3:0] V_DONE   = 4'd8;

  logic [3:0]    st;
  logic [15:0]   w;          // word index on the input side
  logic [SW-1:0] ph;         // sub-block / phase within the word
  logic [1:0]    fl;         // flush counter
  logic [AW-1:0] base_r;
  logic [15:0]   n_r;
  logic [15:0]   w_next;
  logic          last_w;

  assign w_next = w + 16'd1;
  assign last_w = (w_next == n_r);

  // --------------------------------------------------------------- lanes
  logic               lane_in;
  logic               pass2;
  logic [VL*32-1:0]   c32_flat;
  logic [15:0]        c16;
  logic [5:0]         l_sh1;
  logic [5:0]         l_sh2;
  logic [VL-1:0]      lane_ov;
  logic [VL*32-1:0]   y_flat;
  logic [VL*64-1:0]   p64_flat;
  logic [VL*32-1:0]   amax_flat;
  logic [VL*32-1:0]   sat_flat;
  logic [15:0]        rc_m;
  logic [31:0]        gs;        // xorshift32 state (gamma source)
  logic [31:0]        gs_next;
  logic [7:0]         sub_off;   // bit offset of the current sub-block in rd_b

  assign gs_next = ((gs ^ (gs << 13)) ^ ((gs ^ (gs << 13)) >> 17)) ^ (((gs ^ (gs << 13)) ^ ((gs ^ (gs << 13)) >> 17)) << 5);
  assign sub_off = {{(8-SW){1'b0}}, ph} * 8'(VL * 32);

  genvar l;
  generate
    for (l = 0; l < VL; l++) begin : g_lane
      logic [31:0] xe;
      logic [15:0] gam;
      logic [7:0]  e_off;
      assign e_off = sub_off + 8'(l * 32);
      assign xe    = rd_b[e_off +: 32];
      assign gam   = gs[l*4 +: 16];
      assign c32_flat[l*32 +: 32] = pass2 ? {{16{gam[15]}}, gam} : xe;

      probe_vpu_lane u_lane (
        .clk       (clk),
        .rst       (rst),
        .clr       (start),
        .in_valid  (lane_in),
        .x         (xe),
        .c32       (c32_flat[l*32 +: 32]),
        .c16       (c16),
        .sh1       (l_sh1),
        .sh2       (l_sh2),
        .out_valid (lane_ov[l]),
        .y         (y_flat[l*32 +: 32]),
        .p64       (p64_flat[l*64 +: 64]),
        .absmax    (amax_flat[l*32 +: 32]),
        .sat_cnt   (sat_flat[l*32 +: 32])
      );
    end
  endgenerate

  assign c16   = pass2 ? rc_m : 16'd1;
  assign l_sh1 = pass2 ? sh1 : 6'd0;
  assign l_sh2 = pass2 ? sh2 : 6'd0;

  // lane-input tags delayed by the 2-stage lane pipeline
  logic          t1_p2, t2_p2;
  logic [SW-1:0] t1_ph, t2_ph;
  logic [AW-1:0] t1_w, t2_w;

  always_ff @(posedge clk) begin
    if (rst) begin
      t1_p2 <= 1'b0; t2_p2 <= 1'b0;
      t1_ph <= {SW{1'b0}}; t2_ph <= {SW{1'b0}};
      t1_w <= {AW{1'b0}}; t2_w <= {AW{1'b0}};
    end else begin
      t1_p2 <= pass2;     t2_p2 <= t1_p2;
      t1_ph <= ph;        t2_ph <= t1_ph;
      t1_w  <= w[AW-1:0]; t2_w  <= t1_w;
    end
  end

  logic out_v;
  assign out_v = &lane_ov;

  // sum of squares (pass 1)
  logic [63:0] ss;
  logic [63:0] psum;
  always_comb begin
    psum = 64'd0;
    for (int i = 0; i < VL; i++) psum = psum + p64_flat[i*64 +: 64];
  end

  // leading-one detect on ss -> normalized top 16 bits -> fake rsqrt mantissa
  logic [5:0]  msb;
  logic [15:0] m16;
  always_comb begin
    msb = 6'd0;
    for (int i = 0; i < 64; i++) if (ss[i]) msb = 6'(i);
  end
  assign m16 = 16'((ss << (6'd63 - msb)) >> 6'd48);

  // output-side word assembly (pass 2)
  logic [255:0] ybuf;
  logic [255:0] wd_merge;
  logic [7:0]   t2_off;
  assign t2_off = {{(8-SW){1'b0}}, t2_ph} * 8'(VL * 32);
  always_comb begin
    wd_merge = ybuf;
    for (int i = 0; i < VL; i++) wd_merge[t2_off + 8'(i * 32) +: 32] = y_flat[i*32 +: 32];
  end

  always_ff @(posedge clk) begin
    if (out_v && t2_p2) ybuf[t2_off +: 32*VL] <= y_flat;
  end

  // --------------------------------------------------------------- FSM
  logic wr_fire;
  assign wr_fire = out_v && t2_p2 && (t2_ph == SW'(SUB - 1));

  always_ff @(posedge clk) begin
    if (rst) begin
      st      <= V_IDLE;
      w       <= 16'd0;
      ph      <= {SW{1'b0}};
      fl      <= 2'd0;
      base_r  <= {AW{1'b0}};
      n_r     <= 16'd0;
      lane_in <= 1'b0;
      pass2   <= 1'b0;
      rc_m    <= 16'h8000;
      gs      <= 32'h9E37_79B9;
      ss      <= 64'd0;
      re_b    <= 1'b0;
      we_b    <= 1'b0;
      addr_b  <= {AW{1'b0}};
      wd_b    <= 256'd0;
      done    <= 1'b0;
      chk_valid <= 1'b0;
      chk_data  <= 32'd0;
    end else begin
      lane_in   <= 1'b0;
      re_b      <= 1'b0;
      we_b      <= 1'b0;
      done      <= 1'b0;
      chk_valid <= 1'b0;

      // sum of squares accumulates lane products of pass-1 inputs
      if (out_v && !t2_p2) ss <= ss + psum;

      // pass-2 write-back (registered; never collides with the read slot)
      if (wr_fire) begin
        we_b      <= 1'b1;
        addr_b    <= base_r + t2_w;
        wd_b      <= wd_merge;
        chk_valid <= 1'b1;
        chk_data  <= wd_merge[31:0] ^ wd_merge[63:32] ^ wd_merge[95:64] ^ wd_merge[127:96]
                   ^ wd_merge[159:128] ^ wd_merge[191:160] ^ wd_merge[223:192] ^ wd_merge[255:224];
      end

      case (st)
        V_IDLE: begin
          if (start) begin
            base_r <= base;
            n_r    <= n_words;
            pass2  <= 1'b0;
            ss     <= 64'd0;
            st     <= V_P1_RD0;
          end
        end

        V_P1_RD0: begin
          re_b   <= 1'b1;
          addr_b <= base_r;
          w      <= 16'd0;
          ph     <= {SW{1'b0}};
          st     <= V_P1;
        end

        V_P1: begin
          lane_in <= 1'b1;
          if (ph == SW'(SUB - 1)) begin
            ph <= {SW{1'b0}};
            w  <= w_next;
            if (last_w) begin
              fl <= 2'd0;
              st <= V_P1_FL;
            end else begin
              re_b   <= 1'b1;
              addr_b <= base_r + w_next[AW-1:0];
            end
          end else begin
            ph <= ph + {{(SW-1){1'b0}}, 1'b1};
          end
        end

        V_P1_FL: begin
          fl <= fl + 2'd1;
          if (fl == 2'd3) st <= V_RC;
        end

        V_RC: begin
          rc_m  <= {m16[15], ~m16[14:0]};
          pass2 <= 1'b1;
          st    <= V_P2_RD0;
        end

        V_P2_RD0: begin
          re_b   <= 1'b1;
          addr_b <= base_r;
          w      <= 16'd0;
          ph     <= {SW{1'b0}};
          st     <= V_P2;
        end

        V_P2: begin
          if (ph == SW'(SUB)) begin
            // write bubble: issue the next read here
            ph <= {SW{1'b0}};
            w  <= w_next;
            if (last_w) begin
              fl <= 2'd0;
              st <= V_P2_FL;
            end else begin
              re_b   <= 1'b1;
              addr_b <= base_r + w_next[AW-1:0];
            end
          end else begin
            lane_in <= 1'b1;
            gs      <= gs_next;
            ph      <= ph + {{(SW-1){1'b0}}, 1'b1};
          end
        end

        V_P2_FL: begin
          fl <= fl + 2'd1;
          if (fl == 2'd3) st <= V_DONE;
        end

        V_DONE: begin
          done      <= 1'b1;
          chk_valid <= 1'b1;
          chk_data  <= ss[31:0] ^ ss[63:32] ^ {16'd0, rc_m};
          pass2     <= 1'b0;
          st        <= V_IDLE;
        end

        default: st <= V_IDLE;
      endcase
    end
  end

  assign busy = (st != V_IDLE);

  // absmax / saturation summary across lanes
  always_comb begin
    absmax  = 32'd0;
    sat_cnt = 32'd0;
    for (int i = 0; i < VL; i++) begin
      if (amax_flat[i*32 +: 32] > absmax) absmax = amax_flat[i*32 +: 32];
      sat_cnt = sat_cnt + sat_flat[i*32 +: 32];
    end
  end
endmodule
