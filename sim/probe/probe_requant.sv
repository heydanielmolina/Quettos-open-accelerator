// Quettos Core -- speed probe: a simulation cost model. The accelerator RTL lives in rtl/.
// Requant stub: drains WB accumulators one per cycle at tile end.
//   t = sat40( round_shift(acc * Sw_m, 16) )          (40 x 17 -> 57 bit product)
//   y = sat32( round_shift(t * Sx_m, S) ) + bias      (40 x 17 -> 57 bit product)
//   S = sbias - (Sw_e + Sx_e) clamped to [0,63]; m == 0 -> output 0.
// Per-channel meta {i32 bias, u16 Sw_m, i8 Sw_e, u8 pad} is taken from the
// meta side-FIFO (WB/8 metas per WB*8-bit beat). Outputs are assembled 8 per
// 256-bit word and written to vsram port B, or fed to a running argmax
// (strict greater). Tracks absmax and saturation / shift-clamp counters.
module probe_requant #(
  parameter int WB    = 64,
  parameter int ACC_W = 40,
  parameter int AW    = 12
) (
  input  logic                  clk,
  input  logic                  rst,
  input  logic                  gemv_start,   // clears absmax / argmax
  input  logic                  start,        // begin draining a finished tile
  input  logic [15:0]           tile,
  input  logic [WB*ACC_W-1:0]   acc_flat,     // inactive accumulator set
  input  logic                  meta_valid,
  input  logic [WB*8-1:0]       meta_data,
  output logic                  meta_pop,
  input  logic [15:0]           sx_m,
  input  logic [7:0]            sx_e,
  input  logic [7:0]            sbias,
  input  logic                  argmax_mode,
  input  logic [AW-1:0]         vs_dst,
  output logic                  busy,
  output logic                  wr_en,
  output logic [AW-1:0]         wr_addr,
  output logic [255:0]          wr_data,
  output logic                  y_valid,
  output logic [31:0]           y_out,
  output logic [31:0]           absmax,
  output logic [31:0]           argmax_idx,
  output logic [31:0]           argmax_val,
  output logic [31:0]           sat_count,
  output logic [31:0]           err_count
);
  localparam int MPB = WB / 8;                 // metas per beat
  localparam int MB  = $clog2(MPB);            // meta index bits within a beat
  localparam int JW  = $clog2(WB);             // channel index bits
  localparam int OW  = $clog2(WB * ACC_W);     // bit offset into acc_flat
  localparam int MW  = $clog2(WB * 8);         // bit offset into meta_data
  localparam int PW  = ACC_W + 17;             // 57-bit products

  localparam logic [OW-1:0] ACC_W_OW = OW'(ACC_W);
  localparam logic [MW-1:0] META_W   = MW'(64);

  function automatic logic signed [ACC_W-1:0] sat_acc(input logic signed [PW-1:0] x);
    logic ovf;
    ovf     = ~((&x[PW-1:ACC_W-1]) | ~(|x[PW-1:ACC_W-1]));
    sat_acc = ovf ? {x[PW-1], {(ACC_W-1){~x[PW-1]}}} : x[ACC_W-1:0];
  endfunction

  function automatic logic signed [31:0] sat32_57(input logic signed [PW-1:0] x);
    logic ovf;
    ovf      = ~((&x[PW-1:31]) | ~(|x[PW-1:31]));
    sat32_57 = ovf ? {x[PW-1], {31{~x[PW-1]}}} : x[31:0];
  endfunction

  function automatic logic signed [31:0] sat32_33(input logic signed [32:0] x);
    logic ovf;
    ovf      = x[32] ^ x[31];
    sat32_33 = ovf ? {x[32], {31{~x[32]}}} : x[31:0];
  endfunction

  // ---------------------------------------------------------------- stage 0
  logic          draining;
  logic [JW-1:0] j;
  logic [15:0]   tile_r;
  logic [MB-1:0] jm;
  logic          s0_fire;
  logic [OW-1:0] acc_off;
  logic [MW-1:0] meta_off;
  logic [55:0]   meta_sel;      // {sw_e[7:0], sw_m[15:0], bias[31:0]} (pad byte ignored)
  logic signed [ACC_W-1:0] acc_sel;

  assign jm       = j[MB-1:0];
  assign s0_fire  = draining && meta_valid;
  assign meta_pop = s0_fire && (jm == {MB{1'b1}});
  assign acc_off  = {{(OW-JW){1'b0}}, j} * ACC_W_OW;
  assign meta_off = {{(MW-MB){1'b0}}, jm} * META_W;
  assign acc_sel  = acc_flat[acc_off +: ACC_W];
  assign meta_sel = meta_data[meta_off +: 56];

  always_ff @(posedge clk) begin
    if (rst) begin
      draining <= 1'b0;
      j        <= {JW{1'b0}};
      tile_r   <= 16'd0;
    end else begin
      if (start) begin
        draining <= 1'b1;
        j        <= {JW{1'b0}};
        tile_r   <= tile;
      end else if (s0_fire) begin
        j <= j + {{(JW-1){1'b0}}, 1'b1};
        if (j == {JW{1'b1}}) draining <= 1'b0;
      end
    end
  end

  // ---------------------------------------------------------------- stage 1
  logic                    s1_valid;
  logic signed [ACC_W-1:0] s1_acc;
  logic [31:0]             s1_bias;
  logic [15:0]             s1_swm;
  logic [7:0]              s1_swe;
  logic [31:0]             s1_idx;

  always_ff @(posedge clk) begin
    if (rst) begin
      s1_valid <= 1'b0;
      s1_acc   <= {ACC_W{1'b0}};
      s1_bias  <= 32'd0;
      s1_swm   <= 16'd0;
      s1_swe   <= 8'd0;
      s1_idx   <= 32'd0;
    end else begin
      s1_valid <= s0_fire;
      if (s0_fire) begin
        s1_acc  <= acc_sel;
        s1_bias <= meta_sel[31:0];
        s1_swm  <= meta_sel[47:32];
        s1_swe  <= meta_sel[55:48];
        s1_idx  <= {{(16-JW){1'b0}}, tile_r, j};
      end
    end
  end

  logic signed [PW-1:0] p1;
  logic signed [PW-1:0] p1_sh;
  logic signed [ACC_W-1:0] t1;
  logic                 t1_sat;
  logic signed [9:0]    s_raw;      // sbias - (sw_e + sx_e), 10-bit signed
  logic [5:0]           s_raw_lo;
  logic [5:0]           s_clamped;
  logic                 s_err;

  assign p1    = $signed({{17{s1_acc[ACC_W-1]}}, s1_acc}) * $signed({{(ACC_W+1){1'b0}}, s1_swm});
  assign p1_sh = (p1 >>> 16) + $signed({{(PW-1){1'b0}}, p1[15]});
  assign t1    = sat_acc(p1_sh);
  assign t1_sat = ~((&p1_sh[PW-1:ACC_W-1]) | ~(|p1_sh[PW-1:ACC_W-1]));
  assign s_raw = $signed({{2{sbias[7]}}, sbias}) - ($signed({{2{s1_swe[7]}}, s1_swe}) + $signed({{2{sx_e[7]}}, sx_e}));
  assign s_raw_lo = s_raw[5:0];

  always_comb begin
    if (s_raw < 10'sd0) begin
      s_clamped = 6'd0;
      s_err     = 1'b1;
    end else if (s_raw > 10'sd63) begin
      s_clamped = 6'd63;
      s_err     = 1'b1;
    end else begin
      s_clamped = s_raw_lo;
      s_err     = 1'b0;
    end
  end

  // ---------------------------------------------------------------- stage 2
  logic                    s2_valid;
  logic signed [ACC_W-1:0] s2_t;
  logic [5:0]              s2_s;
  logic [31:0]             s2_bias;
  logic [31:0]             s2_idx;
  logic                    s2_mzero;
  logic                    s2_sat;
  logic                    s2_err;

  always_ff @(posedge clk) begin
    if (rst) begin
      s2_valid <= 1'b0;
      s2_t     <= {ACC_W{1'b0}};
      s2_s     <= 6'd0;
      s2_bias  <= 32'd0;
      s2_idx   <= 32'd0;
      s2_mzero <= 1'b0;
      s2_sat   <= 1'b0;
      s2_err   <= 1'b0;
    end else begin
      s2_valid <= s1_valid;
      if (s1_valid) begin
        s2_t     <= t1;
        s2_s     <= s_clamped;
        s2_bias  <= s1_bias;
        s2_idx   <= s1_idx;
        s2_mzero <= (s1_swm == 16'd0);
        s2_sat   <= t1_sat;
        s2_err   <= s_err;
      end
    end
  end

  logic signed [PW-1:0] p2;
  logic [63:0]          p2_bits;
  logic [5:0]           s_m1;
  logic                 rbit;
  logic signed [PW-1:0] p2_sh;
  logic signed [31:0]   y0;
  logic                 y0_sat;
  logic signed [32:0]   y_sum;
  logic signed [31:0]   y1;
  logic                 y1_sat;

  assign p2      = $signed({{17{s2_t[ACC_W-1]}}, s2_t}) * $signed({{(ACC_W+1){1'b0}}, sx_m});
  assign p2_bits = {{(64-PW){p2[PW-1]}}, p2};
  assign s_m1    = s2_s - 6'd1;
  assign rbit    = (s2_s != 6'd0) & p2_bits[s_m1];
  assign p2_sh   = (p2 >>> s2_s) + $signed({{(PW-1){1'b0}}, rbit});
  assign y0      = s2_mzero ? 32'sd0 : sat32_57(p2_sh);
  assign y0_sat  = !s2_mzero && ~((&p2_sh[PW-1:31]) | ~(|p2_sh[PW-1:31]));
  assign y_sum   = $signed({y0[31], y0}) + $signed({s2_bias[31], s2_bias});
  assign y1      = s2_mzero ? 32'sd0 : sat32_33(y_sum);
  assign y1_sat  = !s2_mzero && (y_sum[32] ^ y_sum[31]);

  // ---------------------------------------------------------------- stage 3
  logic        s3_valid;
  logic [31:0] s3_y;
  logic [31:0] s3_idx;
  logic [31:0] s3_absy;
  logic [31:0] y_abs;
  logic [223:0] wbuf;
  logic [7:0]  wb_off;

  assign y_abs  = y1[31] ? (~y1 + 32'd1) : y1;
  assign wb_off = {s3_idx[2:0], 5'b00000};

  always_ff @(posedge clk) begin
    if (rst) begin
      s3_valid  <= 1'b0;
      s3_y      <= 32'd0;
      s3_idx    <= 32'd0;
      s3_absy   <= 32'd0;
      sat_count <= 32'd0;
      err_count <= 32'd0;
    end else begin
      s3_valid <= s2_valid;
      if (s2_valid) begin
        s3_y    <= y1;
        s3_idx  <= s2_idx;
        s3_absy <= y_abs;
        if (!s2_mzero && (s2_sat || y0_sat || y1_sat)) sat_count <= sat_count + 32'd1;
        if (!s2_mzero && s2_err)                        err_count <= err_count + 32'd1;
      end
    end
  end

  // absmax / argmax / word assembly / write-out
  logic [255:0] word_full;
  assign word_full = {s3_y, wbuf};

  always_ff @(posedge clk) begin
    if (rst) begin
      absmax     <= 32'd0;
      argmax_idx <= 32'd0;
      argmax_val <= 32'h8000_0000;
      wr_en      <= 1'b0;
      wr_addr    <= {AW{1'b0}};
      wr_data    <= 256'd0;
      y_valid    <= 1'b0;
      y_out      <= 32'd0;
    end else begin
      y_valid <= s3_valid;
      y_out   <= s3_y;
      wr_en   <= 1'b0;
      if (gemv_start) begin
        absmax     <= 32'd0;
        argmax_idx <= 32'd0;
        argmax_val <= 32'h8000_0000;
      end else if (s3_valid) begin
        if (s3_absy > absmax) absmax <= s3_absy;
        if (argmax_mode) begin
          if ($signed(s3_y) > $signed(argmax_val)) begin
            argmax_val <= s3_y;
            argmax_idx <= s3_idx;
          end
        end else if (s3_idx[2:0] == 3'd7) begin
          wr_en   <= 1'b1;
          wr_addr <= vs_dst + s3_idx[AW+2:3];
          wr_data <= word_full;
        end
      end
    end
  end

  always_ff @(posedge clk) begin
    if (s3_valid && (s3_idx[2:0] != 3'd7)) wbuf[wb_off +: 32] <= s3_y;
  end

  assign busy = draining || s1_valid || s2_valid || s3_valid || wr_en;
endmodule
