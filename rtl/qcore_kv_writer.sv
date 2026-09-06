// Quettos Core KV writer: turns the 64 int8 values a KVWRITE names in VSRAM
// into the KV cache writes of docs/ISA.md 0x30. Participating rows run
// ascending. Per row: read the words holding elements vs_src .. vs_src+63 on
// VSRAM port B, one word per cycle with the data one cycle behind the address,
// and keep the low byte of each element; then issue one write beat per cycle
// whenever the write port accepts -- 64 single-byte-strobe beats of the
// transposed K cache at addr_a + (POS/WB)*64*WB + d*WB + POS%WB, or
// ceil(64/WB) full-width beats of the row-major V cache at
// addr_a + (t*k + POS)*WB -- and last the 8-byte meta record {0, m, e} at
// addr_m + POS*8. A row at or past the token capacity writes nothing; it and a
// source range leaving the VSRAM each raise one err_bounds count for that row,
// and elements past the end read as 0. done pulses the cycle after the last
// row's meta beat is accepted; busy is high from the issue until then.
`include "qcore_csr_defs.svh"
module qcore_kv_writer #(
  parameter int WB          = 64,
  parameter int B_MAX       = 1,
  parameter int VSRAM_WORDS = 4096
) (
  input  logic                           clk,
  input  logic                           rst,
  // descriptor issue; the bundle holds from the pulse to done
  input  logic                           cmd_valid_kv,
  input  logic                           cmd_kv_transposed,
  input  logic [31:0]                    cmd_addr_a,
  input  logic [31:0]                    cmd_addr_m,
  input  logic [15:0]                    cmd_k_stride,
  input  logic [15:0]                    cmd_vs_src,
  input  logic [B_MAX-1:0]               cmd_rows,
  input  logic [B_MAX*16-1:0]            cmd_sx_m,
  input  logic [B_MAX*8-1:0]             cmd_sx_e,
  input  logic [31:0]                    cmd_pos,
  // VSRAM port B read of bank src_row + vsb_row; vsb_row holds while a read runs
  output logic [3:0]                     vsb_row,
  output logic                           vsb_en,
  output logic [$clog2(VSRAM_WORDS)-1:0] vsb_addr,
  input  logic [255:0]                   vsb_rdata,
  // QMEM write beats
  output logic                           k_wr_valid,
  input  logic                           k_wr_ready,
  output logic [31:0]                    k_wr_addr,
  output logic [WB*8-1:0]                k_wr_data,
  output logic [WB-1:0]                  k_wr_strb,
  // per-cycle count: POS >= k, or vs_src + 64 past the end, once per row
  output logic [1:0]                     err_bounds_inc,
  output logic                           done,
  output logic                           busy
);
  localparam int DW    = WB * 8;
  localparam int LWB   = $clog2(WB);
  localparam int AW    = $clog2(VSRAM_WORDS);
  localparam int HD    = `QCORE_HEAD_DIM;           // 64 dimensions per head
  localparam int NE    = `QCORE_VSRAM_WORD_ELEMS;   // 8 elements per VSRAM word
  localparam int NW    = HD / NE + 1;               // words spanning 64 elements from any offset
  localparam int WINW  = NW * NE * 8;               // the gathered read window, in bits
  localparam int BUFW  = HD * 8;                    // the 64 bytes the beats slice
  localparam int TILES = (HD + WB - 1) / WB;        // V beats per row
  localparam int VBW   = TILES * DW;                // the window a V beat slices
  localparam int WAW   = 14;                        // VSRAM word address arithmetic
  localparam int MB    = `QCORE_META_BYTES;         // 8-byte meta record

  localparam logic [31:0] WB_MASK  = 32'(WB) - 32'd1;
  localparam logic [6:0]  LAST_KT  = 7'(HD);        // beat index of the meta beat, transposed
  localparam logic [6:0]  LAST_V   = 7'(TILES);     // and row-major

  typedef enum logic [2:0] {
    S_IDLE  = 3'd0,
    S_ROW   = 3'd1,
    S_READ  = 3'd2,
    S_ALIGN = 3'd3,
    S_WRITE = 3'd4
  } state_t;

  // ------------------------------------------------------------------ descriptor
  state_t                  state;
  logic                    d_trans;
  logic                    d_pos_bad;    // POS >= the token capacity
  logic                    d_range_err;  // vs_src + 64 leaves the VSRAM
  logic [31:0]             d_kt_base;    // transposed beat 0 of every row
  logic [31:0]             d_v_base;     // row-major beat 0 of every row
  logic [31:0]             d_v_step;     // k * WB, the V tile stride in bytes
  logic [31:0]             d_meta_addr;
  logic [12:0]             d_word_base;  // vs_src / 8
  logic [2:0]              d_off;        // vs_src % 8
  logic [3:0]              d_nw;         // words to read, 8 or 9
  logic [B_MAX:0]          rows_left;    // participation of row_i and above
  logic [(B_MAX+1)*16-1:0] sx_m_left;    // the same rows' scale mantissas
  logic [(B_MAX+1)*8-1:0]  sx_e_left;    // and exponents
  logic [3:0]              row_i;
  logic [15:0]             cur_m;
  logic [7:0]              cur_e;

  // ------------------------------------------------------------------ VSRAM read
  logic [3:0]              rd_i;
  logic                    cap_v;        // a word arrives on vsb_rdata this cycle
  logic                    cap_z;        // that word is past the end: zeros
  logic [3:0]              cap_i;
  logic [WINW-1:0]         win;
  logic [BUFW-1:0]         dbuf;
  logic [WAW-1:0]          wa_full;
  logic                    wa_ok;
  logic                    issue_rd;
  logic [NE*8-1:0]         rd_low;

  assign wa_full  = {1'b0, d_word_base} + {{(WAW-4){1'b0}}, rd_i};
  assign wa_ok    = wa_full < WAW'(VSRAM_WORDS);
  assign issue_rd = (state == S_READ) && (rd_i < d_nw);
  assign vsb_en   = issue_rd && wa_ok;
  assign vsb_addr = wa_full[AW-1:0];

  always_comb begin
    for (int j = 0; j < NE; j++) begin
      rd_low[j*8 +: 8] = vsb_rdata[j*32 +: 8];
    end
  end

  // cap_i selects one of the NW word slots of the gathered window. Comparing it
  // against each slot in turn keeps every part-select constant, so the whole of
  // win is driven; a variable base would leave the slots past NW undriven.
  always_ff @(posedge clk) begin
    if (cap_v) begin
      for (int w = 0; w < NW; w++) begin
        if (cap_i == 4'(w)) win[w*NE*8 +: NE*8] <= cap_z ? {(NE*8){1'b0}} : rd_low;
      end
    end
  end

  // ------------------------------------------------------------------ beat assembly
  logic [6:0]     bi;        // beat index within the row: data beats, then the meta beat
  logic [6:0]     last_bi;
  logic           is_meta;
  logic [31:0]    waddr;
  logic [VBW-1:0] vbuf;
  logic [DW-1:0]  v_data;
  logic [DW-1:0]  beat_data;
  logic [WB-1:0]  beat_strb;
  logic [31:0]    beat_addr;

  assign last_bi = d_trans ? LAST_KT : LAST_V;
  assign is_meta = (bi == last_bi);
  assign vbuf    = VBW'(dbuf);

  always_comb begin
    v_data = {DW{1'b0}};
    for (int t = 0; t < TILES; t++) begin
      if (bi == 7'(t)) v_data = vbuf[t*DW +: DW];
    end
  end

  assign beat_addr = is_meta ? d_meta_addr : waddr;
  assign beat_data = is_meta ? DW'({8'd0, cur_e, cur_m, 32'd0})
                             : (d_trans ? {{(DW-8){1'b0}}, dbuf[{bi[5:0], 3'd0} +: 8]} : v_data);
  assign beat_strb = is_meta ? {{(WB-MB){1'b0}}, {MB{1'b1}}}
                             : (d_trans ? {{(WB-1){1'b0}}, 1'b1} : {WB{1'b1}});

  // ------------------------------------------------------------------ sequencer
  logic          k_wr_valid_q;
  logic [31:0]   k_wr_addr_q;
  logic [DW-1:0] k_wr_data_q;
  logic [WB-1:0] k_wr_strb_q;
  logic          done_q;
  logic [1:0]    err_q;
  logic          wr_free;

  assign wr_free = !k_wr_valid_q || k_wr_ready;

  always_ff @(posedge clk) begin
    if (rst) begin
      state        <= S_IDLE;
      row_i        <= 4'd0;
      rd_i         <= 4'd0;
      bi           <= 7'd0;
      cap_v        <= 1'b0;
      cap_z        <= 1'b0;
      cap_i        <= 4'd0;
      rows_left    <= {(B_MAX+1){1'b0}};
      vsb_row      <= 4'd0;
      k_wr_valid_q <= 1'b0;
      done_q       <= 1'b0;
      err_q        <= 2'd0;
    end else begin
      done_q <= 1'b0;
      err_q  <= 2'd0;
      cap_v  <= issue_rd;
      cap_i  <= rd_i;
      cap_z  <= issue_rd && !wa_ok;
      case (state)
        S_IDLE: begin
          if (cmd_valid_kv) begin
            d_trans     <= cmd_kv_transposed;
            d_pos_bad   <= cmd_pos >= {16'd0, cmd_k_stride};
            d_range_err <= ({1'b0, cmd_vs_src} + 17'(HD)) > 17'(VSRAM_WORDS * NE);
            d_kt_base   <= cmd_addr_a + ((cmd_pos & ~WB_MASK) << 6) + (cmd_pos & WB_MASK);
            d_v_base    <= cmd_addr_a + (cmd_pos << LWB);
            d_v_step    <= {16'd0, cmd_k_stride} << LWB;
            d_meta_addr <= cmd_addr_m + (cmd_pos << 3);
            d_word_base <= cmd_vs_src[15:3];
            d_off       <= cmd_vs_src[2:0];
            d_nw        <= (cmd_vs_src[2:0] == 3'd0) ? 4'(NW - 1) : 4'(NW);
            rows_left   <= {1'b0, cmd_rows};
            sx_m_left   <= {16'd0, cmd_sx_m};
            sx_e_left   <= {8'd0, cmd_sx_e};
            row_i       <= 4'd0;
            state       <= S_ROW;
          end
        end
        S_ROW: begin
          if (row_i == 4'(B_MAX)) begin
            done_q <= 1'b1;
            state  <= S_IDLE;
          end else begin
            row_i     <= row_i + 4'd1;
            rows_left <= {1'b0, rows_left[B_MAX:1]};
            sx_m_left <= {16'd0, sx_m_left[(B_MAX+1)*16-1:16]};
            sx_e_left <= {8'd0, sx_e_left[(B_MAX+1)*8-1:8]};
            if (rows_left[0]) begin
              err_q <= (d_pos_bad || d_range_err) ? 2'd1 : 2'd0;
              if (!d_pos_bad) begin
                cur_m   <= sx_m_left[15:0];
                cur_e   <= sx_e_left[7:0];
                vsb_row <= row_i;
                rd_i    <= 4'd0;
                bi      <= 7'd0;
                waddr   <= d_trans ? d_kt_base : d_v_base;
                state   <= S_READ;
              end
            end
          end
        end
        S_READ: begin
          // The word issued last is captured in the cycle rd_i reaches d_nw.
          if (rd_i < d_nw) rd_i <= rd_i + 4'd1;
          else             state <= S_ALIGN;
        end
        S_ALIGN: begin
          dbuf  <= BUFW'(win >> {d_off, 3'd0});
          state <= S_WRITE;
        end
        S_WRITE: begin
          if (wr_free) begin
            if (bi <= last_bi) begin
              k_wr_valid_q <= 1'b1;
              k_wr_addr_q  <= beat_addr;
              k_wr_data_q  <= beat_data;
              k_wr_strb_q  <= beat_strb;
              bi           <= bi + 7'd1;
              if (!is_meta) waddr <= waddr + (d_trans ? 32'(WB) : d_v_step);
            end else begin
              k_wr_valid_q <= 1'b0;
              state        <= S_ROW;
            end
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  assign k_wr_valid     = k_wr_valid_q;
  assign k_wr_addr      = k_wr_addr_q;
  assign k_wr_data      = k_wr_data_q;
  assign k_wr_strb      = k_wr_strb_q;
  assign err_bounds_inc = err_q;
  assign done           = done_q;
  assign busy           = (state != S_IDLE) || done_q;
endmodule
