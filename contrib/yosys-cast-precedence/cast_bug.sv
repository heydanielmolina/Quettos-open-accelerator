// A unary operator in front of a size cast: `~N'(expr)`.
// IEEE 1800-2017 A.2.2.1 allows only a constant_primary as the casting type,
// so the only legal parse is `~(N'(expr))`.  Yosys parses `(~N)'(expr)`.
module dut (
  input  logic [31:0] a,
  output logic [31:0] y,   // expect 32'hfffffffc
  output logic [31:0] m    // expect a & 32'hffffffc0  (clear the low 6 bits)
);
  assign y = ~4'(3);
  assign m = a & ~32'(63);
endmodule

module tb;
  logic [31:0] a = 32'hffff_ffff, y, m;
  dut u (.a(a), .y(y), .m(m));
  initial begin
    #1;
    $display("  a                = 0x%08h", a);
    $display("  y = ~4'(3)       = 0x%08h", y);
    $display("  m = a & ~32'(63) = 0x%08h", m);
  end
endmodule
