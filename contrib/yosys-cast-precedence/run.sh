#!/bin/sh
# Yosys mis-parses a unary operator placed in front of a size cast.
# Runs Yosys, Icarus Verilog and Verilator over cast_bug.sv and prints what
# each one computes.  Needs only those three tools on PATH; writes to ./build.
set -u
cd "$(dirname "$0")"
SRC=cast_bug.sv
B=build
rm -rf "$B"; mkdir "$B"
YS="read_verilog -sv $SRC; hierarchy -top dut; proc; opt -full; clean"

echo "=== tool versions ==="
yosys -V
iverilog -V 2>&1 | head -1
verilator --version

echo
echo "=== yosys: synthesized netlist ==="
echo "\$ yosys -p \"$YS; write_verilog -noattr\""
yosys -q -p "$YS; write_verilog -noattr $B/dut_synth.v" || exit 1
grep 'assign' "$B/dut_synth.v"

echo
echo "=== yosys: its own parse tree (read_verilog -dump_ast1) ==="
yosys -p "read_verilog -sv -no_dump_ptr -dump_ast1 $SRC" 2>&1 \
  | awk "/AST_ASSIGN <$SRC:9\./{n=5} n-->0" \
  | sed -E "s/bits='[01]+'\([0-9]+\) //"

echo
echo "=== iverilog + vvp ==="
iverilog -g2012 -s tb -o "$B/tb.vvp" "$SRC" || exit 1
vvp -n "$B/tb.vvp" | tee "$B/icarus.txt"

echo
echo "=== verilator --binary ==="
verilator --binary --quiet -Wno-WIDTHEXPAND -Mdir "$B/vobj" \
  --top-module tb "$SRC" -o sim >/dev/null 2>&1 || exit 1
"$B/vobj/sim" 2>/dev/null | grep '=' | tee "$B/verilator.txt"

# Yosys's own constant evaluation of the same netlist, for a numeric comparison.
yosys -p "$YS; eval -set a 4294967295 -show y -show m" 2>&1 \
  | sed -n 's/^Eval result: .\([ym]\) = \([0-9]*\)\./\1 \2/p' > "$B/yosys.txt"
yv=$(printf '0x%08x' "$(awk '$1=="y"{print $2}' "$B/yosys.txt")")
ym=$(printf '0x%08x' "$(awk '$1=="m"{print $2}' "$B/yosys.txt")")
iv=$(sed -n '2s/.*= //p' "$B/icarus.txt");    im=$(sed -n '3s/.*= //p' "$B/icarus.txt")
vv=$(sed -n '2s/.*= //p' "$B/verilator.txt"); vm=$(sed -n '3s/.*= //p' "$B/verilator.txt")

echo
echo "=== comparison (a = 0xffffffff) ==="
fmt='%-16s %-12s %-12s %-12s %s\n'
printf "$fmt" expression expected yosys icarus verilator
printf "$fmt" "~4'(3)"       0xfffffffc "$yv" "$iv" "$vv"
printf "$fmt" "a & ~32'(63)" 0xffffffc0 "$ym" "$im" "$vm"
