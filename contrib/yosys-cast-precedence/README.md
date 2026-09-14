# Yosys size-cast precedence

A unary operator written in front of a bare size cast reaches the three tools
this project uses as two different expressions. Yosys 0.65 binds the operator to
the cast *width*:

```systemverilog
assign m = a & ~32'(63);   // yosys:              assign m = { 26'h0000000, a[5:0] };
assign m = a & ~(32'(63)); // yosys, and both simulators: a & 32'hffffffc0
```

The first line synthesizes `a & 63` — the complement of the mask that was
written. Verilator and Icarus build `a & ~63`. Nothing warns, so simulation
passes against RTL that the netlist does not implement.

That is why `scripts/lint.sh` rejects `~ - + ! & | ^` in front of a bare size
cast and every size cast under an operator in `rtl/` is parenthesized:
`~(25'(WB - 1))`, never `~25'(WB - 1)`. The parenthesized form is read the same
way by all three front ends, which `make gatesim` then confirms cell by cell
(`sim/gatesim/README.md`).

## Contents

| file | what it is |
|---|---|
| `cast_bug.sv` | the construct, 22 lines, in a module and a bench |
| `run.sh` | runs Yosys, Icarus and Verilator over it and prints a comparison table |
| `transcript.txt` | that script's output on macOS 15 / arm64 with Homebrew tools |
| `ISSUE.md` | the upstream report, with the IEEE 1800-2017 grammar that fixes the parse and the two lines of `verilog_parser.y` that produce it |

Nothing here is wired into the build. It depends only on `yosys`, `iverilog` and
`verilator` being on `PATH`, and runs from any directory:

```
$ contrib/yosys-cast-precedence/run.sh
...
=== comparison (a = 0xffffffff) ===
expression       expected     yosys        icarus       verilator
~4'(3)           0xfffffffc   0x00000003   0xfffffffc   0xfffffffc
a & ~32'(63)     0xffffffc0   0x0000003f   0xffffffc0   0xffffffc0
```

Scratch output goes to a `build/` directory beside the script, which the repo
already ignores.
