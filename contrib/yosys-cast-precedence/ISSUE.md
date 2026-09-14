# A unary operator in front of a size cast binds to the cast width, not the cast

Yosys reads `~N'(expr)` as `(~N)'(expr)` — the operator is applied to the cast
width and the value passes through unchanged — so `assign y = ~4'(3);`
synthesizes to `assign y = 32'd3;` with no warning.

## Minimal case

```systemverilog
module dut (
  input  logic [31:0] a,
  output logic [31:0] y,   // expect 32'hfffffffc
  output logic [31:0] m    // expect a & 32'hffffffc0  (clear the low 6 bits)
);
  assign y = ~4'(3);
  assign m = a & ~32'(63);
endmodule
```

```
$ yosys -p "read_verilog -sv cast_bug.sv; hierarchy -top dut; proc; opt -full; clean; write_verilog -noattr"
  assign m = { 26'h0000000, a[5:0] };
  assign y = 32'd3;
```

`m` is `a & 63` — the complement of the mask that was written.

## What each tool computes, with a = 32'hffffffff

| expression | expected | yosys 0.65 | iverilog 13 | verilator 5.048 |
|---|---|---|---|---|
| `~4'(3)` | `0xfffffffc` | `0x00000003` | `0xfffffffc` | `0xfffffffc` |
| `a & ~32'(63)` | `0xffffffc0` | `0x0000003f` | `0xffffffc0` | `0xffffffc0` |

The two simulators agree bit for bit. The yosys column is `eval -set a
4294967295 -show y -show m` on the synthesized netlist.

A self-contained reproducer that runs all three tools and prints this table is
attached (`cast_bug.sv` + `run.sh`, two files, no other setup).

## The parse Yosys performs

`read_verilog -dump_ast1` shows the complement sitting in the cast's *size*
slot:

```
      AST_ASSIGN <cast_bug.sv:9.10-9.20>
        AST_IDENTIFIER <cast_bug.sv:9.10-9.11> str='\y' in_lvalue
        AST_CAST_SIZE <cast_bug.sv:9.14-9.19>
          AST_BIT_NOT <cast_bug.sv:9.14-9.16>
            AST_CONSTANT <cast_bug.sv:9.15-9.16> ... int=4
```

So the cast width is `~4` = -5. `AstNode::bitsAsConst(int width, ...)` in
`frontends/ast/ast.cc` guards its resize with `width >= 0`, so a negative width
is a no-op and the operand's bits pass through untouched — hence the constant 3.

Writing the parentheses explicitly gives the expected result, which is
consistent with this reading:

```
assign y = ~(4'(3));        ->  assign y = 32'd4294967292;
assign m = a & ~(32'(63));  ->  assign m = { a[31:6], 6'h00 };
```

## Grammar

IEEE 1800-2017 Syntax 6-7 / A.2.2.1:

```
cast ::= casting_type ' ( expression )
casting_type ::= simple_type | constant_primary | signing | string | const
```

and A.8.4 / Syntax 11-7:

```
constant_primary ::=
      primary_literal
    | ps_parameter_identifier constant_select
    | ...
    | ( constant_mintypmax_expression )
    | constant_cast
    | constant_assignment_pattern_expression
    | type_reference
    | null
```

`constant_primary` has no unary alternative; the unary form is one level up, in
A.8.3:

```
constant_expression ::=
      constant_primary
    | unary_operator { attribute_instance } constant_primary
```

So `~4` is a `constant_expression`, which `casting_type` does not accept, and
`~4'(3)` has exactly one derivation: `unary_operator` applied to a `primary`
that is a cast. A compound width reaches `casting_type` only through
`( constant_mintypmax_expression )`, which is why §6.24.1's own example writes
`(P+1)'(x - 2)` with parentheses. Yosys handles that parenthesized form
correctly.

Separately, §6.24.1 says "It shall be an error if the size specified is zero or
negative", so even under Yosys's own parse the width `~4` = -5 calls for a
diagnostic rather than a value. Yosys does apply that rule when the cast operand
is not constant — `assign y = ~4'(a);` gives `ERROR: Static cast with zero or
negative size!` — so the same expression errors or silently folds to a wrong
constant depending only on whether the operand is foldable.

## Where it comes from

`frontends/verilog/verilog_parser.y`, lines 567-580 (v0.65, v0.69 and `main`
are identical here):

```
// operator precedence from low to high
...
%left OP_POW
%precedence OP_CAST
%precedence UNARY_OPS
```

In bison a later declaration ranks higher, so `UNARY_OPS` outranks `OP_CAST`
and the parser reduces `~4` rather than shifting the apostrophe. The cast rule
(line 3555 in v0.65 and v0.69, 3532 in `main`) also takes a full expression as
the casting type:

```
	basic_expr OP_CAST TOK_LPAREN expr TOK_RPAREN {
```

where the LRM allows only a `constant_primary`. The `TOK_SIGNED OP_CAST`,
`TOK_UNSIGNED OP_CAST` and `typedef_base_type OP_CAST` productions do not go
through `basic_expr`, and those forms behave correctly:
`~signed'(4'd3)`, `~unsigned'(4'd3)`, `~int'(3)` and `~byte'(3)` all give
`32'd4294967292`.

All ten unary operators before a size cast are mis-parsed, not just `~`, and
the result in each case follows from the mis-parsed width (`assign y = <expr>;`,
`y` 32 bits, iverilog and verilator agree on every row):

| expr | yosys | correct | mis-parsed width |
|---|---|---|---|
| `~4'(3)` | `32'd3` | `0xfffffffc` | -5 |
| `-4'(3)` | `32'd3` | `0xfffffffd` | -4 |
| `+4'(3)` | `32'd3` | `0x00000003` | 4 (identity, so it coincides) |
| `!4'(3)` | `32'hxxxxxxxx` | `0x00000000` | 0 |
| `&4'(3)` | `32'hxxxxxxxx` | `0x00000000` | 0 |
| `~\|4'(3)` | `32'hxxxxxxxx` | `0x00000000` | 0 |
| `~^4'(3)` | `32'hxxxxxxxx` | `0x00000001` | 0 |
| `\|4'(3)` | `32'd4294967295` | `0x00000001` | 1 |
| `^4'(3)` | `32'd4294967295` | `0x00000000` | 1 |
| `~&4'(15)` | `32'd4294967295` | `0x00000000` | 1 |

Literal, `parameter`, `localparam` and macro widths behave the same.

## Version and platform

```
Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)
Icarus Verilog version 13.0 (stable) (v13_0)
Verilator 5.048 2026-04-26 rev vUNKNOWN-built20260426
```

macOS 15 (Darwin 25.5.0), arm64, all three from Homebrew.

## Not checked

I ran Yosys 0.65 only. I have not run 0.69 or `main`, so I cannot say from
execution whether they still behave this way — I only checked that the two
`%precedence` lines and the `basic_expr OP_CAST` production are unchanged in
those trees. If this was fixed after 0.65 by some other route, please close it.

I also did not build a patched Yosys, so I have no evidence about what the right
fix is — swapping the two `%precedence` declarations and narrowing the casting
type to a primary-like nonterminal are both plausible, and I have not tested
either for side effects on the rest of the grammar.
