# Contributing to Quettos Core

Thanks for looking. This repository is a small, opinionated hardware project.
The rules below exist so that the same SystemVerilog passes Verilator 5,
Yosys 0.65 and Icarus 13 with zero rewrites, and so that every number in the
README can be traced to a command that produced it.

## 1. Pull requests and commits

- Fork, branch, and open a pull request against `main`. Keep PRs small and
  single-purpose, and say in the description exactly what you ran to validate
  the change (the commands, verbatim).
- Commit messages: a short imperative subject line (under 72 characters), a
  blank line, then the why. One logical change per commit.
- CI must be green (`make lint`, `make style`, `uv run pytest -q sw/tests`, and
  the RTL jobs) before a PR is reviewed.
- Never commit anything under `build/`. Model weights are downloaded at build
  time and are not redistributed (see `NOTICE`).

## 2. Development environment

- Python is **uv-managed** only (`uv 0.9.x`, Python 3.13 via `.python-version`).
  Do not use a system or Homebrew Python. `torch` is never a core dependency; it
  lives in the optional `ref` group and CI never installs it.
- No secrets and no Hugging Face tokens. Both models are ungated.
- Do not vendor code from other projects without a license review.

## 3. SystemVerilog subset and coding rules

The RTL lives in `rtl/`, every module is named `qcore_*`, the synthesis top is
`qcore_top`, and the package is `qcore_pkg`. The subset below is the
intersection accepted by Verilator 5.048 (`--lint-only -Wall -Wpedantic`),
Yosys 0.65 (`read_verilog -sv`) and Icarus 13 (`-g2012`).

### Allowed

- `module` with ANSI port lists; ports are `logic`/`logic signed` packed
  vectors only.
- `logic`, `logic signed`, `localparam`, `parameter`, `typedef` of packed
  structs and packed vectors, `enum logic [N:0]` for FSM states.
- `always_ff @(posedge clk)` for state, `always_comb` for combinational logic,
  `assign` for wires. One process per register group.
- `generate` / `for` generate loops with a named block, `genvar`.
- Functions in `qcore_pkg` that are pure, single-assignment to the function
  name, referenced with explicit scoping: `qcore_pkg::round_shift(...)`.
- `case` with a `default` arm; `if/else`; ternaries.
- `$signed()` on every arithmetic operand; explicitly sized products
  (`logic signed [23:0] prod`).
- `$display`/`$error`/`$fatal` only inside `` `ifndef SYNTHESIS `` ... `` `endif ``.
- Memories as **1-D unpacked arrays of packed words** (`logic [255:0] mem
  [0:VSRAM_WORDS-1]`) with **registered reads** (read address or read data
  goes through a flop) and separate `always_ff` read and write processes.
- ROM initialization only through an untyped `parameter ROM_FILE = ""` (or
  `ROM_FILE_<TABLE>` where a module holds several; Yosys 0.65 rejects
  `parameter string`), set to an **absolute path** by the Makefile / `.ys`
  script: `$readmemh(ROM_FILE, mem);`. Yosys resolves `$readmemh` while
  reading, so the script uses `read_verilog -sv -defer` followed by `chparam
  -set ROM_FILE "<abs path>" <module>` before `hierarchy`, with the image paths
  ahead of any other `-set` in that `chparam`.
- ISA constants (opcodes, flag masks, descriptor field positions, CSR offsets,
  PERF indices) come from the generated include `rtl/qcore_csr_defs.svh`
  (`uv run quettos csr-defs`), which defines one `` `define QCORE_<NAME> ``
  macro per constant; a module or package restates the ones it uses as
  `localparam` (`` localparam int OP_GEMV = `QCORE_OP_GEMV; ``). Macros rather
  than `localparam` in the include because Verilator `-Wall` reports every
  unused `localparam`, in module and package scope alike, and waivers are not
  allowed.

### Forbidden (each one is a lint failure or a Yosys hard error)

- **Unpacked arrays on module ports.** Flatten to a packed vector and index with
  `qcore_pkg::` helper functions. (Yosys 0.65 hard error.)
- **Wildcard imports** (`import qcore_pkg::*;`). Always write `qcore_pkg::NAME`.
- **`return` inside functions.** Assign to the function name instead.
- **Asynchronous reset** or any `negedge` sensitivity. Synchronous reset only:
  `always_ff @(posedge clk) if (rst) ... else ...`.
- **`unique case` / `priority case`**, `unique if`, `priority if`.
- **`interface`**, **`class`**, `modport`, `program`, `clocking`, `assert
  property`, `always_latch`, `initial` blocks in synthesizable modules -- with
  one exception, for one reason. The table images the ROM rule above mandates
  reach their memory through `$readmemh`, and an initializing `$readmemh` has
  no placement outside an `initial` block that Verilator, Yosys and Icarus all
  read the same way. `rtl/qcore_lut_rom.sv` therefore carries
  `initial $readmemh(ROM_FILE, mem);` and is the only file under `rtl/` with an
  `initial` block (`docs/RTL.md` 3.15). State a reset can reach is initialized
  in `always_ff` under `rst` instead, and a second initial block needs a reason
  of the same kind.
- Literal **`$readmemh("...")`** with a string constant (CI greps for it).
- Multi-dimensional unpacked memories, `automatic` variables, `real`,
  `string` variables and `string` parameters, `$clog2` on anything but
  constants, dynamic arrays, queues.
- Lint waivers of any kind (`/* verilator lint_off */`). `UNOPTFLAT` must be
  fixed, not silenced.
- **A unary operator on a bare size cast**: write `~(25'(WB - 1))`, never
  `~25'(WB - 1)`. Yosys 0.65 binds the operator to the size literal, so the
  second form synthesizes the mask itself where Verilator and Icarus simulate
  its complement; the same goes for `& | ^ - + !`. `scripts/lint.sh` greps for
  it and `make gatesim` compares the netlist against the source, so the two
  tools cannot quietly disagree.
- Any fixed-point format knowledge in RTL. **Numerics conventions live only in
  `sw/quettos/numerics.py`**; every shift, exponent and constant reaches the
  RTL as a descriptor field. If you want to change rounding, change
  `numerics.py`, regenerate, and let the tests tell you what broke.

### Style

- Two-space indent, `snake_case` identifiers, `UPPER_CASE` parameters and
  localparams, one module per file, file name equals module name.
- Every module header comment states the module's contract in a few lines:
  inputs, outputs, latency, and what it does not do.
- Keep modules small enough to lint individually. A file over 1,000 lines needs
  a header comment giving the reason the module does not split, and one file
  carries such a note: `rtl/qcore_vpu_top.sv`, whose passes are stages of one
  shift register, so a boundary drawn through them would carry the stage indices
  across it and put the issue decision on a combinational round trip through an
  interface -- the shape the handshake rules of `docs/RTL.md` section 1 exist to
  keep out. Without a reason of that kind, split the module.

## 4. Three-parser lint recipe

`make lint` runs `scripts/lint.sh`, which lints every `rtl/*.sv` file -- the
seventeen modules and the package -- and every `sim/cocotb/wrappers/*.sv` block
assembly as its own top: nineteen tops. The package is taken the way each tool
takes a package (Verilator elaborates it as the top unit, Yosys parses it, and
Icarus, which needs a module, gets an empty wrapper). The file list is
`qcore_pkg.sv` first and then every other file once: Yosys and Icarus resolve
`qcore_pkg::` references only after parsing the package, and a file named twice
is a duplicate declaration (Verilator `MODDUP`, an Icarus syntax error).

```sh
SV="rtl/qcore_pkg.sv $(ls rtl/*.sv | grep -v qcore_pkg.sv | tr '\n' ' ')"
verilator --lint-only -Wall -Wpedantic -Irtl --top-module <mod> $SV [sim/cocotb/wrappers/*.sv]
yosys -q -p "read_verilog -sv -defer -Irtl $SV [...]; hierarchy -check -top <mod>; proc; opt; check -assert"
iverilog -g2012 -Irtl -s <mod> -o /dev/null $SV [...]
```

A wrapper top is elaborated with the `rtl/*.sv` files plus the wrappers on the
command line; a module top is elaborated with the `rtl/*.sv` files alone. A top
that declares a `ROM_FILE*` parameter also takes its image path -- `-G` for
Verilator, a `chparam -set ... <mod>;` ahead of `hierarchy` for Yosys,
`-P<mod>.<param>` for Icarus (`rtl/cfg/README.md`). The
forbidden-construct greps cover both directories. Two of them are heuristics
with a comment in `scripts/lint.sh` explaining the shape they match: the
unpacked-port check (a port declaration whose identifier is followed by a `[..]`
range) and `unary operator on a bare size cast (write ~(N'(x)); Yosys 0.65 binds
it to N)`, which greps the comment-stripped line with its whitespace removed.

The lint proves the three tools accept the RTL. `make gatesim` proves two of
them read it the same way: Yosys maps each block to Xilinx cells and one Icarus
bench drives the source and that netlist from the same stimulus, comparing every
output every cycle (`sim/gatesim/README.md`).

`make lint TOPS="qcore_row qcore_requant"` restricts the run to named tops.

`make style` is the Python half of the lint: `uv run ruff check .` and `uv run
ruff format --check .` over every `.py` in the repository, at the ruff version
`uv.lock` pins, with the rules and the line length in the `[tool.ruff]` tables
of `pyproject.toml`. Both commands only report; `uv run ruff format .` rewrites.

The glob is `rtl/*.sv` plus `sim/cocotb/wrappers/*.sv`: the generated include
`rtl/qcore_csr_defs.svh` is linted through the modules that include it, checked
against `isa.py` by `uv run quettos csr-defs --check`, and run through all three
parsers on an include wrapper by `sw/tests/test_isa.py` (skipped when the tools
are not on `PATH`).

Run it as you write a module. It is fast.

## 5. Pre-commit hook

```sh
scripts/install-hooks.sh
```

installs `.git/hooks/pre-commit`, which runs `make lint`, `make style` and
`uv run pytest -q sw/tests -x`. The hook only checks; it never rewrites files.
If `uv` is not on PATH the style check and the test step are skipped with a
message.

## 6. Measured numbers (README rule)

- Every performance, synthesis or quality number in `README.md` comes from a
  command in this repository (`make perf`, `make synth`, `make bringup-sweep`,
  `uv run quettos check`) and the row names that command.
- Synthesis figures are never typed by hand: `scripts/synth_report.py` writes
  each `syn/reports/*.md` from the Yosys log of the run that produced it -- the
  tool version, the elaborated parameters, the `stat -tech xilinx` table, the
  hard-block instances with the source line each was inferred from, and the
  `ltp` path. `make synth` regenerates the page and fails when the run no
  longer reproduces it: byte for byte on the Yosys build the report records,
  and on any other build against its parameters and its hard-block inventory,
  since LUT packing and path length belong to the build. One section of a
  report is a person's, `## Notes (hand-written)` at the end, which the
  generator carries forward unchanged; whoever changes a block owns keeping its
  note true.
- A wall-clock figure is a median with its range and the number of runs, and
  it is quoted once, on the page that owns it: `docs/PERFORMANCE.md` for a
  harness run, that tool's own `sim/<tool>/README.md` for a `make` target of
  its own. Everywhere else names the command and points at that page.
- Analytical projections live in `docs/`, labeled **estimate**.
- FPGA tokens/s figures state the clock and memory bandwidth they are derived
  from (100 MHz and 6.4 GB/s for WB=64).
- Measurement conventions (what "bit-exact", "utilization" and "quality" mean
  here) are defined once, in `docs/ARCHITECTURE.md`, and used consistently.

## 7. Tests

- `uv run pytest -q sw/tests` must pass without network access once models are
  cached. Tests that need a downloaded model skip with a clear reason when the
  model is missing.
- New RTL needs either a cocotb block test on the tiny config or a step-mode
  per-op compare against `isa_sim.py`. It also needs a case in
  `sim/gatesim/gatesim.py`, or a line in `sim/gatesim/README.md` saying which
  primitive keeps it out. New numerics need a property test in
  `sw/tests/test_numerics.py`.
- Tests marked `slow` need the quantized models under `build/quant/` (from
  `uv run quettos quantize <alias>`), write the compiled images to
  `build/images/<name>/` and take a few minutes;
  `uv run pytest -q sw/tests -m "not slow"` is the quick loop.
