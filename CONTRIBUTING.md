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
- CI must be green (`make lint`, `uv run pytest -q sw/tests`, and the RTL jobs
  once they exist) before a PR is reviewed.
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
- ROM initialization only through a `ROM_FILE` string parameter with **no
  default**, set to an **absolute path** by the Makefile / `.ys` script:
  `$readmemh(ROM_FILE, mem);`.

### Forbidden (each one is a lint failure or a Yosys hard error)

- **Unpacked arrays on module ports.** Flatten to a packed vector and index with
  `qcore_pkg::` helper functions. (Yosys 0.65 hard error.)
- **Wildcard imports** (`import qcore_pkg::*;`). Always write `qcore_pkg::NAME`.
- **`return` inside functions.** Assign to the function name instead.
- **Asynchronous reset** or any `negedge` sensitivity. Synchronous reset only:
  `always_ff @(posedge clk) if (rst) ... else ...`.
- **`unique case` / `priority case`**, `unique if`, `priority if`.
- **`interface`**, **`class`**, `modport`, `program`, `clocking`, `assert
  property`, `always_latch`, `initial` blocks in synthesizable modules.
- Literal **`$readmemh("...")`** with a string constant (CI greps for it).
- Multi-dimensional unpacked memories, `automatic` variables, `real`,
  `string` variables (a `string` *parameter* for `ROM_FILE` is the one
  exception), `$clog2` on anything but constants, dynamic arrays, queues.
- Lint waivers of any kind (`/* verilator lint_off */`). `UNOPTFLAT` must be
  fixed, not silenced.
- Any fixed-point format knowledge in RTL. **Numerics conventions live only in
  `sw/quettos/numerics.py`**; every shift, exponent and constant reaches the
  RTL as a descriptor field. If you want to change rounding, change
  `numerics.py`, regenerate, and let the tests tell you what broke.

### Style

- Two-space indent, `snake_case` identifiers, `UPPER_CASE` parameters and
  localparams, one module per file, file name equals module name.
- Every module header comment states the module's contract in a few lines:
  inputs, outputs, latency, and what it does not do.
- Keep modules small enough to lint individually. No file over ~1,000 lines.

## 4. Three-parser lint recipe

`make lint` runs `scripts/lint.sh`, which for every `rtl/*.sv` runs:

```sh
verilator --lint-only -Wall -Wpedantic --top-module qcore_top rtl/*.sv
yosys -q -p "read_verilog -sv rtl/*.sv; hierarchy -check -top qcore_top; proc; opt; check -assert"
iverilog -g2012 -o /dev/null rtl/*.sv
```

followed by greps that fail on the forbidden constructs listed above. The
unpacked-port check is a heuristic (a port declaration whose identifier is
followed by a `[..]` range); see the comment in `scripts/lint.sh`.

Run it on the day you write a module. It is fast.

## 5. Pre-commit hook

```sh
scripts/install-hooks.sh
```

installs `.git/hooks/pre-commit`, which runs `make lint` and
`uv run pytest -q sw/tests -x`. The hook only checks; it never rewrites files.
If `uv` is not on PATH the pytest step is skipped with a message.

## 6. Measured numbers (README rule)

- Every performance, synthesis or quality number in `README.md` comes from a
  command in this repository (`make perf`, `make synth`, `uv run quettos check`)
  and the row names that command.
- Analytical projections live in `docs/`, labeled **estimate**.
- FPGA tokens/s figures state the clock and memory bandwidth they are derived
  from (100 MHz and 6.4 GB/s for WB=64).
- Measurement conventions (what "bit-exact", "utilization" and "quality" mean
  here) are defined once, in `docs/ARCHITECTURE.md`, and used consistently.

## 7. Tests

- `uv run pytest -q sw/tests` must pass without network access once models are
  cached. Tests that need a downloaded model skip with a clear reason when the
  model is missing.
- New RTL needs either a cocotb unit test on the tiny config or a step-mode
  per-op compare against `isa_sim.py`. New numerics need a property test in
  `sw/tests/test_numerics.py`.
- Tests marked `slow` need the quantized models under `build/quant/` (from
  `uv run quettos quantize <alias>`) and take a few minutes;
  `uv run pytest -q sw/tests -m "not slow"` is the quick loop.
