#!/usr/bin/env bash
# Three-parser lint for Quettos Core RTL, plus greps for forbidden constructs.
# Run from the repo root: `make lint` or `bash scripts/lint.sh`.
#
# Parsers (all must pass):
#   1. verilator --lint-only -Wall -Wpedantic --top-module $TOP
#   2. yosys read_verilog -sv; hierarchy -check -top $TOP; proc; opt; check -assert
#   3. iverilog -g2012 parse-only
#
# Environment overrides: TOP (default qcore_top), VERILATOR, YOSYS, IVERILOG,
# RTL_DIR (default rtl), ROM_FILE (absolute path handed to the ROM_FILE
# parameter; defaults to rtl/gen/exp2.hex if it exists so lint does not fail on
# the no-default parameter).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

TOP="${TOP:-qcore_top}"
VERILATOR="${VERILATOR:-verilator}"
YOSYS="${YOSYS:-yosys}"
IVERILOG="${IVERILOG:-iverilog}"
RTL_DIR="${RTL_DIR:-rtl}"

shopt -s nullglob
SV_FILES=("$RTL_DIR"/*.sv)
shopt -u nullglob

if [ "${#SV_FILES[@]}" -eq 0 ]; then
  echo "lint: no .sv files under $RTL_DIR/ yet; nothing to lint. OK."
  exit 0
fi

echo "lint: ${#SV_FILES[@]} file(s) under $RTL_DIR/, top module $TOP"

# --- ROM_FILE: the parameter has no default by design; supply an absolute path.
ROM_ARGS_VERILATOR=()
ROM_ARGS_YOSYS=""
ROM_ARGS_IVERILOG=()
if [ -n "${ROM_FILE:-}" ] || [ -f "$REPO_ROOT/rtl/gen/exp2.hex" ]; then
  ROM_FILE="${ROM_FILE:-$REPO_ROOT/rtl/gen/exp2.hex}"
  case "$ROM_FILE" in
    /*) ;;
    *) echo "lint: ROM_FILE must be an absolute path (got '$ROM_FILE')"; exit 1 ;;
  esac
  ROM_ARGS_VERILATOR=(-G"ROM_FILE=\"$ROM_FILE\"")
  ROM_ARGS_YOSYS="-chparam ROM_FILE \"$ROM_FILE\""
  ROM_ARGS_IVERILOG=(-P"$TOP.ROM_FILE=\"$ROM_FILE\"")
fi

FAIL=0

# --- 1. Verilator -------------------------------------------------------------
echo "lint: [1/4] $VERILATOR --lint-only -Wall -Wpedantic"
if ! "$VERILATOR" --lint-only -Wall -Wpedantic --top-module "$TOP" \
     ${ROM_ARGS_VERILATOR[@]+"${ROM_ARGS_VERILATOR[@]}"} "${SV_FILES[@]}"; then
  echo "lint: verilator FAILED"; FAIL=1
fi

# --- 2. Yosys -----------------------------------------------------------------
echo "lint: [2/4] $YOSYS read_verilog -sv; hierarchy -check; proc; opt; check -assert"
YOSYS_LOG="$(mktemp "${TMPDIR:-/tmp}/qcore_lint_yosys.XXXXXX")"
YOSYS_CMD="read_verilog -sv ${SV_FILES[*]}; hierarchy -check -top $TOP $ROM_ARGS_YOSYS; proc; opt; check -assert"
if ! "$YOSYS" -q -l "$YOSYS_LOG" -p "$YOSYS_CMD" >/dev/null 2>&1; then
  echo "lint: yosys FAILED; log follows"
  tail -n 60 "$YOSYS_LOG"
  FAIL=1
fi
rm -f "$YOSYS_LOG"

# --- 3. Icarus ----------------------------------------------------------------
echo "lint: [3/4] $IVERILOG -g2012 parse"
if ! "$IVERILOG" -g2012 -s "$TOP" -o /dev/null \
     ${ROM_ARGS_IVERILOG[@]+"${ROM_ARGS_IVERILOG[@]}"} "${SV_FILES[@]}"; then
  echo "lint: iverilog FAILED"; FAIL=1
fi

# --- 4. Forbidden-construct greps --------------------------------------------
# Each pattern below must produce zero matches across the RTL. Comments are
# stripped (// and /* */) before matching so documentation does not trip it.
echo "lint: [4/4] forbidden-construct greps"

strip_comments() {
  # remove /* ... */ (multi-line) then // to end of line
  sed -e ':a' -e 'N' -e '$!ba' -e 's#/\*[^*]*\*\+\([^/*][^*]*\*\+\)*/##g' "$1" | sed -e 's#//.*$##'
}

grep_fail() {
  # $1 = description, $2 = extended regex
  local desc="$1" re="$2" hits=0 f
  for f in "${SV_FILES[@]}"; do
    if strip_comments "$f" | grep -nE -- "$re" | sed "s#^#$f:#"; then
      hits=1
    fi
  done
  if [ "$hits" -ne 0 ]; then
    echo "lint: FORBIDDEN: $desc"; FAIL=1
  fi
}

# Unpacked array ports. Heuristic: within a module header (ANSI port list) a
# line that starts with input/output/inout and has a `[...]` range AFTER the
# port identifier, i.e. `identifier [` or `identifier[` before the trailing
# `,` or `)`. Packed ranges sit before the identifier (`logic [7:0] x`) so they
# do not match. False negatives are possible for exotic formatting (a range on
# the following line); those are still caught by the Yosys hard error.
grep_fail "unpacked array on module port (flatten to a packed vector)" \
  '^[[:space:]]*(input|output|inout)[[:space:]].*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\[[^]]*\][[:space:]]*(,|\)|$)'
grep_fail "wildcard package import (use explicit qcore_pkg:: scoping)" \
  'import[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*::[[:space:]]*\*'
grep_fail "'return' in a function (assign to the function name instead)" \
  '(^|[^A-Za-z0-9_])return([^A-Za-z0-9_]|$)'
grep_fail "literal \$readmemh(\"...\") (use the ROM_FILE parameter)" \
  '\$readmemh[[:space:]]*\([[:space:]]*"'
grep_fail "negedge sensitivity (sync reset only)" \
  '(^|[^A-Za-z0-9_])negedge([^A-Za-z0-9_]|$)'
grep_fail "asynchronous reset (posedge clk or posedge rst; sync reset only)" \
  '(^|[^A-Za-z0-9_])(pos|neg)edge[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]+or[[:space:]]+(pos|neg)edge'
grep_fail "'unique case'" \
  '(^|[^A-Za-z0-9_])unique[[:space:]]+case'
grep_fail "'priority case'" \
  '(^|[^A-Za-z0-9_])priority[[:space:]]+case'
grep_fail "'interface' construct" \
  '(^|[^A-Za-z0-9_])interface[[:space:]]'
grep_fail "'class' construct" \
  '(^|[^A-Za-z0-9_])class[[:space:]]'
grep_fail "verilator lint waiver (fix the warning instead)" \
  'verilator[[:space:]]+lint_off'

if [ "$FAIL" -ne 0 ]; then
  echo "lint: FAILED"
  exit 1
fi
echo "lint: OK"
