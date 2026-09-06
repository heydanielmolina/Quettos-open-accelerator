#!/usr/bin/env bash
# Three-parser lint for Quettos Core RTL, plus greps for forbidden constructs.
# Run from the repo root: `make lint` or `bash scripts/lint.sh`.
#
# Every rtl/*.sv file and every sim/cocotb/wrappers/*.sv assembly is linted as
# its own top, with the files it needs on the command line so that submodules
# and the package resolve:
#   1. verilator --lint-only -Wall -Wpedantic --top-module <top>   (zero warnings)
#   2. yosys read_verilog -sv -defer; [chparam -set ROM_FILE ...;] hierarchy -check -top <top>; proc; opt; check -assert
#   3. iverilog -g2012 -s <top>
# A package file (qcore_pkg.sv) is elaborated through an empty wrapper module
# for Icarus and without -top for Yosys. ROM image parameters (ROM_FILE,
# ROM_FILE_<TABLE>: untyped, empty default) receive absolute paths under
# rtl/gen/: -G for Verilator, -P for Icarus, and for Yosys the chparam pass
# on a deferred read (`read_verilog -defer`), the one form of Yosys 0.65 that
# takes a string value before $readmemh runs.
#
# The wrapper assemblies under sim/cocotb/wrappers/ are block-level tops built
# from the same modules; they are linted with the rtl/*.sv files plus the
# wrapper on the command line, and the forbidden-construct greps cover them.
#
# Environment overrides: TOPS (space-separated subset of tops; default all),
# VERILATOR, YOSYS, IVERILOG, RTL_DIR (default rtl), WRAP_DIR (default
# sim/cocotb/wrappers), ROM_FILE (absolute path used for every ROM_FILE
# parameter; default the matching rtl/gen/<table>.hex).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

VERILATOR="${VERILATOR:-verilator}"
YOSYS="${YOSYS:-yosys}"
IVERILOG="${IVERILOG:-iverilog}"
RTL_DIR="${RTL_DIR:-rtl}"
WRAP_DIR="${WRAP_DIR:-sim/cocotb/wrappers}"

shopt -s nullglob
# The package is listed first: Yosys and Icarus resolve qcore_pkg:: references
# only after the package has been parsed.
SV_FILES=()
for f in "$RTL_DIR"/*.sv; do
  case "$(basename "$f")" in
    qcore_pkg.sv) SV_FILES=("$f" ${SV_FILES[@]+"${SV_FILES[@]}"}) ;;
    *) SV_FILES+=("$f") ;;
  esac
done
WRAP_FILES=()
for f in "$WRAP_DIR"/*.sv; do
  WRAP_FILES+=("$f")
done
shopt -u nullglob

if [ "${#SV_FILES[@]}" -eq 0 ]; then
  echo "lint: no .sv files under $RTL_DIR/ yet; nothing to lint. OK."
  exit 0
fi

# Every file the greps cover, and the list a wrapper top is elaborated with.
ALL_FILES=("${SV_FILES[@]}" ${WRAP_FILES[@]+"${WRAP_FILES[@]}"})

if [ -n "${TOPS:-}" ]; then
  read -r -a TOP_LIST <<< "$TOPS"
else
  TOP_LIST=()
  for f in "${ALL_FILES[@]}"; do
    TOP_LIST+=("$(basename "$f" .sv)")
  done
fi

echo "lint: ${#SV_FILES[@]} file(s) under $RTL_DIR/, ${#WRAP_FILES[@]} under $WRAP_DIR/, ${#TOP_LIST[@]} top(s)"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/qcore_lint.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
WRAP="$WORK/qcore_lint_wrap.sv"
echo "module qcore_lint_wrap; endmodule" > "$WRAP"

# --- ROM images: every `parameter [string] ROM_FILE*` of the top gets an absolute path.
rom_path() {
  # $1 = parameter name -> absolute .hex path
  local name="$1" table
  if [ -n "${ROM_FILE:-}" ]; then
    case "$ROM_FILE" in
      /*) echo "$ROM_FILE"; return ;;
      *) echo "lint: ROM_FILE must be an absolute path (got '$ROM_FILE')" >&2; exit 1 ;;
    esac
  fi
  case "$name" in
    ROM_FILE_*) table="$(echo "${name#ROM_FILE_}" | tr '[:upper:]' '[:lower:]')" ;;
    *) table="exp2" ;;
  esac
  echo "$REPO_ROOT/rtl/gen/$table.hex"
}

string_params() {
  # $1 = file -> names of the ROM image parameters declared in it, one per line
  sed -n -E 's/^[[:space:]]*parameter[[:space:]]+(string[[:space:]]+)?(ROM_FILE[A-Za-z0-9_]*).*/\2/p' "$1"
}

is_package() {
  grep -qE '^[[:space:]]*package[[:space:]]+[A-Za-z_]' "$1" && ! grep -qE '^[[:space:]]*module[[:space:]]+[A-Za-z_]' "$1"
}

FAIL=0

for top in "${TOP_LIST[@]}"; do
  # A top is either an rtl/ module or a wrapper assembly; a wrapper is
  # elaborated with the rtl files as well.
  if [ -f "$RTL_DIR/$top.sv" ]; then
    file="$RTL_DIR/$top.sv"
    FILES=("${SV_FILES[@]}")
  elif [ -f "$WRAP_DIR/$top.sv" ]; then
    file="$WRAP_DIR/$top.sv"
    FILES=("${ALL_FILES[@]}")
  else
    echo "lint: $top: no $RTL_DIR/$top.sv or $WRAP_DIR/$top.sv"; FAIL=1; continue
  fi
  ROM_V=()
  ROM_Y=""
  ROM_I=()
  while IFS= read -r pname; do
    [ -z "$pname" ] && continue
    path="$(rom_path "$pname")"
    ROM_V+=(-G"$pname=\"$path\"")
    ROM_Y="$ROM_Y chparam -set $pname \"$path\" $top;"
    ROM_I+=(-P"$top.$pname=\"$path\"")
  done < <(string_params "$file")

  if is_package "$file"; then
    # 1. Verilator elaborates the package as the top unit; 2. Yosys parses it;
    # 3. Icarus needs a module: the empty wrapper.
    echo "lint: [$top] package: verilator / yosys / iverilog"
    if ! "$VERILATOR" --lint-only -Wall -Wpedantic --top-module "$top" -I"$RTL_DIR" "${FILES[@]}"; then
      echo "lint: $top: verilator FAILED"; FAIL=1
    fi
    YCMD="read_verilog -sv -defer -I$RTL_DIR $file; hierarchy -check; proc; opt; check -assert"
    if ! "$YOSYS" -q -l "$WORK/$top.yosys.log" -p "$YCMD" >/dev/null 2>&1; then
      echo "lint: $top: yosys FAILED; log follows"; tail -n 40 "$WORK/$top.yosys.log"; FAIL=1
    fi
    if ! "$IVERILOG" -g2012 -I"$RTL_DIR" -s qcore_lint_wrap -o /dev/null "$file" "$WRAP"; then
      echo "lint: $top: iverilog FAILED"; FAIL=1
    fi
    continue
  fi

  echo "lint: [$top] verilator / yosys / iverilog"
  if ! "$VERILATOR" --lint-only -Wall -Wpedantic --top-module "$top" -I"$RTL_DIR" \
       ${ROM_V[@]+"${ROM_V[@]}"} "${FILES[@]}"; then
    echo "lint: $top: verilator FAILED"; FAIL=1
  fi
  YCMD="read_verilog -sv -defer -I$RTL_DIR ${FILES[*]};$ROM_Y hierarchy -check -top $top; proc; opt; check -assert"
  if ! "$YOSYS" -q -l "$WORK/$top.yosys.log" -p "$YCMD" >/dev/null 2>&1; then
    echo "lint: $top: yosys FAILED; log follows"; tail -n 40 "$WORK/$top.yosys.log"; FAIL=1
  fi
  if ! "$IVERILOG" -g2012 -I"$RTL_DIR" -s "$top" -o /dev/null \
       ${ROM_I[@]+"${ROM_I[@]}"} "${FILES[@]}"; then
    echo "lint: $top: iverilog FAILED"; FAIL=1
  fi
done

# --- Forbidden-construct greps --------------------------------------------
# Each pattern below must produce zero matches across the RTL. Comments are
# stripped (// and /* */) before matching so documentation does not trip it.
echo "lint: forbidden-construct greps"

strip_comments() {
  # remove /* ... */ (multi-line) then // to end of line
  sed -e ':a' -e 'N' -e '$!ba' -e 's#/\*[^*]*\*\+\([^/*][^*]*\*\+\)*/##g' "$1" | sed -e 's#//.*$##'
}

grep_fail() {
  # $1 = description, $2 = extended regex
  local desc="$1" re="$2" hits=0 f
  for f in "${ALL_FILES[@]}"; do
    if strip_comments "$f" | grep -nE -- "$re" | sed "s#^#$f:#"; then
      hits=1
    fi
  done
  if [ "$hits" -ne 0 ]; then
    echo "lint: FORBIDDEN: $desc"; FAIL=1
  fi
}

grep_fail_packed() {
  # As grep_fail, with every space removed from the line first, so a pattern can
  # name the token that precedes an operator without spelling out the spacing.
  local desc="$1" re="$2" hits=0 f
  for f in "${ALL_FILES[@]}"; do
    if strip_comments "$f" | sed -e 's/[[:space:]]//g' | grep -nE -- "$re" | sed "s#^#$f:#"; then
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
# A unary operator applied to a bare size cast: Yosys 0.65 binds the operator to
# the size literal instead of the cast, so `x & ~25'(WB-1)` synthesizes to
# `x & 25'(WB-1)` while Verilator and Icarus complement the mask -- the netlist
# then computes different values from the simulation. Parenthesize the cast:
# `~(25'(WB-1))`. The operator is in unary position when the token before it is
# not the end of an operand, which is what the leading class tests.
grep_fail_packed "unary operator on a bare size cast (write ~(N'(x)); Yosys 0.65 binds it to N)" \
  "(^|[^]A-Za-z0-9_)}])[-+~&|^!]\`?[0-9A-Za-z_]+'\\("

if [ "$FAIL" -ne 0 ]; then
  echo "lint: FAILED"
  exit 1
fi
echo "lint: OK"
