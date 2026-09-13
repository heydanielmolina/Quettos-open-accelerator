#!/usr/bin/env bash
# Installs the Quettos Core pre-commit hook into .git/hooks/pre-commit.
#
# The hook is sized to run on every commit: `make lint` over the RTL, `make
# style` over the Python, and the test files that belong to what is staged --
# `sw/quettos/<name>.py` and `sw/tests/test_<name>.py` are one pair, and a
# staged test file is its own. Complete-model runs (`-m 'not slow'`) stay out of
# it. The whole suite is `make test`, which CI runs on every push.
#
# `make style` and pytest need `uv` and are skipped with a message when it is
# not on PATH. The hook only checks; it never rewrites files and never commits
# anything -- `make style` reports the files `uv run ruff format .` would
# rewrite rather than rewriting them.
set -euo pipefail

cd "$(dirname "$0")/.."
HOOK_DIR="$(git rev-parse --git-path hooks)"
mkdir -p "$HOOK_DIR"
HOOK="$HOOK_DIR/pre-commit"

cat > "$HOOK" <<'HOOK_EOF'
#!/usr/bin/env bash
# Quettos Core pre-commit hook (installed by scripts/install-hooks.sh).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "pre-commit: make lint"
make lint

if ! command -v uv >/dev/null 2>&1; then
  echo "pre-commit: uv not found on PATH; skipping the style check and pytest"
  echo "pre-commit: OK"
  exit 0
fi

echo "pre-commit: make style"
make style

# The test files that belong to what is staged.
TESTS=""
while IFS= read -r f; do
  case $f in
    sw/tests/test_*.py) t=$f ;;
    sw/quettos/*.py)    t="sw/tests/test_$(basename "$f")" ;;
    *)                  continue ;;
  esac
  [ -f "$t" ] || continue
  case " $TESTS " in *" $t "*) ;; *) TESTS="$TESTS $t" ;; esac
done < <(git diff --cached --name-only --diff-filter=ACMR)

if [ -n "$TESTS" ]; then
  echo "pre-commit: uv run pytest -q -x -m 'not slow'$TESTS"
  rc=0
  uv run pytest -q -x -m "not slow" $TESTS || rc=$?
  # 5 is pytest's "nothing ran": every test in those files is a slow one.
  [ "$rc" = 0 ] || [ "$rc" = 5 ] || exit "$rc"
else
  echo "pre-commit: nothing staged that sw/tests covers; make test runs the suite"
fi
echo "pre-commit: OK"
HOOK_EOF

chmod +x "$HOOK"
echo "installed $HOOK"
