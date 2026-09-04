#!/usr/bin/env bash
# Installs the Quettos Core pre-commit hook into .git/hooks/pre-commit.
# The hook runs `make lint` and `uv run pytest -q sw/tests -x` (pytest is
# skipped with a message when `uv` is not on PATH). It only checks; it never
# rewrites files and never commits anything.
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

if command -v uv >/dev/null 2>&1; then
  if [ -d sw/tests ]; then
    echo "pre-commit: uv run pytest -q sw/tests -x"
    uv run pytest -q sw/tests -x
  else
    echo "pre-commit: sw/tests not present yet; skipping pytest"
  fi
else
  echo "pre-commit: uv not found on PATH; skipping pytest"
fi
echo "pre-commit: OK"
HOOK_EOF

chmod +x "$HOOK"
echo "installed $HOOK"
