#!/usr/bin/env bash
# Installs the Quettos Core pre-commit hook into .git/hooks/pre-commit.
# The hook runs the three checks CI runs on every push: `make lint`,
# `make style` and `uv run pytest -q sw/tests -x`. The last two need `uv` and
# are skipped with a message when it is not on PATH. It only checks; it never
# rewrites files and never commits anything -- `make style` reports the files
# `uv run ruff format .` would rewrite rather than rewriting them.
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
  echo "pre-commit: make style"
  make style
  if [ -d sw/tests ]; then
    echo "pre-commit: uv run pytest -q sw/tests -x"
    uv run pytest -q sw/tests -x
  else
    echo "pre-commit: sw/tests not present yet; skipping pytest"
  fi
else
  echo "pre-commit: uv not found on PATH; skipping the style check and pytest"
fi
echo "pre-commit: OK"
HOOK_EOF

chmod +x "$HOOK"
echo "installed $HOOK"
