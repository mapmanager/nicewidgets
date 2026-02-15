#!/usr/bin/env bash
# Install git hooks for large-file check.
# Run once after cloning (or after creating a new repo) to enable pre-commit and pre-push checks.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"
SCRIPT_DIR="$REPO_ROOT/scripts"

if [ ! -f "$SCRIPT_DIR/git-large-file-check.sh" ]; then
  echo "Error: $SCRIPT_DIR/git-large-file-check.sh not found" >&2
  exit 1
fi

chmod +x "$SCRIPT_DIR/git-large-file-check.sh"

# pre-commit: check staged files before commit
cat > "$HOOKS_DIR/pre-commit" << 'HOOK'
#!/usr/bin/env bash

# Call the shared checker script in "staged" mode
REPO_ROOT="$(git rev-parse --show-toplevel)"
exec "$REPO_ROOT/scripts/git-large-file-check.sh" staged
HOOK

# pre-push: check all tracked files before push
cat > "$HOOKS_DIR/pre-push" << 'HOOK'
#!/usr/bin/env bash

# Call the shared checker script in "tracked" mode
REPO_ROOT="$(git rev-parse --show-toplevel)"
exec "$REPO_ROOT/scripts/git-large-file-check.sh" tracked
HOOK

chmod +x "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-push"

echo "✓ Git hooks installed:"
echo "  - pre-commit  (checks staged files on git commit)"
echo "  - pre-push    (checks all tracked files on git push)"
echo "  Limit: 60MB per file"
