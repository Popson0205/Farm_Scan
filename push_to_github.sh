#!/usr/bin/env bash
# Initializes (if needed) and pushes this repo to GitHub.
#
# Usage:
#   bash push_to_github.sh <github-repo-url> [branch] [commit-message] [--force]
#
# Examples:
#   bash push_to_github.sh https://github.com/yourname/farmscan.git
#   bash push_to_github.sh git@github.com:yourname/farmscan.git main "Initial commit"
#   bash push_to_github.sh https://github.com/yourname/farmscan.git main "Update" --force
#
# Requirements: git installed and authenticated (HTTPS: a GitHub personal
# access token when prompted, or SSH: your key already added to GitHub).
# The MobileSAM weights are intentionally NOT included (see .gitignore) --
# run backend/scripts/download_sam_weights.py after cloning to fetch them.
#
# UPDATING AN EXISTING REPO: if this folder was freshly extracted (e.g. from
# a new zip) and has no git history of its own, but the target GitHub repo
# already has commits (from an earlier push), a normal push will be
# rejected as "non-fast-forward" -- git has no way to know the new folder
# is a newer version of the same project rather than an unrelated one. This
# script detects that case automatically and merges the histories
# (--allow-unrelated-histories) so your existing commit history is kept.
# Pass --force instead if you'd rather just overwrite the remote entirely
# (destructive: discards any commits only present on GitHub).

set -euo pipefail

REPO_URL="${1:-}"
BRANCH="${2:-main}"
COMMIT_MSG="${3:-Update}"
FORCE=false
for arg in "$@"; do
  [ "$arg" = "--force" ] && FORCE=true
done

if [ -z "$REPO_URL" ]; then
  echo "Usage: bash push_to_github.sh <github-repo-url> [branch] [commit-message] [--force]" >&2
  echo "Example: bash push_to_github.sh https://github.com/yourname/farmscan.git" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "== Working directory: $SCRIPT_DIR =="

# --- Sanity checks -----------------------------------------------------
if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is not installed." >&2
  exit 1
fi

# Warn (don't fail) if the SAM weights somehow ended up present -- they
# should stay out of git per .gitignore, but double-check just in case
# someone re-added them locally, since a 39MB binary in a commit defeats
# the point of keeping this repo small.
if [ -f "backend/models/mobile_sam.pt" ]; then
  echo "Note: backend/models/mobile_sam.pt exists locally but is gitignored --"
  echo "      it will NOT be pushed. That's expected; see README.md."
fi

# --- Init repo if needed -------------------------------------------------
FRESH_REPO=false
if [ ! -d ".git" ]; then
  echo "== No git repo found -- running 'git init' =="
  git init
  git branch -M "$BRANCH"
  FRESH_REPO=true
else
  echo "== Existing git repo found =="
  git branch -M "$BRANCH" 2>/dev/null || true
fi

# --- Configure remote ------------------------------------------------------
if git remote get-url origin >/dev/null 2>&1; then
  CURRENT_URL="$(git remote get-url origin)"
  if [ "$CURRENT_URL" != "$REPO_URL" ]; then
    echo "== Updating existing 'origin' remote: $CURRENT_URL -> $REPO_URL =="
    git remote set-url origin "$REPO_URL"
  else
    echo "== 'origin' remote already set to $REPO_URL =="
  fi
else
  echo "== Adding 'origin' remote: $REPO_URL =="
  git remote add origin "$REPO_URL"
fi

# --- Stage, commit ---------------------------------------------------------
echo "== Staging files =="
git add -A

if git diff --cached --quiet 2>/dev/null; then
  echo "== Nothing to commit (working tree matches last commit) =="
else
  echo "== Committing: \"$COMMIT_MSG\" =="
  git commit -m "$COMMIT_MSG"
fi

# --- Push, handling the "existing remote repo + fresh local history" case --
echo "== Pushing to origin/$BRANCH =="

if [ "$FORCE" = true ]; then
  echo "-- --force passed: overwriting remote branch entirely --"
  git push -u --force origin "$BRANCH"
elif git push -u origin "$BRANCH" 2>/tmp/push_error.log; then
  cat /tmp/push_error.log >&2 || true
else
  ERR="$(cat /tmp/push_error.log)"
  echo "$ERR" >&2
  if echo "$ERR" | grep -qi "rejected\|fetch first\|non-fast-forward"; then
    if [ "$FRESH_REPO" = true ]; then
      echo ""
      echo "== Push rejected: the remote repo already has commits this local"
      echo "   folder doesn't (expected for a freshly-extracted project)."
      echo "   Merging histories with --allow-unrelated-histories to keep both =="
      git fetch origin "$BRANCH"
      git merge "origin/$BRANCH" --allow-unrelated-histories --no-edit -X ours
      git push -u origin "$BRANCH"
    else
      echo ""
      echo "== Push rejected. Pulling remote changes first =="
      git pull origin "$BRANCH" --no-edit
      git push -u origin "$BRANCH"
    fi
  else
    echo "Push failed for a reason other than a history conflict -- see error above." >&2
    exit 1
  fi
fi
rm -f /tmp/push_error.log

echo ""
echo "Done. Pushed to $REPO_URL ($BRANCH)."
echo "Reminder: after cloning elsewhere, run"
echo "  python3 backend/scripts/download_sam_weights.py"
echo "to fetch the MobileSAM weights (not stored in git)."

