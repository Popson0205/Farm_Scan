#!/usr/bin/env bash
# Downloads the MobileSAM pretrained weights (~39MB) needed for the "sam"
# plot-extraction method. Not committed to git (see backend/models/.gitkeep)
# so the repo stays small enough for GitHub's web-upload UI and doesn't bloat
# git history with a large binary. Run this once after cloning, and again
# after any fresh deploy where backend/models/ isn't persisted.
#
# Usage:
#   bash backend/scripts/download_sam_weights.sh
#
# Safe to re-run — skips the download if the file already exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
DEST="$MODELS_DIR/mobile_sam.pt"
URL="https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt"

mkdir -p "$MODELS_DIR"

if [ -f "$DEST" ]; then
  echo "mobile_sam.pt already present at $DEST — skipping download."
  exit 0
fi

echo "Downloading MobileSAM weights (~39MB) from $URL ..."
if command -v curl >/dev/null 2>&1; then
  curl -fL --progress-bar -o "$DEST" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -q --show-progress -O "$DEST" "$URL"
else
  echo "Error: need curl or wget installed to download the weights." >&2
  exit 1
fi

echo "Saved to $DEST"
echo "Verify: python3 -c \"import sys; sys.path.insert(0,'$SCRIPT_DIR/..'); import sam_plots; print(sam_plots.sam_available())\""
