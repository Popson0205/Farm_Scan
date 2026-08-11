#!/usr/bin/env python3
"""
Cross-platform equivalent of download_sam_weights.sh -- downloads the
MobileSAM pretrained weights (~39MB) needed for the "sam" plot-extraction
method. Not committed to git (see backend/models/.gitkeep) so the repo stays
small. Run this once after cloning, and again after any fresh deploy where
backend/models/ isn't persisted.

Usage:
    python3 backend/scripts/download_sam_weights.py

Safe to re-run -- skips the download if the file already exists.
"""
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt"


def main():
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models"
    dest = models_dir / "mobile_sam.pt"
    models_dir.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"mobile_sam.pt already present at {dest} -- skipping download.")
        return

    print(f"Downloading MobileSAM weights (~39MB) from {URL} ...")

    def _progress(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size)
        sys.stdout.write(f"\r  {pct}% ({downloaded // 1_000_000}MB / {total_size // 1_000_000}MB)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(URL, dest, reporthook=_progress)
        print(f"\nSaved to {dest}")
    except Exception as e:
        print(f"\nDownload failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
