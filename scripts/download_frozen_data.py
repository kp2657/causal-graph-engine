#!/usr/bin/env python3
"""
download_frozen_data.py — Download frozen reference files for exact result reproduction.

Downloads four files from the GitHub release assets:
  - data/api_cache.sqlite                        (46 MB) — frozen OT L2G, GTEx, gnomAD API responses
  - data/gps_bgrd/BGRD__size500.pkl              (11 MB) — GPS null distribution (BGRD)
  - data/checkpoints/coronary_artery_disease__tier3.json  (5.6 MB) — CAD Tier 3 checkpoint
  - data/checkpoints/rheumatoid_arthritis__tier3.json     (9.2 MB) — RA Tier 3 checkpoint

Without these files:
  - API calls hit live endpoints → OT L2G scores may differ from published results
  - GPS BGRD is recomputed from scratch (~1 hour)
  - run_tier4 requires the full pipeline re-run (Tiers 1–3) to regenerate checkpoints

Usage:
    python scripts/download_frozen_data.py
    python scripts/download_frozen_data.py --release v0.2.3
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# GitHub release URL base — update tag when cutting a new release
_GH_RELEASE_BASE = "https://github.com/{owner}/{repo}/releases/download/{tag}"
_OWNER = "kp2657"        # update to actual GitHub username/org
_REPO  = "causal-graph-engine"
_DEFAULT_TAG = "v0.2.3"

_ASSETS = [
    {
        "name":    "api_cache.sqlite",
        "dest":    ROOT / "data" / "api_cache.sqlite",
        "size_mb": 46,
    },
    {
        "name":    "BGRD__size500.pkl",
        "dest":    ROOT / "data" / "gps_bgrd" / "BGRD__size500.pkl",
        "size_mb": 11,
    },
    {
        "name":    "coronary_artery_disease__tier3.json",
        "dest":    ROOT / "data" / "checkpoints" / "coronary_artery_disease__tier3.json",
        "size_mb": 6,
    },
    {
        "name":    "rheumatoid_arthritis__tier3.json",
        "dest":    ROOT / "data" / "checkpoints" / "rheumatoid_arthritis__tier3.json",
        "size_mb": 9,
    },
]


def _download(url: str, dest: Path, size_mb: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name} (~{size_mb} MB) …", flush=True)
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress(size_mb))
    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        print(f"  URL: {url}")
        print("  Download the file manually and place it at:")
        print(f"    {dest}")
        return
    print(f"  → {dest}")


def _progress(size_mb: int):
    def hook(count, block, total):
        if total > 0:
            pct = min(100, count * block * 100 // total)
        else:
            pct = min(100, count * block // (size_mb * 1024 * 1024) * 100)
        print(f"\r  {pct:3d}%", end="", flush=True)
    return hook


def run(tag: str) -> None:
    base = _GH_RELEASE_BASE.format(owner=_OWNER, repo=_REPO, tag=tag)
    print(f"Release: {base}\n")

    for asset in _ASSETS:
        dest = asset["dest"]
        if dest.exists():
            print(f"  [SKIP] {dest.name} already present at {dest}")
            continue
        url = f"{base}/{asset['name']}"
        _download(url, dest, asset["size_mb"])

    print("\nDone. Run python scripts/check_data.py to verify all data is in place.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default=_DEFAULT_TAG,
                        help=f"GitHub release tag (default: {_DEFAULT_TAG})")
    args = parser.parse_args()
    run(args.release)
