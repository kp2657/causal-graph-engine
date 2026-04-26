#!/usr/bin/env python3
"""
fetch_static_data.py — One-shot downloader for static reference datasets.

These files change only on periodic public releases (gnomAD ~yearly,
HGNC daily but idempotent, Reactome quarterly).  Having them on disk
lets `pipelines.static_lookups` answer per-gene queries without HTTP,
which cuts thousands of round-trips per pipeline run.

Usage:
    python scripts/fetch_static_data.py               # fetch all, skip existing
    python scripts/fetch_static_data.py --force       # re-download everything
    python scripts/fetch_static_data.py --datasets gnomad,hgnc
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(os.getenv("STATIC_DATA_DIR", ROOT / "data" / "static"))

# Public releases — update versions here when bumping.
DATASETS: dict[str, dict] = {
    "gnomad": {
        "url":   "https://storage.googleapis.com/gcp-public-data--gnomad/"
                 "release/4.1/constraint/gnomad.v4.1.constraint_metrics.tsv",
        "path":  STATIC_DIR / "gnomad_constraint.tsv",
        "bytes": 5_000_000,   # approximate; used only for progress display
        "about": "gnomAD v4.1 per-gene constraint (pLI, LOEUF, missense-z)",
    },
    "hgnc": {
        "url":   "https://storage.googleapis.com/public-download-files/hgnc/"
                 "tsv/tsv/hgnc_complete_set.txt",
        "path":  STATIC_DIR / "hgnc_complete.tsv",
        "bytes": 7_000_000,
        "about": "HGNC canonical symbol ↔ Ensembl gene ID ↔ HGNC ID mapping",
    },
    "reactome": {
        "url":   "https://reactome.org/download/current/Ensembl2Reactome_All_Levels.txt",
        "path":  STATIC_DIR / "reactome_ensembl2pathways.tsv",
        "bytes": 180_000_000,
        "about": ("Reactome pathway membership keyed by Ensembl gene ID "
                  "(All Levels: leaf + parent pathways, matches Reactome"
                  " content-service cluster mode)"),
    },
}


def _fetch_one(name: str, spec: dict, force: bool) -> bool:
    target: Path = spec["path"]
    if target.exists() and not force:
        print(f"[skip] {name:10s} exists: {target}  ({target.stat().st_size:,} B)")
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {name:10s} → {target}")
    print(f"         {spec['about']}")
    print(f"         {spec['url']}")

    tmp = target.with_suffix(target.suffix + ".part")
    req = Request(spec["url"], headers={"User-Agent": "causal-graph-engine/0.2 static-data-fetcher"})
    try:
        with urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            chunk = 1 << 16
            total = 0
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                total += len(buf)
        tmp.rename(target)
        print(f"         downloaded {total:,} B")
        # Write companion metadata for reproducibility auditing.
        meta = {
            "source":        spec["about"],
            "url":           spec["url"],
            "download_date": datetime.date.today().isoformat(),
            "size_bytes":    total,
        }
        target.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        return True
    except Exception as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        print(f"         ERROR: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if file exists")
    ap.add_argument(
        "--datasets",
        default=",".join(DATASETS.keys()),
        help=f"comma-separated subset of {list(DATASETS.keys())}",
    )
    args = ap.parse_args()

    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in wanted if d not in DATASETS]
    if unknown:
        print(f"unknown datasets: {unknown}", file=sys.stderr)
        return 2

    print(f"Static data directory: {STATIC_DIR}")
    print(f"Datasets: {wanted}")
    print()

    all_ok = True
    for name in wanted:
        ok = _fetch_one(name, DATASETS[name], force=args.force)
        all_ok = all_ok and ok
        print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
