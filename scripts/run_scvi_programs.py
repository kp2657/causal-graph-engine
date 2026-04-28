"""
scripts/run_scvi_programs.py — Run scVI program extraction in the dedicated scvi-env.

Must be run from the scvi-env (NOT the main causal-graph env):

    conda activate scvi-env
    python scripts/run_scvi_programs.py --disease coronary_artery_disease --n_latent 10

Writes program JSON to data/cnmf_programs/{cell_type}_scvi_programs.json,
which get_programs_for_disease() picks up automatically when the key
"source": "scVI_*_h5ad" is present.

Setup (one-time):
    conda create -n scvi-env python=3.11
    conda activate scvi-env
    pip install scvi-tools anndata scanpy scipy numpy

Why a separate env: torch 2.2.x (only version available on arm64 Rosetta2)
conflicts with NumPy 2.x at the ABI level. The main causal-graph env needs
NumPy >= 2 for scanpy 1.12. scvi-env pins NumPy < 2 to satisfy torch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scVI program extraction")
    parser.add_argument("--disease", required=True,
                        help="Disease key, e.g. coronary_artery_disease")
    parser.add_argument("--h5ad", default=None,
                        help="Path to h5ad file (auto-detected from disease_registry if omitted)")
    parser.add_argument("--n_latent", type=int, default=10,
                        help="Number of scVI latent dimensions (= number of programs)")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--min_genes", type=int, default=10)
    parser.add_argument("--max_genes", type=int, default=200)
    parser.add_argument("--output_dir", default="./data/cnmf_programs")
    parser.add_argument("--cell_type", default=None,
                        help="Label for program IDs (auto-derived from disease if omitted)")
    args = parser.parse_args()

    # Auto-detect h5ad path from disease registry
    h5ad_path = args.h5ad
    if h5ad_path is None:
        try:
            from models.disease_registry import DISEASE_REGISTRY
            reg = DISEASE_REGISTRY.get(args.disease, {})
            h5ad_files = reg.get("h5ad_files", [])
            if h5ad_files:
                h5ad_path = h5ad_files[0]
            else:
                print(f"ERROR: no h5ad found for {args.disease} in disease_registry")
                sys.exit(1)
        except ImportError:
            print("ERROR: disease_registry not importable — provide --h5ad directly")
            sys.exit(1)

    cell_type = args.cell_type or args.disease.replace("_", " ")

    print(f"Running scVI program extraction:")
    print(f"  h5ad:      {h5ad_path}")
    print(f"  n_latent:  {args.n_latent}")
    print(f"  max_epochs:{args.max_epochs}")
    print(f"  output:    {args.output_dir}")

    from pipelines.cnmf_programs import run_scvi_program_extraction
    result = run_scvi_program_extraction(
        h5ad_path=h5ad_path,
        n_latent=args.n_latent,
        max_epochs=args.max_epochs,
        min_genes_per_program=args.min_genes,
        max_genes_per_program=args.max_genes,
        output_dir=args.output_dir,
        cell_type=cell_type,
    )

    if result.get("status") == "error":
        print(f"ERROR: {result['message']}")
        sys.exit(1)

    print(f"\nDone: {result['n_programs']} programs extracted")
    for prog in result["programs"]:
        print(f"  {prog['program_id']}: {prog['n_genes']} genes — {prog['top_genes'][:5]}")

    out_path = Path(args.output_dir) / f"{cell_type}_scvi_programs.json"
    print(f"\nCache written: {out_path}")


if __name__ == "__main__":
    main()
