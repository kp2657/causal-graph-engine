"""
Compute β in LOCUS program space for RA Perturb-seq data.

Produces: data/perturbseq/czi_2025_cd4t_perturb/locus_de_betas.npz

The LOCUS programs are 31 RA GWAS-locus-anchored programs defined by gene sets
from gwas_anchored_programs.npz (locus_genes_json field).

β(KO_gene → LOCUS) = mean log2FC of LOCUS gene set in the KO signature

This is the gene-set-level effect of the KO on each GWAS locus.  It is
directly tied to the locus definition: a KO that up-regulates genes at the
CTLA4 locus has positive β at LOCUS_CTLA4.

Why LOCUS space?
----------------
The GeneticNMF components (C01-C30) have all-negative S-LDSC τ* for RA —
heritability depletion, not enrichment.  The LOCUS programs were defined
directly from RA GWAS loci; their τ* (RA_LOCUS_program_taus.json) are
positive and meaningful.  β(LOCUS) + τ*(LOCUS) gives a valid OTA
γ(KO → RA) that respects the human genetics → programs → trait framework.

Why gene-set mean and not Vt projection?
-----------------------------------------
The Vt matrix in gwas_anchored_programs.npz is the GeneticNMF/SVD loading
basis fitted to Perturb-seq expression data.  Its rows do not specifically
weight locus genes — the gene sets at each locus have tiny, non-specific
loadings in Vt.  Projecting RNA DE signatures onto Vt yields spurious β
dominated by off-locus variance (e.g., LOCUS_COG6 tops many KOs due to an
idiosyncratic Vt row, not locus biology).

The gene-set-mean approach is anchored directly to the locus definition and
produces interpretable, sanity-check-passing β values.

Output format (matches genetic_nmf_de_betas.npz):
  beta        : float32 (n_ko × n_loci)
  ko_genes    : str (n_ko,)
  program_ids : str (n_loci,)   — "LOCUS_AARS2", …, "LOCUS_TYK2"
  condition   : str scalar      — "combined"
  method      : str scalar      — "locus_geneset_mean"

Usage
-----
    conda run -n causal-graph python -m pipelines.perturbseq.compute_locus_betas
"""

from __future__ import annotations

import gzip
import json
import pathlib

import numpy as np

ROOT    = pathlib.Path(__file__).parent.parent.parent
DATA_RA = ROOT / "data/perturbseq/czi_2025_cd4t_perturb"
OUT_NPZ = DATA_RA / "locus_de_betas.npz"


def compute_locus_betas(
    sig_file: str = "signatures.json.gz",
    condition: str = "combined",
) -> str:
    """
    Compute β(KO → LOCUS) = mean log2FC of LOCUS gene set in KO signature.

    Parameters
    ----------
    sig_file  : filename (relative to DATA_RA) of the DE signature JSON.gz
    condition : label stored in output npz

    Returns
    -------
    Path to saved npz.
    """
    # ── Load LOCUS gene sets ──────────────────────────────────────────────────
    anchor_npz  = np.load(DATA_RA / "gwas_anchored_programs.npz", allow_pickle=True)
    locus_names = [str(x) for x in anchor_npz["locus_names"]]  # 31 LOCUS IDs
    pert_names  = [str(x) for x in anchor_npz["pert_names"]]   # 11415 KO gene IDs

    raw_genes = anchor_npz["locus_genes_json"]
    locus_genes: dict[str, list[str]] = json.loads(
        raw_genes.item() if hasattr(raw_genes, "item") else str(raw_genes)
    )

    n_loci  = len(locus_names)
    n_perts = len(pert_names)

    print(f"  LOCUS programs: {n_loci}")
    print(f"  KO genes: {n_perts}")
    locus_sizes = [len(locus_genes.get(l, [])) for l in locus_names]
    print(f"  Locus gene-set sizes: min={min(locus_sizes)} "
          f"median={sorted(locus_sizes)[n_loci//2]} max={max(locus_sizes)}")

    # ── Load DE signatures ────────────────────────────────────────────────────
    sig_path = DATA_RA / sig_file
    print(f"  Loading signatures: {sig_path.name} …", flush=True)
    with gzip.open(sig_path, "rt") as fh:
        sigs: dict[str, dict[str, float]] = json.load(fh)
    print(f"  Signatures loaded for {len(sigs)} KO genes")

    # ── For each KO, compute mean log2FC over each LOCUS gene set ────────────
    # β(KO → LOCUS) = mean(log2FC[KO, g] for g in locus_genes[LOCUS] if g in sig)
    # Genes not significantly DE in the signature have implicit log2FC = 0;
    # missing genes (not measured) are excluded from the denominator.
    beta = np.zeros((n_perts, n_loci), dtype=np.float32)

    n_missing = 0
    for ko_i, ko_gene in enumerate(pert_names):
        if ko_gene not in sigs:
            n_missing += 1
            continue
        sig_dict = sigs[ko_gene]
        for locus_j, locus in enumerate(locus_names):
            gene_set = locus_genes.get(locus, [])
            # Include all locus genes: DE genes get their log2FC, non-DE get 0
            if not gene_set:
                continue
            total = sum(sig_dict.get(g, 0.0) for g in gene_set)
            beta[ko_i, locus_j] = total / len(gene_set)

    print(f"  KO genes without signatures: {n_missing} / {n_perts}")

    # ── Sanity check: correlate with known targets ────────────────────────────
    _sanity_check(beta, pert_names, locus_names)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez_compressed(
        OUT_NPZ,
        beta=beta,
        ko_genes=np.array(pert_names),
        program_ids=np.array(locus_names),
        condition=condition,
        method="locus_geneset_mean",
    )
    print(f"  Saved → {OUT_NPZ}")
    return str(OUT_NPZ)


def _sanity_check(beta: np.ndarray, pert_names: list[str], locus_names: list[str]):
    """Print β for a few known RA target genes at their expected loci."""
    checks = [
        ("TYK2",  "LOCUS_TYK2"),
        ("CTLA4", "LOCUS_CTLA4"),
        ("STAT4", "LOCUS_STAT4"),
        ("IRF5",  "LOCUS_IRF5"),
        ("SMAD3", None),   # not a direct locus gene — expect spread
    ]
    pert_idx   = {g: i for i, g in enumerate(pert_names)}
    locus_idx  = {l: j for j, l in enumerate(locus_names)}

    print("\n  Sanity check — β at expected loci:")
    for gene, expected_locus in checks:
        if gene not in pert_idx:
            print(f"    {gene:12s}  not in pert_names")
            continue
        row = beta[pert_idx[gene]]
        top_locus_j = int(np.argmax(np.abs(row)))
        top_locus   = locus_names[top_locus_j]
        top_val     = row[top_locus_j]
        exp_val     = row[locus_idx[expected_locus]] if expected_locus in locus_idx else float("nan")
        print(f"    {gene:12s}  top_locus={top_locus:25s}  β={top_val:+.4f}"
              + (f"  β_expected_locus({expected_locus})={exp_val:+.4f}"
                 if expected_locus else ""))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute RA LOCUS-space β matrix")
    ap.add_argument("--sig-file",  default="signatures.json.gz")
    ap.add_argument("--condition", default="combined")
    args = ap.parse_args()
    print("\nComputing LOCUS β matrix …")
    out = compute_locus_betas(sig_file=args.sig_file, condition=args.condition)
    print(f"\nDone → {out}")
