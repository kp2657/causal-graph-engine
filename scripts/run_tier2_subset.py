from __future__ import annotations

"""
Run Tier 2 (perturbation_genomics_agent) on a non-virtual-only subset, then
rerun on a 3-gene subset.

Usage:
  PYTHONPATH="causal-graph-engine" TIER2_NONVIRTUAL_ONLY=1 conda run -n causal-graph \
    python -u causal-graph-engine/scripts/run_tier2_subset.py

Optional:
  TIER2_MAX_GENES=200  # cap after non-virtual filtering
"""

import json
import os
from pathlib import Path

from mcp_servers.open_targets_server import get_open_targets_disease_targets
from agents.tier2_pathway.perturbation_genomics_agent import run as tier2_run


def main() -> None:
    efo = "EFO_0001645"
    disease_name = "coronary artery disease"
    mode = os.getenv("TIER2_SUBSET_MODE", "ot_top").strip().lower()
    # "Take them all": OT Platform is paginated and may enforce server-side caps.
    # We approximate "all" by requesting a large page size.
    _ot_env = os.getenv("OT_MAX_TARGETS", "").strip().lower()
    if _ot_env in {"", "all", "max"}:
        ot_max_targets = 5000
    else:
        ot_max_targets = int(_ot_env)
    # Modes:
    #  - ot_top: use OT top targets (default)
    #  - perturbseq_ot: use intersection(Perturb-seq cache genes, OT targets)

    print("Fetching Open Targets disease targets...", flush=True)
    ot = get_open_targets_disease_targets(efo, max_targets=ot_max_targets, min_overall_score=0.0)
    print("OT response keys:", list(ot.keys()), flush=True)
    if "error" in ot:
        print("OT error:", ot["error"], flush=True)

    rows = ot.get("targets") or ot.get("results") or []
    print("OT rows:", len(rows), flush=True)
    if rows and isinstance(rows[0], dict):
        print("sample row keys:", list(rows[0].keys()), flush=True)

    ot_genes: list[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        g = r.get("gene_symbol") or r.get("symbol") or r.get("gene") or r.get("approvedSymbol")
        if g and g not in ot_genes:
            ot_genes.append(g)

    print("OT genes:", len(ot_genes), flush=True)
    print("OT genes example:", ot_genes[:15], flush=True)
    if not ot_genes:
        raise SystemExit("No genes returned by Open Targets; cannot run Tier 2 subset.")

    ot_scores: dict[str, float] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        g = r.get("gene_symbol") or r.get("symbol") or r.get("gene")
        if not g:
            continue
        s = r.get("genetic_score") or r.get("overall_score") or r.get("score") or r.get("ot_score")
        if s is None:
            continue
        try:
            ot_scores[g] = float(s)
        except Exception:
            pass

    # Optional: perturb-seq ∩ OT (sanity pass)
    genes = ot_genes
    if mode == "perturbseq_ot":
        cache_path = Path(
            "causal-graph-engine/data/perturbseq/Schnitzler_GSE210681/beta_cache_8cea9f625b26.json"
        )
        ps_genes: set[str] = set()
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            if isinstance(data, dict) and isinstance(data.get("genes"), list):
                ps_genes = {str(g) for g in data["genes"] if g}
            elif isinstance(data, dict):
                ps_genes = {str(k) for k in data.keys() if k and not str(k).startswith("_")}

        if not ps_genes:
            raise SystemExit(f"Perturb-seq cache not found/empty at {cache_path}")

        ot_set = set(ot_genes)
        genes = sorted(ps_genes & ot_set)
        print(f"Mode perturbseq_ot: perturbseq_genes={len(ps_genes)}, ot_genes={len(ot_set)}, intersection={len(genes)}", flush=True)
        print("intersection example:", genes[:20], flush=True)

    disease_query = {
        "disease_name": disease_name,
        "efo_id": efo,
        "gwas_genes": genes,
        "ot_genetic_scores": ot_scores,
    }

    res = tier2_run(genes, disease_query)
    print("\n--- Tier2 non-virtual subset run ---")
    print("n_genes:", len(res.get("genes", [])))
    print(
        "counts:",
        {
            "tier1": res.get("n_tier1"),
            "tier2": res.get("n_tier2"),
            "tier3": res.get("n_tier3"),
            "virtual": res.get("n_virtual"),
        },
    )
    warnings = res.get("warnings") or []
    print("warnings_head:", warnings[:5], flush=True)
    print("warnings_tail:", warnings[-5:], flush=True)

    et = res.get("evidence_tier_per_gene", {}) or {}
    nonvirtual = [g for g, t in et.items() if t != "provisional_virtual"]
    print("nonvirtual_genes:", len(nonvirtual), flush=True)
    print("nonvirtual_genes example:", nonvirtual[:15], flush=True)

    subset3 = nonvirtual[:3]
    res3 = tier2_run(subset3, disease_query)
    print("\n--- Tier2 top-3 non-virtual run ---")
    print("genes:", subset3)
    print(
        "counts:",
        {
            "tier1": res3.get("n_tier1"),
            "tier2": res3.get("n_tier2"),
            "tier3": res3.get("n_tier3"),
            "virtual": res3.get("n_virtual"),
        },
    )
    warnings3 = res3.get("warnings") or []
    print("warnings_tail:", warnings3[-5:], flush=True)


if __name__ == "__main__":
    main()

