"""
chemistry_agent.py — Tier 4 agent: chemical space characterization.

Finds existing compounds via ChEMBL + PubChem, checks ADMET properties,
retrieves CMap L1000 signatures, and identifies repurposing opportunities.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Pre-known drug-target pairs for CAD
KNOWN_DRUG_TARGET_CAD: dict[str, str] = {
    "HMGCR":  "atorvastatin",
    "PCSK9":  "evolocumab",
    "IL6R":   "tocilizumab",
    "TET2":   "azacitidine",
    "DNMT3A": "azacitidine",
}


def _fetch_ic50(gene: str, max_results: int = 20) -> tuple[str, float | None, str | None]:
    """Fetch best IC50/Ki for a gene from ChEMBL. Returns (gene, best_ic50, chembl_id)."""
    from mcp_servers.chemistry_server import get_chembl_target_activities
    try:
        activities = get_chembl_target_activities(gene, max_results=max_results)
        acts = activities.get("activities", [])
        ic50_values = [
            a.get("standard_value") for a in acts
            if a.get("standard_type") in ("IC50", "Ki")
            and a.get("standard_value") is not None
        ]
        best_ic50 = min(ic50_values) if ic50_values else None
        chembl_id = acts[0].get("target_chembl_id") if acts else None
        return gene, best_ic50, chembl_id
    except Exception:
        return gene, None, None


def run(target_prioritization_result: dict, disease_query: dict) -> dict:
    """
    Characterize the chemical space for prioritized targets.

    Strategy:
      1. Batch-fetch tractability + drugs + max_phase for ALL genes from OT in parallel (~5s)
      2. Parallel ChEMBL IC50 lookup only for genes that need it (~10s with ThreadPoolExecutor)
      3. CMap L1000 batch query for known drugs

    Args:
        target_prioritization_result: Output of target_prioritization_agent.run
        disease_query:                DiseaseQuery dict

    Returns:
        dict with target_chemistry, repurposing_candidates, warnings
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from mcp_servers.chemistry_server import search_chembl_compound
    from mcp_servers.viral_somatic_server import get_cmap_drug_signatures
    from mcp_servers.open_targets_server import get_open_targets_targets_bulk

    targets = target_prioritization_result.get("targets", [])
    warnings: list[str] = []
    target_chemistry: dict[str, dict] = {}
    repurposing_candidates: list[dict] = []

    if not targets:
        return {"target_chemistry": {}, "repurposing_candidates": [], "warnings": warnings}

    # -------------------------------------------------------------------------
    # Step 1 — Batch OT prefetch: tractability + drugs + max_phase for all genes
    # This single parallel call replaces N serial per-gene OT + ChEMBL calls.
    # -------------------------------------------------------------------------
    all_gene_symbols = [rec.get("target_gene", "") for rec in targets if rec.get("target_gene")]
    ot_bulk: dict[str, dict] = {}
    try:
        bulk_result = get_open_targets_targets_bulk(all_gene_symbols)
        ot_bulk = bulk_result.get("targets", {})
    except Exception as exc:
        warnings.append(f"OT bulk prefetch failed: {exc}")

    # -------------------------------------------------------------------------
    # Step 2 — Build per-gene chemistry record using OT data
    # -------------------------------------------------------------------------
    genes_needing_ic50: list[str] = []

    for rec in targets:
        gene        = rec.get("target_gene", "")
        max_phase   = rec.get("max_phase", 0)
        known_drugs = list(rec.get("known_drugs", []) or [])

        # Seed from pre-known CAD drug map if still empty
        if not known_drugs and gene in KNOWN_DRUG_TARGET_CAD:
            known_drugs = [KNOWN_DRUG_TARGET_CAD[gene]]

        # Enrich from OT bulk results
        ot_info       = ot_bulk.get(gene, {})
        ot_drugs      = [d["name"] for d in ot_info.get("known_drugs", []) if d.get("name")]
        ot_phase      = ot_info.get("max_phase", 0)
        tractability  = ot_info.get("tractability_class", "unknown")

        # Merge drug lists (preserve order, deduplicate)
        for d in ot_drugs:
            if d not in known_drugs:
                known_drugs.append(d)
        max_phase = max(max_phase, ot_phase)

        target_chemistry[gene] = {
            "chembl_id":      None,
            "max_phase":      max_phase,
            "best_ic50_nM":   None,
            "tractability":   tractability,
            "ro5_violations": None,
            "cmap_available": False,
            "drugs_found":    known_drugs,
        }

        # Queue IC50 lookup for tractable or known-drug targets
        if tractability in ("small_molecule", "antibody") or known_drugs:
            genes_needing_ic50.append(gene)

    # -------------------------------------------------------------------------
    # Step 3 — Parallel ChEMBL IC50 lookup (only for tractable genes)
    # -------------------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(_fetch_ic50, gene): gene
            for gene in genes_needing_ic50
        }
        for future in as_completed(futures):
            gene, best_ic50, chembl_id = future.result()
            if gene in target_chemistry:
                target_chemistry[gene]["best_ic50_nM"] = best_ic50
                if chembl_id:
                    target_chemistry[gene]["chembl_id"] = chembl_id

    # -------------------------------------------------------------------------
    # Step 4 — CMap L1000 signatures (batch)
    # -------------------------------------------------------------------------
    all_drugs = [
        drug
        for chem in target_chemistry.values()
        for drug in chem.get("drugs_found", [])[:2]
    ]
    if all_drugs:
        try:
            cmap_result = get_cmap_drug_signatures(list(set(all_drugs)))
            sig_map = cmap_result.get("signatures", {})
            for chem in target_chemistry.values():
                if any(d in sig_map for d in chem.get("drugs_found", [])):
                    chem["cmap_available"] = True
        except Exception as exc:
            warnings.append(f"CMap signatures failed: {exc}")

    # -------------------------------------------------------------------------
    # Step 5 — Repurposing opportunities
    # -------------------------------------------------------------------------
    for gene, chem in target_chemistry.items():
        drugs  = chem.get("drugs_found", [])
        max_ph = chem.get("max_phase", 0)
        if max_ph >= 2 and drugs:
            repurposing_candidates.append({
                "drug":            drugs[0],
                "target":          gene,
                "cmap_similarity": None,  # requires L1000 download
                "rationale": (
                    f"{drugs[0]} approved (Phase {max_ph}); "
                    f"causal evidence for {gene} in {disease_query.get('disease_name', '')}"
                ),
            })

    return {
        "target_chemistry":       target_chemistry,
        "repurposing_candidates": repurposing_candidates,
        "warnings":               warnings,
    }
