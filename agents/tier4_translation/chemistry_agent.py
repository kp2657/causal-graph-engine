"""
chemistry_agent.py — Tier 4 agent: GPS screening + reverse-target normalization.

This agent is intentionally minimal:
  - Runs GPS disease/program reversal screens (`pipelines/gps_disease_screen.py`)
  - Aggregates putative target labels from GPS hit annotations
  - Normalizes those labels to HGNC symbols (reverse target search)

We intentionally do NOT do heavy chemistry enrichment here (ChEMBL IC50 mining,
OT tractability/drugs bulk pulls, ADMET, CMap signatures, disease trial checks).
Those either duplicate other agents or add runtime/dependency burden without
changing the core OTA ranking.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _collect_gps_putative_targets(
    disease_reversers: list[dict],
    program_reversers: dict[str, list[dict]],
    *,
    max_unique: int = 200,
) -> list[str]:
    """
    Union ChEMBL-derived putative targets from GPS hit annotations (disease-state
    and program reversal). Names are ChEMBL target_pref_name strings, not always HGNC.
    """
    seen: dict[str, None] = {}
    for hit in disease_reversers or []:
        ann = hit.get("annotation") or {}
        for t in ann.get("putative_targets") or []:
            s = str(t).strip()
            if s:
                seen.setdefault(s, None)
        for sim in ann.get("chembl_similarity_hits") or []:
            for t in sim.get("targets") or []:
                s = str(t).strip()
                if s:
                    seen.setdefault(s, None)
    for hits in (program_reversers or {}).values():
        for hit in hits or []:
            ann = hit.get("annotation") or {}
            for t in ann.get("putative_targets") or []:
                s = str(t).strip()
                if s:
                    seen.setdefault(s, None)
            for sim in ann.get("chembl_similarity_hits") or []:
                for t in sim.get("targets") or []:
                    s = str(t).strip()
                    if s:
                        seen.setdefault(s, None)
    out = list(seen.keys())
    return out[:max_unique]


def run(target_prioritization_result: dict, disease_query: dict) -> dict:
    """
    Run GPS disease/program reversal screens and normalize inferred target labels.

    Args:
        target_prioritization_result: Output of target_prioritization_agent.run
        disease_query:                DiseaseQuery dict

    Returns:
        dict with GPS reversal hits + normalized putative targets (HGNC).
    """
    targets = target_prioritization_result.get("targets", [])
    warnings: list[str] = []
    # Keep output keys stable for downstream code, but chemistry enrichment is empty by design.
    target_chemistry: dict[str, dict] = {}
    repurposing_candidates: list[dict] = []

    if not targets:
        return {
            "target_chemistry": {},
            "repurposing_candidates": [],
            "gps_emulation_candidates": [],
            "gps_disease_reversers": [],
            "gps_program_reversers": {},
            "gps_programs_screened": [],
            "gps_priority_compounds": [],
            "gps_putative_targets_union": [],
            "gps_putative_hgnc": {"genes": [], "n_resolved": 0, "n_unresolved": 0, "mapping_sample": []},
            "warnings": warnings,
        }

    # Step 6 — GPS KO emulation for top-3 Tier 2 genetic anchors.
    # Gated on dominant_tier.startswith("Tier2") and ota_gamma > 0.1 — only
    # high-confidence causal genes with Perturb-seq KO signature coverage are screened.
    # Answers: "does any compound pharmacologically mimic loss-of-function of this target?"
    gps_emulation_candidates: list[dict] = []
    try:
        from pipelines.gps_screen import _docker_available, screen_target_for_emulators
        if _docker_available():
            from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
            tier2_anchors = sorted(
                [
                    t for t in targets
                    if (t.get("dominant_tier") or "").startswith("Tier2")
                    and abs(t.get("ota_gamma") or 0.0) > 0.1
                ],
                key=lambda t: abs(t.get("ota_gamma") or 0.0),
                reverse=True,
            )[:3]

            def _emulate(tgt):
                gene = tgt.get("target_gene", "")
                if not gene:
                    return []
                hits = screen_target_for_emulators(
                    gene=gene,
                    disease_query=disease_query,
                    ota_gamma=tgt.get("ota_gamma"),
                    top_n=20,
                )
                for h in hits:
                    h["target"]          = gene
                    h["emulation_tier"]  = tgt.get("dominant_tier")
                    h["emulation_gamma"] = tgt.get("ota_gamma")
                return hits

            with ThreadPoolExecutor(max_workers=len(tier2_anchors) or 1) as pool:
                futures = {pool.submit(_emulate, t): t for t in tier2_anchors}
                for f in _as_completed(futures):
                    gps_emulation_candidates.extend(f.result())
    except Exception as exc:
        warnings.append(f"GPS KO emulation failed: {exc}")

    # -------------------------------------------------------------------------
    # Step 7 — GPS disease-state and NMF-program reversal screens.
    #
    # Requires Docker + GPS image. Each hit is annotated (PubChem + ChEMBL
    # Tanimoto similarity) with putative_targets — inferred protein names from
    # similar registered drugs, not a full target deconvolution.
    #
    # program→trait γ is read from prioritization_result["_gamma_estimates"]
    # (injected by the orchestrator after Tier 3).
    # -------------------------------------------------------------------------
    gps_disease_reversers:  list[dict] = []
    gps_program_reversers:  dict[str, list[dict]] = {}
    gps_programs_screened:  list[dict] = []
    try:
        from pipelines.gps_disease_screen import run_gps_disease_screens
        _whitelist = disease_query.get("_gps_target_whitelist") or None
        disease_screen_result = run_gps_disease_screens(
            targets=targets,
            disease_query=disease_query,
            gamma_estimates=(
                target_prioritization_result.get("_gamma_estimates")
                or disease_query.get("_gamma_estimates_for_gps")
            ),
            top_n_programs=5,
            top_n_compounds=20,
            target_gene_whitelist=_whitelist,
            genetic_anchors=targets or None,
        )
        gps_disease_reversers = disease_screen_result.get("disease_reversers", [])
        gps_program_reversers = disease_screen_result.get("program_reversers", {})
        gps_programs_screened = disease_screen_result.get("programs_screened", [])
        warnings.extend(disease_screen_result.get("warnings", []))
    except Exception as exc:
        warnings.append(f"GPS disease/program screen failed: {exc}")

    # -------------------------------------------------------------------------
    # Step 8 — Cross-reference: compounds appearing across multiple GPS screens.
    # These have the strongest multi-evidence support: they both correct the
    # disease transcriptional state AND mimic specific causal target perturbations.
    # -------------------------------------------------------------------------
    gps_priority_compounds: list[dict] = []
    try:
        from pipelines.gps_disease_screen import find_overlapping_compounds
        gps_priority_compounds = find_overlapping_compounds(
            disease_reversers=gps_disease_reversers,
            emulation_candidates=gps_emulation_candidates,
            program_reversers=gps_program_reversers,
        )
    except Exception as exc:
        warnings.append(f"GPS cross-reference failed: {exc}")

    gps_putative_targets_union = _collect_gps_putative_targets(
        gps_disease_reversers, gps_program_reversers
    )

    gps_putative_hgnc: dict = {"genes": [], "n_resolved": 0, "n_unresolved": 0, "mapping_sample": []}
    if gps_putative_targets_union:
        try:
            from mcp_servers.chemistry_server import resolve_gps_putative_target_labels_to_hgnc
            gps_putative_hgnc = resolve_gps_putative_target_labels_to_hgnc(
                gps_putative_targets_union,
                max_labels=150,
            )
        except Exception as exc:
            warnings.append(f"GPS putative target → HGNC normalization failed: {exc}")

    return {
        "target_chemistry":         target_chemistry,
        "repurposing_candidates":   repurposing_candidates,
        "gps_emulation_candidates": gps_emulation_candidates,
        "gps_disease_reversers":    gps_disease_reversers,
        "gps_program_reversers":    gps_program_reversers,
        "gps_programs_screened":    gps_programs_screened,
        "gps_priority_compounds":   gps_priority_compounds,
        "gps_putative_targets_union": gps_putative_targets_union,
        "gps_putative_hgnc":          gps_putative_hgnc,
        "warnings":                 warnings,
    }
