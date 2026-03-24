"""
target_prioritization_agent.py — Tier 4 agent: composite target scoring & ranking.

Combines causal evidence (Ota γ), Open Targets score, clinical trial phase bonus,
cell-type specificity (tau index + bimodality), and safety penalty to rank targets.

Scoring formula (v2):
  score = W_CAUSAL × causal + W_OT × ot + W_TRIAL × trial
        + W_SPECIFICITY × specificity - W_SAFETY × safety_penalty

Specificity sub-score:
  specificity = 0.6 × tau + 0.4 × bimodality_coefficient

This implements the Virtual Biotech recommendation: use the minimum tau
across a drug's protein targets (when the drug is multi-target) so that a
high-specificity component doesn't mask a low-specificity co-target.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Scoring weights — must sum to 1.0 (W_SAFETY is subtracted separately)
W_CAUSAL      = 0.35
W_OT          = 0.25
W_TRIAL       = 0.15
W_SPECIFICITY = 0.15
W_SAFETY      = 0.10

# Clinical trial phase bonuses
TRIAL_BONUS: dict[int, float] = {1: 0.1, 2: 0.3, 3: 0.6, 4: 1.0}

# Safety penalty thresholds
PLI_ESSENTIAL = 0.9
PLI_PENALTY   = 0.3
AE_PENALTY    = 0.2

# Tier multipliers for causal score normalization
TIER_MULTIPLIER: dict[str, float] = {
    "Tier1_Interventional": 1.0,
    "Tier2_Convergent":     0.8,
    "Tier3_Provisional":    0.5,
    "moderate_transferred": 0.5,
    "moderate_grn":         0.5,
    "provisional_virtual":  0.1,
}

CHIP_GENES = {"TET2", "DNMT3A", "ASXL1", "JAK2", "TET2_chip", "DNMT3A_chip"}


def _phase_bonus(max_phase: int) -> float:
    return TRIAL_BONUS.get(max_phase, 0.0)


def run(
    causal_discovery_result: dict,
    kg_completion_result: dict,
    disease_query: dict,
) -> dict:
    """
    Rank therapeutic targets using the composite scoring formula.

    Args:
        causal_discovery_result: Output of causal_discovery_agent.run
        kg_completion_result:    Output of kg_completion_agent.run
        disease_query:           DiseaseQuery dict

    Returns:
        list of TargetRecord-compatible dicts, sorted by target_score
    """
    from mcp_servers.open_targets_server import get_open_targets_disease_targets
    from mcp_servers.clinical_trials_server import get_trials_for_target
    from mcp_servers.gwas_genetics_server import query_gnomad_lof_constraint
    from mcp_servers.single_cell_server import (
        get_gene_tau_specificity,
        get_gene_bimodality_scores,
    )

    efo_id       = disease_query.get("efo_id", "")
    disease_name = disease_query.get("disease_name", "")
    top_genes    = causal_discovery_result.get("top_genes", [])
    drug_target_summary = kg_completion_result.get("drug_target_summary", [])
    brg_candidates = kg_completion_result.get("brg_novel_candidates", [])
    warnings: list[str] = []

    # Index BRG scores for quick lookup
    brg_score_map: dict[str, float] = {
        c["gene"]: c["brg_score"] for c in brg_candidates if c.get("gene")
    }

    if not top_genes:
        return {"targets": [], "warnings": ["No genes in causal_discovery_result"]}

    # -------------------------------------------------------------------------
    # Load Open Targets scores for all top genes at once
    # -------------------------------------------------------------------------
    ot_scores: dict[str, float] = {}
    ot_genetic: dict[str, float] = {}
    ot_max_phase: dict[str, int] = {}
    ot_known_drugs: dict[str, list[str]] = {}
    try:
        ot_result = get_open_targets_disease_targets(efo_id)
        for t in ot_result.get("targets", []):
            # Live OT GraphQL returns "gene_symbol"; cached fallback uses "symbol"
            gene = t.get("gene_symbol") or t.get("symbol") or t.get("gene", "")
            if not gene:
                continue
            ot_scores[gene]    = float(t.get("overall_score") or 0.0)
            ot_genetic[gene]   = float(t.get("genetic_score") or 0.0)
            ot_max_phase[gene] = int(t.get("max_clinical_phase") or 0)
            if t.get("known_drugs"):
                ot_known_drugs[gene] = t["known_drugs"]
    except Exception as exc:
        warnings.append(f"Open Targets disease targets failed: {exc}")

    # Drug-target map — seed from OT batch results, then augment from KG completion
    drug_for_gene: dict[str, list[str]] = {g: list(drugs) for g, drugs in ot_known_drugs.items()}
    phase_from_kg: dict[str, int] = {}
    for entry in drug_target_summary:
        gene = entry.get("target", "")
        drug = entry.get("drug")
        phase = entry.get("max_phase", 0)
        if gene:
            if drug and drug not in drug_for_gene.get(gene, []):
                drug_for_gene.setdefault(gene, []).append(drug)
            phase_from_kg[gene] = max(phase_from_kg.get(gene, 0), phase)

    # gnomAD constraint
    # query_gnomad_lof_constraint returns {"genes": [{"symbol": ..., "pLI": ...}, ...]}
    gene_names = [r["gene"] for r in top_genes]
    pli_map: dict[str, float] = {}
    try:
        constraint = query_gnomad_lof_constraint(gene_names)
        for item in constraint.get("genes", []):
            gene = item.get("symbol", "")
            pli  = item.get("pLI") or item.get("pli")
            if gene and pli is not None:
                pli_map[gene] = float(pli)
    except Exception as exc:
        warnings.append(f"gnomAD constraint lookup failed: {exc}")

    # -------------------------------------------------------------------------
    # Cell-type specificity: tau index + bimodality coefficient
    # Per Virtual Biotech recommendation: use minimum tau across drug targets
    # so multi-target drugs are not misleadingly high-scored.
    # -------------------------------------------------------------------------
    gene_names = [r["gene"] for r in top_genes]
    tau_map: dict[str, float | None]  = {}
    bc_map:  dict[str, float | None]  = {}
    try:
        tau_result = get_gene_tau_specificity(gene_names)
        for gene, entry in tau_result.get("tau_scores", {}).items():
            tau_map[gene] = entry.get("tau")
    except Exception as exc:
        warnings.append(f"Tau specificity lookup failed: {exc}")
    try:
        bc_result = get_gene_bimodality_scores(gene_names)
        for gene, entry in bc_result.get("bimodality_scores", {}).items():
            bc_map[gene] = entry.get("bc")
    except Exception as exc:
        warnings.append(f"Bimodality coefficient lookup failed: {exc}")

    # -------------------------------------------------------------------------
    # Normalize causal scores
    # -------------------------------------------------------------------------
    max_gamma = max(
        (abs(r.get("ota_gamma", 0.0)) for r in top_genes),
        default=1.0,
    ) or 1.0

    # -------------------------------------------------------------------------
    # Score each gene
    # -------------------------------------------------------------------------
    target_records: list[dict] = []

    for rec in top_genes:
        gene         = rec["gene"]
        ota_gamma    = abs(rec.get("ota_gamma", 0.0))
        ota_sigma    = rec.get("ota_gamma_sigma", 0.0) or 0.0
        tier         = rec.get("tier", "provisional_virtual")
        tier_mult    = TIER_MULTIPLIER.get(tier, 0.1)
        top_programs = rec.get("programs", [])

        # Causal score
        causal_score = (ota_gamma / max_gamma) * tier_mult

        # OT score
        ot_score = ot_scores.get(gene, 0.0)
        genetic_contrib = ot_genetic.get(gene, 0.0) * 0.5
        ot_combined = ot_score + genetic_contrib * 0.3

        # Trial bonus — prefer OT max phase, fallback to KG
        max_phase = max(
            ot_max_phase.get(gene, 0),
            phase_from_kg.get(gene, 0),
        )
        trial_bonus = _phase_bonus(max_phase)

        # Specificity score: 0.6 × tau + 0.4 × bimodality_coeff (both in [0,1])
        tau = tau_map.get(gene)
        bc  = bc_map.get(gene)
        # Normalize bimodality: BC in [0, 1] scale (cap at 1.0)
        bc_norm = min(bc / 1.0, 1.0) if bc is not None else 0.5  # 0.5 = neutral prior
        tau_val = tau if tau is not None else 0.5  # 0.5 = neutral prior when unknown
        specificity_score = 0.6 * tau_val + 0.4 * bc_norm

        # For multi-target drugs: use minimum tau across drug's co-targets
        # (prevents a specific co-target masking a ubiquitous co-target)
        co_targets = [d for d in drug_for_gene.get(gene, [])]
        if len(co_targets) > 1:
            min_tau = min(
                (tau_map.get(t, tau_val) or tau_val for t in co_targets[:4]),
                default=tau_val,
            )
            if min_tau < tau_val:
                specificity_score = 0.6 * min_tau + 0.4 * bc_norm

        # Safety penalty
        pli = pli_map.get(gene)
        safety_pen = 0.0
        safety_flags: list[str] = []
        if pli is not None and pli > PLI_ESSENTIAL:
            safety_pen += PLI_PENALTY
            safety_flags.append(f"pLI={pli:.2f} > 0.9 — essential gene; on-target toxicity")

        # Composite score (v2 — includes specificity)
        target_score = (
            W_CAUSAL      * causal_score
            + W_OT        * ot_combined
            + W_TRIAL     * trial_bonus
            + W_SPECIFICITY * specificity_score
            - W_SAFETY    * safety_pen
        )

        # Flags
        flags: list[str] = []
        if max_phase >= 2 and ot_score > 0.5:
            flags.append("repurposing_candidate")
        if max_phase == 0 and causal_score > 0.7:
            flags.append("first_in_class")
        if tier in ("Tier1_Interventional", "Tier2_Convergent"):
            flags.append("genetic_anchor")
        if gene in CHIP_GENES or gene.replace("_chip", "") in CHIP_GENES:
            flags.append("chip_mechanism")
        if tier == "provisional_virtual":
            flags.append("provisional_virtual")
        if tau is not None and tau >= 0.70:
            flags.append("highly_specific")
        if bc is not None and bc > 0.555:
            flags.append("bimodal_expression")
        brg_score = brg_score_map.get(gene)
        if brg_score is not None:
            flags.append("brg_novel_candidate")

        # Evidence keys from drug_target_summary
        key_evidence: list[str] = []
        if drug_for_gene.get(gene):
            key_evidence.append(f"Drug: {', '.join(drug_for_gene[gene][:3])}")

        # Gamma 95% CI — propagated from delta-method uncertainty in causal_discovery
        gamma_ci_lower = rec.get("ota_gamma_ci_lower")
        gamma_ci_upper = rec.get("ota_gamma_ci_upper")

        target_records.append({
            "target_gene":        gene,
            "rank":               0,  # filled after sort
            "target_score":       round(target_score, 4),
            "ota_gamma":          rec.get("ota_gamma", 0.0),
            "ota_gamma_raw":      rec.get("ota_gamma_raw", rec.get("ota_gamma", 0.0)),
            "ota_gamma_sigma":    round(ota_sigma, 4),
            "ota_gamma_ci_lower": round(gamma_ci_lower, 4) if gamma_ci_lower is not None else None,
            "ota_gamma_ci_upper": round(gamma_ci_upper, 4) if gamma_ci_upper is not None else None,
            "evidence_tier":      tier,
            "ot_score":           ot_score,
            "max_phase":          max_phase,
            "known_drugs":        drug_for_gene.get(gene, []),
            "pli":                pli,
            "tau_specificity":    tau,
            "bimodality_coeff":   bc,
            "specificity_score":  round(specificity_score, 4),
            "brg_score":          brg_score,
            "scone_confidence":   rec.get("scone_confidence"),
            "scone_flags":        rec.get("scone_flags", []),
            "flags":              flags,
            "top_programs":       top_programs,
            "key_evidence":       key_evidence,
            "safety_flags":       safety_flags,
        })

    # Sort and assign ranks
    target_records.sort(key=lambda r: r["target_score"], reverse=True)
    for i, r in enumerate(target_records, 1):
        r["rank"] = i

    return {
        "targets":  target_records,
        "warnings": warnings,
    }
