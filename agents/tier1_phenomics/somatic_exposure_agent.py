"""
somatic_exposure_agent.py — Tier 1 agent: somatic (CHIP) + viral + drug exposures.

Retrieves CHIP associations, viral MR results, and drug MR evidence;
converts each to CausalEdge format with appropriate evidence tiers.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.evidence import CausalEdge, EvidenceTier


# Disease full name → short form used by viral_somatic_server data tables
DISEASE_SHORT_NAMES: dict[str, str] = {
    "coronary artery disease": "CAD",
    "ischemic heart disease":  "CAD",
    "myocardial infarction":   "CAD",
    "rheumatoid arthritis":    "RA",
    "systemic lupus erythematosus": "SLE",
}

# CHIP gene → expected evidence tier (based on replication across cohorts)
CHIP_TIER_MAP: dict[str, EvidenceTier] = {
    "CHIP_any": "Tier2_Convergent",
    "TET2":     "Tier2_Convergent",
    "DNMT3A":   "Tier2_Convergent",
    "ASXL1":    "Tier3_Provisional",
    "JAK2":     "Tier3_Provisional",
}

# Drug → gene → disease for known drug-target MR
DRUG_TARGET_TRIPLES: list[dict] = [
    {"drug": "statin",       "gene": "HMGCR",  "disease": "CAD"},
    {"drug": "evolocumab",   "gene": "PCSK9",  "disease": "CAD"},
    {"drug": "tocilizumab",  "gene": "IL6R",   "disease": "CAD"},
]


def _safe_log(value: float) -> float:
    """Log-transform a ratio while handling ≤0 gracefully."""
    if value <= 0:
        return 0.0
    return math.log(value)


def run(disease_query: dict) -> dict:
    """
    Collect somatic, viral, and drug exposure evidence for a disease.

    Args:
        disease_query: DiseaseQuery dict

    Returns:
        dict with chip_edges, viral_edges, drug_edges, summary, warnings
    """
    from mcp_servers.viral_somatic_server import (
        get_chip_disease_associations,
        get_viral_disease_mr_results,
        get_drug_exposure_mr,
    )
    from mcp_servers.open_targets_server import get_open_targets_drug_info
    from mcp_servers.clinical_trials_server import search_clinical_trials

    disease_name = disease_query.get("disease_name", "")
    # Normalize to short form used by the hardcoded data tables (e.g. "CAD", "RA")
    disease_lookup = DISEASE_SHORT_NAMES.get(disease_name.lower(), disease_name)
    modifier_types = disease_query.get("modifier_types", ["germline"])
    warnings: list[str] = []

    # -------------------------------------------------------------------------
    # 1. CHIP associations
    # -------------------------------------------------------------------------
    chip_edges: list[dict] = []
    if "somatic_chip" in modifier_types:
        try:
            chip_result = get_chip_disease_associations(disease_lookup)
            associations = chip_result.get("associations", [])
            for assoc in associations:
                gene = assoc.get("gene", "CHIP_any")
                hr = assoc.get("hr")
                log_effect = _safe_log(hr) if hr else assoc.get("log_or", 0.0)
                ci_low = assoc.get("ci_lower")
                ci_hi  = assoc.get("ci_upper")
                log_ci_low = _safe_log(ci_low) if ci_low else None
                log_ci_hi  = _safe_log(ci_hi)  if ci_hi  else None
                tier: EvidenceTier = CHIP_TIER_MAP.get(gene, "Tier3_Provisional")

                try:
                    edge = CausalEdge(
                        from_node=f"{gene}_chip",
                        from_type="gene",
                        to_node=disease_lookup,  # use short form matching anchor edges
                        to_type="trait",
                        effect_size=log_effect,
                        ci_lower=log_ci_low,
                        ci_upper=log_ci_hi,
                        evidence_type="somatic_chip",
                        evidence_tier=tier,
                        method="mr",
                        data_source=assoc.get("source", "Bick2020/Kar2022"),
                        data_source_version="2020/2022",
                    )
                    chip_edges.append(edge.model_dump())
                except Exception as exc:
                    warnings.append(f"CHIP edge creation failed for {gene}: {exc}")

        except Exception as exc:
            warnings.append(f"CHIP associations lookup failed: {exc}")

    # -------------------------------------------------------------------------
    # 2. Viral exposure MR
    # -------------------------------------------------------------------------
    viral_edges: list[dict] = []
    if "viral" in modifier_types:
        for virus in ["EBV"]:
            try:
                viral_mr = get_viral_disease_mr_results(virus, disease_name)
                mr_beta = viral_mr.get("beta")
                mr_p    = viral_mr.get("p_value", 1.0)
                if mr_beta is None:
                    continue

                viral_tier: EvidenceTier
                if mr_p < 5e-8:
                    viral_tier = "Tier1_Interventional"
                elif mr_p < 0.05:
                    viral_tier = "Tier3_Provisional"
                else:
                    warnings.append(
                        f"{virus} → {disease_name}: MR p={mr_p:.2e}, not significant"
                    )
                    continue

                edge = CausalEdge(
                    from_node=virus,
                    from_type="virus",
                    to_node=disease_name,
                    to_type="trait",
                    effect_size=mr_beta,
                    evidence_type="viral",
                    evidence_tier=viral_tier,
                    method="mr",
                    data_source=viral_mr.get("source", "Nyeo2026"),
                    data_source_version="2026",
                    mr_ivw=mr_beta,
                )
                viral_edges.append(edge.model_dump())

            except Exception as exc:
                warnings.append(f"{virus} viral MR failed: {exc}")

    # -------------------------------------------------------------------------
    # 3. Drug exposure MR
    # -------------------------------------------------------------------------
    drug_edges: list[dict] = []
    if "drug" in modifier_types:
        for triple in DRUG_TARGET_TRIPLES:
            # Match against both the full name and the normalized short name
            if (triple["disease"].lower() not in disease_name.lower()
                    and triple["disease"].lower() not in disease_lookup.lower()):
                continue
            try:
                drug_mr = get_drug_exposure_mr(triple["drug"], triple["disease"])
                mr_beta = drug_mr.get("beta")
                if mr_beta is None:
                    continue

                # Confirm trial evidence
                trial_info = search_clinical_trials(
                    condition=triple["disease"],
                    intervention=triple["drug"],
                    phase="PHASE3",
                )
                max_phase = 0
                for t in trial_info.get("trials", []):
                    phases = t.get("phase", [])
                    for ph in phases:
                        try:
                            ph_num = int(str(ph).replace("PHASE", "").strip())
                            max_phase = max(max_phase, ph_num)
                        except ValueError:
                            pass

                drug_tier: EvidenceTier = "Tier1_Interventional" if max_phase >= 3 else "Tier2_Convergent"

                edge = CausalEdge(
                    from_node=triple["gene"],
                    from_type="gene",
                    to_node=triple["disease"],
                    to_type="trait",
                    effect_size=mr_beta,
                    evidence_type="drug",
                    evidence_tier=drug_tier,
                    method="mr",
                    data_source=drug_mr.get("source", "published_MR"),
                    data_source_version="various",
                    mr_ivw=mr_beta,
                )
                drug_edges.append(edge.model_dump())

            except Exception as exc:
                warnings.append(
                    f"Drug MR failed for {triple['drug']} → {triple['gene']}: {exc}"
                )

    return {
        "chip_edges":  chip_edges,
        "viral_edges": viral_edges,
        "drug_edges":  drug_edges,
        "summary": {
            "n_chip_genes":   len(chip_edges),
            "n_viral_viruses": len(viral_edges),
            "n_drug_targets":  len(drug_edges),
        },
        "warnings": warnings,
    }
