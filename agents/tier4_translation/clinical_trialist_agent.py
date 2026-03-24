"""
clinical_trialist_agent.py — Tier 4 agent: clinical trial landscape assessment.

Maps the trial landscape for prioritized targets, assesses development risk,
and identifies repurposing gaps.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Key CAD trials to cross-check
KEY_CAD_TRIALS: list[dict] = [
    {"nct_id": "NCT01764633", "drug": "evolocumab",  "target": "PCSK9", "phase": [3]},
    {"nct_id": "NCT01663402", "drug": "alirocumab",  "target": "PCSK9", "phase": [3]},
    {"nct_id": "NCT01740453", "drug": "tocilizumab", "target": "IL6R",  "phase": [3]},
    {"nct_id": "NCT03705286", "drug": "inclisiran",  "target": "PCSK9", "phase": [3]},
]


def _parse_max_phase(phases: list) -> int:
    max_ph = 0
    for ph in phases:
        try:
            ph_num = int(str(ph).replace("PHASE", "").strip())
            max_ph = max(max_ph, ph_num)
        except (ValueError, AttributeError):
            pass
    return max_ph


def run(
    target_prioritization_result: dict,
    disease_query: dict,
) -> dict:
    """
    Assess the clinical development landscape for top targets.

    Args:
        target_prioritization_result: Output of target_prioritization_agent.run
        disease_query:                DiseaseQuery dict

    Returns:
        dict with trial_summary, key_trials, development_risk,
        repurposing_opportunities, warnings
    """
    from mcp_servers.clinical_trials_server import (
        search_clinical_trials,
        get_trial_details,
        get_trials_for_target,
    )
    from mcp_servers.open_targets_server import get_open_targets_drug_info

    disease_name = disease_query.get("disease_name", "")
    targets      = target_prioritization_result.get("targets", [])
    warnings: list[str] = []

    trial_summary: dict[str, dict] = {}
    development_risk: dict[str, str] = {}
    repurposing_opportunities: list[str] = []
    key_trials: list[dict] = []

    # -------------------------------------------------------------------------
    # Key trials: look up NCT IDs for CAD
    # -------------------------------------------------------------------------
    is_cad = "CAD" in disease_name.upper() or "coronary" in disease_name.lower()
    if is_cad:
        for kt in KEY_CAD_TRIALS:
            try:
                details = get_trial_details(kt["nct_id"])
                key_trials.append({
                    "nct_id":          kt["nct_id"],
                    "drug":            kt["drug"],
                    "phase":           kt["phase"],
                    "status":          details.get("status", "UNKNOWN"),
                    "primary_outcome": details.get("primary_outcome", ""),
                    "n_enrolled":      details.get("enrollment"),
                })
            except Exception as exc:
                warnings.append(f"Key trial {kt['nct_id']} lookup failed: {exc}")
                key_trials.append({
                    "nct_id":          kt["nct_id"],
                    "drug":            kt["drug"],
                    "phase":           kt["phase"],
                    "status":          "LOOKUP_FAILED",
                    "primary_outcome": "",
                    "n_enrolled":      None,
                })

    # -------------------------------------------------------------------------
    # Per-target trial landscape
    # -------------------------------------------------------------------------
    for rec in targets:
        gene      = rec.get("target_gene", "")
        known_drugs = rec.get("known_drugs", []) or []
        n_total = n_active = n_completed = n_terminated = 0
        max_phase_reached = rec.get("max_phase", 0)
        safety_signals: list[str] = []

        try:
            # All-status trial search
            trial_result = search_clinical_trials(
                condition=disease_name,
                intervention=gene,
                status=None,
            )
            all_trials = trial_result.get("trials", [])
            n_total = len(all_trials)

            for t in all_trials:
                status = (t.get("status") or "").upper()
                phases = t.get("phase", [])
                ph = _parse_max_phase(phases)
                max_phase_reached = max(max_phase_reached, ph)

                if status in ("RECRUITING", "ACTIVE_NOT_RECRUITING"):
                    n_active += 1
                elif status == "COMPLETED":
                    n_completed += 1
                elif status in ("TERMINATED", "WITHDRAWN"):
                    n_terminated += 1
                    # Investigate reason
                    reason = (t.get("why_stopped") or "").lower()
                    if any(kw in reason for kw in ("safety", "adverse", "toxicity")):
                        safety_signals.append(
                            f"Trial {t.get('nct_id', '?')} terminated: {reason[:100]}"
                        )
                    elif any(kw in reason for kw in ("efficacy", "futility")):
                        warnings.append(
                            f"{gene}: Trial {t.get('nct_id', '?')} terminated for efficacy failure "
                            f"— reconsider causal hypothesis"
                        )

        except Exception as exc:
            warnings.append(f"{gene}: Trial search failed: {exc}")

        # Also try drug-specific search
        for drug in known_drugs[:2]:
            try:
                drug_trials = search_clinical_trials(
                    condition=disease_name,
                    intervention=drug,
                    status=None,
                )
                drug_trial_list = drug_trials.get("trials", [])
                n_total += len(drug_trial_list)
                for t in drug_trial_list:
                    ph = _parse_max_phase(t.get("phase", []))
                    max_phase_reached = max(max_phase_reached, ph)
            except Exception:
                pass

        trial_summary[gene] = {
            "n_trials_total":   n_total,
            "n_active":         n_active,
            "n_completed":      n_completed,
            "n_terminated":     n_terminated,
            "max_phase_reached": max_phase_reached,
            "safety_signals":   safety_signals,
        }

        # Risk assessment
        pli = rec.get("pli")
        risk = "low"
        if safety_signals:
            risk = "high"
            warnings.append(f"{gene}: Safety signal detected — HIGH risk")
        elif n_terminated > 0:
            risk = "medium"
        elif pli is not None and pli > 0.9:
            risk = "medium"
            warnings.append(f"{gene}: pLI={pli:.2f} — essential gene, narrow therapeutic window")

        development_risk[gene] = risk

        # Repurposing check
        if max_phase_reached >= 2 and known_drugs:
            try:
                ot_drug = get_open_targets_drug_info(known_drugs[0])
                drug_indications = ot_drug.get("indications", [])
                other_indications = [
                    ind for ind in drug_indications
                    if disease_name.lower() not in str(ind).lower()
                ]
                if other_indications:
                    repurposing_opportunities.append(
                        f"{known_drugs[0]} (target: {gene}) approved for "
                        f"{other_indications[0]}; repurposing opportunity for {disease_name}"
                    )
            except Exception:
                pass

    return {
        "trial_summary":             trial_summary,
        "key_trials":                key_trials,
        "development_risk":          development_risk,
        "repurposing_opportunities": repurposing_opportunities,
        "warnings":                  warnings,
    }
