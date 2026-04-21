"""
agents/tier5_writer/red_team_agent.py — Phase T: Adversarial Red Team Agent.

For each top-5 target the red team agent formulates:
  1. The strongest counterargument against pursuing the target
  2. An assessment of evidence vulnerability (rank stability under evidence removal)
  3. SCONE bootstrap confidence as a quantified robustness metric
  4. Literature contradiction signals (from Phase Q)

Red team does NOT kill targets — it surfaces risk so the CSO and reviewers can
make informed decisions. Findings are additive to the pipeline, not gating.

Output schema per target:
  {
    "target_gene":          str,
    "rank":                 int,
    "scone_confidence":     float | None,
    "scone_flags":          list[str],
    "ota_gamma_ci_lower":   float | None,
    "ota_gamma_ci_upper":   float | None,
    "ci_width":             float | None,     upper - lower
    "confidence_level":     "HIGH" | "MODERATE" | "LOW" | "REJECTED",
    "evidence_vulnerability": str,           prose: weakest evidence link
    "counterargument":      str,             strongest single argument against
    "rank_stability":       str,             "STABLE" | "FRAGILE" | "TIER-DEPENDENT"
    "rank_stability_rationale": str,
    "literature_flag":      str | None,      from lit_validation (CONTRADICTED targets)
    "red_team_verdict":     "PROCEED" | "CAUTION" | "DEPRIORITIZE",
    "counterfactual":       dict | None,     simulate_perturbation results (50% inhibition + KO)
  }
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIGH_CONFIDENCE_THRESHOLD  = 0.80
_LOW_CONFIDENCE_THRESHOLD   = 0.50
_WIDE_CI_THRESHOLD          = 0.30   # |upper - lower| > 0.30 → wide

_FRAGILE_TIERS = {"Tier3_Provisional", "provisional_virtual", "moderate_grn", "state_nominated"}


# ---------------------------------------------------------------------------
# Core analysis helpers
# ---------------------------------------------------------------------------

def _confidence_level(scone_confidence: float | None, scone_flags: list[str]) -> str:
    if "bootstrap_rejected" in scone_flags:
        return "REJECTED"
    if scone_confidence is None:
        return "MODERATE"   # unknown — not a rejection
    if scone_confidence >= _HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH"
    if scone_confidence >= _LOW_CONFIDENCE_THRESHOLD:
        return "MODERATE"
    return "LOW"


def _ci_width(ci_lower: float | None, ci_upper: float | None) -> float | None:
    if ci_lower is None or ci_upper is None:
        return None
    return round(abs(ci_upper - ci_lower), 4)


def _evidence_vulnerability(target: dict) -> str:
    """Prose description of the weakest evidence link."""
    tier = target.get("evidence_tier", "provisional_virtual")
    scone_flags = target.get("scone_flags") or []
    ota_sigma = target.get("ota_gamma_sigma") or 0.0
    ci_lower = target.get("ota_gamma_ci_lower")
    ci_upper = target.get("ota_gamma_ci_upper")
    width = _ci_width(ci_lower, ci_upper)

    parts: list[str] = []

    if tier in _FRAGILE_TIERS:
        parts.append(
            f"Evidence rests on {tier} tier — no direct interventional β estimate. "
            "If this tier is excluded, the target may drop out of top-5."
        )

    if "bootstrap_rejected" in scone_flags:
        parts.append(
            "SCONE bootstrap reweighting zeroed out this target's γ in >50% of bootstrap "
            "replicates, indicating the causal signal is not robust to β noise."
        )
    elif "low_bootstrap_confidence" in scone_flags:
        parts.append(
            "Bootstrap confidence is below the 0.5 threshold. The causal edge is sensitive "
            "to perturbation in β estimates."
        )

    if width is not None and width > _WIDE_CI_THRESHOLD:
        parts.append(
            f"The 95% CI for OTA γ spans {width:.3f} — uncertainty is high relative to the "
            "point estimate."
        )
    elif ota_sigma > 0.1:
        parts.append(
            f"Delta-method SE = {ota_sigma:.3f}; propagated uncertainty from β and γ estimates "
            "is non-trivial."
        )

    if not parts:
        return "No major evidence vulnerability detected."

    return " ".join(parts)


def _rank_stability(target: dict) -> tuple[str, str]:
    """
    Returns (stability_label, rationale).

    STABLE:           Tier1/2 + high SCONE confidence; rank would survive evidence removal
    FRAGILE:          bootstrap_rejected or Tier3/virtual; rank loss likely if evidence removed
    TIER-DEPENDENT:   Moderate confidence with dependency on a single tier
    """
    tier = target.get("evidence_tier", "provisional_virtual")
    scone_confidence = target.get("scone_confidence")
    scone_flags = target.get("scone_flags") or []

    if "bootstrap_rejected" in scone_flags:
        return (
            "FRAGILE",
            f"SCONE bootstrap rejection means the target's ranking depends entirely on "
            f"the {tier} β estimate. Removing that tier would eliminate the target.",
        )

    if tier in _FRAGILE_TIERS:
        return (
            "TIER-DEPENDENT",
            f"Ranking is gated on {tier} evidence. A shift to Tier1/2 would increase "
            "confidence; removal would likely move this target out of top-5.",
        )

    if scone_confidence is not None and scone_confidence < _LOW_CONFIDENCE_THRESHOLD:
        return (
            "FRAGILE",
            f"SCONE confidence of {scone_confidence:.2f} indicates bootstrap instability. "
            "The causal edge is sensitive to β noise across programs.",
        )

    if tier in ("Tier1_Interventional", "Tier2_Convergent"):
        label = "STABLE"
        rationale = (
            f"Backed by {tier} evidence with"
            + (f" SCONE confidence {scone_confidence:.2f}." if scone_confidence is not None else " no SCONE data.")
        )
    else:
        label = "TIER-DEPENDENT"
        rationale = (
            f"Moderate confidence. {tier} evidence provides a signal, "
            "but cross-tier validation would strengthen ranking stability."
        )

    return label, rationale


def _counterargument(target: dict, literature_confidence: str | None) -> str:
    """Strongest single counterargument against pursuing this target."""
    gene = target.get("target_gene", "GENE")
    tier = target.get("evidence_tier", "provisional_virtual")
    scone_flags = target.get("scone_flags") or []
    pli = target.get("pli")
    max_phase = target.get("max_phase", 0)
    flags = target.get("flags", [])
    target_score = target.get("target_score", 0.0)

    # Priority order: bootstrap rejection > lit contradiction > safety > tier weakness > score

    if "bootstrap_rejected" in scone_flags:
        return (
            f"SCONE bootstrap analysis rejected {gene}'s causal edge (confidence < 0.50). "
            "The apparent γ may reflect estimation noise rather than true causal signal, "
            "making this a high-risk first-in-class bet without orthogonal validation."
        )

    if literature_confidence == "CONTRADICTED":
        return (
            f"Published literature explicitly contradicts {gene} as a causal driver of "
            "this disease. Proceeding would require explaining published negative data — "
            "a significant regulatory and reputational risk."
        )

    if pli is not None and pli > 0.9:
        return (
            f"{gene} has pLI = {pli:.2f}, indicating it is an essential gene with high "
            "on-target toxicity risk. Selective inhibition may be intractable, and "
            "therapeutic window is likely narrow."
        )

    if tier in _FRAGILE_TIERS:
        return (
            f"{gene}'s causal evidence is {tier} — derived from computational inference "
            "rather than direct perturbation experiments. Without a Tier1/2 β estimate, "
            "the causal edge could be confounded by co-expression or pathway proximity."
        )

    if "low_bootstrap_confidence" in scone_flags:
        return (
            f"{gene} shows low SCONE bootstrap confidence, meaning its causal γ is sensitive "
            "to β estimation noise. Small changes in Perturb-seq effect sizes could materially "
            "alter the ranking."
        )

    if max_phase == 0 and target_score < 0.3:
        return (
            f"{gene} is unvalidated clinically (max phase 0) with a relatively low target "
            "score ({target_score:.3f}). The risk/reward profile for first-in-class drug "
            "development is unfavourable without stronger causal evidence."
        )

    return (
        f"No dominant vulnerability identified for {gene}. The strongest caveat is "
        "the inherent uncertainty in translating causal graph edges to clinical efficacy — "
        "OTA γ predicts population-level genetic causality, not pharmacological response."
    )


def _red_team_verdict(
    confidence_level: str,
    stability: str,
    literature_confidence: str | None,
) -> str:
    if confidence_level == "REJECTED":
        return "DEPRIORITIZE"
    if literature_confidence == "CONTRADICTED":
        return "DEPRIORITIZE"
    if confidence_level == "LOW" or stability == "FRAGILE":
        return "CAUTION"
    return "PROCEED"


# ---------------------------------------------------------------------------
# Main run()
# ---------------------------------------------------------------------------

def _reshape_gamma_for_trait(
    gamma_estimates: dict,
    trait: str,
) -> dict[str, dict]:
    """
    Extract per-program γ for a single trait from the full gamma_estimates dict.

    gamma_estimates structure: {program: {trait: value}} where value is
    float, dict{"gamma": float, ...}, or nested dict.

    Returns {program: {"gamma": float, "evidence_tier": str}}.
    """
    result: dict[str, dict] = {}
    for prog, prog_gammas in gamma_estimates.items():
        if not isinstance(prog_gammas, dict):
            continue
        g_val = prog_gammas.get(trait, prog_gammas.get("gamma", 0.0))
        if isinstance(g_val, (int, float)):
            result[prog] = {"gamma": float(g_val), "evidence_tier": "Tier3_Provisional"}
        elif isinstance(g_val, dict):
            result[prog] = g_val
    return result


def _run_counterfactuals(
    gene: str,
    disease_query: dict,
    beta_matrix: dict,
    gamma_estimates: dict,
) -> dict | None:
    """
    Run 50% inhibition and full knockout simulations for a gene.

    Returns {"inhibition_50pct": ..., "knockout": ..., "primary_trait": str}
    or None if beta_matrix/gamma_estimates are unavailable.
    """
    gene_betas = beta_matrix.get(gene)
    if not gene_betas or not gamma_estimates:
        return None

    # Determine primary trait: first trait from gamma_estimates programs
    primary_trait: str | None = None
    for prog_gammas in gamma_estimates.values():
        if isinstance(prog_gammas, dict):
            for t in prog_gammas:
                primary_trait = t
                break
        if primary_trait:
            break

    # Prefer disease-name-derived trait
    disease_name = (disease_query or {}).get("disease_name", "")
    from graph.schema import DISEASE_TRAIT_MAP, _DISEASE_SHORT_NAMES_FOR_ANCHORS
    short = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_name.lower(), "")
    traits = DISEASE_TRAIT_MAP.get(short, [])
    if traits:
        primary_trait = traits[0]

    if not primary_trait:
        return None

    trait_gammas = _reshape_gamma_for_trait(gamma_estimates, primary_trait)
    if not trait_gammas:
        return None

    from pipelines.counterfactual import simulate_perturbation_with_uncertainty
    try:
        inhibition = simulate_perturbation_with_uncertainty(
            gene=gene,
            delta_beta_fraction=-0.5,
            beta_estimates=gene_betas,
            gamma_estimates=trait_gammas,
            trait=primary_trait,
            n_bootstrap=50,
        )
        knockout = simulate_perturbation_with_uncertainty(
            gene=gene,
            delta_beta_fraction=-1.0,
            beta_estimates=gene_betas,
            gamma_estimates=trait_gammas,
            trait=primary_trait,
            n_bootstrap=50,
        )
        return {
            "primary_trait":    primary_trait,
            "inhibition_50pct": {
                "baseline_gamma":   inhibition["baseline_gamma"],
                "perturbed_gamma":  inhibition["perturbed_gamma"],
                "delta_gamma":      inhibition["delta_gamma"],
                "percent_change":   inhibition["percent_change"],
                "ci_lower":         inhibition.get("ci_lower"),
                "ci_upper":         inhibition.get("ci_upper"),
                "se_perturbed":     inhibition.get("se_perturbed"),
                "uncertainty_note": inhibition.get("uncertainty_note"),
                "interpretation":   inhibition["interpretation"],
                "dominant_program": inhibition["dominant_program"],
            },
            "knockout": {
                "baseline_gamma":   knockout["baseline_gamma"],
                "perturbed_gamma":  knockout["perturbed_gamma"],
                "delta_gamma":      knockout["delta_gamma"],
                "percent_change":   knockout["percent_change"],
                "ci_lower":         knockout.get("ci_lower"),
                "ci_upper":         knockout.get("ci_upper"),
                "se_perturbed":     knockout.get("se_perturbed"),
                "uncertainty_note": knockout.get("uncertainty_note"),
                "interpretation":   knockout["interpretation"],
            },
        }
    except Exception as exc:
        return {"error": str(exc), "primary_trait": primary_trait}


def run(
    prioritization_result: dict,
    literature_result: dict | None = None,
    disease_query: dict | None = None,
    *,
    beta_matrix_result: dict | None = None,
    gamma_estimates: dict | None = None,
) -> dict:
    """
    Adversarial red team assessment for top-5 targets.

    Args:
        prioritization_result: Output of target_prioritization_agent (requires "targets").
        literature_result:     Optional Phase Q output — used to flag CONTRADICTED targets.
        disease_query:         Disease context dict.
        beta_matrix_result:    Optional beta_matrix_result from Tier 2; enables counterfactuals.
        gamma_estimates:       Optional _gamma_estimates {program: {trait: ...}}; enables counterfactuals.

    Returns:
        {
            "red_team_assessments": list[dict],   per-target adversarial analysis
            "n_targets_assessed":   int,
            "n_flagged_caution":    int,
            "n_flagged_deprioritize": int,
            "overall_confidence":   "HIGH" | "MODERATE" | "LOW",
            "red_team_summary":     str,
        }
    """
    targets = prioritization_result.get("targets", [])
    top5 = sorted(targets, key=lambda t: t.get("rank", 999))[:5]

    lit_evidence = (literature_result or {}).get("literature_evidence", {})
    disease_name = (disease_query or {}).get("disease_name", "the disease")
    beta_matrix = (beta_matrix_result or {}).get("beta_matrix", {})
    gamma_est = gamma_estimates or {}

    assessments: list[dict] = []

    for target in top5:
        gene = target.get("target_gene", "")
        scone_confidence = target.get("scone_confidence")
        scone_flags = target.get("scone_flags") or []
        ci_lower = target.get("ota_gamma_ci_lower")
        ci_upper = target.get("ota_gamma_ci_upper")

        lit_ev = lit_evidence.get(gene, {})
        lit_confidence = lit_ev.get("literature_confidence")

        conf_level = _confidence_level(scone_confidence, scone_flags)
        stability, stability_rationale = _rank_stability(target)
        vulnerability = _evidence_vulnerability(target)
        counterarg = _counterargument(target, lit_confidence)
        verdict = _red_team_verdict(conf_level, stability, lit_confidence)
        width = _ci_width(ci_lower, ci_upper)

        lit_flag: str | None = None
        if lit_confidence == "CONTRADICTED":
            lit_flag = f"Literature CONTRADICTED: published data refutes {gene} as a {disease_name} driver."

        # Phase S counterfactual simulation
        counterfactual = _run_counterfactuals(gene, disease_query or {}, beta_matrix, gamma_est)

        assessments.append({
            "target_gene":              gene,
            "rank":                     target.get("rank", 0),
            "scone_confidence":         scone_confidence,
            "scone_flags":              scone_flags,
            "ota_gamma_ci_lower":       ci_lower,
            "ota_gamma_ci_upper":       ci_upper,
            "ci_width":                 width,
            "confidence_level":         conf_level,
            "evidence_vulnerability":   vulnerability,
            "counterargument":          counterarg,
            "rank_stability":           stability,
            "rank_stability_rationale": stability_rationale,
            "literature_flag":          lit_flag,
            "red_team_verdict":         verdict,
            "counterfactual":           counterfactual,
        })

    n_caution      = sum(1 for a in assessments if a["red_team_verdict"] == "CAUTION")
    n_deprioritize = sum(1 for a in assessments if a["red_team_verdict"] == "DEPRIORITIZE")
    n_proceed      = sum(1 for a in assessments if a["red_team_verdict"] == "PROCEED")

    # Overall confidence: dominated by worst-case target
    if n_deprioritize > 0:
        overall = "LOW"
    elif n_caution >= 2:
        overall = "LOW"
    elif n_caution == 1:
        overall = "MODERATE"
    else:
        overall = "HIGH"

    # Summary paragraph
    summary_parts: list[str] = []
    if n_proceed > 0:
        proceed_genes = [a["target_gene"] for a in assessments if a["red_team_verdict"] == "PROCEED"]
        summary_parts.append(
            f"{', '.join(proceed_genes)} passed red team review with no major vulnerabilities."
        )
    if n_caution > 0:
        caution_genes = [a["target_gene"] for a in assessments if a["red_team_verdict"] == "CAUTION"]
        summary_parts.append(
            f"{', '.join(caution_genes)} require additional validation before advancing — "
            "causal evidence is present but bootstrap stability or evidence tier is marginal."
        )
    if n_deprioritize > 0:
        deprio_genes = [a["target_gene"] for a in assessments if a["red_team_verdict"] == "DEPRIORITIZE"]
        summary_parts.append(
            f"{', '.join(deprio_genes)} are flagged for deprioritisation: "
            "SCONE bootstrap rejection or literature contradiction undermines the causal case."
        )

    red_team_summary = " ".join(summary_parts) or "No targets assessed."

    return {
        "red_team_assessments":    assessments,
        "n_targets_assessed":      len(assessments),
        "n_flagged_caution":       n_caution,
        "n_flagged_deprioritize":  n_deprioritize,
        "overall_confidence":      overall,
        "red_team_summary":        red_team_summary,
    }
