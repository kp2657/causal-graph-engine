"""
pipelines/evidence_disagreement.py

Rule-based evidence disagreement and transportability detection.

Four rules (locked design decisions):

  1. genetics_vs_perturbation
     sign(MR-IVW gamma) ≠ sign(T1/T2 ConditionalBeta)
     Genetic instruments point one direction; perturbation evidence points the other.
     Indicates possible reverse causation, confounding, or cell-type mismatch in the MR.

  2. perturbation_vs_chemical
     sign(Perturb-seq beta) ≠ sign(LINCS L1000 beta)
     Direct genetic perturbation (Perturb-seq KO) disagrees with chemical perturbation (LINCS).
     Indicates on-target vs off-target divergence or incomplete phenocopy.

  3. bulk_vs_singlecell
     LINCS |β| > 0.3 (strong bulk signal) but sc beta near-zero (|β| < 0.05).
     Bulk signal does not replicate in single-cell disease-relevant context.
     Indicates cell-type dilution in bulk or cell-type-specific resistance.

  4. cross_context_sign_flip
     context-verified ConditionalBetas flip sign across cell types for the same program.
     Indicates the gene plays opposing roles in different disease-relevant compartments.
     Strong transportability concern for cell-type-agnostic drug targeting.

All rules return EvidenceDisagreementRecord objects (models/latent_mediator.py).
Severity levels: "warning" (informational) | "flag" (review required) | "block" (do not translate).

Phase I adds build_disagreement_profile() — structured 5-dimension profile with 6 pattern categories.

Public API:
    check_genetics_vs_perturbation(gene, mr_gamma, conditional_betas)
    check_perturbation_vs_chemical(gene, disease, perturb_beta, lincs_beta)
    check_bulk_vs_singlecell(gene, disease, lincs_beta, sc_betas)
    check_cross_context_sign_flip(gene, disease, conditional_betas)
    run_all_disagreement_checks(gene, disease, ...)
    build_disagreement_profile(gene, disease, ...)            ← Phase I
"""
from __future__ import annotations

import math
from typing import Any

from models.latent_mediator import ConditionalBeta, EvidenceDisagreementRecord

# Thresholds (consistent with CLAUDE.md and design doc)
_LINCS_STRONG_THRESHOLD = 0.30    # |LINCS beta| > this = "strong bulk signal"
_SC_NEAR_ZERO_THRESHOLD = 0.05    # |sc beta| < this = "near zero"
_SIGN_FLIP_MIN_BETAS = 2          # minimum context-verified betas to call a sign flip


def _sign(v: float) -> int:
    """Return +1, -1, or 0 for NaN/zero."""
    if not math.isfinite(v) or v == 0.0:
        return 0
    return 1 if v > 0 else -1


# ---------------------------------------------------------------------------
# Rule 1: genetics_vs_perturbation
# ---------------------------------------------------------------------------

def check_genetics_vs_perturbation(
    gene: str,
    disease: str,
    mr_gamma: float,
    conditional_betas: list[ConditionalBeta],
    program_id: str | None = None,
) -> EvidenceDisagreementRecord | None:
    """
    Detect sign conflict between MR-IVW gamma and T1/T2 perturbation betas.

    Args:
        gene:               Gene symbol.
        disease:            Short disease key.
        mr_gamma:           Signed MR-IVW gamma for gene → disease (from ota_gamma_estimation).
        conditional_betas:  ConditionalBeta list for this gene.
        program_id:         If set, restrict to betas for this program only.

    Returns EvidenceDisagreementRecord if sign conflict detected, else None.
    """
    if not math.isfinite(mr_gamma) or mr_gamma == 0.0:
        return None

    # Collect T1/T2 betas (perturbational evidence; not pooled proxies)
    tier_ok = {"Tier1_TrajectoryDirect", "Tier1_Interventional",
               "Tier2_TrajectoryInferred", "Tier2_Convergent"}
    relevant = [
        cb for cb in conditional_betas
        if math.isfinite(cb.beta)
        and cb.evidence_tier in tier_ok
        and (program_id is None or cb.program_id == program_id)
    ]
    if not relevant:
        return None

    # Use the mean beta (sign-preserving) of all relevant T1/T2 estimates
    mean_beta = sum(cb.beta for cb in relevant) / len(relevant)
    if mean_beta == 0.0:
        return None

    if _sign(mr_gamma) != _sign(mean_beta):
        severity = "flag" if len(relevant) >= 2 else "warning"
        return EvidenceDisagreementRecord(
            gene=gene,
            disease=disease,
            rule="genetics_vs_perturbation",
            value_a=mr_gamma,
            value_b=mean_beta,
            explanation=(
                f"MR-IVW gamma={mr_gamma:+.3f} vs mean T1/T2 beta={mean_beta:+.3f}; "
                f"n_betas={len(relevant)}. "
                "Possible reverse causation, confounding, or cell-type mismatch in MR."
            ),
            severity=severity,
        )
    return None


# ---------------------------------------------------------------------------
# Rule 2: perturbation_vs_chemical
# ---------------------------------------------------------------------------

def check_perturbation_vs_chemical(
    gene: str,
    disease: str,
    perturb_beta: float,
    lincs_beta: float,
    program_id: str | None = None,
    cell_type: str | None = None,
) -> EvidenceDisagreementRecord | None:
    """
    Detect sign conflict between Perturb-seq beta and LINCS L1000 beta.

    Args:
        gene:          Gene symbol.
        disease:       Short disease key.
        perturb_beta:  Beta from Perturb-seq (T1/T2 ConditionalBeta).
        lincs_beta:    Beta from LINCS L1000 (T3 ConditionalBeta or raw overlap score).
        program_id:    Program context (for explanation only).
        cell_type:     Cell type context (for explanation only).

    Returns EvidenceDisagreementRecord if sign conflict detected, else None.
    """
    if not math.isfinite(perturb_beta) or not math.isfinite(lincs_beta):
        return None
    if perturb_beta == 0.0 or lincs_beta == 0.0:
        return None

    if _sign(perturb_beta) != _sign(lincs_beta):
        return EvidenceDisagreementRecord(
            gene=gene,
            disease=disease,
            rule="perturbation_vs_chemical",
            value_a=perturb_beta,
            value_b=lincs_beta,
            cell_type_a=cell_type,
            explanation=(
                f"Perturb-seq beta={perturb_beta:+.3f} vs LINCS beta={lincs_beta:+.3f}"
                + (f" in {cell_type}" if cell_type else "")
                + (f" for {program_id}" if program_id else "")
                + ". On-target vs off-target divergence or incomplete phenocopy."
            ),
            severity="flag",
        )
    return None


# ---------------------------------------------------------------------------
# Rule 3: bulk_vs_singlecell
# ---------------------------------------------------------------------------

def check_bulk_vs_singlecell(
    gene: str,
    disease: str,
    lincs_beta: float,
    sc_betas: list[ConditionalBeta],
    program_id: str | None = None,
) -> EvidenceDisagreementRecord | None:
    """
    Detect bulk-strong / sc-near-zero discordance.

    Args:
        gene:         Gene symbol.
        disease:      Short disease key.
        lincs_beta:   LINCS L1000 beta (bulk perturbation).
        sc_betas:     ConditionalBeta list from single-cell sources (T1/T2).
        program_id:   Program context.

    Returns EvidenceDisagreementRecord if |LINCS| > 0.3 and mean|sc| < 0.05, else None.
    """
    if not math.isfinite(lincs_beta):
        return None
    if abs(lincs_beta) <= _LINCS_STRONG_THRESHOLD:
        return None

    relevant_sc = [
        cb for cb in sc_betas
        if math.isfinite(cb.beta)
        and cb.evidence_tier in {"Tier1_TrajectoryDirect", "Tier1_Interventional",
                                  "Tier2_TrajectoryInferred", "Tier2_Convergent"}
        and (program_id is None or cb.program_id == program_id)
    ]
    if not relevant_sc:
        return None

    mean_sc_abs = sum(abs(cb.beta) for cb in relevant_sc) / len(relevant_sc)
    if mean_sc_abs < _SC_NEAR_ZERO_THRESHOLD:
        return EvidenceDisagreementRecord(
            gene=gene,
            disease=disease,
            rule="bulk_vs_singlecell",
            value_a=lincs_beta,
            value_b=mean_sc_abs,
            explanation=(
                f"LINCS |beta|={abs(lincs_beta):.3f} (>{_LINCS_STRONG_THRESHOLD}) "
                f"but mean sc |beta|={mean_sc_abs:.4f} (<{_SC_NEAR_ZERO_THRESHOLD}); "
                f"n_sc={len(relevant_sc)}. "
                "Bulk signal does not replicate in disease-relevant single-cell context. "
                "Possible cell-type dilution in bulk or context-specific resistance."
            ),
            severity="flag",
        )
    return None


# ---------------------------------------------------------------------------
# Rule 4: cross_context_sign_flip
# ---------------------------------------------------------------------------

def check_cross_context_sign_flip(
    gene: str,
    disease: str,
    conditional_betas: list[ConditionalBeta],
    program_id: str | None = None,
) -> EvidenceDisagreementRecord | None:
    """
    Detect sign flip of context-verified betas across cell types.

    Only considers betas where context_verified=True (cell-type-specific T1 data).
    At least 2 context-verified betas from different cell types are required.

    Returns EvidenceDisagreementRecord with cell_type_a / cell_type_b showing
    the first conflicting pair, else None.
    """
    verified = [
        cb for cb in conditional_betas
        if cb.context_verified
        and math.isfinite(cb.beta)
        and cb.beta != 0.0
        and (program_id is None or cb.program_id == program_id)
    ]

    # Deduplicate to one beta per cell type (use first encountered)
    seen_ct: dict[str, ConditionalBeta] = {}
    for cb in verified:
        if cb.cell_type not in seen_ct:
            seen_ct[cb.cell_type] = cb

    if len(seen_ct) < _SIGN_FLIP_MIN_BETAS:
        return None

    cell_types = list(seen_ct.keys())
    for i in range(len(cell_types)):
        for j in range(i + 1, len(cell_types)):
            ct_a = cell_types[i]
            ct_b = cell_types[j]
            beta_a = seen_ct[ct_a].beta
            beta_b = seen_ct[ct_b].beta
            if _sign(beta_a) != _sign(beta_b):
                return EvidenceDisagreementRecord(
                    gene=gene,
                    disease=disease,
                    rule="cross_context_sign_flip",
                    value_a=beta_a,
                    value_b=beta_b,
                    cell_type_a=ct_a,
                    cell_type_b=ct_b,
                    explanation=(
                        f"Context-verified beta={beta_a:+.3f} in {ct_a} vs "
                        f"beta={beta_b:+.3f} in {ct_b}. "
                        "Gene plays opposing roles across disease-relevant cell types. "
                        "Strong transportability concern for cell-type-agnostic targeting."
                    ),
                    severity="block",   # highest severity — directly contradictory evidence
                )
    return None


# ---------------------------------------------------------------------------
# Public API: run all checks for a gene
# ---------------------------------------------------------------------------

def run_all_disagreement_checks(
    gene: str,
    disease: str,
    conditional_betas: list[ConditionalBeta],
    *,
    mr_gamma: float | None = None,
    lincs_betas_by_program: dict[str, float] | None = None,
    program_id: str | None = None,
) -> list[EvidenceDisagreementRecord]:
    """
    Run all four disagreement rules for a gene and return every detected conflict.

    Args:
        gene:                    Gene symbol.
        disease:                 Short disease key.
        conditional_betas:       All ConditionalBeta objects for this gene.
        mr_gamma:                Signed MR-IVW gamma (optional; skips rule 1 if None).
        lincs_betas_by_program:  {program_id: LINCS_beta} (optional; needed for rules 2+3).
        program_id:              If set, restrict to this program only.

    Returns:
        List of EvidenceDisagreementRecord (may be empty).
    """
    records: list[EvidenceDisagreementRecord] = []

    # Rule 1: genetics vs perturbation
    if mr_gamma is not None:
        r1 = check_genetics_vs_perturbation(gene, disease, mr_gamma, conditional_betas, program_id)
        if r1:
            records.append(r1)

    # Rules 2 + 3: need LINCS betas
    if lincs_betas_by_program:
        # Collect T1/T2 perturb betas for comparison
        perturb_tier = {"Tier1_TrajectoryDirect", "Tier1_Interventional",
                        "Tier2_TrajectoryInferred", "Tier2_Convergent"}
        t1t2_betas = [
            cb for cb in conditional_betas
            if math.isfinite(cb.beta) and cb.evidence_tier in perturb_tier
            and (program_id is None or cb.program_id == program_id)
        ]

        for prog, lincs_beta in lincs_betas_by_program.items():
            if program_id is not None and prog != program_id:
                continue
            if not math.isfinite(lincs_beta):
                continue

            # Use mean of T1/T2 betas as "perturb_beta" for rule 2
            if t1t2_betas:
                mean_perturb = sum(cb.beta for cb in t1t2_betas) / len(t1t2_betas)
                r2 = check_perturbation_vs_chemical(gene, disease, mean_perturb, lincs_beta,
                                                     program_id=prog)
                if r2:
                    records.append(r2)

            # Rule 3: bulk vs sc
            r3 = check_bulk_vs_singlecell(gene, disease, lincs_beta, conditional_betas,
                                           program_id=prog)
            if r3:
                records.append(r3)

    # Rule 4: cross-context sign flip
    r4 = check_cross_context_sign_flip(gene, disease, conditional_betas, program_id)
    if r4:
        records.append(r4)

    return records


# ---------------------------------------------------------------------------
# Phase I — Structured DisagreementProfile
# ---------------------------------------------------------------------------

_HIGH_TIERS: frozenset[str] = frozenset({
    "Tier1_Interventional", "Tier1_TrajectoryDirect",
    "Tier2_Convergent", "Tier2_TrajectoryInferred",
    "moderate_transferred", "moderate_grn",
})

_TIER_WEIGHT: dict[str, float] = {
    "Tier1_Interventional":    1.00,
    "Tier1_TrajectoryDirect":  1.00,
    "Tier2_Convergent":        0.80,
    "Tier2_TrajectoryInferred": 0.75,
    "moderate_transferred":    0.60,
    "moderate_grn":            0.50,
    "Tier3_Provisional":       0.35,
    "Tier3_TrajectoryProxy":   0.30,
    "provisional_virtual":     0.10,
}

_SUPPORT_THRESHOLD = 0.35   # dimension score above this = "actively supports"


def _dim_genetics(
    mr_gamma: float | None,
    evidence_tier: str,
) -> tuple[float, list[str], list[str]]:
    """genetics_support dimension [0,1]."""
    tw = _TIER_WEIGHT.get(evidence_tier, 0.20)
    if mr_gamma is None or not math.isfinite(mr_gamma):
        gamma_score = 0.0
        g_items: list[str] = ["no_gamma"]
    else:
        gamma_score = min(abs(mr_gamma) / 0.7, 1.0)
        g_items = [f"gamma={mr_gamma:+.3f}"]
    score = 0.5 * gamma_score + 0.5 * tw
    support = g_items + [f"tier={evidence_tier}(w={tw:.2f})"]
    contra: list[str] = []
    if tw <= 0.10:
        contra.append("virtual_tier")
    if gamma_score < 0.1:
        contra.append("near_zero_gamma")
    return score, support, contra


def _dim_expression(
    transition_profile: Any | None,
    das: float,
) -> tuple[float, list[str], list[str]]:
    """expression_coupling dimension [0,1] from Phase G or DAS fallback."""
    if transition_profile is not None:
        candidates = [
            ("entry",       transition_profile.entry_score),
            ("persistence", transition_profile.persistence_score),
            ("recovery",    transition_profile.recovery_score),
        ]
        best_name, best_score = max(candidates, key=lambda x: x[1])
        boundary_contrib = transition_profile.boundary_score * 0.5
        score = max(best_score, boundary_contrib)
        support = [f"dominant={best_name}({best_score:.2f})",
                   f"boundary={transition_profile.boundary_score:.2f}"]
        contra: list[str] = []
        if score < 0.10:
            contra.append("weak_transition_signal")
        return score, support, contra
    # DAS fallback
    score = min(abs(das), 1.0)
    support = [f"das={das:.3f}"] if das > 0.01 else []
    contra = [] if score > 0.1 else ["no_expression_signal"]
    return score, support, contra


def _dim_perturbation(
    conditional_betas: list[ConditionalBeta],
) -> tuple[float, list[str], list[str]]:
    """perturbation_support dimension [0,1]."""
    t1t2 = [
        cb for cb in conditional_betas
        if math.isfinite(cb.beta) and cb.beta != 0.0
        and cb.evidence_tier in _HIGH_TIERS
    ]
    if not t1t2:
        return 0.05, [], ["no_t1t2_perturbation"]
    mean_beta = sum(cb.beta for cb in t1t2) / len(t1t2)
    if mean_beta == 0.0:
        return 0.05, [], ["zero_mean_beta"]
    consistent = sum(1 for cb in t1t2 if _sign(cb.beta) == _sign(mean_beta))
    consistency = consistent / len(t1t2)
    coverage = min(len(t1t2) / 3.0, 1.0)
    score = consistency * coverage
    support = [f"n_t1t2={len(t1t2)}", f"consistency={consistency:.2f}",
               f"mean_beta={mean_beta:+.3f}"]
    contra: list[str] = []
    if consistency < 0.70:
        contra.append(f"low_consistency({consistency:.2f})")
    return score, support, contra


def _dim_cell_type_specificity(
    conditional_betas: list[ConditionalBeta],
) -> tuple[float, list[str], list[str]]:
    """cell_type_specificity dimension [0,1]. Returns 0.5 (neutral) when data insufficient."""
    verified = [
        cb for cb in conditional_betas
        if cb.context_verified and math.isfinite(cb.beta)
    ]
    if len(verified) < 2:
        return 0.5, ["insufficient_ct_data(neutral_prior)"], []
    # Coefficient of variation of |beta| across cell types
    abs_betas = [abs(cb.beta) for cb in verified]
    mean_abs = sum(abs_betas) / len(abs_betas)
    var = sum((b - mean_abs) ** 2 for b in abs_betas) / len(abs_betas)
    std = var ** 0.5
    cv = std / (mean_abs + 1e-8)
    score = min(cv, 1.0)
    n_ct = len({cb.cell_type for cb in verified})
    support = [f"n_cell_types={n_ct}", f"cv={cv:.2f}"]
    contra: list[str] = []
    if mean_abs < 0.05:
        contra.append("near_zero_betas")
    return score, support, contra


def _dim_cross_context(
    disagreement_records: list[EvidenceDisagreementRecord],
) -> tuple[float, list[str], list[str]]:
    """cross_context_consistency dimension [0,1]. Penalised by disagreement records."""
    score = 1.0
    support: list[str] = []
    contra: list[str] = []
    if not disagreement_records:
        support.append("no_disagreement")
        return score, support, contra
    for rec in disagreement_records:
        if rec.severity == "block":
            score -= 0.50
            contra.append(f"block:{rec.rule}")
        elif rec.severity == "flag":
            score -= 0.25
            contra.append(f"flag:{rec.rule}")
        else:
            score -= 0.10
            contra.append(f"warning:{rec.rule}")
    return max(0.0, score), support, contra


def _assign_label(
    profile_scores: dict[str, float],
    mr_gamma: float | None,
    conditional_betas: list[ConditionalBeta],
    transition_profile: Any | None,
    controller_annotation: Any | None,
    has_rule4: bool,
) -> tuple[str, float]:
    """
    Assign mechanistic label from strict rules (priority order).

    Rules (evaluated top-to-bottom; first match wins):
      discordant           sign(genetics) ≠ sign(perturbation)
      context_dependent    Rule 4 (cross_context_sign_flip) fired
      likely_upstream_controller  T1/T2 perturbation OR (early_pt AND tf_ann AND entry>0.2)
      likely_marker        persistence>0.5 AND late_pt AND virtual_tier AND cl<0.3
      likely_non_transportable  expression_coupling>0.5 AND cross_context_consistency<0.3
      supported            ≥4 dimensions above threshold
      unknown              none of the above
    """
    # context_dependent: Rule 4 fired — takes priority over discordant
    # (a cross-context sign flip explains any sign disagreement with gamma)
    if has_rule4:
        return "context_dependent", 0.90

    # discordant: genetics ≠ perturbation sign
    if mr_gamma is not None and math.isfinite(mr_gamma) and abs(mr_gamma) > 0.05:
        t1t2 = [cb for cb in conditional_betas
                 if math.isfinite(cb.beta) and cb.beta != 0.0
                 and cb.evidence_tier in _HIGH_TIERS]
        if t1t2:
            mean_beta = sum(cb.beta for cb in t1t2) / len(t1t2)
            if _sign(mr_gamma) != _sign(mean_beta):
                confidence = 0.85 if len(t1t2) >= 2 else 0.65
                return "discordant", confidence

    # likely_upstream_controller
    has_t1t2 = any(cb.evidence_tier in _HIGH_TIERS for cb in conditional_betas
                   if math.isfinite(cb.beta))
    is_profile_controller = False
    if (transition_profile is not None and controller_annotation is not None
            and transition_profile.entry_score > 0.20):
        sigs = controller_annotation.get("supporting_signals", []) \
            if isinstance(controller_annotation, dict) \
            else (controller_annotation.supporting_signals or [])
        has_tf = "tf_annotation" in sigs
        has_early = any("early_pseudotime" in s for s in sigs)
        is_profile_controller = has_tf and has_early

    if has_t1t2 or is_profile_controller:
        confidence = 0.75 if has_t1t2 else 0.55
        return "likely_upstream_controller", confidence

    # likely_marker: persistence>0.5 AND late pseudotime AND virtual tier AND cl<0.3
    if transition_profile is not None and transition_profile.persistence_score > 0.50:
        is_virtual = not has_t1t2
        is_low_controller = True
        if controller_annotation is not None:
            cl = controller_annotation.get("controller_likelihood", 0.0) \
                if isinstance(controller_annotation, dict) \
                else controller_annotation.controller_likelihood
            is_low_controller = cl < 0.30
        is_late = False
        if controller_annotation is not None:
            sigs = controller_annotation.get("supporting_signals", []) \
                if isinstance(controller_annotation, dict) \
                else (controller_annotation.supporting_signals or [])
            is_late = any("late_pseudotime" in s for s in sigs)
        if is_virtual and is_low_controller and is_late:
            return "likely_marker", 0.75

    # likely_non_transportable
    if (profile_scores.get("expression_coupling", 0) > 0.50
            and profile_scores.get("cross_context_consistency", 1.0) < 0.30):
        return "likely_non_transportable", 0.60

    # supported: ≥4 dimensions above threshold
    n_above = sum(
        1 for v in profile_scores.values()
        if v > _SUPPORT_THRESHOLD
    )
    if n_above >= 4:
        return "supported", min(n_above / 5.0, 1.0)

    return "unknown", 0.0


def build_disagreement_profile(
    gene: str,
    disease: str,
    conditional_betas: list[ConditionalBeta],
    *,
    mr_gamma: float | None = None,
    evidence_tier: str = "provisional_virtual",
    transition_profile: Any | None = None,
    controller_annotation: Any | None = None,
    disease_axis_score: float = 0.0,
    lincs_betas_by_program: dict[str, float] | None = None,
) -> "DisagreementProfile":
    """
    Build a structured Phase I DisagreementProfile for a gene.

    Computes 5 independent dimensions, then assigns one of 6 pattern labels
    using strict priority rules (see module docstring).

    Args:
        gene:                    Gene symbol.
        disease:                 Short disease key (e.g. "IBD").
        conditional_betas:       All ConditionalBeta objects for this gene.
        mr_gamma:                Signed OTA γ (optional).
        evidence_tier:           Best evidence tier for this gene (from Tier 3).
        transition_profile:      Phase G TransitionGeneProfile (optional).
        controller_annotation:   Phase H ControllerAnnotation or its model_dump() (optional).
        disease_axis_score:      Legacy DAS fallback when no transition_profile (optional).
        lincs_betas_by_program:  {program_id: LINCS_beta} for rules 2+3 (optional).

    Returns:
        DisagreementProfile with all 5 dimension scores, mechanistic_label, label_confidence.
    """
    from models.evidence import DisagreementProfile

    # Run existing 4 disagreement rules
    records = run_all_disagreement_checks(
        gene, disease, conditional_betas,
        mr_gamma=mr_gamma,
        lincs_betas_by_program=lincs_betas_by_program,
    )
    has_rule4 = any(r.rule == "cross_context_sign_flip" for r in records)

    # Compute 5 dimensions
    g_s, g_sup, g_con = _dim_genetics(mr_gamma, evidence_tier)
    e_s, e_sup, e_con = _dim_expression(transition_profile, disease_axis_score)
    p_s, p_sup, p_con = _dim_perturbation(conditional_betas)
    ct_s, ct_sup, ct_con = _dim_cell_type_specificity(conditional_betas)
    cc_s, cc_sup, cc_con = _dim_cross_context(records)

    all_support = g_sup + e_sup + p_sup + ct_sup + cc_sup
    all_contra  = g_con + e_con + p_con + ct_con + cc_con

    profile_scores = {
        "genetics_support":         round(g_s, 4),
        "expression_coupling":      round(e_s, 4),
        "perturbation_support":     round(p_s, 4),
        "cell_type_specificity":    round(ct_s, 4),
        "cross_context_consistency": round(cc_s, 4),
    }

    label, confidence = _assign_label(
        profile_scores, mr_gamma, conditional_betas,
        transition_profile, controller_annotation, has_rule4,
    )

    return DisagreementProfile(
        gene=gene,
        disease=disease,
        **profile_scores,
        mechanistic_label=label,
        label_confidence=round(confidence, 4),
        supporting_evidence=all_support,
        contradicting_evidence=all_contra,
    )
