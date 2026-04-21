"""
agents/cso/chief_of_staff_agent.py — CSO as Active Reasoning Hub (Phase P).

The CSO synthesises across pipeline tiers at three decision points:

  1. run_briefing(disease_query)
     Called BEFORE Tier 1. Identifies known scientific challenges,
     recommends tissues, sets anchor gene expectations, emits tier_guidance
     that gets injected into downstream agents via _cso_guidance.

  2. run_conflict_analysis(pipeline_outputs)
     Called AFTER Tier 3. Compares GWAS top genes vs Perturb-seq top
     regulators. Identifies overlap and divergence; formulates a written
     hypothesis about mechanistic levels when they differ.

  3. run_exec_summary(pipeline_outputs)
     Called AFTER Tier 5 + re-delegation. Produces a PI-ready executive
     summary: top insight, confidence assessment, recommended next
     experiments.

Local mode uses deterministic Python synthesis. SDK mode (set via
runner.set_mode("chief_of_staff_agent", "sdk")) delegates to Claude for
richer reasoning on unusual patterns.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipelines.evidence_landscape import build_evidence_landscape, summarize_landscape

# ---------------------------------------------------------------------------
# Disease → tissue priors
# ---------------------------------------------------------------------------

_DISEASE_TISSUE_PRIORS: dict[str, list[str]] = {
    "inflammatory bowel disease": ["colon", "ileum", "small_intestine", "immune_cell"],
    "ibd":                        ["colon", "ileum", "small_intestine", "immune_cell"],
    "crohn":                      ["ileum", "small_intestine", "colon"],
    "ulcerative colitis":         ["colon", "rectum"],
    "coronary artery disease":    ["heart", "artery", "blood_vessel"],
    "cad":                        ["heart", "artery", "blood_vessel"],
    "type 2 diabetes":            ["pancreas", "liver", "adipose"],
    "alzheimer":                  ["brain", "hippocampus", "cortex"],
    "rheumatoid arthritis":       ["synovium", "blood", "immune_cell"],
}

_DISEASE_AREA_MAP: dict[str, str] = {
    "inflammatory bowel disease": "autoimmune/inflammatory",
    "ibd":                        "autoimmune/inflammatory",
    "crohn":                      "autoimmune/inflammatory",
    "ulcerative colitis":         "autoimmune/inflammatory",
    "coronary artery disease":    "cardiovascular",
    "cad":                        "cardiovascular",
    "type 2 diabetes":            "metabolic",
    "alzheimer":                  "neurological",
    "rheumatoid arthritis":       "autoimmune/inflammatory",
}

_DISEASE_ANCHOR_EXPECTATIONS: dict[str, list[str]] = {
    "inflammatory bowel disease": ["NOD2", "IL23R", "ATG16L1", "TNF"],
    "ibd":                        ["NOD2", "IL23R", "ATG16L1", "TNF"],
    "coronary artery disease":    ["PCSK9", "LDLR", "HMGCR"],
    "cad":                        ["PCSK9", "LDLR", "HMGCR"],
    "type 2 diabetes":            ["TCF7L2", "GCK", "PPARG"],
    "alzheimer":                  ["APOE", "APP", "PSEN1"],
}

_DISEASE_KNOWN_CHALLENGES: dict[str, list[str]] = {
    "inflammatory bowel disease": [
        "Heterogeneous disease (CD vs UC) with distinct causal architectures",
        "Multiple cell types implicated — requires cell-type-specific β estimates",
        "Known NOD2/IL23R anchors should recover > 80%; failure indicates tissue mismatch",
    ],
    "coronary artery disease": [
        "GWAS instruments are LDL-pathway dominant — non-lipid mechanisms may be missed",
        "Plaque biology requires vascular smooth muscle and macrophage cell types",
        "PCSK9 is the gold-standard positive control — must appear in top targets",
    ],
    "type 2 diabetes": [
        "Pancreatic islet cell heterogeneity complicates β estimation",
        "Liver and adipose tissue have independent causal contributions",
    ],
}


def _infer_disease_key(disease_name: str) -> str:
    """Normalise disease name to a known key, or return the original."""
    dn = disease_name.lower()
    for key in sorted(_DISEASE_TISSUE_PRIORS, key=len, reverse=True):
        if key in dn:
            return key
    return dn


# ---------------------------------------------------------------------------
# 1. Pre-pipeline briefing
# ---------------------------------------------------------------------------

def run_briefing(disease_query: dict) -> dict:
    """
    CSO briefing called before Tier 1.

    Analyses the disease query and produces:
      - key_challenges:            known scientific difficulties
      - recommended_tissues:       cell/tissue types to prioritise in h5ad lookups
      - anchor_gene_expectations:  expected positive-control genes for anchor recovery
      - tier_guidance:             per-tier hints injected into downstream agents
      - briefing_notes:            one-sentence framing for the PI log
    """
    disease_name = disease_query.get("disease_name", "unknown disease")
    efo_id       = disease_query.get("efo_id", "")
    key          = _infer_disease_key(disease_name)

    recommended_tissues    = _DISEASE_TISSUE_PRIORS.get(key, [])
    anchor_expectations    = _DISEASE_ANCHOR_EXPECTATIONS.get(key, [])
    known_challenges       = _DISEASE_KNOWN_CHALLENGES.get(key, [
        "No prior knowledge loaded for this disease — apply general causal discovery protocol.",
    ])
    disease_area           = _DISEASE_AREA_MAP.get(key, "unknown")

    tissue_hint = (
        f"Prioritise these tissues for β estimation: {', '.join(recommended_tissues[:3])}."
        if recommended_tissues else
        "No tissue priors available — use default tissue sweep."
    )

    tier_guidance: dict[str, str] = {
        "tier1": (
            f"Disease area: {disease_area}. "
            + tissue_hint
        ),
        "tier2": (
            f"Focus Perturb-seq β on cell types: {', '.join(recommended_tissues[:2]) or 'all available'}. "
            "Prioritise Tier 1 Perturb-seq over virtual evidence."
        ),
        "tier3": (
            f"Expect anchor genes: {', '.join(anchor_expectations[:4]) or 'none pre-loaded'}. "
            "Recovery < 80% indicates tissue or gene-list mismatch — investigate before proceeding."
        ),
        "tier4": (
            "Confirm tractability for top-3 targets before ranking finalises. "
            "Flag undruggable targets early for deprioritisation."
        ),
    }

    briefing_notes = (
        f"CSO briefing for {disease_name} ({efo_id or 'EFO unknown'}): "
        f"{disease_area} disease. "
        f"{'Anchor genes pre-loaded: ' + ', '.join(anchor_expectations[:3]) + '.' if anchor_expectations else 'No anchor priors — build from GWAS.'}"
    )

    return {
        "disease_name":           disease_name,
        "disease_area":           disease_area,
        "key_challenges":         known_challenges,
        "recommended_tissues":    recommended_tissues,
        "anchor_gene_expectations": anchor_expectations,
        "tier_guidance":          tier_guidance,
        "briefing_notes":         briefing_notes,
    }


# ---------------------------------------------------------------------------
# 2. Post-tier-3 conflict analysis
# ---------------------------------------------------------------------------

def run_conflict_analysis(pipeline_outputs: dict) -> dict:
    """
    CSO conflict analysis called after Tier 3.

    Compares GWAS-derived top genes vs Perturb-seq top regulators.
    When they diverge, formulates a written hypothesis about what the
    divergence means scientifically.

    Returns:
      gwas_top_genes, perturbseq_top_regulators, overlap_genes,
      divergence_detected, divergence_hypothesis,
      recommended_focus_genes, evidence_conflict_notes
    """
    genetics_result   = pipeline_outputs.get("genetics_result", {})
    beta_result       = pipeline_outputs.get("beta_matrix_result", {})
    causal_result     = pipeline_outputs.get("causal_result", {})
    regulator_nom     = pipeline_outputs.get("regulator_nomination_evidence", {})

    # Extract GWAS top genes (genetic instrument genes)
    gwas_genes: list[str] = []
    for g in genetics_result.get("top_genes", []):
        sym = g.get("gene_symbol") or g.get("gene") if isinstance(g, dict) else g
        if sym:
            gwas_genes.append(str(sym))

    # Extract Perturb-seq regulator nominations
    perturbseq_regs: list[str] = []
    for nom in regulator_nom.get("nominators", []):
        gene = nom.get("gene") or nom.get("gene_symbol") if isinstance(nom, dict) else nom
        if gene:
            perturbseq_regs.append(str(gene))
    # Also check beta_result for Tier1 genes
    for gene, tier in beta_result.get("evidence_tier_per_gene", {}).items():
        if tier in ("Tier1_Interventional", "Tier1_Perturb_seq") and gene not in perturbseq_regs:
            perturbseq_regs.append(gene)

    # Compute overlap
    gwas_set       = set(g.upper() for g in gwas_genes)
    perturb_set    = set(g.upper() for g in perturbseq_regs)
    overlap_set    = gwas_set & perturb_set
    overlap_genes  = sorted(overlap_set)

    divergence_detected = (
        len(gwas_set) > 0
        and len(perturb_set) > 0
        and len(overlap_set) < min(len(gwas_set), len(perturb_set)) * 0.4
    )

    # Get anchor recovery for confidence framing
    anchor_recovery = causal_result.get("anchor_recovery", {}).get("recovery_rate", 0.0)

    # Formulate divergence hypothesis
    if not gwas_genes and not perturbseq_regs:
        divergence_hypothesis = "Insufficient data from both tracks to perform conflict analysis."
        evidence_conflict_notes = "No genes from either track — pipeline may have stalled in Tier 1/2."
    elif not gwas_genes:
        divergence_hypothesis = (
            "No GWAS instruments recovered — all candidate targets come from Perturb-seq. "
            "These are transcriptional regulators of disease-relevant programs, not genetic risk genes. "
            "Interpret as mechanistic targets rather than genetically-validated disease genes."
        )
        evidence_conflict_notes = "GWAS track empty — verify GWAS server connectivity and trait mapping."
    elif not perturbseq_regs:
        divergence_hypothesis = (
            "No Perturb-seq regulators nominated — all targets are GWAS-derived. "
            "Genetic evidence is strong but mechanistic pathway context is missing. "
            "Consider expanding h5ad dataset coverage."
        )
        evidence_conflict_notes = "Perturb-seq track empty — verify h5ad dataset availability for this disease."
    elif overlap_genes:
        gwas_ex   = [g for g in gwas_genes[:3] if g.upper() not in overlap_set]
        perturb_ex = [g for g in perturbseq_regs[:3] if g.upper() not in overlap_set]
        divergence_hypothesis = (
            f"Convergent evidence: {', '.join(overlap_genes[:5])} appear in both GWAS and Perturb-seq tracks. "
            "These are the highest-confidence targets — genetic risk + functional validation. "
            + (
                f"GWAS-only genes ({', '.join(gwas_ex[:3])}) are genetic risk genes without Perturb-seq validation — "
                "may require cell-type expansion. "
                if gwas_ex else ""
            )
            + (
                f"Perturb-seq-only regulators ({', '.join(perturb_ex[:3])}) are transcriptional controllers "
                "without GWAS support — strong mechanistic rationale but higher genetic uncertainty."
                if perturb_ex else ""
            )
        )
        evidence_conflict_notes = f"No true conflict — {len(overlap_genes)} convergent gene(s) detected."
    else:
        # Full divergence — most interesting case
        gwas_str    = ", ".join(gwas_genes[:5])
        perturb_str = ", ".join(perturbseq_regs[:5])
        divergence_hypothesis = (
            f"Track divergence detected: GWAS identified {gwas_str} as genetic risk genes "
            f"(germline variant → disease susceptibility), while Perturb-seq nominated "
            f"{perturb_str} as transcriptional regulators of disease-relevant cellular programs. "
            "This divergence likely reflects two complementary mechanistic levels: "
            "GWAS genes represent upstream genetic predisposition (variant-driven risk), "
            "Perturb-seq regulators represent downstream cellular execution (program control). "
            "Both tracks are scientifically valid. "
            "Prioritise convergent candidates if they emerge after causal scoring; "
            "treat GWAS-only and Perturb-seq-only genes as complementary hypotheses, "
            "not contradictory evidence."
        )
        evidence_conflict_notes = (
            f"GWAS track: {len(gwas_genes)} genes. "
            f"Perturb-seq track: {len(perturbseq_regs)} regulators. "
            "Overlap: 0 — recommend verifying both tracks are using the same gene namespace."
        )

    # Recommended focus: convergent genes first, then GWAS (genetically validated)
    recommended_focus = overlap_genes + [g for g in gwas_genes if g.upper() not in overlap_set]

    return {
        "gwas_top_genes":          gwas_genes,
        "perturbseq_top_regulators": perturbseq_regs,
        "overlap_genes":           overlap_genes,
        "divergence_detected":     divergence_detected,
        "divergence_hypothesis":   divergence_hypothesis,
        "recommended_focus_genes": recommended_focus[:10],
        "evidence_conflict_notes": evidence_conflict_notes,
        "anchor_recovery":         anchor_recovery,
    }


# ---------------------------------------------------------------------------
# 3. Post-tier-5 executive summary
# ---------------------------------------------------------------------------

def run_exec_summary(pipeline_outputs: dict) -> dict:
    """
    CSO executive summary called after Tier 5 + re-delegation.

    Synthesises the full pipeline into a PI-ready briefing:
      - executive_summary:      2–3 sentence narrative for the PI
      - top_insight:            single most important finding
      - confidence_assessment:  HIGH / MEDIUM / LOW with rationale
      - next_experiments:       list of specific follow-up proposals
      - pipeline_health:        structured QC summary
    """
    disease_name     = (
        pipeline_outputs.get("phenotype_result", {}).get("disease_name")
        or pipeline_outputs.get("disease_name")
        or "unknown disease"
    )
    # Support both full pipeline_outputs (local mode) and scoped input (SDK dispatch via _run_cso_exec_summary)
    targets          = (
        pipeline_outputs.get("prioritization_result", {}).get("targets", [])
        or pipeline_outputs.get("top_targets", [])
    )
    reviewer_result  = pipeline_outputs.get("review_result", {})
    # Also support scoped input where reviewer verdict/counts are flat keys
    if not reviewer_result and "reviewer_verdict" in pipeline_outputs:
        reviewer_result = {
            "verdict":    pipeline_outputs.get("reviewer_verdict", "UNKNOWN"),
            "n_critical": pipeline_outputs.get("n_critical", 0),
            "n_major":    pipeline_outputs.get("n_major", 0),
            "issues":     pipeline_outputs.get("open_issues", []),
        }
    anchor_recovery  = (
        pipeline_outputs.get("causal_result", {})
        .get("anchor_recovery", {}).get("recovery_rate", 0.0)
        or pipeline_outputs.get("anchor_recovery", 0.0)
    )
    score_adjustments = pipeline_outputs.get("prioritization_result", {}).get("score_adjustments", [])
    redelegation_actions = pipeline_outputs.get("_redelegation_actions", [])

    reviewer_verdict = reviewer_result.get("verdict", "UNKNOWN")
    n_critical       = reviewer_result.get("n_critical", 0)
    n_major          = reviewer_result.get("n_major", 0)

    all_targets      = sorted(targets, key=lambda t: t.get("rank", 99))
    n_targets        = len(all_targets)

    # Build evidence landscape summary for CSO synthesis
    # Use pre-built landscape if available (from _build_final_output), else compute
    landscape_data   = pipeline_outputs.get("evidence_landscape") or {}
    if landscape_data and isinstance(landscape_data, dict):
        landscape_summary = landscape_data.get("summary") or {}
        profiles          = landscape_data.get("profiles", [])
    else:
        try:
            profiles = build_evidence_landscape(pipeline_outputs)
            landscape_summary = summarize_landscape(profiles)
        except Exception:
            profiles = []
            landscape_summary = {}

    n_convergent     = landscape_summary.get("by_class", {}).get("convergent", 0)
    n_genetic_anchor = landscape_summary.get("by_class", {}).get("genetic_anchor", 0)
    n_perturb_only   = landscape_summary.get("by_class", {}).get("perturb_seq_regulator", 0)
    n_state_nom      = landscape_summary.get("by_class", {}).get("state_nominated", 0)
    n_lit_supported  = landscape_summary.get("n_with_literature_support", 0)
    n_with_cf        = landscape_summary.get("n_with_counterfactual", 0)

    # Confidence assessment
    if anchor_recovery >= 0.80 and reviewer_verdict == "APPROVE" and n_critical == 0:
        confidence     = "HIGH"
        confidence_rationale = (
            f"Anchor recovery={anchor_recovery:.0%}, reviewer verdict=APPROVE, "
            f"no critical issues."
        )
    elif anchor_recovery >= 0.80 and n_critical == 0:
        confidence     = "MEDIUM"
        confidence_rationale = (
            f"Anchor recovery={anchor_recovery:.0%} but {n_major} major reviewer issue(s). "
            "Results are usable — address major issues before clinical prioritisation."
        )
    else:
        confidence     = "LOW"
        confidence_rationale = (
            f"Anchor recovery={anchor_recovery:.0%} and/or {n_critical} critical reviewer issue(s). "
            "Results require additional validation before clinical prioritisation."
        )

    # Score adjustment narrative
    adjusted_genes = [a["gene"] for a in score_adjustments]
    adjustment_note = (
        f" Chemistry/clinical feedback flagged {', '.join(adjusted_genes)}."
        if adjusted_genes else ""
    )

    # Top insight: describe the evidence landscape, not a single top gene
    if profiles:
        # Find strongest convergent or genetic_anchor gene by ota_gamma
        def _gamma(p: dict) -> float:
            return abs((p.get("genetic_evidence") or {}).get("ota_gamma") or 0.0)
        genetic_profiles = [p for p in profiles if p["evidence_class"] in ("convergent", "genetic_anchor")]
        if genetic_profiles:
            anchor = max(genetic_profiles, key=_gamma)
            anchor_gene = anchor["gene"]
            anchor_gamma = _gamma(anchor)
            anchor_tier = (anchor.get("genetic_evidence") or {}).get("evidence_tier", "")
            convergence_note = (
                " (convergent genetic + Perturb-seq)"
                if anchor["evidence_class"] == "convergent"
                else f" ({anchor_tier})"
            )
            top_insight = (
                f"{anchor_gene} has the strongest genetic causal signal "
                f"(γ={anchor_gamma:.3f}{convergence_note})."
            )
        else:
            top_insight = (
                f"No genes with genetic causal instruments identified; "
                f"{n_state_nom} state-nominated candidates require experimental validation."
            )
    elif all_targets:
        top_gene = all_targets[0].get("target_gene", "unknown")
        top_insight = f"{top_gene} is the highest-scored target by internal ranking."
    else:
        top_insight = "No targets — pipeline produced no prioritised genes."

    # Landscape narrative
    landscape_clauses: list[str] = []
    if n_convergent:
        landscape_clauses.append(f"{n_convergent} gene(s) with convergent genetic+Perturb-seq evidence")
    if n_genetic_anchor:
        landscape_clauses.append(f"{n_genetic_anchor} gene(s) with genetic causal instrument only")
    if n_perturb_only:
        landscape_clauses.append(f"{n_perturb_only} upstream Perturb-seq regulator(s) without strong GWAS link")
    if n_state_nom:
        landscape_clauses.append(f"{n_state_nom} state-nominated exploratory candidate(s)")
    landscape_narrative = (
        "Evidence landscape: " + "; ".join(landscape_clauses) + "."
        if landscape_clauses else ""
    )
    lit_note = (
        f" {n_lit_supported} gene(s) have literature support; "
        f"{n_with_cf} have counterfactual perturbation predictions."
        if n_lit_supported or n_with_cf else ""
    )

    # Executive summary
    reviewer_phrase = (
        "Reviewer approved the pipeline with no critical issues."
        if reviewer_verdict == "APPROVE" and n_critical == 0
        else f"Reviewer flagged {n_critical} critical and {n_major} major issues."
    )
    redel_phrase = (
        f" {len(redelegation_actions)} agent(s) were re-delegated to address issues."
        if redelegation_actions else ""
    )

    executive_summary = (
        f"{disease_name.title()} pipeline analysed {n_targets} candidate gene(s). "
        f"{landscape_narrative}{lit_note}{adjustment_note} "
        f"{reviewer_phrase}{redel_phrase} "
        f"Overall confidence: {confidence}."
    )

    # Next experiments — sample across evidence classes, not purely by rank
    next_experiments: list[str] = []

    def _experiment_line(p: dict) -> str:
        gene  = p["gene"]
        cls   = p["evidence_class"]
        phase = (p.get("translational_evidence") or {}).get("max_phase", 0) or 0
        adv   = (p.get("adversarial_assessment") or {}).get("verdict")
        if adv == "DEPRIORITIZE":
            return f"{gene}: red team DEPRIORITIZE — address adversarial concerns before advancing."
        if phase >= 3:
            return f"{gene}: approved/Phase III compound — evaluate repurposing; search ongoing trials."
        if phase >= 1:
            return f"{gene}: Phase {phase} compound — review trial design; assess dose/biomarker strategy."
        if cls in ("convergent", "genetic_anchor"):
            return (
                f"{gene}: strong causal evidence, no clinical compound — "
                "ChEMBL screen or structure-based design recommended."
            )
        if cls == "perturb_seq_regulator":
            return (
                f"{gene}: upstream Perturb-seq regulator without genetic instrument — "
                "CRISPR KO in disease-relevant cell type to confirm causal role."
            )
        return (
            f"{gene}: state-nominated (exploratory) — "
            "Perturb-seq with functional disease readout required before prioritisation."
        )

    # Prioritise: convergent → genetic_anchor → perturb_seq_regulator → state_nominated
    _class_order = ["convergent", "genetic_anchor", "perturb_seq_regulator", "state_nominated", "gwas_provisional"]
    _by_class: dict[str, list] = {c: [] for c in _class_order}
    for p in profiles:
        cls = p.get("evidence_class", "state_nominated")
        _by_class.get(cls, _by_class["state_nominated"]).append(p)

    seen: set[str] = set()
    for cls in _class_order:
        for p in _by_class[cls][:2]:  # up to 2 per class
            if len(next_experiments) >= 5:
                break
            line = _experiment_line(p)
            if line not in seen:
                next_experiments.append(line)
                seen.add(line)
        if len(next_experiments) >= 5:
            break

    # Fallback: if profiles were unavailable, generate from raw target list
    if not next_experiments and all_targets:
        for t in all_targets[:5]:
            gene  = t.get("target_gene", "?")
            phase = t.get("max_phase", 0) or 0
            tier  = t.get("evidence_tier", "")
            if phase >= 3:
                next_experiments.append(
                    f"{gene}: approved/Phase III compound — evaluate repurposing; search ongoing trials."
                )
            elif phase >= 1:
                next_experiments.append(
                    f"{gene}: Phase {phase} compound — review trial design; assess dose/biomarker strategy."
                )
            elif tier in ("Tier1_Interventional", "Tier2_Convergent", "Tier1_Perturb_seq", "Tier2_eQTL_MR"):
                next_experiments.append(
                    f"{gene}: strong causal evidence, no clinical compound — "
                    "ChEMBL screen or structure-based design recommended."
                )
            else:
                next_experiments.append(
                    f"{gene}: Perturb-seq validation in disease-relevant cell type required before prioritisation."
                )

    # If state_nominated dominate, add a summary note instead of listing each
    n_state_listed = sum(1 for e in next_experiments if "state-nominated" in e)
    if n_state_nom > n_state_listed + 1:
        next_experiments.append(
            f"{n_state_nom - n_state_listed} additional state-nominated candidate(s) — "
            "all require Perturb-seq validation before clinical prioritisation."
        )

    # Reviewer issues that remain unresolved
    open_issues = [
        i for i in reviewer_result.get("issues", [])
        if i.get("severity") in ("CRITICAL", "MAJOR")
    ]
    if open_issues:
        next_experiments.append(
            f"Address {len(open_issues)} open reviewer issue(s) before finalising report: "
            + "; ".join(i["check"] for i in open_issues[:3])
        )

    pipeline_health = {
        "anchor_recovery":     anchor_recovery,
        "reviewer_verdict":    reviewer_verdict,
        "n_critical_issues":   n_critical,
        "n_major_issues":      n_major,
        "redelegation_rounds": len(redelegation_actions),
        "score_adjustments":   len(score_adjustments),
    }

    return {
        "executive_summary":     executive_summary,
        "top_insight":           top_insight,
        "confidence_assessment": confidence,
        "confidence_rationale":  confidence_rationale,
        "next_experiments":      next_experiments,
        "pipeline_health":       pipeline_health,
    }


# ---------------------------------------------------------------------------
# Unified entry point for AgentRunner._call_local dispatch
# ---------------------------------------------------------------------------

def run(disease_query: dict, upstream_results: dict) -> dict:
    """
    Unified entry point called by AgentRunner.

    Dispatch is determined by upstream_results["_cso_mode"]:
      "briefing"          → run_briefing(disease_query)
      "conflict_analysis" → run_conflict_analysis(upstream_results)
      "exec_summary"      → run_exec_summary(upstream_results)
    """
    mode = upstream_results.get("_cso_mode", "briefing")
    if mode == "briefing":
        return run_briefing(disease_query)
    if mode == "conflict_analysis":
        return run_conflict_analysis(upstream_results)
    if mode == "exec_summary":
        return run_exec_summary(upstream_results)
    raise ValueError(f"chief_of_staff_agent.run: unknown _cso_mode={mode!r}")
