"""
scientific_writer_agent.py — Tier 5 agent: synthesis into publication-grade output.

Synthesizes all tier outputs into an executive summary, ranked target table,
causal pathway narratives, evidence quality report, and limitations section.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


PIPELINE_VERSION = "0.1.0"

# Number of top targets to write full causal narratives for
N_TOP_NARRATIVES = 3


def _format_target_table(targets: list[dict]) -> str:
    """Produce a Markdown table of the top ranked targets."""
    header = (
        "| Rank | Gene | Ota γ | Tier | OT Score | Max Phase | Key Evidence |\n"
        "|------|------|-------|------|----------|-----------|--------------|"
    )
    rows = []
    for t in targets:
        gene       = t.get("target_gene", "?")
        gamma      = t.get("ota_gamma", 0.0)
        tier       = t.get("evidence_tier", "?")
        ot         = t.get("ot_score", 0.0)
        phase      = t.get("max_phase", 0)
        evidence   = "; ".join(t.get("key_evidence", [])[:2]) or "—"
        rows.append(
            f"| {t.get('rank', '?')} | {gene} | {gamma:.3f} | {tier} "
            f"| {ot:.2f} | Phase {phase} | {evidence} |"
        )
    return header + "\n" + "\n".join(rows)


def _format_state_space_table(state_targets: list[dict]) -> str:
    """Produce a Markdown table of state-space nominated targets (separate section)."""
    header = (
        "| Rank | Gene | StateEffect | CI95 | Confidence | Score | OT Score | Max Phase |\n"
        "|------|------|------------:|:-----|-----------:|------:|---------:|----------:|"
    )
    rows = []
    for t in state_targets:
        gene = t.get("target_gene", "?")
        eff  = float(t.get("state_edge_effect") or 0.0)
        conf = float(t.get("state_edge_confidence") or 0.0)
        score = float(t.get("state_target_score") or (eff * conf))
        ci_lo = t.get("state_edge_ci_lower")
        ci_hi = t.get("state_edge_ci_upper")
        if ci_lo is not None and ci_hi is not None:
            ci_str = f"[{float(ci_lo):.3f}, {float(ci_hi):.3f}]"
        else:
            ci_str = "—"
        ot = float(t.get("ot_score") or 0.0)
        phase = int(t.get("max_phase") or 0)
        rows.append(
            f"| {t.get('rank', '?')} | {gene} | {eff:.3f} | {ci_str} | {conf:.3f} | {score:.3f} | {ot:.2f} | {phase} |"
        )
    return header + "\n" + ("\n".join(rows) if rows else "| — | — | — | — | — | — | — | — |")


def _causal_narrative(
    target: dict,
    chemistry: dict,
    trials: dict,
) -> str:
    """Generate a causal pathway narrative for a single target."""
    gene       = target.get("target_gene", "?")
    gamma      = target.get("ota_gamma", 0.0)
    tier       = target.get("evidence_tier", "?")
    programs_raw = target.get("top_programs", [])
    # top_programs may be a dict {prog: contribution} (post-fix) or a list
    programs = list(programs_raw.keys()) if isinstance(programs_raw, dict) else list(programs_raw)
    drugs      = target.get("known_drugs", [])
    flags      = target.get("flags", [])
    chem_info  = chemistry.get("target_chemistry", {}).get(gene, {})
    trial_info = trials.get("trial_summary", {}).get(gene, {})
    ic50       = chem_info.get("best_ic50_nM")
    max_phase  = trial_info.get("max_phase_reached", target.get("max_phase", 0))
    key_evidence = "; ".join(target.get("key_evidence", [])[:3]) or "GWAS + Perturb-seq"

    # Program description: use names if available, otherwise infer from flags
    if programs:
        prog_str = ", ".join(programs[:3])
    elif "chip_mechanism" in flags:
        prog_str = "inflammatory and clonal hematopoiesis programs"
    elif tier in ("Tier1_Interventional", "Tier2_Convergent"):
        prog_str = "lipid metabolism and inflammatory programs"
    else:
        prog_str = "disease-relevant biological programs (pending cNMF)"

    # Direction: infer from gamma sign
    if gamma < -0.05:
        direction = "downregulates disease-promoting"
    elif gamma > 0.05:
        direction = "upregulates disease-relevant"
    else:
        direction = "modulates"

    drug_str = drugs[0] if drugs else "no approved drug"
    ic50_str = f"IC50={ic50:.0f} nM" if ic50 else "no in vitro activity data"
    virtual_label = (
        " [provisional_virtual — in silico estimate, awaiting experimental data]"
        if tier == "provisional_virtual"
        else ""
    )

    return (
        f"{gene} {direction} {prog_str} → disease trait{virtual_label}.\n"
        f"Evidence: {key_evidence}.\n"
        f"Composite Ota estimate: γ_{{{gene}→trait}} = {gamma:.3f} ({tier}).\n"
        f"Best compound: {drug_str} ({ic50_str}; Phase {max_phase})."
    )


def _evidence_quality_section(
    causal_result: dict,
    n_virtual: int,
) -> dict:
    """Compile mandatory evidence quality metrics."""
    top_genes = causal_result.get("top_genes", [])

    n_tier1 = sum(1 for g in top_genes if g.get("tier") == "Tier1_Interventional")
    n_tier2 = sum(1 for g in top_genes if g.get("tier") == "Tier2_Convergent")
    n_tier3 = sum(
        1 for g in top_genes
        if g.get("tier") in ("Tier3_Provisional", "moderate_transferred", "moderate_grn")
    )
    n_virtual_edges = sum(1 for g in top_genes if g.get("tier") == "provisional_virtual")

    return {
        "n_tier1_edges":   n_tier1,
        "n_tier2_edges":   n_tier2,
        "n_tier3_edges":   n_tier3,
        "n_virtual_edges": n_virtual_edges,
        "min_evalue":      None,
    }


def _limitations_text(n_virtual: int) -> str:
    return (
        f"About {n_virtual} gene–program β rows are labelled `provisional_virtual` (in silico or "
        "weakly supported cell-model estimates). That label applies to those β edges, not to the "
        "entire analysis: GWAS/L2G and other tiers may still be experimentally grounded. Treat "
        "virtual-tier rows as hypothesis-generating. Full quantitative Perturb-seq β coverage "
        "may require local GEO GSE246756 downloads (~50GB; Replogle 2022) when not cached. "
        "S-LDSC γ and MR layers remain context-dependent; validate lead findings experimentally."
    )


def run(
    phenotype_result:    dict,
    genetics_result:     dict,
    beta_matrix_result:  dict,
    regulatory_result:   dict,
    causal_result:       dict,
    kg_result:           dict,
    prioritization_result: dict,
    chemistry_result:    dict,
    trials_result:       dict,
    somatic_result:      dict | None = None,  # kept for backward compat, ignored
) -> dict:
    """
    Synthesize all tier outputs into a publication-grade GraphOutput.

    Returns:
        GraphOutput-compatible dict
    """
    disease_name = phenotype_result.get("disease_name", "")
    efo_id       = phenotype_result.get("efo_id", "")
    targets      = prioritization_result.get("targets", [])
    state_space_targets = prioritization_result.get("state_space_targets", [])
    warnings_all: list[str] = []

    # Collect warnings from all tiers
    for res in [
        genetics_result, beta_matrix_result,
        regulatory_result, causal_result, kg_result,
        prioritization_result, chemistry_result, trials_result,
    ]:
        warnings_all.extend(res.get("warnings", []))

    # -------------------------------------------------------------------------
    # Executive summary
    # -------------------------------------------------------------------------
    top3 = targets[:3]
    top3_genes = [t["target_gene"] for t in top3]
    n_written = causal_result.get("n_edges_written", 0)
    n_virtual_beta = beta_matrix_result.get("n_virtual", 0)
    # Anchor recovery is optional (Tier 3 may omit it). Default to 0/12.
    _anchor = causal_result.get("anchor_recovery") or {}
    _anchor_total = _anchor.get("n_reference") or _anchor.get("total") or 12
    _anchor_recov = _anchor.get("n_recovered") or _anchor.get("recovered") or 0
    try:
        _anchor_total = int(_anchor_total)
    except Exception:
        _anchor_total = 12
    try:
        _anchor_recov = int(_anchor_recov)
    except Exception:
        _anchor_recov = 0
    recovery_rate = (_anchor_recov / _anchor_total) if _anchor_total else 0.0

    executive_summary = (
        f"Causal graph analysis of {disease_name} (EFO: {efo_id}) identified "
        f"{n_written} significant causal edges spanning {len(targets)} therapeutic targets. "
        f"Top-ranked targets are {', '.join(top3_genes) if top3_genes else 'none identified'}. "
        f"Anchor edge recovery: {recovery_rate:.0%} of 12 reference edges. "
        f"{n_virtual_beta} gene–program β estimate(s) use the `provisional_virtual` evidence tier "
        "(in silico / weak perturbation support); other evidence streams may still be experimental."
    )

    # -------------------------------------------------------------------------
    # Target table
    # -------------------------------------------------------------------------
    target_table = _format_target_table(targets[:10])
    state_space_table = _format_state_space_table(state_space_targets[:10])

    # -------------------------------------------------------------------------
    # Causal narratives for top 3
    # -------------------------------------------------------------------------
    top_target_narratives: list[str] = []
    for target in top3:
        narrative = _causal_narrative(target, chemistry_result, trials_result)
        top_target_narratives.append(narrative)

    # -------------------------------------------------------------------------
    # Evidence quality
    # -------------------------------------------------------------------------
    evidence_quality = _evidence_quality_section(
        causal_result, n_virtual_beta
    )

    # -------------------------------------------------------------------------
    # Limitations
    # -------------------------------------------------------------------------
    limitations = _limitations_text(n_virtual_beta)

    # -------------------------------------------------------------------------
    # TargetRecord-compatible list
    # -------------------------------------------------------------------------
    target_list = [
        {
            "target_gene":        t["target_gene"],
            "rank":               t["rank"],
            "target_score":       t["target_score"],
            "ota_gamma":          t["ota_gamma"],
            "ota_gamma_raw":      t.get("ota_gamma_raw"),
            "ota_gamma_sigma":    t.get("ota_gamma_sigma"),
            "ota_gamma_ci_lower": t.get("ota_gamma_ci_lower"),
            "ota_gamma_ci_upper": t.get("ota_gamma_ci_upper"),
            "scone_confidence":   t.get("scone_confidence"),
            "scone_flags":        t.get("scone_flags", []),
            "evidence_tier":      t["evidence_tier"],
            "ot_score":           t["ot_score"],
            "max_phase":          t["max_phase"],
            "known_drugs":        t["known_drugs"],
            "pli":                t.get("pli"),
            "flags":              t.get("flags", []),
            "top_programs":       t.get("top_programs", []),
            "key_evidence":       t.get("key_evidence", []),
            # Phase R: disease-state τ specificity — direct keys take precedence (cover GWAS genes
            # without TR), TR dict is fallback (covers state-space genes patched into TR)
            "tau_disease_specificity": (
                t.get("tau_disease_specificity")
                or (t.get("therapeutic_redirection_result") or {}).get("tau_disease_specificity")
            ),
            "tau_disease_log2fc": (
                t.get("disease_log2fc")
                or (t.get("therapeutic_redirection_result") or {}).get("disease_log2fc")
            ),
            "tau_specificity_class": (
                t.get("tau_specificity_class")
                or (t.get("therapeutic_redirection_result") or {}).get("tau_specificity_class")
            ),
            # Multi-dimensional ranking fields (Phase 39 redesign)
            # beta_amd_concentration: pleiotropic diagnostic — fraction of β-mass in AMD-specific
            # programs; low = spread across generic programs (e.g. JAZF1). Reporting only.
            "beta_amd_concentration":  t.get("beta_amd_concentration"),
            "causal_gamma":            t.get("causal_gamma"),
            "genetic_evidence_score":  t.get("genetic_evidence_score"),
            "ot_genetic_score":        t.get("ot_genetic_score"),
            "partition":               t.get("partition"),
            "mechanistic_score":       t.get("mechanistic_score"),
            "therapeutic_redirection": t.get("therapeutic_redirection"),
            "stability_score":         t.get("stability_score"),
            "entry_score":             t.get("entry_score"),
            "persistence_score":       t.get("persistence_score"),
            "recovery_score":          t.get("recovery_score"),
            "boundary_score":          t.get("boundary_score"),
            "marker_score":            t.get("marker_score"),
            "specificity_score":       t.get("specificity_score"),
            "brg_score":               t.get("brg_score"),
            "bimodality_coeff":        t.get("bimodality_coeff"),
            "tau_specificity":         t.get("tau_specificity"),
            # Program classification (Phase Z6)
            "ot_l2g_score":            t.get("ot_l2g_score"),
            "program_drivers":         t.get("program_drivers"),
        }
        for t in targets
    ]

    state_space_target_list = [
        {
            "target_gene": t.get("target_gene"),
            "rank": t.get("rank"),
            "evidence_tier": t.get("evidence_tier", "state_nominated"),
            "evidence_type": t.get("evidence_type", "state_space"),
            "state_edge_effect": t.get("state_edge_effect"),
            "state_edge_confidence": t.get("state_edge_confidence"),
            "state_target_score": t.get("state_target_score"),
            "state_edge_ci_lower": t.get("state_edge_ci_lower"),
            "state_edge_ci_upper": t.get("state_edge_ci_upper"),
            "state_edge_cv": t.get("state_edge_cv"),
            "genetic_evidence_score": t.get("genetic_evidence_score"),
            "ot_score": t.get("ot_score"),
            "max_phase": t.get("max_phase"),
            "known_drugs": t.get("known_drugs", []),
            "controller_annotation": t.get("controller_annotation"),
            "therapeutic_redirection_result": t.get("therapeutic_redirection_result"),
            "evidence_disagreement": t.get("evidence_disagreement", []),
            "scone_confidence": t.get("scone_confidence"),
            "scone_flags": t.get("scone_flags", []),
            "tau_disease_specificity": t.get("tau_disease_specificity"),
            "tau_disease_log2fc": t.get("disease_log2fc"),
            "tau_specificity_class": t.get("tau_specificity_class"),
            "beta_amd_concentration": t.get("beta_amd_concentration"),
        }
        for t in state_space_targets
    ]

    # -------------------------------------------------------------------------
    # Evidence profiles — three categories replace the single ranked list.
    #
    # Genetic anchors:       OT / pipeline genetic score ≥ 0.5 (strong L2G proxy).
    # Locus-nominated:       score in [0.05, 0.5). Sub-sorted by disease-specific |γ|.
    # Repurposing opportunities: Genetic anchor + any drug in clinical trials
    #                        (max_phase ≥ 1). Phase ≥ 1 rather than ≥ 2 to capture
    #                        earlier-stage compounds with strong genetic backing.
    #                        This reveals non-obvious repurposing beyond known drugs.
    # Mechanistic candidates: Mechanistic-only targets (partition == mechanistic_only).
    #                        Hypothesis-generating; require orthogonal validation.
    #
    # A target can appear in multiple sections. Sections do NOT compete.
    # -------------------------------------------------------------------------
    def _is_strong_anchor(t: dict) -> bool:
        """OT L2G ≥ 0.5: unambiguous causal gene attribution at GWAS locus."""
        return (t.get("ot_genetic_score") or 0.0) >= 0.5

    def _is_locus_nominated(t: dict) -> bool:
        """OT genetic score 0.05–0.5: complex GWAS loci (CFB, C2, ADAMTS9).
        Previously missed because prioritization used max_targets=20; now fixed."""
        s = t.get("ot_genetic_score") or 0.0
        return 0.05 <= s < 0.5

    def _is_genetic_anchor(t: dict) -> bool:
        return _is_strong_anchor(t) or _is_locus_nominated(t)

    def _disease_specific_gamma(t: dict) -> float:
        """Sort key: disease-specific gamma = |ota_gamma| × (disease_specific_pct / 100).
        Deprioritises targets whose gamma comes mostly from generic programs."""
        gamma = abs(t.get("ota_gamma") or 0.0)
        drivers = t.get("program_drivers") or {}
        # disease_specific_pct preferred; amd_specific_pct kept as backward-compat alias
        pct = drivers.get("disease_specific_pct") or drivers.get("amd_specific_pct") or 0
        return gamma * (pct / 100.0) if pct > 0 else gamma
    # backward-compat alias used in sort calls below
    _amd_specific_gamma = _disease_specific_gamma

    # High-confidence anchors (L2G ≥ 0.5)
    genetic_anchors: list[dict] = sorted(
        [t for t in target_list if _is_strong_anchor(t)],
        key=_amd_specific_gamma,
        reverse=True,
    )

    # Locus-nominated anchors (complex GWAS loci, 0.05 ≤ L2G < 0.5)
    locus_nominated_anchors: list[dict] = sorted(
        [t for t in target_list if _is_locus_nominated(t)],
        key=_amd_specific_gamma,
        reverse=True,
    )

    # Repurposing: genetic anchor + drug in clinical trials globally,
    # but NOT already in disease-specific trials (those are validation, not repurposing).
    repurposing_opportunities: list[dict] = sorted(
        [
            t for t in target_list
            if (t.get("max_phase") or 0) >= 1
            and _is_genetic_anchor(t)
            and t.get("known_drugs")
            and not t.get("already_in_disease_trial")
        ],
        key=lambda x: (x.get("max_phase") or 0, abs(x.get("ota_gamma") or 0.0)),
        reverse=True,
    )

    # Disease-specific pipeline: drugs already in trials for THIS disease.
    # Framework validation signal — genetic causal evidence agrees with active clinical development.
    disease_specific_pipeline: list[dict] = sorted(
        [t for t in target_list if t.get("already_in_disease_trial")],
        key=lambda x: abs(x.get("ota_gamma") or 0.0),
        reverse=True,
    )

    mechanistic_candidates: list[dict] = sorted(
        [t for t in target_list if t.get("partition") == "mechanistic_only"],
        key=lambda x: abs(x.get("ota_gamma") or 0.0),
        reverse=True,
    )

    return {
        "disease_name":          disease_name,
        "efo_id":                efo_id,
        "target_list":           target_list,
        "state_space_targets":   state_space_target_list,
        # Evidence profiles (preferred output — replaces single ranked table)
        "genetic_anchors":           genetic_anchors,        # OT L2G ≥ 0.5
        "locus_nominated_anchors":   locus_nominated_anchors, # complex GWAS loci 0.05–0.5
        "repurposing_opportunities": repurposing_opportunities,
        "disease_specific_pipeline": disease_specific_pipeline,  # validation: drug already in disease trials
        "mechanistic_candidates":    mechanistic_candidates,
        "n_tier1_edges":         evidence_quality["n_tier1_edges"],
        "n_tier2_edges":         evidence_quality["n_tier2_edges"],
        "n_tier3_edges":         evidence_quality["n_tier3_edges"],
        "n_virtual_edges":       evidence_quality["n_virtual_edges"],
        "executive_summary":     executive_summary,
        "target_table":          target_table,
        "state_space_table":     state_space_table,
        "top_target_narratives": top_target_narratives,
        "evidence_quality":      evidence_quality,
        "limitations":           limitations,
        "pipeline_version":      PIPELINE_VERSION,
        "generated_at":          datetime.now(tz=timezone.utc).isoformat(),
        "warnings":              warnings_all,
    }
