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


def _causal_narrative(
    target: dict,
    chemistry: dict,
    trials: dict,
) -> str:
    """Generate a causal pathway narrative for a single target."""
    gene       = target.get("target_gene", "?")
    gamma      = target.get("ota_gamma", 0.0)
    tier       = target.get("evidence_tier", "?")
    programs   = target.get("top_programs", [])
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
    elif "genetic_anchor" in flags:
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
    anchor_recovery: dict,
) -> dict:
    """Compile mandatory evidence quality metrics."""
    n_written  = causal_result.get("n_edges_written", 0)
    shd        = causal_result.get("shd", 0)
    top_genes  = causal_result.get("top_genes", [])

    n_tier1 = sum(1 for g in top_genes if g.get("tier") == "Tier1_Interventional")
    n_tier2 = sum(1 for g in top_genes if g.get("tier") == "Tier2_Convergent")
    n_tier3 = sum(
        1 for g in top_genes
        if g.get("tier") in ("Tier3_Provisional", "moderate_transferred", "moderate_grn")
    )
    n_virtual_edges = sum(1 for g in top_genes if g.get("tier") == "provisional_virtual")

    recovery_rate = anchor_recovery.get("recovery_rate", 0.0)
    n_anchors = 12  # total defined in schema

    return {
        "anchor_edge_recovery_rate": f"{recovery_rate:.0%} ({int(recovery_rate * n_anchors)}/{n_anchors} anchors)",
        "n_tier1_edges":   n_tier1,
        "n_tier2_edges":   n_tier2,
        "n_tier3_edges":   n_tier3,
        "n_virtual_edges": n_virtual_edges,
        "min_evalue":      None,  # STUB: requires per-edge E-value storage
        "shd_from_reference": shd,
    }


def _limitations_text(n_virtual: int) -> str:
    return (
        f"The current analysis includes {n_virtual} provisional_virtual edges where no experimental "
        "perturbation or genetic instrument data is available. These edges rely on in silico "
        "prediction and should be treated as hypothesis-generating rather than causal claims. "
        "Full quantitative β estimates await download of GEO GSE246756 (~50GB; Replogle 2022). "
        "GWAS S-LDSC γ estimates are provisional; Mendelian randomization will be run after "
        "summary statistic download."
    )


def run(
    phenotype_result:    dict,
    genetics_result:     dict,
    somatic_result:      dict,
    beta_matrix_result:  dict,
    regulatory_result:   dict,
    causal_result:       dict,
    kg_result:           dict,
    prioritization_result: dict,
    chemistry_result:    dict,
    trials_result:       dict,
) -> dict:
    """
    Synthesize all tier outputs into a publication-grade GraphOutput.

    Returns:
        GraphOutput-compatible dict
    """
    disease_name = phenotype_result.get("disease_name", "")
    efo_id       = phenotype_result.get("efo_id", "")
    targets      = prioritization_result.get("targets", [])
    warnings_all: list[str] = []

    # Collect warnings from all tiers
    for res in [
        genetics_result, somatic_result, beta_matrix_result,
        regulatory_result, causal_result, kg_result,
        prioritization_result, chemistry_result, trials_result,
    ]:
        warnings_all.extend(res.get("warnings", []))

    # -------------------------------------------------------------------------
    # Executive summary
    # -------------------------------------------------------------------------
    top3 = targets[:3]
    top3_genes = [t["target_gene"] for t in top3]
    anchor_recovery = causal_result.get("anchor_recovery", {})
    recovery_rate = anchor_recovery.get("recovery_rate", 0.0)
    n_written = causal_result.get("n_edges_written", 0)
    n_virtual_beta = beta_matrix_result.get("n_virtual", 0)

    executive_summary = (
        f"Causal graph analysis of {disease_name} (EFO: {efo_id}) identified "
        f"{n_written} significant causal edges spanning {len(targets)} therapeutic targets. "
        f"Top-ranked targets are {', '.join(top3_genes) if top3_genes else 'none identified'}. "
        f"Anchor edge recovery: {recovery_rate:.0%} of 12 reference edges. "
        f"{n_virtual_beta} gene-program β estimates remain provisional_virtual pending "
        "Perturb-seq data download (GEO GSE246756)."
    )

    # -------------------------------------------------------------------------
    # Target table
    # -------------------------------------------------------------------------
    target_table = _format_target_table(targets[:10])

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
        causal_result, n_virtual_beta, anchor_recovery
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
            "target_gene":   t["target_gene"],
            "rank":          t["rank"],
            "target_score":  t["target_score"],
            "ota_gamma":     t["ota_gamma"],
            "evidence_tier": t["evidence_tier"],
            "ot_score":      t["ot_score"],
            "max_phase":     t["max_phase"],
            "known_drugs":   t["known_drugs"],
            "pli":           t.get("pli"),
            "flags":         t.get("flags", []),
            "top_programs":  t.get("top_programs", []),
            "key_evidence":  t.get("key_evidence", []),
        }
        for t in targets
    ]

    return {
        "disease_name":          disease_name,
        "efo_id":                efo_id,
        "target_list":           target_list,
        "anchor_edge_recovery":  recovery_rate,
        "n_tier1_edges":         evidence_quality["n_tier1_edges"],
        "n_tier2_edges":         evidence_quality["n_tier2_edges"],
        "n_tier3_edges":         evidence_quality["n_tier3_edges"],
        "n_virtual_edges":       evidence_quality["n_virtual_edges"],
        "executive_summary":     executive_summary,
        "target_table":          target_table,
        "top_target_narratives": top_target_narratives,
        "evidence_quality":      evidence_quality,
        "limitations":           limitations,
        "pipeline_version":      PIPELINE_VERSION,
        "generated_at":          datetime.now(tz=timezone.utc).isoformat(),
        "warnings":              warnings_all,
    }
