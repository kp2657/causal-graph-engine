"""
pipelines/evidence_landscape.py — Build per-gene evidence profiles from pipeline outputs.

Rather than a single ranked score, each gene gets a structured evidence profile
showing every line of evidence side-by-side:

  genetic_evidence      — OTA γ, SCONE CI, evidence tier
  perturbseq_evidence   — upstream regulator status, cNMF programs, β values
  functional_evidence   — state-space scores, expression flags
  translational_evidence — drugs, max_phase, tractability, OT score
  literature_evidence   — n_papers, citations, recency
  counterfactual_evidence — 50% inhibition CI, knockout prediction
  adversarial_assessment  — red team verdict, counterargument, rank stability

evidence_class groups genes by the strongest line of evidence:
  convergent          Tier1/2 genetic + Perturb-seq regulator
  genetic_anchor      Tier1/2 genetic evidence, no Perturb-seq
  perturb_seq_regulator Perturb-seq only (no strong genetic instrument)
  gwas_provisional    Tier3 genetic evidence
  state_nominated     State-space nomination only (no genetic instrument)
"""
from __future__ import annotations

from typing import Any


_PERTURB_REGULATORS_KEY = "upstream_regulator_evidence"


def _evidence_class(target: dict, perturb_genes: set[str]) -> str:
    """Classify a gene by its strongest evidence class."""
    tier = target.get("evidence_tier", "")
    gene = target.get("target_gene", "")
    gamma = target.get("ota_gamma", 0.0) or 0.0
    is_perturb = gene in perturb_genes

    if tier in ("Tier1_Interventional", "Tier2_Convergent") and gamma != 0.0:
        return "convergent" if is_perturb else "genetic_anchor"
    if is_perturb:
        return "perturb_seq_regulator"
    if tier == "Tier3_Provisional":
        return "gwas_provisional"
    return "state_nominated"


def _genetic_track(target: dict) -> dict:
    ci_lower = target.get("ota_gamma_ci_lower")
    ci_upper = target.get("ota_gamma_ci_upper")
    return {
        "ota_gamma":        target.get("ota_gamma"),
        "ota_gamma_raw":    target.get("ota_gamma_raw"),
        "ota_gamma_sigma":  target.get("ota_gamma_sigma"),
        "ota_gamma_ci":     [ci_lower, ci_upper] if ci_lower is not None else None,
        "scone_confidence": target.get("scone_confidence"),
        "scone_flags":      target.get("scone_flags", []),
        "evidence_tier":    target.get("evidence_tier"),
    }


def _perturbseq_track(gene: str, perturb_genes: set[str], target: dict) -> dict:
    programs = target.get("top_programs", [])
    return {
        "is_upstream_regulator": gene in perturb_genes,
        "top_programs":          programs,
        "n_programs":            len(programs),
    }


def _functional_track(target: dict, tr_data: dict | None = None) -> dict:
    flags = target.get("flags", [])
    # Phase R τ: prefer flattened keys on target (writer surfaces them);
    # fall back to tr_data dict if available
    tau_disease    = target.get("tau_disease_specificity")
    disease_log2fc = target.get("tau_disease_log2fc")
    tau_class      = target.get("tau_specificity_class")
    if tr_data and tau_disease is None:
        tau_disease    = tr_data.get("tau_disease_specificity")
        disease_log2fc = tr_data.get("disease_log2fc")
        tau_class      = tr_data.get("tau_specificity_class")
    # Treat the default 0.5 neutral sentinel as "not computed" for display
    if tau_disease == 0.5:
        tau_disease = None
    return {
        "flags":               flags,
        "bimodal_expression":  "bimodal_expression" in flags,
        "chip_mechanism":      "chip_mechanism" in flags,
        "escape_risk":         "escape_risk" in flags,
        # Phase R: disease-state τ specificity
        "tau_disease":         tau_disease,
        "disease_log2fc":      disease_log2fc,
        "tau_specificity_class": tau_class,
    }


def _translational_track(target: dict, chem_info: dict, trial_info: dict) -> dict:
    tractability = chem_info.get("tractability") or chem_info.get("tractability_class")
    ic50         = chem_info.get("best_ic50_nM")
    approved     = chem_info.get("approved_drugs", [])
    return {
        "max_phase":    target.get("max_phase", 0),
        "known_drugs":  target.get("known_drugs", []),
        "approved_drugs": approved,
        "tractability": tractability,
        "best_ic50_nM": ic50,
        "ot_score":     target.get("ot_score"),
        "pli":          target.get("pli"),
    }


def _literature_track(gene: str, lit_evidence: dict) -> dict:
    rec = lit_evidence.get(gene) or lit_evidence.get(gene.upper()) or {}
    citations = rec.get("key_citations", [])
    # Compact citation list: pmid + title
    compact_cites = [
        {"pmid": c.get("pmid"), "title": c.get("title", "")[:100], "year": c.get("year")}
        for c in citations[:3]
    ]
    return {
        "n_papers_found":      rec.get("n_papers_found", 0),
        "n_supporting":        rec.get("n_supporting", 0),
        "n_contradicting":     rec.get("n_contradicting", 0),
        "literature_confidence": rec.get("literature_confidence"),
        "recency_score":       rec.get("recency_score"),
        "key_citations":       compact_cites,
        "search_query":        rec.get("search_query"),
    }


def _counterfactual_track(red_team_assessment: dict | None) -> dict | None:
    if not red_team_assessment:
        return None
    cf = red_team_assessment.get("counterfactual")
    if not cf:
        return None

    def _compact_sim(sim: dict | None) -> dict | None:
        if not sim:
            return None
        ci_lower = sim.get("ci_lower")
        ci_upper = sim.get("ci_upper")
        baseline = sim.get("baseline_gamma")
        pct = sim.get("percent_change")
        note = sim.get("uncertainty_note", "")
        # Parse CI percent range from note if available
        return {
            "baseline_gamma":  baseline,
            "perturbed_gamma": sim.get("perturbed_gamma"),
            "percent_change":  pct,
            "ci_lower":        ci_lower,
            "ci_upper":        ci_upper,
            "uncertainty_note": note,
            "dominant_program": sim.get("dominant_program"),
            "interpretation":   sim.get("interpretation"),
        }

    return {
        "primary_trait":    cf.get("primary_trait"),
        "inhibition_50pct": _compact_sim(cf.get("inhibition_50pct")),
        "knockout":         _compact_sim(cf.get("knockout")),
    }


def _adversarial_track(red_team_assessment: dict | None) -> dict | None:
    if not red_team_assessment:
        return None
    return {
        "verdict":              red_team_assessment.get("red_team_verdict"),
        "confidence_level":     red_team_assessment.get("confidence_level"),
        "rank_stability":       red_team_assessment.get("rank_stability"),
        "rank_stability_rationale": red_team_assessment.get("rank_stability_rationale"),
        "counterargument":      red_team_assessment.get("counterargument"),
        "evidence_vulnerability": red_team_assessment.get("evidence_vulnerability"),
        "literature_flag":      red_team_assessment.get("literature_flag"),
    }


def build_evidence_landscape(pipeline_outputs: dict) -> list[dict]:
    """
    Assemble per-gene evidence profiles from all pipeline outputs.

    Args:
        pipeline_outputs: dict containing graph_output, literature_result,
                          red_team_result, chemistry_result, trials_result,
                          and upstream_regulator_evidence.

    Returns:
        List of evidence profile dicts, one per gene, ordered by original rank.
    """
    graph_output     = pipeline_outputs.get("graph_output", {})
    target_list      = graph_output.get("target_list", [])

    # Build gene→TR data index from prioritization_result (writer strips TR from target_list)
    _pr_targets = (
        (pipeline_outputs.get("prioritization_result") or {}).get("targets", []) or []
    )
    tr_by_gene: dict[str, dict] = {
        rec.get("target_gene", ""): rec.get("therapeutic_redirection_result") or {}
        for rec in _pr_targets
        if rec.get("target_gene")
    }

    lit_result       = pipeline_outputs.get("literature_result", {}) or {}
    lit_evidence     = lit_result.get("literature_evidence", {}) or {}

    red_team_result  = pipeline_outputs.get("red_team_result", {}) or {}
    rt_assessments   = red_team_result.get("red_team_assessments", []) or []
    rt_by_gene: dict[str, dict] = {
        a["target_gene"]: a for a in rt_assessments if "target_gene" in a
    }

    chem_result      = pipeline_outputs.get("chemistry_result", {}) or {}
    target_chemistry = chem_result.get("target_chemistry", {}) or {}

    trials_result    = pipeline_outputs.get("trials_result", {}) or {}
    trial_summary    = trials_result.get("trial_summary", {}) or {}

    # Perturb-seq upstream regulators (stored under either key)
    regulator_ev = (
        pipeline_outputs.get(_PERTURB_REGULATORS_KEY)
        or pipeline_outputs.get("regulator_nomination_evidence")
        or {}
    )
    perturb_genes: set[str] = set()
    if isinstance(regulator_ev, dict):
        # Two shapes: {gene: {...}} or {"regulators": [{gene: ..., ...}]}
        if "regulators" in regulator_ev:
            # List-of-records shape
            for rec in regulator_ev["regulators"]:
                g = rec.get("gene") or rec.get("target_gene")
                if g:
                    perturb_genes.add(g)
        else:
            # gene-keyed dict shape
            perturb_genes = set(regulator_ev.keys())
    elif isinstance(regulator_ev, list):
        # Flat list of gene names or records
        for item in regulator_ev:
            if isinstance(item, str):
                perturb_genes.add(item)
            elif isinstance(item, dict):
                g = item.get("gene") or item.get("target_gene")
                if g:
                    perturb_genes.add(g)

    profiles: list[dict] = []
    for target in target_list:
        gene = target.get("target_gene", "")
        chem_info  = target_chemistry.get(gene, {})
        trial_info = trial_summary.get(gene, {})
        rt_assess  = rt_by_gene.get(gene)
        # TR data: prefer prioritization_result (full fields); target_list only has stripped fields
        tr_data    = tr_by_gene.get(gene) or target.get("therapeutic_redirection_result") or {}

        profile = {
            "gene":          gene,
            "evidence_class": _evidence_class(target, perturb_genes),

            # Internal rank retained for ordering but de-emphasized
            "_rank":         target.get("rank"),
            "_target_score": target.get("target_score"),

            "genetic_evidence":       _genetic_track(target),
            "perturbseq_evidence":    _perturbseq_track(gene, perturb_genes, target),
            "functional_evidence":    _functional_track(target, tr_data),
            "translational_evidence": _translational_track(target, chem_info, trial_info),
            "literature_evidence":    _literature_track(gene, lit_evidence),
            "counterfactual_evidence": _counterfactual_track(rt_assess),
            "adversarial_assessment":  _adversarial_track(rt_assess),
        }
        profiles.append(profile)

    return profiles


def summarize_landscape(profiles: list[dict]) -> dict:
    """
    High-level counts across evidence classes.

    Returns:
        {
          "n_genes": int,
          "by_class": {class: count},
          "n_with_genetic_instrument": int,
          "n_with_literature_support": int,
          "n_with_perturb_seq": int,
          "n_with_counterfactual": int,
          "n_proceed": int,   # red team verdict
          "n_caution": int,
          "n_deprioritize": int,
        }
    """
    by_class: dict[str, int] = {}
    n_genetic   = 0
    n_lit       = 0
    n_perturb   = 0
    n_cf        = 0
    n_proceed   = 0
    n_caution   = 0
    n_depriori  = 0

    for p in profiles:
        cls = p.get("evidence_class", "unknown")
        by_class[cls] = by_class.get(cls, 0) + 1

        gamma = (p.get("genetic_evidence") or {}).get("ota_gamma") or 0.0
        if gamma and gamma != 0.0:
            n_genetic += 1

        lit_conf = (p.get("literature_evidence") or {}).get("literature_confidence")
        if lit_conf == "SUPPORTED":
            n_lit += 1

        if (p.get("perturbseq_evidence") or {}).get("is_upstream_regulator"):
            n_perturb += 1

        if p.get("counterfactual_evidence"):
            n_cf += 1

        verdict = (p.get("adversarial_assessment") or {}).get("verdict")
        if verdict == "PROCEED":
            n_proceed += 1
        elif verdict == "CAUTION":
            n_caution += 1
        elif verdict == "DEPRIORITIZE":
            n_depriori += 1

    return {
        "n_genes":                   len(profiles),
        "by_class":                  by_class,
        "n_with_genetic_instrument": n_genetic,
        "n_with_literature_support": n_lit,
        "n_with_perturb_seq":        n_perturb,
        "n_with_counterfactual":     n_cf,
        "n_proceed":                 n_proceed,
        "n_caution":                 n_caution,
        "n_deprioritize":            n_depriori,
    }
