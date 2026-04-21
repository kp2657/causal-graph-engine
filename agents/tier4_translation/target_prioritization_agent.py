"""
target_prioritization_agent.py — Tier 4 agent: multi-dimensional target ranking.

Output dimensions (reported as separate columns, not folded into a formula):
  causal_gamma          — OTA γ (primary sort key; genetic grounding from Tier 3)
  genetic_evidence_score — L2G / OT genetic association proxy in [0, 1] when available
  mechanistic_score     — |TR| + 0.3×state_influence + 0.2×transition_avg [0, 1]
  max_phase / known_drugs / pli — translatability (reported verbatim)

Partition label (informational; no effect on rank):
  "genetically_grounded": ot_genetic_score ≥ 0.05  OR  max_phase > 0
  "high_reward_mechanistic": ot_genetic_score < 0.05 AND TR ≥ 0.4 AND stability ≥ 0.8
  "mechanistic_only":     ot_genetic_score < 0.05  AND max_phase == 0

Flags replace discounts — no dimension is modified by evidence classification:
  "no_genetic_grounding"  — mechanistic_only gene; review genetic_evidence_score
  "convergent_controller" — high TR + high Stability; candidate for high_reward_mechanistic
  "marker_gene"           — marker_score ≥ 0.3; likely disease-state consequence

Ranking: purely by causal_gamma (ota_gamma) descending. Consumers filter by flags.

target_score = causal_gamma (= ota_gamma) — kept for backward compat with writer/orchestrator.
"""
from __future__ import annotations

import os

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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
    "state_nominated":      0.2,   # state-space nominated; no genetic instrument
}

CHIP_GENES = {"TET2", "DNMT3A", "ASXL1", "JAK2", "TET2_chip", "DNMT3A_chip"}


# Programs known to be pleiotropic / not disease-specific
_GENERIC_PROGRAMS = frozenset({
    "bile_acid_metabolism",
    "coagulation_program",
    "allograft_rejection",
    "estrogen_signaling",
    "protein_secretion",
    "inflammatory_NF-kB",
    "cell_cycle",
    "generic_stress",
})

# Disease-specific program vocabularies — used to score how disease-relevant a
# gene's β distribution is (vs. generic pleiotropic programs).
_DISEASE_PROGRAMS: dict[str, frozenset] = {
    "AMD": frozenset({
        "complement_program",
        "__protein_channel__",
        "visual_transduction",
        "photoreceptor_maintenance",
        "angiogenesis",
        "lipid_efflux",
        "oxidative_stress",
    }),
    "CAD": frozenset({
        "HALLMARK_CHOLESTEROL_HOMEOSTASIS",
        "HALLMARK_FATTY_ACID_METABOLISM",
        "HALLMARK_COAGULATION",
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_ANGIOGENESIS",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
        "lipid_metabolism",
        "foam_cell_program",
        "plaque_inflammation",
    }),
    "IBD": frozenset({
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "gut_epithelial_barrier",
        "innate_immune_program",
    }),
}
# Backward-compat alias
_AMD_PROGRAMS = _DISEASE_PROGRAMS["AMD"]


def _classify_program_drivers(
    top_programs: dict | list,
    ota_gamma: float,
    disease_key: str = "AMD",
) -> dict:
    """
    Summarise which NMF programs drive a gene's OTA gamma and whether those
    programs are disease-specific or generic/pleiotropic.

    Args:
        top_programs: dict {program_name: contribution} or list (ignored → empty)
        ota_gamma:    total OTA gamma for the gene
        disease_key:  Short disease key (AMD, CAD, IBD) — selects which program
                      vocabulary is used for "disease-specific" classification.

    Returns:
        {
            top_program, top_program_pct, n_programs_contributing,
            spread, disease_specific_pct, generic_pct, program_flag,
            top_programs_ranked  (list of {program, pct})
        }
    """
    _empty = {
        "top_program":              None,
        "top_program_pct":          0,
        "n_programs_contributing":  0,
        "spread":                   "unknown",
        "disease_specific_pct":     0,
        "amd_specific_pct":         0,  # backward-compat alias
        "generic_pct":              0,
        "program_flag":             "no_program_data",
        "top_programs_ranked":      [],
    }
    if not isinstance(top_programs, dict) or not top_programs or not ota_gamma:
        return _empty

    _disease_progs = _DISEASE_PROGRAMS.get(disease_key.upper(), _AMD_PROGRAMS)

    total = sum(abs(v) for v in top_programs.values())
    if total <= 1e-9:
        return _empty

    sorted_progs = sorted(top_programs.items(), key=lambda x: -abs(x[1]))
    top_prog, top_val = sorted_progs[0]
    top_pct = round(abs(top_val) / total * 100, 1)

    n_contributing = sum(1 for _, v in sorted_progs if abs(v) / total >= 0.10)

    dis_pct = round(
        sum(abs(v) for p, v in top_programs.items() if p in _disease_progs) / total * 100, 1
    )
    gen_pct = round(
        sum(abs(v) for p, v in top_programs.items() if p in _GENERIC_PROGRAMS) / total * 100, 1
    )

    spread = (
        "concentrated" if top_pct >= 70
        else "moderate"  if top_pct >= 40
        else "diffuse"
    )

    if dis_pct >= 70:
        flag = f"disease_specific:{top_prog} ({dis_pct:.0f}%)"
    elif gen_pct >= 50:
        flag = f"generic_programs:{gen_pct:.0f}%_bile_acid-coag-allograft"
    elif top_pct >= 60:
        flag = f"concentrated:{top_prog} ({top_pct:.0f}%)"
    else:
        flag = f"diffuse:{n_contributing}_programs_each_<40pct"

    return {
        "top_program":              top_prog,
        "top_program_pct":          top_pct,
        "n_programs_contributing":  n_contributing,
        "spread":                   spread,
        "disease_specific_pct":     dis_pct,
        "amd_specific_pct":         dis_pct,  # backward-compat alias
        "generic_pct":              gen_pct,
        "program_flag":             flag,
        "top_programs_ranked": [
            {"program": p, "pct": round(abs(v) / total * 100, 1)}
            for p, v in sorted_progs[:5]
        ],
    }


def _phase_bonus(max_phase: int) -> float:
    return TRIAL_BONUS.get(max_phase, 0.0)


def run(
    causal_discovery_result: dict,
    kg_completion_result: dict,
    disease_query: dict,
) -> dict:
    """
    Rank therapeutic targets using multi-dimensional scoring.

    Args:
        causal_discovery_result: Output of causal_discovery_agent.run
        kg_completion_result:    Output of kg_completion_agent.run
        disease_query:           DiseaseQuery dict

    Returns:
        list of TargetRecord-compatible dicts, sorted by partition then causal_gamma
    """
    from mcp_servers.open_targets_server import get_open_targets_disease_targets
    from mcp_servers.gwas_genetics_server import query_gnomad_lof_constraint
    from mcp_servers.single_cell_server import (
        get_gene_tau_specificity,
        get_gene_bimodality_scores,
    )
    _tier4_ctx = disease_query.get("_tier4_context") or {}
    minimal_tier4 = bool(_tier4_ctx.get("minimal")) or os.getenv("MINIMAL_TIER4", "").strip().lower() in {"1", "true", "yes", "on"}

    efo_id       = disease_query.get("efo_id", "")
    disease_name = disease_query.get("disease_name", "")
    # Derive short disease key for disease-specific program classification
    from graph.schema import _DISEASE_SHORT_NAMES_FOR_ANCHORS as _DSN
    _short_key = _DSN.get(disease_name.lower(), "AMD")
    top_genes_all = causal_discovery_result.get("top_genes", [])
    drug_target_summary = kg_completion_result.get("drug_target_summary", [])
    brg_candidates = kg_completion_result.get("brg_novel_candidates", [])
    warnings: list[str] = []

    # Index BRG scores for quick lookup
    brg_score_map: dict[str, float] = {
        c["gene"]: c["brg_score"] for c in brg_candidates if c.get("gene")
    }

    if not top_genes_all:
        return {"targets": [], "warnings": ["No genes in causal_discovery_result"]}

    # Separate instrumented (genetic-causal) targets from state-space nominees.
    # Policy: state-space targets are reported separately (not mixed into top ranks).
    top_genes: list[dict] = []
    state_space_genes: list[dict] = []
    for rec in top_genes_all:
        tier = rec.get("tier")
        ev_type = rec.get("evidence_type")
        if tier == "state_nominated" or ev_type == "state_space":
            state_space_genes.append(rec)
        else:
            top_genes.append(rec)

    # -------------------------------------------------------------------------
    # Load Open Targets scores for all top genes at once (or L2G-only if assoc API off)
    # -------------------------------------------------------------------------
    ot_scores: dict[str, float] = {}
    ot_genetic: dict[str, float] = {}
    ot_max_phase: dict[str, int] = {}
    ot_known_drugs: dict[str, list[str]] = {}
    _assoc_off = os.getenv("OPEN_TARGETS_ASSOC_DISABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    _ot_cache = disease_query.get("_ot_disease_targets_cache") or {}
    try:
        if _assoc_off and efo_id:
            from mcp_servers.gwas_genetics_server import aggregate_l2g_scores_for_program_genes
            genes = [r["gene"] for r in top_genes if r.get("gene")]
            agg = aggregate_l2g_scores_for_program_genes(efo_id, genes)
            gmap = agg.get("gene_scores") or {}
            for g in genes:
                sc = float(gmap.get(g.upper(), 0.0))
                ot_genetic[g] = sc
                ot_scores[g] = sc
        else:
            # Prefer orchestrator-cached OT disease→targets payload to avoid a second large call.
            if isinstance(_ot_cache, dict) and (_ot_cache.get("efo_id") == efo_id) and _ot_cache.get("targets"):
                ot_targets = _ot_cache.get("targets") or []
            else:
                ot_result = get_open_targets_disease_targets(efo_id, max_targets=500)
                ot_targets = ot_result.get("targets", []) or []

            for t in ot_targets:
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
    pli_map: dict[str, float] = dict(_tier4_ctx.get("pli_map") or {})
    if not pli_map:
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
    if not minimal_tier4:
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
    # Multi-dimensional scoring — transparent per-dimension, no composite formula
    # -------------------------------------------------------------------------
    max_gamma = max(
        (abs(r.get("ota_gamma") or 0.0) for r in top_genes),
        default=1.0,
    ) or 1.0

    # -------------------------------------------------------------------------
    # Score each gene
    # -------------------------------------------------------------------------
    target_records: list[dict] = []
    n_mech_no_geno = 0
    mech_example_genes: list[str] = []

    for rec in top_genes:
        gene         = rec["gene"]
        ota_gamma    = abs(rec.get("ota_gamma") or 0.0)
        ota_sigma    = rec.get("ota_gamma_sigma", 0.0) or 0.0
        tier         = rec.get("tier", "provisional_virtual")
        top_programs = rec.get("programs", [])

        # ---- Dimension 1: Causal magnitude (ota_gamma ± CI) -----------------
        gamma_ci_lower = rec.get("ota_gamma_ci_lower")
        gamma_ci_upper = rec.get("ota_gamma_ci_upper")

        # ---- Dimension 2: Genetic evidence (OT GWAS/coloc/burden) -----------
        ot_gen_score = ot_genetic.get(gene, 0.0)
        ot_score     = ot_scores.get(gene, 0.0)

        # ---- Dimension 3: Mechanistic signal — state-space + TR --------------
        tr_data   = rec.get("therapeutic_redirection_result")
        traj_data = rec.get("trajectory")
        mechanistic_score    = 0.0
        mechanistic_category = "unknown"
        entry_s = persist_s = recovery_s = boundary_s = 0.0
        if tr_data is not None:
            tr_val     = abs(float(tr_data.get("therapeutic_redirection", 0.0)))
            si_val     = float(tr_data.get("state_influence_score", 0.0))
            entry_s    = float(tr_data.get("entry_score", 0.0))
            persist_s  = float(tr_data.get("persistence_score", 0.0))
            recovery_s = float(tr_data.get("recovery_score", 0.0))
            boundary_s = float(tr_data.get("boundary_score", 0.0))
            transition_avg = 0.25 * (entry_s + persist_s + recovery_s + boundary_s)
            mechanistic_score    = min(tr_val + 0.3 * si_val + 0.2 * transition_avg, 1.0)
            mechanistic_category = str(tr_data.get("mechanistic_category", "unknown"))
        elif traj_data:
            legacy_mech = (
                traj_data.get("expected_pathology_reduction", 0.0)
                - 0.5 * traj_data.get("escape_risk_score", 0.0)
                - 0.5 * traj_data.get("negative_memory_penalty", 0.0)
            )
            mechanistic_score = max(0.0, min(legacy_mech, 1.0))

        # ---- Dimension 4: Translatability ------------------------------------
        max_phase = max(
            ot_max_phase.get(gene, 0),
            phase_from_kg.get(gene, 0),
        )

        # Safety
        pli = pli_map.get(gene)
        safety_pen   = 0.0
        safety_flags: list[str] = []
        if pli is not None and pli > PLI_ESSENTIAL:
            safety_pen += PLI_PENALTY
            safety_flags.append(f"pLI={pli:.2f} > 0.9 — essential gene; on-target toxicity")

        # Cell-type specificity (reporting only; not a sort key)
        tau    = tau_map.get(gene)
        bc     = bc_map.get(gene)
        bc_norm = min(bc / 1.0, 1.0) if bc is not None else 0.5
        tau_val = tau if tau is not None else 0.5
        # Convenience composite for downstream consumers; not used in ranking
        specificity_score = round(0.6 * tau_val + 0.4 * bc_norm, 4)

        # ---- Partitioning: grounded vs mechanistic-only ----------------------
        # Genetically grounded: pipeline genetic score (L2G aggregate or OT
        # genetic_association when assoc API enabled) ≥ 0.05, OR drug development
        # history (max_phase > 0).
        #
        # Phase Z7: High-Risk/High-Reward (HRHR) path for "Convergent Controllers"
        # Genes with high Therapeutic Redirection and high Stability score,
        # even without GWAS support.
        is_grounded = ot_gen_score >= 0.05 or max_phase > 0
        
        tr_val_raw = abs(float(tr_data.get("therapeutic_redirection", 0.0))) if tr_data else 0.0
        # Use stability_score from tr_data if present (Phase Z7)
        stability_val = float(tr_data.get("stability_score", 1.0)) if tr_data else 1.0
        is_hrhr = not is_grounded and tr_val_raw >= 0.4 and stability_val >= 0.8

        if is_grounded:
            partition = "genetically_grounded"
        elif is_hrhr:
            partition = "high_reward_mechanistic"
        else:
            partition = "mechanistic_only"

        # ---- target_score = causal_gamma (backward compat) ------------------
        # Primary sort key within each partition.  Kept as "target_score" so
        # downstream writer / orchestrator fields are unchanged.
        target_score = ota_gamma

        # Phase H: compute marker score for flagging only — does NOT modify any dimension.
        # A gene classified as a disease marker (consequence of disease state, not cause)
        # is noted via the "marker_gene" flag so consumers can filter; scores are unchanged.
        marker_score = 0.0
        if tr_data is not None:
            ctrl_data = rec.get("controller_annotation")
            if ctrl_data is not None:
                try:
                    from models.evidence import ControllerAnnotation, TransitionGeneProfile
                    from pipelines.state_space.controller_classifier import compute_marker_discount
                    ctrl_ann = ControllerAnnotation.model_validate(ctrl_data)
                    profile_for_discount = TransitionGeneProfile(
                        gene=gene, disease=disease_name,
                        entry_score=entry_s,
                        persistence_score=persist_s,
                        recovery_score=recovery_s,
                        boundary_score=boundary_s,
                    )
                    marker_score = compute_marker_discount(ctrl_ann, profile_for_discount)
                except Exception:
                    pass

        # ---- Flags -----------------------------------------------------------
        flags: list[str] = []
        if max_phase >= 2 and ot_score > 0.5:
            flags.append("repurposing_candidate")
        if max_phase == 0 and ota_gamma > 0.5 * max_gamma:
            flags.append("first_in_class")
        if gene in CHIP_GENES or gene.replace("_chip", "") in CHIP_GENES:
            flags.append("chip_mechanism")
        if tier == "provisional_virtual":
            flags.append("provisional_virtual")
        if tau is not None and tau >= 0.70:
            flags.append("highly_specific")
        if bc is not None and bc > 0.555:
            flags.append("bimodal_expression")
        if is_hrhr:
            flags.append("convergent_controller")
            warnings.append(f"{gene}: high TR ({tr_val_raw:.2f}) + high Stability ({stability_val:.2f}) — high_reward_mechanistic")
        elif not is_grounded:
            flags.append("no_genetic_grounding")
            n_mech_no_geno += 1
            if len(mech_example_genes) < 10:
                mech_example_genes.append(gene)
        if marker_score >= 0.3:
            flags.append("marker_gene")
            warnings.append(
                f"{gene}: marker_score={marker_score:.2f} — likely disease-state consequence; "
                "review mechanistic_score before prioritising"
            )
        brg_score = brg_score_map.get(gene)
        if brg_score is not None:
            flags.append("brg_novel_candidate")
        if tr_data is not None:
            if tr_data.get("therapeutic_redirection", 0) > 0.1:
                flags.append("strong_trajectory_signal")
            if tr_data.get("escape_risk", 0) > 0.5:
                flags.append("high_escape_risk")
            if tr_data.get("context_confidence_warning"):
                flags.append("context_confidence_warning")
        elif traj_data:
            if traj_data.get("escape_risk_score", 0) > 0.5:
                flags.append("high_escape_risk")
            if traj_data.get("negative_memory_penalty", 0) > 0.6:
                flags.append("negative_memory_risk")
            if traj_data.get("expected_pathology_reduction", 0) > 0.4:
                flags.append("strong_trajectory_signal")
        disagreement_records = rec.get("evidence_disagreement", [])
        has_block = any(r.get("severity") == "block" for r in disagreement_records)
        has_flag  = any(r.get("severity") == "flag"  for r in disagreement_records)
        if has_block:
            flags.append("evidence_disagreement_block")
        elif has_flag:
            flags.append("evidence_disagreement_flag")

        # Evidence keys from drug_target_summary
        key_evidence: list[str] = []
        if drug_for_gene.get(gene):
            key_evidence.append(f"Drug: {', '.join(drug_for_gene[gene][:3])}")

        target_records.append({
            "target_gene":        gene,
            "rank":               0,  # filled after sort

            # Dimension 1 — causal magnitude
            "causal_gamma":       round(ota_gamma, 4),
            "target_score":       round(target_score, 4),   # = causal_gamma; backward compat
            "ota_gamma":          rec.get("ota_gamma", 0.0),
            "ota_gamma_raw":      rec.get("ota_gamma_raw", rec.get("ota_gamma", 0.0)),
            "ota_gamma_sigma":    round(ota_sigma, 4),
            "ota_gamma_ci_lower": round(gamma_ci_lower, 4) if gamma_ci_lower is not None else None,
            "ota_gamma_ci_upper": round(gamma_ci_upper, 4) if gamma_ci_upper is not None else None,
            "evidence_tier":      tier,

            # Dimension 2 — genetic evidence
            "genetic_evidence_score": round(ot_gen_score, 4),
            "ot_genetic_score":   ot_gen_score,   # backward compat alias
            "ot_l2g_score":       round(ot_gen_score, 4),  # same as genetic_evidence_score (L2G aggregate or OT genetic)
            "ot_score":           ot_score,
            "partition":          partition,       # genetically_grounded | mechanistic_only

            # Dimension 3 — mechanistic signal
            "mechanistic_score":     round(mechanistic_score, 4),
            "mechanistic_category":  mechanistic_category,
            "therapeutic_redirection": round(tr_val_raw, 4),
            "stability_score":       round(stability_val, 4),
            "entry_score":           round(entry_s, 4),
            "persistence_score":     round(persist_s, 4),
            "recovery_score":        round(recovery_s, 4),
            "boundary_score":        round(boundary_s, 4),
            "marker_score":          round(marker_score, 4),   # informational; not a discount

            # Dimension 4 — translatability
            "max_phase":          max_phase,
            "known_drugs":        drug_for_gene.get(gene, []),
            "pli":                pli,
            "safety_flags":       safety_flags,

            # Specificity (reporting; not a sort key)
            "tau_specificity":    None if minimal_tier4 else tau,
            "bimodality_coeff":   None if minimal_tier4 else bc,
            "specificity_score":  None if minimal_tier4 else specificity_score,   # reporting only

            # Other
            "brg_score":          brg_score,
            "scone_confidence":   rec.get("scone_confidence"),
            "scone_flags":        rec.get("scone_flags", []),
            "flags":              flags,
            "top_programs":       top_programs,
            "_max_prog_contrib":  (
                max((abs(v) for v in top_programs.values() if isinstance(v, (int, float))), default=0.0)
                if isinstance(top_programs, dict) else 0.0
            ),  # internal; stripped before output (see below)
            "program_drivers":    None if minimal_tier4 else _classify_program_drivers(top_programs, abs(rec.get("ota_gamma", 0.0)), disease_key=_short_key),
            "key_evidence":       key_evidence,
            "therapeutic_redirection_result": tr_data,
            "evidence_disagreement":          disagreement_records,
            "controller_annotation":          rec.get("controller_annotation"),
            "trajectory":                     traj_data,   # kept for backward compat
            # Phase R pass-through
            "tau_disease_specificity": rec.get("tau_disease_specificity"),
            "disease_log2fc":          rec.get("disease_log2fc"),
            "tau_specificity_class":   rec.get("tau_specificity_class"),
            # beta_amd_concentration: fraction of β-mass in AMD-specific programs [0,1].
            # Diagnostic for pleiotropic genes: low value = β spread across generic programs
            # (e.g. JAZF1 with high β everywhere gets low concentration).
            # ota_gamma is NOT modified — this is reporting only; CSO interprets it.
            "beta_amd_concentration": rec.get("beta_amd_concentration"),
            # tier_upgrade_log: records any re-scoring events applied after initial tier assignment.
            # Format: [{"from_tier": str, "to_tier": str, "reason": str, "data_source": str}]
            # Empty list means the gene kept its originally assigned tier throughout the run.
            "tier_upgrade_log": rec.get("tier_upgrade_log", []),
        })

    if n_mech_no_geno > 0:
        ex = ", ".join(mech_example_genes)
        more = f" (+{n_mech_no_geno - len(mech_example_genes)} more)" if n_mech_no_geno > len(mech_example_genes) else ""
        warnings.append(
            f"Partition: {n_mech_no_geno} target(s) classified as mechanistic_only "
            f"(pipeline genetic score <0.05 and no drug with trials phase >0). "
            f"Tier1 Perturb-seq / trajectory evidence can still be strong. Examples: {ex}{more}"
        )

    # Sort by causal_gamma, but require co-evidence: genes with no genetic grounding
    # (mechanistic_only partition) have their effective rank score discounted 80%.
    # Phase Z7: HRHR genes get a smaller discount (40% → multiplier 0.6) instead
    # of 80%, as their mechanistic evidence is validated by high TR + Stability.
    def _ranking_key(r: dict) -> float:
        gamma = abs(r.get("ota_gamma", 0.0))
        partition = r.get("partition", "mechanistic_only")
        
        if partition == "genetically_grounded":
            multiplier = 1.0
        elif partition == "high_reward_mechanistic":
            multiplier = 0.6  # 40% discount for high-confidence mechanistic
        else:
            multiplier = 0.2  # 80% discount for ungrounded/low-confidence
            
        return -(gamma * multiplier)

    target_records.sort(key=_ranking_key)

    # Essential/housekeeping gene sink: genes whose ota_gamma is inflated because their
    # Perturb-seq β saturates every program equally (RNA Pol, primase, proteasome, etc.).
    # Signature: |ota_gamma|>2.0 AND the single largest program contribution exceeds 0.8.
    # Real disease genes (LIPC, SORT1, HMGCR) have max program contributions ≤0.48.
    # The gap between 0.48 (real) and 0.93+ (housekeeping) is clean across AMD and CAD runs.
    _essential_flagged: list[str] = []
    for r in target_records:
        max_contrib = r.get("_max_prog_contrib", 0.0)
        if max_contrib == 0.0:
            # Fallback: recompute from top_programs in case field was missing
            progs = r.get("top_programs") or {}
            if isinstance(progs, dict):
                max_contrib = max((abs(v) for v in progs.values() if isinstance(v, (int, float))), default=0.0)
        if (
            abs(r.get("ota_gamma", 0.0)) > 2.0
            and max_contrib > 0.8
        ):
            r["flags"] = list(r.get("flags", [])) + ["inflated_gamma_essential"]
            _essential_flagged.append(r["target_gene"])

    if _essential_flagged:
        warnings.append(
            f"Essential gene sink: {len(_essential_flagged)} gene(s) with |ota_gamma|>2.0 and "
            f"ot_genetic_score<0.3 ranked last (inflated housekeeping β). "
            f"Examples: {', '.join(_essential_flagged[:5])}"
        )
        _normal = [r for r in target_records if "inflated_gamma_essential" not in r.get("flags", [])]
        _sunk   = [r for r in target_records if "inflated_gamma_essential" in r.get("flags", [])]
        target_records = _normal + _sunk

    for i, r in enumerate(target_records, 1):
        r["rank"] = i

    # Phase J: upstream regulator rank benchmark — INFORMATIONAL ONLY
    # GWAS and Perturb-seq are independent evidence tracks. Essential upstream TFs
    # (SPI1, JAK2, STAT1) have high pLI and low GWAS gamma → correctly score low as
    # drug targets. Their causal evidence lives in upstream_regulator_evidence, not here.
    # upstream_recovery_ratio is recorded for tracking but never causes a pipeline warning.
    benchmark_result: dict | None = None
    if not minimal_tier4:
        try:
            from pipelines.upstream_recovery_benchmark import load_benchmark_config, evaluate_ranking
            _benchmarks_dir = Path(__file__).parent.parent.parent / "data" / "benchmarks"
            _disease_key = disease_name.lower().replace(" ", "_").replace("-", "_")
            _cfg_candidates = list(_benchmarks_dir.glob(f"{_disease_key}_upstream_regulators_*.json"))
            _ALIAS_MAP = {
                "ibd": ("inflammatory_bowel_disease", "ibd"),
                "cad": ("coronary_artery_disease", "cad"),
            }
            if not _cfg_candidates:
                for _short, _variants in _ALIAS_MAP.items():
                    if any(v == _disease_key or v in _disease_key for v in _variants):
                        _cfg_candidates = list(_benchmarks_dir.glob(f"{_short}_upstream_regulators_*.json"))
                        break
            if _cfg_candidates:
                import math as _math
                _cfg = load_benchmark_config(sorted(_cfg_candidates)[-1])
                _ranked_genes = [r.get("gene") or r.get("target_gene", "") for r in target_records]
                benchmark_result = evaluate_ranking(_ranked_genes, _cfg)
                benchmark_result = {
                    k: (None if isinstance(v, float) and not _math.isfinite(v) else v)
                    for k, v in benchmark_result.items()
                }
                benchmark_result["note"] = (
                    "Drug-target rank ordering only. Upstream TF causal evidence is in "
                    "upstream_regulator_evidence (independent Perturb-seq track)."
                )
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # State-space targets (reported separately; not mixed into ranks)
    # -------------------------------------------------------------------------
    state_space_targets: list[dict] = []
    for rec in state_space_genes:
        gene = rec.get("gene") or rec.get("target_gene") or ""
        if not gene:
            continue
        tr_data = rec.get("therapeutic_redirection_result")
        state_eff = float(rec.get("state_edge_effect") or 0.0)
        state_conf = float(rec.get("state_edge_confidence") or 0.5)
        # independent score for within-section ordering only
        state_score = state_eff * state_conf

        # Optional enrichment: keep OT fields if available
        ot_gen_score = ot_genetic.get(gene, 0.0)
        ot_score     = ot_scores.get(gene, 0.0)
        max_phase = max(
            ot_max_phase.get(gene, 0),
            phase_from_kg.get(gene, 0),
        )

        state_space_targets.append({
            "target_gene": gene,
            "rank": 0,  # filled after sort
            "evidence_tier": "state_nominated",
            "evidence_type": "state_space",
            "state_edge_effect": round(state_eff, 6),
            "state_edge_confidence": round(state_conf, 6),
            "state_target_score": round(state_score, 6),
            "state_edge_ci_lower": rec.get("state_edge_ci_lower"),
            "state_edge_ci_upper": rec.get("state_edge_ci_upper"),
            "state_edge_cv": rec.get("state_edge_cv"),
            # Reporting-only dimensions
            "genetic_evidence_score": round(float(ot_gen_score or 0.0), 4),
            "ot_score": float(ot_score or 0.0),
            "max_phase": int(max_phase or 0),
            "known_drugs": drug_for_gene.get(gene, []),
            "therapeutic_redirection_result": tr_data,
            "controller_annotation": rec.get("controller_annotation"),
            "evidence_disagreement": rec.get("evidence_disagreement", []),
            "scone_confidence": rec.get("scone_confidence"),
            "scone_flags": rec.get("scone_flags", []),
            "tau_disease_specificity": rec.get("tau_disease_specificity"),
            "disease_log2fc": rec.get("disease_log2fc"),
            "tau_specificity_class": rec.get("tau_specificity_class"),
            "beta_amd_concentration": rec.get("beta_amd_concentration"),
        })

    state_space_targets.sort(key=lambda r: -(r.get("state_target_score") or 0.0))
    for i, r in enumerate(state_space_targets, 1):
        r["rank"] = i

    # Strip internal fields (prefixed _) before returning
    for r in target_records:
        r.pop("_max_prog_contrib", None)

    return {
        "targets":   target_records,
        "state_space_targets": state_space_targets,
        "warnings":  warnings,
        "benchmark": benchmark_result,
    }
