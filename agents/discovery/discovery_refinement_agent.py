"""
agents/discovery/discovery_refinement_agent.py — Second-pass discovery refinement.

Evaluates the initial pipeline evidence landscape and runs targeted follow-up
analyses to surface novel, high-value therapeutic opportunities:

  1. Evidence gap filling  — upgrades genes lacking one track of evidence
     (gwas_provisional → genetic_anchor via eQTL; state_nominated → druggable
     via OT tractability; NOVEL literature genes → deeper search)

  2. Genetic instrument recovery — checks GTEx eQTLs and OT genetics for genes
     that lack GWAS instruments but have strong functional evidence

  3. Druggability sweep — queries ChEMBL + OT for state-nominated genes with
     high disease-state τ (disease-specific expression = biologically credible)

  4. Upstream chokepoint analysis — identifies Perturb-seq regulators that
     control the most disease programs; flags the most druggable

  5. Cross-disease colocalization — checks H4 posteriors linking top targets
     to shared causal variants across related traits

Local mode: pure heuristic classification (no API calls).
SDK mode  : Claude actively queries tools and synthesises findings.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Opportunity classes
# ---------------------------------------------------------------------------

OPPORTUNITY_CLASSES = {
    "convergent_needs_drug":      "Convergent genetic+Perturb-seq, no clinical compound — high-priority for HTS/structure-based design",
    "gwas_provisional_upgradable": "Tier3 GWAS signal — eQTL instrument check could upgrade to Tier2",
    "druggable_state_nominated":   "State-nominated with high disease-τ and known small-molecule tool compounds",
    "novel_unexplored":            "No prior literature in disease context — high upside if mechanism confirmed",
    "upstream_chokepoint":         "Perturb-seq regulator controlling ≥3 disease programs — proxy for multiple downstream targets",
    "cross_disease_overlap":       "Convergent evidence + colocalization signal in a related disease — repurposing opportunity",
    "escaped_ranking":             "High state-space signal but depressed by genetic_grounding=0 — warrants direct genetic instrument search",
}


# ---------------------------------------------------------------------------
# Local (heuristic) mode
# ---------------------------------------------------------------------------

def _local_run(pipeline_outputs: dict) -> dict:
    """
    Heuristic second-pass without API calls.

    Classifies genes in the evidence landscape into opportunity buckets
    based on their existing evidence profile, without querying external APIs.
    """
    disease_name = (
        pipeline_outputs.get("disease_name")
        or pipeline_outputs.get("phenotype_result", {}).get("disease_name", "unknown")
    )
    efo_id = (
        pipeline_outputs.get("efo_id")
        or pipeline_outputs.get("phenotype_result", {}).get("efo_id", "")
    )

    # Pull evidence profiles — accept both formats:
    #   compact (from scoped_input["evidence_profiles"]): flat keys at top level
    #   nested  (from pipeline_outputs["evidence_landscape"]["profiles"]): sub-dicts per track
    profiles: list[dict] = []
    if pipeline_outputs.get("evidence_profiles"):
        # Compact format — already flattened by _run_discovery_refinement
        profiles = pipeline_outputs["evidence_profiles"]
        _compact = True
    else:
        landscape_data = pipeline_outputs.get("evidence_landscape") or {}
        if isinstance(landscape_data, dict):
            profiles = landscape_data.get("profiles", [])
        _compact = False

    # Pull upstream regulators
    upstream_regs = list(pipeline_outputs.get("upstream_regulators") or [])
    if not upstream_regs:
        regulator_ev = pipeline_outputs.get("upstream_regulator_evidence") or {}
        if isinstance(regulator_ev, dict) and "regulators" in regulator_ev:
            upstream_regs = [r.get("gene") or r.get("target_gene", "") for r in regulator_ev.get("regulators", [])]
        elif isinstance(regulator_ev, dict):
            upstream_regs = list(regulator_ev.keys())

    novel_high_value: list[dict] = []
    evidence_gaps: list[dict] = []
    upgraded: list[dict] = []
    chokepoints: list[dict] = []

    for p in profiles:
        gene = p.get("gene", "")
        cls  = p.get("evidence_class", "state_nominated")

        if _compact:
            # Compact profile — fields are at top level
            gamma    = p.get("ota_gamma") or 0.0
            tau      = p.get("tau_disease")
            tau_cls  = p.get("tau_class")
            lit_conf = p.get("lit_confidence")
            rt_verd  = p.get("rt_verdict")
            max_ph   = p.get("max_phase") or 0
            drugs    = p.get("known_drugs") or []
            n_prog   = p.get("n_programs", 0)
            is_reg   = p.get("is_regulator", False)
        else:
            # Nested profile — fields in per-track sub-dicts
            ge   = p.get("genetic_evidence") or {}
            fe   = p.get("functional_evidence") or {}
            le   = p.get("literature_evidence") or {}
            ae   = p.get("adversarial_assessment") or {}
            te   = p.get("translational_evidence") or {}
            pe   = p.get("perturbseq_evidence") or {}
            gamma    = ge.get("ota_gamma") or 0.0
            tau      = fe.get("tau_disease")
            tau_cls  = fe.get("tau_specificity_class")
            lit_conf = le.get("literature_confidence")
            rt_verd  = ae.get("verdict")
            max_ph   = te.get("max_phase") or 0
            drugs    = te.get("known_drugs") or []
            n_prog   = pe.get("n_programs", 0)
            is_reg   = pe.get("is_upstream_regulator", False)

        if rt_verd == "DEPRIORITIZE":
            continue

        # ---- Convergent with no drug compound ----
        if cls == "convergent" and max_ph == 0 and not drugs:
            novel_high_value.append({
                "gene": gene,
                "opportunity_class": "convergent_needs_drug",
                "rationale": (
                    f"Convergent genetic+Perturb-seq evidence (γ={gamma:.3f}) with no known compound. "
                    "Highest-confidence mechanism, clear druggability gap."
                ),
                "evidence_gap_filled": None,
                "recommended_experiment": (
                    f"ChEMBL target sweep for {gene}; if no tool compound, commission "
                    "structure-based design using cryo-EM structure (if available) or "
                    "AlphaFold model for active-site identification."
                ),
                "urgency": "high",
            })

        # ---- Genetic anchor with no drug ----
        elif cls == "genetic_anchor" and max_ph == 0 and not drugs:
            novel_high_value.append({
                "gene": gene,
                "opportunity_class": "convergent_needs_drug",
                "rationale": (
                    f"Genetic causal instrument (γ={gamma:.3f}) — no Perturb-seq data but "
                    "genetically validated. No clinical compound identified."
                ),
                "evidence_gap_filled": None,
                "recommended_experiment": (
                    f"Query ChEMBL/OT for {gene} tool compounds; if absent, run "
                    "Perturb-seq in disease-relevant cell type to establish mechanistic context "
                    "before committing to drug discovery."
                ),
                "urgency": "high",
            })

        # ---- GWAS provisional — potential upgrade via eQTL ----
        elif cls == "gwas_provisional":
            evidence_gaps.append({
                "gene": gene,
                "gap_type": "missing_eqtl_instrument",
                "current_class": cls,
                "potential_upgrade": "genetic_anchor",
                "fill_experiment": (
                    f"Query GTEx eQTL for {gene} in IBD-relevant tissues (colon, ileum, immune cells). "
                    "If fine-mapped eQTL colocalizes with GWAS signal (H4 > 0.5), upgrade to Tier2."
                ),
                "priority": "medium",
            })

        # ---- State-nominated with high disease-τ ----
        elif cls == "state_nominated" and tau is not None and tau > 0.6:
            opp = {
                "gene": gene,
                "opportunity_class": "druggable_state_nominated",
                "rationale": (
                    f"State-space nominated, τ={tau:.2f} ({tau_cls}) — "
                    "disease-specifically expressed in macrophages. "
                    "Mechanistic credibility is high; genetic instrument is missing."
                ),
                "evidence_gap_filled": None,
                "recommended_experiment": (
                    f"(1) CRISPR KO of {gene} in primary macrophages with IBD-relevant functional readout "
                    "(cytokine secretion, phagocytosis, barrier function). "
                    "(2) Query OT for tractability; if small-molecule pocket exists, "
                    "commission ChEMBL screen. "
                    "(3) Search FinnGen/UK Biobank for coding variants in {gene} → genetic instrument."
                ),
                "urgency": "medium",
            }
            # If a drug exists for this state-nominated gene, it's higher value
            if drugs or max_ph > 0:
                opp["opportunity_class"] = "druggable_state_nominated"
                opp["urgency"] = "high"
                opp["rationale"] += (
                    f" Phase {max_ph} compound already exists ({', '.join(drugs[:2]) or 'unnamed'}) — "
                    "disease-specific expression makes this a strong repurposing candidate."
                )
                opp["recommended_experiment"] = (
                    f"Evaluate {', '.join(drugs[:1]) or 'existing compound'} in IBD organoid/macrophage model. "
                    "Confirm mechanistic target engagement via {gene} KO rescue experiment."
                )
            novel_high_value.append(opp)

        # ---- Novel — no literature ----
        if lit_conf == "NOVEL" and cls not in ("convergent",):
            novel_high_value.append({
                "gene": gene,
                "opportunity_class": "novel_unexplored",
                "rationale": (
                    f"{gene} has {cls} evidence class but no prior literature in this disease. "
                    "First-mover advantage if causal role confirmed."
                ),
                "evidence_gap_filled": None,
                "recommended_experiment": (
                    f"Systematic literature review: search '{gene} inflammation', '{gene} macrophage', "
                    f"'{gene} intestinal'. If no mechanism papers: annotate as de novo discovery target — "
                    "begin with CRISPR screen in disease model to confirm essentiality."
                ),
                "urgency": "medium",
            })

        # ---- Upstream chokepoint ----
        if is_reg and n_prog >= 2:
            chokepoints.append({
                "gene": gene,
                "n_programs": n_prog,
                "druggable": bool(drugs or max_ph > 0),
                "rationale": (
                    f"Perturb-seq upstream regulator controlling {n_prog} cNMF disease programs. "
                    "Inhibiting this single node affects multiple downstream disease mechanisms."
                ),
                "recommended_experiment": (
                    f"Map {gene}'s target gene network using CUT&RUN or ChIP-seq in macrophages. "
                    "Identify the most tractable downstream effector if {gene} itself is undruggable."
                ),
            })

    # Genes near evidence class boundary — escaped ranking
    for p in profiles:
        gene = p.get("gene", "")
        cls  = p.get("evidence_class", "state_nominated")
        if _compact:
            gamma = abs(p.get("ota_gamma") or 0.0)
        else:
            ge = p.get("genetic_evidence") or {}
            gamma = abs(ge.get("ota_gamma") or 0.0)
        if cls == "state_nominated" and gamma == 0.0:
            # No genetic anchor at all — check if it's in upstream regulators
            if gene in upstream_regs:
                evidence_gaps.append({
                    "gene": gene,
                    "gap_type": "escaped_ranking_perturb_regulator",
                    "current_class": cls,
                    "potential_upgrade": "perturb_seq_regulator",
                    "fill_experiment": (
                        f"{gene} is a Perturb-seq upstream regulator missing from top genetic hits. "
                        "Search GWAS catalog for common variants near this gene; check OT genetics L2G scores."
                    ),
                    "priority": "medium",
                })

    # Deduplicate novel_high_value by gene (keep first occurrence)
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in novel_high_value:
        g = item.get("gene", "")
        if g not in seen:
            deduped.append(item)
            seen.add(g)
    novel_high_value = deduped

    # Sort chokepoints by n_programs desc
    chokepoints.sort(key=lambda c: c.get("n_programs", 0), reverse=True)

    # Analysis summary
    n_high   = sum(1 for x in novel_high_value if x.get("urgency") == "high")
    n_medium = sum(1 for x in novel_high_value if x.get("urgency") == "medium")
    analysis_summary = (
        f"Heuristic second-pass for {disease_name}: "
        f"{n_high} high-urgency and {n_medium} medium-urgency novel opportunities identified. "
        f"{len(evidence_gaps)} evidence gaps flagged for targeted follow-up. "
        f"{len(chokepoints)} upstream regulatory chokepoint(s) identified."
    )

    return {
        "novel_high_value_targets":  novel_high_value[:10],
        "upgraded_evidence":         upgraded,
        "evidence_gaps":             evidence_gaps[:10],
        "chokepoint_regulators":     chokepoints[:5],
        "analysis_summary":          analysis_summary,
        "mode":                      "local_heuristic",
        "n_queries_run":             0,
        "disease_name":              disease_name,
        "efo_id":                    efo_id,
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def run(pipeline_outputs: dict, disease_query: dict) -> dict:
    """
    Entry point called by AgentRunner in local mode.

    In SDK mode, the agent receives a scoped dict and queries tools autonomously.
    In local mode, returns a heuristic classification without API calls.
    """
    merged = {**pipeline_outputs, **disease_query}
    return _local_run(merged)
