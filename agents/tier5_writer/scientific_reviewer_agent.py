"""
scientific_reviewer_agent.py — QA loop: structured rubric review of pipeline output.

Implements the Virtual Biotech Supplementary Note II reviewer pattern.
After the scientific_writer_agent produces a draft report, this agent:

  1. Checks that every β in the top-ranked targets has causal evidence
     (Tier1/Tier2/Tier3 — not provisional_virtual)
  2. Verifies that top targets are benchmarked against known disease anchors
  3. Checks that tractability claims reference actual tractability_class data
  4. Verifies that effect sizes are reported (non-null ota_gamma + beta)
  5. Flags any provisional_virtual evidence leaking into the top-5 targets
  6. Checks SCONE bootstrap confidence where available

Returns:
  verdict: "APPROVE" | "REVISE"
  issues:  list of structured issues (severity, check, description, agent_to_revisit)
  summary: brief narrative for the PI

The reviewer does NOT rerun agents — it flags issues for human or automated
re-dispatch.  Critical issues (severity "CRITICAL") must be resolved before
the report is finalized.  Minor issues are logged as warnings.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Severity levels
SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_MAJOR    = "MAJOR"
SEVERITY_MINOR    = "MINOR"

# Virtual evidence should not appear in top-N clinical translation claims
TOP_N_VIRTUAL_CHECK = 5

# Minimum anchor recovery to approve
MIN_ANCHOR_RECOVERY = 0.80

# Minimum SCONE bootstrap confidence for edge to pass
MIN_SCONE_CONFIDENCE = 0.50


def run(pipeline_outputs: dict, disease_query: dict) -> dict:
    """
    Review all pipeline outputs against the structured rubric.

    Args:
        pipeline_outputs: Dict of all tier outputs (same keys as orchestrator collects)
        disease_query:    DiseaseQuery dict

    Returns:
        {
          "verdict":           "APPROVE" | "REVISE",
          "issues":            list[Issue dict],
          "n_critical":        int,
          "n_major":           int,
          "n_minor":           int,
          "summary":           str,
          "agent_to_revisit":  str | None,   # first critical issue's responsible agent
          "approved_targets":  list[str],    # genes that passed all checks
          "flagged_targets":   list[str],    # genes with issues
          "warnings":          list[str],
        }
    """
    issues: list[dict] = []
    warnings: list[str] = []

    disease_name       = disease_query.get("disease_name", "")
    beta_result        = pipeline_outputs.get("beta_matrix_result", {})
    causal_result      = pipeline_outputs.get("causal_result", {})
    prioritization     = pipeline_outputs.get("prioritization_result", {})
    chemistry_result   = pipeline_outputs.get("chemistry_result", {})
    writer_result      = pipeline_outputs.get("graph_output", {})

    targets            = prioritization.get("targets", [])
    top_targets        = targets[:TOP_N_VIRTUAL_CHECK]
    evidence_tier_map  = beta_result.get("evidence_tier_per_gene", {})
    anchor_recovery    = causal_result.get("anchor_recovery", {})

    # =========================================================================
    # Check A: No provisional_virtual β in top-5 targets
    # =========================================================================
    for rec in top_targets:
        gene  = rec.get("target_gene", "")
        tier  = rec.get("evidence_tier") or evidence_tier_map.get(gene, "provisional_virtual")
        score = rec.get("target_score", 0.0)

        if tier == "provisional_virtual":
            issues.append({
                "severity":         SEVERITY_CRITICAL,
                "check":            "A_virtual_in_top5",
                "gene":             gene,
                "description": (
                    f"{gene} (rank {rec.get('rank')}, score={score:.3f}) has "
                    f"evidence_tier=provisional_virtual. "
                    "Provisional-virtual β cannot support clinical translation claims. "
                    "Requires Tier 1 Perturb-seq or Tier 2 eQTL-MR evidence."
                ),
                "agent_to_revisit": "perturbation_genomics_agent",
            })

    # =========================================================================
    # Check B: Top targets benchmarked against known disease anchors
    # =========================================================================
    recovery_rate = anchor_recovery.get("recovery_rate", 0.0)
    missing_anchors = anchor_recovery.get("missing", [])

    if recovery_rate < MIN_ANCHOR_RECOVERY:
        issues.append({
            "severity":         SEVERITY_CRITICAL,
            "check":            "B_anchor_recovery",
            "gene":             None,
            "description": (
                f"Anchor recovery = {recovery_rate:.0%} < {MIN_ANCHOR_RECOVERY:.0%}. "
                f"Missing known causal edges: {missing_anchors}. "
                "Pipeline is missing validated disease-gene associations. "
                "Check gene list collection and causal edge writing."
            ),
            "agent_to_revisit": "causal_discovery_agent",
        })
    elif missing_anchors:
        issues.append({
            "severity":         SEVERITY_MINOR,
            "check":            "B_anchor_partial",
            "gene":             None,
            "description": (
                f"Recovery {recovery_rate:.0%} ≥ 80% but {len(missing_anchors)} anchors missing: "
                f"{missing_anchors}. Report should note these gaps."
            ),
            "agent_to_revisit": None,
        })

    # =========================================================================
    # Check C: Tractability claims have data backing
    # =========================================================================
    target_chemistry = chemistry_result.get("target_chemistry", {})
    for rec in top_targets:
        gene = rec.get("target_gene", "")
        chem = target_chemistry.get(gene, {})
        tractability = chem.get("tractability")

        # If writer claims druggable but chemistry has no tractability data
        if not tractability or tractability == "unknown":
            if rec.get("max_phase", 0) == 0:
                issues.append({
                    "severity":         SEVERITY_MAJOR,
                    "check":            "C_tractability_unsupported",
                    "gene":             gene,
                    "description": (
                        f"{gene}: no tractability_class data in chemistry result "
                        f"and max_phase=0. Any 'druggable' claim is unsupported. "
                        "Requires Open Targets tractability query or ChEMBL IC50."
                    ),
                    "agent_to_revisit": "chemistry_agent",
                })

    # =========================================================================
    # Check D: Effect sizes reported (non-null ota_gamma + non-zero beta)
    # =========================================================================
    for rec in top_targets:
        gene      = rec.get("target_gene", "")
        ota_gamma = rec.get("ota_gamma")

        if ota_gamma is None or ota_gamma == 0.0:
            issues.append({
                "severity":         SEVERITY_MAJOR,
                "check":            "D_missing_effect_size",
                "gene":             gene,
                "description": (
                    f"{gene}: ota_gamma is {ota_gamma!r}. "
                    "All reported targets must have a quantified causal effect size. "
                    "Zero-gamma edges should be filtered before ranking."
                ),
                "agent_to_revisit": "causal_discovery_agent",
            })

    # =========================================================================
    # Check E: SCONE bootstrap confidence (if available)
    # =========================================================================
    for rec in top_targets:
        gene          = rec.get("target_gene", "")
        scone_conf    = rec.get("scone_confidence")
        scone_flags   = rec.get("scone_flags", [])

        if scone_conf is not None and scone_conf < MIN_SCONE_CONFIDENCE:
            issues.append({
                "severity":         SEVERITY_MAJOR,
                "check":            "E_low_scone_confidence",
                "gene":             gene,
                "description": (
                    f"{gene}: SCONE bootstrap confidence={scone_conf:.2f} < {MIN_SCONE_CONFIDENCE}. "
                    "Edge did not survive majority of bootstrap resamples. "
                    "Consider downgrading to provisional or removing from top rankings."
                ),
                "agent_to_revisit": "causal_discovery_agent",
            })
        if "bootstrap_rejected" in scone_flags:
            issues.append({
                "severity":         SEVERITY_CRITICAL,
                "check":            "E_scone_bootstrap_rejected",
                "gene":             gene,
                "description": (
                    f"{gene}: SCONE bootstrap rejected this edge (confidence < {MIN_SCONE_CONFIDENCE}). "
                    "Edge should not appear in top targets."
                ),
                "agent_to_revisit": "causal_discovery_agent",
            })

    # =========================================================================
    # Check F: Cell-type specificity sanity — warn if highly-ranked target
    #           is ubiquitous (tau < 0.3)
    # =========================================================================
    for rec in top_targets[:3]:  # top 3 only — focus reviewer attention
        gene = rec.get("target_gene", "")
        tau  = rec.get("tau_specificity")
        if tau is not None and tau < 0.30:
            issues.append({
                "severity":         SEVERITY_MINOR,
                "check":            "F_low_specificity",
                "gene":             gene,
                "description": (
                    f"{gene}: tau={tau:.2f} (ubiquitous expression across tissues). "
                    "High causal score but low specificity increases systemic AE risk. "
                    "Add specificity caveat to report."
                ),
                "agent_to_revisit": "scientific_writer_agent",
            })

    # =========================================================================
    # Determine verdict
    # =========================================================================
    n_critical = sum(1 for i in issues if i["severity"] == SEVERITY_CRITICAL)
    n_major    = sum(1 for i in issues if i["severity"] == SEVERITY_MAJOR)
    n_minor    = sum(1 for i in issues if i["severity"] == SEVERITY_MINOR)

    verdict = "APPROVE" if n_critical == 0 and n_major <= 1 else "REVISE"

    # First critical issue's agent
    first_critical = next(
        (i for i in issues if i["severity"] == SEVERITY_CRITICAL), None
    )
    agent_to_revisit = first_critical["agent_to_revisit"] if first_critical else None

    # Summary narrative
    if verdict == "APPROVE":
        summary = (
            f"APPROVE: {disease_name} pipeline passed QA review. "
            f"Anchor recovery={recovery_rate:.0%}, "
            f"n_critical=0, n_major={n_major}, n_minor={n_minor}."
        )
    else:
        summary = (
            f"REVISE: {disease_name} pipeline has {n_critical} critical and "
            f"{n_major} major issues. "
            f"Primary concern: {first_critical['description'][:120] if first_critical else 'see issues list'}. "
            f"Route to {agent_to_revisit} for correction."
        )

    # Approved vs flagged targets
    flagged_genes = {i["gene"] for i in issues if i.get("gene")}
    approved_targets = [r["target_gene"] for r in top_targets if r["target_gene"] not in flagged_genes]
    flagged_targets  = [r["target_gene"] for r in top_targets if r["target_gene"] in flagged_genes]

    return {
        "verdict":           verdict,
        "issues":            issues,
        "n_critical":        n_critical,
        "n_major":           n_major,
        "n_minor":           n_minor,
        "summary":           summary,
        "agent_to_revisit":  agent_to_revisit,
        "approved_targets":  approved_targets,
        "flagged_targets":   flagged_targets,
        "anchor_recovery":   recovery_rate,
        "warnings":          warnings,
    }
