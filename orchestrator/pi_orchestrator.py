"""
pi_orchestrator.py — Principal Investigator top-level orchestrator.

Enforces quality gates, scientific standards, and escalation rules.
Coordinates with chief_of_staff for agent dispatch and scientific_reviewer
for evidence validation.

Entry point for the full pipeline:
    from orchestrator.pi_orchestrator import analyze_disease
    result = analyze_disease("coronary artery disease")
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Quality thresholds (from prompt spec)
ANCHOR_RECOVERY_MIN      = 0.80
MR_P_THRESHOLD           = 5e-8
F_STATISTIC_MIN          = 10.0
EVALUE_MIN               = 2.0
VIRTUAL_TOP5_ESCALATE    = True   # escalate if all top-5 are provisional_virtual


def _escalate(message: str, context: dict) -> None:
    """Print a clearly formatted escalation notice to the user."""
    print(f"\n{'!'*60}")
    print(f"[ESCALATION REQUIRED]")
    print(f"{'!'*60}")
    print(message)
    print(f"Context: {context}")
    print(f"{'!'*60}\n")


def _run_quality_gate_tier1(disease_query: dict, genetics_result: dict) -> list[str]:
    """
    Quality gate after Tier 1: instrument validity checks.
    Returns list of escalation messages (empty = OK).
    """
    issues: list[str] = []

    # Weak instruments check
    for inst in genetics_result.get("instruments", []):
        f_stat = inst.get("f_statistic")
        if f_stat is not None and float(f_stat) < F_STATISTIC_MIN:
            issues.append(
                f"ESCALATE: {inst['exposure']} instruments F-statistic={f_stat:.1f} < {F_STATISTIC_MIN} "
                "(weak instruments — MR estimates unreliable)"
            )

    # Anchor gene validation failures
    not_validated = [
        gene for gene, ok in genetics_result.get("anchor_genes_validated", {}).items()
        if not ok
    ]
    if not_validated:
        issues.append(
            f"WARNING: Anchor genes not validated by eQTL: {not_validated} "
            "(check tissue / gene ID)"
        )

    return issues


def _run_quality_gate_tier3(causal_result: dict) -> list[str]:
    """
    Quality gate after Tier 3: anchor recovery and E-value.
    Returns escalation messages; critical issues raise ValueError.
    """
    issues: list[str] = []

    recovery = causal_result.get("anchor_recovery", {}).get("recovery_rate", 0.0)
    missing  = causal_result.get("anchor_recovery", {}).get("missing", [])

    if recovery < ANCHOR_RECOVERY_MIN:
        msg = (
            f"CRITICAL: Anchor edge recovery {recovery:.0%} < {ANCHOR_RECOVERY_MIN:.0%}. "
            f"Missing anchors: {missing}. "
            "Pipeline HALTED — do not proceed to Tier 4 without resolving missing anchors."
        )
        issues.append(msg)
        raise ValueError(msg)  # Hard stop per spec

    # E-value warnings from causal result
    for warning in causal_result.get("warnings", []):
        if "E-value" in warning:
            issues.append(f"QUALITY: {warning}")

    return issues


def _run_quality_gate_tier4(prioritization_result: dict) -> list[str]:
    """
    Quality gate after Tier 4: virtual target escalation.
    """
    issues: list[str] = []
    targets = prioritization_result.get("targets", [])

    if not targets:
        issues.append("ESCALATE: No targets returned from Tier 4 — check pipeline inputs")
        return issues

    top5_tiers = [t.get("evidence_tier") for t in targets[:5]]
    if VIRTUAL_TOP5_ESCALATE and all(t == "provisional_virtual" for t in top5_tiers):
        issues.append(
            "ESCALATE: All top-5 targets are provisional_virtual — "
            "pipeline output is hypothesis-generating only; no Tier1/2 evidence available. "
            "Download Replogle 2022 h5ad data before clinical translation."
        )

    # Check for safety signals in high-ranked targets
    for t in targets[:3]:
        if t.get("safety_flags"):
            issues.append(
                f"SAFETY: {t['target_gene']} (Rank {t['rank']}): {t['safety_flags']}"
            )

    return issues


def _build_final_output(pipeline_outputs: dict) -> dict:
    """
    Package the final GraphOutput from pipeline outputs.
    """
    graph_output = pipeline_outputs.get("graph_output", {})
    disease_query = pipeline_outputs.get("phenotype_result", {})

    # Enrich with PI-level metadata
    return {
        **graph_output,
        "pi_reviewed":       True,
        "pipeline_version":  graph_output.get("pipeline_version", "0.1.0"),
        "generated_at":      datetime.now(tz=timezone.utc).isoformat(),
        "disease_name":      disease_query.get("disease_name", ""),
        "efo_id":            disease_query.get("efo_id", ""),
        "pipeline_duration_s": pipeline_outputs.get("pipeline_duration_s"),
        "pipeline_status":   pipeline_outputs.get("pipeline_status", "UNKNOWN"),
        "n_escalations":     len([
            w for w in pipeline_outputs.get("all_warnings", [])
            if "ESCALATE" in w or "CRITICAL" in w
        ]),
    }


def analyze_disease(disease_name: str) -> dict:
    """
    Top-level entry point: analyze a disease through the full 5-tier pipeline.

    Args:
        disease_name: Human-readable disease name, e.g. "coronary artery disease"

    Returns:
        GraphOutput-compatible dict with PI review metadata.

    Raises:
        ValueError: if critical quality gate fails (anchor recovery < 80%)
    """
    from orchestrator.chief_of_staff import run_pipeline

    print(f"\n{'='*60}")
    print(f"PI ORCHESTRATOR: {disease_name.upper()}")
    print(f"Started: {datetime.now(tz=timezone.utc).isoformat()}")
    print(f"{'='*60}")

    # Run the full pipeline via Chief of Staff
    pipeline_outputs = run_pipeline(disease_name)

    # Extract individual tier results for quality gates
    genetics_result       = pipeline_outputs.get("genetics_result", {})
    causal_result         = pipeline_outputs.get("causal_result", {})
    prioritization_result = pipeline_outputs.get("prioritization_result", {})
    all_warnings          = pipeline_outputs.get("all_warnings", [])

    # Run quality gates
    escalations: list[str] = []

    # Tier 1 gate
    tier1_issues = _run_quality_gate_tier1(
        pipeline_outputs.get("phenotype_result", {}), genetics_result
    )
    escalations.extend(tier1_issues)
    for issue in tier1_issues:
        if "ESCALATE" in issue:
            _escalate(issue, {"disease": disease_name})

    # Tier 3 gate (may raise ValueError)
    try:
        tier3_issues = _run_quality_gate_tier3(causal_result)
        escalations.extend(tier3_issues)
    except ValueError as exc:
        _escalate(str(exc), {
            "disease":         disease_name,
            "anchor_recovery": causal_result.get("anchor_recovery"),
        })
        pipeline_outputs["pipeline_status"] = "HALTED_ANCHOR_RECOVERY"
        pipeline_outputs["all_warnings"] = all_warnings + [str(exc)]
        return _build_final_output(pipeline_outputs)

    # Tier 4 gate
    tier4_issues = _run_quality_gate_tier4(prioritization_result)
    escalations.extend(tier4_issues)
    for issue in tier4_issues:
        if "ESCALATE" in issue:
            _escalate(issue, {"disease": disease_name})

    # Attach escalations to all_warnings
    pipeline_outputs["all_warnings"] = all_warnings + escalations

    # Build and return final output
    final = _build_final_output(pipeline_outputs)

    print(f"\n[PI REVIEW COMPLETE]")
    print(f"  Disease:          {disease_name}")
    print(f"  Targets ranked:   {len(final.get('target_list', []))}")
    print(f"  Anchor recovery:  {final.get('anchor_edge_recovery', 0):.0%}")
    print(f"  Escalations:      {final.get('n_escalations', 0)}")
    print(f"  Pipeline status:  {final.get('pipeline_status')}")

    return final


def review_single_edge(edge: dict) -> dict:
    """
    Convenience wrapper: run Scientific Reviewer on a single edge.
    """
    from orchestrator.scientific_reviewer import review_edge
    return review_edge(edge)


def check_graph_contradictions(new_edges: list[dict], disease_name: str) -> dict:
    """
    Convenience wrapper: run Contradiction Agent on a list of proposed edges.
    """
    from orchestrator.contradiction_agent import run as check_contradictions
    return check_contradictions(new_edges, disease_name)
