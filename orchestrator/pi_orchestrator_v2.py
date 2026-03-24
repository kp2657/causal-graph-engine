"""
pi_orchestrator_v2.py — Multiagent PI orchestrator using AgentRunner.

Replaces chief_of_staff._call_with_retry() with AgentRunner.dispatch(),
enabling per-agent mode switching (local → sdk) for gradual Claude API migration.

Key differences from pi_orchestrator.py / chief_of_staff.py:
  - Direct runner.dispatch() calls instead of chief_of_staff.run_pipeline()
  - T1b+T1c (genetics + somatic) run in parallel after phenotype_architect
  - T2a+T2b (perturbation + regulatory) run in parallel
  - T4b+T4c (chemistry + clinical) run in parallel after target_prioritization
  - All outputs wrapped in typed AgentOutput envelopes (edges_written, warnings, escalate)
  - Any agent can be flipped to Claude API: runner.set_mode("somatic_exposure_agent", "sdk")

Entry point:
    from orchestrator.pi_orchestrator_v2 import analyze_disease_v2
    result = analyze_disease_v2("coronary artery disease")

SDK mode proof-of-concept:
    result = analyze_disease_v2(
        "coronary artery disease",
        mode_overrides={"somatic_exposure_agent": "sdk"},
    )
"""
from __future__ import annotations

import concurrent.futures
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.agent_runner import AgentRunner
from orchestrator.message_contracts import AgentInput, AgentOutput

# Reuse quality gate logic from v1 — no need to duplicate
from orchestrator.pi_orchestrator import (
    ANCHOR_RECOVERY_MIN,
    EVALUE_MIN,
    VIRTUAL_TOP5_ESCALATE,
    _escalate,
    _run_quality_gate_tier1,
    _run_quality_gate_tier3,
    _run_quality_gate_tier4,
    _build_final_output,
)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S")


def _log(tag: str, agent: str, msg: str) -> None:
    print(f"[{tag:8s}] {agent:<30s} {msg}  ({_ts()})")


# ---------------------------------------------------------------------------
# Somatic edge write — must happen before Tier 3 (anchor recovery requires it)
# ---------------------------------------------------------------------------

def _write_somatic_edges(somatic_output: AgentOutput, disease_name: str) -> list[str]:
    """
    Write CHIP / drug / viral edges from the somatic agent to Kùzu.
    Returns a list of warning strings (empty = OK).
    """
    somatic_result = somatic_output.results
    edges: list[dict] = (
        somatic_result.get("chip_edges", [])
        + somatic_result.get("drug_edges", [])
        + somatic_result.get("viral_edges", [])
    )
    if not edges:
        return []

    warnings: list[str] = []
    try:
        from orchestrator.scientific_reviewer import review_batch
        from mcp_servers.graph_db_server import write_causal_edges

        reviewed = review_batch(edges)
        approved = reviewed.get("approved_edges", [])
        if approved:
            write_causal_edges(approved, disease_name)
            print(f"[SOMATIC  ] Wrote {len(approved)}/{len(edges)} CHIP/drug/viral edges to graph")
        else:
            warnings.append(f"Somatic edge review approved 0/{len(edges)} edges")
    except Exception as exc:
        warnings.append(f"Somatic edge write failed: {exc}")

    return warnings


# ---------------------------------------------------------------------------
# Gamma estimates — identical to chief_of_staff._get_gamma_estimates
# ---------------------------------------------------------------------------

def _get_gamma_estimates(disease_query: dict) -> dict:
    """
    Compute program→trait γ estimates using DISEASE_TRAIT_MAP + ThreadPoolExecutor.

    Returns {program: {trait: gamma_dict}} where each gamma_dict includes
    gamma, gamma_se, evidence_tier, and data_source.  When efo_id is present
    and program gene sets are available, live OT genetic association scores
    replace the hardcoded PROVISIONAL_GAMMAS table (estimate_gamma_live path).
    """
    from pipelines.ota_gamma_estimation import estimate_gamma
    from mcp_servers.burden_perturb_server import get_cnmf_program_info, get_program_gene_loadings
    from graph.schema import DISEASE_TRAIT_MAP, _DISEASE_SHORT_NAMES_FOR_ANCHORS

    disease_name = disease_query.get("disease_name", "coronary artery disease")
    efo_id       = disease_query.get("efo_id") or None
    short_name   = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_name.lower(), disease_name.upper())
    traits       = DISEASE_TRAIT_MAP.get(short_name, [short_name])

    programs_info = get_cnmf_program_info()
    raw_programs  = programs_info.get("programs", [])
    program_names = [
        p if isinstance(p, str) else (p.get("program_id") or p.get("name", ""))
        for p in raw_programs
    ]

    # Pre-fetch program gene sets once — needed for live OT γ estimation.
    program_gene_sets: dict[str, set[str]] = {}
    for pid in program_names:
        try:
            loadings_info = get_program_gene_loadings(pid)
            program_gene_sets[pid] = {
                g if isinstance(g, str) else g.get("gene", "")
                for g in loadings_info.get("top_genes", [])
                if g
            }
        except Exception:
            program_gene_sets[pid] = set()

    work = [(prog, trait) for prog in program_names for trait in traits]

    def _fetch(prog_trait: tuple[str, str]) -> tuple[str, str, dict]:
        prog, trait = prog_trait
        try:
            result = estimate_gamma(
                prog, trait,
                program_gene_set=program_gene_sets.get(prog) or None,
                efo_id=efo_id,
            )
            return prog, trait, result
        except Exception:
            return prog, trait, {
                "gamma": 0.0, "gamma_se": None,
                "evidence_tier": "provisional_virtual", "data_source": "error",
            }

    gamma_matrix: dict[str, dict[str, dict]] = {p: {} for p in program_names}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for prog, trait, val in pool.map(_fetch, work):
            gamma_matrix[prog][trait] = val

    return gamma_matrix


# ---------------------------------------------------------------------------
# Gene list — identical to chief_of_staff._collect_gene_list
# ---------------------------------------------------------------------------

def _collect_gene_list(
    disease_query: dict,
    genetics_output: AgentOutput,
    somatic_output: AgentOutput,
) -> list[str]:
    from graph.schema import ANCHOR_EDGES

    genetics_result = genetics_output.results
    somatic_result  = somatic_output.results
    genes: list[str] = []

    # Seed from disease-specific anchor edges — ensures all disease anchors are processed
    disease_name = disease_query.get("disease_name", "").lower()
    _disease_short_map = {
        "coronary artery disease": "CAD", "cad": "CAD",
        "inflammatory bowel disease": "IBD", "ibd": "IBD",
        "crohn's disease": "IBD", "crohns disease": "IBD", "ulcerative colitis": "IBD",
        "rheumatoid arthritis": "RA", "ra": "RA",
        "alzheimer's disease": "AD", "alzheimer disease": "AD",
        "type 2 diabetes": "T2D", "t2d": "T2D",
    }
    disease_short = _disease_short_map.get(disease_name, "")
    for edge in ANCHOR_EDGES:
        gene = edge.get("from", "")
        target = edge.get("to", "")
        # Include gene-level anchors for this disease (skip program and somatic edges)
        if (
            target.upper() == disease_short
            and gene
            and not gene.endswith("_chip")
            and not gene.endswith("_exposure")
            and not gene.endswith("_program")
            and gene not in genes
        ):
            genes.append(gene)

    # Add genes from somatic exposure (CHIP, drug targets)
    for edge in somatic_result.get("chip_edges", []):
        gene = edge.get("from_node", "").replace("_chip", "")
        if gene and gene not in genes:
            genes.append(gene)

    for edge in somatic_result.get("drug_edges", []):
        gene = edge.get("from_node", "")
        if gene and gene not in genes:
            genes.append(gene)

    # Add any additional validated anchor genes from the geneticist
    for gene in genetics_result.get("anchor_genes_validated", {}):
        if gene not in genes:
            genes.append(gene)

    return genes if genes else ["PCSK9", "LDLR", "TET2", "DNMT3A"]


# ---------------------------------------------------------------------------
# Tier runners
# ---------------------------------------------------------------------------

def _run_tier1(
    runner: AgentRunner,
    disease_name: str,
    run_id: str,
) -> tuple[AgentOutput, AgentOutput, AgentOutput]:
    """
    Tier 1: phenotype_architect → parallel(statistical_geneticist, somatic_exposure_agent).

    Returns:
        (phenotype_output, genetics_output, somatic_output)
    """
    print(f"\n{'='*60}\nTIER 1 — Phenomics: {disease_name}\n{'='*60}")

    # 1a — Phenotype Architect (serial; its output feeds 1b + 1c)
    _log("DISPATCH", "phenotype_architect", f"disease={disease_name!r}")
    t0 = time.time()
    phenotype_input = AgentInput(
        disease_query={"disease_name": disease_name},
        run_id=run_id,
    )
    phenotype_output = runner.dispatch("phenotype_architect", phenotype_input)
    _log("COMPLETE", "phenotype_architect",
         f"efo_id={phenotype_output.results.get('efo_id')}  {time.time()-t0:.1f}s")

    disease_query = phenotype_output.results

    # 1b + 1c — Genetics and Somatic in parallel
    _log("DISPATCH", "statistical_geneticist + somatic_exposure_agent", "parallel")

    genetics_input = AgentInput(
        disease_query=disease_query,
        upstream_results={"phenotype_architect": disease_query},
        run_id=run_id,
    )
    somatic_input = AgentInput(
        disease_query=disease_query,
        upstream_results={"phenotype_architect": disease_query},
        run_id=run_id,
    )

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        genetics_fut = pool.submit(runner.dispatch, "statistical_geneticist", genetics_input)
        somatic_fut  = pool.submit(runner.dispatch, "somatic_exposure_agent",  somatic_input)
        genetics_output = genetics_fut.result()
        somatic_output  = somatic_fut.result()

    _log("COMPLETE", "statistical_geneticist",
         f"n_instruments={len(genetics_output.results.get('instruments', []))}  {time.time()-t0:.1f}s")
    _log("COMPLETE", "somatic_exposure_agent",
         f"chip={somatic_output.results.get('summary', {}).get('n_chip_genes', 0)}, "
         f"drug={somatic_output.results.get('summary', {}).get('n_drug_targets', 0)}")

    return phenotype_output, genetics_output, somatic_output


def _run_tier2(
    runner: AgentRunner,
    disease_query: dict,
    gene_list: list[str],
    run_id: str,
) -> tuple[AgentOutput, AgentOutput]:
    """
    Tier 2: perturbation_genomics + regulatory_genomics in parallel.

    Returns:
        (beta_output, regulatory_output)
    """
    print(f"\n{'='*60}\nTIER 2 — Pathway: {len(gene_list)} genes\n{'='*60}")
    _log("DISPATCH", "perturbation + regulatory genomics", "parallel")

    t2_upstream = {"_gene_list": gene_list}

    beta_input = AgentInput(
        disease_query=disease_query,
        upstream_results=t2_upstream,
        run_id=run_id,
    )
    regulatory_input = AgentInput(
        disease_query=disease_query,
        upstream_results=t2_upstream,
        run_id=run_id,
    )

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        beta_fut       = pool.submit(runner.dispatch, "perturbation_genomics_agent", beta_input)
        regulatory_fut = pool.submit(runner.dispatch, "regulatory_genomics_agent",   regulatory_input)
        beta_output       = beta_fut.result()
        regulatory_output = regulatory_fut.result()

    _log("COMPLETE", "perturbation_genomics_agent",
         f"tier1={beta_output.results.get('n_tier1', 0)}, "
         f"virtual={beta_output.results.get('n_virtual', 0)}  {time.time()-t0:.1f}s")
    _log("COMPLETE", "regulatory_genomics_agent",
         f"tier2_upgrades={len(regulatory_output.results.get('tier2_upgrades', []))}")

    return beta_output, regulatory_output


def _run_tier3(
    runner: AgentRunner,
    disease_query: dict,
    beta_output: AgentOutput,
    gamma_estimates: dict,
    run_id: str,
) -> tuple[AgentOutput, AgentOutput]:
    """
    Tier 3: causal_discovery → kg_completion (sequential — KG needs causal graph).

    Returns:
        (causal_output, kg_output)
    """
    print(f"\n{'='*60}\nTIER 3 — Causal Discovery\n{'='*60}")

    # 3a — Causal Discovery
    _log("DISPATCH", "causal_discovery_agent", "beta_matrix + gamma_estimates")
    t0 = time.time()
    causal_input = AgentInput(
        disease_query=disease_query,
        upstream_results={
            "perturbation_genomics_agent": beta_output.results,
            "_gamma_estimates": gamma_estimates,
        },
        run_id=run_id,
    )
    causal_output = runner.dispatch("causal_discovery_agent", causal_input)
    recovery = causal_output.results.get("anchor_recovery", {}).get("recovery_rate", 0.0)
    _log("COMPLETE", "causal_discovery_agent",
         f"n_written={causal_output.results.get('n_edges_written', 0)}, "
         f"anchor_recovery={recovery:.0%}  {time.time()-t0:.1f}s")

    # 3b — KG Completion
    _log("DISPATCH", "kg_completion_agent", "causal graph")
    t0 = time.time()
    kg_input = AgentInput(
        disease_query=disease_query,
        upstream_results={"causal_discovery_agent": causal_output.results},
        run_id=run_id,
    )
    kg_output = runner.dispatch("kg_completion_agent", kg_input)
    _log("COMPLETE", "kg_completion_agent",
         f"pathways={kg_output.results.get('n_pathway_edges_added', 0)}, "
         f"drugs={kg_output.results.get('n_drug_target_edges_added', 0)}  {time.time()-t0:.1f}s")

    return causal_output, kg_output


def _run_tier4(
    runner: AgentRunner,
    disease_query: dict,
    causal_output: AgentOutput,
    kg_output: AgentOutput,
    run_id: str,
) -> tuple[AgentOutput, AgentOutput, AgentOutput]:
    """
    Tier 4: target_prioritization → parallel(chemistry_agent, clinical_trialist_agent).

    Returns:
        (prioritization_output, chemistry_output, clinical_output)
    """
    print(f"\n{'='*60}\nTIER 4 — Translation\n{'='*60}")

    # 4a — Target Prioritization (serial; chemistry + clinical depend on its output)
    _log("DISPATCH", "target_prioritization_agent", "causal graph + KG")
    t0 = time.time()
    prioritization_input = AgentInput(
        disease_query=disease_query,
        upstream_results={
            "causal_discovery_agent": causal_output.results,
            "kg_completion_agent":    kg_output.results,
        },
        run_id=run_id,
    )
    prioritization_output = runner.dispatch("target_prioritization_agent", prioritization_input)
    n_targets = len(prioritization_output.results.get("targets", []))
    _log("COMPLETE", "target_prioritization_agent",
         f"n_targets={n_targets}  {time.time()-t0:.1f}s")

    # 4b + 4c — Chemistry and Clinical in parallel
    _log("DISPATCH", "chemistry_agent + clinical_trialist_agent",
         f"parallel, n_targets={n_targets}")

    chemistry_input = AgentInput(
        disease_query=disease_query,
        upstream_results={"target_prioritization_agent": prioritization_output.results},
        run_id=run_id,
    )
    clinical_input = AgentInput(
        disease_query=disease_query,
        upstream_results={"target_prioritization_agent": prioritization_output.results},
        run_id=run_id,
    )

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        chemistry_fut = pool.submit(runner.dispatch, "chemistry_agent",         chemistry_input)
        clinical_fut  = pool.submit(runner.dispatch, "clinical_trialist_agent", clinical_input)
        chemistry_output = chemistry_fut.result()
        clinical_output  = clinical_fut.result()

    _log("COMPLETE", "chemistry_agent",
         f"n_repurposing={len(chemistry_output.results.get('repurposing_candidates', []))}  "
         f"{time.time()-t0:.1f}s")
    _log("COMPLETE", "clinical_trialist_agent",
         f"n_key_trials={len(clinical_output.results.get('key_trials', []))}")

    return prioritization_output, chemistry_output, clinical_output


def _run_tier5(
    runner: AgentRunner,
    all_outputs: dict[str, Any],
    run_id: str,
) -> tuple[AgentOutput, AgentOutput]:
    """
    Tier 5: scientific_writer_agent → scientific_reviewer_agent.

    The reviewer runs after the writer and applies the structured QA rubric.
    A REVISE verdict is logged as a warning but does not halt the pipeline
    (halting is reserved for anchor recovery < 80%).

    Returns:
        (writer_output, reviewer_output)
    """
    print(f"\n{'='*60}\nTIER 5 — Scientific Writer\n{'='*60}")
    _log("DISPATCH", "scientific_writer_agent", "all tier outputs")

    t0 = time.time()
    writer_input = AgentInput(
        disease_query=all_outputs.get("phenotype_result", {}),
        upstream_results={k: v for k, v in all_outputs.items() if k != "phenotype_result"},
        run_id=run_id,
    )
    writer_output = runner.dispatch("scientific_writer_agent", writer_input)
    _log("COMPLETE", "scientific_writer_agent",
         f"n_targets={len(writer_output.results.get('target_list', []))}, "
         f"anchor_recovery={writer_output.results.get('anchor_edge_recovery', 0):.0%}  "
         f"{time.time()-t0:.1f}s")

    # 5b — Scientific Reviewer (QA gate)
    _log("DISPATCH", "scientific_reviewer_agent", "QA rubric review")
    t0 = time.time()
    reviewer_input = AgentInput(
        disease_query=all_outputs.get("phenotype_result", {}),
        upstream_results=all_outputs,
        run_id=run_id,
    )
    reviewer_output = runner.dispatch("scientific_reviewer_agent", reviewer_input)
    verdict = reviewer_output.results.get("verdict", "APPROVE")
    n_crit  = reviewer_output.results.get("n_critical", 0)
    _log("COMPLETE", "scientific_reviewer_agent",
         f"verdict={verdict}, critical={n_crit}, "
         f"major={reviewer_output.results.get('n_major', 0)}  {time.time()-t0:.1f}s")

    if verdict == "REVISE":
        print(f"[REVIEWER ] {reviewer_output.results.get('summary', '')}")

    return writer_output, reviewer_output


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def analyze_disease_v2(
    disease_name: str,
    mode_overrides: dict[str, str] | None = None,
) -> dict:
    """
    Analyze a disease through the full 5-tier multiagent pipeline.

    Args:
        disease_name:    Human-readable name, e.g. "coronary artery disease"
        mode_overrides:  Optional per-agent mode overrides, e.g.
                         {"somatic_exposure_agent": "sdk"} to flip one agent
                         to Claude API while keeping all others local.

    Returns:
        GraphOutput-compatible dict with PI review metadata (same shape as v1).

    Raises:
        ValueError: if critical quality gate fails (anchor recovery < 80%).
    """
    run_id = datetime.now(tz=timezone.utc).isoformat()
    pipeline_start = time.time()
    all_warnings: list[str] = []

    print(f"\n{'='*60}")
    print(f"PI ORCHESTRATOR v2: {disease_name.upper()}")
    print(f"Started: {run_id}")
    print(f"{'='*60}")

    # Build runner and apply mode overrides
    runner = AgentRunner()
    if mode_overrides:
        for agent_name, mode in mode_overrides.items():
            runner.set_mode(agent_name, mode)
            print(f"[MODE     ] {agent_name} → {mode}")

    pipeline_outputs: dict[str, Any] = {}

    # =========================================================================
    # TIER 1
    # =========================================================================
    phenotype_output, genetics_output, somatic_output = _run_tier1(
        runner, disease_name, run_id
    )

    # Collect warnings + check contract
    for output in (phenotype_output, genetics_output, somatic_output):
        all_warnings.extend(output.warnings)

    disease_query = phenotype_output.results
    pipeline_outputs["phenotype_result"] = disease_query
    pipeline_outputs["genetics_result"]  = genetics_output.results
    pipeline_outputs["somatic_result"]   = somatic_output.results

    if phenotype_output.stub_fallback:
        pipeline_outputs.update({
            "pipeline_status": "FAILED_TIER1_PHENOTYPE",
            "all_warnings": all_warnings,
        })
        return _build_final_output(pipeline_outputs)

    # Tier 1 quality gate
    tier1_issues = _run_quality_gate_tier1(disease_query, genetics_output.results)
    all_warnings.extend(tier1_issues)
    for issue in tier1_issues:
        if "ESCALATE" in issue:
            _escalate(issue, {"disease": disease_name})

    # Write somatic edges to Kùzu (must happen before Tier 3 anchor check)
    somatic_write_warnings = _write_somatic_edges(somatic_output, disease_name)
    all_warnings.extend(somatic_write_warnings)

    gene_list = _collect_gene_list(disease_query, genetics_output, somatic_output)

    # =========================================================================
    # TIER 2
    # =========================================================================
    beta_output, regulatory_output = _run_tier2(
        runner, disease_query, gene_list, run_id
    )
    all_warnings.extend(beta_output.warnings)
    all_warnings.extend(regulatory_output.warnings)
    pipeline_outputs["beta_matrix_result"] = beta_output.results
    pipeline_outputs["regulatory_result"]  = regulatory_output.results

    gamma_estimates = _get_gamma_estimates(disease_query)

    # =========================================================================
    # TIER 3
    # =========================================================================
    causal_output, kg_output = _run_tier3(
        runner, disease_query, beta_output, gamma_estimates, run_id
    )
    all_warnings.extend(causal_output.warnings)
    all_warnings.extend(kg_output.warnings)
    pipeline_outputs["causal_result"] = causal_output.results
    pipeline_outputs["kg_result"]     = kg_output.results

    # Tier 3 quality gate (may raise ValueError / halt pipeline)
    try:
        tier3_issues = _run_quality_gate_tier3(causal_output.results)
        all_warnings.extend(tier3_issues)
    except ValueError as exc:
        _escalate(str(exc), {
            "disease":         disease_name,
            "anchor_recovery": causal_output.results.get("anchor_recovery"),
        })
        pipeline_outputs.update({
            "pipeline_status": "HALTED_ANCHOR_RECOVERY",
            "all_warnings":    all_warnings + [str(exc)],
        })
        return _build_final_output(pipeline_outputs)

    # =========================================================================
    # TIER 4
    # =========================================================================
    prioritization_output, chemistry_output, clinical_output = _run_tier4(
        runner, disease_query, causal_output, kg_output, run_id
    )
    all_warnings.extend(prioritization_output.warnings)
    all_warnings.extend(chemistry_output.warnings)
    all_warnings.extend(clinical_output.warnings)
    pipeline_outputs["prioritization_result"] = prioritization_output.results
    pipeline_outputs["chemistry_result"]      = chemistry_output.results
    pipeline_outputs["trials_result"]         = clinical_output.results

    # Tier 4 quality gate
    tier4_issues = _run_quality_gate_tier4(prioritization_output.results)
    all_warnings.extend(tier4_issues)
    for issue in tier4_issues:
        if "ESCALATE" in issue:
            _escalate(issue, {"disease": disease_name})

    # =========================================================================
    # TIER 5: Writer + Reviewer
    # =========================================================================
    writer_output, reviewer_output = _run_tier5(runner, pipeline_outputs, run_id)
    all_warnings.extend(writer_output.warnings)
    all_warnings.extend(reviewer_output.warnings)
    pipeline_outputs["graph_output"]    = writer_output.results
    pipeline_outputs["review_result"]   = reviewer_output.results

    # Surface reviewer issues as warnings so they appear in the output JSON
    for issue in reviewer_output.results.get("issues", []):
        if issue.get("severity") in ("CRITICAL", "MAJOR"):
            all_warnings.append(
                f"[REVIEWER/{issue['severity']}] {issue['check']}: {issue['description'][:200]}"
            )

    # Collect total edges written across all agents
    total_edges = sum([
        somatic_output.edges_written,
        causal_output.edges_written,
        kg_output.edges_written,
    ])

    total_duration = time.time() - pipeline_start
    pipeline_outputs.update({
        "pipeline_status":    "SUCCESS",
        "pipeline_duration_s": round(total_duration, 1),
        "all_warnings":       all_warnings,
        "total_edges_written": total_edges,
    })

    final = _build_final_output(pipeline_outputs)

    print(f"\n[PI v2 COMPLETE]")
    print(f"  Disease:          {disease_name}")
    print(f"  Targets ranked:   {len(final.get('target_list', []))}")
    print(f"  Anchor recovery:  {final.get('anchor_edge_recovery', 0):.0%}")
    print(f"  Escalations:      {final.get('n_escalations', 0)}")
    print(f"  Edges written:    {total_edges}")
    print(f"  Duration:         {total_duration:.1f}s")
    print(f"  Pipeline status:  {final.get('pipeline_status')}")

    return final
