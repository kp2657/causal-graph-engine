"""
chief_of_staff.py — Operational coordinator between PI and tier agents.

Handles:
  - Sequential/parallel dispatch of tier agents
  - Schema validation of agent outputs
  - Retry logic (max 2 attempts per agent)
  - Structured logging of all dispatches
  - Scientific reviewer gate before graph writes
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))


MAX_RETRIES = 2


def _ts() -> str:
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S")


def _log_dispatch(agent: str, input_summary: str) -> None:
    print(f"[DISPATCH] {agent} ← {input_summary} ({_ts()})")


def _log_complete(agent: str, output_summary: str, duration: float) -> None:
    print(f"[COMPLETE] {agent} → {output_summary} ({duration:.1f}s)")


def _log_error(agent: str, error: str, action: str) -> None:
    print(f"[ERROR]    {agent} → {error} [{action}]")


def _call_with_retry(fn, *args, agent_name: str, **kwargs) -> dict:
    """
    Call an agent function with up to MAX_RETRIES attempts.

    Returns the result dict or a stub-fallback dict on final failure.
    """
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            _log_error(
                agent_name,
                str(exc),
                "RETRY" if attempt < MAX_RETRIES - 1 else "FALLBACK",
            )
    # Return stub fallback
    return {
        "error":         str(last_exc),
        "stub_fallback": True,
        "agent":         agent_name,
        "warnings":      [f"Agent {agent_name} failed after {MAX_RETRIES} attempts: {last_exc}"],
    }


def _validate_required_keys(result: dict, required: list[str], agent_name: str) -> list[str]:
    """Return list of missing required keys (empty = valid)."""
    return [k for k in required if k not in result]


# Required output keys per agent — used to catch format contract violations early.
_REQUIRED_OUTPUT_KEYS: dict[str, list[str]] = {
    "phenotype_architect":         ["disease_name", "efo_id"],
    "statistical_geneticist":      ["instruments", "anchor_genes_validated"],
    "somatic_exposure_agent":      ["chip_edges", "drug_edges", "viral_edges"],
    "perturbation_genomics_agent": ["genes", "beta_matrix", "evidence_tier_per_gene"],
    "regulatory_genomics_agent":   ["gene_eqtl_summary", "tier2_upgrades"],
    "causal_discovery_agent":      ["n_edges_written", "anchor_recovery", "top_genes"],
    "kg_completion_agent":         ["n_pathway_edges_added"],
    "target_prioritization_agent": ["targets"],
    "chemistry_agent":             ["repurposing_candidates"],
    "clinical_trialist_agent":     ["trial_summary", "key_trials"],
    "scientific_writer_agent":     ["target_list", "executive_summary"],
}


def _check_agent_output(result: dict, agent_name: str, all_warnings: list[str]) -> None:
    """
    Warn loudly if a non-stub agent result is missing required keys.
    Avoids silent propagation of empty dicts that cause downstream failures.
    """
    if result.get("stub_fallback"):
        return
    missing = _validate_required_keys(
        result, _REQUIRED_OUTPUT_KEYS.get(agent_name, []), agent_name
    )
    if missing:
        msg = f"[CONTRACT] {agent_name} output missing required keys: {missing}"
        print(msg)
        all_warnings.append(msg)


# ---------------------------------------------------------------------------
# Individual agent dispatch helpers
# ---------------------------------------------------------------------------

def _run_phenotype_architect(disease_name: str) -> dict:
    from agents.tier1_phenomics.phenotype_architect import run
    return run(disease_name)


def _run_statistical_geneticist(disease_query: dict) -> dict:
    from agents.tier1_phenomics.statistical_geneticist import run
    return run(disease_query)


def _run_somatic_exposure(disease_query: dict) -> dict:
    from agents.tier1_phenomics.somatic_exposure_agent import run
    return run(disease_query)


def _run_perturbation_genomics(gene_list: list[str], disease_query: dict) -> dict:
    from agents.tier2_pathway.perturbation_genomics_agent import run
    return run(gene_list, disease_query)


def _run_regulatory_genomics(gene_list: list[str], disease_query: dict) -> dict:
    from agents.tier2_pathway.regulatory_genomics_agent import run
    return run(gene_list, disease_query)


def _run_causal_discovery(beta_result: dict, gamma_estimates: dict, disease_query: dict) -> dict:
    from agents.tier3_causal.causal_discovery_agent import run
    return run(beta_result, gamma_estimates, disease_query)


def _run_kg_completion(causal_result: dict, disease_query: dict) -> dict:
    from agents.tier3_causal.kg_completion_agent import run
    return run(causal_result, disease_query)


def _run_target_prioritization(causal_result: dict, kg_result: dict, disease_query: dict) -> dict:
    from agents.tier4_translation.target_prioritization_agent import run
    return run(causal_result, kg_result, disease_query)


def _run_chemistry(prioritization_result: dict, disease_query: dict) -> dict:
    from agents.tier4_translation.chemistry_agent import run
    return run(prioritization_result, disease_query)


def _run_clinical_trialist(prioritization_result: dict, disease_query: dict) -> dict:
    from agents.tier4_translation.clinical_trialist_agent import run
    return run(prioritization_result, disease_query)


def _run_scientific_writer(all_outputs: dict) -> dict:
    from agents.tier5_writer.scientific_writer_agent import run
    return run(**all_outputs)


# ---------------------------------------------------------------------------
# Gamma estimates stub (until full ota_gamma_estimation pipeline runs)
# ---------------------------------------------------------------------------

def _get_gamma_estimates(disease_query: dict) -> dict:
    """
    Retrieve or compute program→trait γ estimates.

    Returns {program: {trait: gamma_float}} for use in causal_discovery_agent.
    Uses DISEASE_TRAIT_MAP from schema as single source of truth for which traits
    each disease requires, so adding a new disease only requires updating schema.py.
    """
    import concurrent.futures
    from pipelines.ota_gamma_estimation import estimate_gamma
    from mcp_servers.burden_perturb_server import get_cnmf_program_info
    from graph.schema import DISEASE_TRAIT_MAP, _DISEASE_SHORT_NAMES_FOR_ANCHORS

    disease_name = disease_query.get("disease_name", "coronary artery disease")
    short_name   = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_name.lower(), disease_name.upper())
    traits       = DISEASE_TRAIT_MAP.get(short_name, [short_name])

    programs_info = get_cnmf_program_info()
    raw_programs  = programs_info.get("programs", [])
    program_names = [
        p if isinstance(p, str) else (p.get("program_id") or p.get("name", ""))
        for p in raw_programs
    ]

    # Build the (prog, trait) work list and fan out in a thread pool.
    # estimate_gamma is I/O-light today but will hit GWAS/LDSC APIs when wired.
    work = [(prog, trait) for prog in program_names for trait in traits]

    def _fetch(prog_trait: tuple[str, str]) -> tuple[str, str, float]:
        prog, trait = prog_trait
        try:
            result = estimate_gamma(prog, trait)
            return prog, trait, float(result.get("gamma", 0.0))
        except Exception:
            return prog, trait, 0.0

    gamma_matrix: dict[str, dict[str, float]] = {p: {} for p in program_names}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for prog, trait, val in pool.map(_fetch, work):
            gamma_matrix[prog][trait] = val

    return gamma_matrix


# ---------------------------------------------------------------------------
# Main dispatch pipeline
# ---------------------------------------------------------------------------

def run_pipeline(disease_name: str) -> dict:
    """
    Execute the full 5-tier agent pipeline for a disease.

    Args:
        disease_name: Human-readable disease name, e.g. "coronary artery disease"

    Returns:
        Full pipeline outputs keyed by tier and agent name.
    """
    pipeline_start = time.time()
    all_outputs: dict[str, Any] = {}
    all_warnings: list[str] = []

    # =====================================================================
    # TIER 1 — Phenomics
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"TIER 1 — Phenomics: {disease_name}")
    print(f"{'='*60}")

    # 1a. Phenotype Architect
    _log_dispatch("phenotype_architect", f"disease_name={disease_name!r}")
    t0 = time.time()
    disease_query = _call_with_retry(
        _run_phenotype_architect, disease_name,
        agent_name="phenotype_architect",
    )
    _log_complete("phenotype_architect", f"efo_id={disease_query.get('efo_id')}", time.time() - t0)
    _check_agent_output(disease_query, "phenotype_architect", all_warnings)
    all_outputs["phenotype_result"] = disease_query
    all_warnings.extend(disease_query.get("warnings", []))

    if disease_query.get("stub_fallback"):
        all_outputs["pipeline_status"] = "FAILED_TIER1_PHENOTYPE"
        return all_outputs

    # 1b. Statistical Geneticist
    _log_dispatch("statistical_geneticist", f"efo_id={disease_query.get('efo_id')}")
    t0 = time.time()
    genetics_result = _call_with_retry(
        _run_statistical_geneticist, disease_query,
        agent_name="statistical_geneticist",
    )
    _log_complete(
        "statistical_geneticist",
        f"n_instruments={len(genetics_result.get('instruments', []))}",
        time.time() - t0,
    )
    _check_agent_output(genetics_result, "statistical_geneticist", all_warnings)
    all_outputs["genetics_result"] = genetics_result
    all_warnings.extend(genetics_result.get("warnings", []))

    # 1c. Somatic Exposure Agent
    _log_dispatch("somatic_exposure_agent", f"modifier_types={disease_query.get('modifier_types')}")
    t0 = time.time()
    somatic_result = _call_with_retry(
        _run_somatic_exposure, disease_query,
        agent_name="somatic_exposure_agent",
    )
    _log_complete(
        "somatic_exposure_agent",
        f"chip={somatic_result.get('summary', {}).get('n_chip_genes', 0)}, "
        f"drug={somatic_result.get('summary', {}).get('n_drug_targets', 0)}",
        time.time() - t0,
    )
    _check_agent_output(somatic_result, "somatic_exposure_agent", all_warnings)
    all_outputs["somatic_result"] = somatic_result
    all_warnings.extend(somatic_result.get("warnings", []))

    # Write CHIP and drug edges to Kùzu now — they are needed for anchor recovery in Tier 3
    somatic_edges: list[dict] = (
        somatic_result.get("chip_edges", [])
        + somatic_result.get("drug_edges", [])
        + somatic_result.get("viral_edges", [])
    )
    if somatic_edges:
        try:
            from orchestrator.scientific_reviewer import review_batch
            from mcp_servers.graph_db_server import write_causal_edges
            reviewed = review_batch(somatic_edges)
            approved = reviewed.get("approved_edges", [])
            if approved:
                write_causal_edges(approved, disease_name)
                print(f"[SOMATIC] Wrote {len(approved)} CHIP/drug/viral edges to graph")
        except Exception as exc:
            all_warnings.append(f"Somatic edge write failed: {exc}")

    # Collect gene list from all Tier 1 outputs
    gene_list: list[str] = _collect_gene_list(disease_query, genetics_result, somatic_result)

    # =====================================================================
    # TIER 2 — Pathway
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"TIER 2 — Pathway: {len(gene_list)} genes")
    print(f"{'='*60}")

    # 2a. Perturbation Genomics
    _log_dispatch("perturbation_genomics_agent", f"n_genes={len(gene_list)}")
    t0 = time.time()
    beta_result = _call_with_retry(
        _run_perturbation_genomics, gene_list, disease_query,
        agent_name="perturbation_genomics_agent",
    )
    _log_complete(
        "perturbation_genomics_agent",
        f"tier1={beta_result.get('n_tier1', 0)}, "
        f"virtual={beta_result.get('n_virtual', 0)}",
        time.time() - t0,
    )
    _check_agent_output(beta_result, "perturbation_genomics_agent", all_warnings)
    all_outputs["beta_matrix_result"] = beta_result
    all_warnings.extend(beta_result.get("warnings", []))

    # 2b. Regulatory Genomics
    _log_dispatch("regulatory_genomics_agent", f"n_genes={len(gene_list)}")
    t0 = time.time()
    regulatory_result = _call_with_retry(
        _run_regulatory_genomics, gene_list, disease_query,
        agent_name="regulatory_genomics_agent",
    )
    _log_complete(
        "regulatory_genomics_agent",
        f"tier2_upgrades={len(regulatory_result.get('tier2_upgrades', []))}",
        time.time() - t0,
    )
    all_outputs["regulatory_result"] = regulatory_result
    all_warnings.extend(regulatory_result.get("warnings", []))

    # Get γ estimates
    gamma_estimates = _get_gamma_estimates(disease_query)

    # =====================================================================
    # TIER 3 — Causal
    # =====================================================================
    print(f"\n{'='*60}")
    print("TIER 3 — Causal Discovery")
    print(f"{'='*60}")

    # 3a. Causal Discovery
    _log_dispatch("causal_discovery_agent", "beta_matrix + gamma_estimates")
    t0 = time.time()
    causal_result = _call_with_retry(
        _run_causal_discovery, beta_result, gamma_estimates, disease_query,
        agent_name="causal_discovery_agent",
    )
    recovery = causal_result.get("anchor_recovery", {}).get("recovery_rate", 0.0)
    _log_complete(
        "causal_discovery_agent",
        f"n_written={causal_result.get('n_edges_written', 0)}, "
        f"anchor_recovery={recovery:.0%}",
        time.time() - t0,
    )
    _check_agent_output(causal_result, "causal_discovery_agent", all_warnings)
    all_outputs["causal_result"] = causal_result
    all_warnings.extend(causal_result.get("warnings", []))

    # Quality gate: anchor recovery ≥ 80%
    if recovery < 0.80:
        all_outputs["pipeline_status"] = f"QUALITY_GATE_FAILED: anchor_recovery={recovery:.0%} < 80%"
        all_outputs["all_warnings"] = all_warnings
        return all_outputs

    # 3b. KG Completion
    _log_dispatch("kg_completion_agent", "causal graph")
    t0 = time.time()
    kg_result = _call_with_retry(
        _run_kg_completion, causal_result, disease_query,
        agent_name="kg_completion_agent",
    )
    _log_complete(
        "kg_completion_agent",
        f"pathways={kg_result.get('n_pathway_edges_added', 0)}, "
        f"drugs={kg_result.get('n_drug_target_edges_added', 0)}",
        time.time() - t0,
    )
    all_outputs["kg_result"] = kg_result
    all_warnings.extend(kg_result.get("warnings", []))

    # =====================================================================
    # TIER 4 — Translation
    # =====================================================================
    print(f"\n{'='*60}")
    print("TIER 4 — Translation")
    print(f"{'='*60}")

    # 4a. Target Prioritization
    _log_dispatch("target_prioritization_agent", "causal graph + KG")
    t0 = time.time()
    prioritization_result = _call_with_retry(
        _run_target_prioritization, causal_result, kg_result, disease_query,
        agent_name="target_prioritization_agent",
    )
    n_targets = len(prioritization_result.get("targets", []))
    _log_complete("target_prioritization_agent", f"n_targets={n_targets}", time.time() - t0)
    all_outputs["prioritization_result"] = prioritization_result
    all_warnings.extend(prioritization_result.get("warnings", []))

    # Quality gate: at least one non-virtual target
    top_targets = prioritization_result.get("targets", [])
    if top_targets:
        top5_tiers = [t.get("evidence_tier") for t in top_targets[:5]]
        if all(t == "provisional_virtual" for t in top5_tiers):
            all_warnings.append(
                "ESCALATE: All top-5 targets are provisional_virtual — "
                "real evidence pipeline required before clinical translation"
            )

    # 4b. Chemistry (parallel dispatch conceptually; sequential here)
    _log_dispatch("chemistry_agent", f"n_targets={n_targets}")
    t0 = time.time()
    chemistry_result = _call_with_retry(
        _run_chemistry, prioritization_result, disease_query,
        agent_name="chemistry_agent",
    )
    _log_complete(
        "chemistry_agent",
        f"n_repurposing={len(chemistry_result.get('repurposing_candidates', []))}",
        time.time() - t0,
    )
    all_outputs["chemistry_result"] = chemistry_result
    all_warnings.extend(chemistry_result.get("warnings", []))

    # 4c. Clinical Trialist
    _log_dispatch("clinical_trialist_agent", f"n_targets={n_targets}")
    t0 = time.time()
    trials_result = _call_with_retry(
        _run_clinical_trialist, prioritization_result, disease_query,
        agent_name="clinical_trialist_agent",
    )
    _log_complete(
        "clinical_trialist_agent",
        f"n_key_trials={len(trials_result.get('key_trials', []))}",
        time.time() - t0,
    )
    all_outputs["trials_result"] = trials_result
    all_warnings.extend(trials_result.get("warnings", []))

    # =====================================================================
    # TIER 5 — Scientific Writer
    # =====================================================================
    print(f"\n{'='*60}")
    print("TIER 5 — Scientific Writer")
    print(f"{'='*60}")

    writer_inputs = {
        "phenotype_result":      all_outputs["phenotype_result"],
        "genetics_result":       all_outputs["genetics_result"],
        "somatic_result":        all_outputs["somatic_result"],
        "beta_matrix_result":    all_outputs["beta_matrix_result"],
        "regulatory_result":     all_outputs["regulatory_result"],
        "causal_result":         all_outputs["causal_result"],
        "kg_result":             all_outputs["kg_result"],
        "prioritization_result": all_outputs["prioritization_result"],
        "chemistry_result":      all_outputs["chemistry_result"],
        "trials_result":         all_outputs["trials_result"],
    }

    _log_dispatch("scientific_writer_agent", "all tier outputs")
    t0 = time.time()
    graph_output = _call_with_retry(
        _run_scientific_writer, writer_inputs,
        agent_name="scientific_writer_agent",
    )
    _log_complete(
        "scientific_writer_agent",
        f"n_targets={len(graph_output.get('target_list', []))}, "
        f"anchor_recovery={graph_output.get('anchor_edge_recovery', 0):.0%}",
        time.time() - t0,
    )
    all_outputs["graph_output"] = graph_output
    all_warnings.extend(graph_output.get("warnings", []))

    total_duration = time.time() - pipeline_start
    all_outputs["pipeline_status"] = "SUCCESS"
    all_outputs["pipeline_duration_s"] = round(total_duration, 1)
    all_outputs["all_warnings"] = all_warnings

    print(f"\n[DONE] Pipeline completed in {total_duration:.1f}s")
    return all_outputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_gene_list(
    disease_query: dict,
    genetics_result: dict,
    somatic_result: dict,
) -> list[str]:
    """
    Aggregate gene targets from Tier 1 outputs into a deduplicated list.
    """
    genes: list[str] = []

    # Anchor genes from phenotype
    efo_id = disease_query.get("efo_id", "")
    if efo_id == "EFO_0001645":  # CAD
        genes.extend(["PCSK9", "LDLR", "HMGCR", "IL6R"])

    # CHIP genes from somatic result
    for edge in somatic_result.get("chip_edges", []):
        gene = edge.get("from_node", "").replace("_chip", "")
        if gene and gene not in genes:
            genes.append(gene)

    # Drug target genes
    for edge in somatic_result.get("drug_edges", []):
        gene = edge.get("from_node", "")
        if gene and gene not in genes:
            genes.append(gene)

    # Anchor gene validation failures → still include
    for gene, validated in genetics_result.get("anchor_genes_validated", {}).items():
        if gene not in genes:
            genes.append(gene)

    return genes if genes else ["PCSK9", "LDLR", "TET2", "DNMT3A"]
