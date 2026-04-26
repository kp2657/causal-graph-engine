"""
pi_orchestrator_v2.py — Static 5-tier causal genomics pipeline.

Implements the OTA formula: γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait})
as a direct function-call chain with no agent dispatch, SDK, or AgentRunner.
All tiers execute synchronously; state is passed as plain dicts between tiers.

Pipeline tiers
--------------
Tier 1  Phenomics   — disease ontology, GWAS instruments, genetic anchors
Tier 2  Pathway     — β estimation (eQTL-MR, Perturb-seq, pQTL, Tier 2L transfer)
Tier 3  Causal      — SCONE causal discovery, γ estimation, program decomposition
Tier 4  Translation — target prioritization, GPS phenotypic screening, chemistry
Tier 5  Writing     — structured JSON/Markdown report generation

Entry points (CLI)
------------------
    python -m orchestrator.pi_orchestrator_v2 run_tier4 "age-related macular degeneration"
    python -m orchestrator.pi_orchestrator_v2 run_tier4 "coronary artery disease"

    # Full pipeline (Tiers 1-5, ~6-8h per disease)
    python -m orchestrator.pi_orchestrator_v2 analyze_disease_v2 "<disease name>"

Validated diseases: AMD (RPE + Müller h5ad), CAD (smooth muscle cell h5ad).
Other diseases in DISEASE_CELLXGENE_MAP are supported but not regularly validated.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force line-buffered stdout so print() output appears immediately when piped
# (conda run and tee both trigger block-buffering otherwise).
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

import os

# ---------------------------------------------------------------------------
# Quality gate constants and helpers (inlined from v1)
# ---------------------------------------------------------------------------

VIRTUAL_TOP5_ESCALATE = True


def _escalate(message: str, context: dict) -> None:
    print(f"\n{'!'*60}\n[ESCALATION REQUIRED]\n{'!'*60}")
    print(message)
    print(f"Context: {context}")
    print(f"{'!'*60}\n")


def _run_quality_gate_tier1(disease_query: dict, genetics_result: dict) -> list[str]:
    issues: list[str] = []
    not_validated = [
        g for g, ok in genetics_result.get("anchor_genes_validated", {}).items() if not ok
    ]
    if not_validated:
        issues.append(f"WARNING: Anchor genes not validated by eQTL: {not_validated}")
    return issues


def _run_quality_gate_tier3(causal_result: dict) -> list[str]:
    issues: list[str] = []
    for w in causal_result.get("warnings", []):
        if "E-value" in w:
            issues.append(f"QUALITY: {w}")
    return issues


def _run_quality_gate_tier4(prioritization_result: dict) -> list[str]:
    issues: list[str] = []
    targets = prioritization_result.get("targets", [])
    if not targets:
        issues.append("ESCALATE: No targets returned from Tier 4 — check pipeline inputs")
        return issues
    top5_tiers = [t.get("evidence_tier") for t in targets[:5]]
    if VIRTUAL_TOP5_ESCALATE and all(t == "provisional_virtual" for t in top5_tiers):
        issues.append(
            "ESCALATE: All top-5 targets are provisional_virtual — "
            "pipeline output is hypothesis-generating only; no Tier1/2 evidence available."
        )
    for t in targets[:3]:
        if t.get("safety_flags"):
            issues.append(f"SAFETY: {t['target_gene']} (Rank {t['rank']}): {t['safety_flags']}")
    return issues


def _build_final_output(pipeline_outputs: dict) -> dict:
    graph_output  = pipeline_outputs.get("graph_output", {})
    disease_query = pipeline_outputs.get("phenotype_result", {})
    all_warnings  = pipeline_outputs.get("all_warnings", [])

    # Collect degradation warnings that affect output completeness.
    # These are surfaced to the user so they know when a data source was skipped.
    pipeline_warnings: list[str] = [
        w for w in all_warnings
        if any(kw in w for kw in (
            "SKIPPED", "not found", "download failed", "unavailable",
            "Docker", "h5ad", "GPS", "fallback", "timeout",
        ))
    ]

    # Summarise which data sources were actually used (vs. unavailable).
    beta_result   = pipeline_outputs.get("beta_result", {})
    chemistry     = pipeline_outputs.get("chemistry_result", {})
    data_completeness = {
        "perturb_seq_dataset":     disease_query.get("perturb_seq_dataset"),
        "h5ad_disease_sig_loaded": bool(pipeline_outputs.get("h5ad_disease_sig")),
        "gps_screen_run":          bool(chemistry.get("gps_disease_reversers")),
        "n_gps_reversers":         len(chemistry.get("gps_disease_reversers") or []),
    }

    return {
        **graph_output,
        "pi_reviewed":        True,
        "pipeline_version":   "0.2.0",
        "generated_at":       datetime.now(tz=timezone.utc).isoformat(),
        "disease_name":       disease_query.get("disease_name", ""),
        "efo_id":             disease_query.get("efo_id", ""),
        "pipeline_duration_s": pipeline_outputs.get("pipeline_duration_s"),
        "pipeline_status":    pipeline_outputs.get("pipeline_status", "UNKNOWN"),
        "total_edges_written": pipeline_outputs.get("total_edges_written"),
        "edge_write_breakdown": pipeline_outputs.get("edge_write_breakdown"),
        "n_escalations":      len([w for w in all_warnings if "ESCALATE" in w or "CRITICAL" in w]),
        "pipeline_warnings":  pipeline_warnings,
        "data_completeness":  data_completeness,
        # GPS compound screens — always present; empty list when GPS skipped
        "gps_disease_state_reversers": chemistry.get("gps_disease_reversers") or [],
        "gps_program_reversers":       chemistry.get("gps_program_reversers") or [],
        "gps_priority_compounds":      chemistry.get("gps_priority_compounds") or [],
    }


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S")


def _log(tag: str, agent: str, msg: str) -> None:
    print(f"[{tag:8s}] {agent:<30s} {msg}  ({_ts()})")


# ---------------------------------------------------------------------------
# Gamma estimates — identical to chief_of_staff._get_gamma_estimates
# ---------------------------------------------------------------------------

def _get_gamma_estimates(disease_query: dict) -> dict:
    """
    Compute program→trait γ estimates using DISEASE_TRAIT_MAP + ThreadPoolExecutor.

    Returns {program: {trait: gamma_dict}} where each gamma_dict includes
    gamma, gamma_se, evidence_tier, and data_source.  When efo_id is present
    and program gene sets are available, live aggregated L2G scores
    replace the hardcoded PROVISIONAL_GAMMAS table (estimate_gamma_live path).
    """
    from pipelines.ota_gamma_estimation import estimate_gamma
    from mcp_servers.burden_perturb_server import get_program_gene_loadings
    from pipelines.cnmf_programs import get_programs_for_disease
    from graph.schema import DISEASE_TRAIT_MAP, _DISEASE_SHORT_NAMES_FOR_ANCHORS

    disease_name = disease_query.get("disease_name", "coronary artery disease")
    efo_id       = disease_query.get("efo_id") or None
    short_name   = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_name.lower(), disease_name.upper())
    traits       = DISEASE_TRAIT_MAP.get(short_name, [short_name])

    # Use discovery-aware program source: cNMF on disk → MSigDB Hallmark → hardcoded fallback
    programs_info = get_programs_for_disease(short_name)
    raw_programs  = programs_info.get("programs", [])
    program_names = [
        p if isinstance(p, str) else (p.get("program_id") or p.get("name", ""))
        for p in raw_programs
    ]

    # Pre-fetch program gene sets once — needed for live OT γ estimation.
    # Priority: gene_set from programs_info (MSigDB = 50-200 genes, cNMF = 20-50 genes)
    # Fallback: get_program_gene_loadings (L1000 landmark genes only — 1-5 per program,
    # too sparse for OT enrichment queries).
    program_gene_sets: dict[str, set[str]] = {}
    for p in raw_programs:
        pid = p if isinstance(p, str) else (p.get("program_id") or p.get("name", ""))
        if isinstance(p, dict) and p.get("gene_set"):
            program_gene_sets[pid] = set(p["gene_set"])
        else:
            try:
                loadings_info = get_program_gene_loadings(pid)
                program_gene_sets[pid] = {
                    g if isinstance(g, str) else g.get("gene", "")
                    for g in loadings_info.get("top_genes", [])
                    if g
                }
            except Exception as _e:
                print(f"[WARN] gene_set lookup failed for program {pid!r}: {_e}")
                program_gene_sets[pid] = set()

    # Precomputed OT genetic scores from disease_query (populated by Tier 1 GWAS fetch).
    # Used as fallback when estimate_gamma returns None — covers diseases like AMD where
    # NMF program genes are RPE-specific and absent from OT L2G credible sets, but the
    # real AMD GWAS genes (CFH, ARMS2, C3...) do overlap with program gene sets.
    precomputed_ot_scores: dict[str, float] = disease_query.get("ot_genetic_scores") or {}

    # h5ad-based program gamma fallback: mean(log2FC of program genes in disease vs normal).
    # Applied when both OT L2G and precomputed OT score overlap fail. Covers cell-type-
    # specific programs (e.g. AMD RPE NMF programs) whose genes have no GWAS credible sets
    # but ARE differentially expressed in disease — providing a transcriptomic γ proxy.
    h5ad_disease_sig: dict[str, float] = {}
    try:
        from pipelines.gps_disease_screen import _build_sig_from_h5ad
        # elbow_trim=False: return all ~18k DEGs for program γ overlap.
        # GPS screen uses a separate trimmed call; here we need genome-wide
        # coverage so cNMF program genes (30 RPE-specific genes each) can match.
        _h5ad_sig = _build_sig_from_h5ad(disease_query, gps_genes=None, elbow_trim=False)
        if _h5ad_sig:
            h5ad_disease_sig = _h5ad_sig
    except Exception:
        pass

    work = [(prog, trait) for prog in program_names for trait in traits]

    # Load S-LDSC pre-computed τ coefficients (Mode 2 γ) — available after
    # running: python -m pipelines.ldsc.runner run {disease_key}
    _ldsc_gammas: dict[str, dict] = {}
    try:
        from pipelines.ldsc.gamma_loader import get_all_program_gammas_ldsc, ldsc_available
        if ldsc_available(short_name):
            _ldsc_gammas = get_all_program_gammas_ldsc(short_name)
            if _ldsc_gammas:
                print(f"[S-LDSC] Loaded τ coefficients for {len(_ldsc_gammas)} programs ({short_name})")
    except Exception as _ldsc_err:
        pass  # S-LDSC not yet run — falls back to OT L2G only

    def _fetch(prog_trait: tuple[str, str]) -> tuple[str, str, dict]:
        prog, trait = prog_trait
        try:
            # Mode 2: S-LDSC partitioned heritability (highest priority when available)
            # τ coefficient = per-SNP heritability enrichment from PLAtlas MVP+UKB+FinnGen
            if prog in _ldsc_gammas:
                ldsc_g = _ldsc_gammas[prog]
                ldsc_g.setdefault("gamma_source_type", "s_ldsc")
                ldsc_g["trait"] = trait
                return prog, trait, ldsc_g

            result = estimate_gamma(
                prog, trait,
                program_gene_set=program_gene_sets.get(prog) or None,
                efo_id=efo_id,
            )
            # estimate_gamma returns {"gamma": None, ...} for no_evidence — treat as
            # a miss so the OT/h5ad fallbacks below can fire.
            if result is not None and result.get("gamma") is not None:
                result.setdefault("gamma_source_type", "ot_l2g")
                return prog, trait, result

            prog_genes = program_gene_sets.get(prog) or set()

            # Fallback 1: precomputed GWAS OT genetic score overlap.
            matched_ot = {g: precomputed_ot_scores[g] for g in prog_genes if g in precomputed_ot_scores}
            if matched_ot:
                mean_score = sum(matched_ot.values()) / len(matched_ot)
                if mean_score >= 0.05:
                    gamma_val = round(mean_score * 0.65, 4)
                    return prog, trait, {
                        "gamma":            gamma_val,
                        "gamma_se":         round(gamma_val * 0.5, 4),
                        "evidence_tier":    "Tier3_Provisional",
                        "gamma_source_type": "gwas_ot_overlap",
                        "data_source":      f"OT_precomputed_overlap_{len(matched_ot)}_genes",
                        "program":          prog,
                        "trait":            trait,
                        "note":             (
                            f"Gamma from overlap of {len(matched_ot)} program genes with "
                            "precomputed GWAS OT genetic scores."
                        ),
                    }

            # Fallback 2: h5ad DEG-based gamma — mean log2FC of program genes in disease vs normal.
            # Positive mean → program upregulated in disease → risk program (γ > 0).
            # This covers cell-type-specific programs (e.g. AMD RPE NMF) whose genes are absent
            # from GWAS credible sets but are differentially expressed in disease.
            if h5ad_disease_sig:
                matched_h5ad = {g: h5ad_disease_sig[g] for g in prog_genes if g in h5ad_disease_sig}
                if len(matched_h5ad) >= 3:
                    mean_lfc = sum(matched_h5ad.values()) / len(matched_h5ad)
                    # Scale: 0.25 cap (down from 0.8).  h5ad DEG overlap fires for programs
                    # that are differentially expressed but not GWAS-supported — capping at 0.25
                    # prevents Perturb-seq essential-gene artefacts from dominating the OTA rank.
                    # Programs with real GWAS support already fire Fallback 1 (OT overlap, ~0.4–0.5)
                    # which is higher than this cap, so GWAS-anchored programs are unaffected.
                    gamma_val = round(min(abs(mean_lfc), 0.25), 4)
                    if gamma_val >= 0.01:  # filter near-zero noise (mean_lfc < 0.01)
                        # Sign: positive mean_lfc = risk (upregulated in disease)
                        signed_gamma = gamma_val if mean_lfc >= 0 else -gamma_val
                        return prog, trait, {
                            "gamma":             signed_gamma,
                            "gamma_se":          round(gamma_val * 0.4, 4),
                            "evidence_tier":     "Tier3_Provisional",
                            "gamma_source_type": "h5ad_deg",
                            "data_source":       f"h5ad_DEG_mean_lfc_{len(matched_h5ad)}_genes",
                            "program":           prog,
                            "trait":             trait,
                            "note":              (
                                f"Gamma from mean log2FC ({mean_lfc:+.3f}) of {len(matched_h5ad)} "
                                "program genes in disease vs normal h5ad (full DEG, no elbow trim). "
                                "Transcriptomic — not GWAS-derived."
                            ),
                        }
        except Exception:
            pass
        # Return None (unknown, not zero) so compute_ota_gamma skips this
        # program rather than contributing a phantom 0×γ product.
        return prog, trait, None

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
    genetics_result: dict,
) -> tuple[list[str], dict[str, float], dict]:
    """
    Returns (gwas_gene_list, ot_genetic_scores, ot_disease_targets_cache).

    Gene list is entirely data-derived: OT L2G GWAS associations + GWAS-validated genes
    from the statistical geneticist. CHIP / somatic excluded — GWAS/perturb-seq targets only.
    """
    genes: list[str] = []
    ot_scores: dict[str, float] = {}
    ot_cache: dict = {"targets": [], "source": None, "efo_id": disease_query.get("efo_id") or ""}

    # Add any GWAS-validated genes from the statistician (data-derived, not hardcoded)
    for gene in genetics_result.get("anchor_genes_validated", {}):
        if gene not in genes:
            genes.append(gene)

    # Seed trait-linked genes: either OT disease→target associations (genetic_score)
    # or, when OPEN_TARGETS_ASSOC_DISABLED=1, Open Targets L2G only (no associatedTargets GraphQL).
    efo_id = disease_query.get("efo_id") or ""
    _assoc_off = os.getenv("OPEN_TARGETS_ASSOC_DISABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    if efo_id and not _assoc_off:
        try:
            from mcp_servers.open_targets_server import get_open_targets_disease_targets
            ot_result = get_open_targets_disease_targets(
                efo_id, max_targets=500, min_overall_score=0.1
            )
            ot_cache = {"targets": ot_result.get("targets", []) or [], "source": "get_open_targets_disease_targets", "efo_id": efo_id}
            ot_added: list[str] = []
            for t in ot_result.get("targets", []):
                gene = t.get("gene_symbol", "")
                score = float(t.get("genetic_score", 0.0))
                if gene and score >= 0.1:
                    ot_scores[gene] = max(ot_scores.get(gene, 0.0), score)
                    if gene not in genes:
                        genes.append(gene)
                        ot_added.append(gene)
            if ot_added:
                _log("OT_SEED", "_collect_gene_list",
                     f"added {len(ot_added)} GWAS genes from OT Platform: {ot_added[:10]}{'...' if len(ot_added) > 10 else ''}")
        except Exception as _ot_exc:
            _log("OT_SEED_FAIL", "_collect_gene_list",
                 f"OT gene seeding failed: {_ot_exc} — gene list will be GWAS-validated hits only")
    elif efo_id and _assoc_off:
        try:
            from mcp_servers.gwas_genetics_server import get_l2g_prioritized_gene_list_for_efo
            l2g_res = get_l2g_prioritized_gene_list_for_efo(efo_id, max_genes=500)
            # Cache for downstream consumers that only need a gene list (no OT association payload).
            ot_cache = {"targets": l2g_res.get("genes") or [], "source": "get_l2g_prioritized_gene_list_for_efo", "efo_id": efo_id}
            ot_added: list[str] = []
            for row in l2g_res.get("genes") or []:
                gene = row.get("gene_symbol") or ""
                score = float(row.get("l2g_score") or 0.0)
                if gene and score >= 0.1:
                    ot_scores[gene] = max(ot_scores.get(gene, 0.0), score)
                    if gene not in genes:
                        genes.append(gene)
                        ot_added.append(gene)
            if ot_added:
                _log("L2G_SEED", "_collect_gene_list",
                     f"OPEN_TARGETS_ASSOC_DISABLED: seeded {len(ot_added)} genes from L2G: "
                     f"{ot_added[:10]}{'...' if len(ot_added) > 10 else ''}")
        except Exception as _l2g_exc:
            _log("L2G_SEED_FAIL", "_collect_gene_list",
                 f"L2G gene seeding failed: {_l2g_exc} — gene list will be GWAS-validated hits only")

    # Return empty list if no genes found — surface the failure rather than silently
    # injecting CAD-specific genes into an AMD/other-disease run.
    return genes, ot_scores, ot_cache


def _collect_perturbseq_genes(disease_query: dict) -> list[str]:
    """
    Return all gene symbols present in the Perturb-seq beta cache for this disease.

    Reads the existing cache file (JSON keys) without recomputing any betas.
    Falls back to empty list if no cache exists yet (first run before β computation).
    """
    dataset_id: str | None = None
    disease_name = (disease_query.get("disease_name") or "").lower()
    try:
        from graph.schema import DISEASE_CELL_TYPE_MAP, _DISEASE_SHORT_NAMES_FOR_ANCHORS
        short_key = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_name, "")
        if short_key and short_key in DISEASE_CELL_TYPE_MAP:
            dataset_id = DISEASE_CELL_TYPE_MAP[short_key].get("scperturb_dataset")
    except Exception:
        pass

    if not dataset_id:
        return []

    try:
        from pipelines.replogle_parser import _PERTURBSEQ_DIR
        cache_dir = _PERTURBSEQ_DIR / dataset_id
        # Any beta cache for this dataset works — we only need the gene keys
        cache_files = list(cache_dir.glob("beta_cache_*.json")) if cache_dir.exists() else []
        if not cache_files:
            _log("PERTURB_SEED", "_collect_perturbseq_genes",
                 f"No beta cache found for {dataset_id} — Perturb-seq genes will be added after first β computation")
            return []
        cache_path = max(cache_files, key=lambda p: p.stat().st_size)
        with open(cache_path) as f:
            data = json.load(f)
        genes = list(data.keys())
        _log("PERTURB_SEED", "_collect_perturbseq_genes",
             f"{len(genes)} Perturb-seq KO genes loaded from {cache_path.name}")
        return genes
    except Exception as exc:
        _log("PERTURB_SEED_FAIL", "_collect_perturbseq_genes", str(exc))
        return []


# ---------------------------------------------------------------------------
# Tier 4 feedback helper
# ---------------------------------------------------------------------------

def _build_tier4_context(
    disease_query: dict,
    top_genes: list[dict],
) -> dict:
    """
    Build a single Tier 4 context snapshot once per run.

    Goal: centralize metadata fetches so Tier 4 agents can be lightweight and avoid
    duplicated calls. This context is attached to disease_query as `_tier4_context`.
    """
    minimal = os.getenv("MINIMAL_TIER4", "").strip().lower() in {"1", "true", "yes", "on"}
    genes = [r.get("gene") for r in top_genes if r.get("gene")]
    pli_map: dict[str, float] = {}
    try:
        from mcp_servers.gwas_genetics_server import query_gnomad_lof_constraint
        constraint = query_gnomad_lof_constraint(genes)
        for item in constraint.get("genes", []) or []:
            g = item.get("symbol", "")
            pli = item.get("pLI") or item.get("pli")
            if g and pli is not None:
                pli_map[g] = float(pli)
    except Exception:
        pass

    return {
        "minimal": minimal,
        # OT disease-target payload is cached earlier during gene seeding:
        "ot_disease_targets_cache": disease_query.get("_ot_disease_targets_cache") or {},
        "pli_map": pli_map,
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def analyze_disease_v2(
    disease_name: str,
    _ckpt_dir: "Path | None" = None,
) -> dict[str, Any]:
    """
    Run the full 5-tier OTA causal pipeline for a disease.

    Args:
        disease_name: Human-readable disease name (e.g. "age-related macular degeneration").
                      Canonical names are in models/disease_registry.py.
        _ckpt_dir: Override checkpoint directory (tests only).  Production code leaves
                   this as None so checkpoints go to data/checkpoints/.

    Returns:
        Dict with keys: disease_name, target_list, genetic_anchors, gps_disease_state_reversers,
        gps_program_reversers, pipeline_version, pipeline_warnings, data_completeness, and more.
        See docs/OUTPUT_SCHEMA.md for the full field reference.

    Side effects:
        Writes JSON to data/analyze_{disease_slug}.json and Markdown to data/analyze_{disease_slug}.md.
        Saves tier checkpoints to data/checkpoints/ for use with run_tier4_from_checkpoint.
    """
    run_id         = datetime.now(tz=timezone.utc).isoformat()
    pipeline_start = time.time()
    all_warnings:   list[str] = []
    pipeline_outputs: dict[str, Any] = {}

    print(f"\n{'='*60}")
    print(f"PI ORCHESTRATOR v2: {disease_name.upper()}")
    print(f"Started: {run_id}")
    print(f"{'='*60}")

    # =========================================================================
    # TIER 1 — Phenomics
    # =========================================================================
    print(f"\n{'='*60}\nTIER 1 — Phenomics: {disease_name}\n{'='*60}")

    from agents.tier1_phenomics.phenotype_architect import run as _pa_run
    from agents.tier1_phenomics.statistical_geneticist import run as _sg_run

    t0 = time.time()
    # disease_query["disease_key"] (e.g. "CAD", "AMD") is set by phenotype_architect
    # and propagated as-is through all tiers. Agents must read it from disease_query,
    # not re-derive it via get_disease_key() — single source of truth.
    disease_query = _pa_run(disease_name)
    _log("COMPLETE", "phenotype_architect",
         f"efo_id={disease_query.get('efo_id')}  {time.time()-t0:.1f}s")

    if disease_query.get("stub_fallback"):
        if any(x in disease_name.lower() for x in ("glaucoma", "poag")):
            disease_query["efo_id"] = "EFO_0004190"
            disease_query["disease_name"] = "primary open-angle glaucoma"
            _log("FIXUP", "orchestrator", "Applied manual EFO_0004190 fallback for Glaucoma")
        else:
            pipeline_outputs.update({
                "pipeline_status": "FAILED_TIER1_PHENOTYPE",
                "phenotype_result": disease_query,
                "all_warnings": all_warnings,
            })
            return _build_final_output(pipeline_outputs)

    t0 = time.time()
    genetics_result = _sg_run(disease_query)
    _log("COMPLETE", "statistical_geneticist",
         f"n_instruments={len(genetics_result.get('instruments', []))}  {time.time()-t0:.1f}s")

    pipeline_outputs["phenotype_result"] = disease_query
    pipeline_outputs["genetics_result"]  = genetics_result
    all_warnings.extend(genetics_result.get("warnings", []))

    # Tier 1 quality gate
    tier1_issues = _run_quality_gate_tier1(disease_query, genetics_result)
    all_warnings.extend(tier1_issues)
    for issue in tier1_issues:
        if "ESCALATE" in issue:
            _escalate(issue, {"disease": disease_name})

    # Gene list: OT L2G + GWAS-validated hits
    gene_list, gwas_ot_scores, ot_disease_targets_cache = _collect_gene_list(disease_query, genetics_result)
    gwas_gene_list = list(gene_list)

    # Union all Perturb-seq KO genes — GWAS genes retain priority; non-GWAS
    # Perturb-seq genes are appended so OTA scores them via program-level γ.
    perturb_genes = _collect_perturbseq_genes(disease_query)
    perturb_only_genes = [g for g in perturb_genes if g not in gwas_gene_list]
    gene_list = gwas_gene_list + perturb_only_genes
    if perturb_only_genes:
        _log("PERTURB_UNION", "gene_list",
             f"added {len(perturb_only_genes)} Perturb-seq-only genes "
             f"(total {len(gene_list)}: {len(gwas_gene_list)} GWAS + {len(perturb_only_genes)} novel)")

    _disease_key_for_pareto = disease_query.get("disease_key") or ""

    disease_query = {
        **disease_query,
        "gwas_genes": gwas_gene_list,          # unchanged — used for γ grounding & anchor QC
        "perturb_only_genes": perturb_only_genes,  # novel candidates with no GWAS prior
        "ot_genetic_scores": gwas_ot_scores,
        "_ot_disease_targets_cache": ot_disease_targets_cache,
    }

    # Hard gate: require h5ad before running any β estimation.
    # Without h5ad, Perturb-seq β falls back to provisional_virtual which has no causal
    # basis and produces misleading output.  Fail loudly so the user downloads the file.
    import pathlib as _pl
    from graph.schema import _DISEASE_SHORT_NAMES_FOR_ANCHORS as _DSN2
    _short2 = _DSN2.get(disease_query.get("disease_name", "").lower(), "")
    if _short2:
        _h5ad_dir = _pl.Path(f"data/cellxgene/{_short2}")
        _h5ad_candidates = sorted(
            p for p in _h5ad_dir.glob(f"{_short2}_*.h5ad")
            if "latent_cache" not in p.name and "state_cache" not in p.name
        ) if _h5ad_dir.exists() else []
        if not _h5ad_candidates:
            raise RuntimeError(
                f"\n\nMissing h5ad for {_short2.upper()}.\n"
                f"Run:\n\n"
                f"  python -m pipelines.discovery.cellxgene_downloader download_all {_short2}\n\n"
                f"Expected location: data/cellxgene/{_short2}/{_short2}_<cell_type>.h5ad\n"
                f"Without this file the pipeline cannot compute causal β estimates."
            )

    # Pre-run NMF on disease h5ad so {disease}_programs.json exists on disk before
    # both Tier 2 (beta matrix) and gamma estimation call get_programs_for_disease.
    # Must run BEFORE Tier 2 to guarantee consistent program names across both.
    try:
        from pipelines.discovery.cnmf_runner import run_nmf_programs as _run_nmf
        if _short2 and _h5ad_candidates:
            _run_nmf(str(_h5ad_candidates[0]), _short2, n_programs=20, n_top_genes=50)
    except Exception as _nmf_pre_exc:
        pass  # NMF failure is non-fatal — programs fall back to MSigDB Hallmarks

    # =========================================================================
    # TIER 2 — Pathway
    # =========================================================================
    print(f"\n{'='*60}\nTIER 2 — Pathway: {len(gene_list)} genes\n{'='*60}")

    from agents.tier2_pathway.perturbation_genomics_agent import run as _pga_run
    from agents.tier2_pathway.regulatory_genomics_agent import run as _rga_run

    t0 = time.time()
    # Independent: run Tier 2 agents in parallel (no shared mutable state).
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_beta = pool.submit(_pga_run, gene_list, disease_query)
        fut_reg  = pool.submit(_rga_run, gene_list, disease_query)
        beta_result       = fut_beta.result()
        regulatory_result = fut_reg.result()
    _log("COMPLETE", "perturbation_genomics_agent",
         f"tier1={beta_result.get('n_tier1',0)}, virtual={beta_result.get('n_virtual',0)}  "
         f"{time.time()-t0:.1f}s")
    _log("COMPLETE", "regulatory_genomics_agent",
         f"tier2_upgrades={len(regulatory_result.get('tier2_upgrades', []))}")

    pipeline_outputs["beta_matrix_result"] = beta_result
    pipeline_outputs["regulatory_result"]  = regulatory_result
    all_warnings.extend(beta_result.get("warnings", []))
    all_warnings.extend(regulatory_result.get("warnings", []))

    # Forward Tier 2 program-membership map into disease_query so Tier 3 can
    # reconstruct `top_programs` for genes that reach gamma via the OT genetic
    # fallback (n_programs_contributing == 0).  Without this, classify_program_drivers
    # returns the "no_program_data" sentinel for every GWAS-only target.
    disease_query["gene_program_overlap"] = regulatory_result.get("gene_program_overlap", {})

    # Gamma estimates — live OT + S-LDSC per program
    gamma_estimates = _get_gamma_estimates(disease_query)
    pipeline_outputs["_gamma_estimates"] = gamma_estimates

    # Probe h5ad disease sig so data_completeness.h5ad_disease_sig_loaded is accurate.
    # _build_sig_from_h5ad is cached; this second call is cheap when the file is on disk.
    try:
        from pipelines.gps_disease_screen import _build_sig_from_h5ad as _probe_h5ad
        _h5ad_probe = _probe_h5ad(disease_query, gps_genes=None)
        if _h5ad_probe:
            pipeline_outputs["h5ad_disease_sig"] = _h5ad_probe
    except Exception:
        pass

    # =========================================================================
    # TIER 3 — Causal Discovery
    # =========================================================================
    print(f"\n{'='*60}\nTIER 3 — Causal Discovery\n{'='*60}")

    from agents.tier3_causal.causal_discovery_agent import run as _cda_run
    from agents.tier3_causal.kg_completion_agent import run as _kgc_run

    t0 = time.time()
    causal_result = _cda_run(beta_result, gamma_estimates, disease_query)
    _log("COMPLETE", "causal_discovery_agent",
         f"n_written={causal_result.get('n_edges_written',0)}  {time.time()-t0:.1f}s")

    # Write program→trait DrivesTrait edges (program nodes + γ edges)
    try:
        from mcp_servers.graph_db_server import write_program_gamma_edges as _wpge
        from graph.schema import DISEASE_CELL_TYPE_MAP, _DISEASE_SHORT_NAMES_FOR_ANCHORS as _DSN
        _dkey = _DSN.get(disease_name.lower(), disease_name.upper())
        _cell_type = (DISEASE_CELL_TYPE_MAP.get(_dkey, {}).get("cell_types") or ["unknown"])[0]
        _pg_result = _wpge(
            gamma_estimates,
            disease=disease_name,
            efo_id=disease_query.get("efo_id"),
            cell_type=_cell_type,
        )
        _log("COMPLETE", "program_gamma_edges",
             f"written={_pg_result['written']} rejected={_pg_result['rejected']}")
    except Exception as _exc_pg:
        all_warnings.append(f"program gamma edge write failed (non-fatal): {_exc_pg}")

    t0 = time.time()
    kg_result = _kgc_run(causal_result, disease_query)
    _log("COMPLETE", "kg_completion_agent",
         f"pathways={kg_result.get('n_pathway_edges_added',0)}  {time.time()-t0:.1f}s")

    pipeline_outputs["causal_result"] = causal_result
    pipeline_outputs["kg_result"]     = kg_result
    all_warnings.extend(causal_result.get("warnings", []))
    all_warnings.extend(kg_result.get("warnings", []))

    # Tier 3 quality gate
    tier3_issues = _run_quality_gate_tier3(causal_result)
    all_warnings.extend(tier3_issues)

    # Scientific reviewer — plain function call on written edges
    from orchestrator.scientific_reviewer import review_batch
    causal_edges = causal_result.get("edges_written", []) or []
    if causal_edges:
        review_result = review_batch(causal_edges)
        pipeline_outputs["review_result"] = review_result
        n_rej = review_result.get("n_rejected", 0)
        if n_rej:
            all_warnings.append(f"Scientific reviewer rejected {n_rej}/{len(causal_edges)} edges")

    # Tier 3 checkpoint — saves all inputs needed to re-run Tier 4 independently.
    try:
        ckpt_dir = _ckpt_dir if _ckpt_dir is not None else Path(__file__).parent.parent / "data" / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        _disease_slug_t3 = disease_name.lower().replace(" ", "_").replace("-", "_")
        ckpt3_path = ckpt_dir / f"{_disease_slug_t3}__tier3.json"
        with ckpt3_path.open("w") as _cf3:
            json.dump({
                "run_id":            run_id,
                "disease_name":      disease_name,
                "disease_query":     disease_query,
                "genetics_result":   genetics_result,
                "beta_result":       beta_result,
                "regulatory_result": regulatory_result,
                "gamma_estimates":   gamma_estimates,
                "causal_result":     causal_result,
                "kg_result":         kg_result,
            }, _cf3, indent=2, default=str)
        _log("CHECKPOINT", "tier3", f"saved → {ckpt3_path}")
    except Exception as _ckpt3_exc:
        _log("CHECKPOINT", "tier3", f"save failed: {_ckpt3_exc}")

    # =========================================================================
    # TIER 4 — Translation
    # =========================================================================
    print(f"\n{'='*60}\nTIER 4 — Translation\n{'='*60}")

    from agents.tier4_translation.target_prioritization_agent import run as _tpa_run
    from agents.tier4_translation.chemistry_agent import run as _chem_run
    from agents.tier4_translation.clinical_trialist_agent import run as _ct_run

    t0 = time.time()
    # Build Tier 4 context once; passed via disease_query for agent reuse.
    disease_query = {
        **disease_query,
        "_tier4_context": _build_tier4_context(disease_query, causal_result.get("top_genes", []) or []),
    }

    prioritization_result = _tpa_run(causal_result, kg_result, disease_query)
    # Chemistry / GPS disease-program screens need program→trait γ; inject from orchestrator.
    prioritization_result["_gamma_estimates"] = gamma_estimates
    _log("COMPLETE", "target_prioritization_agent",
         f"n_targets={len(prioritization_result.get('targets',[]))}  {time.time()-t0:.1f}s")

    # Gate ChEMBL annotation to post-Tier-3 gene whitelist (Phase G).
    _post_filter_genes = {r["gene"] for r in causal_result.get("top_genes", [])}
    if _post_filter_genes:
        disease_query = {**disease_query, "_gps_target_whitelist": _post_filter_genes}

    # Independent: run chemistry + clinical trialist in parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_chem = pool.submit(_chem_run, prioritization_result, disease_query)
        fut_ct   = pool.submit(_ct_run, prioritization_result, disease_query)
        chemistry_result = fut_chem.result()
        clinical_result  = fut_ct.result()

    # GPS × OTA convergence — cross-reference reversers against ranked targets.
    try:
        from pipelines.gps_convergence import compute_gps_convergence, annotate_targets_with_gps
        _gps_for_conv = {
            "disease_reversers": chemistry_result.get("gps_disease_reversers") or [],
            "program_reversers": chemistry_result.get("gps_program_reversers") or {},
        }
        _targets_for_conv = prioritization_result.get("targets") or []
        if _targets_for_conv and (_gps_for_conv["disease_reversers"] or _gps_for_conv["program_reversers"]):
            _conv = compute_gps_convergence(_targets_for_conv, _gps_for_conv, top_n=200)
            annotate_targets_with_gps(_targets_for_conv, _conv)
            chemistry_result["gps_convergence"] = _conv
            _log("GPS_CONV", "convergence",
                 f"converged={_conv['n_converged']} novel_mechanism={_conv['n_novel_mechanism']}")
    except Exception as _conv_exc:
        all_warnings.append(f"GPS convergence failed (non-fatal): {_conv_exc}")

    # No Tier4 feedback loop: ranking is owned by target_prioritization_agent (OTA-first).
    pipeline_outputs["prioritization_result"] = prioritization_result
    pipeline_outputs["chemistry_result"]      = chemistry_result
    pipeline_outputs["trials_result"]         = clinical_result
    all_warnings.extend(prioritization_result.get("warnings", []))
    all_warnings.extend(chemistry_result.get("warnings", []))
    all_warnings.extend(clinical_result.get("warnings", []))

    # Checkpoint — save Tier 4 output so writer can be re-run independently
    _disease_slug = disease_name.lower().replace(" ", "_").replace("-", "_")
    try:
        ckpt_dir = _ckpt_dir if _ckpt_dir is not None else Path(__file__).parent.parent / "data" / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f"{_disease_slug}__tier4.json"
        with ckpt_path.open("w") as _cf:
            json.dump({
                "run_id":                run_id,
                "disease_name":          disease_name,
                "prioritization_result": prioritization_result,
                "chemistry_result":      chemistry_result,
                "trials_result":         clinical_result,
                "phenotype_result":      disease_query,
            }, _cf, indent=2, default=str)
        _log("CHECKPOINT", "tier4", f"saved → {ckpt_path}")
    except Exception as _ckpt_exc:
        _log("CHECKPOINT", "tier4", f"save failed: {_ckpt_exc}")

    # Tier 4 quality gate
    tier4_issues = _run_quality_gate_tier4(prioritization_result)
    all_warnings.extend(tier4_issues)
    for issue in tier4_issues:
        if "ESCALATE" in issue:
            _escalate(issue, {"disease": disease_name})

    # =========================================================================
    # TIER 5 — Output assembly (collapsed, no agent dispatch)
    # =========================================================================
    print(f"\n{'='*60}\nTIER 5 — Output Assembly\n{'='*60}")

    from agents.tier5_writer.scientific_writer_agent import run as _writer_run

    graph_output = _writer_run(
        phenotype_result      = disease_query,
        genetics_result       = genetics_result,
        beta_matrix_result    = beta_result,
        regulatory_result     = regulatory_result,
        causal_result         = causal_result,
        kg_result             = kg_result,
        prioritization_result = prioritization_result,
        chemistry_result      = chemistry_result,
        trials_result         = clinical_result,
    )
    pipeline_outputs["graph_output"] = graph_output
    all_warnings.extend(graph_output.get("warnings", []))


    # =========================================================================
    # Finalise
    # =========================================================================
    _n_causal = int(causal_result.get("n_edges_written", 0) or 0)
    _n_path   = int(kg_result.get("n_pathway_edges_added", 0) or 0)
    total_edges = _n_causal + _n_path
    total_duration = time.time() - pipeline_start
    pipeline_outputs.update({
        "pipeline_status":     "SUCCESS",
        "pipeline_duration_s": round(total_duration, 1),
        "all_warnings":        all_warnings,
        "total_edges_written": total_edges,
        "edge_write_breakdown": {
            "tier3_causal_graph_writes":  _n_causal,
            "tier3_kg_reactome_pathways": _n_path,
        },
    })

    benchmark = prioritization_result.get("benchmark")
    final = _build_final_output(pipeline_outputs)
    if benchmark is not None:
        final["benchmark"] = benchmark

    print(f"\n[PI v2 COMPLETE]")
    print(f"  Disease:         {disease_name}")
    print(f"  Targets ranked:  {len(final.get('target_list', []))}")
    print(f"  Escalations:     {final.get('n_escalations', 0)}")
    print(f"  Edges written:   {total_edges}  (Tier3 causal graph: {_n_causal}; Reactome pathways: {_n_path})")
    print(f"  Duration:        {total_duration:.1f}s")
    print(f"  Status:          {final.get('pipeline_status')}")

    return final


def _write_pipeline_output(result: dict, data_dir: Path) -> None:
    """Write pipeline result JSON to data/ directory."""
    disease_slug = (
        result.get("disease_name", "unknown")
        .lower().replace(" ", "_").replace("-", "_")
    )
    out_path = data_dir / f"analyze_{disease_slug}.json"
    with out_path.open("w") as _f:
        json.dump(result, _f, indent=2, default=str)
    print(f"  Output written:  {out_path}")


def run_tier4_from_checkpoint(disease_name: str) -> dict[str, Any]:
    """
    Re-run Tier 4 (+ Tier 5) from a saved Tier 3 checkpoint.

    Use this after updating scoring logic, GPS Docker, or compound library without
    re-running the expensive Tiers 1–3 (GWAS, eQTL-MR, Perturb-seq, SCONE).

    Args:
        disease_name: Same string used in the original analyze_disease_v2 call.

    Returns:
        Same structure as analyze_disease_v2. Nested under prioritization_result
        when called via CLI; flat when imported and called directly.

    Raises:
        FileNotFoundError: If no tier3 or tier4 checkpoint exists for the disease.

    Usage:
      python -m orchestrator.pi_orchestrator_v2 run_tier4 "coronary artery disease"
    """
    import time as _time
    _disease_slug = disease_name.lower().replace(" ", "_").replace("-", "_")
    ckpt_dir = Path(__file__).parent.parent / "data" / "checkpoints"
    ckpt3_path = ckpt_dir / f"{_disease_slug}__tier3.json"
    ckpt4_path = ckpt_dir / f"{_disease_slug}__tier4.json"

    from agents.tier4_translation.target_prioritization_agent import run as _tpa_run
    from agents.tier4_translation.chemistry_agent import run as _chem_run
    from agents.tier4_translation.clinical_trialist_agent import run as _ct_run
    from agents.tier5_writer.scientific_writer_agent import run as _writer_run

    if ckpt3_path.exists():
        # Full Tier 4 re-run: re-ranks targets + re-runs GPS + writes report
        with ckpt3_path.open() as _f:
            ckpt = json.load(_f)
        print(f"\nLoaded Tier 3 checkpoint: {ckpt3_path}")
        print(f"  Disease:  {ckpt.get('disease_name')}  |  Run ID: {ckpt.get('run_id')}")

        disease_query     = ckpt["disease_query"]
        genetics_result   = ckpt.get("genetics_result", {})
        beta_result       = ckpt.get("beta_result", {})
        regulatory_result = ckpt.get("regulatory_result", {})
        gamma_estimates   = ckpt.get("gamma_estimates", {})
        causal_result     = ckpt["causal_result"]
        kg_result         = ckpt["kg_result"]

        print(f"\n{'='*60}\nTIER 4 — Translation (from tier3 checkpoint)\n{'='*60}")
        t0 = _time.time()
        disease_query = {
            **disease_query,
            "_tier4_context": _build_tier4_context(disease_query, causal_result.get("top_genes", []) or []),
        }
        prioritization_result = _tpa_run(causal_result, kg_result, disease_query)
        prioritization_result["_gamma_estimates"] = gamma_estimates
        _log("COMPLETE", "target_prioritization_agent",
             f"n_targets={len(prioritization_result.get('targets',[]))}  {_time.time()-t0:.1f}s")

    elif ckpt4_path.exists():
        # GPS-only re-run: keep existing targets, re-run chemistry + trials, write report
        with ckpt4_path.open() as _f:
            ckpt = json.load(_f)
        print(f"\nNo tier3 checkpoint found — using tier4 checkpoint: {ckpt4_path}")
        print(f"  Disease:  {ckpt.get('disease_name')}  |  Run ID: {ckpt.get('run_id')}")
        print(f"  Mode:     chemistry-only re-run (target ranking unchanged)")

        disease_query         = ckpt.get("phenotype_result", {})
        prioritization_result = ckpt["prioritization_result"]
        # Stub upstream results — writer uses these for metadata only
        genetics_result       = {}
        beta_result           = {"n_virtual": 0}
        regulatory_result     = {}
        gamma_estimates       = {}
        causal_result         = {
            "top_genes": prioritization_result.get("targets", []),
            "n_edges_written": 0,
            "edges_written": [],
        }
        kg_result             = {}

        print(f"\n{'='*60}\nTIER 4 — Chemistry only (from tier4 checkpoint)\n{'='*60}")
        _log("SKIP", "target_prioritization_agent",
             f"n_targets={len(prioritization_result.get('targets',[]))} (from checkpoint)")

    else:
        raise FileNotFoundError(
            f"No checkpoint found for '{disease_name}'.\n"
            f"  Expected (tier3): {ckpt3_path}\n"
            f"  Expected (tier4): {ckpt4_path}\n"
            f"Run the full pipeline first: analyze_disease_v2 \"{disease_name}\""
        )

    # Gate ChEMBL annotation to post-Tier-3 gene whitelist (Phase G).
    _post_filter_genes_t4 = {r["gene"] for r in causal_result.get("top_genes", [])}
    if _post_filter_genes_t4:
        disease_query = {**disease_query, "_gps_target_whitelist": _post_filter_genes_t4}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_chem = pool.submit(_chem_run, prioritization_result, disease_query)
        fut_ct   = pool.submit(_ct_run, prioritization_result, disease_query)
        chemistry_result = fut_chem.result()
        clinical_result  = fut_ct.result()

    # GPS × OTA convergence
    try:
        from pipelines.gps_convergence import compute_gps_convergence, annotate_targets_with_gps
        _gps_for_conv = {
            "disease_reversers": chemistry_result.get("gps_disease_reversers") or [],
            "program_reversers": chemistry_result.get("gps_program_reversers") or {},
        }
        _targets_for_conv = prioritization_result.get("targets") or []
        if _targets_for_conv and (_gps_for_conv["disease_reversers"] or _gps_for_conv["program_reversers"]):
            _conv = compute_gps_convergence(_targets_for_conv, _gps_for_conv, top_n=200)
            annotate_targets_with_gps(_targets_for_conv, _conv)
            chemistry_result["gps_convergence"] = _conv
            _log("GPS_CONV", "convergence",
                 f"converged={_conv['n_converged']} novel_mechanism={_conv['n_novel_mechanism']}")
    except Exception as _conv_exc:
        all_warnings.append(f"GPS convergence (tier4 path) failed (non-fatal): {_conv_exc}")

    # Save updated tier4 checkpoint
    try:
        ckpt4_path = ckpt_dir / f"{_disease_slug}__tier4.json"
        with ckpt4_path.open("w") as _cf4:
            json.dump({
                "run_id":                ckpt.get("run_id"),
                "disease_name":          disease_name,
                "prioritization_result": prioritization_result,
                "chemistry_result":      chemistry_result,
                "trials_result":         clinical_result,
                "phenotype_result":      disease_query,
            }, _cf4, indent=2, default=str)
        _log("CHECKPOINT", "tier4", f"saved → {ckpt4_path}")
    except Exception as _e:
        _log("CHECKPOINT", "tier4", f"save failed: {_e}")

    print(f"\n{'='*60}\nTIER 5 — Output Assembly (from checkpoint)\n{'='*60}")

    graph_output = _writer_run(
        phenotype_result      = disease_query,
        genetics_result       = genetics_result,
        beta_matrix_result    = beta_result,
        regulatory_result     = regulatory_result,
        causal_result         = causal_result,
        kg_result             = kg_result,
        prioritization_result = prioritization_result,
        chemistry_result      = chemistry_result,
        trials_result         = clinical_result,
    )

    targets = prioritization_result.get("targets", [])
    print(f"\n[COMPLETE] Tier 4+5 from checkpoint")
    print(f"  Targets ranked:  {len(targets)}")
    print(f"  GPS disease reversers:   {len(chemistry_result.get('gps_disease_reversers', []))}")
    print(f"  GPS program reversers:   {sum(len(v) for v in chemistry_result.get('gps_program_reversers', {}).values())} across {len(chemistry_result.get('gps_program_reversers', {}))} programs")
    print(f"  GPS priority compounds:  {len(chemistry_result.get('gps_priority_compounds', []))}")

    result = {
        "disease_name":          disease_name,
        "genetics_result":       genetics_result,
        "beta_matrix_result":    beta_result,
        "regulatory_result":     regulatory_result,
        "causal_result":         causal_result,
        "kg_result":             kg_result,
        "prioritization_result": prioritization_result,
        "chemistry_result":      chemistry_result,
        "trials_result":         clinical_result,
        "graph_output":          graph_output,
        "pipeline_status":       "SUCCESS",
    }
    return result


if __name__ == "__main__":
    import argparse as _argparse
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    _parser = _argparse.ArgumentParser(
        description="PI Orchestrator v2 — causal genomics pipeline"
    )
    _sub = _parser.add_subparsers(dest="command")
    _p = _sub.add_parser("analyze_disease_v2", help="Run full pipeline for a disease")
    _p.add_argument("disease_name", help='Disease name, e.g. "inflammatory bowel disease"')
    _p4 = _sub.add_parser(
        "run_tier4",
        help="Re-run Tier 4+5 from saved Tier 3 checkpoint (skips Tiers 1-3)",
    )
    _p4.add_argument("disease_name", help='Disease name matching existing checkpoint')
    _args = _parser.parse_args()
    if _args.command == "analyze_disease_v2":
        _result = analyze_disease_v2(_args.disease_name)
        _data_dir = Path(__file__).parent.parent / "data"
        _write_pipeline_output(_result, _data_dir)
    elif _args.command == "run_tier4":
        _result = run_tier4_from_checkpoint(_args.disease_name)
        _data_dir = Path(__file__).parent.parent / "data"
        _write_pipeline_output(_result, _data_dir)
    else:
        _parser.print_help()
