"""
pi_orchestrator_v2.py — Static 5-tier causal genomics pipeline.

Implements the OTA formula: γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait})
as a direct function-call chain with no agent dispatch, SDK, or AgentRunner.
All tiers execute synchronously; state is passed as plain dicts between tiers.

Pipeline tiers
--------------
Tier 1  Phenomics   — disease ontology, GWAS instruments, genetic anchors
Tier 2  Pathway     — β estimation (eQTL-MR, Perturb-seq, pQTL, Tier 2L transfer)
Tier 3  Causal      — OTA γ estimation, program decomposition, causal graph construction
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


def _strip_gps_annotation_bloat(gps_program_reversers: dict) -> dict:
    """Strip convergent_genetic_targets_hypothesis from compound annotations before output serialization."""
    stripped = {}
    for prog, compounds in gps_program_reversers.items():
        if isinstance(compounds, list):
            clean = []
            for c in compounds:
                ann = c.get("annotation") or {}
                safe_ann = {k: v for k, v in ann.items() if k != "convergent_genetic_targets_hypothesis"}
                clean.append({**{k: v for k, v in c.items() if k != "annotation"}, "annotation": safe_ann})
            stripped[prog] = clean
        else:
            stripped[prog] = compounds
    return stripped


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
        # Strip convergent_genetic_targets_hypothesis from annotations (O(N_targets) per compound → bloat).
        "gps_disease_state_reversers": chemistry.get("gps_disease_reversers") or [],
        "gps_program_reversers":       _strip_gps_annotation_bloat(chemistry.get("gps_program_reversers") or {}),
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
    from pipelines.cnmf_programs import get_programs_for_disease, get_svd_programs_for_disease
    from graph.schema import DISEASE_TRAIT_MAP, _DISEASE_SHORT_NAMES_FOR_ANCHORS, DISEASE_CELL_TYPE_MAP

    disease_name = disease_query.get("disease_name", "coronary artery disease")
    efo_id       = disease_query.get("efo_id") or None
    short_name   = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_name.lower(), disease_name.upper())
    traits       = DISEASE_TRAIT_MAP.get(short_name, [short_name])

    # Program source: must match what beta_matrix_builder selects (same program IDs).
    # Priority 0: signatures-backed datasets (Schnitzler/CZI) → cNMF paper programs
    #   (P01-P60). SVD programs require h5ad which these datasets lack.
    # Priority 1: SVD components for h5ad-backed datasets.
    # Priority 2: NMF from disk / MSigDB.
    gwas_genes_for_gamma = set(disease_query.get("gwas_genes", []))
    raw_programs: list = []
    programs_info: dict = {}

    try:
        from pipelines.perturbseq_beta_loader import _DATASET_SIGNATURES_REGISTRY as _SIG_REG
        _gctx = DISEASE_CELL_TYPE_MAP.get(short_name, {})
        _gds = _gctx.get("scperturb_dataset")
        if _gds and _gds in _SIG_REG:
            import json as _gjson
            _gp = _SIG_REG[_gds].parent / "cnmf_program_gene_sets.json"
            if _gp.exists():
                _gs = _gjson.loads(_gp.read_text())
                raw_programs = [{"program_id": pid, "gene_set": genes} for pid, genes in _gs.items()]
                programs_info = {"programs": raw_programs, "source": f"cNMF_paper_programs_{_gds}"}
    except Exception:
        pass

    if not raw_programs:
        svd_info = get_svd_programs_for_disease(short_name, gwas_genes=gwas_genes_for_gamma or None)
        if svd_info.get("programs"):
            programs_info = svd_info
        else:
            programs_info = get_programs_for_disease(short_name)
        raw_programs = programs_info.get("programs", [])
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

    work = [(prog, trait) for prog in program_names for trait in traits]

    def _fetch(prog_trait: tuple[str, str]) -> tuple[str, str, dict]:
        prog, trait = prog_trait
        try:
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

    The full result is frozen to data/checkpoints/{disease_key}_gene_list.json on first run
    and reloaded on subsequent runs so results are stable across OT Platform releases.
    Force a refresh by deleting that file.
    """
    import pathlib as _pl, json as _json
    _dkey = (disease_query.get("disease_key") or "unknown").lower()
    _freeze_path = _pl.Path(f"data/checkpoints/{_dkey}_gene_list.json")

    if _freeze_path.exists():
        _frozen = _json.loads(_freeze_path.read_text())
        _log("GENE_LIST_CACHE", "_collect_gene_list",
             f"loaded frozen gene list ({len(_frozen['genes'])} genes) from {_freeze_path}")
        return _frozen["genes"], _frozen["ot_scores"], _frozen["ot_cache"]

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

    # Hard-seed curated anchors from disease_registry as a fallback for genes that
    # OT Platform misses (e.g. AHNAK2, DNASE1L3 in RA — strong L2G but absent from
    # OT association query due to database lag or score cutoff differences).
    # These are only added when NOT already in the gene list.
    try:
        from models.disease_registry import DISEASE_GWAS_ANCHORS
        _dkey = (disease_query.get("disease_key") or "").upper()
        _anchors = DISEASE_GWAS_ANCHORS.get(_dkey, frozenset())
        _anchor_added = []
        for _ag in sorted(_anchors):
            if _ag not in genes:
                genes.append(_ag)
                ot_scores[_ag] = max(ot_scores.get(_ag, 0.0), 0.5)  # treat as moderate L2G
                _anchor_added.append(_ag)
        if _anchor_added:
            _log("ANCHOR_SEED", "_collect_gene_list",
                 f"hard-seeded {len(_anchor_added)} registry anchors missing from OT: {sorted(_anchor_added)}")
    except Exception:
        pass

    # Freeze gene list so subsequent runs are stable across OT Platform releases.
    try:
        _freeze_path.parent.mkdir(parents=True, exist_ok=True)
        _freeze_path.write_text(_json.dumps(
            {"genes": genes, "ot_scores": ot_scores, "ot_cache": ot_cache},
            indent=2,
        ))
        _log("GENE_LIST_CACHE", "_collect_gene_list",
             f"froze gene list ({len(genes)} genes) → {_freeze_path}")
    except Exception as _fe:
        _log("GENE_LIST_CACHE_FAIL", "_collect_gene_list", str(_fe))

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
        import pipelines.perturbseq_beta_loader as _pbl
        cache_dir = _pbl._PERTURBSEQ_DIR / dataset_id
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


def _collect_disease_fingerprint_nominees(
    disease_query: dict,
    gwas_gene_set: set[str],
) -> list[str]:
    """
    Return Perturb-seq genes whose KO anti-correlates with the disease DEG profile.

    Complements SVD cosine nomination (genetic-architecture proximity) with functional
    reversal: genes whose knockout transcriptionally reverses the disease state are
    therapeutic candidates even if they don't strongly co-load with GWAS anchors.

    Uses map_disease_to_fingerprints (Pearson r) on the full h5ad DEG vector.
    Nominees must satisfy r ≤ −FINGERPRINT_DISEASE_R_THRESHOLD.
    """
    disease_name = (disease_query.get("disease_name") or "").lower()
    dataset_id: str | None = None
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
        import json as _json
        import pathlib as _pathlib
        from config.scoring_thresholds import FINGERPRINT_DISEASE_R_THRESHOLD, FINGERPRINT_MAX_FP_NOMINEES

        # Cache: reuse existing match JSON if present — map_disease_to_fingerprints is
        # expensive (~1 min for 11k KOs) and the result is stable across runs.
        from mcp_servers.perturbseq_server import _CACHE_DIR
        cached_path = _pathlib.Path(_CACHE_DIR) / dataset_id / "disease_fingerprint_match.json"
        if cached_path.exists():
            with open(cached_path) as fh:
                full = _json.load(fh)
            _log("FINGERPRINT_CACHE", "_collect_disease_fingerprint_nominees",
                 f"loaded {full.get('n_matched', '?')} matched KOs from cache ({dataset_id})")
        else:
            from pipelines.gps_disease_screen import _build_sig_from_h5ad
            from mcp_servers.perturbseq_server import map_disease_to_fingerprints

            disease_de = _build_sig_from_h5ad(disease_query, gps_genes=None, elbow_trim=False)
            if not disease_de:
                _log("FINGERPRINT_NOM_FAIL", "_collect_disease_fingerprint_nominees",
                     "no h5ad DEG available")
                return []

            match = map_disease_to_fingerprints(disease_de, dataset_id)
            if "error" in match:
                _log("FINGERPRINT_NOM_FAIL", "_collect_disease_fingerprint_nominees", match["error"])
                return []

            with open(match["output_path"]) as fh:
                full = _json.load(fh)

        # r floor gate, then top-N cap (r distribution is smooth — no natural gap)
        threshold = -FINGERPRINT_DISEASE_R_THRESHOLD
        # results are sorted r descending (most positive first); reverse for most negative
        by_r = sorted(full["results"], key=lambda x: x["r"])
        passed_floor = [
            r["gene_ko"]
            for r in by_r
            if r["r"] <= threshold and r["gene_ko"] not in gwas_gene_set
        ]
        nominees = passed_floor[:FINGERPRINT_MAX_FP_NOMINEES]
        _log(
            "FINGERPRINT_NOM",
            "_collect_disease_fingerprint_nominees",
            f"{len(nominees)} disease-fingerprint nominees (r≤{threshold:.2f}, top-{FINGERPRINT_MAX_FP_NOMINEES}) from {dataset_id}",
        )
        return nominees
    except Exception as exc:
        _log("FINGERPRINT_NOM_FAIL", "_collect_disease_fingerprint_nominees", str(exc))
        return []


def _collect_svd_nominees(disease_query: dict, gwas_gene_list: list[str]) -> list[str]:
    """
    Return non-GWAS Perturb-seq genes nominated by SVD cosine similarity.

    Computes cosine similarity of each non-GWAS perturbed gene to the GWAS
    centroid in truncated SVD latent space (from svd_loadings.npz written by
    preprocess_rna_fingerprints). Returns only genes above
    FINGERPRINT_SVD_COSINE_MIN, capped at FINGERPRINT_MAX_NONGWAS_NOMINEES.

    Falls back to empty list when svd_loadings.npz is absent (e.g. first run
    before preprocessing) so the pipeline degrades gracefully.
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

    if not dataset_id or not gwas_gene_list:
        return []

    try:
        from mcp_servers.perturbseq_server import compute_svd_nomination_scores
        nominees = compute_svd_nomination_scores(dataset_id, gwas_genes=gwas_gene_list)
        if isinstance(nominees, dict):  # error dict
            _log("SVD_NOM_FAIL", "_collect_svd_nominees", nominees.get("error", "unknown"))
            return []
        genes = [n["gene"] for n in nominees]
        _log("SVD_NOM", "_collect_svd_nominees",
             f"{len(genes)} SVD-nominated non-GWAS genes (cosine≥threshold) from {dataset_id}")
        return genes
    except Exception as exc:
        _log("SVD_NOM_FAIL", "_collect_svd_nominees", str(exc))
        return []


def _collect_cache_first_nominees(
    disease_query: dict,
    ot_scores: dict[str, float],
    exclude_genes: set[str],
) -> tuple[list[dict], set[str]]:
    """
    Single-step cache-first nomination preserving the Ota/Zhu genetic framework.

    Ranking score per gene: |pre_OTA| × l2g_weight × eqtl_coherence_weight

    - pre_OTA = Σ_P β(gene,P) × γ(P,trait)  [S-LDSC γ = genetic link]
    - l2g_weight = f(OT L2G score)            [V2G specificity]
    - eqtl_coherence_weight                   [eQTL direction × OTA direction]

    Returns:
        (nominees, uncached_ot_genes)
        nominees: list of dicts sorted by effective_score desc
        uncached_ot_genes: OT L2G genes not in beta cache (need h5ad β estimation)
    """
    disease_name = (disease_query.get("disease_name") or "").lower()
    dataset_id: str | None = None
    disease_key: str | None = None
    try:
        from graph.schema import DISEASE_CELL_TYPE_MAP, _DISEASE_SHORT_NAMES_FOR_ANCHORS
        short_key = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_name, "")
        if short_key and short_key in DISEASE_CELL_TYPE_MAP:
            dataset_id = DISEASE_CELL_TYPE_MAP[short_key].get("scperturb_dataset")
        disease_key = short_key.upper() if short_key else None
    except Exception:
        pass

    if not dataset_id or not disease_key:
        return [], set()

    try:
        import numpy as _np
        import pathlib as _pl
        from pipelines.ldsc.gamma_loader import get_genetic_nmf_program_gammas, get_cnmf_program_gammas
        from config.scoring_thresholds import (
            SLDSC_GAMMA_FLOOR,
            CACHE_FIRST_L2G_STRONG, CACHE_FIRST_L2G_MODERATE,
            CACHE_FIRST_L2G_WEIGHT_STRONG, CACHE_FIRST_L2G_WEIGHT_MODERATE,
            CACHE_FIRST_L2G_WEIGHT_PROXIMAL, CACHE_FIRST_L2G_WEIGHT_NONE,
            CACHE_FIRST_EQTL_CONCORDANT_WEIGHT, CACHE_FIRST_EQTL_DISCORDANT_WEIGHT,
            CACHE_FIRST_EFFECTIVE_SCORE_MIN, CACHE_FIRST_MAX_NOMINEES,
            PERTURB_NOMINATED_GAMMA_MIN,
        )

        # --- GeneticNMF program gammas (Stim48hr primary; cNMF for CAD) ---
        gnmf_gammas = get_genetic_nmf_program_gammas(disease_key, condition="Stim48hr") \
                      or get_genetic_nmf_program_gammas(disease_key)
        cnmf_gammas = get_cnmf_program_gammas(disease_key)
        gammas = {**gnmf_gammas, **cnmf_gammas}
        active_gammas = {p: d["gamma"] for p, d in gammas.items() if abs(d["gamma"]) >= SLDSC_GAMMA_FLOOR}
        if not active_gammas:
            return [], set()

        # --- GeneticNMF U_scaled loadings (gene × program matrix) ---
        _perturb_dir = _pl.Path(__file__).parent.parent / "data" / "perturbseq" / dataset_id
        _npz_path = _perturb_dir / "genetic_nmf_loadings_stim48hr.npz"
        if not _npz_path.exists():
            _npz_path = _perturb_dir / "genetic_nmf_loadings.npz"
        if not _npz_path.exists():
            return [], set()
        _ld = _np.load(_npz_path, allow_pickle=True)
        _U   = _ld["U_scaled"]   # shape: (n_genes, k)
        _gnames = [str(g) for g in _ld["gene_names"]]
        _gene_idx: dict[str, int] = {g: i for i, g in enumerate(_gnames)}

        # Map active GeneticNMF program names → column indices in U_scaled
        # GeneticNMF programs: RA_GeneticNMF_Stim48hr_C01 → col 0
        import re as _re
        _CPFX = _re.compile(r'_C(\d+)$')
        _prog_col: dict[str, int] = {}
        for pname in active_gammas:
            if "_GeneticNMF_" not in pname and "_cNMF_" not in pname.replace("cNMF", ""):
                m = _CPFX.search(pname)
                if m:
                    col = int(m.group(1)) - 1
                    if 0 <= col < _U.shape[1]:
                        _prog_col[pname] = col
            elif "_GeneticNMF_" in pname:
                m = _CPFX.search(pname)
                if m:
                    col = int(m.group(1)) - 1
                    if 0 <= col < _U.shape[1]:
                        _prog_col[pname] = col

        # --- cNMF MAST betas (CAD-specific; RA has no cNMF) ---
        # Loads cnmf_mast_betas.npz to include Schnitzler-style programs in pre-OTA
        _cnmf_B: _np.ndarray | None = None
        _cnmf_ko_idx: dict[str, int] = {}
        _cnmf_gamma_vec: _np.ndarray | None = None
        _PPFX = _re.compile(r'_P(\d+)$')
        _cnmf_npz_path = _perturb_dir / "cnmf_mast_betas.npz"
        if _cnmf_npz_path.exists():
            _cld = _np.load(_cnmf_npz_path, allow_pickle=True)
            _cnmf_B = _cld["beta"].astype(_np.float32)   # (n_ko, n_prog)
            _cnmf_ko_genes = [str(g) for g in _cld["ko_genes"]]
            _cnmf_prog_ids = [str(p) for p in _cld["program_ids"]]
            _cnmf_ko_idx = {g: i for i, g in enumerate(_cnmf_ko_genes)}
            # Build γ vector for cNMF programs (P01..P60 → CAD_cNMF_P01..P60)
            _cnmf_gamma_vec = _np.zeros(len(_cnmf_prog_ids))
            for j, pid in enumerate(_cnmf_prog_ids):
                for pname, gamma in active_gammas.items():
                    m = _PPFX.search(pname)
                    if m and int(m.group(1)) == int(pid.lstrip("P")):
                        _cnmf_gamma_vec[j] = gamma
                        break
            # Normalise cNMF gamma scale to GeneticNMF scale for unified threshold
            _gnmf_scale = max(abs(active_gammas[p]) for p in _prog_col) if _prog_col else 1.0
            _cnmf_scale = float(_np.max(_np.abs(_cnmf_gamma_vec))) if _cnmf_gamma_vec.any() else 1.0
            if _cnmf_scale > 0:
                _cnmf_gamma_vec = _cnmf_gamma_vec * (_gnmf_scale / _cnmf_scale)

        if not _prog_col and _cnmf_B is None:
            return [], set()

        # --- eQTL index for direction coherence ---
        _eqtl_dir = _pl.Path(__file__).parent.parent / "data" / "eqtl" / "indices"
        eqtl_betas: dict[str, float] = {}
        for eqtl_file in _eqtl_dir.glob(f"{disease_key}_*_top_eqtls.json"):
            try:
                with open(eqtl_file) as f:
                    idx = json.load(f)
                if isinstance(idx, dict):
                    for gene, rec in idx.items():
                        if isinstance(rec, dict) and "beta" in rec:
                            eqtl_betas.setdefault(gene, float(rec["beta"]))
            except Exception:
                pass

        gnmf_gene_set: set[str] = set(_gnames)

        def _l2g_weight(gene: str) -> float:
            score = ot_scores.get(gene, 0.0)
            if isinstance(score, dict):
                score = score.get("score", 0.0) or 0.0
            score = float(score or 0.0)
            if score >= CACHE_FIRST_L2G_STRONG:
                return CACHE_FIRST_L2G_WEIGHT_STRONG
            if score >= CACHE_FIRST_L2G_MODERATE:
                return CACHE_FIRST_L2G_WEIGHT_MODERATE
            if gene in gnmf_gene_set:
                return CACHE_FIRST_L2G_WEIGHT_PROXIMAL  # in Perturb library
            return CACHE_FIRST_L2G_WEIGHT_NONE

        def _eqtl_weight(gene: str, pre_ota: float) -> float:
            eqtl_b = eqtl_betas.get(gene)
            if eqtl_b is None or pre_ota == 0.0:
                return 1.0
            return CACHE_FIRST_EQTL_CONCORDANT_WEIGHT if (eqtl_b * pre_ota) < 0 else CACHE_FIRST_EQTL_DISCORDANT_WEIGHT

        _GENE_SYMBOL_RE = _re.compile(r'^[A-Z][A-Z0-9\-]{1,19}$')
        _LNCRNA_RE      = _re.compile(r'^A[CL]\d{6}')
        _SAFE_TARGET_RE = _re.compile(r'^(safe|non)-', _re.I)

        from config.scoring_thresholds import CACHE_FIRST_STRESS_MEAN_THRESHOLD

        # Vectorized pre-OTA from GeneticNMF: (n_genes,) = U_scaled @ γ_vec
        _gamma_vec = _np.zeros(_U.shape[1])
        for pname, col in _prog_col.items():
            _gamma_vec[col] = active_gammas[pname]
        _pre_ota_gnmf: _np.ndarray = _U @ _gamma_vec

        # cNMF contribution: (n_ko_genes,) = cnmf_B @ cnmf_γ_vec
        _pre_ota_cnmf_map: dict[str, float] = {}
        if _cnmf_B is not None and _cnmf_gamma_vec is not None:
            _pre_ota_cnmf_all = _cnmf_B @ _cnmf_gamma_vec
            _cnmf_ko_genes_list = list(_cnmf_ko_idx.keys())
            for _ci, _cg in enumerate(_cnmf_ko_genes_list):
                _pre_ota_cnmf_map[_cg] = float(_pre_ota_cnmf_all[_ci])

        # All genes: union of GeneticNMF genes and cNMF KO genes
        _all_candidate_genes: dict[str, tuple[float, float]] = {}  # gene → (gnmf_pre_ota, gnmf_mean_abs_b)
        for i, gene in enumerate(_gnames):
            _all_candidate_genes[gene] = (float(_pre_ota_gnmf[i]), float(_np.mean(_np.abs(_U[i]))))
        # cNMF-only genes (not in GeneticNMF)
        for _cg in _pre_ota_cnmf_map:
            if _cg not in _all_candidate_genes:
                _all_candidate_genes[_cg] = (0.0, 0.0)

        gnmf_gene_set = set(_gnames)
        # Extend with cNMF KO genes for l2g_weight proximity check
        _all_perturb_genes = gnmf_gene_set | set(_cnmf_ko_idx.keys())

        # Override l2g_weight helper to use combined perturb gene set
        def _l2g_weight(gene: str) -> float:  # type: ignore[override]
            score = ot_scores.get(gene, 0.0)
            if isinstance(score, dict):
                score = score.get("score", 0.0) or 0.0
            score = float(score or 0.0)
            if score >= CACHE_FIRST_L2G_STRONG:
                return CACHE_FIRST_L2G_WEIGHT_STRONG
            if score >= CACHE_FIRST_L2G_MODERATE:
                return CACHE_FIRST_L2G_WEIGHT_MODERATE
            if gene in _all_perturb_genes:
                return CACHE_FIRST_L2G_WEIGHT_PROXIMAL
            return CACHE_FIRST_L2G_WEIGHT_NONE

        nominees: list[dict] = []
        for gene, (gnmf_pre_ota, gnmf_mean_abs_b) in _all_candidate_genes.items():
            if gene in exclude_genes:
                continue
            if not _GENE_SYMBOL_RE.match(gene):
                continue
            if _LNCRNA_RE.match(gene) or _SAFE_TARGET_RE.match(gene):
                continue
            # Stress filter on GeneticNMF betas (cNMF-only genes skip this)
            if gnmf_mean_abs_b > CACHE_FIRST_STRESS_MEAN_THRESHOLD:
                continue
            # Combined pre-OTA: GeneticNMF + cNMF contributions
            pre_ota = gnmf_pre_ota + _pre_ota_cnmf_map.get(gene, 0.0)
            if pre_ota == 0.0:
                continue

            l2g_w   = _l2g_weight(gene)
            eqtl_w  = _eqtl_weight(gene, pre_ota)
            eff     = abs(pre_ota) * l2g_w * eqtl_w

            l2g_score = ot_scores.get(gene, 0.0)
            if isinstance(l2g_score, dict):
                l2g_score = l2g_score.get("score", 0.0) or 0.0
            has_l2g = float(l2g_score or 0.0) >= CACHE_FIRST_L2G_MODERATE

            if has_l2g:
                if eff < CACHE_FIRST_EFFECTIVE_SCORE_MIN:
                    continue
            else:
                if abs(pre_ota) < PERTURB_NOMINATED_GAMMA_MIN:
                    continue

            nominees.append({
                "gene":             gene,
                "pre_ota":          round(pre_ota, 5),
                "l2g_weight":       l2g_w,
                "eqtl_weight":      eqtl_w,
                "effective_score":  round(eff, 5),
                "ot_l2g_score":     float(l2g_score or 0.0),
                "eqtl_beta":        eqtl_betas.get(gene),
                "nomination_tier":  (
                    "Tier1_Interventional"  if float(l2g_score or 0.0) >= CACHE_FIRST_L2G_STRONG else
                    "Tier2_Convergent"      if float(l2g_score or 0.0) >= CACHE_FIRST_L2G_MODERATE else
                    "Tier2_PerturbNominated"
                ),
            })

        nominees.sort(key=lambda x: -x["effective_score"])
        nominees = nominees[:CACHE_FIRST_MAX_NOMINEES]

        # OT L2G genes not in GeneticNMF or cNMF gene sets still need h5ad β estimation
        uncached_ot_genes = {
            g for g in ot_scores
            if g not in _all_perturb_genes and g not in exclude_genes
        }

        n_tier1 = sum(1 for n in nominees if n["nomination_tier"] == "Tier1_Interventional")
        n_tier2p = sum(1 for n in nominees if n["nomination_tier"] == "Tier2_PerturbNominated")
        _log(
            "CACHE_FIRST_NOM",
            "_collect_cache_first_nominees",
            f"{len(nominees)} nominees ({n_tier1} Tier1, {n_tier2p} PerturbNominated) "
            f"+ {len(uncached_ot_genes)} uncached OT genes | "
            f"top: {[n['gene'] for n in nominees[:5]]}",
        )
        return nominees, uncached_ot_genes

    except Exception as exc:
        _log("CACHE_FIRST_NOM_FAIL", "_collect_cache_first_nominees", str(exc))
        return [], set()


def _collect_high_ota_perturbseq_nominees(
    disease_query: dict,
    exclude_genes: set[str],
) -> list[str]:
    """
    Nominate Perturb-seq genes with |OTA γ| >= PERTURB_NOMINATED_GAMMA_MIN
    that are not already in the gene list via OT L2G or fingerprint paths.

    Uses GeneticNMF U_scaled loadings + program gammas to pre-estimate OTA
    before the full pipeline runs.

    Returns: list of gene symbols, sorted by |pre-OTA| descending.
    """
    disease_name = (disease_query.get("disease_name") or "").lower()
    dataset_id: str | None = None
    disease_key: str | None = None
    try:
        from graph.schema import DISEASE_CELL_TYPE_MAP, _DISEASE_SHORT_NAMES_FOR_ANCHORS
        short_key = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_name, "")
        if short_key and short_key in DISEASE_CELL_TYPE_MAP:
            dataset_id = DISEASE_CELL_TYPE_MAP[short_key].get("scperturb_dataset")
        disease_key = short_key.upper() if short_key else None
    except Exception:
        pass

    if not dataset_id or not disease_key:
        return []

    try:
        import numpy as _np
        import pathlib as _pl
        from pipelines.ldsc.gamma_loader import get_genetic_nmf_program_gammas, get_cnmf_program_gammas
        from config.scoring_thresholds import PERTURB_NOMINATED_GAMMA_MIN, SLDSC_GAMMA_FLOOR

        # GeneticNMF + cNMF program gammas
        gnmf_gammas = get_genetic_nmf_program_gammas(disease_key, condition="Stim48hr") \
                      or get_genetic_nmf_program_gammas(disease_key)
        gammas = {**gnmf_gammas, **get_cnmf_program_gammas(disease_key)}
        active_gammas = {p: d["gamma"] for p, d in gammas.items() if abs(d["gamma"]) >= SLDSC_GAMMA_FLOOR}
        if not active_gammas:
            return []

        # GeneticNMF U_scaled loadings
        _perturb_dir = _pl.Path(__file__).parent.parent / "data" / "perturbseq" / dataset_id
        _npz_path = _perturb_dir / "genetic_nmf_loadings_stim48hr.npz"
        if not _npz_path.exists():
            _npz_path = _perturb_dir / "genetic_nmf_loadings.npz"
        if not _npz_path.exists():
            return []
        _ld = _np.load(_npz_path, allow_pickle=True)
        _U   = _ld["U_scaled"]
        _gnames = [str(g) for g in _ld["gene_names"]]

        import re as _re
        _CPFX = _re.compile(r'_C(\d+)$')
        _gamma_vec = _np.zeros(_U.shape[1])
        for pname, gamma in active_gammas.items():
            m = _CPFX.search(pname)
            if m:
                col = int(m.group(1)) - 1
                if 0 <= col < _U.shape[1]:
                    _gamma_vec[col] = gamma

        _pre_ota_all: _np.ndarray = _U @ _gamma_vec

        nominees: list[tuple[str, float]] = []
        for i, gene in enumerate(_gnames):
            if gene in exclude_genes:
                continue
            pre_ota = float(_pre_ota_all[i])
            if abs(pre_ota) >= PERTURB_NOMINATED_GAMMA_MIN:
                nominees.append((gene, pre_ota))

        nominees.sort(key=lambda x: -abs(x[1]))
        from config.scoring_thresholds import PERTURB_NOMINATED_MAX
        genes = [g for g, _ in nominees[:PERTURB_NOMINATED_MAX]]
        _log(
            "PERTURB_OTA_NOM",
            "_collect_high_ota_perturbseq_nominees",
            f"{len(genes)} high-OTA Perturb-seq nominees (|pre-OTA|≥{PERTURB_NOMINATED_GAMMA_MIN}) "
            f"from {dataset_id} (top: {genes[:5]})",
        )
        return genes
    except Exception as exc:
        _log("PERTURB_OTA_NOM_FAIL", "_collect_high_ota_perturbseq_nominees", str(exc))
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

    from steps.tier1_phenomics.disease_query_builder import run as _pa_run
    from steps.tier1_phenomics.gwas_anchor_validator import run as _sg_run

    t0 = time.time()
    # disease_query["disease_key"] (e.g. "CAD", "AMD") is set by disease_query_builder
    # and propagated as-is through all tiers. Agents must read it from disease_query,
    # not re-derive it via get_disease_key() — single source of truth.
    disease_query = _pa_run(disease_name)
    _log("COMPLETE", "disease_query_builder",
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
    _log("COMPLETE", "gwas_anchor_validator",
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

    # Non-GWAS Perturb-seq nominees: two complementary nomination paths, then union.
    #
    # Path 1 — SVD cosine (USE_SVD_GENE_NOMINATION=True, default):
    #   Genes co-regulated with GWAS anchors in truncated SVD space (genetic-architecture
    #   proximity). Gate: cosine ≥ FINGERPRINT_SVD_COSINE_MIN. No count cap — the
    #   cosine threshold is the principled selector.
    #
    # Path 2 — Disease fingerprint (always active when Path 1 is active):
    #   Genes whose KO anti-correlates with the disease DEG profile (functional reversal).
    #   Gate: Pearson r ≤ −FINGERPRINT_DISEASE_R_THRESHOLD. Recovers known targets
    #   that co-load weakly with GWAS anchors (e.g. IL6R in RA).
    #
    # Legacy fallback (USE_SVD_GENE_NOMINATION=False): full Perturb-seq union.
    try:
        from config.scoring_thresholds import USE_SVD_GENE_NOMINATION, USE_CACHE_FIRST_NOMINATION
    except Exception:
        USE_SVD_GENE_NOMINATION = False
        USE_CACHE_FIRST_NOMINATION = False

    _cf_nominee_tiers: dict[str, str] = {}
    _perturb_nominated_new: set[str] = set()
    ota_nominees: list[str] = []

    if USE_CACHE_FIRST_NOMINATION:
        gwas_set = set(gwas_gene_list)
        cf_nominees, uncached_ot_genes = _collect_cache_first_nominees(
            disease_query,
            ot_scores=gwas_ot_scores,
            exclude_genes=gwas_set,
        )
        perturb_only_genes = [n["gene"] for n in cf_nominees]
        _cf_nominee_tiers = {n["gene"]: n["nomination_tier"] for n in cf_nominees}
        _perturb_nominated_new = {
            n["gene"] for n in cf_nominees if n["nomination_tier"] == "Tier2_PerturbNominated"
        }
        # Uncached OT genes (L2G but no Perturb-seq β) still go through h5ad estimation
        for g in uncached_ot_genes:
            if g not in gwas_set and g not in set(perturb_only_genes):
                perturb_only_genes.append(g)

    elif USE_SVD_GENE_NOMINATION:
        gwas_set = set(gwas_gene_list)
        svd_nominees = _collect_svd_nominees(disease_query, gwas_gene_list)
        svd_set = set(svd_nominees)
        fp_nominees = _collect_disease_fingerprint_nominees(disease_query, gwas_set)
        fp_novel = [g for g in fp_nominees if g not in gwas_set and g not in svd_set]
        # Path 3 — High-OTA Perturb-seq: genes with |pre-OTA γ| ≥ PERTURB_NOMINATED_GAMMA_MIN
        # that aren't already covered by OT L2G, SVD cosine, or fingerprint paths.
        # These are genes Schnitzler-validated as causal regulators of CAD programs but
        # whose GWAS signal is too distal or diffuse for OT L2G to capture.
        already_covered = gwas_set | svd_set | set(fp_novel)
        ota_nominees = _collect_high_ota_perturbseq_nominees(disease_query, already_covered)
        perturb_only_genes = svd_nominees + fp_novel + ota_nominees
        _log(
            "NOMINATION_UNION",
            "gene_list",
            f"{len(svd_nominees)} SVD + {len(fp_novel)} fingerprint-novel + "
            f"{len(ota_nominees)} high-OTA = "
            f"{len(perturb_only_genes)} non-GWAS nominees "
            f"(total {len(gwas_gene_list) + len(perturb_only_genes)}: "
            f"{len(gwas_gene_list)} GWAS + {len(perturb_only_genes)} nominated)",
        )
    else:
        perturb_genes = _collect_perturbseq_genes(disease_query)
        perturb_only_genes = [g for g in perturb_genes if g not in gwas_gene_list]
        if perturb_only_genes:
            _log("PERTURB_UNION", "gene_list",
                 f"added {len(perturb_only_genes)} Perturb-seq-only genes "
                 f"(total {len(gene_list)}: {len(gwas_gene_list)} GWAS + {len(perturb_only_genes)} novel)")
    gene_list = gwas_gene_list + perturb_only_genes

    _disease_key_for_pareto = disease_query.get("disease_key") or ""

    disease_query = {
        **disease_query,
        "gwas_genes": gwas_gene_list,
        "perturb_only_genes": perturb_only_genes,
        "perturb_nominated_genes": (
            _perturb_nominated_new if USE_CACHE_FIRST_NOMINATION
            else (set(ota_nominees) if USE_SVD_GENE_NOMINATION else set())
        ),
        "cache_first_nominee_tiers": _cf_nominee_tiers,  # gene → tier label for ota_gamma_calculator
        "ot_genetic_scores": gwas_ot_scores,
        "_ot_disease_targets_cache": ot_disease_targets_cache,
    }

    # Persist GWAS gene list so gamma_loader can use Zhu et al. GWAS enrichment γ_P
    # in standalone analysis without a full pipeline run.
    if gwas_gene_list:
        import pathlib as _ot_pl, json as _ot_json
        _ot_cache_dir = _ot_pl.Path(__file__).parent.parent / "data" / "ot_cache"
        _ot_cache_dir.mkdir(exist_ok=True)
        _dk_up = (disease_query.get("disease_key") or "").upper()
        if _dk_up:
            (_ot_cache_dir / f"{_dk_up}_genetic_genes.json").write_text(
                _ot_json.dumps(list(gwas_gene_list))
            )

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

    from steps.tier2_pathway.beta_matrix_builder import run as _pga_run
    from steps.tier2_pathway.eqtl_coloc_mapper import run as _rga_run

    t0 = time.time()
    # Independent: run Tier 2 agents in parallel (no shared mutable state).
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_beta = pool.submit(_pga_run, gene_list, disease_query)
        fut_reg  = pool.submit(_rga_run, gene_list, disease_query)
        beta_result       = fut_beta.result()
        regulatory_result = fut_reg.result()
    _log("COMPLETE", "beta_matrix_builder",
         f"tier1={beta_result.get('n_tier1',0)}, virtual={beta_result.get('n_virtual',0)}  "
         f"{time.time()-t0:.1f}s")
    _log("COMPLETE", "eqtl_coloc_mapper",
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

    from steps.tier3_causal.ota_gamma_calculator import run as _cda_run
    from steps.tier3_causal.drug_target_graph_enricher import run as _kgc_run

    t0 = time.time()
    causal_result = _cda_run(beta_result, gamma_estimates, disease_query)
    _log("COMPLETE", "ota_gamma_calculator",
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
    _log("COMPLETE", "drug_target_graph_enricher",
         f"pathways={kg_result.get('n_pathway_edges_added',0)}  {time.time()-t0:.1f}s")

    pipeline_outputs["causal_result"] = causal_result
    pipeline_outputs["kg_result"]     = kg_result
    all_warnings.extend(causal_result.get("warnings", []))
    all_warnings.extend(kg_result.get("warnings", []))

    # Tier 3 quality gate
    tier3_issues = _run_quality_gate_tier3(causal_result)
    all_warnings.extend(tier3_issues)


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

    from steps.tier4_translation.target_ranker import run as _tpa_run
    from steps.tier4_translation.gps_compound_screener import run as _chem_run
    from steps.tier4_translation.trial_landscape_mapper import run as _ct_run

    t0 = time.time()
    # Build Tier 4 context once; passed via disease_query for agent reuse.
    disease_query = {
        **disease_query,
        "_tier4_context": _build_tier4_context(disease_query, causal_result.get("top_genes", []) or []),
    }

    prioritization_result = _tpa_run(causal_result, kg_result, disease_query)
    # Chemistry / GPS disease-program screens need program→trait γ; inject from orchestrator.
    prioritization_result["_gamma_estimates"] = gamma_estimates
    _log("COMPLETE", "target_ranker",
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

    # No Tier4 feedback loop: ranking is owned by target_ranker (OTA-first).
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

    from steps.tier5_writer.report_builder import run as _writer_run

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
    """Write pipeline result JSON and Markdown report to data/ directory."""
    disease_slug = (
        result.get("disease_name", "unknown")
        .lower().replace(" ", "_").replace("-", "_")
    )
    out_path = data_dir / f"analyze_{disease_slug}.json"
    with out_path.open("w") as _f:
        json.dump(result, _f, indent=2, default=str)
    print(f"  Output written:  {out_path}")

    # Write Markdown report alongside JSON
    md_path = data_dir / f"analyze_{disease_slug}.md"
    try:
        sections = []
        if result.get("executive_summary"):
            sections.append(result["executive_summary"])
        if result.get("target_table"):
            sections.append("\n## Target Rankings\n\n" + result["target_table"])
        narratives = result.get("top_target_narratives") or []
        if narratives:
            sections.append("\n## Top Target Narratives\n\n" + "\n\n".join(narratives))
        if result.get("limitations"):
            sections.append("\n## Limitations\n\n" + result["limitations"])
        if sections:
            md_path.write_text("\n\n".join(sections))
            print(f"  Markdown written: {md_path}")
    except Exception as _e:
        print(f"  Markdown write skipped: {_e}")


def _write_plots(disease_name: str) -> None:
    """Generate drug target validation, SVD + NMF plots, and refresh RDF/JSON-LD/CSV exports."""
    from models.disease_registry import get_disease_key
    disease_key = (get_disease_key(disease_name) or disease_name).lower()
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    # Drug target validation — runs first, before plots, so QC is visible before rendering
    try:
        from steps.tier5_writer.drug_target_validator import print_drug_target_report
        import json
        from pathlib import Path as _Path
        _ck_path = _Path(__file__).parent.parent / "data" / "checkpoints" / f"{disease_key.replace(' ', '_')}__tier4.json"
        if not _ck_path.exists():
            # Try slug form
            _slug = disease_name.lower().replace(" ", "_")
            _ck_path = _Path(__file__).parent.parent / "data" / "checkpoints" / f"{_slug}__tier4.json"
        if _ck_path.exists():
            with open(_ck_path) as _fh:
                _ck4 = json.load(_fh)
            _targets = (_ck4.get("prioritization_result") or {}).get("targets") or []
            _dk_upper = (get_disease_key(disease_name) or disease_name).upper()
            print_drug_target_report(_dk_upper, _targets)
        else:
            print(f"  Drug target validation skipped — checkpoint not found: {_ck_path}")
    except Exception as _dv_exc:
        print(f"  Drug target validation skipped ({_dv_exc})")

    # Per-program plots: condition-split GeneticNMF (RA) or cNMF MAST (CAD)
    from outputs.plot_long_island import plot_gnmf_programs
    if disease_key == "ra":
        for _cond in ("Stim48hr", "Rest"):
            try:
                plot_gnmf_programs(disease_key, condition=_cond)
            except Exception as _e:
                print(f"  Plot (gnmf/{_cond}) skipped: {_e}")
    else:
        try:
            plot_gnmf_programs(disease_key)
        except Exception as _e:
            print(f"  Plot (gnmf) skipped ({disease_key}): {_e}")

    # β-program cluster heatmap: shows which gene KOs co-regulate heritable programs
    # and whether winner-takes-all program assignment is biologically coherent.
    try:
        from outputs.plot_beta_heatmap import plot_beta_heatmap
        plot_beta_heatmap(disease_key)
    except Exception as _e:
        print(f"  Beta heatmap skipped ({disease_key}): {_e}")

    # De novo anchor discovery: cluster β profiles, score clusters by GWAS grounding.
    try:
        import json as _json
        from outputs.anchor_discovery import (
            rank_denovo_anchors, plot_anchor_cluster_landscape, _benchmark_direction_map,
        )
        _anchors = rank_denovo_anchors(disease_key, min_n_genes=3)
        _anchor_path = f"outputs/anchor_clusters_{disease_key}.json"
        with open(_anchor_path, "w") as _af:
            _json.dump(_anchors, _af, indent=2)
        print(f"  Anchor clusters: {len(_anchors)} clusters → {_anchor_path}")
        for _r in _anchors[:5]:
            print(f"    #{_r['cluster_id']:2d}  n={_r['n_genes']:4d}  "
                  f"score={_r['anchor_score']:.4f}  dir={_r.get('kd_direction','?'):20s}  "
                  f"anchor={_r['top_gwas_gene']}")
        # Cluster-specific landscape plots for top 5 anchors by score
        _plotted: set = set()
        for _r in _anchors[:5]:
            _ag = _r["top_gwas_gene"]
            _plotted.add(_ag)
            try:
                plot_anchor_cluster_landscape(disease_key, _ag)
            except Exception as _le:
                print(f"    Landscape skipped ({_ag}): {_le}")
        # Benchmark-centric landscape plots: one per cluster containing ≥1 benchmark gene
        # (deduplicated so each cluster only generates one plot)
        _bench_map = _benchmark_direction_map(disease_key)
        _seen_clusters: set = set()
        for _r in _anchors:
            _bench_in_cluster = [g for g in _r["genes"] if g in _bench_map]
            if not _bench_in_cluster or _r["cluster_id"] in _seen_clusters:
                continue
            _seen_clusters.add(_r["cluster_id"])
            _ag = _r["top_gwas_gene"]
            if _ag in _plotted:
                continue
            _plotted.add(_ag)
            try:
                plot_anchor_cluster_landscape(disease_key, _ag)
            except Exception as _le:
                print(f"    Benchmark landscape skipped ({_ag}): {_le}")
    except Exception as _e:
        print(f"  Anchor discovery skipped ({disease_key}): {_e}")

    # Per-program biological annotations
    try:
        from pipelines.program_annotator import annotate_programs
        ann = annotate_programs(disease_key)
        print(f"  Program annotations written: {len(ann)} programs")
    except Exception as _e:
        print(f"  Program annotations skipped ({disease_key}): {_e}")

    # RDF / JSON-LD / CSV graph export
    try:
        from graph.export import export_disease_graph
        result = export_disease_graph(disease_name)
        print(f"  Exports written: {result.get('n_edges', '?')} edges → data/exports/")
    except Exception as _e:
        print(f"  Graph export skipped: {_e}")


def run_tier3_from_checkpoint(disease_name: str) -> dict[str, Any]:
    """
    Re-run Tier 3 (OTA γ calculator) using beta_result saved in an existing
    Tier 3 checkpoint, then proceed through Tier 4+5.

    Use after wiring new program tracks (e.g. cNMF gammas/betas) without
    re-running the expensive Tiers 1–2 (GWAS, eQTL, beta_matrix_builder).
    Overwrites the Tier 3 checkpoint before handing off to run_tier4.
    """
    import time as _time
    _disease_slug = disease_name.lower().replace(" ", "_").replace("-", "_")
    ckpt_dir = Path(__file__).parent.parent / "data" / "checkpoints"
    ckpt3_path = ckpt_dir / f"{_disease_slug}__tier3.json"

    if not ckpt3_path.exists():
        raise FileNotFoundError(
            f"No Tier 3 checkpoint for {disease_name!r}: {ckpt3_path}\n"
            f"Run the full pipeline first: analyze_disease_v2 \"{disease_name}\""
        )

    with ckpt3_path.open() as _f:
        ckpt = json.load(_f)
    print(f"\nLoaded Tier 3 checkpoint (for Tier 3 re-run): {ckpt3_path}")
    print(f"  Disease: {ckpt.get('disease_name')}  |  Run ID: {ckpt.get('run_id')}")

    disease_query     = ckpt["disease_query"]
    genetics_result   = ckpt.get("genetics_result", {})
    beta_result       = ckpt.get("beta_result", {})
    regulatory_result = ckpt.get("regulatory_result", {})
    gamma_estimates   = ckpt.get("gamma_estimates", {})
    kg_result         = ckpt.get("kg_result", {})

    beta_matrix = beta_result.get("beta_matrix", {})
    print(f"\n{'='*60}\nTIER 3 — OTA γ (re-run, {len(beta_matrix)} genes)\n{'='*60}")
    t0 = _time.time()

    from steps.tier3_causal.ota_gamma_calculator import run as _cda_run
    from steps.tier3_causal.drug_target_graph_enricher import run as _kgc_run

    causal_result = _cda_run(beta_result, gamma_estimates, disease_query)
    print(f"  Tier 3 done — n_written={causal_result.get('n_edges_written', 0)}  {_time.time()-t0:.1f}s")
    kg_result = _kgc_run(causal_result, disease_query)

    # Overwrite Tier 3 checkpoint with updated causal_result
    try:
        with ckpt3_path.open("w") as _cf3:
            json.dump({
                **ckpt,
                "causal_result": causal_result,
                "kg_result":     kg_result,
            }, _cf3, indent=2, default=str)
        print(f"  Tier 3 checkpoint updated: {ckpt3_path}")
    except Exception as _e:
        print(f"  Tier 3 checkpoint save failed: {_e}")

    # Hand off to Tier 4+5 using the refreshed checkpoint
    return run_tier4_from_checkpoint(disease_name)


def run_tier4_from_checkpoint(disease_name: str) -> dict[str, Any]:
    """
    Re-run Tier 4 (+ Tier 5) from a saved Tier 3 checkpoint.

    Use this after updating scoring logic, GPS Docker, or compound library without
    re-running the expensive Tiers 1–3 (GWAS, eQTL-MR, Perturb-seq, OTA).

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

    from steps.tier4_translation.target_ranker import run as _tpa_run
    from steps.tier4_translation.gps_compound_screener import run as _chem_run
    from steps.tier4_translation.trial_landscape_mapper import run as _ct_run
    from steps.tier5_writer.report_builder import run as _writer_run

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
        _log("COMPLETE", "target_ranker",
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
        _log("SKIP", "target_ranker",
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

    # Inject benchmark KD targets for GPS emulation regardless of tier
    # (genes validated by targeted KD/KO in disease-relevant cells but absent from
    # genome-scale Perturb-seq atlas — GPS will skip if no signature is available).
    from models.disease_registry import GPS_FORCE_GENES
    _dkey = disease_query.get("disease_key", "")
    _force = GPS_FORCE_GENES.get(_dkey, [])
    if _force:
        disease_query = {**disease_query, "gps_force_genes": _force}

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

    # Save updated tier4 checkpoint (strip GPS annotation bloat before writing).
    try:
        ckpt4_path = ckpt_dir / f"{_disease_slug}__tier4.json"
        _chem_for_ckpt = {
            **chemistry_result,
            "gps_program_reversers": _strip_gps_annotation_bloat(
                chemistry_result.get("gps_program_reversers") or {}
            ),
        }
        with ckpt4_path.open("w") as _cf4:
            json.dump({
                "run_id":                ckpt.get("run_id"),
                "disease_name":          disease_name,
                "prioritization_result": prioritization_result,
                "chemistry_result":      _chem_for_ckpt,
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

    # Build trimmed output using the same path as analyze_disease_v2 — avoids
    # serialising full intermediate dicts (prioritization_result, chemistry_result
    # gps_convergence, causal_result) which can exceed 10 GB for large diseases.
    all_warnings: list[str] = []
    all_warnings.extend(graph_output.get("warnings", []))
    all_warnings.extend(chemistry_result.get("warnings", []))
    all_warnings.extend(clinical_result.get("warnings", []))
    pipeline_outputs = {
        "graph_output":        graph_output,
        "chemistry_result":    chemistry_result,
        "phenotype_result":    disease_query,
        "all_warnings":        all_warnings,
        "pipeline_status":     "SUCCESS",
        "pipeline_duration_s": None,
        "total_edges_written": 0,
        "edge_write_breakdown": {},
    }
    return _build_final_output(pipeline_outputs)


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
    _p3 = _sub.add_parser(
        "run_tier3",
        help="Re-run Tier 3 (OTA γ) from saved beta_result in Tier 3 checkpoint, then Tier 4+5",
    )
    _p3.add_argument("disease_name", help='Disease name matching existing checkpoint')
    _args = _parser.parse_args()
    if _args.command == "analyze_disease_v2":
        _result = analyze_disease_v2(_args.disease_name)
        _data_dir = Path(__file__).parent.parent / "data"
        _write_pipeline_output(_result, _data_dir)
        _write_plots(_args.disease_name)
    elif _args.command == "run_tier4":
        _result = run_tier4_from_checkpoint(_args.disease_name)
        _data_dir = Path(__file__).parent.parent / "data"
        _write_pipeline_output(_result, _data_dir)
        _write_plots(_args.disease_name)
    elif _args.command == "run_tier3":
        _result = run_tier3_from_checkpoint(_args.disease_name)
        _data_dir = Path(__file__).parent.parent / "data"
        _write_pipeline_output(_result, _data_dir)
        _write_plots(_args.disease_name)
    else:
        _parser.print_help()
