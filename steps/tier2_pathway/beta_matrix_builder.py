"""
beta_matrix_builder.py — Tier 2: β_{gene→program} estimation.

Uses the 4-tier fallback hierarchy (Perturb-seq → eQTL → GRN → virtual)
to populate the ProgramBetaMatrix for all genes in the target list.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Expected biology cross-checks: (gene, program) → "positive" | "negative"
BETA_EXPECTATIONS: dict[tuple[str, str], str] = {
    ("TET2",   "inflammatory_NF-kB"):           "positive",
    ("DNMT3A", "DNA_methylation_maintenance"):   "negative",
    ("PCSK9",  "lipid_metabolism"):              "negative",
    ("HLA-DRA","MHC_class_II_presentation"):     "negative",
}

# Tier rank for best-tier tracking (lower = better)
_TIER_RANK: dict[str, int] = {
    "Tier1_Interventional":   1,
    "Tier2_Convergent":       2,
    "Tier2L_LatentHijack":    2,
    "Tier2c_scEQTL":          2,
    "Tier2c_scEQTL_direction": 2,
    "Tier2p_pQTL_MR":         2,
    "Tier2_eQTL_direction":   2,
    "Tier2rb_RareBurden":     2,
    "Tier3_Provisional":      3,
    "provisional_virtual":    4,
}


def run(gene_list: list[str], disease_query: dict) -> dict:
    """
    Estimate β_{gene→program} for all genes using the 4-tier fallback.

    Args:
        gene_list: Genes to estimate β for (from Tier 1 outputs)
        disease_query: DiseaseQuery dict (for tissue/disease context)

    Returns:
        ProgramBetaMatrix-compatible dict
    """
    from mcp_servers.burden_perturb_server import (
        get_gene_perturbation_effect,
        get_program_gene_loadings,
    )
    from mcp_servers.gwas_genetics_server import query_gtex_eqtl
    from mcp_servers.open_targets_server import (
        get_ot_genetic_instruments,
        get_ot_genetic_instruments_bulk,
    )
    from mcp_servers.perturbseq_server import get_perturbseq_signature
    from pipelines.ota_beta_estimation import estimate_beta
    from pipelines.cnmf_programs import (
        get_programs_for_disease,
        get_svd_programs_for_disease,
        run_cnmf_pipeline,
    )
    from graph.schema import DISEASE_CELL_TYPE_MAP

    # h5ad paths for NMF auto-trigger (keyed by perturb_seq_source)
    _base = Path(__file__).parent.parent.parent
    _CNMF_H5AD_MAP: dict[str, str] = {
        "Replogle2022_RPE1":       str(_base / "data/perturbseq/replogle_2022_rpe1/rpe1_normalized_bulk_01.h5ad"),
        # Schnitzler cardiac endothelial (HAEC) Perturb-seq; MI vs non-MI disease axis
        "schnitzler_cad_vascular": str(_base / "data/cellxgene/CAD/CAD_cardiac_endothelial_cell.h5ad"),
    }

    disease_key_for_programs = disease_query.get("disease_key") or ""
    # gwas_gene_set is injected by the orchestrator; used for GWAS-aligned SVD program ranking.
    gwas_gene_set_for_programs: set[str] = set(disease_query.get("gwas_genes", []))

    # Disease DE vector from scRNA-seq h5ad: {gene: log2FC(disease vs healthy)}.
    # Computed once here; passed to SVD program scorer for joint GWAS + DE alignment.
    # genome-wide (elbow_trim=False) so all SVD component gene sets are covered.
    de_vector_for_programs: dict[str, float] | None = None
    try:
        from pipelines.gps_disease_screen import _build_sig_from_h5ad
        _de = _build_sig_from_h5ad(disease_query, gps_genes=None, elbow_trim=False)
        if _de:
            de_vector_for_programs = _de
            logger.info(
                "Disease DE vector loaded: %d genes for SVD program alignment (%s)",
                len(_de), disease_key_for_programs,
            )
    except Exception as _de_exc:
        logger.debug("DE vector unavailable for SVD alignment: %s", _de_exc)

    # Program source priority:
    # 0. cNMF Perturb-seq paper programs (signatures-backed datasets without h5ad).
    #    Schnitzler/CZI: cnmf_program_gene_sets.json (P01-P60) + MAST betas.
    #    SVD Vt z-scores require the h5ad, which these datasets don't have.
    # 1. GWAS + DE aligned SVD — programs ranked by gwas_t × de_pearson;
    #    island axes and program colors converge; genetic north star + disease
    #    transcriptional grounding both satisfied (h5ad datasets only)
    # 2. cNMF from disease-specific h5ad (scRNA-seq NMF — separate from Perturb-seq space)
    # 3. MSigDB Hallmark fallback
    raw_programs: list = []
    programs_info: dict = {}

    # Priority 0: signatures-backed datasets have pre-computed cNMF program gene sets.
    # Use them directly — SVD betas can't be loaded without h5ad.
    _sig_backed_programs_loaded = False
    try:
        from pipelines.perturbseq_beta_loader import _DATASET_SIGNATURES_REGISTRY as _SIG_REG
        _early_ctx = DISEASE_CELL_TYPE_MAP.get(disease_key_for_programs, {})
        _early_dataset = _early_ctx.get("scperturb_dataset")
        if _early_dataset and _early_dataset in _SIG_REG:
            _gene_sets_path = _SIG_REG[_early_dataset].parent / "cnmf_program_gene_sets.json"
            if _gene_sets_path.exists():
                import json as _json
                _cnmf_gs = _json.loads(_gene_sets_path.read_text())
                raw_programs = [
                    {"program_id": pid, "gene_set": genes}
                    for pid, genes in _cnmf_gs.items()
                ]
                programs_info = {
                    "programs": raw_programs,
                    "source": f"cNMF_paper_programs_{_early_dataset}",
                    "cell_type": _early_dataset,
                }
                _sig_backed_programs_loaded = True
                logger.info(
                    "Using %d cNMF paper programs for %s (source=%s, dataset=%s)",
                    len(raw_programs), disease_key_for_programs,
                    programs_info["source"], _early_dataset,
                )
    except Exception as _sig_prog_exc:
        logger.debug("Signatures-backed program load failed, falling through: %s", _sig_prog_exc)

    if not _sig_backed_programs_loaded and disease_key_for_programs:
        svd_info = get_svd_programs_for_disease(
            disease_key_for_programs,
            gwas_genes=gwas_gene_set_for_programs or None,
            de_vector=de_vector_for_programs,
        )
        if svd_info.get("programs"):
            programs_info = svd_info
            raw_programs  = svd_info["programs"]
            logger.info(
                "Using %d SVD-component programs for %s (source=%s)",
                len(raw_programs), disease_key_for_programs, svd_info.get("source"),
            )

    # Fallback: cNMF on disk → auto-trigger NMF → MSigDB
    if not raw_programs:
        programs_info = get_programs_for_disease(disease_key_for_programs) if disease_key_for_programs \
            else get_programs_for_disease("")
        raw_programs = programs_info.get("programs", [])

        if not raw_programs and disease_key_for_programs:
            cell_type = programs_info.get("cell_type", "")
            h5ad_path = _CNMF_H5AD_MAP.get(cell_type, "")
            if h5ad_path and Path(h5ad_path).exists():
                logger.info("No program cache for %s; running cNMF on %s (cell_type=%s)",
                            disease_key_for_programs, Path(h5ad_path).name, cell_type)
                nmf_result = run_cnmf_pipeline(
                    h5ad_path=h5ad_path,
                    cell_type=cell_type,
                    output_dir=str(_base / "data/cnmf_programs"),
                )
                raw_programs = nmf_result.get("programs", [])
                programs_info = nmf_result
            else:
                logger.warning("No NMF cache and no h5ad for cell_type=%r; continuing with empty program set",
                               cell_type)


    # raw_programs is a list of strings (program names) from the registry
    program_ids = [
        p if isinstance(p, str) else (p.get("program_id") or p.get("name", ""))
        for p in raw_programs
    ]

    beta_matrix: dict[str, dict[str, float | None]] = {}
    evidence_tier_per_gene: dict[str, str] = {}
    coloc_h4_per_gene: dict[str, float] = {}  # Phase E: PIP-weighted proxy H4 per gene
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # Optional: pre-filter to genes with any non-virtual evidence.
    #
    # Why: "provisional_virtual" is expensive to discover (requires running the
    # full per-program estimation loop). This pre-screen does a single pass per
    # gene to see if *any* Tier1/2 evidence is available, and drops genes that
    # would be virtual-only.
    #
    # Controls:
    #   - disease_query["tier2_nonvirtual_only"] = True
    #   - env TIER2_NONVIRTUAL_ONLY=1
    #   - env TIER2_MAX_GENES=<int> (cap after filtering; keeps original order)
    # ------------------------------------------------------------------
    nonvirtual_only = bool(disease_query.get("tier2_nonvirtual_only")) or os.getenv("TIER2_NONVIRTUAL_ONLY", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    max_genes_env = os.getenv("TIER2_MAX_GENES", "").strip()
    max_genes = int(max_genes_env) if max_genes_env.isdigit() else None

    # ------------------------------------------------------------------
    # Determine disease-relevant GTEx tissue from DISEASE_CELL_TYPE_MAP
    # ------------------------------------------------------------------
    disease_key = disease_query.get("disease_key") or ""
    ctx = DISEASE_CELL_TYPE_MAP.get(disease_key, {})

    # Two-tier gene split: GWAS genes get full API stack; Perturb-seq
    # nominees (regulators added by reverse-lookup) only need Perturb-seq.
    # gwas_genes is injected by the orchestrator from _collect_gene_list.
    # Falls back to treating ALL genes as GWAS if not provided (safe default).
    gwas_gene_set: set[str] = set(disease_query.get("gwas_genes", []))
    gtex_tissue   = ctx.get("gtex_tissue", "Whole_Blood")
    # Ordered list of fallback tissues to try when the primary has no eQTL for a gene.
    # Disease-specific rationale is in DISEASE_CELL_TYPE_MAP (schema.py).
    # Example: AMD → ["Liver", "Whole_Blood"] catches CFH/C3/CFD (liver-synthesised complement).
    gtex_tissues_secondary: list[str] = ctx.get("gtex_tissues_secondary", [])

    # Diseases whose primary cell type is not a GTEx bulk tissue (e.g. RA → CD4+ T cell)
    # get sc-eQTL (eQTL Catalogue) as primary instrument; GTEx is a last-resort fallback.
    _GTEX_BULK_TISSUES = frozenset({
        "Whole_Blood", "Liver", "Lung", "Kidney_Cortex", "Heart_Left_Ventricle",
        "Artery_Coronary", "Artery_Aorta", "Artery_Tibial", "Spleen", "Muscle_Skeletal",
        "Adipose_Subcutaneous", "Brain_Cortex", "Thyroid", "Skin_Sun_Exposed_Lower_leg",
        "Colon_Sigmoid", "Colon_Transverse", "Small_Intestine_Terminal_Ileum",
    })
    _prefer_sc_eqtl = ctx.get("primary_tissue", "Whole_Blood") not in _GTEX_BULK_TISSUES
    efo_id: str   = disease_query.get("efo_id", "")

    # Phase Z7: Motif and Library for Latent Hijack
    current_disease_motif = disease_query.get("current_disease_motif")
    motif_library = disease_query.get("motif_library")

    # ------------------------------------------------------------------
    # Pre-fetch program gene loadings once per program (cached)
    # Priority: gene_set embedded in raw_programs (cNMF / MSigDB source)
    # Fallback: get_program_gene_loadings (L1000 landmark genes — sparse)
    # ------------------------------------------------------------------
    program_loadings_cache: dict[str, dict] = {}
    # program_gene_sets[pid] = set of gene symbols in that program (for pathway_member check)
    program_gene_sets: dict[str, set[str]] = {}
    # Build a lookup from program_id → gene_set from programs_info directly
    _embedded_gene_sets: dict[str, set[str]] = {
        (p.get("program_id") or p.get("name", "")): set(p.get("gene_set", []))
        for p in raw_programs
        if isinstance(p, dict) and p.get("gene_set")
    }
    for pid in program_ids:
        if pid in _embedded_gene_sets and _embedded_gene_sets[pid]:
            program_loadings_cache[pid] = {"top_genes": list(_embedded_gene_sets[pid])}
            program_gene_sets[pid] = _embedded_gene_sets[pid]
        else:
            try:
                loadings_info = get_program_gene_loadings(pid)
                program_loadings_cache[pid] = loadings_info
                program_gene_sets[pid] = {
                    g if isinstance(g, str) else g.get("gene", "")
                    for g in loadings_info.get("top_genes", [])
                    if g
                }
            except Exception:
                program_loadings_cache[pid] = {}
                program_gene_sets[pid] = set()

    # ------------------------------------------------------------------
    # Pre-load Replogle 2022 quantitative Perturb-seq β (Tier 1 upgrade)
    # If the h5ad is present, all genes get SE-bearing quantitative β.
    # Reuses program_gene_sets built above. Gracefully skipped if absent.
    # ------------------------------------------------------------------
    perturbseq_data: dict | None = None
    # Cell type actually loaded — used for cell-type relevance check in estimate_beta.
    # Defaults to "K562" because burden_perturb_server (qualitative fallback) is K562-based.
    loaded_perturb_cell_type: str = "K562"
    try:
        from pipelines.perturbseq_beta_loader import (
            load_perturbseq_betas, _get_h5ad_path, _download_h5ad,
            _DATASET_SIGNATURES_REGISTRY,
        )
        # Use disease-specific dataset (e.g. replogle_2022_rpe1 for AMD, replogle_2022_k562 generic)
        scperturb_dataset = ctx.get("scperturb_dataset")
        _h5ad_path = _get_h5ad_path(scperturb_dataset) if scperturb_dataset else None
        if _h5ad_path and (not _h5ad_path.exists()) and scperturb_dataset:
            # Auto-download the Replogle pseudo-bulk h5ad when missing.
            # This is required for quantitative Tier1 Perturb-seq β estimates.
            try:
                _download_h5ad(scperturb_dataset)
            except Exception as _dl_exc:
                warnings.append(
                    f"Replogle h5ad download failed for {scperturb_dataset}: {_dl_exc}"
                )

        if _h5ad_path and _h5ad_path.exists():
            # SVD programs: use z-scored Vt values as β (bypasses NES saturation cap).
            # For each SVD component c, β = (Vt[c,g] - mean_c) / std_c, giving ±1-2
            # scale per gene with component-specific differentiation.  For non-SVD
            # programs (NMF, MSigDB), fall back to NES via load_perturbseq_betas.
            _is_svd = programs_info.get("source", "").startswith("SVD_")
            if _is_svd and scperturb_dataset:
                try:
                    from mcp_servers.perturbseq_server import load_svd_vt_betas_zscored
                    disease_key_for_vt = disease_key_for_programs or disease_key or ""
                    perturbseq_data = load_svd_vt_betas_zscored(
                        scperturb_dataset, disease_key_for_vt
                    )
                    logger.info(
                        "SVD z-scored Vt betas loaded: %d perturbed genes (%s)",
                        len(perturbseq_data), scperturb_dataset,
                    )
                except Exception as _vt_exc:
                    warnings.append(f"SVD Vt-z load failed, falling back to NES: {_vt_exc}")
                    _is_svd = False  # fall through to NES path

            if not _is_svd:
                prog_gene_sets_list = {pid: list(gset) for pid, gset in program_gene_sets.items()}
                if any(prog_gene_sets_list.values()):
                    perturbseq_data = load_perturbseq_betas(
                        program_gene_sets=prog_gene_sets_list,
                        dataset_id=scperturb_dataset,
                        gwas_gene_set=gwas_gene_set if gwas_gene_set else None,
                    )
            # Use the dataset ID as the cell_type token so _is_cell_type_matched can
            # look it up in _CELL_TYPE_MATCHED_DATASETS (e.g. "Schnitzler_GSE210681"
            # for CAD, "replogle_2022_rpe1" for AMD).
            if scperturb_dataset:
                loaded_perturb_cell_type = scperturb_dataset
        elif scperturb_dataset and scperturb_dataset in _DATASET_SIGNATURES_REGISTRY:
            # Signatures-backed datasets (e.g. Schnitzler GSE210681, CZI CD4+ T):
            # load_perturbseq_betas handles cnmf_mast_betas.npz and signatures.json.gz
            # internally — no h5ad required.
            _is_svd = programs_info.get("source", "").startswith("SVD_")
            if not _is_svd:
                prog_gene_sets_list = {pid: list(gset) for pid, gset in program_gene_sets.items()}
                if any(prog_gene_sets_list.values()):
                    perturbseq_data = load_perturbseq_betas(
                        program_gene_sets=prog_gene_sets_list,
                        dataset_id=scperturb_dataset,
                        gwas_gene_set=gwas_gene_set if gwas_gene_set else None,
                    )
            loaded_perturb_cell_type = scperturb_dataset
        elif _h5ad_path:
            warnings.append(
                f"Replogle h5ad not found for {scperturb_dataset}: {_h5ad_path}. "
                "Run load_perturbseq_betas(auto_download=True) to download."
            )

        # SVD cosine nomination: extend gene_list with non-GWAS perturb-seq genes
        # that co-load with the GWAS centroid in latent SVD space.
        if scperturb_dataset and gwas_gene_set and perturbseq_data is not None:
            try:
                from mcp_servers.perturbseq_server import compute_svd_nomination_scores
                svd_nominees = compute_svd_nomination_scores(
                    dataset_id=scperturb_dataset,
                    gwas_genes=list(gwas_gene_set),
                )
                current_gene_set = set(gene_list)
                added = 0
                for nom in svd_nominees:
                    g = nom["gene"]
                    if g not in current_gene_set and g in perturbseq_data:
                        gene_list = list(gene_list) + [g]
                        current_gene_set.add(g)
                        added += 1
                if added:
                    logger.info(
                        "SVD nomination: added %d non-GWAS perturb-seq nominees (cosine ≥ %.2f)",
                        added,
                        nom.get("cosine_score", 0.0),
                    )
            except Exception as _svd_exc:
                warnings.append(f"SVD nomination skipped: {_svd_exc}")

    except Exception as exc:
        warnings.append(f"Replogle h5ad pre-load skipped: {exc}")

    # ------------------------------------------------------------------
    # Bulk-prefetch Open Targets genetic instruments for every GWAS gene
    # up-front.  One bulk call primes the SQLite cache under the exact
    # keys that the per-gene `get_ot_genetic_instruments(gene, efo_id)`
    # calls inside the pre-screen and main loop below will look up — so
    # the loops execute against a warm cache with zero extra network
    # round-trips.  Saves ~2-3 HTTPS calls per GWAS gene (hundreds per
    # run for AMD/CAD).  Silent-skip on any failure; falls back to the
    # per-gene path.
    # ------------------------------------------------------------------
    if efo_id:
        _bulk_target_genes = list(gwas_gene_set) if gwas_gene_set else list(gene_list)
        if _bulk_target_genes:
            try:
                get_ot_genetic_instruments_bulk(_bulk_target_genes, efo_id)
            except Exception as _bulk_exc:
                warnings.append(f"OT bulk prefetch failed (falling back to per-gene): {_bulk_exc}")

    def _has_nonvirtual_evidence(gene: str) -> bool:
        """
        Best-effort evidence existence check (cheap; no per-program loop).
        Returns True if any Tier1/2 evidence source appears available.
        """
        # 1) Replogle quantitative perturb-seq betas (fast dict lookup if loaded)
        try:
            if isinstance(perturbseq_data, dict):
                # load_perturbseq_betas outputs a dict keyed by gene in most versions
                if gene in perturbseq_data:
                    return True
        except Exception:
            pass

        # 2) Direct perturb-seq qualitative effect (single call)
        try:
            eff = get_gene_perturbation_effect(gene)
            if (eff.get("top_programs_up") or eff.get("top_programs_dn")):
                return True
        except Exception:
            pass

        # 3) Perturb-seq signature presence (single call; cached upstream)
        try:
            ps = get_perturbseq_signature(gene, disease_context=disease_key or None)
            if ps.get("signature"):
                return True
        except Exception:
            pass

        # 4) GTEx eQTL (single call; cached by api_cache)
        if efo_id is not None:  # keep consistent behaviour for GWAS genes
            try:
                r = query_gtex_eqtl(gene, gtex_tissue)
                if (r.get("eqtls") or r.get("data")):
                    return True
            except Exception:
                pass

        # 5) Open Targets genetics instruments (single call)
        if efo_id:
            try:
                ot = get_ot_genetic_instruments(gene, efo_id)
                if ot.get("instruments"):
                    return True
            except Exception:
                pass

        # 6) pQTL / sc-eQTL / burden are more expensive; omit from pre-screen.
        return False

    if nonvirtual_only:
        kept: list[str] = []
        for g in gene_list:
            if _has_nonvirtual_evidence(g):
                kept.append(g)
        dropped = len(gene_list) - len(kept)
        warnings.append(
            f"TIER2_NONVIRTUAL_ONLY enabled: kept {len(kept)}/{len(gene_list)} genes; dropped {dropped} virtual-only candidates"
        )
        gene_list = kept

    if max_genes is not None and len(gene_list) > max_genes:
        warnings.append(f"TIER2_MAX_GENES cap applied: {max_genes} (from {len(gene_list)})")
        gene_list = gene_list[:max_genes]

    for gene in gene_list:
        gene_betas: dict[str, float | None] = {pid: None for pid in program_ids}

        # Two-tier split: Perturb-seq nominees only have knockout data — no
        # GWAS instruments, no eQTLs, no pQTLs.  Skip all genomics API calls
        # and go straight to Perturb-seq signature fetch below.
        # If gwas_gene_set is empty (old callers without injection), treat all
        # genes as GWAS genes so behaviour is unchanged (safe fallback).
        is_gwas_gene = (not gwas_gene_set) or (gene in gwas_gene_set)

        # --------------------------------------------------------------
        # Fetch eQTL data for this gene once (reused across all programs).
        #
        # Routing priority:
        #   _prefer_sc_eqtl=True  (RA, immune): eQTL Catalogue CD4+ T → GTEx fallback (flagged)
        #   _prefer_sc_eqtl=False (CAD, AMD):   GTEx primary → GTEx secondary → eQTL Catalogue
        #
        # eQTL Catalogue returns beta; GTEx returns nes.  Both are normalised to
        # the "nes" key that estimate_beta expects.
        # --------------------------------------------------------------
        eqtl_data_for_gene: dict | None = None
        if is_gwas_gene and _prefer_sc_eqtl:
            # --- sc-eQTL first path (CD4+ T cell diseases) ---
            _sc_cell_types = ctx.get("sc_eqtl_cell_types", [])
            _sc_study      = ctx.get("sc_eqtl_study")
            if _sc_cell_types and _sc_study:
                try:
                    from mcp_servers.eqtl_catalogue_server import get_sc_eqtl as _get_sc_eqtl
                    _sc_result = _get_sc_eqtl(
                        gene,
                        cell_type=_sc_cell_types[0],
                        study_label=_sc_study,
                        disease=disease_key or None,
                    )
                    _top_sc = _sc_result.get("top_eqtl")
                    if _top_sc and _sc_result.get("n_found", 0) > 0:
                        _beta = _top_sc.get("beta") or _top_sc.get("nes") or _top_sc.get("effect_size")
                        if _beta is not None:
                            eqtl_data_for_gene = {
                                "nes":    float(_beta),
                                "se":     _top_sc.get("se"),
                                "pvalue": float(_top_sc.get("pvalue") or 1.0),
                                "tissue": _sc_result.get("cell_type", _sc_cell_types[0]),
                                "source": f"eQTL_Catalogue_{_sc_study}_{_sc_cell_types[0]}",
                            }
                except Exception as exc:
                    warnings.append(f"{gene}: sc-eQTL (primary) prefetch failed: {exc}")

            # GTEx Whole Blood as last-resort fallback — flagged as cell-type-mismatched
            if eqtl_data_for_gene is None:
                try:
                    eqtl_result = query_gtex_eqtl(gene, gtex_tissue)
                    eqtls = eqtl_result.get("eqtls", []) or eqtl_result.get("data", [])
                    if eqtls:
                        top = eqtls[0]
                        nes = top.get("nes") or top.get("effect_size")
                        if nes is not None:
                            eqtl_data_for_gene = {
                                "nes":              float(nes),
                                "se":               top.get("se"),
                                "pvalue":           float(top.get("pval_nominal") or top.get("pvalue") or 1.0),
                                "tissue":           gtex_tissue,
                                "source":           f"GTEx_fallback_mismatched({gtex_tissue})",
                                "cell_type_mismatch": True,
                            }
                except Exception as exc:
                    warnings.append(f"{gene}: GTEx eQTL fallback failed: {exc}")

        elif is_gwas_gene:
            # --- GTEx-first path (CAD, AMD, tissue-matched bulk diseases) ---
            try:
                eqtl_result = query_gtex_eqtl(gene, gtex_tissue)
                eqtls = eqtl_result.get("eqtls", []) or eqtl_result.get("data", [])
                if eqtls:
                    top = eqtls[0]
                    nes = top.get("nes") or top.get("effect_size")
                    if nes is not None:
                        eqtl_data_for_gene = {
                            "nes":    float(nes),
                            "se":     top.get("se"),
                            "pvalue": float(top.get("pval_nominal") or top.get("pvalue") or 1.0),
                            "tissue": gtex_tissue,
                        }
            except Exception as exc:
                warnings.append(f"{gene}: GTEx eQTL prefetch failed: {exc}")

            # Secondary-tissue fallback: try disease-specific tissue priority list.
            # Covers genes whose causal tissue differs from the disease primary tissue.
            # Example: AMD complement genes (CFH, C3, CFD) are liver-synthesised →
            # Retina eQTL is near-zero; Liver eQTL gives a real instrument.
            if eqtl_data_for_gene is None and gtex_tissues_secondary:
                for _sec_tissue in gtex_tissues_secondary:
                    try:
                        _sec_result = query_gtex_eqtl(gene, _sec_tissue)
                        _sec_eqtls = _sec_result.get("eqtls", []) or _sec_result.get("data", [])
                        if _sec_eqtls:
                            _top = _sec_eqtls[0]
                            _nes = _top.get("nes") or _top.get("effect_size")
                            if _nes is not None:
                                eqtl_data_for_gene = {
                                    "nes":    float(_nes),
                                    "se":     _top.get("se"),
                                    "pvalue": float(_top.get("pval_nominal") or _top.get("pvalue") or 1.0),
                                    "tissue": _sec_tissue,
                                    "source": f"GTEx_secondary_tissue({_sec_tissue})",
                                }
                                break  # first tissue with a hit wins
                    except Exception as _exc_sec:
                        pass  # non-fatal — try next secondary tissue

        # --------------------------------------------------------------
        # Pre-fetch OT genetic instruments (Tier 2b): GWAS credible sets
        # and eQTL catalogue betas from Open Targets.
        # Only attempted when an EFO ID is available.
        # --------------------------------------------------------------
        ot_instruments_for_gene: dict | None = None
        if is_gwas_gene and efo_id:
            try:
                ot_result = get_ot_genetic_instruments(gene, efo_id)
                if ot_result.get("instruments"):
                    ot_instruments_for_gene = ot_result
            except Exception as exc:
                warnings.append(f"{gene}: OT instruments prefetch failed: {exc}")

        # --------------------------------------------------------------
        # Phase E: PIP-weighted proxy COLOC H4 (Pritchard P4).
        #
        # eqtl_coloc_mapper runs in parallel with this module so true COLOC
        # posteriors are unavailable here.  When both a cis-eQTL and an OT GWAS
        # instrument exist for the same gene, proxy H4 = 0.85 (strong signal that
        # the eQTL shares a causal variant with the GWAS locus).  This is then
        # downscaled by the gene's fine-mapped PIP:
        #
        #   proxy_h4 = 0.85 × min(pip / PIP_ANCHOR, 1.0)
        #
        # where PIP_ANCHOR = 0.10.  A gene at a well-fine-mapped locus (PIP≥0.10)
        # keeps proxy_h4=0.85 ≥ COLOC_H4_MIN (0.80) → Tier2.
        # A gene with PIP=0.05 → proxy_h4=0.43 < 0.80 → routed to Tier2.5
        # (direction-only, higher sigma).  This is the first time coloc_h4 is
        # non-None in the live pipeline; it activates the gating in estimate_beta_tier2.
        # --------------------------------------------------------------
        coloc_h4_for_gene: float | None = None
        _PIP_ANCHOR = 0.10
        _PROXY_H4_MAX = 0.85
        if eqtl_data_for_gene is not None and ot_instruments_for_gene is not None:
            _gwas_instrs = [
                i for i in ot_instruments_for_gene.get("instruments", [])
                if i.get("instrument_type") == "gwas_credset"
            ]
            if _gwas_instrs:
                try:
                    from mcp_servers.gwas_genetics_server import get_gene_max_pip_for_trait
                    _pip_res = get_gene_max_pip_for_trait(gene, efo_id)
                    _pip = float(_pip_res.get("max_pip") or 0.0)
                except Exception:
                    _pip = 0.0
                # Scale proxy H4 by PIP: full credit at PIP≥0.10, proportional below
                coloc_h4_for_gene = _PROXY_H4_MAX * min(_pip / _PIP_ANCHOR, 1.0) if _pip > 0 else _PROXY_H4_MAX * 0.5

        # --------------------------------------------------------------
        # Pre-fetch pQTL instruments (Tier 2p): protein QTL from UKB-PPP /
        # INTERVAL / deCODE via eQTL Catalogue.  Used for genes with coding
        # variants where cis-eQTL is absent (CFH Y402H, LPA, TREM2 R47H).
        # Only fetch if GTEx eQTL was not found.
        # --------------------------------------------------------------
        pqtl_data_for_gene: dict | None = None
        if is_gwas_gene and eqtl_data_for_gene is None:
            _pqtl_key_genes = set(ctx.get("pqtl_key_genes", []))
            if gene in _pqtl_key_genes or not _pqtl_key_genes:
                try:
                    from mcp_servers.eqtl_catalogue_server import get_pqtl_instruments
                    _pqtl_result = get_pqtl_instruments(gene, disease=disease_key or None)
                    if _pqtl_result.get("n_found", 0) > 0:
                        pqtl_data_for_gene = _pqtl_result
                except Exception as exc:
                    warnings.append(f"{gene}: pQTL pre-fetch failed: {exc}")

        # --------------------------------------------------------------
        # Pre-fetch sc-eQTL (Tier 2c): cell-type-specific eQTL from
        # eQTL Catalogue (OneK1K, Blueprint).
        # For _prefer_sc_eqtl diseases this was already run above as primary;
        # skip here to avoid a duplicate API call.
        # For GTEx-first diseases, run as gap-filler when GTEx found nothing.
        # --------------------------------------------------------------
        sc_eqtl_data_for_gene: dict | None = None
        if is_gwas_gene and not _prefer_sc_eqtl and eqtl_data_for_gene is None:
            _sc_cell_types = ctx.get("sc_eqtl_cell_types", [])
            _sc_study      = ctx.get("sc_eqtl_study")
            try:
                from mcp_servers.eqtl_catalogue_server import get_sc_eqtl
                _sc_result = get_sc_eqtl(
                    gene,
                    cell_type=_sc_cell_types[0] if _sc_cell_types else None,
                    study_label=_sc_study,
                    disease=disease_key or None,
                )
                if _sc_result.get("n_found", 0) > 0:
                    sc_eqtl_data_for_gene = _sc_result
            except Exception as exc:
                warnings.append(f"{gene}: sc-eQTL pre-fetch failed: {exc}")

        # --------------------------------------------------------------
        # Pre-fetch UKB WES rare variant burden (Tier 2rb).
        # Used for structural genes (FBN2, HMCN1) or any gene where eQTL
        # and pQTL are both absent.  Provides direction constraint from
        # LoF carrier enrichment in disease cohorts.
        # Only fetch when both GTEx eQTL and pQTL are absent.
        # --------------------------------------------------------------
        burden_data_for_gene: dict | None = None
        if is_gwas_gene and eqtl_data_for_gene is None and pqtl_data_for_gene is None:
            try:
                from mcp_servers.ukb_wes_server import get_burden_direction_for_gene
                _burden = get_burden_direction_for_gene(gene, disease=disease_key or None)
                if _burden is not None:
                    burden_data_for_gene = _burden
            except Exception as exc:
                warnings.append(f"{gene}: UKB WES burden pre-fetch failed: {exc}")

        # Vt-direct fast path was removed: Vt values (~0.01) are 10-20x smaller
        # than NES betas (~0.3-0.8), causing OTA gamma to collapse below ranking
        # thresholds. NES computed by load_perturbseq_betas against SVD component
        # gene sets (top-|U_scaled[:,c]| genes) gives the correct scale.
        tier = "provisional_virtual"
        best_tier_rank = _TIER_RANK["provisional_virtual"]

        try:
            any_filled = False
            for pid in program_ids:

                # Standard path: extract gene loading for this (gene, program) pair
                loadings_info = program_loadings_cache.get(pid, {})
                loading: float | None = None
                for g in loadings_info.get("top_genes", []):
                    g_name = g if isinstance(g, str) else g.get("gene", "")
                    g_wt = None if isinstance(g, str) else (g.get("weight") or g.get("loading"))
                    if g_name == gene:
                        loading = float(g_wt) if g_wt is not None else 1.0
                        break

                pathway_member = gene in program_gene_sets.get(pid, set())

                single = estimate_beta(
                    gene, pid,
                    perturbseq_data=perturbseq_data,
                    eqtl_data=eqtl_data_for_gene,
                    coloc_h4=coloc_h4_for_gene,
                    ot_instruments=ot_instruments_for_gene,
                    sc_eqtl_data=sc_eqtl_data_for_gene,
                    pqtl_data=pqtl_data_for_gene,
                    burden_data=burden_data_for_gene,
                    program_gene_set=program_gene_sets.get(pid),
                    cell_type=loaded_perturb_cell_type,
                    disease=disease_key,
                    program_loading=loading,
                    pathway_member=pathway_member,
                    # Phase Z7: Latent Hijack
                    current_disease_motif=current_disease_motif,
                    motif_library=motif_library,
                )
                beta_val = single.get("beta") if single else None
                if beta_val is not None:
                    et = single.get("evidence_tier", "provisional_virtual")
                    gene_betas[pid] = {
                        "beta":          float(beta_val),
                        "evidence_tier": et,
                        "beta_sigma":    single.get("beta_sigma"),
                        "program_gamma": single.get("program_gamma"),
                    }
                    any_filled = True
                    r = _TIER_RANK.get(et, 4)
                    if r < best_tier_rank:
                        best_tier_rank = r
                        tier = et
            if not any_filled:
                tier = "provisional_virtual"

        except Exception as exc:
            warnings.append(f"{gene}: estimate_beta failed ({exc}); falling back")
            tier = "provisional_virtual"

        # --------------------------------------------------------------
        # If still virtual, try direct Perturb-seq lookup (Tier 1)
        # This handles the sign-only qualitative path directly, bypassing
        # estimate_beta for cases where the server has explicit up/dn lists.
        # --------------------------------------------------------------
        if tier == "provisional_virtual":
            try:
                perturb = get_gene_perturbation_effect(gene)
                # top_programs_up/dn are lists of program names (qualitative direction)
                up_progs = perturb.get("top_programs_up", [])
                dn_progs = perturb.get("top_programs_dn", [])
                if up_progs or dn_progs:
                    for pid in program_ids:
                        if pid in up_progs:
                            gene_betas[pid] = {"beta": 1.0, "evidence_tier": "Tier1_Interventional"}
                        elif pid in dn_progs:
                            gene_betas[pid] = {"beta": -1.0, "evidence_tier": "Tier1_Interventional"}
                    if any(isinstance(v, dict) for v in gene_betas.values()):
                        tier = "Tier1_Interventional"
            except Exception as exc:
                warnings.append(f"{gene}: Perturb-seq lookup failed: {exc}")

        beta_matrix[gene] = gene_betas
        evidence_tier_per_gene[gene] = tier
        if coloc_h4_for_gene is not None:
            coloc_h4_per_gene[gene] = round(coloc_h4_for_gene, 3)

    # ------------------------------------------------------------------
    # Consolidate tier counts from evidence_tier_per_gene
    # ------------------------------------------------------------------
    n_tier1 = sum(1 for t in evidence_tier_per_gene.values() if t == "Tier1_Interventional")
    n_tier2 = sum(1 for t in evidence_tier_per_gene.values() if t in ("Tier2_Convergent", "Tier2_eQTL_direction", "Tier2_PerturbNominated"))
    n_tier3 = sum(1 for t in evidence_tier_per_gene.values() if t in ("Tier3_Provisional", "moderate_transferred", "moderate_grn"))
    n_virtual = sum(1 for t in evidence_tier_per_gene.values() if t == "provisional_virtual")

    # ------------------------------------------------------------------
    # Biology cross-check
    # ------------------------------------------------------------------
    for (gene, prog), expected_dir in BETA_EXPECTATIONS.items():
        if gene not in beta_matrix:
            continue
        raw = beta_matrix[gene].get(prog)
        if raw is None:
            continue
        # beta_matrix stores dicts {"beta": float, ...} or plain floats (defensive)
        beta_val = raw.get("beta") if isinstance(raw, dict) else raw
        if beta_val is None:
            continue
        direction_ok = (
            (expected_dir == "positive" and beta_val > 0)
            or (expected_dir == "negative" and beta_val < 0)
        )
        if not direction_ok:
            warnings.append(
                f"Biology mismatch: {gene} → {prog} β={beta_val:.3f} "
                f"(expected {expected_dir}) — flag for Scientific Reviewer"
            )

    # ------------------------------------------------------------------
    # Ticket 5: state_space output block (additive — no-op if h5ad absent)
    # ------------------------------------------------------------------
    state_space = _maybe_run_state_space(disease_key, warnings)

    # ------------------------------------------------------------------
    # Activation-stratified NMF (RA/SLE only): compute per-program
    # Stim8hr/Rest bias from CZI varm data for use in compute_ota_gamma.
    # Only available after precompute_activation_biases() is run offline.
    # ------------------------------------------------------------------
    program_activation_biases: dict[str, float] | None = None
    if disease_key == "RA":
        try:
            from mcp_servers.perturbseq_server import load_program_activation_biases
            _pg_sets_list = {pid: list(gset) for pid, gset in program_gene_sets.items() if gset}
            program_activation_biases = load_program_activation_biases(
                "czi_2025_cd4t_perturb", _pg_sets_list
            )
            if program_activation_biases:
                logger.info("Loaded activation biases for %d programs (disease=%s)",
                            len(program_activation_biases), disease_key)
        except Exception as _ab_exc:
            logger.debug("Activation biases not available: %s", _ab_exc)

    return {
        "genes":                      gene_list,
        "programs":                   program_ids,
        "beta_matrix":                beta_matrix,
        "evidence_tier_per_gene":     evidence_tier_per_gene,
        "coloc_h4_per_gene":          coloc_h4_per_gene,   # Phase E: PIP-weighted proxy H4
        "n_tier1":                    n_tier1,
        "n_tier2":                    n_tier2,
        "n_tier3":                    n_tier3,
        "n_virtual":                  n_virtual,
        "warnings":                   warnings,
        "state_space":                state_space,
        "program_activation_biases":  program_activation_biases,
    }


# ---------------------------------------------------------------------------
# Ticket 5: state_space block helper
# ---------------------------------------------------------------------------

def _maybe_run_state_space(disease_key: str, warnings: list[str]) -> dict:
    """
    Run state-space pipeline if a cached h5ad exists for this disease.

    Returns a state_space block:
        {"available": True,  "states": ..., "transitions": ..., "basin_summary": ...}  # h5ad present
        {"available": False, "reason": str}                                             # no h5ad
        {"available": False, "reason": str, "error": str}                               # pipeline error
    """
    if not disease_key:
        return {"available": False, "reason": "disease_key not resolved"}

    # Check cache only — do not trigger a download in the middle of a pipeline run
    try:
        from pipelines.discovery.cellxgene_downloader import (
            DISEASE_CELLXGENE_MAP, _CACHE_ROOT,
        )
        ctx = DISEASE_CELLXGENE_MAP.get(disease_key, {})
        if not ctx:
            return {"available": False, "reason": f"disease_key '{disease_key}' not in DISEASE_CELLXGENE_MAP"}
        from pathlib import Path
        # Try priority_cell h5ad first; fall back to any other cell type that has a cached file.
        # This handles the transition from old SMC-priority to new endothelial-priority for CAD
        # while the endothelial h5ad is not yet downloaded.
        priority_cell = ctx["priority_cell"]
        _candidates = [priority_cell] + [c for c in ctx.get("cell_types", []) if c != priority_cell]
        h5ad_path: str | None = None
        for _ct in _candidates:
            _p = _CACHE_ROOT / disease_key / f"{disease_key}_{_ct.replace(' ', '_')}.h5ad"
            if _p.exists():
                h5ad_path = str(_p)
                if _ct != priority_cell:
                    import warnings as _w
                    _w.warn(
                        f"[state_space] {disease_key} priority h5ad ({priority_cell}) not cached; "
                        f"using fallback: {_ct}. Download correct cell type for full cell-context alignment.",
                        UserWarning, stacklevel=2,
                    )
                break
        if h5ad_path is None:
            return {"available": False, "reason": f"h5ad not cached for {disease_key} — run download_disease_scrna first"}
    except Exception as exc:
        return {"available": False, "reason": "cache check failed", "error": str(exc)}

    try:
        from pipelines.state_space.latent_model import build_disease_latent_space
        from pipelines.state_space.state_definition import define_cell_states
        from pipelines.state_space.transition_graph import infer_state_transition_graph, get_basin_summary

        latent = build_disease_latent_space(disease_key, [h5ad_path])
        if latent.get("error"):
            warnings.append(f"state_space: latent model failed: {latent['error']}")
            return {"available": False, "reason": "latent build error", "error": latent["error"]}

        states = define_cell_states(latent, disease_key)
        transitions = infer_state_transition_graph(latent, states, disease_key)

        # Compact summary for downstream agents (full objects available if needed)
        state_summary = {
            res_name: [
                {
                    "state_id":            s.state_id,
                    "cell_type":           s.cell_type,
                    "n_cells":             s.n_cells,
                    "pathological_score":  s.pathological_score,
                    "stability_score":     s.stability_score,
                    "marker_genes":        s.marker_genes[:5],
                    "program_labels":      s.program_labels,
                    "context_tags":        s.context_tags,
                }
                for s in slist
            ]
            for res_name, slist in states.items()
        }

        return {
            "available":         True,
            "h5ad_path":         h5ad_path,
            "backend":           latent["backend"],
            "n_cells":           latent["adata"].n_obs,
            "states":            state_summary,
            "n_transitions":     len(transitions["transitions"]),
            "pathologic_basins": transitions["pathologic_basin_ids"],
            "healthy_basins":    transitions["healthy_basin_ids"],
            "escape_basins":     transitions["escape_basin_ids"],
            "basin_summary":     get_basin_summary(transitions),
            "confidence":        transitions["confidence_summary"],
        }

    except Exception as exc:
        err = str(exc)
        warnings.append(f"state_space pipeline failed: {err}")
        return {"available": False, "reason": "pipeline exception", "error": err}
