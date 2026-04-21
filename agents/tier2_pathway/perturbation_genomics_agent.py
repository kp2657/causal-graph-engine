"""
perturbation_genomics_agent.py — Tier 2 agent: β_{gene→program} estimation.

Uses the 4-tier fallback hierarchy (Perturb-seq → eQTL → GRN → virtual)
to populate the ProgramBetaMatrix for all genes in the target list.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Expected biology cross-checks: (gene, program) → "positive" | "negative"
BETA_EXPECTATIONS: dict[tuple[str, str], str] = {
    ("TET2",   "inflammatory_NF-kB"):           "positive",
    ("DNMT3A", "DNA_methylation_maintenance"):   "negative",
    ("PCSK9",  "lipid_metabolism"):              "negative",
    ("HLA-DRA","MHC_class_II_presentation"):     "negative",
}

from models.disease_registry import get_disease_key as _get_disease_key

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
    "Tier2pt_ProteinChannel":  2,
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
    from mcp_servers.open_targets_server import get_ot_genetic_instruments
    from mcp_servers.perturbseq_server import get_perturbseq_signature
    from pipelines.ota_beta_estimation import estimate_beta
    from pipelines.cnmf_programs import get_programs_for_disease
    from graph.schema import DISEASE_CELL_TYPE_MAP

    # Use discovery-aware program source: cNMF on disk → MSigDB Hallmark → hardcoded
    disease_key_for_programs = _get_disease_key(
        disease_query.get("disease_name", "").lower()
    ) or ""
    programs_info = get_programs_for_disease(disease_key_for_programs) if disease_key_for_programs \
        else get_programs_for_disease("")
    raw_programs = programs_info.get("programs", [])
    # raw_programs is a list of strings (program names) from the registry
    program_ids = [
        p if isinstance(p, str) else (p.get("program_id") or p.get("name", ""))
        for p in raw_programs
    ]

    beta_matrix: dict[str, dict[str, float | None]] = {}
    evidence_tier_per_gene: dict[str, str] = {}
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
    disease_name = disease_query.get("disease_name", "").lower()
    disease_key = _get_disease_key(disease_name) or ""
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
    lincs_cell_line = ctx.get("lincs_cell_line")
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
        from pipelines.replogle_parser import load_replogle_betas, _get_h5ad_path, _download_h5ad
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
            prog_gene_sets_list = {pid: list(gset) for pid, gset in program_gene_sets.items()}
            if any(prog_gene_sets_list.values()):
                perturbseq_data = load_replogle_betas(
                    program_gene_sets=prog_gene_sets_list,
                    dataset_id=scperturb_dataset,
                )
            # Use the dataset ID as the cell_type token so _is_cell_type_matched can
            # look it up in _CELL_TYPE_MATCHED_DATASETS (e.g. "Schnitzler_GSE210681"
            # for CAD, "replogle_2022_rpe1" for AMD).
            if scperturb_dataset:
                loaded_perturb_cell_type = scperturb_dataset
        elif _h5ad_path:
            warnings.append(
                f"Replogle h5ad not found for {scperturb_dataset}: {_h5ad_path}. "
                "Run load_replogle_betas(auto_download=True) to download."
            )
    except Exception as exc:
        warnings.append(f"Replogle h5ad pre-load skipped: {exc}")

    def _has_nonvirtual_evidence(gene: str) -> bool:
        """
        Best-effort evidence existence check (cheap; no per-program loop).
        Returns True if any Tier1/2 evidence source appears available.
        """
        # 1) Replogle quantitative perturb-seq betas (fast dict lookup if loaded)
        try:
            if isinstance(perturbseq_data, dict):
                # load_replogle_betas outputs a dict keyed by gene in most versions
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
        # Fetch eQTL data for this gene once (reused across all programs)
        # Supports both real GTEx format {"eqtls": [...]} and test mocks
        # that return {"data": [...]}.
        # --------------------------------------------------------------
        eqtl_data_for_gene: dict | None = None
        if is_gwas_gene:
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
                                    "tissue": _sec_tissue,
                                    "source": f"GTEx_secondary_tissue({_sec_tissue})",
                                }
                                break  # first tissue with a hit wins
                    except Exception as _exc_sec:
                        pass  # non-fatal — try next secondary tissue

            # eQTL Catalogue fallback: immune datasets (IBD/RA/SLE relevance)
            # Only attempted when GTEx returned no eQTLs and disease is immune-relevant.
            if eqtl_data_for_gene is None:
                _IMMUNE_DISEASE_KEYS = frozenset({"IBD", "RA", "SLE", "MS", "T1D"})
                if disease_key in _IMMUNE_DISEASE_KEYS:
                    try:
                        from mcp_servers.gwas_genetics_server import query_eqtl_catalogue
                        catalogue_result = query_eqtl_catalogue(gene)
                        best_beta = catalogue_result.get("best_beta")
                        if best_beta is not None:
                            eqtl_data_for_gene = {
                                "nes":    float(best_beta),
                                "se":     catalogue_result.get("best_se"),
                                "tissue": catalogue_result.get("best_dataset", "immune_catalogue"),
                                "source": "eQTL_Catalogue",
                            }
                    except Exception as exc:
                        warnings.append(f"{gene}: eQTL Catalogue fallback failed: {exc}")

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
        # eQTL Catalogue (OneK1K, Blueprint).  Fills gaps where GTEx bulk
        # dilutes signal present in a minority cell type.
        # Only fetch if GTEx eQTL was not found.
        # --------------------------------------------------------------
        sc_eqtl_data_for_gene: dict | None = None
        if is_gwas_gene and eqtl_data_for_gene is None:
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

        # --------------------------------------------------------------
        # Pre-fetch Perturb-seq CRISPR signature for this gene (Tier 3).
        # Uses disease-context-matched cell line from perturbseq registry.
        # One call per gene; reused across all programs.
        # --------------------------------------------------------------
        lincs_signature_for_gene: dict | None = None
        try:
            ps_result = get_perturbseq_signature(gene, disease_context=disease_key or None)
            sig = ps_result.get("signature")
            if sig:
                lincs_signature_for_gene = sig
        except Exception as exc:
            warnings.append(f"{gene}: Perturb-seq pre-fetch failed: {exc}")

        # --------------------------------------------------------------
        # Main β estimation via 4-tier fallback (Tier1 → Tier2 → ... )
        # estimate_beta now receives eqtl_data and program_loading so
        # Tier 2 (eQTL-MR) activates when Tier 1 data is absent.
        # --------------------------------------------------------------
        tier = "provisional_virtual"
        best_tier_rank = _TIER_RANK["provisional_virtual"]
        try:
            any_filled = False
            for pid in program_ids:
                # Extract gene loading for this (gene, program) pair
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
                    ot_instruments=ot_instruments_for_gene,
                    sc_eqtl_data=sc_eqtl_data_for_gene,
                    pqtl_data=pqtl_data_for_gene,
                    burden_data=burden_data_for_gene,
                    lincs_signature=lincs_signature_for_gene,
                    program_gene_set=program_gene_sets.get(pid),
                    cell_line=lincs_cell_line,
                    cell_type=loaded_perturb_cell_type,
                    disease=disease_key,
                    program_loading=loading,
                    pathway_member=pathway_member,
                    # Phase Z7: Latent Hijack
                    current_disease_motif=current_disease_motif,
                    motif_library=motif_library,
                )
                beta_val = single.get("beta")
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

        # --------------------------------------------------------------
        # Protein channel: additive virtual arc for protein-level mechanisms.
        # Runs after the main fallback loop so it never blocks Tier1/2 evidence —
        # it adds a "__protein_channel__" slot to beta_matrix that carry their
        # own embedded program_gamma (used by compute_ota_gamma as fallback).
        # This captures genes like CFH/C3/PCSK9 whose mechanism is post-
        # translational (complement, secreted proteins, enzyme activity)
        # and are absent from Perturb-seq transcriptional programs.
        # --------------------------------------------------------------
        try:
            from pipelines.ota_beta_estimation import estimate_beta_tier2pt
            _gene_ot_score: float = float(disease_query.get("ot_genetic_scores", {}).get(gene, 0.0))
            _pt = estimate_beta_tier2pt(
                gene=gene,
                program="__protein_channel__",
                ot_instruments=ot_instruments_for_gene,
                pqtl_data=pqtl_data_for_gene,
                ot_score=_gene_ot_score,
            )
            if _pt is not None:
                gene_betas["__protein_channel__"] = {
                    "beta":          _pt["beta"],
                    "evidence_tier": _pt["evidence_tier"],
                    "beta_sigma":    _pt.get("beta_sigma"),
                    "program_gamma": _pt.get("program_gamma"),
                }
                # Upgrade tier tracking if Tier2pt fires and gene was virtual
                if _TIER_RANK.get(_pt["evidence_tier"], 99) < best_tier_rank:
                    best_tier_rank = _TIER_RANK[_pt["evidence_tier"]]
                    tier = _pt["evidence_tier"]
        except Exception as exc:
            warnings.append(f"{gene}: protein channel failed: {exc}")

        beta_matrix[gene] = gene_betas
        evidence_tier_per_gene[gene] = tier

    # ------------------------------------------------------------------
    # Consolidate tier counts from evidence_tier_per_gene
    # ------------------------------------------------------------------
    n_tier1 = sum(1 for t in evidence_tier_per_gene.values() if t == "Tier1_Interventional")
    n_tier2 = sum(1 for t in evidence_tier_per_gene.values() if t == "Tier2_Convergent")
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

    return {
        "genes":                  gene_list,
        "programs":               program_ids,
        "beta_matrix":            beta_matrix,
        "evidence_tier_per_gene": evidence_tier_per_gene,
        "n_tier1":                n_tier1,
        "n_tier2":                n_tier2,
        "n_tier3":                n_tier3,
        "n_virtual":              n_virtual,
        "warnings":               warnings,
        "state_space":            state_space,
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
        priority_cell = ctx["priority_cell"]
        h5ad_path = str(
            _CACHE_ROOT / disease_key / f"{disease_key}_{priority_cell.replace(' ', '_')}.h5ad"
        )
        from pathlib import Path
        if not Path(h5ad_path).exists():
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
