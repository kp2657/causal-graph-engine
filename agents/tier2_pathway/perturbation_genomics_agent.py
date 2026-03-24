"""
perturbation_genomics_agent.py — Tier 2 agent: β_{gene→program} estimation.

Uses the 4-tier fallback hierarchy (Perturb-seq → eQTL → GRN → virtual)
to populate the ProgramBetaMatrix for all genes in the target list.
"""
from __future__ import annotations

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

# Map lower-cased disease names → DISEASE_CELL_TYPE_MAP keys
_DISEASE_KEY_MAP: dict[str, str] = {
    "coronary artery disease": "CAD",
    "cad":                     "CAD",
    "inflammatory bowel disease": "IBD",
    "ibd":                     "IBD",
    "crohn's disease":         "IBD",
    "crohns disease":          "IBD",
    "ulcerative colitis":      "IBD",
    "rheumatoid arthritis":    "RA",
    "ra":                      "RA",
    "alzheimer's disease":     "AD",
    "type 2 diabetes":         "T2D",
}

# Tier rank for best-tier tracking (lower = better)
_TIER_RANK: dict[str, int] = {
    "Tier1_Interventional": 1,
    "Tier2_Convergent":     2,
    "Tier3_Provisional":    3,
    "provisional_virtual":  4,
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
        get_cnmf_program_info,
        get_program_gene_loadings,
    )
    from mcp_servers.gwas_genetics_server import query_gtex_eqtl
    from mcp_servers.open_targets_server import get_ot_genetic_instruments
    from pipelines.ota_beta_estimation import estimate_beta
    from graph.schema import DISEASE_CELL_TYPE_MAP

    programs_info = get_cnmf_program_info()
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
    # Determine disease-relevant GTEx tissue from DISEASE_CELL_TYPE_MAP
    # ------------------------------------------------------------------
    disease_name = disease_query.get("disease_name", "").lower()
    disease_key = _DISEASE_KEY_MAP.get(disease_name, "")
    ctx = DISEASE_CELL_TYPE_MAP.get(disease_key, {})
    gtex_tissue = ctx.get("gtex_tissue", "Whole_Blood")
    efo_id: str = disease_query.get("efo_id", "")

    # ------------------------------------------------------------------
    # Pre-fetch program gene loadings once per program (cached)
    # ------------------------------------------------------------------
    program_loadings_cache: dict[str, dict] = {}
    # program_gene_sets[pid] = set of gene symbols in that program (for pathway_member check)
    program_gene_sets: dict[str, set[str]] = {}
    for pid in program_ids:
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
    try:
        from pipelines.replogle_parser import load_replogle_betas, _H5AD_PATH
        if _H5AD_PATH.exists():
            prog_gene_sets_list = {pid: list(gset) for pid, gset in program_gene_sets.items()}
            if any(prog_gene_sets_list.values()):
                perturbseq_data = load_replogle_betas(program_gene_sets=prog_gene_sets_list)
    except Exception as exc:
        warnings.append(f"Replogle h5ad pre-load skipped: {exc}")

    for gene in gene_list:
        gene_betas: dict[str, float | None] = {pid: None for pid in program_ids}

        # --------------------------------------------------------------
        # Fetch eQTL data for this gene once (reused across all programs)
        # Supports both real GTEx format {"eqtls": [...]} and test mocks
        # that return {"data": [...]}.
        # --------------------------------------------------------------
        eqtl_data_for_gene: dict | None = None
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

        # --------------------------------------------------------------
        # Pre-fetch OT genetic instruments (Tier 2b): GWAS credible sets
        # and eQTL catalogue betas from Open Targets.
        # Only attempted when an EFO ID is available.
        # --------------------------------------------------------------
        ot_instruments_for_gene: dict | None = None
        if efo_id:
            try:
                ot_result = get_ot_genetic_instruments(gene, efo_id)
                if ot_result.get("instruments"):
                    ot_instruments_for_gene = ot_result
            except Exception as exc:
                warnings.append(f"{gene}: OT instruments prefetch failed: {exc}")

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
                    program_loading=loading,
                    pathway_member=pathway_member,
                )
                beta_val = single.get("beta")
                if beta_val is not None:
                    et = single.get("evidence_tier", "provisional_virtual")
                    gene_betas[pid] = {
                        "beta":          float(beta_val),
                        "evidence_tier": et,
                        "beta_sigma":    single.get("beta_sigma"),
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
    }
