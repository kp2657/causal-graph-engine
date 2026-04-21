"""
regulatory_genomics_agent.py — Tier 2 agent: eQTL, COLOC, and program overlap.

Identifies cis-eQTLs in disease-relevant tissues, checks colocalization
candidates, and maps genes to cNMF programs for β_{gene→program} prioritization.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.disease_registry import get_disease_key as _get_disease_key

_DEFAULT_TISSUE = "Whole_Blood"


def run(gene_list: list[str], disease_query: dict) -> dict:
    """
    Perform regulatory genomics characterization for a list of genes.

    Args:
        gene_list: Target gene list (output of Tier 1 agents)
        disease_query: DiseaseQuery dict

    Returns:
        dict with gene_eqtl_summary, gene_program_overlap, tier2_upgrades, warnings
    """
    from mcp_servers.gwas_genetics_server import query_gtex_eqtl, get_snp_associations
    from mcp_servers.burden_perturb_server import get_program_gene_loadings, get_cnmf_program_info
    from graph.schema import DISEASE_CELL_TYPE_MAP

    efo_id    = disease_query.get("efo_id", "")
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # Resolve disease-appropriate primary and secondary GTEx tissues
    # ------------------------------------------------------------------
    disease_name = disease_query.get("disease_name", "").lower()
    disease_key  = _get_disease_key(disease_name) or ""
    ctx          = DISEASE_CELL_TYPE_MAP.get(disease_key, {})
    primary_tissue     = ctx.get("gtex_tissue", _DEFAULT_TISSUE)
    secondary_tissues: list[str] = ctx.get("gtex_tissues_secondary", [])

    gene_eqtl_summary: dict[str, dict] = {}
    gene_program_overlap: dict[str, list[str]] = {}
    tier2_upgrades: list[str] = []

    # Load program definitions once
    programs_info = get_cnmf_program_info()
    raw_programs = programs_info.get("programs", [])
    program_ids = [
        p if isinstance(p, str) else (p.get("program_id") or p.get("name", ""))
        for p in raw_programs
    ]

    # Pre-load top genes for each program
    program_top_genes: dict[str, set[str]] = {}
    for pid in program_ids:
        try:
            loadings = get_program_gene_loadings(pid)
            top_genes = loadings.get("top_genes", [])
            gene_set: set[str] = set()
            for g in top_genes:
                gene_set.add(g if isinstance(g, str) else g.get("gene", ""))
            program_top_genes[pid] = gene_set
        except Exception as exc:
            warnings.append(f"Program loading failed for {pid}: {exc}")
            program_top_genes[pid] = set()

    for gene in gene_list:
        top_eqtl_p: float | None = None
        top_eqtl_nes: float | None = None
        coloc_candidate = False
        l2g_score: float | None = None
        used_tissue = primary_tissue

        # eQTL lookup — try primary tissue first, then secondary tissues
        tissues_to_try = [primary_tissue] + [t for t in secondary_tissues if t != primary_tissue]
        eqtl_found = False
        for tissue in tissues_to_try:
            try:
                eqtl_result = query_gtex_eqtl(gene, tissue)
                # `query_gtex_eqtl` returns canonical list under `eqtls`
                data = eqtl_result.get("eqtls") or eqtl_result.get("data") or eqtl_result.get("results") or []
                if isinstance(data, list) and data:
                    top = data[0]
                    nes = top.get("nes") if top.get("nes") is not None else top.get("effect_size")
                    if nes is not None:
                        top_eqtl_p   = (
                            top.get("pvalue")
                            if top.get("pvalue") is not None
                            else top.get("p_value") or top.get("pval_nominal")
                        )
                        top_eqtl_nes = nes
                        used_tissue  = tissue
                        eqtl_found   = True

                        # Check if top eQTL SNP is also a GWAS hit (COLOC candidate)
                        top_snp = top.get("variantId") or top.get("variant_id") or top.get("rsid") or top.get("snpId")
                        if top_snp and top_eqtl_p and top_eqtl_p < 1e-5:
                            try:
                                snp_assoc = get_snp_associations(top_snp)
                                traits = snp_assoc.get("associations", [])
                                for t_assoc in traits:
                                    p_exp = t_assoc.get("pvalue_exp", t_assoc.get("p_value_exponent", 0))
                                    if (p_exp or 0) <= -8:
                                        coloc_candidate = True
                                        break
                            except Exception:
                                pass  # SNP lookup is best-effort
                        break  # stop at first tissue with a valid eQTL
            except Exception as exc:
                warnings.append(f"{gene}: GTEx eQTL lookup failed in {tissue}: {exc}")

        gene_eqtl_summary[gene] = {
            "top_tissue":       used_tissue,
            "top_eqtl_p":       top_eqtl_p,
            "top_eqtl_nes":     top_eqtl_nes,
            "coloc_candidate":  coloc_candidate,
            "l2g_score":        l2g_score,
        }

        # Validation: PCSK9 must have a significant Liver eQTL when Liver is a relevant tissue
        # (PCSK9 is hepatically synthesised; Liver is primary tissue for lipid/CAD genes)
        if gene == "PCSK9" and "Liver" in tissues_to_try:
            liver_ok = used_tissue == "Liver" and top_eqtl_p is not None and top_eqtl_p < 1e-5
            if not liver_ok:
                warnings.append(
                    "PCSK9 Liver eQTL not significant — check API or gene ID"
                )

        # Program overlap
        programs_containing_gene: list[str] = []
        for pid, top_genes in program_top_genes.items():
            if gene in top_genes:
                programs_containing_gene.append(pid)
        gene_program_overlap[gene] = programs_containing_gene

        # L2G score lookup: high-confidence causal gene at GWAS locus (OT Platform v4)
        if efo_id:
            try:
                from mcp_servers.gwas_genetics_server import get_l2g_scores
                primary_gwas_id = disease_query.get("primary_gwas_id", "")
                if primary_gwas_id:
                    l2g_result = get_l2g_scores(primary_gwas_id, top_n=20)
                    for l2g_rec in l2g_result.get("l2g_genes", []):
                        if l2g_rec.get("gene_symbol", "").upper() == gene.upper():
                            l2g_score = l2g_rec.get("l2g_score")
                            gene_eqtl_summary[gene]["l2g_score"] = l2g_score
                            break
            except Exception:
                pass  # L2G lookup is best-effort

        # Tier 2 upgrade: eQTL significant + COLOC candidate, OR L2G score ≥ 0.5
        if (coloc_candidate and top_eqtl_p and top_eqtl_p < 1e-5) or (
            (gene_eqtl_summary.get(gene, {}).get("l2g_score") or 0) >= 0.5
        ):
            tier2_upgrades.append(gene)

    return {
        "gene_eqtl_summary":  gene_eqtl_summary,
        "gene_program_overlap": gene_program_overlap,
        "tier2_upgrades":     tier2_upgrades,
        "warnings":           warnings,
    }
