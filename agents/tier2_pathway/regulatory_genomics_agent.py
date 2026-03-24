"""
regulatory_genomics_agent.py — Tier 2 agent: eQTL, COLOC, and program overlap.

Identifies cis-eQTLs in disease-relevant tissues, checks colocalization
candidates, and maps genes to cNMF programs for β_{gene→program} prioritization.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# CAD tissue priority per gene class
CAD_TISSUE_MAP: dict[str, str] = {
    "PCSK9":   "Liver",
    "LDLR":    "Liver",
    "HMGCR":   "Liver",
    "IL6R":    "Liver",
    "TET2":    "Whole_Blood",
    "DNMT3A":  "Whole_Blood",
    "ASXL1":   "Whole_Blood",
    "JAK2":    "Whole_Blood",
    "HLA-DRA": "Whole_Blood",
    "CIITA":   "Whole_Blood",
}
DEFAULT_TISSUE = "Whole_Blood"


def _pick_tissue(gene: str, efo_id: str) -> str:
    return CAD_TISSUE_MAP.get(gene, DEFAULT_TISSUE)


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

    efo_id    = disease_query.get("efo_id", "")
    warnings: list[str] = []

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
        tissue = _pick_tissue(gene, efo_id)
        top_eqtl_p: float | None = None
        top_eqtl_nes: float | None = None
        coloc_candidate = False
        l2g_score: float | None = None

        # eQTL lookup
        try:
            eqtl_result = query_gtex_eqtl(gene, tissue)
            data = eqtl_result.get("data") or eqtl_result.get("results") or []
            if isinstance(data, list) and data:
                top = data[0]
                top_eqtl_p   = top.get("p_value") or top.get("pval_nominal")
                top_eqtl_nes = top.get("nes") or top.get("effect_size")

                # Check if top eQTL SNP is also a GWAS hit (COLOC candidate)
                top_snp = top.get("variant_id") or top.get("rsid")
                if top_snp and top_eqtl_p and top_eqtl_p < 1e-5:
                    try:
                        snp_assoc = get_snp_associations(top_snp)
                        traits = snp_assoc.get("associations", [])
                        for t in traits:
                            p_exp = t.get("p_value_exponent", 0)
                            if (p_exp or 0) <= -8:
                                coloc_candidate = True
                                break
                    except Exception:
                        pass  # SNP lookup is best-effort

            gene_eqtl_summary[gene] = {
                "top_tissue":       tissue,
                "top_eqtl_p":       top_eqtl_p,
                "top_eqtl_nes":     top_eqtl_nes,
                "coloc_candidate":  coloc_candidate,
                "l2g_score":        l2g_score,
            }

        except Exception as exc:
            warnings.append(f"{gene}: GTEx eQTL lookup failed: {exc}")
            gene_eqtl_summary[gene] = {
                "top_tissue":      tissue,
                "top_eqtl_p":      None,
                "top_eqtl_nes":    None,
                "coloc_candidate": False,
                "l2g_score":       None,
            }

        # Special validation: PCSK9 liver eQTL must be significant
        if gene == "PCSK9" and tissue == "Liver":
            p = gene_eqtl_summary[gene]["top_eqtl_p"]
            if p is None or p >= 1e-5:
                warnings.append(
                    "PCSK9 Liver eQTL not significant — check API or gene ID"
                )

        # Program overlap
        programs_containing_gene: list[str] = []
        for pid, top_genes in program_top_genes.items():
            if gene in top_genes:
                programs_containing_gene.append(pid)
        gene_program_overlap[gene] = programs_containing_gene

        # Tier 2 upgrade: eQTL significant + COLOC candidate
        if coloc_candidate and top_eqtl_p and top_eqtl_p < 1e-5:
            tier2_upgrades.append(gene)

    return {
        "gene_eqtl_summary":  gene_eqtl_summary,
        "gene_program_overlap": gene_program_overlap,
        "tier2_upgrades":     tier2_upgrades,
        "warnings":           warnings,
    }
