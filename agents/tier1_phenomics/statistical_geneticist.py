"""
statistical_geneticist.py — Tier 1 agent: anchor gene eQTL validation and GWAS hit counting.

Validates known anchor genes via GTEx eQTL direction checks and counts
genome-wide-significant GWAS hits for the disease.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Anchor gene eQTL validation expectations — disease-specific
# "direction" = expected eQTL NES sign in the relevant tissue
_ANCHOR_GENE_EXPECTATIONS_BY_DISEASE: dict[str, dict[str, dict]] = {
    "CAD": {
        "PCSK9": {"direction": "negative", "tissue": "Liver",      "note": "LoF → lower LDL → lower CAD risk"},
        "LDLR":  {"direction": "positive", "tissue": "Liver",      "note": "LoF → higher LDL → higher CAD risk"},
        "HMGCR": {"direction": "positive", "tissue": "Liver",      "note": "Expression → LDL pathway (GTEx Liver eQTL)"},
        "IL6R":  {"direction": "negative", "tissue": "Whole_Blood","note": "rs2228145 → reduced IL-6R signaling → lower CAD"},
    },
    "SLE": {
        "TNFSF13B": {"direction": "positive", "tissue": "Whole_Blood", "note": "BAFF/BLyS elevated in SLE; belimumab target"},
        "IFNAR1":   {"direction": "positive", "tissue": "Whole_Blood", "note": "Type I IFN receptor; anifrolumab target (approved 2021)"},
        "IRF5":     {"direction": "positive", "tissue": "Whole_Blood", "note": "Transcription factor; rs2004640 coding variant associated with SLE"},
    },
    "RA": {
        "PTPN22": {"direction": "positive", "tissue": "Whole_Blood", "note": "R620W gain-of-function raises T cell activation threshold; strongest non-HLA RA locus"},
        "IL6R":   {"direction": "positive", "tissue": "Whole_Blood", "note": "IL-6 receptor drives synovitis; tocilizumab (approved 2010) blocks it"},
        "MS4A1":  {"direction": "positive", "tissue": "Whole_Blood", "note": "CD20 B cell surface marker; rituximab target (approved 2006 for RA)"},
    },
}
_ANCHOR_GENE_EXPECTATIONS_DEFAULT: dict[str, dict] = {
    "PCSK9":   {"direction": "negative", "tissue": "Liver",      "note": "LoF → lower LDL → lower CAD risk"},
    "LDLR":    {"direction": "positive", "tissue": "Liver",      "note": "LoF → higher LDL → higher CAD risk"},
    "IL6R":    {"direction": "negative", "tissue": "Whole_Blood","note": "rs2228145 → reduced IL-6R signaling"},
}



def _validate_anchor_gene_generic(gene: str, eqtl_nes: float | None, spec: dict) -> bool:
    """Check eQTL NES direction against a spec dict with 'direction' key."""
    if eqtl_nes is None:
        return True  # can't validate without data; not a failure
    expected = spec.get("direction", "unclear")
    if expected == "negative":
        return eqtl_nes < 0
    elif expected == "positive":
        return eqtl_nes > 0
    return True  # "unclear" → always pass


def _get_ot_tissue_for_gene(gene: str, disease_name: str) -> str:
    """
    Query Open Targets for the best GTEx tissue to use for a gene's eQTL validation.
    Falls back to Whole_Blood if OT has no tissue data.
    """
    try:
        from mcp_servers.open_targets_server import get_open_targets_target_info
        info = get_open_targets_target_info(gene)
        tissues = info.get("associated_tissues") or info.get("tissue_specificity") or []
        if isinstance(tissues, list) and tissues:
            # Prefer tissue names that map to GTEx
            gtex_map = {
                "liver": "Liver", "blood": "Whole_Blood", "colon": "Colon_Sigmoid",
                "pancreas": "Pancreas", "adipose": "Adipose_Subcutaneous",
                "heart": "Heart_Left_Ventricle", "brain": "Brain_Cortex",
                "lung": "Lung", "kidney": "Kidney_Cortex", "eye": "Retina",
                "retina": "Retina", "skin": "Skin_Sun_Exposed_Lower_leg",
            }
            for t in tissues:
                t_lower = str(t).lower()
                for key, gtex in gtex_map.items():
                    if key in t_lower:
                        return gtex
    except Exception:
        pass
    return "Whole_Blood"




def run(disease_query: dict) -> dict:
    """
    Validate anchor genes via GTEx eQTL and count GWAS significant hits.

    Args:
        disease_query: DiseaseQuery dict (output of phenotype_architect.run)

    Returns:
        dict with anchor_genes_validated, n_gw_significant_hits, warnings
    """
    from mcp_servers.gwas_genetics_server import (
        get_gwas_catalog_associations,
        query_gnomad_lof_constraint,
        query_gtex_eqtl,
    )
    efo_id = disease_query.get("efo_id")
    primary_outcome = disease_query.get("primary_gwas_id", "ieu-a-7")
    warnings: list[str] = []
    # Validate anchor genes via GTEx eQTL — tissue from OT for unknown diseases
    anchor_genes_validated: dict[str, bool] = {}
    disease_name = disease_query.get("disease_name", "")
    disease_key = disease_query.get("disease_key") or ""
    anchor_expectations = _ANCHOR_GENE_EXPECTATIONS_BY_DISEASE.get(
        disease_key, _ANCHOR_GENE_EXPECTATIONS_DEFAULT
    )
    for gene, spec in anchor_expectations.items():
        # Use hardcoded tissue if available; otherwise query OT
        tissue = spec.get("tissue") or _get_ot_tissue_for_gene(gene, disease_name)
        try:
            eqtl_result = query_gtex_eqtl(gene, tissue)
            top_nes = None
            eqtls = eqtl_result.get("eqtls") or eqtl_result.get("data") or eqtl_result.get("results") or []
            if isinstance(eqtls, list) and eqtls:
                top_nes = eqtls[0].get("nes") or eqtls[0].get("effect_size")
            anchor_genes_validated[gene] = _validate_anchor_gene_generic(gene, top_nes, spec)
            if not anchor_genes_validated[gene]:
                warnings.append(
                    f"{gene}: eQTL NES direction inconsistent with expected biology "
                    f"({spec['note']})"
                )
        except Exception as exc:
            warnings.append(f"{gene} GTEx lookup failed: {exc}")
            anchor_genes_validated[gene] = False

    # Get GW-significant hit count for the disease
    gw_hits: list[dict] = []
    if efo_id:
        try:
            assoc_result = get_gwas_catalog_associations(efo_id, page_size=100)
            raw = assoc_result.get("associations", [])
            gw_hits = [
                a for a in raw
                if (a.get("p_value_exponent") or 0) <= -8
            ]
        except Exception as exc:
            warnings.append(f"GWAS catalog associations failed: {exc}")

    # Constraint check for anchor genes
    try:
        constraint_res = query_gnomad_lof_constraint(list(anchor_expectations.keys()))
        genes_block = (
            constraint_res.get("genes", [])
            if isinstance(constraint_res, dict)
            else constraint_res
            if isinstance(constraint_res, list)
            else []
        )
        for c in genes_block:
            if not isinstance(c, dict):
                continue
            gene = c.get("symbol", "")
            pli = c.get("pLI") or c.get("pli")
            if pli is not None and pli > 0.9:
                warnings.append(
                    f"{gene}: pLI={pli:.2f} > 0.9 — essential gene; on-target toxicity risk"
                )
    except Exception as exc:
        warnings.append(f"gnomAD constraint lookup failed: {exc}")

    return {
        "anchor_genes_validated": anchor_genes_validated,
        "n_gw_significant_hits":  len(gw_hits),
        "warnings":               warnings,
    }
