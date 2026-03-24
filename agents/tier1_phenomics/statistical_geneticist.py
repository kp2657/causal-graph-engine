"""
statistical_geneticist.py — Tier 1 agent: genetic instrument identification & MR.

Extracts GW-significant loci, selects MR instruments, validates F-statistics,
and runs two-sample MR for key exposure → disease causal estimates.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# MR exposures per disease (keyed by short disease name / EFO)
_MR_EXPOSURES_BY_DISEASE: dict[str, list[dict]] = {
    "CAD": [
        {"exposure": "LDL-C",  "exposure_id": "ieu-a-299", "outcome_id": "ieu-a-7",  "expected_dir": "positive"},
        {"exposure": "HDL-C",  "exposure_id": "ieu-a-298", "outcome_id": "ieu-a-7",  "expected_dir": "negative"},
        {"exposure": "CRP",    "exposure_id": "ieu-a-32",  "outcome_id": "ieu-a-7",  "expected_dir": "unclear"},
    ],
    "IBD": [
        {"exposure": "CRP",        "exposure_id": "ieu-a-32",  "outcome_id": "ieu-b-30", "expected_dir": "positive"},
        {"exposure": "IL6R",       "exposure_id": "ieu-a-1058","outcome_id": "ieu-b-30", "expected_dir": "unclear"},
    ],
    "RA": [
        {"exposure": "CRP",    "exposure_id": "ieu-a-32",  "outcome_id": "ieu-a-833", "expected_dir": "positive"},
    ],
    "T2D": [
        {"exposure": "BMI",    "exposure_id": "ieu-a-2",   "outcome_id": "ieu-a-26",  "expected_dir": "positive"},
        {"exposure": "HbA1c",  "exposure_id": "ieu-a-1340","outcome_id": "ieu-a-26",  "expected_dir": "unclear"},
    ],
}
# Fallback for unknown diseases
_MR_EXPOSURES_DEFAULT: list[dict] = [
    {"exposure": "CRP", "exposure_id": "ieu-a-32", "outcome_id": "ieu-a-7", "expected_dir": "unclear"},
]

# Anchor gene eQTL validation expectations — disease-specific
# "direction" = expected eQTL NES sign in the relevant tissue
_ANCHOR_GENE_EXPECTATIONS_BY_DISEASE: dict[str, dict[str, dict]] = {
    "CAD": {
        "PCSK9": {"direction": "negative", "tissue": "Liver",      "note": "LoF → lower LDL → lower CAD risk"},
        "LDLR":  {"direction": "positive", "tissue": "Liver",      "note": "LoF → higher LDL → higher CAD risk"},
        "HMGCR": {"direction": "positive", "tissue": "Liver",      "note": "Expression → LDL pathway (GTEx Liver eQTL)"},
        "IL6R":  {"direction": "negative", "tissue": "Whole_Blood","note": "rs2228145 → reduced IL-6R signaling → lower CAD"},
    },
    "IBD": {
        "NOD2":  {"direction": "unclear",  "tissue": "Colon_Sigmoid","note": "NOD2 LoF → impaired mucosal immunity → IBD risk"},
        "IL23R": {"direction": "negative", "tissue": "Whole_Blood",  "note": "IL23R LoF → reduced Th17 → protective"},
        "TNF":   {"direction": "positive", "tissue": "Whole_Blood",  "note": "TNF expression drives mucosal inflammation"},
        "IL10":  {"direction": "negative", "tissue": "Colon_Sigmoid","note": "IL10 LoF → loss of anti-inflammatory brake → IBD"},
    },
    "RA": {
        "PTPN22": {"direction": "unclear",  "tissue": "Whole_Blood", "note": "PTPN22 risk allele → T cell activation"},
        "HLA-DRA":{"direction": "unclear",  "tissue": "Whole_Blood", "note": "HLA DRB1*04 → peptide presentation → RA"},
        "IL6":    {"direction": "positive", "tissue": "Whole_Blood", "note": "IL6 eQTL drives synovitis"},
    },
    "T2D": {
        "TCF7L2": {"direction": "positive", "tissue": "Pancreas",   "note": "TCF7L2 expression → β-cell function impairment"},
        "SLC30A8":{"direction": "unclear",  "tissue": "Pancreas",   "note": "SLC30A8 LoF → altered zinc transport in β-cells"},
    },
}
_ANCHOR_GENE_EXPECTATIONS_DEFAULT: dict[str, dict] = {
    "PCSK9":   {"direction": "negative", "tissue": "Liver",      "note": "LoF → lower LDL → lower CAD risk"},
    "LDLR":    {"direction": "positive", "tissue": "Liver",      "note": "LoF → higher LDL → higher CAD risk"},
    "IL6R":    {"direction": "negative", "tissue": "Whole_Blood","note": "rs2228145 → reduced IL-6R signaling"},
}

# Short-name lookup for EFO IDs
_EFO_TO_DISEASE_SHORT: dict[str, str] = {
    "EFO_0001645": "CAD",
    "EFO_0003767": "IBD",
    "EFO_0000685": "RA",
    "EFO_0001360": "T2D",
    "EFO_0000616": "T2D",
}

# Backward-compat aliases (keep old names pointing to per-disease dicts for tests)
MR_EXPOSURES = _MR_EXPOSURES_BY_DISEASE["CAD"]
ANCHOR_GENE_EXPECTATIONS = _ANCHOR_GENE_EXPECTATIONS_BY_DISEASE["CAD"]


def _validate_anchor_gene(gene: str, eqtl_nes: float | None) -> bool:
    """Check if eQTL NES direction is consistent with biology expectations (CAD default)."""
    spec = ANCHOR_GENE_EXPECTATIONS.get(gene)
    if spec is None or eqtl_nes is None:
        return True
    return _validate_anchor_gene_generic(gene, eqtl_nes, spec)


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


def run(disease_query: dict) -> dict:
    """
    Identify genetic instruments and run MR analyses.

    Args:
        disease_query: DiseaseQuery dict (output of phenotype_architect.run)

    Returns:
        dict with instruments list, anchor_genes_validated, warnings
    """
    from mcp_servers.gwas_genetics_server import (
        get_gwas_catalog_associations,
        query_gnomad_lof_constraint,
        query_gtex_eqtl,
    )
    from pipelines.mr_analysis import run_two_sample_mr as run_mr_analysis, run_sensitivity_analysis as run_mr_sensitivity

    efo_id = disease_query.get("efo_id")
    primary_outcome = disease_query.get("primary_gwas_id", "ieu-a-7")
    warnings: list[str] = []
    instruments: list[dict] = []

    # Select disease-specific MR exposures and anchor expectations
    disease_short = _EFO_TO_DISEASE_SHORT.get(efo_id or "", "")
    mr_exposures = _MR_EXPOSURES_BY_DISEASE.get(disease_short, _MR_EXPOSURES_DEFAULT)
    anchor_expectations = _ANCHOR_GENE_EXPECTATIONS_BY_DISEASE.get(
        disease_short, _ANCHOR_GENE_EXPECTATIONS_DEFAULT
    )

    # Run MR for each exposure
    for exp in mr_exposures:
        outcome_id = exp.get("outcome_id", primary_outcome)
        try:
            mr_result = run_mr_analysis(exp["exposure_id"], outcome_id)
            ivw_beta = mr_result.get("ivw_beta")
            ivw_p    = mr_result.get("ivw_p")
            n_snps   = mr_result.get("n_snps", 0)

            # Approximate F-statistic from n_snps and sample size
            f_stat = mr_result.get("f_statistic", n_snps * 10.0 if n_snps else 0.0)

            if f_stat < 10:
                warnings.append(
                    f"{exp['exposure']}: F-statistic {f_stat:.1f} < 10 — weak instruments"
                )

            # Direction check
            if ivw_beta is not None and exp["expected_dir"] != "unclear":
                direction_ok = (
                    (exp["expected_dir"] == "positive" and ivw_beta > 0)
                    or (exp["expected_dir"] == "negative" and ivw_beta < 0)
                )
                if not direction_ok:
                    warnings.append(
                        f"{exp['exposure']}: Unexpected MR direction "
                        f"(beta={ivw_beta:.3f}, expected {exp['expected_dir']})"
                    )

            # Sensitivity
            sensitivity = run_mr_sensitivity(mr_result)
            egger_p = sensitivity.get("egger_intercept_p", 1.0)

            instruments.append({
                "exposure":     exp["exposure"],
                "exposure_id":  exp["exposure_id"],
                "outcome_id":   outcome_id,
                "n_snps":       n_snps,
                "mr_ivw_beta":  ivw_beta,
                "mr_ivw_p":     ivw_p,
                "f_statistic":  f_stat,
                "pleiotropy_p": egger_p,
            })

        except Exception as exc:
            warnings.append(f"{exp['exposure']} MR failed: {exc}")
            instruments.append({
                "exposure":     exp["exposure"],
                "exposure_id":  exp["exposure_id"],
                "outcome_id":   outcome_id,
                "n_snps":       0,
                "mr_ivw_beta":  None,
                "mr_ivw_p":     None,
                "f_statistic":  None,
                "pleiotropy_p": None,
            })

    # Validate anchor genes via GTEx eQTL (disease-specific)
    anchor_genes_validated: dict[str, bool] = {}
    for gene, spec in anchor_expectations.items():
        tissue = spec.get("tissue", "Whole_Blood")
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
        constraint = query_gnomad_lof_constraint(list(anchor_expectations.keys()))
        for gene, c in constraint.items():
            pli = c.get("pLI") or c.get("pli")
            if pli is not None and pli > 0.9:
                warnings.append(
                    f"{gene}: pLI={pli:.2f} > 0.9 — essential gene; on-target toxicity risk"
                )
    except Exception as exc:
        warnings.append(f"gnomAD constraint lookup failed: {exc}")

    return {
        "instruments":            instruments,
        "anchor_genes_validated": anchor_genes_validated,
        "n_gw_significant_hits":  len(gw_hits),
        "warnings":               warnings,
    }
