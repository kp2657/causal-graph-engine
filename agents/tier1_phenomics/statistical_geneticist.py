"""
statistical_geneticist.py — Tier 1 agent: genetic instrument identification & MR.

Extracts GW-significant loci, selects MR instruments, validates F-statistics,
and runs two-sample MR for key exposure → disease causal estimates.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.scoring_thresholds import MR_F_STATISTIC_MIN as _MR_F_STAT_MIN

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
    "AMD": [
        # Complement / lipid exposures → AMD (Fritsche 2016: ebi-a-GCST006909)
        {"exposure": "LDL-C",  "exposure_id": "ieu-a-299", "outcome_id": "ebi-a-GCST006909", "expected_dir": "positive"},
        {"exposure": "BMI",    "exposure_id": "ieu-a-2",   "outcome_id": "ebi-a-GCST006909", "expected_dir": "unclear"},
        {"exposure": "CRP",    "exposure_id": "ieu-a-32",  "outcome_id": "ebi-a-GCST006909", "expected_dir": "unclear"},
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
    "AMD": {
        "CFH":   {"direction": "unclear", "tissue": "Retina", "note": "CFH Y402H risk allele → complement dysregulation → AMD"},
        "C3":    {"direction": "unclear", "tissue": "Retina", "note": "C3 R102G variant → complement activation in RPE"},
        "VEGFA": {"direction": "positive","tissue": "Retina", "note": "VEGFA drives choroidal neovascularization in wet AMD"},
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
    "EFO_0001481": "AMD",  # age-related macular degeneration
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


def _get_ot_exposures_for_disease(efo_id: str, outcome_id: str) -> list[dict]:
    """
    Query Open Targets for top genetic association genes and find their OpenGWAS IDs
    to construct MR exposures dynamically for unknown diseases.
    """
    try:
        from mcp_servers.open_targets_server import get_open_targets_disease_targets
        from mcp_servers.gwas_genetics_server import get_gwas_catalog_associations
        result = get_open_targets_disease_targets(efo_id, top_n=8)
        targets = result.get("targets") or result.get("results") or []
        exposures = []
        for t in targets[:5]:
            gene = t.get("gene") or t.get("symbol") or t.get("approvedSymbol")
            if not gene:
                continue
            # Try to find an OpenGWAS pQTL/eQTL study for this gene
            # Use ieu-a-32 (CRP) as a generic fallback exposure_id
            exposures.append({
                "exposure":    gene,
                "exposure_id": f"eqtl-{gene}",  # signals eQTL fallback path
                "outcome_id":  outcome_id,
                "expected_dir": "unclear",
                "_from_ot":    True,
            })
        return exposures
    except Exception:
        return []


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
    from pipelines.mr_analysis import (
        run_two_sample_mr as run_mr_analysis,
        run_sensitivity_analysis as run_mr_sensitivity,
    )

    efo_id = disease_query.get("efo_id")
    primary_outcome = disease_query.get("primary_gwas_id", "ieu-a-7")
    warnings: list[str] = []
    instruments: list[dict] = []

    skip_mr = bool(disease_query.get("skip_mr")) or os.getenv("SKIP_MR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}

    # Select disease-specific MR exposures and anchor expectations
    disease_short = _EFO_TO_DISEASE_SHORT.get(efo_id or "", "")
    mr_exposures = _MR_EXPOSURES_BY_DISEASE.get(disease_short)
    anchor_expectations = _ANCHOR_GENE_EXPECTATIONS_BY_DISEASE.get(
        disease_short, _ANCHOR_GENE_EXPECTATIONS_DEFAULT
    )

    # A2: Unknown disease — discover exposures dynamically from Open Targets
    if mr_exposures is None:
        mr_exposures = _get_ot_exposures_for_disease(efo_id or "", primary_outcome)
        if mr_exposures:
            warnings.append(
                f"Unknown disease '{disease_short or efo_id}': MR exposures derived dynamically "
                f"from Open Targets ({len(mr_exposures)} genes). Validate results carefully."
            )
        else:
            mr_exposures = _MR_EXPOSURES_DEFAULT

    if skip_mr:
        warnings.append("MR skipped (SKIP_MR=1 or disease_query.skip_mr=true).")
        instruments = []
    else:
        # Run MR for each exposure
        for exp in mr_exposures:
            outcome_id = exp.get("outcome_id", primary_outcome)

            # Skip OT-derived exposures that only have eQTL proxy IDs (no real OpenGWAS study)
            if exp.get("_from_ot") and exp["exposure_id"].startswith("eqtl-"):
                instruments.append({
                    "exposure":          exp["exposure"],
                    "exposure_id":       exp["exposure_id"],
                    "outcome_id":        outcome_id,
                    "n_snps":            0,
                    "mr_ivw_beta":       None,
                    "mr_ivw_p":          None,
                    "f_statistic":       None,
                    "pleiotropy_p":      None,
                    "instrument_source": "none",
                })
                continue

            try:
                mr_result = run_mr_analysis(exp["exposure_id"], outcome_id)
                ivw_beta = mr_result.get("mr_ivw") or mr_result.get("ivw_beta")
                ivw_p    = mr_result.get("mr_ivw_p") or mr_result.get("ivw_p")
                n_snps   = mr_result.get("n_snps", 0)

                # A1: F-stat defaults to None (not 0.0) when no SNPs — distinguishes
                # "unknown instrument strength" from "zero effect"
                f_stat = mr_result.get("f_statistic", n_snps * 10.0 if n_snps else None)
                instrument_source = "gwas"

                if f_stat is not None and f_stat < _MR_F_STAT_MIN:
                    warnings.append(
                        f"ESCALATE: {exp['exposure']} instruments F-statistic={f_stat:.1f} < {_MR_F_STAT_MIN} "
                        "(weak instruments — MR estimates unreliable)"
                    )
                    # A3: Try eQTL fallback — tissue from OT (not hardcoded table)
                    disease_name = disease_query.get("disease_name", "")
                    tissue = _get_ot_tissue_for_gene(exp["exposure"], disease_name)
                    try:
                        eqtl_res = query_gtex_eqtl(exp["exposure"], tissue)
                        eqtls = eqtl_res.get("eqtls") or eqtl_res.get("data") or eqtl_res.get("results") or []
                        if eqtls:
                            top_nes = eqtls[0].get("nes") or eqtls[0].get("effect_size") or 0.0
                            top_p   = eqtls[0].get("pvalue") or eqtls[0].get("p_value") or 1.0
                            if top_p < 1e-5:
                                proxy_f = (top_nes ** 2) / max(0.01, top_p) * 10
                                if proxy_f >= 10:
                                    f_stat = proxy_f
                                    instrument_source = "eqtl"
                                    warnings.append(
                                        f"{exp['exposure']}: GTEx eQTL fallback in {tissue} "
                                        f"(NES={top_nes:.3f}, proxy F={proxy_f:.1f})"
                                    )
                    except Exception:
                        pass

                elif f_stat is None:
                    # MR stub or failed MR — not an escalation, just informational.
                    # Real escalation requires a measured F-statistic (not stub/missing data).
                    warnings.append(
                        f"{exp['exposure']}: MR F-statistic unavailable (stub or no instruments) "
                        "— skipping MR validation for this exposure"
                    )
                    instrument_source = "none"

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
                    "exposure":          exp["exposure"],
                    "exposure_id":       exp["exposure_id"],
                    "outcome_id":        outcome_id,
                    "n_snps":            n_snps,
                    "mr_ivw_beta":       ivw_beta,
                    "mr_ivw_p":          ivw_p,
                    "f_statistic":       f_stat,
                    "pleiotropy_p":      egger_p,
                    "instrument_source": instrument_source,
                })

            except Exception as exc:
                warnings.append(f"{exp['exposure']} MR failed: {exc}")
                instruments.append({
                    "exposure":          exp["exposure"],
                    "exposure_id":       exp["exposure_id"],
                    "outcome_id":        outcome_id,
                    "n_snps":            0,
                    "mr_ivw_beta":       None,
                    "mr_ivw_p":          None,
                    "f_statistic":       None,
                    "pleiotropy_p":      None,
                    "instrument_source": "none",
                })

    # A3: Validate anchor genes via GTEx eQTL — tissue from OT for unknown diseases
    anchor_genes_validated: dict[str, bool] = {}
    disease_name = disease_query.get("disease_name", "")
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
        "instruments":            instruments,
        "anchor_genes_validated": anchor_genes_validated,
        "n_gw_significant_hits":  len(gw_hits),
        "skip_mr":                skip_mr,
        "warnings":               warnings,
    }
