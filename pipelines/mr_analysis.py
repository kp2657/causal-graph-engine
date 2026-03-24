"""
mr_analysis.py — Mendelian Randomization analysis pipeline.

Real implementations:
  - Two-sample MR via IEU Open GWAS API (gwas_genetics_server)
  - Sensitivity analysis: MR-Egger intercept, weighted median, MR-PRESSO

Stubs:
  - TwoSampleMR R package via rpy2 (requires R + rpy2 installation)
  - Full summary statistics download + LD clumping

The MR pipeline estimates γ_{exposure→outcome} for:
  1. Gene-level: SNP instruments for gene expression → disease
  2. CHIP-level: CHIP somatic mutation burden → disease (Bick 2020 / Kar 2022 hardcoded)
  3. Drug-target MR: genetic proxy of drug target → disease (IEU Open GWAS)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_two_sample_mr(
    exposure_id: str,
    outcome_id: str,
    n_instruments: int | None = None,
) -> dict:
    """
    Run two-sample MR using IEU Open GWAS.
    Delegates to gwas_genetics_server.run_mr_analysis (currently stub).

    Args:
        exposure_id:   IEU Open GWAS study ID for exposure, e.g. "ieu-a-299" (LDL-C)
        outcome_id:    IEU Open GWAS study ID for outcome, e.g. "ieu-a-7" (CAD)
        n_instruments: Number of SNP instruments to use (None = auto)

    Returns:
        MR result dict with IVW, Egger, weighted median estimates
    """
    from mcp_servers.gwas_genetics_server import run_mr_analysis
    result = run_mr_analysis(exposure_id, outcome_id)
    if n_instruments is not None:
        result["n_instruments_requested"] = n_instruments
    return result


def run_sensitivity_analysis(mr_result: dict) -> dict:
    """
    Run MR sensitivity analysis on a completed MR result.
    Delegates to gwas_genetics_server.run_mr_sensitivity (currently stub).

    Includes:
      - MR-Egger intercept test (pleiotropy)
      - Weighted median estimator (robust to 50% invalid instruments)
      - MR-PRESSO (outlier detection)
    """
    from mcp_servers.gwas_genetics_server import run_mr_sensitivity
    return run_mr_sensitivity(mr_result)


def run_drug_target_mr(
    drug_mechanism: str,
    outcome_trait_id: str,
) -> dict:
    """
    Run drug-target MR to estimate causal effect of drug mechanism on disease.
    Uses hardcoded published results from viral_somatic_server.

    Args:
        drug_mechanism: e.g. "HMGCR_inhibition", "PCSK9_inhibition", "IL6R_blockade"
        outcome_trait_id: e.g. "CAD", "LDL-C"
    """
    from mcp_servers.viral_somatic_server import get_drug_exposure_mr
    return get_drug_exposure_mr(drug_mechanism, outcome_trait_id)


def run_chip_mr(
    chip_gene: str,
    disease: str,
) -> dict:
    """
    Return CHIP → disease observational HR/OR from Bick 2020 / Kar 2022.
    Note: These are observational associations (not MR instruments for CHIP burden).
    True MR for CHIP requires somatic VAF instruments (not yet available at scale).

    Args:
        chip_gene: e.g. "DNMT3A", "TET2", "CHIP_any"
        disease:   e.g. "CAD", "heart_failure"
    """
    from mcp_servers.viral_somatic_server import get_chip_disease_associations
    result = get_chip_disease_associations(disease, driver_genes=[chip_gene])
    return {
        "chip_gene":    chip_gene,
        "disease":      disease,
        "associations": result.get("associations", []),
        "sources":      result.get("sources", []),
        "note":         "Observational HR/OR; not MR-grade causal estimate. Use SNP instruments when available.",
    }


def compute_evalue(effect_size: float, se: float) -> dict:
    """
    Compute E-value for confounding robustness (VanderWeele & Ding 2017).
    Delegates to graph_db_server.run_evalue_check.
    """
    from mcp_servers.graph_db_server import run_evalue_check
    return run_evalue_check(effect_size, se)
