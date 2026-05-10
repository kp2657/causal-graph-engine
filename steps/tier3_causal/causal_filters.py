"""
causal_filters.py — Utility functions for Tier 3 causal ranking and filtering.

Extracted from ota_gamma_calculator.py to keep that module focused on the
main run() orchestration. Functions here:
  - _beta_stress_discount    — detect essential-gene stress artefacts
  - _program_entropy         — Shannon entropy of β-loading across programs
  - _mechanistic_necessity_filter — prune genes with no program evidence
  - _extract_beta_for_program — safe β extraction from tiered beta_matrix shapes
  - _extract_gamma_for_trait  — safe γ extraction from multi-form gamma_estimates
  - _pareto_cutoff            — empiric elbow cutoff on sorted |γ| list
  - _HOUSEKEEPING_PREFIXES / _HOUSEKEEPING_EXACT — non-specific gene sets
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math as _math

from config.scoring_thresholds import (
    CAUSAL_STRESS_MEAN_THRESHOLD  as _STRESS_MEAN_THRESHOLD,
    CAUSAL_STRESS_CV_FLOOR        as _STRESS_CV_FLOOR,
    CAUSAL_STRESS_DISCOUNT        as _STRESS_DISCOUNT,
    PARETO_ELBOW_GAP_THRESHOLD,
    WES_CONCORDANCE_P_THRESHOLD,
    WES_CONCORDANT_BOOST,
)

# Housekeeping gene prefixes: ribosomal proteins and histones have very large
# Perturb-seq β values (global translation failure / chromatin disruption) that
# are biologically non-specific and not therapeutically actionable.
_HOUSEKEEPING_PREFIXES = ("RPL", "RPS", "HIST", "H1-", "H2A", "H2B", "H3-", "H4-")
_HOUSEKEEPING_EXACT = frozenset({
    "UBA52", "UBB", "UBC", "UBD",           # ubiquitin
    "ACTB", "ACTG1",                          # cytoskeletal
    "GAPDH", "LDHA",                          # glycolysis
    "B2M",                                    # MHC presentation component
})


def _beta_stress_discount(gene_beta: dict) -> float:
    """
    Return a gamma multiplier [0.25, 1.0] to penalise essential-gene cell-stress
    artefacts in Perturb-seq data.

    When a gene's KO causes non-specific transcriptional stress (all programs
    perturbed at similar large magnitude), its OTA gamma is inflated and not
    disease-pathway specific.  Criterion:
      mean(|beta|) > _STRESS_MEAN_THRESHOLD  AND
      stdev(|beta|) / mean(|beta|) < _STRESS_CV_FLOOR  (low CV = uniform response)

    Returns _STRESS_DISCOUNT (0.25) when both conditions are met, else 1.0.
    """
    import math
    vals: list[float] = []
    for v in gene_beta.values():
        raw = v.get("beta") if isinstance(v, dict) else v
        try:
            f = float(raw)
            if math.isfinite(f):
                vals.append(abs(f))
        except (TypeError, ValueError):
            pass

    if len(vals) < 3:
        return 1.0

    mean_abs = sum(vals) / len(vals)
    if mean_abs <= _STRESS_MEAN_THRESHOLD:
        return 1.0

    variance = sum((v - mean_abs) ** 2 for v in vals) / len(vals)
    cv = (variance ** 0.5) / mean_abs if mean_abs > 0 else 1.0
    if cv < _STRESS_CV_FLOOR:
        return _STRESS_DISCOUNT
    return 1.0


def _program_entropy(gene: str, beta_matrix: dict, disease_programs: frozenset) -> float:
    """Shannon entropy of gene's β-loading across disease-relevant programs.

    High entropy → gene loads broadly across all programs (housekeeping / non-specific).
    Low entropy → gene concentrated in a few disease programs (likely a causal driver).
    Returns 0.0 when no disease-program betas are available.
    """
    import math as _m
    import numpy as _np
    weights = []
    for prog in disease_programs:
        b = (beta_matrix.get(gene) or {}).get(prog)
        if b is None:
            continue
        bval = b if isinstance(b, (int, float)) else (b.get("beta") or 0.0)
        try:
            bval = float(bval)
        except (TypeError, ValueError):
            continue
        if _m.isfinite(bval):
            weights.append(abs(bval))
    if not weights or sum(weights) == 0:
        return 0.0
    p = _np.array(weights) / sum(weights)
    return float(-_np.sum(p * _np.log(p + 1e-12)))


def _mechanistic_necessity_filter(
    genes: list,
    beta_matrix: dict,
    disease_programs: frozenset,
    beta_threshold: float = 0.05,
    min_keep: int = 50,
) -> list:
    """Hard-remove genes that have GWAS support but β≈0 and no program overlap.

    Exemptions (always kept):
      - dominant_tier starts with "Tier2"  (strong genetic or pQTL tier)
    Keeps a minimum of `min_keep` genes regardless of filter result.
    """
    def _is_exempt(r: dict) -> bool:
        if (r.get("dominant_tier") or "").startswith("Tier2"):
            return True
        return False

    def _has_mechanistic(r: dict) -> bool:
        gene = r["gene"]
        for prog in disease_programs:
            b = (beta_matrix.get(gene) or {}).get(prog)
            if b is None:
                continue
            bval = b if isinstance(b, (int, float)) else (b.get("beta") or 0.0)
            try:
                if abs(float(bval)) > beta_threshold:
                    return True
            except (TypeError, ValueError):
                pass
        tp = r.get("top_programs") or r.get("programs")
        return isinstance(tp, dict) and bool(tp)

    kept = [r for r in genes if _is_exempt(r) or _has_mechanistic(r)]
    return kept if len(kept) >= min_keep else genes[:min_keep]


def _extract_beta_for_program(beta_matrix: dict, gene: str, program: str) -> float | None:
    """Pull the β for (gene, program) out of the tiered beta_matrix shapes used
    by the pipeline.  Returns None when the value is missing or non-finite."""
    import math as _m
    gene_beta = beta_matrix.get(gene, {}) or {}
    raw = gene_beta.get(program)
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = raw.get("beta")
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return None
    return f if _m.isfinite(f) else None


def _extract_gamma_for_trait(prog_gammas, trait: str) -> float | None:
    """Normalise gamma_estimates[program] shape (scalar | trait-keyed dict |
    dict-with-gamma) into a single scalar γ for the given trait."""
    import math as _m
    if prog_gammas is None:
        return None
    # Scalar form
    if isinstance(prog_gammas, (int, float)):
        f = float(prog_gammas)
        return f if _m.isfinite(f) else None
    if not isinstance(prog_gammas, dict):
        return None
    # Trait-keyed dict: first try the specific trait, then the "gamma" shorthand
    val = prog_gammas.get(trait)
    if val is None:
        val = prog_gammas.get("gamma")
    if isinstance(val, dict):
        val = val.get("gamma")
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if _m.isfinite(f) else None



# Genes where CRISPRi knockdown and LoF rare-variant burden are known to
# measure different causal mechanisms — discordance is biologically expected
# and should not be penalised.  Key: gene symbol. Value: explanation.
_EXPECTED_CRISPRI_LOF_DIVERGENCE: dict[str, str] = {
    "TYK2":  "P1104A is a catalytic hypomorph leaving expression near-normal; CRISPRi measures transcription not kinase activity",
    "IL6R":  "Shed soluble receptor and membrane receptor have opposing eQTL/LoF relationships",
    "PTPN22": "R620W is a gain-of-function coding variant; CRISPRi measures total transcript loss not GOF",
    "CTLA4": "LoF increases autoimmunity; abatacept mimics CTLA4 agonism — mechanism inversion expected",
    "IRF5":  "Splice-site LoF variants alter isoform ratio, not expression level measurable by CRISPRi",
    "STAT4": "Multiple independent signals at locus; LoF and eQTL instruments may tag different causal variants",
    "IL23R": "Protective LoF (R381Q) is coding; CRISPRi measures proximal promoter, not distal enhancers driving disease association",
    "PCSK9": "Therapeutic target is secreted protein; LoF reduces LDL via intracellular degradation pathway not gene expression",
    "LPA":   "Lp(a) driven by kringle repeat number, not transcript abundance; CRISPRi and LoF are orthogonal instruments",
}


def _wes_concordance_check(
    ota_gamma: float,
    gb_result: dict | None,
    gene: str = "",
) -> dict:
    """
    Check whether the OTA composite γ direction agrees with WES rare LoF burden direction.

    WES burden is a direct human genetic causal signal for gene→disease direction:
      burden_beta > 0: LoF increases disease risk → gene is protective → OTA should be > 0
      burden_beta < 0: LoF decreases disease risk → gene is harmful   → OTA should be < 0

    Concordance: sign(ota_gamma) == sign(burden_beta)  (same signs = agree on gene direction)
    Discordance: sign(ota_gamma) != sign(burden_beta)

    Concordance and discordance are both annotation-only — ota_gamma is never modified.
    wes_gamma_weight is always 1.0. Discordance is documented in wes_note and wes_concordant
    so downstream consumers (CSO, report) can surface the conflict.

    Returns dict with:
      wes_checked:      bool — True when any WES burden data is available
      wes_concordant:   bool | None — direction agreement (None if no data)
      wes_gamma_weight: float — always 1.0 (pure annotation; ota_gamma unchanged)
      wes_burden_p:     float | None
      wes_burden_beta:  float | None
      wes_note:         str
    """
    result = {
        "wes_checked":               False,
        "wes_concordant":            None,
        "wes_gamma_weight":          1.0,
        "wes_burden_p":              None,
        "wes_burden_beta":           None,
        "wes_note":                  "",
        "mechanism_divergence_note": None,
    }

    if gb_result is None:
        return result

    burden_beta = gb_result.get("burden_beta")
    burden_p    = gb_result.get("burden_p")

    if burden_beta is None or burden_p is None:
        return result

    try:
        burden_beta = float(burden_beta)
        burden_p    = float(burden_p)
    except (TypeError, ValueError):
        return result

    if not _math.isfinite(burden_beta) or not _math.isfinite(burden_p):
        return result

    if _math.isnan(ota_gamma) or abs(ota_gamma) < 1e-10:
        return result

    result["wes_checked"]     = True
    result["wes_burden_p"]    = burden_p
    result["wes_burden_beta"] = burden_beta

    # Same signs = concordant: both say gene drives disease (both <0) or gene protects (both >0)
    concordant = (ota_gamma * burden_beta) > 0
    result["wes_concordant"] = concordant

    expected_divergence = _EXPECTED_CRISPRI_LOF_DIVERGENCE.get(gene)

    if concordant:
        result["wes_note"] = (
            f"WES concordant (p={burden_p:.1e}): burden_beta={burden_beta:+.3f} "
            f"agrees with ota_gamma={ota_gamma:+.3f}."
        )
    else:
        if expected_divergence:
            result["mechanism_divergence_note"] = f"expected divergence: {expected_divergence}"
            result["wes_note"] = (
                f"WES discordant (p={burden_p:.1e}): burden_beta={burden_beta:+.3f} "
                f"opposes ota_gamma={ota_gamma:+.3f}. "
                f"Divergence expected — CRISPRi and LoF measure different mechanisms: {expected_divergence}"
            )
        else:
            result["mechanism_divergence_note"] = "unexplained CRISPRi/LoF discordance — mechanism warrants review"
            result["wes_note"] = (
                f"WES discordant (p={burden_p:.1e}): burden_beta={burden_beta:+.3f} "
                f"opposes ota_gamma={ota_gamma:+.3f}. Perturbation and rare-variant direction conflict."
            )

    return result


def _pareto_cutoff(
    ranked: list[dict],
    fraction: float = 0.5,  # unused; retained for call-site compatibility
    key: str = "ota_gamma",
    min_keep: int = 50,
    max_keep: int | None = None,
    gap_threshold: float = PARETO_ELBOW_GAP_THRESHOLD,
) -> list[dict]:
    """Empiric elbow cutoff: find the first position past min_keep where the
    relative drop between consecutive |γ| values exceeds gap_threshold.

    A step-down at PARETO_ELBOW_GAP_THRESHOLD marks the transition from
    disease signal to the long noise tail without any fixed fraction parameter.
    If no elbow is found all genes are returned — no arbitrary truncation.
    """
    if not ranked:
        return []
    vals = [abs(float(r.get(key) or 0.0)) for r in ranked]

    cut = len(ranked)
    for i in range(min_keep, len(vals) - 1):
        v_curr = vals[i]
        v_next = vals[i + 1]
        if v_curr > 1e-10 and (v_curr - v_next) / v_curr >= gap_threshold:
            cut = i + 1
            break

    cut = max(cut, min_keep)
    if max_keep is not None:
        cut = min(cut, max_keep)
    return ranked[: min(cut, len(ranked))]
