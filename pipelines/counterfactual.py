"""
pipelines/counterfactual.py — Phase S: OTA counterfactual simulation.

simulate_perturbation(gene, delta_beta_fraction, beta_estimates, gamma_estimates, trait)
  Predicts the causal effect of a genetic/pharmacological intervention by
  re-computing OTA γ after scaling β values.

OTA formula (Ota et al.):
  γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait})

Perturbation model:
  β_perturbed = β_original × (1 + delta_beta_fraction)
  e.g. delta_beta_fraction = -1.0  → complete knockout   (β → 0)
       delta_beta_fraction = -0.5  → 50% inhibition
       delta_beta_fraction =  1.0  → 2× over-expression
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.ota_gamma_estimation import compute_ota_gamma

# Tier-specific noise scale (same as SCONE bootstrap_edge_confidence)
_TIER_NOISE: dict[str, float] = {
    "Tier1_Interventional": 0.15,
    "Tier2_Convergent":     0.25,
    "Tier3_Provisional":    0.40,
    "provisional_virtual":  0.60,
}
_DEFAULT_NOISE = 0.40


def simulate_perturbation(
    gene: str,
    delta_beta_fraction: float,
    beta_estimates: dict[str, Any],
    gamma_estimates: dict[str, Any],
    trait: str,
) -> dict:
    """
    Simulate the causal impact of perturbing a gene's β values.

    Args:
        gene:                Gene symbol (e.g. "NOD2")
        delta_beta_fraction: Fractional change applied to all β values.
                             -1.0 = complete knockout, -0.5 = 50% inhibition,
                             0.0 = no change, 1.0 = doubling.
        beta_estimates:      {program → {"beta": float, "evidence_tier": str} | float | None}
                             Beta (effect size) estimates for this gene across programs.
        gamma_estimates:     {program → {"gamma": float, ...}} program→trait effect sizes.
        trait:               Disease trait name (e.g. "inflammatory bowel disease").

    Returns:
        {
            "gene":                           str,
            "trait":                          str,
            "delta_beta_fraction":            float,
            "baseline_gamma":                 float,  OTA γ before perturbation
            "perturbed_gamma":                float,  OTA γ after perturbation
            "delta_gamma":                    float,  perturbed - baseline
            "percent_change":                 float,  delta / |baseline| × 100
            "interpretation":                 str,    human-readable summary
            "dominant_program":               str | None,
            "program_contributions_baseline": list[dict],
            "program_contributions_perturbed":list[dict],
        }
    """
    # ---- Baseline ----
    baseline = compute_ota_gamma(
        gene=gene,
        trait=trait,
        beta_estimates=beta_estimates,
        gamma_estimates=gamma_estimates,
    )
    baseline_gamma: float = baseline["ota_gamma"]

    # ---- Perturbed β ----
    scale = 1.0 + delta_beta_fraction
    perturbed_betas: dict[str, Any] = {}
    for program, entry in beta_estimates.items():
        if entry is None:
            perturbed_betas[program] = None
            continue
        if isinstance(entry, dict):
            new_entry = dict(entry)
            orig_beta = entry.get("beta")
            if orig_beta is not None:
                new_entry["beta"] = orig_beta * scale
            perturbed_betas[program] = new_entry
        else:
            perturbed_betas[program] = float(entry) * scale

    # ---- Perturbed OTA γ ----
    perturbed = compute_ota_gamma(
        gene=gene,
        trait=trait,
        beta_estimates=perturbed_betas,
        gamma_estimates=gamma_estimates,
    )
    perturbed_gamma: float = perturbed["ota_gamma"]

    delta_gamma = perturbed_gamma - baseline_gamma
    percent_change = (
        (delta_gamma / abs(baseline_gamma) * 100.0)
        if abs(baseline_gamma) > 1e-10
        else 0.0
    )

    # ---- Interpretation ----
    direction = "reduction" if delta_gamma < 0 else "increase"
    if abs(delta_beta_fraction) < 0.01:
        interpretation = "No perturbation applied."
    elif delta_beta_fraction <= -0.9:
        interpretation = (
            f"Complete knockout of {gene} predicts "
            f"{direction} of {abs(percent_change):.1f}% in "
            f"γ({gene}→{trait})."
        )
    elif delta_beta_fraction < 0:
        inh_pct = abs(delta_beta_fraction) * 100
        interpretation = (
            f"{inh_pct:.0f}% inhibition of {gene} predicts "
            f"{direction} of {abs(percent_change):.1f}% in "
            f"γ({gene}→{trait})."
        )
    else:
        act_pct = delta_beta_fraction * 100
        interpretation = (
            f"{act_pct:.0f}% activation of {gene} predicts "
            f"{direction} of {abs(percent_change):.1f}% in "
            f"γ({gene}→{trait})."
        )

    # ---- Dominant program (highest |contribution| at baseline) ----
    dominant_program: str | None = None
    max_contrib = 0.0
    for contrib in baseline.get("program_contributions", []):
        c = abs(contrib.get("contribution", 0.0))
        if c > max_contrib:
            max_contrib = c
            dominant_program = contrib.get("program")

    return {
        "gene":                           gene,
        "trait":                          trait,
        "delta_beta_fraction":            delta_beta_fraction,
        "baseline_gamma":                 round(baseline_gamma, 6),
        "perturbed_gamma":                round(perturbed_gamma, 6),
        "delta_gamma":                    round(delta_gamma, 6),
        "percent_change":                 round(percent_change, 2),
        "interpretation":                 interpretation,
        "dominant_program":               dominant_program,
        "program_contributions_baseline": baseline.get("program_contributions", []),
        "program_contributions_perturbed": perturbed.get("program_contributions", []),
    }


def _add_tier_noise(beta_estimates: dict[str, Any]) -> dict[str, Any]:
    """Add tier-specific Gaussian noise to β values. Returns a new dict."""
    noisy: dict[str, Any] = {}
    for program, entry in beta_estimates.items():
        if entry is None:
            noisy[program] = None
            continue
        if isinstance(entry, dict):
            new_entry = dict(entry)
            beta_val  = entry.get("beta") or 0.0
            tier      = entry.get("evidence_tier", "provisional_virtual")
            scale     = _TIER_NOISE.get(tier, _DEFAULT_NOISE)
            noise     = random.gauss(0, abs(beta_val) * scale + 1e-6)
            new_entry["beta"] = beta_val + noise
            noisy[program] = new_entry
        else:
            beta_val = float(entry)
            noise    = random.gauss(0, abs(beta_val) * _DEFAULT_NOISE + 1e-6)
            noisy[program] = beta_val + noise
    return noisy


def simulate_perturbation_with_uncertainty(
    gene: str,
    delta_beta_fraction: float,
    beta_estimates: dict[str, Any],
    gamma_estimates: dict[str, Any],
    trait: str,
    n_bootstrap: int = 50,
) -> dict:
    """
    Simulate the causal impact of a perturbation with bootstrap uncertainty.

    The OTA formula is linear in β, so uniform β scaling gives an exactly
    proportional γ change by definition.  This function adds tier-specific
    Gaussian noise to β before each perturbation, producing a distribution of
    outcomes whose spread reflects genuine β uncertainty:

      - Tier1 genes (β well-measured): tight CI — e.g. -50% ± 3%
      - Tier3/virtual genes (β uncertain): wide CI — e.g. -50% ± 35%

    The 95% CI answers: "Given what we know about β, how confident are we in
    this linear-model prediction?"

    Args:
        gene:                Gene symbol
        delta_beta_fraction: Fractional change (-0.5 = 50% inhibition, -1.0 = KO)
        beta_estimates:      {program → {beta, evidence_tier, ...}}
        gamma_estimates:     {program → {gamma, ...}}
        trait:               Disease trait
        n_bootstrap:         Bootstrap samples (default 50)

    Returns:
        All fields of simulate_perturbation plus:
        {
            "ci_lower":            float,   2.5th percentile of perturbed_gamma
            "ci_upper":            float,   97.5th percentile of perturbed_gamma
            "se_perturbed":        float,   std dev of bootstrap perturbed_gammas
            "n_bootstrap":         int,
            "uncertainty_note":    str,     human-readable CI summary
        }
    """
    # Deterministic point estimate (no noise)
    point = simulate_perturbation(gene, delta_beta_fraction, beta_estimates, gamma_estimates, trait)

    # Bootstrap distribution
    perturbed_gammas: list[float] = []
    for _ in range(n_bootstrap):
        noisy_betas = _add_tier_noise(beta_estimates)
        try:
            result = simulate_perturbation(
                gene, delta_beta_fraction, noisy_betas, gamma_estimates, trait
            )
            perturbed_gammas.append(result["perturbed_gamma"])
        except Exception:
            pass

    if len(perturbed_gammas) < 3:
        # Not enough samples — return point estimate without CI
        return {
            **point,
            "ci_lower":       None,
            "ci_upper":       None,
            "se_perturbed":   None,
            "n_bootstrap":    len(perturbed_gammas),
            "uncertainty_note": "Insufficient bootstrap samples for CI.",
        }

    perturbed_gammas.sort()
    n = len(perturbed_gammas)
    ci_lower = perturbed_gammas[int(0.025 * n)]
    ci_upper = perturbed_gammas[int(0.975 * n)]
    mean_p   = sum(perturbed_gammas) / n
    var_p    = sum((x - mean_p) ** 2 for x in perturbed_gammas) / (n - 1)
    se_p     = math.sqrt(var_p)

    baseline = point["baseline_gamma"]
    if abs(baseline) > 1e-10:
        ci_lower_pct = round((ci_lower - baseline) / abs(baseline) * 100, 1)
        ci_upper_pct = round((ci_upper - baseline) / abs(baseline) * 100, 1)
        note = (
            f"Bootstrap 95% CI on perturbed γ: [{ci_lower:.4f}, {ci_upper:.4f}] "
            f"({ci_lower_pct}% to {ci_upper_pct}% change). "
            f"SE={se_p:.4f} reflects β uncertainty across {n} bootstrap samples."
        )
    else:
        note = f"Baseline γ ≈ 0; CI=[{ci_lower:.4f}, {ci_upper:.4f}] ({n} samples)."

    return {
        **point,
        "ci_lower":        round(ci_lower, 6),
        "ci_upper":        round(ci_upper, 6),
        "se_perturbed":    round(se_p, 6),
        "n_bootstrap":     n,
        "uncertainty_note": note,
    }
