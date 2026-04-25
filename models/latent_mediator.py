"""
Pydantic v2 models for the Phase A latent-mediator OTA refactor.

New scoring formula:
  final_score = therapeutic_redirection + durability - escape_risk - failure_risk
              + 0.15×ot_combined + 0.10×trial_bonus - 0.10×safety_pen

Mediator hierarchy:
  gene → LatentProgram (cNMF) → trait
  beta: gene→program (ConditionalBeta, per cell type)
  gamma: program→trait  (ConditionalGamma, GWAS-anchored + transition-weighted)
  P_loading: 0.7×nmf_loading + 0.3×transition_de_signal
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

ProgramType = Literal[
    "inflammatory",
    "fibrotic",
    "metabolic",
    "stress_response",
    "proliferative",
    "immunoregulatory",
    "angiogenic",
    "unknown",
]

DisagreementRule = Literal[
    "genetics_vs_perturbation",       # sign(MR-IVW) ≠ sign(T1/T2 beta)
    "perturbation_vs_chemical",       # sign(Perturb-seq) ≠ sign(LINCS)
    "bulk_vs_singlecell",             # LINCS strong, sc beta near-zero
    "cross_context_sign_flip",        # context-verified betas flip sign across cell types
]


# ---------------------------------------------------------------------------
# LatentProgram
# ---------------------------------------------------------------------------

class LatentProgram(BaseModel):
    """One cNMF program inferred from a single cell-type h5ad."""

    program_id: str                              # e.g. "IBD_macrophage_P07"
    disease: str
    cell_type: str
    program_index: int                           # 0-based NMF component index
    top_genes: list[str] = Field(default_factory=list)  # top-weighted genes (descending)
    program_type: ProgramType = "unknown"
    hallmark_annotations: list[str] = Field(default_factory=list)  # MSigDB hallmarks (annotation only)
    gwas_enrichment_score: float | None = None   # −log10(p) from GWAS enrichment
    n_cells_expressing: int | None = None
    data_source: str = ""

    @field_validator("program_index")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("program_index must be ≥ 0")
        return v


# ---------------------------------------------------------------------------
# ConditionalBeta  (gene → program, per cell type)
# ---------------------------------------------------------------------------

class ConditionalBeta(BaseModel):
    """
    Estimated β for gene → LatentProgram in a specific cell type.

    pooled_fallback=True means the beta comes from merged/pooled data because
    single-cell-type perturbation data were unavailable.  When >50% of betas
    for a target are pooled, emit a context_confidence_warning.
    """

    gene: str
    program_id: str
    cell_type: str
    disease: str
    beta: float                                  # NaN if not estimable
    beta_se: float | None = None
    pooled_fallback: bool = False                # True → pooled data used
    context_verified: bool = False               # True → cell-type-specific perturbation confirmed
    evidence_tier: str = "Tier3_TrajectoryProxy"
    data_source: str = ""

    @field_validator("beta")
    @classmethod
    def no_zero_for_missing(cls, v: float) -> float:
        # Caller must pass float('nan') for unknown, not 0.0
        # We don't block 0.0 here (it could be a real effect) but document the convention.
        return v


# ---------------------------------------------------------------------------
# ConditionalGamma  (program → trait, mixed GWAS + transition)
# ---------------------------------------------------------------------------

class ConditionalGamma(BaseModel):
    """
    Mixed γ: alpha × gamma_GWAS + (1-alpha) × gamma_transition.

    alpha is chosen by StateEvidenceTier:
      T1 (TrajectoryDirect)   → alpha = 0.35
      T2 (TrajectoryInferred) → alpha = 0.55
      T3 (TrajectoryProxy)    → alpha = 0.70
    """

    program_id: str
    trait: str
    disease: str
    gamma_gwas: float                            # GWAS-anchored estimate; NaN if missing
    gamma_transition: float                      # transition-weighted estimate; NaN if missing
    alpha: float                                 # GWAS weight ∈ [0, 1]
    gamma_mixed: float                           # final mixed value
    evidence_tier: str = "Tier3_TrajectoryProxy"
    uncertainty: float | None = None
    data_source: str = ""

    @field_validator("alpha")
    @classmethod
    def valid_alpha(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {v}")
        return v


# ---------------------------------------------------------------------------
# ProgramLoading  (gene × program × cell_type)
# ---------------------------------------------------------------------------

class ProgramLoading(BaseModel):
    """
    P_loading = 0.7 × nmf_loading + 0.3 × transition_de_signal

    nmf_loading: raw NMF H-matrix weight (gene column in program row), L2-normalised.
    transition_de_signal: |log2FC| of gene between disease and healthy attractor
                          centroids, clipped to [0, 1].
    """

    gene: str
    program_id: str
    cell_type: str
    disease: str
    nmf_loading: float
    transition_de_signal: float
    p_loading: float                             # combined score

    @field_validator("nmf_loading", "transition_de_signal", "p_loading")
    @classmethod
    def finite_float(cls, v: float) -> float:
        import math
        if math.isnan(v) or math.isinf(v):
            raise ValueError("ProgramLoading values must be finite")
        return v


# ---------------------------------------------------------------------------
# TherapeuticRedirectionResult
# ---------------------------------------------------------------------------

class TherapeuticRedirectionResult(BaseModel):
    """
    Per-gene summary of the therapeutic_redirection component.

    therapeutic_redirection = Σ_c w(c) × Σ_P P_loading(gene,P,c)
                              × Σ_T [T_perturbed(c) - T_baseline(c)] × gamma_conditional(P,d|c)

    T_perturbed is capped at 50% of baseline per transition edge before row-renormalisation.
    """

    gene: str
    disease: str
    therapeutic_redirection: float
    durability_score: float = 0.0
    escape_risk: float = 0.0
    failure_risk: float = 0.0
    # additive bonuses / penalties (applied on top)
    ot_combined: float = 0.0
    trial_bonus: float = 0.0
    safety_penalty: float = 0.0
    # mechanistic state-layer fields (Phase F — annotation only from Phase G onward)
    state_influence_score: float = 0.0          # legacy DAS [0,1] — NOT used in formula from Phase G
    directionality: int = 0                     # +1 = higher in pathological; -1 = higher in healthy
    genetic_grounding: float = 0.0              # OTA γ used to ground disease relevance
    # Phase G transition scores (replace state_influence in formula)
    entry_score: float = 0.0                    # enrichment in cells entering pathological basins
    persistence_score: float = 0.0             # enrichment in cells dwelling in pathological basins
    recovery_score: float = 0.0                # enrichment in cells exiting toward healthy basins
    boundary_score: float = 0.0                # max(knn_boundary, pseudotime_boundary)
    mechanistic_category: str = "unknown"      # trigger | maintenance | recovery | mixed
    # Phase Z7: Monte Carlo transition matrix sensitivity
    stability_score: float = 1.0                # 1.0 = perfectly stable, 0.0 = rank collapses on noise
    # metadata
    n_programs_contributing: int = 0
    n_cell_types_contributing: int = 0
    pooled_fraction: float = 0.0                 # fraction of betas from pooled fallback
    context_confidence_warning: bool = False     # True if pooled_fraction > 0.50
    evidence_tiers_used: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)
    # Phase R: disease-state τ specificity (within-cell-type, from h5ad disease groups)
    # 0 = ubiquitous across disease states, 1 = exclusively in one disease state
    # Default 0.5 = neutral (unknown / not computed)
    tau_disease_specificity: float = 0.5

    @property
    def assessment(self) -> dict:
        """
        Structured per-dimension readout replacing the collapsed final_score formula.

        Returns a dict with five labeled dimensions. Primary ranking uses ota_gamma
        (genetic_grounding) directly — not a weighted average of these dimensions.
        Each dimension can be read independently or rendered as prose by the writer.
        """
        ota = float(self.genetic_grounding)

        # Causal dimension: OTA γ and its uncertainty
        if abs(ota) >= 0.3:
            causal_confidence = "high"
        elif abs(ota) >= 0.1:
            causal_confidence = "moderate"
        elif abs(ota) > 0:
            causal_confidence = "low"
        else:
            causal_confidence = "absent"

        # Mechanistic dimension: TR + transition scores
        weighted_transition = (
            0.35 * self.entry_score +
            0.35 * self.persistence_score +
            0.20 * self.recovery_score +
            0.10 * self.boundary_score
        )
        mech_strength = (abs(self.therapeutic_redirection) + weighted_transition) * self.stability_score
        mech_verdict = (
            "strong" if mech_strength >= 0.6
            else "moderate" if mech_strength >= 0.3
            else "weak"
        )

        # Disease specificity dimension
        tau = self.tau_disease_specificity
        specificity_verdict = (
            "disease-specific" if tau >= 0.6
            else "moderate specificity" if tau >= 0.4
            else "broadly expressed"
        )

        # Translational dimension: OT + trials
        translational_verdict = (
            "clinical evidence" if self.trial_bonus >= 0.5
            else "OT-supported" if self.ot_combined >= 0.3
            else "preclinical only"
        )

        # Safety dimension
        total_risk = (
            0.20 * min(self.escape_risk, 1.0)
            + 0.15 * min(self.failure_risk, 1.0)
            + 0.30 * min(self.safety_penalty, 1.0)
        )
        safety_verdict = (
            "high risk" if total_risk >= 0.25
            else "moderate risk" if total_risk >= 0.10
            else "low risk"
        )

        return {
            "causal": {
                "ota_gamma":   round(ota, 4),
                "confidence":  causal_confidence,
            },
            "mechanistic": {
                "therapeutic_redirection": round(float(self.therapeutic_redirection), 4),
                "state_influence":         round(float(self.state_influence_score), 4),
                "weighted_transition":     round(float(weighted_transition), 4),
                "stability":               round(float(self.stability_score), 4),
                "verdict":                 mech_verdict,
            },
            "disease_specificity": {
                "tau":     round(float(tau), 4),
                "verdict": specificity_verdict,
            },
            "translational": {
                "ot_combined":  round(float(self.ot_combined), 4),
                "trial_bonus":  round(float(self.trial_bonus), 4),
                "safety_penalty": round(float(self.safety_penalty), 4),
                "verdict":      translational_verdict,
            },
            "safety": {
                "escape_risk":  round(float(self.escape_risk), 4),
                "failure_risk": round(float(self.failure_risk), 4),
                "verdict":      safety_verdict,
            },
        }


# ---------------------------------------------------------------------------
# EvidenceDisagreementRecord
# ---------------------------------------------------------------------------

class EvidenceDisagreementRecord(BaseModel):
    """
    One rule-based evidence disagreement or transportability flag for a gene.

    Rules:
      1. genetics_vs_perturbation   sign(MR-IVW) ≠ sign(T1/T2 beta)
      2. perturbation_vs_chemical   sign(Perturb-seq) ≠ sign(LINCS)
      3. bulk_vs_singlecell         LINCS strong (|β|>0.3), sc beta near-zero (|β|<0.05)
      4. cross_context_sign_flip    context-verified betas flip sign across cell types
    """

    gene: str
    disease: str
    rule: DisagreementRule
    value_a: float | None = None                 # first comparand
    value_b: float | None = None                 # second comparand
    cell_type_a: str | None = None
    cell_type_b: str | None = None               # used for cross_context_sign_flip
    explanation: str = ""
    severity: Literal["warning", "flag", "block"] = "warning"
