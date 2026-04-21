"""
scoring_thresholds.py — Central registry of scoring thresholds used throughout the pipeline.

All magic numbers that govern tier assignment, evidence gating, and score normalisation
live here with provenance citations or explicit design notes. Import from this module
rather than defining local constants.

Sources:
  [MR-WI]   Staley & Burgess (2017) Genet Epidemiol 41:774 — F-statistic weak-instrument threshold
  [COLOC]   Giambartolomei et al. (2014) PLoS Genet — COLOC H4 posterior threshold
  [L2G]     Mountjoy et al. (2021) Nat Genet — Open Targets Locus2Gene score
  [LDSC]    Finucane et al. (2015) Nat Genet — S-LDSC heritability enrichment
  [GPS]     Lamb et al. (2006) Science; Chen et al. (2017) — connectivity map RGES
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Mendelian Randomisation
# ---------------------------------------------------------------------------

MR_F_STATISTIC_MIN: float = 10.0
# Minimum F-statistic for instrument validity in MR.
# Source: [MR-WI] — widely adopted weak-instrument criterion.

MR_P_VALUE_MAX: float = 5e-8
# Genome-wide significance threshold for GWAS instrument selection.

MR_P_VALUE_MAX_EQTL: float = 1e-4
# Maximum p-value for eQTL instrument inclusion.
# Relaxed from GWAS threshold because cis-eQTLs are directional hypotheses.

MR_PQTL_P_VALUE_MAX: float = 0.05
# Maximum p-value for pQTL instrument.
# Relaxed from GWAS threshold: pQTLs often have moderate effect sizes;
# calibrated empirically against UKB-PPP CFH recovery (AMD Run 8, 2026-03).

# ---------------------------------------------------------------------------
# Colocalization (COLOC)
# ---------------------------------------------------------------------------

COLOC_H4_MIN: float = 0.80
# Minimum COLOC H4 posterior probability (shared causal variant) for Tier 2a eQTL-MR.
# Source: [COLOC] — standard in the colocalization literature.
# Below this threshold: demoted to Tier 2.5 (direction-only, sigma=0.50).

COLOC_H4_DIRECTION_MIN: float = 0.50
# Minimum H4 for direction-only Tier 2.5 beta assignment.
# Below this: eQTL is treated as no evidence.

# ---------------------------------------------------------------------------
# Open Targets L2G / Genetic Score
# ---------------------------------------------------------------------------

OT_L2G_MIN: float = 0.05
# Minimum Open Targets L2G score for gene inclusion in GWAS anchor set.
# Source: [L2G] — 0.05 corresponds to ~80% precision in OTG benchmarks.
# Note: raised to 0.10 for gamma estimation to reduce false-positive program links.

OT_L2G_GAMMA_MIN: float = 0.10
# Minimum OT L2G / genetic score used in gamma estimation (estimate_gamma_fused).
# Higher than OT_L2G_MIN to avoid inflating gamma from weakly-associated genes.

OT_GENETIC_SCORE_MIN: float = 0.05
# Minimum OT genetic score for gamma fallback pathway (precomputed_ot_scores overlap).

# ---------------------------------------------------------------------------
# Beta estimation defaults
# ---------------------------------------------------------------------------

BETA_SIGMA_SYNTHETIC: float = 0.45
# Prior width for synthetic (Reactome-derived) Tier 2s beta estimates.
# Design note: larger than eQTL sigma (0.25) to reflect pathway-level uncertainty.
# Empirical; not derived from a specific reference.

BETA_SIGMA_EQTL_DIRECTION: float = 0.50
# Prior width for direction-only Tier 2.5 betas (eQTL present, COLOC H4 < 0.8).

BETA_SIGMA_RARE_BURDEN: float = 0.60
# Prior width for Tier 2rb rare-burden-direction betas.

BETA_LOADING_DEFAULT: float = 0.30
# Default program loading when gene has no NMF weight.
# Represents a conservative estimate of weak program membership.

BETA_LOADING_PROTEIN_CHANNEL: float = 1.0
# Loading for Tier 2pt protein-channel betas (direct gene→disease arc, not mediated by program).

# ---------------------------------------------------------------------------
# Gamma estimation
# ---------------------------------------------------------------------------

GAMMA_HYPERGEOMETRIC_MIN_OVERLAP: int = 2
# Minimum GWAS–program gene overlap for hypergeometric enrichment.
# Fewer than 2 overlapping genes → not enough evidence; return None.

GAMMA_HYPERGEOMETRIC_P_SIGNIFICANT: float = 0.05
# Fisher/hypergeometric p-value for "Tier3_Provisional" gamma classification.
# Below this → enriched; assigned provisional_virtual above.

GAMMA_HYPERGEOMETRIC_NORMALISER: float = 5.0
# Normalisation divisor: gamma_raw = -log10(p) / 5.0, saturating at p=1e-5 → gamma=1.0.
# Design note: chosen so that a genome-wide-significant enrichment (p≈1e-8) → gamma≈0.6,
# leaving headroom for the Bayesian fused estimate.

GAMMA_LDSC_WEIGHT_CAP: float = 10.0
# Maximum weight assigned to the LDSC Z-score component in gamma fusion.
# Prevents pathologically strong enrichment signals from dominating the OT prior.
# Source: [LDSC] — S-LDSC tau statistics can reach very large Z-scores for small gene sets.

GAMMA_OT_PRIOR_WEIGHT: float = 0.20
# Prior weight for the Open Targets L2G component in Bayesian gamma fusion.
# Represents weak genetic association evidence independent of LDSC.

# ---------------------------------------------------------------------------
# Phase F unified target scoring
# ---------------------------------------------------------------------------

SCORE_GENETIC_WEIGHT: float = 0.60
# Weight on genetic evidence component in core target score.
# Design note: genetic evidence (GWAS-grounded gamma) is the primary causal anchor.
# 60/40 split is a design choice — not statistically derived.
# Sensitivity analysis (50/50, 70/30) produced similar rank-ordering on AMD Run 22.

SCORE_MECHANISTIC_WEIGHT: float = 0.40
# Weight on mechanistic evidence (therapeutic redirection, state influence).

SCORE_TRACTABILITY_BONUS_OT: float = 0.15
# t_mod bonus per unit OT tractability (small molecule / antibody tractable targets).

SCORE_TRIAL_BONUS: float = 0.10
# t_mod bonus for targets with ≥ Phase 2 clinical trial evidence.

SCORE_SAFETY_PENALTY: float = 0.10
# t_mod deduction per safety flag (adverse event signal, on-target toxicity report).

SCORE_T_MOD_MIN: float = 0.50
# Minimum t_mod clamp — prevents extreme safety penalties from zeroing out scores.

SCORE_T_MOD_MAX: float = 1.50
# Maximum t_mod clamp.

SCORE_RISK_ESCAPE_WEIGHT: float = 0.20
# Escape risk penalty weight in risk_discount.

SCORE_RISK_FAILURE_WEIGHT: float = 0.15
# Historical trial failure risk penalty weight.

SCORE_RISK_DISCOUNT_MIN: float = 0.10
# Minimum risk_discount (floor at 10% of core score retained).

# ---------------------------------------------------------------------------
# GPS compound screening
# ---------------------------------------------------------------------------

GPS_TIMEOUT_WITH_BGRD: int = 7200
# GPS Docker timeout (seconds) when BGRD is pre-cached (permutation-free run).

GPS_TIMEOUT_NO_BGRD: int = 21600
# GPS Docker timeout (seconds) when BGRD must be computed from scratch.

GPS_Z_RGES_DEFAULT: float = 2.0
# Default RGES Z-score threshold for calling a compound a disease-state reverser.
# Corresponds to ~5% FDR under N(0,1) null; validated empirically (Session 60).
# Source: [GPS] — connectivity-map approach; Z-threshold selection is application-specific.

GPS_MAX_HITS: int = 100
# Maximum compounds returned from a GPS screen.

GPS_BGRD_MIN_GENES: int = 700
# Minimum signature genes required for BGRD elbow-trim (Session 59 calibration).

GPS_BGRD_MAX_GENES: int = 1000
# Maximum signature genes included after elbow-trim.

# ---------------------------------------------------------------------------
# Co-evidence gate
# ---------------------------------------------------------------------------

CO_EVIDENCE_WEIGHT_UNGROUNDED: float = 0.20
# Multiplicative penalty applied to scores for targets with no genetic co-evidence.
# Prevents purely transcriptional candidates from outranking genetically anchored ones.

# ---------------------------------------------------------------------------
# OTA gamma scoring cap
# ---------------------------------------------------------------------------

OTA_GAMMA_SCORE_CAP: float = 0.70
# Hard cap on |ota_gamma| used as target_score.
# Calibrated to the maximum expected causal gamma for a well-validated target
# (e.g. CFH in AMD, HMGCR in CAD; both ~0.5-0.7).
# Prevents essential/housekeeping genes (RNA Pol subunits, DNA primase) whose
# Perturb-seq knockdown mimics broad transcriptional repression from inflating
# their ota_gamma to ±3-5 and dominating the ranked list.

OTA_GAMMA_DIFFUSE_DISCOUNT: float = 0.40
# Additional multiplicative discount for genes with diffuse program loading
# (top_programs spread ≥ 4 programs contributing, none dominant).
# Rationale: a gene whose β is spread evenly across 5 programs is less likely
# to be a disease-specific driver than one concentrated in a single causal program.
