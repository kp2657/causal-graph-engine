"""
scoring_thresholds.py — Central registry of scoring thresholds used throughout the pipeline.

All magic numbers that govern tier assignment, evidence gating, and score normalisation
live here with provenance citations or explicit design notes. Import from this module
rather than defining local constants.

Sources:
  [MR-WI]   Staley & Burgess (2017) Genet Epidemiol 41:774 — F-statistic weak-instrument threshold
  [COLOC]   Giambartolomei et al. (2014) PLoS Genet — COLOC H4 posterior threshold
  [L2G]     Mountjoy et al. (2021) Nat Genet — Open Targets Locus2Gene score
  [GPS]     Lamb et al. (2006) Science; Chen et al. (2017) — connectivity map RGES
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# GWAS thresholds
# ---------------------------------------------------------------------------

MR_P_VALUE_MAX: float = 5e-8
# Genome-wide significance threshold for GWAS hit inclusion.

MR_PQTL_P_VALUE_MAX: float = 5e-5
# Significance threshold for pQTL association in beta estimation.
# Looser than GWAS threshold to allow instrument discovery for protein biomarkers.

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


# ---------------------------------------------------------------------------
# Causal discovery
# ---------------------------------------------------------------------------

OTA_GAMMA_EDGE_MIN: float = 0.01
# Minimum |ota_gamma| for writing a gene→trait edge to the Kùzu graph.
# Edges below this are noise-level and bloat the graph without adding signal.

CAUSAL_STRESS_MEAN_THRESHOLD: float = 1.2
# Mean |beta| across programs above which a gene is flagged as a global stress responder.
# Genes with uniformly large betas across all programs likely reflect transcriptional
# shutdown rather than a program-specific causal effect.

CAUSAL_STRESS_CV_FLOOR: float = 0.35
# Coefficient of variation below this (betas are uniformly large) triggers stress flag.
# High mean + low CV = indiscriminate perturbation response.

CAUSAL_STRESS_DISCOUNT: float = 0.25
# Multiplicative discount applied to ota_gamma when stress pattern is detected.
# Retains the gene in the ranking but substantially penalises its score.

PARETO_ELBOW_GAP_THRESHOLD: float = 0.20
# Minimum relative drop between consecutive |γ| values to call the Pareto elbow.
# A 20% drop signals the transition from high-signal to noise-floor candidates.
# Source: empirical calibration on AMD Run 22 and CAD Run 3.

# ---------------------------------------------------------------------------
# pLI / safety penalties (target prioritisation)
# ---------------------------------------------------------------------------


PLI_ESSENTIAL_THRESHOLD: float = 0.9
# pLI above this marks a gene as constrained (loss-of-function intolerant).
# Such genes are de-weighted as drug targets due to on-target toxicity risk.
# Source: gnomAD pLI score; threshold follows Karczewski et al. (2020) Nature.

PLI_SAFETY_PENALTY: float = 0.3
# Multiplicative t_mod penalty applied to OT L2G gamma when pLI > PLI_ESSENTIAL_THRESHOLD.
# Reduces but does not zero out constrained genes — genetic evidence still counted.


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

GPS_MIN_DISEASE_SIG_GENES: int = 50
# Minimum genes required in the disease-state differential signature for GPS screen.
# Below this: signature is too sparse to produce reliable RGES scores.

GPS_MIN_PROGRAM_SIG_GENES: int = 5
# Minimum genes required in an NMF program signature for GPS program-reversal screen.

GPS_MAX_PARALLEL: int = 3
# Maximum concurrent GPS Docker calls.
# Bounded by Docker memory: each call loads ~2 GB of LINCS data.

GPS_JACCARD_SKIP_THRESHOLD: float = 0.70
# Skip GPS screening for a gene if its signature Jaccard similarity to an already-run
# signature exceeds this threshold — avoids redundant Docker calls.

GPS_PROGRAM_WEIGHT_FRACTION: float = 0.10
# Fraction of top-weighted genes taken from each NMF program to build the program
# reversal signature (e.g., 0.10 = top 10% of program-loading genes).

GPS_TIMEOUT_WITH_BGRD: int = 7200
# GPS Docker timeout (seconds) when BGRD is pre-cached (permutation-free run).

GPS_TIMEOUT_NO_BGRD: int = 21600
# GPS Docker timeout (seconds) when BGRD must be computed from scratch.

GPS_Z_RGES_DEFAULT: float = 3.5
# Z_RGES threshold for calling a compound a disease-state reverser (3.5σ from permuted null).
# Raised from 2.0: CAD screen has 2,997 compounds at Z < -2.0 but smooth distribution with
# no natural break; 3.5σ (862 CAD compounds) represents genuine reversal signal.
# The threshold governs the z-scored path; GPS_MAX_HITS is only the non-z-scored fallback cap.
# Source: [GPS] — connectivity-map approach; Z-threshold selection is application-specific.

GPS_MAX_HITS: int = 500
# Safety cap for the non-z-scored GPS fallback path (top_n by |RGES|).
# Not applied when GPS output contains Z_RGES column — threshold governs in that case.

GPS_BGRD_MIN_GENES: int = 500
# Minimum signature genes for BGRD elbow-trim. GPS sets n_permutations=n_sig_genes;
# 500 completes in ~2h on Apple Silicon (Rosetta2); 700 exceeds the 6h GPS internal timeout.

GPS_BGRD_MAX_GENES: int = 500
# Maximum signature genes after elbow-trim. Pinned to 500 so all disease screens share
# BGRD__size500.pkl regardless of DEG count.

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

# ---------------------------------------------------------------------------
# Leave-locus-out (LOO) γ stability
# ---------------------------------------------------------------------------

LOO_STABILITY_THRESHOLD: int = 10
# Rank change ≤ 10 = leave-locus-out stable; Pritchard 2025 criterion.
# Genes whose rank changes by > 10 when their own locus (±1 Mb) is excluded
# from the GWAS SNP pool may have a circularly inflated γ.
