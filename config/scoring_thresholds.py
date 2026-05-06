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

# ---------------------------------------------------------------------------
# Cache-first OTA nomination (replaces SVD cosine / fingerprint gates)
# Effective score = |pre_OTA| × l2g_weight × eqtl_coherence_weight
# All three are genetic signals from the Ota/Zhu framework.
# ---------------------------------------------------------------------------

CACHE_FIRST_L2G_STRONG: float = 0.5   # OT L2G ≥ 0.5 → l2g_weight = 1.0 (specific causal gene)
CACHE_FIRST_L2G_MODERATE: float = 0.1  # OT L2G ≥ 0.1 → l2g_weight = 0.7 (probable V2G)
CACHE_FIRST_L2G_WEIGHT_STRONG: float = 1.0
CACHE_FIRST_L2G_WEIGHT_MODERATE: float = 0.7
CACHE_FIRST_L2G_WEIGHT_PROXIMAL: float = 0.55  # in Perturb library (GWAS-proximal) but no specific V2G
CACHE_FIRST_L2G_WEIGHT_NONE: float = 0.1      # no GWAS link

CACHE_FIRST_EQTL_CONCORDANT_WEIGHT: float = 1.0   # sign(eQTL_β × OTA) < 0 → concordant
CACHE_FIRST_EQTL_DISCORDANT_WEIGHT: float = 0.7   # sign(eQTL_β × OTA) > 0 → discordant
# Concordant: risk allele increases expression (eQTL_β > 0) AND KO reduces disease (OTA < 0)
# = more expression promotes disease = inhibiting gene is therapeutic

CACHE_FIRST_EFFECTIVE_SCORE_MIN: float = 0.04
# Min effective_score = |pre_OTA| × l2g_weight × eqtl_weight for cache-first nomination.
# Calibrated so KRIT1 (|pre_OTA|=0.137 × 0.4 × 1.0 = 0.055) and
# PLPP3 (0.115 × 0.4 × 1.0 = 0.046) pass, while noise genes below 0.04 are excluded.

CACHE_FIRST_MAX_NOMINEES: int = 1500
# Safety cap on total cache-first nominees per disease (sorted by effective_score desc).
# Set to 1500 to include the full Schnitzler library (1293 genes pass |pre_OTA|≥0.10);
# the library was pre-selected for GWAS proximity so all passing genes are legitimate.

USE_CACHE_FIRST_NOMINATION: bool = True
# Feature flag — flip to True after Phase 3 validation. When False, existing
# SVD cosine + fingerprint + high-OTA paths run unchanged.

PERTURB_NOMINATED_GAMMA_MIN: float = 0.10
# Legacy gate used by _collect_high_ota_perturbseq_nominees (active when
# USE_CACHE_FIRST_NOMINATION=False). Kept for backward compat; superseded by
# CACHE_FIRST_EFFECTIVE_SCORE_MIN when USE_CACHE_FIRST_NOMINATION=True.

PERTURB_NOMINATED_MAX: int = 200
# Legacy cap for _collect_high_ota_perturbseq_nominees.

CAUSAL_STRESS_MEAN_THRESHOLD: float = 1.2
# Mean |beta| across programs above which a gene is flagged as a global stress responder.
# Genes with uniformly large betas across all programs likely reflect transcriptional
# shutdown rather than a program-specific causal effect.

CAUSAL_STRESS_CV_FLOOR: float = 0.35
# Coefficient of variation below this (betas are uniformly large) triggers stress flag.
# High mean + low CV = indiscriminate perturbation response.
# NOTE: CV-based filter only valid for NMF/NES-scale betas. For SVD z-score betas
# (Schnitzler library), CV is uniformly high (0.5–0.8) even for genuine stress genes.
# Use CACHE_FIRST_STRESS_MEAN_THRESHOLD instead for cache-first nomination.

CACHE_FIRST_STRESS_MEAN_THRESHOLD: float = 2.5
# Hard mean|β| cutoff for SVD z-score betas in cache-first nomination.
# Calibrated so ACTR3C (5.5) and TP53 (3.1) are excluded, while CCM2 (1.9),
# PLPP3 (1.7), and KRIT1 (0.8) are retained. No CV gate applied.

CAUSAL_STRESS_DISCOUNT: float = 0.25
# Multiplicative discount applied to ota_gamma when stress pattern is detected.
# Retains the gene in the ranking but substantially penalises its score.

PARETO_ELBOW_GAP_THRESHOLD: float = 0.20
# Minimum relative drop between consecutive |γ| values to call the Pareto elbow.
# A 20% drop signals the transition from high-signal to noise-floor candidates.
# Source: empirical calibration on AMD Run 22 and CAD Run 3.

# ---------------------------------------------------------------------------
# WES burden directional concordance check
# ---------------------------------------------------------------------------

WES_CONCORDANCE_P_THRESHOLD: float = 0.05
# Minimum burden-test significance for WES concordance boost to apply.
# Relaxed to p<0.05 so that nominally significant burden signals inform ranking;
# boost scales continuously with stat strength so weak signals get a small lift.

WES_CONCORDANT_BOOST: float = 1.50
# Multiplicative boost on ota_gamma when WES burden direction agrees with OTA direction
# (p < WES_CONCORDANCE_P_THRESHOLD). Concordance with a direct human genetic causal
# signal increases confidence in the OTA estimate. Scales linearly from 1.0 at
# p=0.05 to 1.5 at p=5e-8 (GWS), linear in -log10(p) space.

WES_CODING_MECHANISM_P: float = 1e-4
# WES burden p-value below which a gene is assumed to act via a coding mechanism.
# When p < WES_CODING_MECHANISM_P, Tier 2a/2b/2c eQTL-MR β is suppressed in
# ota_beta_estimation.estimate_beta() — eQTL (mRNA level) is not the right instrument
# when the variant effect is coding/protein-level. Tier 2p (pQTL), 2rb, and 2.5 are unaffected.

# ---------------------------------------------------------------------------
# GWAS direction vs eQTL direction concordance (ota_beta_estimation)
# ---------------------------------------------------------------------------

EQTL_GWAS_DIRECTION_SIGMA_FACTOR: float = 0.75
# sigma multiplier applied to Tier2/2c eQTL beta_sigma when eQTL NES direction
# agrees with GWAS risk allele direction (at COLOC H4 ≥ 0.5).
# Tightens uncertainty when two independent lines of genetic evidence agree on direction.
# 0.75 = 25% sigma reduction; preserves most uncertainty while rewarding concordance.

EQTL_GWAS_DIRECTION_COLOC_MIN: float = 0.50
# Minimum COLOC H4 for GWAS direction vs eQTL concordance check.
# Below this the signals may not share a causal variant — don't reward agreement.

# ---------------------------------------------------------------------------
# Fine-mapped posterior inclusion probability (target_ranker)
# ---------------------------------------------------------------------------

FINE_MAPPED_PIP_MIN: float = 0.10
# Minimum fine-mapped PIP to report a gene's credible set membership.
# Genes below this threshold are not reported as fine-mapped; PIP field = 0.0.
# Source: Schaid et al. (2018) recommend PIP ≥ 0.1 as "plausibly fine-mapped".

# ---------------------------------------------------------------------------
# S-LDSC heritability enrichment — program ranking and bystander filtering
# ---------------------------------------------------------------------------

SLDSC_BYSTANDER_WEIGHT: float = 0.30
# Multiplicative weight applied to the β×γ contribution of programs with τ ≤ 0
# (no GWAS heritability enrichment). Programs not enriched for disease heritability
# are likely bystander/reactive programs (cellular response to disease, not causal).
# When S-LDSC data is unavailable this weight is not applied (graceful fallback).

SLDSC_TAU_SIGNIFICANT_P: float = 0.05
# τ_p threshold below which a program is considered significantly heritability-enriched.
# Used to annotate program contributions in the OTA sum output.

SLDSC_GAMMA_FLOOR: float = 0.02
# Minimum |γ(P→trait)| for a program to contribute to the OTA sum β×γ.
# Programs below this threshold have near-zero heritability enrichment; large β on them
# adds noise without signal (γ_SE-weighting is insufficient when SE << |γ|).
# Matches MIN_PROG_GAMMA in the long-island plot for consistency.
# Derived from S-LDSC τ distribution: max=0.113, floor cuts programs with |τ|<0.02 (~15%).

# ---------------------------------------------------------------------------
# Multi-timepoint Perturb-seq — condition-specific β quality
# ---------------------------------------------------------------------------

TIMEPOINT_CONCORDANCE_SIGMA_FACTOR: float = 0.82
# beta_sigma multiplier when Stim8hr and Stim48hr perturbation effects show
# concordant sign for the same NMF program. Same-assay concordance is weaker
# evidence than GWAS/eQTL concordance (0.75), hence the milder 0.82 reward.
# Zhu et al. (2025): regulator-burden correlations are enriched in both Stim8hr
# and Stim48hr, supporting concordant effects as a quality signal.

TIMEPOINT_ACTIVATION_BIAS_MIN: float = 1.5
# Minimum ratio of (Stim8hr n_regulators) / (Rest n_regulators), weighted by
# NMF program gene loadings, for a program to count as activation-enriched.
# Programs below this threshold are discounted by SLDSC_BYSTANDER_WEIGHT (0.30):
# Zhu et al. (2025) Fig 7A: Rest-dominant regulator clusters show no RA/SLE/T1D
# enrichment; Stim8hr/Stim48hr clusters drive autoimmune disease association.

# ---------------------------------------------------------------------------
# RNA fingerprinting — SVD-denoised Perturb-seq β
# ---------------------------------------------------------------------------

FINGERPRINT_SVD_RANK: int = 30
# Number of latent factors for truncated SVD denoising of the gene × perturbation
# log2FC matrix. Matches the default k=30 in the RNA fingerprinting paper
# (Elorbany et al. 2025, PMC12458102). Higher k retains more variance but less
# denoising; lower k is smoother but may lose perturbation-specific signal.

FINGERPRINT_N_BOOTSTRAP: int = 50
# Bootstrap resampling iterations for disease-to-fingerprint correlation uncertainty.

# ---------------------------------------------------------------------------
# Genetic-direction NMF — WES-regularized program decomposition
# ---------------------------------------------------------------------------

GENETIC_NMF_RANK: int = 30
# Number of programs for the WES-regularized NMF decomposition. Kept equal to
# FINGERPRINT_SVD_RANK so downstream consumers (S-LDSC, nomination scorer) see
# the same program dimensionality regardless of which decomposition is active.

GENETIC_NMF_LAMBDA: float = 0.10
# Incoherence penalty strength: λ × Σ_k pos_k × neg_k.
# pos_k = Σ_atherogenic W[i,k]×w_i, neg_k = Σ_protective W[i,k]×w_i.
# Calibration target: programs should move from genetic_direction_score ~0 (SVD
# baseline) toward |score| ≥ 0.4 without collapsing to single-gene programs.
# 0.10 is a conservative starting point; increase if programs remain mixed.

GENETIC_NMF_MAX_ITER: int = 300
# Maximum multiplicative update iterations. 300 is sufficient for convergence
# at rank-30; higher values add runtime without measurable gain.

GENETIC_NMF_WES_Z_MIN: float = 1.5
# Minimum WES burden |Z-score| (|β/SE|) for a gene to contribute to the
# incoherence penalty. Genes with weaker WES signal than this threshold get
# w_i = 0 and do not influence the regularization, preventing noise injection.

GENETIC_NMF_SHET_SCALE: float = 5.0
# Multiplier applied to the GeneBayes shet posterior when computing the final
# gene weight: w_i = burden_z × (1 + SHET_SCALE × shet_post). Constrained genes
# (shet ~ 0.05–0.2) get up to a 2× boost; near-zero shet genes are unaffected.
# Each iteration resamples 80% of shared genes; the SD of bootstrap correlations
# is the SE used for z-score computation and uncertainty quantification.

GENETIC_NMF_EQTL_CONCORDANT_BOOST: float = 1.5
# Weight multiplier when WES and eQTL signals agree on direction. Applied to
# the final gene weight w_i before NMF to amplify genes with converging evidence.

GENETIC_NMF_EQTL_DISCORDANT_DISCOUNT: float = 0.5
# Weight multiplier when WES and eQTL signals disagree on direction. Reduces
# but does not zero the weight — the gene may still have real signal through
# one pathway; we just trust it less.

GENETIC_NMF_EQTL_ONLY_WEIGHT: float = 0.5
# Relative weight given to genes with eQTL evidence but no WES signal.
# w_i = |eQTL_Z| × EQTL_ONLY_WEIGHT. The 0.5 discount reflects that eQTL
# alone (without rare-variant orthogonal support) is a weaker direction signal.

FINGERPRINT_MIN_GENE_OVERLAP: int = 50
# Minimum shared genes required between the disease DE vector and a perturbation
# fingerprint for correlation to be computed. Below this threshold the correlation
# is too noisy to be meaningful given the bootstrap SE.

USE_FINGERPRINT_BETA: bool = True
# When True, `load_replogle_betas` prefers `signatures_fingerprint.json.gz` for
# GWAS-anchored genes and raw `signatures.json.gz` for non-GWAS nominees.
# Set to False to revert to raw log2FC signatures everywhere (baseline mode).

FINGERPRINT_SVD_COSINE_MIN: float = 0.30
# Minimum cosine similarity between a non-GWAS perturbed gene's SVD loading vector
# and the mean GWAS gene loading vector for the gene to be nominated as a
# fingerprint-based candidate. Genes below this threshold co-vary in latent
# directions unrelated to the GWAS signal and are not nominated.

FINGERPRINT_DISEASE_R_THRESHOLD: float = 0.20
# Minimum |r| required for a Perturb-seq KO to be nominated as a disease-fingerprint
# candidate. Only KOs with r ≤ −FINGERPRINT_DISEASE_R_THRESHOLD (i.e. KO anti-correlates
# with the disease DEG profile) pass the floor gate. After this gate, nominees are
# further capped at FINGERPRINT_MAX_FP_NOMINEES (top-N by most negative r) to prevent
# the smooth r distribution from flooding the gene list.
# RA calibration: IL6R r=−0.241 passes at 0.20; DHODH r=−0.137 / TYK2 r=−0.168 do not
# but are recovered via SVD cosine path (cosine ≥ 0.30 in GWAS centroid space).

FINGERPRINT_MAX_FP_NOMINEES: int = 200
# Hard cap on the number of disease-fingerprint nominees added per run (after r floor gate).
# The r distribution is smooth with no natural gap, so a top-N cap prevents the fingerprint
# path from dominating the gene list. 200 adds O(200) functionally-driven reversal candidates
# on top of O(50–450) SVD cosine nominees. Genes are taken in order of most negative r.

USE_SVD_GENE_NOMINATION: bool = True
# When True, the orchestrator replaces the full Perturb-seq union (all KO genes)
# with SVD cosine-nominated genes only (those co-regulated with the GWAS centroid
# in latent SVD space, cosine ≥ FINGERPRINT_SVD_COSINE_MIN). This collapses
# ~2000 mechanistic candidates to ~50 principled nominees. Set False to revert to
# the full Perturb-seq union (session 92 behaviour).

# ---------------------------------------------------------------------------
# GPS disease signature — genetic credibility weighting
# ---------------------------------------------------------------------------

GPS_CAUSAL_DE_L2G_THRESHOLD: float = 0.20
# L2G score above which a differentially expressed gene is considered
# "genetically causal DE" (expression change driven by a GWAS-implicated variant).
# Genes above this threshold get upweighted in the GPS disease signature.

GPS_CAUSAL_DE_WEIGHT: float = 1.50
# Weight multiplier for GWAS-colocalized DE genes in the GPS disease signature.
# Compounds that reverse these genes' expression are more likely to be on-mechanism.

GPS_REACTIVE_DE_WEIGHT: float = 0.60
# Weight multiplier for DE genes with no genetic instrument (L2G < threshold).
# These are likely downstream of disease, not causal; de-emphasised in the signature.

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

GPS_Z_RGES_PROGRAM: float = 3.5
# Z_RGES threshold for program-level GPS screens. Matched to GPS_Z_RGES_DEFAULT (3.5σ):
# the BGRD permutation normalises for signature size, so Z-scores are comparable across
# disease-state and program screens. 2.0σ produced ~4,149 hits (19.6% of the LINCS
# library) — no discriminative power. 3.5σ gives O(10–100) hits per program.

GPS_MAX_HITS: int = 500
# Safety cap for the non-z-scored GPS fallback path (top_n by |RGES|).
# Not applied when GPS output contains Z_RGES column — threshold governs in that case.

GPS_Z_STEPOFF_MIN_RATIO: float = 3.0
# Gap-detection sensitivity for the Z_RGES step-off threshold.
# The step-off is the largest gap in the sorted Z distribution.
# A gap is accepted as a real signal/noise boundary only when it is at least
# GPS_Z_STEPOFF_MIN_RATIO × the median inter-compound gap.
# Larger values = stricter (fewer hits); 3.0 requires the gap to be 3× typical spacing.

GPS_BGRD_MIN_GENES: int = 500
# Minimum signature genes for BGRD elbow-trim. GPS sets n_permutations=n_sig_genes;
# 500 completes in ~2h on Apple Silicon (Rosetta2); 700 exceeds the 6h GPS internal timeout.

GPS_BGRD_MAX_GENES: int = 500
# Maximum signature genes after elbow-trim. Pinned to 500 so all disease screens share
# BGRD__size500.pkl regardless of DEG count.

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
