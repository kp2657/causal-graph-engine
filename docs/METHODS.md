# Methodological Sources

This document details the statistical and computational methods used in each pipeline tier, with primary citations.

---

## Core causal-effect formula (OTA)

The pipeline scores every gene–disease pair using the **Ota et al. causal-effect estimator**:

```
γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait})
```

where:
- **P** indexes latent biological programs (NMF components derived from Perturb-seq or Reactome pathways)
- **β_{gene→P}** = gene's perturbation effect on program P (from Perturb-seq KO data, MR, pQTL, or rare-variant burden)
- **γ_{P→trait}** = program's GWAS-derived causal effect on the trait (L2G-weighted hypergeometric enrichment, S-LDSC h² enrichment, or MR-IVW estimate)

**Primary citation:**
> Ota M et al. "Multimodal causal inference integrating genetic perturbation data to identify disease-relevant programs." *bioRxiv* 2024. https://doi.org/10.1101/2024.07.12.603296

---

## Evidence tiers (v0.2.0)

Each gene receives an evidence tier that governs its β estimation and SCONE weighting.

| Tier | Label | β source | Criterion |
|------|-------|----------|-----------|
| 1 | `Tier1_Interventional` | Perturb-seq KO (Replogle 2022) | Direct CRISPRi knockdown β |
| 2 | `Tier2_Convergent` | MR-IVW or OT L2G ≥ 0.10 | ≥2 MR instruments OR strong GWAS co-localisation |
| 2p | `Tier2p_pQTL_MR` | pQTL-MR (Sun 2023 / deCODE) | Protein-level instrument, p < 0.05 |
| 2rb | `Tier2rb_RareBurden` | Rare variant burden (gnomAD WES) | Exome-wide significant burden test |
| 2pt | `Tier2pt_ProteinChannel` | Direct protein→disease arc | Coding variant with phenome evidence |
| 2.5 | `Tier2.5_eQTL_direction` | eQTL direction only | COLOC H4 0.50–0.80; direction assigned, σ=0.50 |
| 3 | `Tier3_Provisional` | Synthetic β (Reactome pathway) | Gene in disease-associated pathway, no direct perturbation |
| 3 | `Tier3_Provisional` (OT fallback) | OT genetic score proxy | ota_gamma = 0.65 × OT_L2G when NMF programs yield zero contribution |

Genes with no evidence above Tier 3 use `provisional_virtual` β estimates.

---

## Tier 1 — Interventional evidence (Perturb-seq)

**Method:** Genome-wide CRISPRi perturbation of RPE-1 cells (Replogle et al. 2022). Each gene's KO produces a transcriptional β vector across NMF programs. β values enter the OTA formula as the mechanistic arm.

**β scale:** allelic-effect scale aligned to Open Targets L2G scores. Beta magnitude is calibrated so that the OTA γ for a well-validated GWAS anchor (e.g. CFH in AMD, HMGCR in CAD) reaches ~0.5–0.7.

**Essential gene sink:** genes whose Perturb-seq KO maximally perturbs all programs equally (RNA polymerase subunits, DNA primase, translation factors) have inflated OTA γ. These are identified by `max(|β_P × γ_P|) > 0.8 AND |ota_gamma| > 2.0` and ranked last regardless of raw OTA γ magnitude.

**Citations:**
> Replogle JM et al. "Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq." *Cell* 185(19):2559–2575 (2022). https://doi.org/10.1016/j.cell.2022.05.013

---

## Tier 2 — Convergent genetic evidence

### Mendelian Randomisation (MR-IVW)

Two-sample MR using GWAS summary statistics. Instruments = genome-wide significant SNPs (p < 5×10⁻⁸) with F-statistic > 10. Effect direction consistency checked across IEU OpenGWAS studies.

**Citations:**
> Burgess S, Butterworth A, Thompson SG. "Mendelian randomization analysis with multiple genetic variants using summarized data." *Genet Epidemiol* 37(7):658–665 (2013). https://doi.org/10.1002/gepi.21758

> Hemani G et al. "The MR-Base platform supports systematic causal inference across the human phenome." *eLife* 7:e34408 (2018). https://doi.org/10.7554/eLife.34408

### GWAS locus-to-gene (L2G)

Open Targets Genetics L2G model: fine-mapping, eQTL co-localisation, protein-coding distance, and chromatin features jointly score the most likely causal gene at each GWAS locus. Minimum score for gene inclusion: 0.05 (gene seeding), 0.10 (gamma estimation).

**Citations:**
> Mountjoy E et al. "An open approach to systematically prioritize causal variants and genes at all published human GWAS trait-associated loci." *Nat Genet* 53:1527–1533 (2021). https://doi.org/10.1038/s41588-021-00945-5

### eQTL integration (GTEx / single-cell eQTL)

GTEx v8 tissue-specific eQTLs; colocalization tested with COLOC (H4 ≥ 0.80 = Tier 2, H4 0.50–0.80 = Tier 2.5 direction-only). Direction concordance with GWAS beta checked.

**Citation:**
> The GTEx Consortium. "The GTEx Consortium atlas of genetic regulatory effects across human tissues." *Science* 369(6509):1318–1330 (2020). https://doi.org/10.1126/science.aaz1776

> Giambartolomei C et al. "Bayesian test for colocalisation between pairs of genetic association studies using summary statistics." *PLoS Genet* 10(5):e1004383 (2014). https://doi.org/10.1371/journal.pgen.1004383

---

## Tier 3 — Provisional evidence

Single-instrument MR signals (F > 10, single GWAS hit). Not cross-validated by independent MR study. Also includes OT L2G fallback: when NMF programs yield zero OTA γ contribution for a genetically-supported gene (OT L2G ≥ 0.05), gamma is set to `0.65 × OT_L2G` and tier is `Tier2_Convergent` if L2G ≥ 0.10, else `Tier3_Provisional`.

---

## Co-evidence gate

Targets with no genetic co-evidence (OT genetic score < 0.05 and no drug in trials with phase > 0) are classified as `mechanistic_only` and receive an 80% effective-score discount during ranking. This prevents purely transcriptional candidates from outranking genetically-anchored targets. The multiplier is relaxed to 60% for `high_reward_mechanistic` targets (high TR + stability + genetic plausibility).

**Constant:** `CO_EVIDENCE_WEIGHT_UNGROUNDED = 0.20` (see `config/scoring_thresholds.py`)

---

## State-space modelling

### cNMF / NMF program discovery

Consensus Non-negative Matrix Factorisation (Kotliar et al.) applied to Perturb-seq expression matrices to define *k* latent programs. Each program is a weighted gene set. Programs are annotated against MSigDB Hallmarks and Reactome pathways.

**Citation:**
> Kotliar D et al. "Identifying gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq." *eLife* 8:e43803 (2019). https://doi.org/10.7554/eLife.43803

### Disease-state axis (CELLxGENE)

Log2 fold-change between disease and normal cells along a continuous disease axis derived from PCA of scRNA-seq data (CZ CELLxGENE census 2025-11-08). Cell types are coarsened into `pathological`, `intermediate`, and `healthy` states via pseudotime-informed clustering. Used to compute `state_influence` per gene.

### Therapeutic redirection (TR)

Direction and magnitude of disease-state shift upon gene KO, computed via Markov-chain transition matrix perturbation. The perturbation reduces both outflow from pathological states AND inflow from healthy states (T[healthy→pathological] edges), correcting for the degenerate case where pathological basins lack self-loops.

```
net_improvement = outflow_improvement + inflow_reduction
outflow_improvement = Σ_{i∈path, j∈healthy} max(T_perturbed[i,j] − T_baseline[i,j], 0)
inflow_reduction    = Σ_{i∈healthy, j∈path} max(T_baseline[i,j] − T_perturbed[i,j], 0)
```

**mechanistic_score** = `min(|TR| + state_influence × 0.3, 1.0)`

---

## Scoring formula (v0.2.0)

The primary ranking key is the raw OTA γ (`ota_gamma`), co-evidence gated:

```
ranking_key = −(|ota_gamma| × partition_multiplier)

partition_multiplier:
  genetically_grounded      → 1.0   (OT genetic score ≥ 0.05 OR drug in trials)
  high_reward_mechanistic   → 0.6   (high TR + stability, mechanistic-only)
  mechanistic_only          → 0.2   (no genetic anchor)
```

Additional reporting fields (not primary sort keys):
- `genetic_evidence_score` = OT L2G / genetic score [0, 1]
- `mechanistic_score` = min(|TR| + state_influence × 0.3, 1.0)
- `therapeutic_redirection` = net Markov-chain state shift [0, 1]
- `stability_score`, `entry_score`, `persistence_score`, `recovery_score`
- `target_score` = `ota_gamma` (alias for backward compatibility)

The **Phase F composite formula** (`core = 0.60 × genetic + 0.40 × mechanistic`) is defined in `CLAUDE.md` as the design target; it is not yet wired as the default sort key pending calibration against known benchmark genes.

**OTA γ cap:** `|ota_gamma|` is never artificially capped during ranking. Instead, essential/housekeeping genes with inflated γ are identified and sunk to the bottom (see Tier 1 section above). Known disease genes top out at |γ| ≈ 1.9 (CAD: LIPC); the hard cap constant `OTA_GAMMA_SCORE_CAP = 0.70` in `config/scoring_thresholds.py` is reserved for future composite-score calibration.

---

## GPS — Gene Perturbation Similarity screen

Phenotypic compound screen using the **Connectivity Map** approach. Disease transcriptional signature (log2FC disease vs normal, up to 1,000 genes after elbow-trimming) is scored against a compound library. Compounds ranked by **RGES** (Reverse Gene Expression Score).

Two screen modes:
1. **Disease-state reversal**: signature = disease vs healthy from CELLxGENE h5ad
2. **Program reversal**: signature = genes driving each causal NMF program (top γ contributors)

Z-score threshold for hit calling: `GPS_Z_RGES_DEFAULT = 2.0` (~5% FDR under N(0,1) null). Maximum 100 compounds returned per screen.

Background distribution (BGRD) cached at `data/gps_bgrd/` after first computation. With cached BGRD, runtime < 2 hours; without, up to 6 hours.

**Scope and limitations:** GPS is a **repurposing and hypothesis-generation tool**, not a validation of causal targets. CMAP profiles are measured in cancer cell lines (MCF7, A549, PC3) which do not match the disease-relevant cell types used for target nomination (cardiac endothelial for CAD, CD4+ T cells for RA). Transcriptional convergence between GPS compounds and genetic anchor genes (`convergent_genetic_targets_hypothesis` field) is an annotation-only signal — it is never used in target scoring or γ estimation.

**Citations:**
> Lamb J et al. "The Connectivity Map: using gene-expression signatures to connect small molecules, genes, and disease." *Science* 313(5795):1929–1935 (2006). https://doi.org/10.1126/science.1132939

> Chen B et al. "Reversal of cancer gene expression correlates with drug efficacy and reveals therapeutic targets." *Nat Commun* 8:16022 (2017). https://doi.org/10.1038/ncomms16022

---

## S-LDSC (stratified LD score regression)

Used to estimate heritability enrichment per NMF program (h² enrichment in programme gene sets vs genome-wide). Programs with significant enrichment get γ uplift.

**Citation:**
> Finucane HK et al. "Partitioning heritability by functional annotation using genome-wide association summary statistics." *Nat Genet* 47:1228–1235 (2015). https://doi.org/10.1038/ng.3404

---

## pQTL integration

Protein quantitative trait loci (plasma pQTLs) from deCODE Genetics and UK Biobank. Used to triangulate genetic effects at the protein level. p-value threshold: 0.05 (relaxed from GWAS threshold; appropriate for coding variants where cis-eQTL is absent).

**Citations:**
> Ferkingstad E et al. "Large-scale integration of the plasma proteome with genetics and disease." *Nat Genet* 53:1712–1721 (2021). https://doi.org/10.1038/s41588-021-00978-w

> Sun BB et al. "Plasma proteomic associations with genetics and health in the UK Biobank." *Nature* 622:329–338 (2023). https://doi.org/10.1038/s41586-023-06592-6

---

## SCONE (Sensitivity-aware Causal Network Optimization)

Post-OTA reweighting step that adjusts gene-γ edge confidence based on:
1. Cross-regime sensitivity (how stable the γ estimate is under β perturbation)
2. Polybic score (BIC-penalised OTA γ magnitude by evidence tier)
3. Bootstrap edge confidence (30 resamples of β matrix, fraction with |γ| > threshold)

Anchor genes (GWAS-validated) bypass SCONE downweighting.

---

## Validation modules

Five independent validation tests are run post-Tier 3 to flag potential confounds. These are **annotation-only** — none modify `ota_gamma` or the primary ranking.

### Pseudotime pre-branch test
NMF programs are labelled `mediator`, `consequence`, or `ambiguous` based on whether they activate before or after the normal→disease branch point in pseudotime. Programs labelled `consequence` receive a 0.5× γ discount (these are downstream of disease, not causal). Implemented in `pipelines/state_space/program_precedence.py`; results cached to `data/ldsc/results/{disease_key}_program_precedence.json`.

### SMR + HEIDI pleiotropic instrument test
For eQTL-MR Tier 2 genes: β_SMR = β_GWAS / β_eQTL (ratio estimator with delta-method SE). HEIDI flag raised when H3/(H3+H4) > 0.3, indicating the GWAS and eQTL signals may be driven by different causal variants (horizontal pleiotropy). Implemented in `pipelines/smr_heidi.py`; wired into `estimate_beta_tier2()`.

**Citation:**
> Zhu Z et al. "Integration of summary data from GWAS and eQTL studies predicts complex trait gene targets." *Nat Genet* 48:481–487 (2016). https://doi.org/10.1038/ng.3538

### Leave-locus-out (LOO) γ stability
The anchor gene's own ±1 Mb locus is excluded from the program χ² enrichment and τ is recomputed. Genes whose τ rank shifts by > 10 positions are flagged as `loo_stable=False` — their γ may be inflated by the anchor gene's own GWAS signal rather than genuine program enrichment. Implemented in `pipelines/ldsc/leave_locus_out.py`.

### FinnGen holdout validation
FinnGen R10 (no UK Biobank overlap) provides an independent GWAS cohort for replication. Program-level enrichment is recomputed using FinnGen summary statistics, and Spearman rank correlation between pipeline scores and FinnGen enrichment tests whether top targets replicate. Endpoints: CAD = I9_CHD, RA = M13_RHEUMA. Implemented in `pipelines/ldsc/finngen_validation.py`.

### Clinical trial AUROC
OpenTargets `knownDrugs` GraphQL query maps Phase 2/3 trial outcomes (success vs failure/discontinued) for all nominated genes. Mann-Whitney AUROC tests whether pipeline rank correlates with clinical success. Returns `skipped` if fewer than 5 labeled genes are available. Implemented in `pipelines/validation/clinical_trial_auc.py`.

---

## Scoring thresholds registry

All numeric thresholds (F-statistic floor, COLOC H4 minimum, L2G cutoffs, GPS Z-score, gamma caps) are centralised in `config/scoring_thresholds.py` with provenance citations. Import from this module rather than defining local constants.
