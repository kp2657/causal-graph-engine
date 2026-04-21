# OTA Causal Algorithm

## Overview

The causal graph engine implements the **Ota et al. framework** (Nature 2026) for identifying genetically causal drug targets. The core idea is to decompose the causal effect of a gene on a disease trait into a sum of contributions mediated through gene programs:

```
γ_{gene→trait} = Σ_P  β_{gene→P} × γ_{P→trait}
```

Where:
- **β_{gene→P}** = the causal effect of perturbing gene *G* on transcriptional program *P*
- **γ_{P→trait}** = the causal enrichment of program *P* for disease trait *T*
- **γ_{gene→trait}** = the aggregate causal score for gene *G* on trait *T*

This product is causal — not correlational — because both β and γ are estimated using **genetic instruments** (GWAS variants, eQTLs, pQTLs) that remove reverse causation.

---

## Why this is causal

Traditional target identification uses:
- Co-expression (confounded by common environmental drivers)
- Differential expression (confounded by disease state)
- Protein interaction networks (no causal directionality)

OTA replaces these with instruments grounded in human genetics:

| Estimation | Source | Why causal |
|---|---|---|
| β_{gene→P} | CRISPR Perturb-seq log2FC, eQTL-MR | Random assignment of CRISPR alleles; eQTL is a natural experiment |
| γ_{P→trait} | GWAS gene set enrichment, S-LDSC | GWAS variants are randomly inherited (Mendelian randomization) |

---

## Worked example: CFH → AMD

**Gene programs** in AMD: complement_activation, RPE_oxidative_stress, lipid_metabolism, VEGF_signaling, ...

**Step 1 — β_{CFH→complement_activation}:**
- CFH is not a perturb target in the Replogle 2022 RPE1 library (CRISPR knockout screen covers ~8k genes)
- → β falls back to eQTL-MR: CFH cis-eQTL in GTEx retina, but weak COLOC H4 → provisional_virtual tier
- → effective β ≈ 0 (no direct Perturb-seq measurement; β estimated via OT L2G proxy)

**Step 2 — γ_{CFH-locus→AMD}:**
- CFH OT L2G score = 0.86 → strong genetic prioritisation by Open Targets
- COLOC H4 between CFH eQTL and AMD GWAS = 0.87 → shared causal variant confirmed
- → causal_gamma = 0.561 (derived from OT L2G + COLOC fusion; Bayesian Mode 3)

**Step 3 — OTA summation:**
- γ_{CFH→AMD} = Σ_P (β_{CFH→P} × γ_{P→AMD}) = 0 (no program-mediated β)
- causal_gamma is sourced directly from genetic instrument (OT L2G = 0.86 → causal_gamma = 0.561)
- → ota_gamma = 0.561, partition = genetically_grounded
- → rank: #2 in AMD target list (v0.2.0 validated run)

**Note on sign convention:** CFH loss-of-function (Y402H variant) increases AMD risk. The positive ota_gamma
reflects that the locus is strongly causal for AMD regardless of expression direction; sign is preserved where
a perturb β is available (Tier1/Tier2) but not for locus-only estimates.

---

## β Estimation Tier Hierarchy

β is estimated for each (gene, program) pair using a priority-ordered evidence chain. Higher tiers provide stronger causal evidence and narrower uncertainty.

| Tier | Label | Data source | Causal basis | Sigma |
|---|---|---|---|---|
| 1 | `Tier1_Interventional` | CRISPR Perturb-seq (Replogle 2022 RPE1/K562; Schnitzler 2023 HCASMC) | Direct intervention | ~0.15 |
| 2a | `Tier2_Convergent` | GTEx v8 eQTL-MR + COLOC H4 ≥ 0.80 | Mendelian randomization | ~0.25 |
| 2b | `Tier2_Convergent` | Open Targets credible-set instruments | GWAS fine-mapping | ~0.25 |
| 2c | `Tier2c_scEQTL` | eQTL Catalogue sc-eQTL (OneK1K, Blueprint) | Cell-type-specific MR | ~0.30 |
| 2p | `Tier2p_pQTL_MR` | pQTL-MR (UKB-PPP, INTERVAL, deCODE) | Protein-level instrument | ~0.30 |
| 2L | `Tier2L_LatentHijack` | Cross-disease motif transfer | Indirect genetic support | ~0.40 |
| 2.5 | `Tier2_eQTL_direction` | GTEx eQTL direction (COLOC H4 < 0.80) | Direction only, no magnitude | 0.50 |
| 2rb | `Tier2rb_RareBurden` | UKB WES rare-variant burden direction | LoF enrichment | 0.60 |
| 3 | `Tier3_Provisional` | LINCS L1000 / Perturb-seq (cell-line mismatched) | Interventional, cell-type mismatch | ~0.50 |
| Virtual | `provisional_virtual` | Pathway membership annotation | No causal basis; annotation proxy | 1.00 |

**Rules:**
- Co-expression is never used (no causal directionality)
- Synthetic pathway betas (Reactome-derived) were removed in v0.2.0 — they lacked causal basis
- Sigma represents prior width in the Bayesian update; larger sigma = less informative

### pQTL-MR rationale

pQTLs (protein quantitative trait loci) are used for genes where the causal variant changes protein abundance rather than mRNA level. Examples:
- **CFH Y402H** (rs1061170): complement regultor; no cis-eQTL in GTEx retina, but strong pQTL in UKB-PPP
- **LPA** kringles: lipoprotein(a); driven by repeat number polymorphism, not transcription
- **TREM2 R47H**: microglia receptor; coding variant changes protein stability

Source: UKB-PPP (Sun et al. 2023, Nature); INTERVAL (Sun 2018, Nature); deCODE (Ferkingstad 2021, Nature Genetics)

---

## γ Estimation

γ_{P→trait} quantifies how strongly a gene program is enriched for disease causal variants. Three estimators are used:

### Mode 1 — Hypergeometric enrichment (cNMF GWAS overlap)

Fisher's exact test for overlap between program gene set and GWAS anchor genes:

```
p_enrich = P(X ≥ k | Hypergeometric(N, K, n))
gamma_raw = min(-log10(p_enrich) / 5.0, 1.0)
```

Where N = genome size (20,000), K = GWAS hit count, n = program size, k = overlap count.

Threshold: odds ratio > 1 and p < 0.1.

### Mode 2 — S-LDSC heritability enrichment

Stratified LD-score regression (Finucane et al. 2015, Nature Genetics) partitions GWAS heritability across functional annotations. The program gene set is used as an annotation; the τ coefficient measures per-SNP heritability enrichment.

```
gamma_ldsc = τ_annotation   (when z_score > 0)
```

### Mode 3 — Bayesian fusion

The final γ fuses LDSC (Mode 2) and OT L2G overlap (Mode 1 proxy) using precision-weighted combination:

```
γ_fused = (τ_ldsc × w_ldsc + γ_ot × w_ot) / (w_ldsc + w_ot)

w_ldsc = min(z_ldsc, 10)     # capped to avoid dominance
w_ot   = 0.20                # prior weight for OT evidence
```

Thresholds: OT L2G score ≥ 0.10 for contribution; LDSC z_score > 0.

---

## Final Target Score (Phase F)

The OTA causal gamma feeds into a multi-component target scoring formula:

```
core_score = 0.60 × genetic_component + 0.40 × mechanistic_component

genetic_component  = ota_gamma / 0.70           (GWAS-grounded causal score)
mechanistic_component = min(|TR| + state_influence × 0.3, 1.0)

final_score = core_score × t_mod × risk_discount

t_mod        = clamp(1 + 0.15×OT_tractability + 0.10×trial_phase - 0.10×safety_flags, 0.5, 1.5)
risk_discount = max(0.10, 1 - 0.20×escape_risk - 0.15×failure_risk)
```

Where:
- **TR** (Therapeutic Redirection): cosine similarity between gene KO signature and disease→normal axis
- **state_influence**: transcriptional influence on disease state axis from CELLxGENE h5ad
- **OT_tractability**: Open Targets small-molecule / antibody tractability score
- **trial_phase**: max clinical trial phase in disease indication
- **escape_risk / failure_risk**: from GPS compound screen and historical MoA failure rates

**Design notes on the 60/40 split:**
This weighting was chosen to prioritize genetic evidence as the primary causal anchor (preventing purely transcriptional candidates from outranking genetically validated ones), while preserving mechanistic signal that predicts druggability. Sensitivity analysis at 50/50 and 70/30 on AMD Run 22 produced consistent top-10 rankings for CFH, ARMS2, C3, CFB.

---

## Program Definition

Gene programs (*P*) represent co-regulated transcriptional modules. Two sources are used:

1. **MSigDB Hallmark gene sets** (50 programs, curated): used for disease-agnostic mechanistic annotation
2. **cNMF (consensus NMF)** from CELLxGENE disease-matched h5ad: data-driven, disease-specific programs

Programs are defined per disease based on cell-type context (RPE cells for AMD, HCASMC/HAEC for CAD, PBMC for IBD/RA/SLE).

---

## Data Flow

```
GWAS anchor genes (OT L2G, IEU OpenGWAS)
        │
        ▼
Tier 1 agent — β estimation (Perturb-seq, eQTL-MR, pQTL-MR)
        │
        ▼
Tier 2 agent — γ estimation (GWAS enrichment, S-LDSC, OT L2G)
        │
        ▼
OTA summation → γ_{gene→trait} per gene
        │
        ▼
Phase F scoring → ranked target list
        │
        ▼
GPS screen → compound reversers of disease state
```

---

## References

- Ota et al. (2026) Nature — original OTA framework
- Replogle et al. (2022) Cell — genome-wide Perturb-seq (RPE1, K562)
- Schnitzler et al. (GEO GSE210681) — CAD vascular Perturb-seq (HCASMC/HAEC)
- Mountjoy et al. (2021) Nature Genetics — Open Targets Locus2Gene (L2G)
- GTEx Consortium (2020) Science — GTEx v8 eQTL atlas
- Finucane et al. (2015) Nature Genetics — S-LDSC
- Giambartolomei et al. (2014) PLoS Genetics — COLOC
- Sun et al. (2023) Nature — UKB-PPP plasma proteomics
- Ferkingstad et al. (2021) Nature Genetics — deCODE proteomics
- Staley & Burgess (2017) Genetic Epidemiology — MR weak-instrument criterion
