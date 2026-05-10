# Causal Graph Engine — Build State
Last updated: 2026-05-09 (v1.0 freeze — GPS results assembled, outputs committed, merged to main)

---

## Status: v1.0 — FROZEN

Active diseases: **CAD + RA**.

Tests: **356 targeted tests passing**.

---

## Active diseases

### CAD (coronary artery disease)
- **Last full run:** Session 132 — 1,689 targets; 7/7 benchmarks found (100%); 4/7 sign correct vs Schnitzler KD phenotype
  - OTA direction correct: PLPP3, NOS3, COL4A1, COL4A2
  - OTA direction wrong (cNMF mechanism mismatch): EXOC3L2, CALCRL (γ<0 but KD↑risk), LOX (γ>0 but KD↓risk)
  - Benchmark convention: expected_sign = Schnitzler KD phenotype (+1=KD↑risk, −1=KD↓risk); PGF absent from CRISPRi library
- γ architecture: **cNMF k=60 only** — GeneticNMF removed from CAD OTA (all-positive τ* overwhelms cNMF sign for atheroprotective suppressors)
- β source: Schnitzler GSE210681 cNMF MAST betas (`cnmf_mast_betas.npz`, 2,643 perts × 60 programs)
- γ source: S-LDSC τ* (cNMF 31 programs with τ*>0, from `CAD_cNMF_program_taus.json`)
- Perturb-seq: Schnitzler GSE210681 (cardiac endothelial)

### RA (rheumatoid arthritis)
- **Last full run:** Session 132 — 496 targets, 11/11 benchmarks found (100% recovery)
  - Sign direction not meaningful: all GeneticNMF τ*>0 → all OTA γ>0; rank by |γ| is the discriminator
  - ICOS γ=+0.661 (expected −1): known open item; may reflect CD4+ T ICOS biology driving heritable programs
  - Benchmarks: TYK2, JAK2, CTLA4, IL12RB2, ICOS, TRAF3IP2, CD226, IL2RA, PTPN22, REL, TRAF1 (11 genes)
- **γ architecture: GeneticNMF OTA** (Stim48hr condition)
  - β = `genetic_nmf_de_betas.npz` (11,281 KO genes × 30 programs; DESeq2 z-score projected onto Vt)
  - τ* = `RA_GeneticNMF_Stim48hr_program_taus.json`, normalized by max_pos (C25 = 1.0)
  - **8 programs with τ*>0**: C25 (1.000), C01 (0.543), C12 (0.195), C04 (0.104), C24 (0.077), C17 (0.037), C02 (0.011), C28 (0.005)
  - Raw `program_taus` used (not eQTL-direction-corrected); direction from β signs
  - All 11,281 perturb-seq KOs scoreable → discovery without prior GWAS locus
- GWAS: GCST90132222 (Sakaue 2021, 35,871 seropositive cases)
- Perturb-seq: CZI CD4+ T — `GWCD4i.DE_stats.h5ad` ✓, `genetic_nmf_de_betas.npz` ✓

---

## γ architecture (settled — Session 131)

### CAD — cNMF k=60 only
`γ(gene→CAD) = Σ_P β(gene→P) × (τ*(P) / max_τ*)`
- β from Schnitzler GSE210681 CRISPRi-seq (MAST z-scores projected onto cNMF Vt)
- τ* from `CAD_cNMF_program_taus.json` (31 programs with τ*>0)
- GeneticNMF NOT used for CAD: all-positive τ* flips sign for atheroprotective suppressors

### RA — GeneticNMF Stim48hr
`γ(gene→RA) = Σ_P β(gene→P) × (τ*(P) / max_τ*)`
- β from CZI CD4+ T genetic_nmf_de_betas.npz
- τ* = raw `program_taus` (not eQTL-direction-corrected); direction from β signs
- 8 programs with τ*>0; all 11,281 CZI KO genes scoreable

### Sign convention (OTA γ)
- `γ < 0`: KO reduces heritable-program activity (gene drives disease programs → inhibitor target)
- `γ > 0`: KO amplifies heritable-program activity
- For RA: all γ>0; rank by |γ| is the meaningful discriminator

---

## Key constants (config/scoring_thresholds.py)

```python
SLDSC_GAMMA_FLOOR    = 0.02   # |γ(P→trait)| < floor → excluded from OTA sum
OTA_GAMMA_MIN        = 0.01   # minimum |OTA γ| to write gene edge to Tier 3 graph
GPS_Z_RGES_DEFAULT   = 3.5    # disease-state reversal threshold
GPS_Z_RGES_PROGRAM   = 3.5    # program reversal threshold (matched to default)
```

---

## v1.0 release checklist

| Item | Status |
|------|--------|
| 356 tests passing | ✅ |
| CAD: 1,689 targets, 7/7 benchmarks, 4/7 sign correct | ✅ |
| RA: 496 targets, 11/11 benchmarks | ✅ |
| GPS results assembled into tier4 checkpoints | ✅ |
| Reproducibility assets pinned (gene lists + τ* JSON + LOO discounts) | ✅ |
| README: full download + compute instructions | ✅ |
| methods_v5.md: GPS threshold + sort order corrected | ✅ |
| outputs/ PNGs + anchor_clusters JSONs committed | ✅ |
| feature/rna-fingerprinting merged to main | ✅ |
| Tagged v1.0.0 | ✅ |

## GPS nomination (v1.0)

### Programs selected
- CAD: P14 (τ*=0.092), P43 (τ*=0.083), P26 (τ*=0.075)
- RA: C25 (τ*_norm=1.000), C01 (0.543), C12 (0.195)

### Novel gene nominations (no GWAS signal)
CAD (ECM atherogenic clusters, all γ<0, KD↓risk, inhibitable proteins):
- PLEKHA1: rank 64, γ=−0.869 (COL4A1 cluster; phosphoinositide-binding)
- GIT1: rank 239, γ=−0.623 (COL4A2/FN1 cluster; focal adhesion ECM integrator)
- ELOVL2: rank 285, γ=−0.587 (LOX cluster; fatty acid elongase, lipid-ECM crosstalk)
- NPR2: rank 313, γ=−0.569 (COL4A2/FN1 cluster; natriuretic peptide receptor B)
- ROCK1: rank 551, γ=−0.447 (COL4A1 cluster; Rho kinase; ROCK inhibitors clinical)

RA (cosine similarity to IL12RB2/CD226/TRAF1):
- NUGGC: rank 8, γ=+0.477 (Criterion B: cosine=0.976)
- CRTAM: rank 17, γ=+0.398 (Criterion B: cosine=0.871, CD226 paralogue)
- MACC1: rank 175, γ=+0.236 (Criterion A+B: CD226 cluster + cosine=0.915)

---

## Architecture quick reference
```
orchestrator/pi_orchestrator_v2.py      — 5-tier pipeline (analyze_disease_v2 / run_tier4)
config/scoring_thresholds.py            — all numeric constants with citations
models/disease_registry.py             — canonical disease / EFO / GWAS mapping
pipelines/ldsc/runner.py               — S-LDSC τ* enrichment + eQTL direction γ
pipelines/ldsc/gamma_loader.py         — load signed program γ for OTA
steps/tier{1-5}_*/                     — per-tier pipeline steps
pipelines/ota_beta_estimation.py        — Tier1/2/3 β estimation
pipelines/gps_disease_screen.py        — GPS disease-state + program reversal (parallel Docker)
mcp_servers/perturbseq_server.py        — fingerprinting + SVD nomination + disease matching
outputs/plot_long_island.py            — long-island multi-panel plot
graph/                                 — Kùzu graph DB + RDF/Turtle export
```

---

## Known non-bugs
- GPSM3 → LDL-C/CAD/CRP E-value=1.76 flagged (harmless instrument quality warning)
- n_tier3_edges=0 is CORRECT (LINCS removed in session 57)
- ICOS γ=+0.661 for RA (expected −1): not a bug; reflects heritable program biology in CD4+ T cells

---

## Citation

Ota M et al. Causal modelling of gene effects from regulators to programs to traits. *Nature* (2026). https://doi.org/10.1038/s41586-025-09866-3
