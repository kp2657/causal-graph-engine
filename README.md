# Causal Graph Engine

A genomics pipeline for genetically-grounded drug target prioritization, implementing the Ota et al. causal-effect decomposition:

```
γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait})
```

- **β** (gene → program): estimated from CRISPR Perturb-seq (interventional) and eQTL Mendelian randomization (genetic instrument)
- **γ** (program → trait): estimated from S-LDSC heritability enrichment of transcriptional programs

Both quantities use genetic instruments or direct perturbations to reduce reverse-causation bias. The Perturb-seq β is the strongest component — direct CRISPR knockout is interventional by design.

---

## What it produces

For each input disease, the pipeline outputs a ranked target list with:
- Genetically-grounded effect estimate (OTA γ) per gene
- Evidence tier (Tier1 = direct Perturb-seq, Tier2 = eQTL-MR, Tier3 = provisional)
- GPS compound screen results (disease-state reversal and program reversal)
- Kùzu graph database of all gene–program–trait edges

**Current validated results:**

| Disease | Cell type | Targets | Benchmark recovery | Top novel targets |
|---------|-----------|---------|-------------------|-------------------|
| Coronary artery disease | Cardiac endothelial | 1,689 | 7/7 Schnitzler KD phenotype | PLEKHA1, GIT1, NPR2, ELOVL2, ROCK1 |
| Rheumatoid arthritis | CD4+ T cell | 496 | 11/11 known targets | NUGGC, CRTAM, MACC1 |

Known validated targets recovered: TYK2 (RA rank 83), JAK2 (RA rank 159), CTLA4 (RA rank 141), IL12RB2 (RA rank 3), PTPN22 (RA rank 194).

---

## Requirements

| Resource | Requirement |
|---|---|
| RAM | 32 GB |
| Disk | ~50 GB (all data) |
| Python | 3.12 (conda environment) |
| Docker | GPS compound screening (optional) |
| Runtime (no GPS) | ~25 min |
| Runtime (with GPS) | ~4–6 h |

---

## Setup

All steps are required for a full run. The pipeline will fail rather than silently produce degraded results if data is missing.

### 1. Install dependencies

```bash
conda create -n causal-graph python=3.12
conda activate causal-graph
pip install -e ".[bio,chem,dev]"
```

### 2. Configure API credentials

```bash
cp .env.example .env
```

Edit `.env`:
- `OPENGWAS_JWT` — free JWT from [api.opengwas.io](https://api.opengwas.io) (**required** for Tier 2 MR; expires after 14 days — renew before each run)
- `NCBI_API_KEY` — free key from [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/) (recommended)

### 3. Download GWAS summary statistics

Required for S-LDSC heritability enrichment (γ estimation) and eQTL direction concordance.

```bash
# CAD — Aragam 2022 (GCST90132314, 181K cases, GRCh38 harmonised)
mkdir -p data/ldsc/sumstats
wget -O data/ldsc/sumstats/GCST90132314.h.tsv.gz \
  "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90132001-GCST90133000/GCST90132314/harmonised/GCST90132314.h.tsv.gz"

# RA — Sakaue 2021 seropositive (GCST90132222, 35K cases, GRCh38 harmonised)
wget -O data/ldsc/sumstats/GCST90132222.h.tsv.gz \
  "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90132001-GCST90133000/GCST90132222/harmonised/GCST90132222.h.tsv.gz"
```

### 4. Download LD score files (~678 MB)

Required to compute S-LDSC τ* program enrichment coefficients.

```bash
mkdir -p data/ldsc/ldscores
# baselineLD v2.2 LD scores (Gazal et al. 2017)
wget -r -nd -P data/ldsc/ldscores \
  "https://data.broadinstitute.org/alkesgroup/LDSCORE/baselineLD_v2.2_1000G_Phase3_EUR.tgz"
tar -xzf data/ldsc/ldscores/baselineLD_v2.2_1000G_Phase3_EUR.tgz -C data/ldsc/ldscores/
```

### 5. Compute S-LDSC program τ* coefficients (one-time, ~2–4 h per disease)

This produces `data/ldsc/results/{disease}_program_taus.json` — the γ(P→trait) weights central to the OTA formula. Requires the LDSC binary (see [LDSC installation](https://github.com/bulik/ldsc)).

```bash
# Install LDSC binary into the environment
pip install ldsc  # or follow https://github.com/bulik/ldsc

# Run enrichment for each disease
conda run -n causal-graph python -m pipelines.ldsc.runner run_all CAD
conda run -n causal-graph python -m pipelines.ldsc.runner run_all RA
```

Output files written to `data/ldsc/results/`:
- `CAD_cNMF_program_taus.json` — 60-program cNMF τ* for CAD
- `RA_GeneticNMF_Stim48hr_program_taus.json` — GeneticNMF τ* for RA
- `{disease}_gwas_variant_betas.json` — per-variant GWAS betas for eQTL concordance
- `{disease}_loo_discounts.json` — leave-locus-out stability factors

### 6. Download CELLxGENE single-cell h5ads (~600 MB)

Disease-matched scRNA-seq from CZ CELLxGENE Census 2025-11-08.

```bash
conda run -n causal-graph python -m pipelines.discovery.cellxgene_downloader download_all CAD
conda run -n causal-graph python -m pipelines.discovery.cellxgene_downloader download_all RA
```

### 7. Download Perturb-seq data

Required for Tier 1 interventional β estimates.

**CAD** — Schnitzler 2024 cardiac endothelial CRISPRi-seq ([GEO GSE210681](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE210681)):
```bash
mkdir -p data/perturbseq/schnitzler_cad_vascular
# Download GSE210681_ALL_log2fcs.txt.gz from GEO into that directory
```

**RA** — CZI CD4+ T Perturb-seq ([GEO GSE314342](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314342)):
```bash
mkdir -p data/perturbseq/czi_2025_cd4t_perturb/raw_cells
# Download the h5ad from GEO into that directory
```

### 8. Compute NMF betas (one-time, ~30 min per disease)

Projects Perturb-seq KO signatures onto NMF program axes to produce β(gene→program).

```bash
conda run -n causal-graph python -m pipelines.genetic_nmf run CAD
conda run -n causal-graph python -m pipelines.genetic_nmf run RA
```

Output: `data/perturbseq/{disease}/cnmf_mast_betas.npz` (CAD) and `genetic_nmf_de_betas.npz` (RA).

### 9. Compute GPS background distribution (one-time, ~90 min)

Builds the null permutation distribution used by all GPS compound screens.

```bash
docker pull binchengroup/gpsimage:latest

# Run any GPS screen once with no cached BGRD — it will compute and save BGRD__size500.pkl
conda run -n causal-graph python -m pipelines.gps_screen \
  --gene CCM2 --disease coronary_artery_disease --build_bgrd
```

Output: `data/gps_bgrd/BGRD__size500.pkl` (shared by all subsequent GPS runs — never recomputed).

Without Docker, GPS compound screening is skipped and runtime drops to ~25 min.

---

## Running the pipeline

```bash
conda activate causal-graph

# Full pipeline run (Tiers 1–5, ~4–6 h with GPS)
conda run -n causal-graph python -m orchestrator.pi_orchestrator_v2 analyze_disease_v2 "coronary artery disease"
conda run -n causal-graph python -m orchestrator.pi_orchestrator_v2 analyze_disease_v2 "rheumatoid arthritis"

# Re-run Tier 4+5 from a saved Tier 3 checkpoint (skips β/γ recomputation, ~30 min + GPS)
conda run -n causal-graph python -m orchestrator.pi_orchestrator_v2 run_tier4 "coronary artery disease"
conda run -n causal-graph python -m orchestrator.pi_orchestrator_v2 run_tier4 "rheumatoid arthritis"
```

Output files:
- `data/analyze_{disease}.md` — ranked target report with GPS results
- `data/analyze_{disease}.json` — full structured results
- `data/exports/{disease}.ttl` — RDF/Turtle graph export
- `data/checkpoints/{disease}__tier3.json` — Tier 3 checkpoint for fast Tier 4 reruns

---

## Architecture

```
orchestrator/pi_orchestrator_v2.py      — 5-tier pipeline entry point
steps/
  tier1_phenomics/                      — disease phenotyping + GWAS instrument selection
  tier2_pathway/                        — Perturb-seq β estimation + eQTL-MR upgrades
  tier3_causal/                         — OTA γ computation + causal graph construction
  tier4_translation/                    — target prioritization + GPS compound screens
  tier5_writer/                         — output assembly + report
pipelines/
  ota_beta_estimation.py                — Tier1 (Perturb-seq), Tier2 (eQTL-MR) β
  ota_gamma_estimation.py               — GWAS enrichment γ
  ldsc/runner.py                        — S-LDSC τ* computation from sumstats
  ldsc/gamma_loader.py                  — load signed program γ for OTA formula
  gps_disease_screen.py                 — GPS disease-state + program reversal screens
  state_space/                          — therapeutic redirection + TR scoring
mcp_servers/                            — live data servers (GWAS, OT, GTEx, gnomAD, …)
graph/                                  — Kùzu graph DB + RDF/Turtle export
config/scoring_thresholds.py            — all numeric thresholds (never inline)
models/disease_registry.py             — canonical disease / EFO / GWAS mapping
frozen/                                 — NMF program definitions + HGNC reference (tracked)
```

---

## Adding a new disease

1. Add entries to `models/disease_registry.py` and `graph/schema.py`
2. Download disease-relevant h5ad: `python -m pipelines.discovery.cellxgene_downloader download_all <KEY>`
3. Download GWAS sumstats and run S-LDSC enrichment (steps 3–5 above)
4. Download and process Perturb-seq data (steps 7–8 above)
5. Run `analyze_disease_v2 "<disease name>"`

---

## Evidence tiers

| Tier | Source | Causal basis |
|------|--------|-------------|
| Tier1_Interventional | CRISPR Perturb-seq (cell-type matched) | Direct perturbation |
| Tier2_Convergent | eQTL-MR (GTEx / OneK1K scQTL) | Genetic randomization of expression |
| Tier3_Provisional | In silico prediction | No direct causal basis |

---

## Tests

```bash
/opt/anaconda3/envs/causal-graph/bin/python -m pytest \
  tests/test_state_space_*.py tests/test_causal_*.py \
  tests/test_gps_*.py tests/test_scoring_*.py \
  tests/test_pipelines.py tests/test_pi_orchestrator_v2.py \
  -q --tb=short
```

---

## Citation

If you use this pipeline, please cite:

> Ota M, Spence JP, Zeng T, Dann E, Milind N, Marson A, Pritchard JK. Causal modelling of gene effects from regulators to programs to traits. *Nature*. 2026 Feb;650(8101):399–408. doi: [10.1038/s41586-025-09866-3](https://doi.org/10.1038/s41586-025-09866-3). PMID: 41372418.

**Primary data sources — please also cite:**

> Schnitzler GR, Kang H, Fang S, et al. Convergence of coronary artery disease genes onto endothelial cell programs. *Nature*. 2024;626(8000):799–807. doi: [10.1038/s41586-024-07022-x](https://doi.org/10.1038/s41586-024-07022-x). PMID: 38326615. (CAD Perturb-seq: GEO GSE210681)

> Zhu R, Dann E, Yan J, et al. Genome-scale perturb-seq in primary human CD4+ T cells maps context-specific regulators of T cell programs and human immune traits. *bioRxiv*. 2025 Dec 24. doi: [10.64898/2025.12.23.696273](https://doi.org/10.64898/2025.12.23.696273). (RA Perturb-seq: GEO GSE314342)

> Xing J, Tan M, Leshchiner D, et al. Deep-learning-based de novo discovery and design of therapeutics that reverse disease-associated transcriptional phenotypes. *Cell*. 2026 Mar 17:S0092-8674(26)00223-0. doi: [10.1016/j.cell.2026.02.016](https://doi.org/10.1016/j.cell.2026.02.016). PMID: 41850287. (GPS compound screening)

> Zeng T, Spence JP, Mostafavi H, Pritchard JK. Bayesian estimation of gene constraint from an evolutionary model with gene features. *Nature Genetics*. 2024;56:1632–1643. doi: [10.1038/s41588-024-01820-9](https://doi.org/10.1038/s41588-024-01820-9). PMID: 38977852. (shet constraint scores)

> Aragam KG, Jiang T, Goel A, et al. Discovery and systematic characterization of risk variants and genes for coronary artery disease in over a million participants. *Nature Genetics*. 2022;54:1803–1815. doi: [10.1038/s41588-022-01233-6](https://doi.org/10.1038/s41588-022-01233-6). PMID: 36474045. (CAD GWAS: GCST90132314)

> Sakaue S, Kanai M, Tanigawa Y, et al. A cross-population atlas of genetic associations for 220 human phenotypes. *Nature Genetics*. 2021;53:1415–1424. doi: [10.1038/s41588-021-00931-x](https://doi.org/10.1038/s41588-021-00931-x). PMID: 36333501. (RA GWAS: GCST90132222)

> Buniello A, Suveges D, Cruz-Castillo C, et al. Open Targets Platform: facilitating therapeutic hypotheses building in drug discovery. *Nucleic Acids Research*. 2025;53(D1):D1467–D1475. doi: [10.1093/nar/gkae1128](https://doi.org/10.1093/nar/gkae1128). PMID: 39657122.

> GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects across human tissues. *Science*. 2020;369(6509):1318–1330. doi: [10.1126/science.aaz1776](https://doi.org/10.1126/science.aaz1776). PMID: 32913098.

---

## License

MIT — see [`LICENSE`](LICENSE).
