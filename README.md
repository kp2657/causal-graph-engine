# Causal Graph Engine

A genomics pipeline for causal drug target discovery, implementing the Ota et al. causal-effect decomposition:

```
γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait})
```

Where:
- **β** (gene → program): estimated from CRISPR Perturb-seq and eQTL Mendelian randomization
- **γ** (program → trait): estimated from GWAS enrichment (OT L2G, S-LDSC, Reactome)

Both quantities are genetic-instrument-grounded, removing reverse causation from target scores. See [`docs/OTA_ALGORITHM.md`](docs/OTA_ALGORITHM.md) for method details.

---

## What it produces

For each input disease, the pipeline outputs a ranked target list with:
- Causal effect estimate (γ) per gene
- Evidence tier (Tier1 = direct Perturb-seq, Tier2 = eQTL-MR, Tier3 = provisional)
- GPS compound screen results (disease-state reversal and program reversal)
- Kùzu graph database of all causal edges

**Validated results (v0.2.3):**

| Disease | Targets | Tier1 edges | Tier2 edges | Top targets |
|---------|---------|-------------|-------------|-------------|
| Coronary artery disease | 1,586 | 1,448 | 184 | LIPC, PROCR, SORT1, LPA, IL6R |
| Rheumatoid arthritis | 244 | 23 | 221 | NFKBIE, UQCC2, ACTR3, MYC, CTLA4 |

Known validated targets recovered: PCSK9 (CAD rank 132), LPA (CAD rank 47), IL6R (CAD rank 90 / RA rank 140), JAK1 (RA rank 201), CTLA4 (RA rank 44).

---

## Requirements

| Resource | Requirement |
|---|---|
| RAM | 32 GB |
| Disk | ~20 GB (all required data) |
| Python | 3.12 |
| Docker | GPS compound screening (optional but recommended) |
| Runtime (CAD, no GPS) | ~25 min |
| Runtime (CAD, with GPS) | ~4 h |

---

## Exact reproduction

The `frozen/` directory (tracked in this repo) contains the reference files used to generate the published results:

| File | Contents |
|---|---|
| `frozen/CAD_program_taus.json` | S-LDSC τ coefficients for CAD programs |
| `frozen/RA_program_taus.json` | S-LDSC τ coefficients for RA programs |
| `frozen/CAD_programs.json` | NMF program definitions for CAD |
| `frozen/RA_programs.json` | NMF program definitions for RA |

Four files are distributed as GitHub Release assets. Download them before running:

```bash
python scripts/download_frozen_data.py
```

This places:
- `data/api_cache.sqlite` — frozen OT L2G, GTEx, gnomAD API responses (46 MB)
- `data/gps_bgrd/BGRD__size500.pkl` — GPS null distribution (11 MB)
- `data/checkpoints/coronary_artery_disease__tier3.json` — CAD Tier 3 checkpoint (5.6 MB)
- `data/checkpoints/rheumatoid_arthritis__tier3.json` — RA Tier 3 checkpoint (9.2 MB)

Without these, `run_tier4` requires a full Tiers 1–3 re-run (~25 min, needs all data downloaded), and OT L2G scores and GPS results may differ from the published values.

---

## Setup

All steps below are required. The pipeline will fail rather than silently produce degraded results if data is missing.

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

### 3. Download static reference data (~574 MB)

gnomAD constraint, HGNC gene table, Reactome pathways.

```bash
python scripts/fetch_static_data.py
```

### 4. Download CellxGENE scRNA-seq h5ads (~600 MB)

Disease-matched single-cell data from CZ CELLxGENE Census 2025-11-08.

```bash
python -m pipelines.discovery.cellxgene_downloader download_all CAD
python -m pipelines.discovery.cellxgene_downloader download_all RA
```

### 5. Download Perturb-seq data (~16 GB)

Required for Tier 1 interventional β estimates.

**CAD** — Schnitzler 2023 HAEC/HCASMC ([GEO GSE210681](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE210681)):
```bash
mkdir -p data/perturbseq/schnitzler_cad_vascular
# Download GSE210681_ALL_log2fcs.txt.gz from GEO into that directory
```

**RA** — CZI CD4+ T Perturb-seq ([GEO GSE314342](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314342)):
```bash
mkdir -p data/perturbseq/czi_2025_cd4t_perturb
# Download the h5ad from GEO into that directory
```

### 6. Download S-LDSC summary statistics (~727 MB)

Required for heritability enrichment γ estimation.

```bash
python -m pipelines.ldsc.setup download_all
```

### 7. Verify all data is in place

```bash
python scripts/check_data.py
```

All checks must pass before running the pipeline.

### 8. Pull GPS Docker image (optional — for compound screening)

```bash
docker pull binchengroup/gpsimage:latest
```

Without Docker, GPS compound screening is skipped and runtime drops to ~25 min.

---

## Running the pipeline

```bash
conda activate causal-graph

# Full pipeline run (Tiers 1–5, ~4h with GPS)
python -m orchestrator.pi_orchestrator_v2 analyze_disease_v2 "coronary artery disease"
python -m orchestrator.pi_orchestrator_v2 analyze_disease_v2 "rheumatoid arthritis"

# Re-run Tier 4+5 from a saved Tier 3 checkpoint (skips β/γ recomputation, ~30 min)
python -m orchestrator.pi_orchestrator_v2 run_tier4 "coronary artery disease"
```

Output files:
- `data/analyze_{disease}.json` — full results (targets, edges, GPS compounds, narrative)
- `data/graph_{disease}.kuzu` — Kùzu graph database
- `data/checkpoints/{disease}__tier3.json` — Tier 3 checkpoint for fast Tier 4 reruns

---

## Architecture

```
orchestrator/pi_orchestrator_v2.py          — 5-tier pipeline entry point
  agents/tier1_phenomics/                   — disease phenotyping + GWAS instrument selection
  agents/tier2_pathway/                     — Perturb-seq β estimation + regulatory upgrades
  agents/tier3_causal/                      — OTA γ computation + causal graph construction
  agents/tier4_translation/                 — target prioritization + GPS compound screens
  agents/tier5_writer/                      — output assembly + narrative report
pipelines/
  ota_beta_estimation.py                    — Tier1 (Perturb-seq), Tier2 (eQTL-MR), virtual β
  ota_gamma_estimation.py                   — GWAS enrichment γ (OT L2G, Reactome, S-LDSC)
  gps_disease_screen.py                     — GPS disease-state + program reversal screens
  state_space/                              — therapeutic redirection, latent model, TR scoring
mcp_servers/                               — 8 live data servers (GWAS, OT, GTEx, gnomAD, …)
graph/                                     — Kùzu graph DB + RDF/Turtle export
config/scoring_thresholds.py               — all numeric thresholds (never inline)
models/disease_registry.py                 — canonical disease name / EFO / slug mapping
```

---

## Adding a new disease

1. Add entries to `models/disease_registry.py` (`DISEASE_SHORT_KEY`, `DISEASE_EFO`, `DISEASE_DISPLAY_NAME`, `DISEASE_SLUG`, `DISEASE_PROGRAMS`)
2. Add the disease to `graph/schema.py` (`DISEASE_CELL_TYPE_MAP`, `DISEASE_TRAIT_MAP`, `_DISEASE_SHORT_NAMES_FOR_ANCHORS`)
3. Download the disease-relevant h5ad:
   ```bash
   python -m pipelines.discovery.cellxgene_downloader download_all <DISEASE_KEY>
   ```
4. Run the pipeline

---

## Evidence tiers

| Tier | Source | Causal basis |
|------|--------|-------------|
| Tier1_Interventional | CRISPR Perturb-seq (cell-type matched) | Direct perturbation |
| Tier2_Convergent | eQTL-MR (GTEx/scQTL) | Genetic randomization of expression |
| Tier3_Provisional | In silico prediction | No direct causal basis |
| provisional_virtual | Virtual cell / pathway membership | Annotation only |

---

## Tests

```bash
/opt/anaconda3/envs/causal-graph/bin/python -m pytest \
  tests/test_causal_*.py tests/test_gps_*.py tests/test_scoring_*.py \
  tests/test_pipelines*.py tests/test_pi_orchestrator_v2.py -q --tb=short
```

---

## Citation

If you use this pipeline, please cite:

> Ota M, Spence JP, Zeng T, Dann E, Milind N, Marson A, Pritchard JK. Causal modelling of gene effects from regulators to programs to traits. *Nature*. 2026 Feb;650(8101):399–408. doi: [10.1038/s41586-025-09866-3](https://doi.org/10.1038/s41586-025-09866-3). PMID: 41372418.

**Primary data sources — please also cite:**

> Schnitzler GR et al. Convergence of coronary artery disease genes onto endothelial cell programs. *Nature*. *(doi/PMID TBD — please verify)* (Perturb-seq: GEO GSE210681)

> Replogle JM, Saunders RA, Pogson AN, et al. Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell*. 2022;185(14):2559–2575. doi: [10.1016/j.cell.2022.05.013](https://doi.org/10.1016/j.cell.2022.05.013). PMID: 35688146. (K562 Perturb-seq)

> CZI CD4+ T cell Perturb-seq (GEO GSE314342). *(preprint — citation pending)*

> Xing J, Tan M, Leshchiner D, et al. Deep-learning-based de novo discovery and design of therapeutics that reverse disease-associated transcriptional phenotypes. *Cell*. 2026 Mar 17:S0092-8674(26)00223-0. doi: [10.1016/j.cell.2026.02.016](https://doi.org/10.1016/j.cell.2026.02.016). PMID: 41850287. (GPS compound screening)

> Spence JP, Zeng T, Mostafavi H, Pritchard JK. Genome-wide Bayesian estimation of the selective effects of heterozygous loss-of-function variants. *Nature Genetics*. 2024;56:1483–1491. doi: [10.1038/s41588-024-01820-9](https://doi.org/10.1038/s41588-024-01820-9). PMID: 38982189. (shet constraint)

> Ochoa D, Hercules A, Carmona M, et al. Open Targets Platform: supporting systematic drug–target identification and prioritisation. *Nucleic Acids Research*. 2021;49(D1):D1302–D1310. doi: [10.1093/nar/gkaa1027](https://doi.org/10.1093/nar/gkaa1027). PMID: 33196642.

> Elsworth B, Lyon M, Alexander T, et al. The MRC IEU OpenGWAS data infrastructure. *bioRxiv*. 2020. doi: [10.1101/2020.08.10.244293](https://doi.org/10.1101/2020.08.10.244293).

> GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects across human tissues. *Science*. 2020;369(6509):1318–1330. doi: [10.1126/science.aaz1776](https://doi.org/10.1126/science.aaz1776). PMID: 32913098.

---

## License

MIT — see [`LICENSE`](LICENSE).
