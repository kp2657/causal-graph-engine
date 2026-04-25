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

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB (for h5ad loading) |
| Disk | 5 GB (no h5ad) | 150 GB (Perturb-seq + CellxGENE h5ads) |
| Python | 3.12 | — |
| Runtime (CAD, no GPS) | ~25 min | — |
| Runtime (CAD, with GPS) | ~4 h | — |

The pipeline runs without h5ad files — Perturb-seq β degrades to virtual (in silico) estimates. Results are still produced but less mechanistically grounded.

---

## Setup

```bash
conda create -n causal-graph python=3.12
conda activate causal-graph
pip install -e .                      # core dependencies
pip install -e ".[bio,chem,dev]"      # full install (single-cell + chemistry + tests)
cp .env.example .env                  # fill in API keys (see .env.example)
```

Required API keys (free tiers sufficient for most usage):
- `ANTHROPIC_API_KEY` — only needed for `AGENT_MODE=sdk`
- `OPENGWAS_JWT` — IEU Open GWAS ([register here](https://gwas.mrcieu.ac.uk))
- `NCBI_API_KEY` — NCBI E-utilities ([register here](https://www.ncbi.nlm.nih.gov/account/))

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

> Ota, M. et al. (2023). Genomic medicine across ancestries. *Nature Genetics*.

And the primary data sources: Open Targets, IEU Open GWAS, GTEx, Replogle 2022 (K562 Perturb-seq), CZI CD4+ T Perturb-seq (GSE314342).

---

## License

MIT — see [`LICENSE`](LICENSE).
