# Causal Graph Engine

Multiagent genomics pipeline for causal drug target discovery, implementing the Ota et al. causal-effect decomposition:

```
γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait})
```

β is estimated from CRISPR Perturb-seq and eQTL-MR; γ from GWAS enrichment and S-LDSC. Both are genetic-instrument-grounded, removing reverse causation from target scores. See [docs/OTA_ALGORITHM.md](docs/OTA_ALGORITHM.md) for a full method description.

---

## What you get

Running the full pipeline on AMD produces a ranked target list like this (v0.2.0, 2026-04-20):

| Rank | Gene | γ (ota_gamma) | Tier | Evidence |
|------|------|---------------|------|----------|
| 1 | LIPC | 0.565 | Tier2_Convergent | GTEx eQTL-MR + OT L2G=0.87 + lipid metabolism program |
| 2 | CFH | 0.561 | provisional_virtual† | OT L2G=0.86 + COLOC H4=0.87 + complement locus |
| 3 | APOE | 0.542 | provisional_virtual† | OT L2G=0.83 + AMD GWAS locus |
| 4 | C3 | 0.535 | provisional_virtual† | OT L2G=0.82 + complement program γ>0 |
| 5 | CFHR5 | 0.511 | provisional_virtual† | OT L2G=0.79 + CFH-locus co-localisation |

† `provisional_virtual`: gene is absent from the Replogle 2022 RPE1 CRISPR screen (not a perturb target in the library). `causal_gamma` is derived from OT L2G + COLOC Bayesian fusion rather than a direct Σ(β×γ) product. The genetic evidence is still strong — CFH, C3, and ARMS2 are among the most replicated AMD GWAS hits — but no Perturb-seq β was measured for these genes. ARMS2 (rank 30, γ=0.322) and VEGFA (rank 130, γ=0.072) are also recovered.

**Why is LIPC rank 1 above CFH?** LIPC (hepatic lipase) has a slightly higher OT L2G score (0.869 vs 0.863) driven by a lipid-metabolism eQTL-MR signal absent from CFH. The difference is within margin of uncertainty (Δγ=0.004); both genes are strong AMD targets. CFH is the canonical therapeutic anchor for the complement pathway.

CAD top-5 (v0.2.0, 2026-04-21): LIPC(1,γ=1.892), PROCR(2,γ=-1.835), BUD13(3,γ=-1.713), SORT1(4,γ=1.519), SF3A3(5,γ=-1.511). HMGCR rank 46, LPA rank 118, PCSK9 rank 160. BUD13 and SF3A3 are RNA splicing factors with Perturb-seq Tier1 evidence (K562 HCASMC screen) and GWAS co-localisation; their high γ reflects strong program loading but they are not conventional druggable CAD targets. PROCR (Protein C receptor) is a coagulation/APC pathway target with OT L2G=0.85.

Plus: GPS compound screen against disease-state signature (100 reversers per disease), RDF/Turtle graph export, and a narrative discovery report.

---

## Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB (for h5ad loading) |
| Disk | 5 GB (no h5ad) | 150 GB (Replogle + CELLxGENE h5ads) |
| CPU | 4 cores | 8 cores (ThreadPoolExecutor) |
| Runtime (AMD, no GPS) | ~45 min | — |
| Runtime (AMD, with GPS) | ~3 h | — |
| Runtime (CAD, with GPS) | ~4 h | — |

The pipeline runs without h5ad files — Perturb-seq data degrades to Tier 3 LINCS signatures, and GPS degrades to OTA-derived signatures. Results are still valid but less mechanistically grounded.

---

## Setup

```bash
conda create -n causal-graph python=3.12
conda activate causal-graph
pip install -e .                      # core: no Anthropic SDK required
pip install -e ".[agentic]"           # + Claude SDK for AGENT_MODE=sdk
pip install -e ".[bio,chem,dev]"      # full install (single-cell + chemistry + tests)
cp .env.example .env                  # fill in API keys (see .env.example for links)
```

---

## Run

```bash
# Full pipeline — AMD or CAD (first run or after changing Tier 1–3 data)
conda run -n causal-graph python -m orchestrator.pi_orchestrator_v2 analyze_disease_v2 \
    "age-related macular degeneration"

conda run -n causal-graph python -m orchestrator.pi_orchestrator_v2 analyze_disease_v2 \
    "coronary artery disease"

# Re-run Tier 4+5 from saved Tier 3 checkpoint (skips Tiers 1–3 re-computation)
# Use after updating scoring logic, GPS Docker, or compound library
conda run -n causal-graph python -m orchestrator.pi_orchestrator_v2 run_tier4 \
    "age-related macular degeneration"

# Validate output against known disease genes
conda run -n causal-graph python scripts/validate_results.py --disease amd
conda run -n causal-graph python scripts/validate_results.py --disease cad
```

### Optional: GPS phenotypic screening

GPS requires the Bin-Chen-Lab Docker image (~8 GB):

```bash
docker pull binchengroup/gpsimage:latest
docker run -d --name gps binchengroup/gpsimage:latest sleep infinity
```

Without Docker, GPS is skipped and logged in `pipeline_warnings` in the output JSON. The target ranking still completes using genetic evidence alone.

### Optional: Large Perturb-seq h5ad files

For quantitative Tier 1 β (strongest evidence), download the Replogle 2022 h5ad:

```bash
# RPE1 (~100 MB) — used for AMD
mkdir -p data/perturb_seq/replogle2022
wget -O data/perturb_seq/replogle2022/RPE1_essential_normalized_bulk_01.h5ad \
    https://ndownloader.figshare.com/files/35780876

# K562 genome-wide (~370 MB) — used for CAD/IBD fallback
wget -O data/perturb_seq/replogle_2022_k562/K562_gwps_normalized_bulk_01.h5ad \
    https://ndownloader.figshare.com/files/35773217
```

Without h5ad, β falls back to eQTL-MR (Tier 2), which is still genetically grounded.

---

## Tests

```bash
# Targeted test run (~30s) — always run this, never pytest tests/ directly
/opt/anaconda3/envs/causal-graph/bin/python -m pytest \
  tests/test_ota_beta_estimation.py \
  tests/test_ota_gamma_estimation.py \
  tests/test_phase_a_models.py \
  tests/test_pipelines.py \
  tests/test_pi_orchestrator_v2.py \
  -q --tb=short
```

---

## Modes

| Mode | Description | Requires |
|---|---|---|
| `AGENT_MODE=local` (default) | All agents run as direct function calls. No API cost. | Free public APIs |
| `AGENT_MODE=sdk` | CSO + discovery-refinement agents use Claude API for richer reasoning. | `ANTHROPIC_API_KEY` |

---

## Architecture

```
orchestrator/pi_orchestrator_v2.py   — 5-tier pipeline coordinator
config/scoring_thresholds.py         — all scoring constants with citations
models/disease_registry.py           — canonical disease name/EFO mapping

agents/tier1_phenomics/              — phenotype, GWAS genetics, somatic exposure
agents/tier2_pathway/                — Perturb-seq β estimation, regulatory networks
agents/tier3_causal/                 — causal discovery, knowledge graph completion
agents/tier4_translation/            — target prioritisation, GPS chemistry, clinical
agents/tier5_writer/                 — scientific writer and reviewer

pipelines/                           — OTA β/γ estimation, GPS screening, state-space
mcp_servers/                         — 8 live data servers (GWAS, GTEx, OT, CELLxGENE, …)
graph/                               — Kùzu graph DB + RDF/Turtle export
```

---

## Outputs

| File | Description |
|---|---|
| `data/analyze_{disease}.md` | Human-readable discovery report |
| `data/analyze_{disease}.json` | Full structured output (targets, GPS, warnings) |
| `data/exports/{disease}.ttl` | RDF/Turtle graph export |
| `data/checkpoints/{disease}__tier*.json` | Per-tier resumable checkpoints |

The JSON output includes `pipeline_warnings` (list of data sources skipped or degraded) and `data_completeness` (which optional datasets were loaded), so you can always tell what quality of evidence went into each run.

---

## APIs used

All public and free. See `.env.example` for registration links.

| API | Used for | Auth |
|---|---|---|
| IEU OpenGWAS | MR instruments, GWAS summary stats | JWT (free, 14-day expiry) |
| GWAS Catalog | GWAS hit lookup, cross-reference | None |
| gnomAD | pLI, LOEUF constraint scores | None |
| GTEx v8 | eQTL-MR instruments | None |
| Open Targets | L2G prioritisation, drug targets | None |
| eQTL Catalogue | sc-eQTL, pQTL instruments | None |
| CELLxGENE Census | Disease-state transcriptomics | None |
| ChEMBL / PubChem | Compound annotation, SMILES | None |
| Semantic Scholar | Literature cross-reference | None (optional key) |
| NCBI E-utilities | Gene aliases, literature | None (free key for higher rate) |

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/OTA_ALGORITHM.md](docs/OTA_ALGORITHM.md) | Full algorithm with worked example (CFH → AMD), tier hierarchy, γ estimation modes |
| [docs/GPS_ALGORITHM.md](docs/GPS_ALGORITHM.md) | GPS RGES scoring, BGRD lifecycle, Z_RGES threshold, Docker setup |
| [docs/METHODS.md](docs/METHODS.md) | Statistical methods with primary citations |
| [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) | Every data source — URL, access, local paths |
| [docs/RUNTIME.md](docs/RUNTIME.md) | Env vars, silent degradation, orchestrator caps |
| [docs/OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md) | Every field in the output JSON — target scores, flags, GPS records |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to add a new disease, run tests, coding conventions |

---

## Citation

If you use this pipeline, please cite:
- Ota et al. (2026) Nature — OTA causal framework
- Replogle et al. (2022) Cell — Perturb-seq β estimation
- Mountjoy et al. (2021) Nature Genetics — Open Targets L2G
- Lamb et al. (2006) Science — GPS / connectivity map
