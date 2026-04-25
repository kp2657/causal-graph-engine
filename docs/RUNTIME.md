# Runtime configuration

Single reference for environment variables, orchestrator behaviour, file layout, and **what degrades without failing loudly**.

---

## Environment variables (by subsystem)

| Variable | Used by | When unset / empty |
|----------|---------|---------------------|
| `GRAPH_DB_PATH` | `graph/db.py`, `graph_db_server` | Defaults to `./data/graph.kuzu` (creates or uses that file). |
| `OPENGWAS_JWT` | `gwas_genetics_server` OpenGWAS tools | OpenGWAS endpoints return empty or auth-error payloads; pipeline continues with GWAS Catalog / OT only. Expires 14 days after issue; renew regularly. |
| `OPEN_TARGETS_ASSOC_DISABLED` | `open_targets_server`, `pi_orchestrator_v2._collect_gene_list`, `target_prioritization_agent` | When `1`/`true`: skips heavy `associatedTargets` GraphQL; seeds genes from OT **L2G** instead. |
| `SKIP_MR` | `statistical_geneticist.run` | When `1`/`true`: skips the Mendelian randomisation step entirely. Use when IEU JWT is absent and MR instruments would be empty anyway. |
| `GPS_REPO_PATH` | `mcp_servers/gps_server.py` only | Optional local GPS4Drug subprocess path. **Not** used by Tier 4 chemistry; that uses `pipelines/gps_disease_screen.py` via Docker. |
| `CROSSREF_MAILTO` | Polite Crossref User-Agent | Optional; improves rate limits. |
| `ANTHROPIC_API_KEY` | SDK / `AGENT_MODE=sdk` | Required only for Claude-dispatched agents. |
| `AGENT_MODE` | `agent_runner` | `local` (default): no API. `sdk`: CSO / discovery agents may call Claude. **`pi_orchestrator_v2` ignores AgentRunner** — it always calls agents as plain functions. |
| `MINIMAL_TIER4` | `target_prioritization_agent` | When `1`/`true`: skips non-essential Tier 4 metrics (tau/bimodality, upstream benchmark, program driver classification). |
| `TIER2_NONVIRTUAL_ONLY` | `perturbation_genomics_agent` | When `1`/`true`: drops genes with no direct Perturb-seq β (virtual-only candidates) after Tier 2. Useful for debugging mechanistic quality without virtual-tier noise. |
| `TIER2_MAX_GENES` | `perturbation_genomics_agent` | Integer cap on Tier 2 gene list after filtering. Default: no cap. Set to e.g. `200` to accelerate debugging runs. |
| `PERTURBSEQ_DATA_DIR` | `mcp_servers/burden_perturb_server.py` | Override path to Replogle h5ad files used by the burden server. Default: `./data/replogle2022`. |
| `STATIC_DATA_DIR` | `pipelines/static_lookups.py`, `scripts/fetch_static_data.py` | Override path to static lookup tables (pLI, OMIM, drug targets). Default: `./data/static/`. |
| `CLUE_API_KEY` | `mcp_servers/viral_somatic_server.py` | Optional CLUE Connectivity Map API key. Free tier available; without it, CLUE perturbational signatures are skipped. |
| `VIRAL_REF_DIR`, `CHIP_CALLS_DIR` | Respective data loaders | Missing dirs → fewer Tier 2 signals; agents emit warnings or fall back to virtual tiers. |
| `NCBI_API_KEY` | Literature / PubMed tools | Optional; raises PubMed rate limit from 3 to 10 req/s. |

Copy `.env.example` to `.env` and fill values. Never commit `.env`.

---

## Run commands

```bash
# Full pipeline (Tiers 1–5)
conda run -n causal-graph python -m orchestrator.pi_orchestrator_v2 analyze_disease_v2 "coronary artery disease"
conda run -n causal-graph python -m orchestrator.pi_orchestrator_v2 analyze_disease_v2 "age-related macular degeneration"

# Re-run Tier 4+5 only from saved Tier 3 checkpoint (skips Tiers 1–3; much faster)
conda run -n causal-graph python -m orchestrator.pi_orchestrator_v2 run_tier4 "coronary artery disease"
```

Output is written to `data/analyze_{disease_slug}.json` and `data/analyze_{disease_slug}.md`.

---

## Orchestrator (`pi_orchestrator_v2`)

- **Gene seeding**: `_collect_gene_list` requests up to **500** Open Targets disease targets (`min_overall_score=0.05`) when `efo_id` is set and `OPEN_TARGETS_ASSOC_DISABLED` is off. If OT fails, logs `OT_SEED_FAIL` and continues with GWAS-validated genes only (may be a short list).
- **Gamma estimates**: `_get_gamma_estimates` runs before Tier 3; the same dict is stored as `pipeline_outputs["_gamma_estimates"]` and injected into `prioritization_result["_gamma_estimates"]` before chemistry so GPS disease/program screens see program→trait γ.
- **Tier 4 context**: a single `_tier4_context` snapshot (e.g., gnomAD pLI map) is attached to `disease_query` once per run and reused inside Tier 4.
- **Output format**: `analyze_disease_v2` returns the `_build_final_output` flat format (all `graph_output` keys spread to top level, including `target_list`). `run_tier4` returns a nested dict with `prioritization_result.targets`.

---

## Checkpoints

Tier 3 and Tier 4 results are saved after each stage:

| File | Written by | Used by |
|------|------------|---------|
| `data/checkpoints/{disease}__tier3.json` | `analyze_disease_v2` after Tier 3 | `run_tier4` to skip Tiers 1–3 |
| `data/checkpoints/{disease}__tier4.json` | `analyze_disease_v2` after Tier 4 | `run_tier4` fallback if no Tier 3 checkpoint |

The Tier 3 checkpoint contains: `genetics_result`, `beta_result`, `regulatory_result`, `gamma_estimates`, `causal_result` (includes `top_genes` with `programs`, `ota_gamma`, evidence tiers), `kg_result`.

**Stale checkpoints:** if a checkpoint's `causal_result.top_genes` is empty (e.g., from a failed prior run), `run_tier4` will produce 0 targets. Delete stale checkpoints and re-run the full pipeline.

---

## State cache

CELLxGENE disease-state inference results (list of `CellState` objects per resolution level) are cached at:

```
data/cellxgene/{DISEASE}/state_cache_coarse.json
data/cellxgene/{DISEASE}/state_cache_intermediate.json
data/cellxgene/{DISEASE}/state_cache_fine.json
```

The cache stores `CellState` metadata (cluster ids, labels, pseudotime, pathological score) but **not** the Markov transition matrix — that is recomputed fresh each run from the h5ad file. Delete state cache files to force re-clustering.

CELLxGENE h5ad files are cached at:
```
data/cellxgene/{DISEASE}/{DISEASE}_{cell_type}.h5ad
```

Census version: `2025-11-08`. Download is automatically triggered on first run per disease/cell-type combination.

---

## GPS compound screening

GPS uses a Docker container (`binchengroup/gpsimage:latest`). The container must be running before the pipeline starts:

```bash
docker run -d --name gps binchengroup/gpsimage:latest sleep infinity
```

Background distribution (BGRD) is pre-computed on first run and cached:
```
data/gps_bgrd/{disease}_bgrd.pkl
```

With cached BGRD: timeout `GPS_TIMEOUT_WITH_BGRD = 7200 s` (2 hours).  
Without cached BGRD: timeout `GPS_TIMEOUT_NO_BGRD = 21600 s` (6 hours).

Signature size: 700–1000 genes after elbow-trim (`GPS_BGRD_MIN_GENES = 700`, `GPS_BGRD_MAX_GENES = 1000`).  
RGES Z-score hit threshold: `GPS_Z_RGES_DEFAULT = 2.0`.  
Maximum compounds returned: `GPS_MAX_HITS = 100`.

All GPS constants are in `config/scoring_thresholds.py`.

---

## Scoring thresholds registry

All numeric thresholds used throughout the pipeline live in `config/scoring_thresholds.py`:

```python
from config.scoring_thresholds import MR_F_STATISTIC_MIN, COLOC_H4_MIN, OT_L2G_MIN, ...
```

Do not define local threshold constants; import from this module. Key thresholds:

| Constant | Value | Purpose |
|----------|-------|---------|
| `MR_F_STATISTIC_MIN` | 10.0 | Weak-instrument floor |
| `MR_P_VALUE_MAX` | 5×10⁻⁸ | GWAS instrument inclusion |
| `MR_PQTL_P_VALUE_MAX` | 0.05 | pQTL instrument inclusion |
| `COLOC_H4_MIN` | 0.80 | Full colocalization (Tier 2) |
| `COLOC_H4_DIRECTION_MIN` | 0.50 | Direction-only (Tier 2.5) |
| `OT_L2G_MIN` | 0.05 | Gene seeding threshold |
| `OT_L2G_GAMMA_MIN` | 0.10 | Gamma estimation threshold |
| `GPS_Z_RGES_DEFAULT` | 2.0 | GPS hit Z-score |
| `CO_EVIDENCE_WEIGHT_UNGROUNDED` | 0.20 | Mechanistic-only rank penalty |
| `OTA_GAMMA_SCORE_CAP` | 0.70 | Reserved for future composite scoring |

---

## FAST / caps (scripts)

`scripts/build_cad_genetics_pack.py` reads:

| Env | Role |
|-----|------|
| `FAST=1` | Skips eQTL and burden annotation passes (large speedup). |
| `MAX_TOPHITS`, `MAX_GENES`, `MAX_OT_STUDIES_USED` | Caps network/GCS work. |
| `ENABLE_GTEX`, `ENABLE_SC_EQTL`, `ENABLE_PQTL`, `ENABLE_FINNGEN`, `ENABLE_UKB_WES` | Set to `0` to disable layers. |

`scripts/run_tier2_subset.py` documents `TIER2_NONVIRTUAL_ONLY=1` for subset runs.

---

## Validated run results (v0.2.2, 2026-04-24)

### AMD (age-related macular degeneration)
- Targets ranked: 145
- Known gene recovery: 11/14 (79%)
- CFH: rank 2, γ=0.56; C3: rank 4; ARMS2: rank 30; VEGFA: rank 130
- GPS disease-state reversers: 100

### CAD (coronary artery disease)
- Targets ranked: 2144 (Tier 3 + 4 complete, GPS program reversers pending re-run)
- PCSK9/HMGCR recovered; LIPC rank 1 (eQTL-MR driven)
- GPS disease-state reversers: 100

### SLE / DED
- Pending first run (CZI Perturb-seq downloads in progress; see STATE.md)

---

## Related docs

- [DATA_SOURCES.md](DATA_SOURCES.md) — URLs and datasets.
- [METHODS.md](METHODS.md) — statistical definitions (OTA formula, evidence tiers, scoring).
