# Contributing

## Environment

```bash
conda create -n causal-graph python=3.12
conda activate causal-graph
pip install -e ".[dev]"        # core + tests + linting
```

## Running tests

Always run targeted test files — never `pytest tests/` (exceeds the 2-minute timeout due to integration tests).

```bash
# Fast core check (~15s)
python -m pytest \
  tests/test_ota_beta_estimation.py \
  tests/test_ota_gamma_estimation.py \
  tests/test_phase_a_models.py \
  tests/test_pipelines.py \
  -q --tb=short -m "not integration"

# Broader check (~45s)
python -m pytest \
  tests/test_ota_beta_estimation.py \
  tests/test_ota_gamma_estimation.py \
  tests/test_phase_a_models.py \
  tests/test_phase_b_conditional.py \
  tests/test_phase_c_therapeutic_redirection.py \
  tests/test_pi_orchestrator_v2.py \
  tests/test_pipelines.py \
  -q --tb=short -m "not integration"
```

Integration tests (live API calls) require credentials and are excluded from CI:

```bash
python -m pytest tests/ -q --tb=short -m "integration"
```

## Coding conventions

### NaN, not 0.0, for missing values

```python
# Wrong — 0.0 × gamma = phantom zero contribution
beta = 0.0

# Right — propagates through OTA summation correctly
beta = float("nan")
```

All tier functions must return `None` (not `0.0`) when evidence is absent. The `estimate_beta()` dispatcher always returns a dict; callers should check `beta.get("beta")`.

### No co-expression or synthetic data

Co-expression is not a valid β source. Synthetic Reactome-derived betas (Tier2s) were removed in v0.2.0 — do not re-introduce them. All β sources must be either interventional (Perturb-seq) or genetic instruments (eQTL-MR, pQTL-MR, GWAS fine-mapping).

### Disease keys

Use `models.disease_registry` for all disease name → short key conversions:

```python
from models.disease_registry import get_disease_key, get_efo_id

key = get_disease_key("age-related macular degeneration")  # "AMD"
efo = get_efo_id("AMD")                                   # "EFO_0001481"
```

Do not define new `_DISEASE_KEY_MAP` dicts in individual files.

### Scoring thresholds

All numeric thresholds must be imported from `config.scoring_thresholds`:

```python
from config.scoring_thresholds import COLOC_H4_MIN, OT_L2G_GAMMA_MIN

if coloc_h4 is not None and coloc_h4 >= COLOC_H4_MIN:
    ...
```

### Type annotations

All public functions (not starting with `_`) must have complete type annotations including return types. Use `-> dict | None` when a function can return None.

### No comments explaining what the code does

Comments should only explain *why* — hidden constraints, citations, or non-obvious invariants. Well-named functions and variables communicate *what*.

```python
# Wrong
# This multiplies beta by loading
beta_scaled = beta * loading

# Right — explains why this particular threshold
# Threshold from Giambartolomei 2014 — H4 >= 0.8 = strong shared-variant evidence
if coloc_h4 >= COLOC_H4_MIN:
```

### Error handling

When a fallback is triggered, always print a `[WARN]` message:

```python
except Exception as _e:
    print(f"[WARN] GTEx eQTL lookup failed for {gene}: {_e}")
    # fallback path
```

Never use bare `except Exception: pass` without a print statement.

## Adding a new disease

1. Add to `models/disease_registry.py`: `DISEASE_SHORT_KEY`, `DISEASE_EFO`, `DISEASE_DISPLAY_NAME`, `DISEASE_SLUG`
2. Add CELLxGENE h5ad download config to `pipelines/discovery/cellxgene_downloader.py`
3. Add Perturb-seq dataset to `mcp_servers/perturbseq_server.py` `_DATASET_REGISTRY` (if disease-specific data exists)
4. Add disease-cell-type context to `graph/schema.py` `DISEASE_CELL_TYPE_MAP`
5. Add known validated genes to `scripts/validate_results.py` for QC
6. Run: `python -m orchestrator.pi_orchestrator_v2 run_tier4 "your disease name"`
7. Check `scripts/validate_results.py --disease <key>` — aim for ≥ 60% anchor recovery

## Pull request checklist

- [ ] All targeted tests pass (`-m "not integration"`)
- [ ] New public functions have type annotations
- [ ] Any new thresholds are added to `config/scoring_thresholds.py` with citations
- [ ] No new disease key maps defined locally — use `models.disease_registry`
- [ ] Any silent failures log `[WARN]` before returning fallback
- [ ] CHANGELOG.md updated with a summary of changes

## Version scheme

This project uses semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: breaking change to output schema or OTA formula
- MINOR: new evidence tier, new disease, or new data source
- PATCH: bug fixes and silent failure improvements

Update `pyproject.toml` version and add a CHANGELOG entry for each release.
