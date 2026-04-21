# Causal Graph Engine — Build State
Last updated: 2026-04-21 (Session 63 — essential gene filter, scoring_thresholds wiring, v0.2.0 validated)

---

## Status: v0.2.0 COMPLETE ✓

Both AMD and CAD pipelines have been validated with the final v0.2.0 code. All pre-publication fixes applied.

---

## Validated pipeline results

### AMD (age-related macular degeneration) — 2026-04-20
- **145 targets**, 100 GPS disease-state reversers, 0 essential genes sunk
- Top 5: LIPC(1,γ=0.565), CFH(2,γ=0.561), APOE(3,γ=0.542), C3(4,γ=0.535), CFHR5(5,γ=0.511)
- Anchor recovery: CFH(✓), C3(✓), ARMS2(✓ rank 30), VEGFA(✓ rank 130) ≈ 79%
- Output: `data/analyze_age_related_macular_degeneration.json`

### CAD (coronary artery disease) — 2026-04-21
- **712 targets**, 100 GPS disease-state reversers, 48 essential genes sunk to bottom
- Top 5: LIPC(1,γ=1.892), PROCR(2,γ=-1.835), BUD13(3,γ=-1.713), SORT1(4,γ=1.519), SF3A3(5,γ=-1.511)
- HMGCR rank 46, LPA rank 118, PCSK9 rank 160
- Essential sunk: POLR2C, PRIM2, RPL/RPS family (48 total, `inflated_gamma_essential` flag)
- Output: `data/analyze_coronary_artery_disease.json`

---

## Test suite
- **576 passing**, 5 failing (all pre-existing SDK integration tests requiring live API keys)
- Run: `/opt/anaconda3/envs/causal-graph/bin/python -m pytest tests/test_ota_beta_estimation.py tests/test_ota_gamma_estimation.py tests/test_phase_a_models.py tests/test_pipelines.py tests/test_pi_orchestrator_v2.py -q --tb=short`

---

## Key v0.2.0 features

### Essential gene sink
`|ota_gamma| > 2.0 AND _max_prog_contrib > 0.8` → `inflated_gamma_essential` flag + sort to bottom.
`_max_prog_contrib = max(|β_P × γ_P|)` pre-computed at record-build time in `target_prioritization_agent.py`.

### Scoring thresholds registry
All numeric constants centralised in `config/scoring_thresholds.py`. Wired into:
- `pipelines/twmr.py` — `MR_F_STATISTIC_MIN`
- `pipelines/ota_beta_estimation.py` — `COLOC_H4_MIN`, `MR_PQTL_P_VALUE_MAX`
- `agents/tier1_phenomics/statistical_geneticist.py` — `MR_F_STATISTIC_MIN`

### TR inflow reduction
`perturb_transition_matrix` now reduces both outflow (pathological→healthy) and inflow (healthy→pathological) edges; `compute_net_trajectory_improvement` sums both components.

### OTA formula
`γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait})`
β: Perturb-seq (Tier1) → eQTL-MR (Tier2) → LINCS (Tier3)
γ: GWAS enrichment via S-LDSC / Reactome pathway programs

### Evidence tiers (output `tier` field)
| Tier | Criterion |
|------|-----------|
| Tier1_Interventional | Perturb-seq β + co-evidence gate pass |
| Tier2_Convergent | eQTL-MR + COLOC H4≥0.8 |
| Tier2p_pQTL_MR | pQTL-MR (UKB-PPP / deCODE) |
| Tier2rb_RareBurden | UKB WES burden p<0.01 |
| Tier2pt_ProteinChannel | Protein abundance pQTL |
| Tier2.5 | eQTL direction-only (no β magnitude) |
| Tier3_Provisional | OT L2G < 0.10 (no genetic instrument) |

### GPS screens
- `gps_disease_state_reversers`: reverses disease-state DEG signature from h5ad
- `gps_program_reversers`: reverses top NMF program signatures (top_n_programs=5)
- Dynamic Z_RGES threshold (z_threshold=2.0, max_hits=100); falls back to top_n if <3 pass

---

## Pending (post-v0.2.0)

1. **git tag v0.2.0** — no tag exists yet
2. **Clean stale data/ files** — `analyze_.json`, `analyze_schizophrenia.json`, `analyze_primary_open-angle_glaucoma.json`, `analyze_inflammatory_bowel_disease.json`
3. **README example table** — shows fabricated scores (CFH 0.847) that don't match actual v0.2.0 AMD output (CFH γ=0.561)
4. **OTA_ALGORITHM.md worked example** — verify CFH numbers match v0.2.0 AMD run (CFH rank 2, γ=0.561)
5. **CLAUDE.md run commands** — still lists `run_tier4` as the default full-pipeline command; should be `analyze_disease_v2`

---

## GPS/h5ad status

| File | Status |
|------|--------|
| AMD RPE1 h5ad | ✅ loaded, DEG sig computed |
| CAD SMC h5ad | ✅ loaded, DEG sig computed |
| AMD BGRD (disease-state) | ✅ recomputed |
| CAD BGRD (disease-state) | ✅ recomputed |
| AMD program BGRDs | ✅ P07 recomputed |
| CAD program BGRDs | ✅ 5 programs recomputed |

---

## Architecture quick reference
```
orchestrator/pi_orchestrator_v2.py   — 5-tier pipeline (analyze_disease_v2 / run_tier4)
config/scoring_thresholds.py         — all numeric constants with citations
models/disease_registry.py           — canonical disease name / EFO mapping
agents/tier{1-5}_*/                  — per-tier agents
pipelines/ota_beta_estimation.py     — Tier1/2/3 β estimation
pipelines/ota_gamma_estimation.py    — GWAS-enrichment γ via Reactome + S-LDSC
pipelines/state_space/               — TR, conditional β/γ, state influence
pipelines/gps_screening.py          — GPS Docker wrapper + BGRD lifecycle
mcp_servers/                         — 8 live data servers
graph/                               — Kùzu graph DB + RDF/Turtle export
```
