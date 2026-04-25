# Causal Graph Engine — Claude Instructions

## What this is
Multiagent causal genomics pipeline implementing Ota et al.:
`γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait})`

Output: ranked drug targets with Kùzu graph + RDF export + Markdown report.

---

## Environment
```bash
conda activate causal-graph   # Python 3.12, pydantic v2
cd causal-graph-engine/
# NEVER use base python/python3 — they are Python 3.8
```

---

## Run commands
```bash
# Full pipeline (first run or after changing Tier 1–3 data)
# Active diseases: CAD (cardiac endothelial cell) + RA (CD4+ T cell)
conda run -n causal-graph python -m orchestrator.pi_orchestrator_v2 analyze_disease_v2 "coronary artery disease"
conda run -n causal-graph python -m orchestrator.pi_orchestrator_v2 analyze_disease_v2 "rheumatoid arthritis"

# Re-run Tier 4+5 from saved Tier 3 checkpoint (skips Tiers 1–3 re-computation)
conda run -n causal-graph python -m orchestrator.pi_orchestrator_v2 run_tier4 "coronary artery disease"
conda run -n causal-graph python -m orchestrator.pi_orchestrator_v2 run_tier4 "rheumatoid arthritis"

# Unit tests (fast, targeted — NEVER run the full suite via pytest tests/)
/opt/anaconda3/envs/causal-graph/bin/python -m pytest tests/test_state_space_*.py tests/test_causal_*.py tests/test_gps_*.py tests/test_scoring_*.py tests/test_pipelines*.py tests/test_pi_orchestrator_v2.py -q --tb=short
```

---

## Session startup
1. Read `STATE.md` — objective + next steps
2. Read `CHANGELOG.md` — recent changes
3. Run unit tests to confirm baseline
4. Never re-implement anything marked ✓ DONE without first checking it's broken

---

## Success criteria
- `anchor_edge_recovery ≥ 0.80` (QC check)
- `n_novel_edges > 0` (primary goal)
- `data/analyze_{disease}.md` + `data/exports/{disease}.ttl` exist

---

## Architecture
```
orchestrator/pi_orchestrator_v2.py   — 5-tier pipeline (the entry point; run this)
orchestrator/sdk/                    — optional Claude SDK dispatch (AGENT_MODE=sdk only)
agents/tier{1-5}_*/                  — per-tier agents (plain functions, called directly)
  tier3_causal/causal_discovery_agent.py — run() only: OTA γ + graph construction
  tier3_causal/causal_filters.py         — utility fns: stress discount, entropy, Pareto
  tier3_causal/causal_therapeutic.py    — _maybe_therapeutic_redirection helper
pipelines/                           — OTA β/γ estimation, GPS screening, state-space
mcp_servers/                         — 8 live data servers (GWAS, gnomAD, GTEx, CELLxGENE, OT, …)
graph/                               — Kùzu graph DB + RDF/Turtle export
config/scoring_thresholds.py         — all numeric constants with citations (import from here, never inline)
models/disease_registry.py           — canonical disease name/EFO/GWAS mapping; disease_key set by phenotype_architect
data/cellxgene/{disease}/            — cached h5ad files
```

---

## Critical rules

| Rule | Why |
|------|-----|
| Pydantic v2 only (`field_validator`, `model_dump()`, `X \| None`) | v1 syntax breaks silently |
| `float('nan')` for missing β/γ, never `0.0` | 0.0 × γ = phantom zero edges |
| Co-expression is NOT a valid β source | No causation without perturbation/genetic instrument |
| `GRAPH_DB_PATH` → temp dir in DB tests | Prevents corrupting `data/graph.kuzu` |
| `model_construct()` for unknown-agent stubs | Bypasses AgentName Literal without crash |
| **Do not use `get_anndata(obs_coords=numpy_array)`** | Segfaults in tiledbsoma 2.3.0 — use axis_query |
| All numeric thresholds in `config/scoring_thresholds.py` | Never inline magic numbers — import from there |
| `disease_query["disease_key"]` is set once by `phenotype_architect.run()` | Never re-derive via `get_disease_key()` in downstream agents |

---

## State-space refactor: Phases A–F COMPLETE ✓

**Target sort order (actual implementation):** partition rank first (Tier1 > Tier2 > Tier3 > Tier4), then `−|ota_gamma|` within partition.

Key modules: `pipelines/state_space/` — `state_influence.py`, `therapeutic_redirection.py`, `conditional_beta.py`, `conditional_gamma.py`, `latent_model.py`, `transition_graph.py`

## CELLxGENE downloads
`pipelines/discovery/cellxgene_downloader.py` — axis_query two-phase (safe).
Census: `2025-11-08`. Column: `feature_type`. Cached: `data/cellxgene/{DISEASE}/{DISEASE}_{cell_type}.h5ad`

---

## Commit discipline
```bash
conda run -n causal-graph python -m pytest tests/ -q -m "not integration" && git add -p && git commit
```
All unit tests must pass before commit.

---

## Background task discipline

### Hard rules — no exceptions

**NEVER background pytest.** Always run tests foreground. pytest backgrounded:
- writes output to tmp files that are empty when the process is killed
- reports exit code 1 on SIGKILL regardless of whether tests were passing
- accumulates as zombie processes silently consuming resources

**NEVER launch a duplicate.** Before issuing any `run_in_background=True` command, run the pre-flight check (below). If a process doing the same thing is already running, read its output or kill it first — do not start another.

**ALWAYS kill superseded tasks immediately.** If the user changes direction mid-run, call `pkill` or `TaskStop` before starting the replacement. Do not leave orphaned jobs running.

### Pre-flight check — run before any background launch
```bash
pgrep -la python | grep -E "pytest|orchestrator|pi_orchestrator"
```
If anything matching is already running: stop it, then proceed.

### Allowed background tasks (only these)
| Task | Condition |
|------|-----------|
| Full IBD/CAD pipeline run | User explicitly asked; expected >5 min; record the PID |
| Multi-disease batch run | Same |

Everything else — pytest, probes, diagnostics, inline python, any command <5 min — runs **foreground only**.

### Test strategy — foreground, targeted
```bash
# Fastest signal (~10s) — run this first
/opt/anaconda3/envs/causal-graph/bin/python -m pytest tests/test_phase_*.py tests/test_state_space_*.py -q --tb=short

# Broader check (~30s) — targeted file list, never the whole directory
/opt/anaconda3/envs/causal-graph/bin/python -m pytest \
  tests/test_state_space_*.py tests/test_causal_*.py \
  tests/test_gps_*.py tests/test_scoring_*.py \
  tests/test_pipelines*.py tests/test_agents.py -q --tb=short
```

**NEVER run `pytest tests/`** — the full suite exceeds the 2-minute Bash tool timeout and is
automatically backgrounded, producing an empty output file and a spurious exit-code-1 on kill.
Always target specific test files.

### On receiving a task-notification with status=failed
1. Check output file line count: `wc -l <output_file>`. If 0 or 1 lines → killed by SIGKILL (timeout), not a real failure.
2. Treat empty/1-line output as: **ignore, do not re-run**.
3. If a real failure is suspected, run the specific relevant test file(s) foreground.
4. Never retry a killed background task with another background task.
5. `pgrep -la python | grep pytest` to confirm no zombie processes remain; kill any found.

### Cleanup
```bash
pkill -9 -f "python -m pytest"       # stale test processes
pkill -9 -f "pi_orchestrator_v2"     # stale pipeline runs
pgrep -la python | grep -E "pytest|orchestrator"  # verify clean
```

---

---

## Agent modes
```
AGENT_MODE=local   # default — all agents run as direct function calls, no API cost
AGENT_MODE=sdk     # CSO + discovery_refinement dispatch via Claude API
```

## Credentials (.env)
```
ANTHROPIC_API_KEY=<key>   # only needed for AGENT_MODE=sdk
OPENGWAS_JWT=<jwt>        # expires 2026-05-06
NCBI_API_KEY=<key>
GRAPH_DB_PATH=./data/graph_test.kuzu   # tests only; unset in prod → per-disease graph_{slug}.kuzu
```
