# Causal Graph Engine — Claude Instructions

This file governs Claude Code behavior for this project across all sessions.
**Read this before starting any work.** Update it as the project evolves.

---

## What This Project Does

A multiagent causal genomics pipeline implementing the **Ota et al. framework**:

```
γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait})
```

- **β**: causal effect of gene X on biological program P (from Perturb-seq / eQTL-MR / LINCS L1000)
- **γ**: causal effect of program P on disease trait (from GWAS S-LDSC + TWMR)
- Output: ranked drug targets with Kùzu graph DB + RDF export + Markdown report

**Current disease**: Coronary Artery Disease (EFO_0001645). Next: IBD.

---

## Environment

```bash
conda activate causal-graph          # Python 3.12, pydantic v2
cd causal-graph-engine/
```

**Never use the base `python` or `python3` — they are Python 3.8.**

---

## How to Run

```bash
# Full pipeline (v2 multiagent orchestrator)
conda run -n causal-graph python -m orchestrator.pi_orchestrator_v2 analyze_disease_v2 "coronary artery disease"

# CLI entry point
conda run -n causal-graph python main.py analyze "coronary artery disease"

# Run tests (fast unit tests only — ~7s)
conda run -n causal-graph python -m pytest tests/ -q -m "not integration"

# Run all tests including live API calls (~5min)
conda run -n causal-graph python -m pytest tests/ -q
```

---

## Commit Discipline

**Before every commit:**
1. Run `conda run -n causal-graph python -m pytest tests/ -q -m "not integration"`
2. All tests must pass. Never commit code that breaks passing tests.
3. Commit after every meaningful unit of work — do not batch unrelated changes.
4. Use a descriptive commit message that explains *why*, not just *what*.

```bash
conda run -n causal-graph python -m pytest tests/ -q -m "not integration" && git add -p && git commit -m "..."
```

---

## Session Resumption

When starting a new session:
1. Read `STATE.md` — current objective, completed steps, NEXT steps
2. Read `CHANGELOG.md` — recent session history and failed approaches
3. Run the test suite to confirm baseline: `pytest tests/ -q -m "not integration"`
4. Check `data/analyze_coronary_artery_disease.json` to see last pipeline output

**Never re-implement something in STATE.md marked ✓ DONE without first checking that the existing implementation is broken.**

---

## Success Criteria

The pipeline is working when:
- `main.py analyze "coronary artery disease"` completes with `pipeline_status: SUCCESS`
- `anchor_edge_recovery ≥ 0.80` (4/5 CAD REQUIRED_ANCHORS)
- `n_edges_written ≥ 8` (Ota γ composite edges)
- `data/analyze_coronary_artery_disease.md` exists with ranked target table
- `data/exports/coronary_artery_disease.ttl` exists (RDF export)

---

## Architecture: Key Files

```
orchestrator/
  pi_orchestrator_v2.py   — Main entry point (5-tier multiagent pipeline)
  agent_runner.py         — AgentRunner: local/sdk mode dispatch
  message_contracts.py    — AgentInput/AgentOutput pydantic v2 envelopes

agents/
  tier1_phenomics/        — phenotype_architect, statistical_geneticist, somatic_exposure_agent
  tier2_pathway/          — perturbation_genomics_agent, regulatory_genomics_agent
  tier3_causal/           — causal_discovery_agent, kg_completion_agent
  tier4_translation/      — target_prioritization_agent, chemistry_agent, clinical_trialist_agent
  tier5_writer/           — scientific_writer_agent

pipelines/
  ota_beta_estimation.py  — β fallback chain (T1 Perturb-seq → T2 eQTL-MR → T3 LINCS → Virtual)
  ota_gamma_estimation.py — γ from GWAS S-LDSC + TWMR

graph/
  db.py                   — Kùzu CRUD
  schema.py               — DISEASE_CELL_TYPE_MAP, DISEASE_TRAIT_MAP, ANCHOR_EDGES
  export.py               — RDF/Turtle, JSON-LD, CSV export

mcp_servers/              — 8 live MCP servers (GWAS, gnomAD, GTEx, CELLxGENE, etc.)

data/
  graph.kuzu              — Live Kùzu graph database
  analyze_*.json          — Pipeline JSON output
  analyze_*.md            — Pipeline Markdown report
  exports/                — RDF/JSON-LD/CSV exports
```

---

## Critical Rules (Do Not Break)

| Rule | Why |
|------|-----|
| Use pydantic v2 API only (`field_validator`, `model_validator`, `X \| None`, `model_dump()`) | Codebase fully migrated; pydantic v1 syntax breaks silently |
| `float('nan')` for missing β/γ, never `0.0` | 0.0 × γ produces phantom zero-effect edges |
| Co-expression is NOT a valid β source | Direction/causation requires perturbation or genetic instrument |
| Every REQUIRED_ANCHOR must also be in ANCHOR_EDGES | Schema consistency; new regression test guards this |
| Set `GRAPH_DB_PATH` to a temp dir in all DB tests | Avoids corrupting the live `data/graph.kuzu` |
| `model_construct()` for unknown-agent stubs in AgentRunner | Bypasses AgentName Literal validation without crashing |

---

## Autonomous Session Instructions

For long-running autonomous work:

1. **Scope before coding**: write out the plan in a brief comment or update STATE.md CURRENT OBJECTIVE before touching any file
2. **Test-driven**: write the test first if adding new functionality
3. **Commit checkpoints**: commit after each test goes green, not at the end of a big batch
4. **Document failures**: if an approach doesn't work, add it to `CHANGELOG.md` under "Failed approaches" before pivoting
5. **Verify before declaring done**: re-run the full test suite and check the actual output files

**If asked to "keep working until done"** — use the following stopping criterion:
- All unit tests pass (`pytest tests/ -q -m "not integration"`)
- `main.py analyze "coronary artery disease"` runs without exception
- STATE.md CURRENT OBJECTIVE is updated to the next step

---

## SDK Mode (Claude API Subagents)

To flip any agent to real Claude API:
```python
from orchestrator.pi_orchestrator_v2 import analyze_disease_v2
result = analyze_disease_v2(
    "coronary artery disease",
    mode_overrides={"chemistry_agent": "sdk"},  # chemistry_agent is the bottleneck
)
```

Requires `ANTHROPIC_API_KEY` in `.env`. Use `claude-haiku-4-5-20251001` for speed/cost.

---

## Known Flaky Tests (Do Not Fix)

These 2 tests fail intermittently due to external API instability — not code bugs:
- `test_chemistry_server.py::TestPubChemLive::test_aspirin_by_name` — PubChem formula field absent
- `test_clinical_trials_server.py::TestClinicalTrialsLive::test_cad_recruiting_trials` — totalCount absent from v2 API

---

## Credentials (.env)

```
ANTHROPIC_API_KEY=<key>         # For SDK mode agent dispatch
OPENGWAS_JWT=<jwt>              # Expires 2026-04-06 — renew at api.opengwas.io
NCBI_API_KEY=<key>
CROSSREF_MAILTO=kenneth.pham@columbia.edu
GRAPH_DB_PATH=./data/graph.kuzu
```
