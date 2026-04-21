# Agent Enhancement Plan

This document describes how autonomous agent layers could be added on top of the
static pipeline (`orchestrator/pi_orchestrator_v2.py`) in the future.  The static
pipeline is the source of truth — agents are enhancements, not replacements.

---

## Current architecture (static)

```
analyze_disease_v2()
  ├── Tier 1: phenotype_architect.run() + statistical_geneticist.run()
  ├── Gene list: _collect_gene_list()  [OT L2G + GWAS-validated]
  ├── Regulator nomination: _run_regulator_nomination()
  ├── Tier 2: perturbation_genomics_agent.run() + regulatory_genomics_agent.run()
  ├── Gamma estimates: _get_gamma_estimates()
  ├── Tier 3: causal_discovery_agent.run() + kg_completion_agent.run()
  ├── Scientific reviewer: review_batch()  [plain function]
  ├── Tier 4: target_prioritization_agent.run() + chemistry_agent.run() + clinical_trialist_agent.run()
  └── Tier 5 (collapsed): scientific_writer_agent.run()
```

Each tier's `run()` is a deterministic Python function — no LLM, no API cost.

---

## Where agents add value

### 1. CSO Pre-Pipeline Briefing (Phase P)

**Where**: Before Tier 1, after `disease_name` is known.

**What**: A Claude agent reads prior run summaries, recent literature, and
open questions, then produces a structured briefing that shapes which gene sets
and programs to prioritise.

**Interface**:
```python
from agents.cso.chief_of_staff_agent import run_briefing
cso_briefing = run_briefing({"disease_name": disease_name})
pipeline_outputs["cso_briefing"] = cso_briefing
```

**Gain**: Directs downstream attention; surfaces conflicts the static pipeline
cannot detect (e.g. "the IL-23 pathway is saturated with approved drugs").

---

### 2. CSO Conflict Analysis (Phase P)

**Where**: After Tier 3, before Tier 4.

**What**: Agent reviews divergences between GWAS signal (γ) and Perturb-seq
effect (β) for the same gene — flags cases where genetic evidence points one
way and functional evidence points the other.

**Interface**:
```python
from agents.cso.chief_of_staff_agent import run_conflict_analysis
cso_conflict = run_conflict_analysis(pipeline_outputs)
pipeline_outputs["cso_conflict_analysis"] = cso_conflict
```

**Gain**: Prioritises genes with convergent vs. divergent evidence before
target ranking happens; replaces manual inspection.

---

### 3. Scientific Reviewer Re-delegation (Phase O)

**Where**: After Tier 5 (writer), before final output.

**What**: A structured reviewer checks the `GraphOutput` against QA criteria.
On `REVISE` verdict it can re-delegate specific agents (e.g. re-run
`causal_discovery_agent` with tighter evidence thresholds).

**Interface**:
```python
from orchestrator.agent_runner import AgentRunner
from orchestrator.agent_messages import build_feedback_from_reviewer
runner = AgentRunner()
writer_output = runner.dispatch("scientific_writer_agent", writer_input)
reviewer_output = runner.dispatch("scientific_reviewer_agent", reviewer_input)
feedback = build_feedback_from_reviewer(reviewer_output.results, run_id)
if feedback.has_critical():
    # re-run flagged agents, then re-run writer
    ...
```

**Gain**: Catches systematic issues (e.g. all top-5 targets are virtual) before
the report is written; provides a machine-readable audit trail.

---

### 4. Literature Validation (Phase Q)

**Where**: After Tier 4 prioritization.

**What**: Agent queries PubMed/Semantic Scholar for each top-ranked gene,
returning confidence scores (`literature_confidence`: HIGH/MEDIUM/LOW) and
paper counts.

**Interface**:
```python
runner.dispatch("literature_validation_agent", lit_input)
```

**Gain**: Filters out targets with no published functional evidence; flags
genes that are well-studied in other diseases but novel in the current one.

---

### 5. Red Team Assessment (Phase T)

**Where**: After literature validation.

**What**: Adversarial agent challenges top-5 targets — generates
counterarguments, identifies confounds, assigns CAUTION / DEPRIORITIZE
verdicts.

**Interface**:
```python
runner.dispatch("red_team_agent", rt_input)
```

**Gain**: Surface-level robustness check before clinical translation; documents
known failure modes for each target.

---

### 6. CSO Executive Summary (Phase P, post-run)

**Where**: After all tiers complete.

**What**: Agent synthesises the full run into a structured executive summary
with top insight, research gaps, and next experiments.

**Interface**:
```python
from agents.cso.chief_of_staff_agent import run_exec_summary
cso_summary = run_exec_summary(pipeline_outputs)
```

**Gain**: Replaces the templated `executive_summary` string in the writer with
a reasoned narrative that accounts for conflicts, literature support, and
red-team verdicts.

---

### 7. Discovery Refinement (Phase P)

**Where**: After the CSO executive summary.

**What**: A Haiku-class agent does a second-pass scan for novel opportunities
missed by the main pipeline — understudied genes with moderate γ×β scores,
cross-disease repurposing candidates, regulatory network hubs.

**Interface**:
```python
runner.set_mode("discovery_refinement_agent", "sdk")
runner.set_model("discovery_refinement_agent", "claude-haiku-4-5-20251001")
runner.dispatch("discovery_refinement_agent", discovery_input)
```

**Gain**: Extends the ranked list beyond GWAS-prominent genes; low cost
(Haiku-class) relative to the biological value.

---

## How to re-introduce agents

1. Restore `AgentRunner` import and instantiation in `analyze_disease_v2`.
2. Wrap tier `run()` calls back into `runner.dispatch()` with `AgentInput`.
3. Add the agent steps above at the annotated insertion points.
4. Set `AGENT_MODE=sdk` in `.env` to activate Claude API for CSO and discovery.

The `agents/cso/chief_of_staff_agent.py` and `orchestrator/agent_runner.py`
modules are preserved and fully functional — no re-implementation needed.
