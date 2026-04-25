# Causal Graph Engine — Build State
Last updated: 2026-04-25 (Session 77 — SLE dropped, dead code removed, CAD+RA validated)

---

## Status: v0.2.3-dev — two-disease focus

Active diseases: **CAD + RA**. SLE dropped (OT L2G returns ~3 genes for EFO_0002690 — insufficient for pipeline). AD, IBD, DED all dropped previously.

117 tests passing, 3 skipped.

---

## Active diseases

### CAD (coronary artery disease) — validated
- Last run: 2026-04-25 (v0.2.3-dev), 1,586 targets, 4,896 causal edges, 15 program γ edges
- Perturb-seq: Schnitzler GSE210681 (cardiac endothelial/HCASMC) + CZI CD4+ T
- eQTL: GTEx Whole_Blood + Liver + Artery_Coronary
- scRNA-seq: cardiac endothelial (CellxGENE myocardial infarction label)
- GPS: 24 novel-mechanism compounds (Z_RGES < −3.5σ)
- Known targets recovered: PCSK9 (rank 132), LPA (rank 47), IL6R (rank 90), APOB (rank 135)

### RA (rheumatoid arthritis) — validated
- Last run: 2026-04-25 (v0.2.3-dev), 244 targets, 488 causal edges, 12 program γ edges
- Perturb-seq: CZI CD4+ T (GSE314342, 11,327 genes)
- eQTL: OneK1K scQTL (14 immune cell types) via eQTL Catalogue
- scRNA-seq: CellxGENE CD4+ T cell
- GPS: 54 novel-mechanism compounds
- Top targets: NFKBIE (Tier1), UQCC2 (Tier1), ACTR3 (Tier1), MYC (Tier1), CTLA4 (rank 44)
- Known targets recovered: IL6R (rank 140), JAK1 (rank 201), CTLA4 (rank 44)

---

## Immediate next steps

1. **Expand RA gene list** — only 447 genes from OT L2G; investigate whether L2G threshold or EFO EFO_0000685 is limiting
2. **GPS program reversers** — CAD has 4 program GPS screens but 0 compounds post-threshold; RA has 5. Check if Z_RGES 3.5σ is too aggressive for program screens (which have smaller gene sets than disease screen)
3. **`tr_track = None` bug** — `compute_therapeutic_redirection_per_celltype()` never writes `tr_track` for Path C (Perturb-seq) genes; fix pending

---

## Known pipeline bugs (carry-forward)
- **KNOWN:** `tr_track = None` for Path C genes (Perturb-seq genes with real β): `compute_therapeutic_redirection_per_celltype()` predates two-track design and never writes `tr_track`; fix pending
- **KNOWN:** GPS program screen compounds = 0 for both CAD/RA — 3.5σ threshold may be too aggressive for program-level screens (smaller gene set → fewer reverters)
- n_tier3_edges=0 is CORRECT: LINCS removed in session 57; Tier3 edges no longer produced

---

## Architecture quick reference
```
orchestrator/pi_orchestrator_v2.py      — 5-tier pipeline (analyze_disease_v2 / run_tier4)
config/scoring_thresholds.py            — all numeric constants with citations
models/disease_registry.py             — canonical disease name / EFO mapping + DISEASE_PROGRAMS
agents/tier{1-5}_*/                    — per-tier agents
pipelines/ota_beta_estimation.py        — Tier1/2/3 β estimation
pipelines/ota_gamma_estimation.py       — GWAS-enrichment γ via Reactome + S-LDSC
pipelines/cnmf_programs.py             — MSigDB Hallmark + cNMF program loading
pipelines/state_space/                  — TR, conditional β/γ, state influence, latent model
pipelines/gps_disease_screen.py        — GPS disease-state + program reversal (parallel Docker)
mcp_servers/                           — 8 live data servers
graph/                                 — Kùzu graph DB + RDF/Turtle export
```
