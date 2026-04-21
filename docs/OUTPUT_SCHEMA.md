# Output Schema Reference

The pipeline writes `data/analyze_{disease_slug}.json`. This document describes every field.

---

## Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `disease_name` | str | Canonical disease name used as pipeline input |
| `efo_id` | str | EFO ontology identifier (from disease_registry.py) |
| `pipeline_version` | str | Semantic version (e.g. "0.2.0") |
| `generated_at` | str | ISO-8601 UTC timestamp |
| `pipeline_status` | str | "SUCCESS" or "PARTIAL" |
| `pipeline_duration_s` | float | Wall-clock runtime in seconds |
| `target_list` | list[TargetRecord] | Primary ranked target list (all evidence tiers) |
| `genetic_anchors` | list[TargetRecord] | Subset: genetically grounded targets only |
| `locus_nominated_anchors` | list[TargetRecord] | Subset: OT L2G ≥ 0.10 locus candidates |
| `state_space_targets` | list[StateRecord] | Top 20 targets from state-space (TR/stability) scoring |
| `repurposing_opportunities` | list[TargetRecord] | Subset: max_phase ≥ 2 with OT score > 0.5 |
| `disease_specific_pipeline` | list[TargetRecord] | Disease-specific targets (tau ≥ 0.70) |
| `mechanistic_candidates` | list[TargetRecord] | Mechanistic-only targets (no genetic grounding) |
| `n_tier1_edges` | int | Number of Tier1 (Perturb-seq) gene–program edges |
| `n_tier2_edges` | int | Number of Tier2 (eQTL-MR / pQTL) edges |
| `n_tier3_edges` | int | Number of Tier3 (LINCS) edges |
| `n_virtual_edges` | int | Number of provisional_virtual (annotation proxy) edges |
| `total_edges_written` | int | Edges written to Kùzu graph |
| `executive_summary` | str | Free-text pipeline narrative (Tier 5 writer) |
| `target_table` | str | Markdown-formatted top-20 target table |
| `state_space_table` | str | Markdown-formatted state-space target table |
| `top_target_narratives` | list[str] | Per-gene mechanistic narratives for top 3 targets |
| `evidence_quality` | dict | Aggregate evidence quality stats |
| `limitations` | str | Auto-generated limitations paragraph (data gaps) |
| `warnings` | list[str] | Per-gene warnings from scoring and QC |
| `pipeline_warnings` | list[str] | Pipeline-level warnings (degraded data sources) |
| `data_completeness` | dict | Which optional data sources were loaded |
| `gps_disease_state_reversers` | list[GPSRecord] | GPS compounds that reverse the disease-state DEG signature |
| `gps_program_reversers` | list[GPSRecord] | GPS compounds that reverse top NMF program signatures |
| `gps_priority_compounds` | list[GPSRecord] | Intersection of disease-state and program reversers |
| `pi_reviewed` | bool | Whether PI-review tier escalation logic ran |
| `n_escalations` | int | Number of targets tier-upgraded by PI review |
| `edge_write_breakdown` | dict | Edge counts broken down by tier |

---

## TargetRecord fields

Each entry in `target_list` (and subset lists) is a dict with these fields:

### Identity and ranking

| Field | Type | Description |
|-------|------|-------------|
| `target_gene` | str | HGNC gene symbol |
| `rank` | int | 1-based rank (sorted by `ota_gamma` descending within partition) |
| `target_score` | float | Alias for `ota_gamma` (backward compatibility) |

### OTA causal score

| Field | Type | Description |
|-------|------|-------------|
| `ota_gamma` | float | Final causal effect score: `Σ_P (β_{gene→P} × γ_{P→trait})` or OT L2G fusion |
| `ota_gamma_raw` | float | Raw pre-fusion score (before Bayesian update) |
| `ota_gamma_sigma` | float | Prior uncertainty (smaller = stronger causal evidence) |
| `ota_gamma_ci_lower` | float | 95% CI lower bound |
| `ota_gamma_ci_upper` | float | 95% CI upper bound |
| `causal_gamma` | float | Same as `ota_gamma` (explicit alias) |
| `causal_gamma_source` | str | `"perturb_x_program"` = direct Σ(β×γ); `"ot_l2g_fusion"` = gene not in Perturb-seq library |

### Evidence tier

| Field | Type | Description |
|-------|------|-------------|
| `evidence_tier` | str | See tier table below |
| `partition` | str | `"genetically_grounded"` (OT L2G ≥ 0.10) or `"mechanistic_only"` |

**Evidence tier values:**

| Value | Meaning |
|-------|---------|
| `Tier1_Interventional` | CRISPR Perturb-seq β (Replogle 2022 RPE1 or K562) |
| `Tier2_Convergent` | GTEx eQTL-MR + COLOC H4 ≥ 0.80 |
| `Tier2p_pQTL_MR` | pQTL-MR (UKB-PPP, deCODE, INTERVAL) |
| `Tier2rb_RareBurden` | UKB WES rare-variant burden direction |
| `Tier2pt_ProteinChannel` | Protein abundance pQTL |
| `Tier2.5` / `Tier2_eQTL_direction` | eQTL direction only (COLOC H4 < 0.80; no β magnitude) |
| `Tier3_Provisional` | LINCS L1000 (cell-line mismatch) |
| `provisional_virtual` | Gene absent from Perturb-seq library; `causal_gamma` from OT L2G + COLOC fusion |

### Genetic evidence

| Field | Type | Description |
|-------|------|-------------|
| `genetic_evidence_score` | float | Aggregate genetic score (OT L2G or equivalent) [0, 1] |
| `ot_l2g_score` | float | Open Targets locus-to-gene score [0, 1] |
| `ot_score` | float | Open Targets overall association score [0, 1] |
| `ot_genetic_score` | float | Alias for `genetic_evidence_score` |

### Mechanistic signal

| Field | Type | Description |
|-------|------|-------------|
| `mechanistic_score` | float | State-space mechanistic composite [0, 1] |
| `mechanistic_category` | str | `"genetically_grounded"`, `"high_reward_mechanistic"`, or `"mechanistic_only"` |
| `therapeutic_redirection` | float | TR score: alignment of gene KO signature with disease→normal axis |
| `stability_score` | float | State-space stability contribution |
| `entry_score` | float | Healthy-state entry probability contribution |
| `persistence_score` | float | Disease-state exit probability contribution |
| `recovery_score` | float | Recovery trajectory score |
| `boundary_score` | float | Disease boundary crossing score |
| `marker_score` | float | Disease-state marker score (high = likely consequence, not cause) |

### Translatability

| Field | Type | Description |
|-------|------|-------------|
| `max_phase` | int | Maximum clinical trial phase across all indications |
| `known_drugs` | list[str] | Drug names from ChEMBL / Open Targets |
| `pli` | float\|null | gnomAD pLI (probability of loss-of-function intolerance) |
| `safety_flags` | list[str] | Safety signals from Open Targets / literature |

### Specificity

| Field | Type | Description |
|-------|------|-------------|
| `tau_disease_specificity` | float\|null | τ specificity score from CELLxGENE h5ad [0, 1] |
| `tau_disease_log2fc` | float\|null | log2FC between disease and normal cells |
| `tau_specificity_class` | str\|null | `"disease_specific"`, `"moderately_specific"`, `"ubiquitous"` |
| `specificity_score` | float | Combined specificity score (reporting only; not a sort key) |
| `bimodality_coeff` | float | Bimodality coefficient from expression distribution |
| `beta_amd_concentration` | float | Fraction of β-mass in AMD-specific programs (AMD only; diagnostic for pleiotropic genes) |

### Program decomposition

| Field | Type | Description |
|-------|------|-------------|
| `top_programs` | list or dict | Top NMF/Hallmark programs driving β×γ product |
| `program_drivers` | dict | Classification of program contribution: top_program, spread, disease_specific_pct |

### Quality / flags

| Field | Type | Description |
|-------|------|-------------|
| `flags` | list[str] | See flag table below |
| `scone_confidence` | float | SCONE causal discovery confidence [0, 1] |
| `scone_flags` | list[str] | SCONE-specific warnings |
| `tier_upgrade_log` | list[dict] | Records any tier re-assignments applied during scoring |
| `evidence_disagreement` | list[dict] | Cross-evidence conflicts (sign flips, bulk vs sc disagreement) |
| `key_evidence` | list[str] | Human-readable evidence summary strings |
| `brg_score` | float\|null | BRG novel candidate score (when available) |

**Flag values:**

| Flag | Meaning |
|------|---------|
| `first_in_class` | No approved drug; high γ — novel target opportunity |
| `repurposing_candidate` | Existing drug in clinical trials (max_phase ≥ 2) |
| `provisional_virtual` | Gene absent from Perturb-seq library; causal_gamma from OT L2G fusion |
| `not_in_perturb_library` | Same as provisional_virtual — explicit label for downstream filtering |
| `highly_specific` | τ ≥ 0.70 (disease-specific expression) |
| `bimodal_expression` | Bimodality coefficient > 0.555 (subpopulation heterogeneity) |
| `convergent_controller` | High TR + High stability — strong mechanistic signal |
| `no_genetic_grounding` | No GWAS/OT L2G evidence; mechanistic-only candidate |
| `marker_gene` | High marker_score — likely disease consequence, not cause |
| `inflated_gamma_essential` | Housekeeping gene with inflated γ (sorted to bottom) |
| `evidence_disagreement_block` | Cross-evidence sign conflict — treat with caution |
| `evidence_disagreement_flag` | Minor cross-evidence inconsistency |
| `chip_mechanism` | Clonal haematopoiesis mechanism |
| `strong_trajectory_signal` | TR > 0.1 — strong state-space redirection |
| `high_escape_risk` | GPS / historical MoA suggests high resistance risk |

---

## GPSRecord fields

Each entry in `gps_disease_state_reversers` and `gps_program_reversers`:

| Field | Type | Description |
|-------|------|-------------|
| `compound_id` | str | CID (PubChem) or compound name |
| `rges` | float | Reversed Gene Expression Score (negative = reversal) |
| `z_rges` | float | Z-score of RGES against permutation null |
| `p_value` | float | Empirical p-value from permutation |
| `rank` | int | Rank within screen (1 = strongest reverser) |
| `source_library` | str | Compound library (e.g. "CMAP", "LINCS_L1000") |
| `screen_type` | str | `"disease_state"` or `"program:{program_name}"` |
| `note` | str | Human-readable annotation |

---

## data_completeness fields

| Key | Meaning |
|-----|---------|
| `h5ad_loaded` | Whether disease-matched h5ad was loaded for GPS |
| `perturb_seq_loaded` | Whether Replogle h5ad was loaded for Tier1 β |
| `gps_docker_available` | Whether GPS Docker container was running |
| `opengwas_available` | Whether IEU OpenGWAS JWT was valid |
| `eqtl_catalogue_available` | Whether eQTL Catalogue responded |
