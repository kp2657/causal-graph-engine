# GPS Compound Screening Algorithm

## Overview

The **Gene Perturbation Signature (GPS)** screen identifies compounds that reverse the disease transcriptional state or mimic the knockdown of high-confidence causal targets. It implements the **connectivity map** approach (Lamb et al. 2006 Science; Xing et al. 2026 Cell) using the Bin-Chen Lab GPS Docker tool against the Enamine HTS library (~2M compounds).

GPS runs as the **Tier 4 chemistry step** — after genetic target prioritisation — and returns three complementary screen types:

| Screen type | Input signature | Question |
|---|---|---|
| Disease-state reversal | Whole-disease DEG (disease vs. normal cells) | What compounds reverse the AMD/CAD transcriptional state? |
| Program reversal | Top NMF program gene loadings | What compounds specifically suppress the complement/lipid program? |
| KO emulation | Perturb-seq KO signature (negated) | What compounds mimic loss-of-function of CFH / PCSK9? |

---

## Disease-State Signature Construction

The disease-state signature is built from a **comparison of disease vs. healthy cell transcriptomes** with a three-tier fallback chain:

### Priority 1: CELLxGENE h5ad pseudo-bulk DEG

```python
disease_cells = adata[adata.obs["disease"] == "age-related macular degeneration"]
normal_cells  = adata[adata.obs["disease"] == "normal"]
log2fc = log2(mean(disease) + 1) - log2(mean(normal) + 1)
```

Branching-probability (BP) cell selection is applied when a latent cache exists: disease cells are selected by highest BP (most transitioning), normal cells by lowest BP (most stable). This sharpens the DEG signal at the disease decision boundary.

Cell types by disease:
| Disease | Cell type | h5ad source |
|---|---|---|
| AMD | RPE (retinal pigment epithelium) | CELLxGENE AMD collection |
| CAD | Cardiac endothelial → SMC → hepatocyte | CELLxGENE CAD + liver |

### Priority 2: `tau_disease_log2fc` from target records

Per-target log2FC values computed during the pipeline run (from CELLxGENE DEG). Used when h5ad is not cached locally.

### Priority 3: OTA γ proxy (last resort)

Normalized OTA γ values (`γ / max_γ`) stand in for log2FC when no h5ad or DEG data is available. This provides ~50–200 GPS-compatible genes for AMD/CAD.

**OTA proxy trimming:** genes are sorted by |γ| descending and trimmed at the point where cumulative |γ| weight reaches 90% of total. Low-γ genes are pure permutation cost — they increase `n_permutations = n_sig_genes` without proportional RGES signal benefit (GPS RGES is rank-weighted). Bounds: min 50 genes, max 1000.

### Signature trimming (h5ad path)

The h5ad DEG path applies **elbow trimming** (kneedle algorithm) to keep 700–1000 genes:
- Sort genes by |log2FC| descending (scree curve)
- Normalise rank and |log2FC| to [0, 1]
- Find gene maximally distant from the diagonal (maximum curvature = elbow)
- Clamp to [700, 1000]

The minimum of 700 ensures ≥700 permutations for a calibrated null distribution (`n_permutations = n_sig_genes` inside GPS). Empirically: 183-gene signature → 183 perms → 0 hits; 1000-gene signature → well-calibrated Z_RGES.

---

## Background Distribution (BGRD)

The BGRD is a permutation-based null distribution for RGES scores. It answers: "how extreme is this compound's RGES under a random signature?"

### Construction

For a signature of size N, the BGRD uses **N permutations** (capped 700–1000). Each permutation randomises gene ordering in the signature, re-scoring all compounds. The resulting RGES distribution forms the null under which Z_RGES is calibrated.

**BGRD path:** `data/gps_bgrd/BGRD__{label}.pkl`

Labels are deterministic: `{disease_key}_disease_state`, `{disease_key}_prog_{sha1[:8]}`, `{gene}_emulation`.

### BGRD lifecycle

1. **First run (no cache):** BGRD is computed from scratch. 0.5–6h depending on signature size. Timeout: 6h.
2. **Subsequent runs (cache hit):** BGRD loaded from disk. GPS runs reversal scoring only. Timeout: 2h.
3. **Stale BGRD:** Delete and recompute if the disease h5ad is updated or the signature composition changes materially:
   ```bash
   rm data/gps_bgrd/BGRD__amd_*.pkl   # AMD
   rm data/gps_bgrd/BGRD__coronary*.pkl  # CAD
   ```
4. **New program or KO target:** First run computes BGRD for that label; all subsequent runs reuse it.

---

## RGES Scoring

The **Reversed Gene Expression Score (RGES)** measures anti-correlation between the disease signature and a compound's expression perturbation profile.

```
Z_RGES = (RGES − mean_BGRD) / std_BGRD
```

A compound is called a **reverser** when `|Z_RGES| > z_threshold` (default 2.0 → two-tailed p < 0.05 ≈ FDR 5% under N(0,1) null). Falls back to top-N by raw RGES if fewer than 3 compounds pass the threshold.

---

## Program-Level GPS Screens

Program reversal screens use the **NMF gene loading vector** as the GPS input signature, signed by causal direction:

- **Risk program** (net OTA contribution > 0): positive loadings passed → reversers inhibit the program
- **Protective program** (net contribution < 0): loadings negated → reversers reinforce it

Up to 5 programs are screened (set by `top_n_programs=5`), filtered to programs with weight ≥ 10% of the top program's weight.

**Signature sources (priority order):**
1. NMF gene loadings from cNMF h5ad analysis (transcriptomic, ideal for GPS RGES)
2. OTA contribution fractions + MSigDB Hallmark gene expansion (fallback when cNMF has < 5 GPS-compatible genes)

**Jaccard deduplication:** before queuing a program, its GPS-compatible gene set is checked against all already-queued programs. If Jaccard similarity ≥ 0.70, the program is skipped — near-identical gene sets produce near-identical BGRD distributions and RGES hit lists, wasting a Docker run.

---

## KO Emulation Screens

For the top-3 Tier2 genetic anchors with `ota_gamma > 0.1`, GPS finds compounds that **mimic knockdown** of that target:

1. Fetch Perturb-seq KO signature for the gene from `mcp_servers/perturbseq_server.py`
2. Negate the signature: `reversers of (−KO) = mimics of KO`
3. Run GPS → compounds that mimic therapeutic loss-of-function

Gate: `dominant_tier.startswith("Tier2") AND abs(ota_gamma) > 0.1`. Sorted by |γ| descending so the highest-confidence anchors are always screened if the cap bites.

For AMD: expects CFH, C3, VEGFA (or nearest Tier2 anchors).
For CAD: expects PCSK9, HMGCR, LPA (or nearest Tier2 anchors).

---

## Parallel Execution

All GPS Docker containers are CPU-bound subprocesses — parallelism is safe with `ThreadPoolExecutor`.

**Execution order:**
1. Disease-state screen — **synchronous** (first; result gates program screens)
2. Program screens — **concurrent** (`ThreadPoolExecutor(max_workers=3)`)
3. KO emulation screens — **concurrent** (`ThreadPoolExecutor(max_workers=3)`, runs before disease screen)
4. ChEMBL annotation — **sequential** (after all Docker screens; fork-safety on macOS)

**Early-exit cascade:** if the disease-state screen runs but returns 0 hits, all program screens are skipped immediately. Root causes to check: (1) BGRD not cached, (2) signature < 700 GPS-compatible genes, (3) GPS Docker image stale.

**Wall-time improvement:**
- Previous (sequential): `1 + 5 + 3 = 9 screens × single_screen_time`
- Current (parallel): `1 + ceil(5/3) + ceil(3/3) ≈ 1 + 2 + 1 = 4 rounds × single_screen_time`

---

## Cross-Screen Compound Overlap

`find_overlapping_compounds` identifies compounds appearing in ≥2 screens. These are the highest-confidence hits: they reverse the whole-disease state AND specifically suppress a causal program AND/OR mimic a KO of a validated target.

Overlap compounds are surfaced as `gps_priority_compounds` in the pipeline output and their putative targets are passed to the discovery refinement agent for latent axis analysis.

---

## Target-Cluster-Centric Analysis

`_target_cluster_analysis` in `discovery_refinement_agent` cross-references GPS putative targets (from ChEMBL Tanimoto annotation) against cNMF + MSigDB program gene sets. Programs where GPS hit targets overlap ≥5% of the program gene set (and ≥2 genes) surface as `latent_therapeutic_axes`.

This reveals programs not explicitly screened by GPS but whose genes appear in compound target annotations — latent therapeutic axes discoverable only through chemical convergence.

---

## Output Fields

```json
{
  "gps_disease_reversers": [
    {
      "compound_id": "CID:12345",
      "rges": -0.41,
      "z_rges": -3.2,
      "rank": 1,
      "screen_type": "disease_state_reversal",
      "annotation": {
        "compound_name": "rapamycin",
        "mechanism_of_action": "mTOR inhibitor",
        "putative_targets": ["MTOR"],
        "max_phase": 4
      }
    }
  ],
  "gps_program_reversers": {
    "complement_program": [ ... ],
    "angiogenesis_program": [ ... ]
  },
  "gps_emulation_candidates": [
    {
      "compound_id": "CID:67890",
      "target": "CFH",
      "emulation_tier": "Tier2_genetic_anchor",
      "emulation_gamma": 0.561
    }
  ],
  "gps_priority_compounds": [
    {
      "compound_id": "CID:12345",
      "n_screens": 3,
      "screens": ["disease_state_reversal", "program_reversal:complement_program", "target_emulation:CFH"],
      "avg_rges": -0.38
    }
  ],
  "gps_programs_screened": [
    {"program_id": "complement_program", "direction": "risk", "net_weight": 0.42, "n_sig_genes": 87, "sig_source": "nmf_loadings"}
  ]
}
```

---

## Docker Setup

```bash
docker pull binchengroup/gpsimage:latest
```

If Docker is unavailable, all GPS screens are skipped. The genetic target ranking still completes. GPS status is reported in `pipeline_warnings`:

```json
{"pipeline_warnings": ["GPS disease screen skipped: Docker / GPS image unavailable"]}
```

---

## References

- Xing et al. (2026) Cell — GPS4Drugs: RGES, Z_RGES normalisation, GPS 2198-gene landmark set
- Lamb et al. (2006) Science 313:1929 — Connectivity Map (CMap)
- Chen et al. (2017) Briefings in Bioinformatics — GPS method
- GPS Docker: github.com/Bin-Chen-Lab/GPS

---

## Known Limitations

1. **GPS 2198 landmark set**: GPS uses a 2198-gene landmark set (not L1000). Programs with structural or cell-type-identity genes may have insufficient overlap → OTA proxy fallback.
2. **Cell-line mismatch**: Most compound profiles in the GPS library are from cancer cell lines. AMD RPE and CAD endothelial biology may not be fully captured.
3. **BGRD permutation count = signature size**: Conservative; may underestimate FDR for signatures near the 700-gene minimum.
4. **KO emulation requires Perturb-seq coverage**: Genes absent from the Replogle 2022 RPE1/K562 screens produce empty signatures → KO emulation silently returns [].
