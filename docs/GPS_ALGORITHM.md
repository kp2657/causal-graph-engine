# GPS Compound Screening Algorithm

## Overview

The **Gene Perturbation Signature (GPS)** screen identifies compounds that reverse the disease transcriptional state. It implements the **connectivity map** approach (Lamb et al. 2006 Science; Chen et al. 2017) using the Bin-Chen Lab GPS tool.

The key output is a ranked list of compounds whose transcriptional signatures are anti-correlated with the disease expression signature — i.e., compounds that push cell state from "disease" back toward "healthy."

---

## Biological rationale

Disease states are characterised by a shift in transcriptional programs. If a compound produces a gene expression signature that is the mirror image of the disease signature, it may therapeutically reverse that state. This is distinct from target-based drug discovery — it is phenotypic, searching for reversal of the entire disease program rather than a single protein.

GPS is used as a **Tier 4** step in the pipeline — after genetic target prioritisation — to identify candidate compounds that simultaneously address the disease state.

---

## Disease-State Signature Construction

The disease-state signature is built from a **comparison of disease vs. healthy cell transcriptomes** in CELLxGENE h5ad files.

### Step 1: Load disease and normal cells from h5ad

```python
# Example: AMD RPE cells
disease_cells = adata[adata.obs["disease"] == "age-related macular degeneration"]
normal_cells  = adata[adata.obs["disease"] == "normal"]
```

Cell types used are disease-specific:
| Disease | Cell type | h5ad source |
|---|---|---|
| AMD | RPE (retinal pigment epithelium) | CELLxGENE AMD collection |
| CAD | SMC (smooth muscle) + hepatocyte | CELLxGENE CAD + liver |
| IBD | Intestinal epithelial + macrophage | CELLxGENE gut collection |

### Step 2: Compute differential expression

Log2 fold-change per gene: `log2FC(disease/normal)` computed as mean log-normalized expression difference.

### Step 3: Signature trimming (elbow trim)

The GPS tool accepts signatures of **700–1000 genes** (empirically calibrated in Session 59). Larger signatures include noise; smaller signatures lack coverage.

An elbow-trim selects the top genes by absolute log2FC magnitude:
- Minimum: 700 genes
- Maximum: 1000 genes

```python
sig_genes = sorted(genes, key=lambda g: abs(log2fc[g]), reverse=True)[:max_genes]
```

---

## Background Distribution (BGRD)

The BGRD is a permutation-based null distribution for RGES scores. It answers: "how extreme is this compound's RGES under a random signature?"

### Construction

For a signature of size N, the BGRD uses **N permutations** (one per gene). Each permutation randomises which genes are "up" vs. "down" in the signature, then re-scores all compounds in the L1000 library. The resulting distribution of RGES scores forms the null.

**BGRD path structure:**
```
data/gps_bgrd/BGRD__{disease}__{cell_type}__{n_genes}genes.pkl
```

### BGRD lifecycle

1. **First run**: BGRD is computed from scratch. This takes 2–6 hours depending on signature size.
2. **Subsequent runs**: BGRD is loaded from disk if the signature has not changed.
3. **Stale BGRD**: If the disease h5ad is updated, the BGRD must be deleted and recomputed:
   ```bash
   rm data/gps_bgrd/BGRD__AMD__*.pkl
   ```
4. **Timeout**: GPS with no BGRD uses a 6-hour timeout; with BGRD, 2-hour timeout.

---

## RGES Scoring

The **Reversed Gene Expression Score (RGES)** measures the anti-correlation between the disease signature and a compound's L1000 perturbation profile.

```
RGES = -Σ_{g ∈ up-genes} rank(g in compound_sig) / N_up
     + Σ_{g ∈ down-genes} rank(g in compound_sig) / N_dn
```

A strongly negative RGES means the compound upregulates genes that are down in disease and downregulates genes that are up in disease — i.e., the compound reverses the disease state.

### Z_RGES — normalised score

RGES is z-scored against the BGRD:

```
Z_RGES = (RGES - mean_BGRD) / std_BGRD
```

A compound is called a **disease-state reverser** when:

```
Z_RGES ≤ -threshold    (default threshold = 2.0)
```

### Dynamic Z_RGES threshold (v0.2.0+)

The threshold is calibrated dynamically based on the BGRD standard deviation and compound library size:
- Default Z = 2.0 (≈5% false-discovery rate under N(0,1) null)
- Raised to Z = 2.5 when < 100 compounds tested (small library → higher FPR)
- Lowered to Z = 1.5 when exploring a novel disease with sparse GWAS anchors

This replaces the previous hard cutoff of `top_n=20` (Session 60 fix).

---

## Program-Level GPS Screens

In addition to disease-state reversal, GPS is run against individual transcriptional programs:

```
Z_RGES per compound per program = program-specific reversal score
```

This identifies compounds that specifically reverse the complement program, the lipid metabolism program, etc. — allowing mechanism-specific drug repurposing.

Up to 5 programs are screened per run (set by `top_n_programs=5`).

---

## Output fields

```json
{
  "compound_id": "CID:12345",
  "compound_name": "rapamycin",
  "z_rges": -3.2,
  "rges": -0.41,
  "rank": 1,
  "annotation": {
    "mechanism_of_action": "mTOR inhibitor",
    "putative_targets": "MTOR",
    "phase": 4
  },
  "program_reversals": {
    "complement_activation": -2.8,
    "RPE_oxidative_stress": -1.9
  }
}
```

---

## Docker requirement

GPS runs inside the Bin-Chen Lab Docker image:

```bash
docker pull binchenlab/gpsimage
```

If Docker is not running, the GPS screen is **skipped**. This is logged in `pipeline_warnings` in the output JSON:

```json
{
  "pipeline_warnings": [
    "GPS screen SKIPPED — Docker daemon not reachable. Run: docker pull binchenlab/gpsimage"
  ]
}
```

The rest of the pipeline (genetic target ranking) proceeds without GPS results.

---

## Running GPS manually

```bash
# Build disease-state signature and run GPS screen
conda run -n causal-graph python -m pipelines.gps_disease_screen \
    --disease "age-related macular degeneration" \
    --cell-type "retinal pigment epithelial cell" \
    --n-genes 850

# Recompute BGRD from scratch
rm data/gps_bgrd/BGRD__AMD__*.pkl
conda run -n causal-graph python -m pipelines.gps_screen \
    --disease AMD --recompute-bgrd
```

---

## References

- Lamb et al. (2006) Science 313:1929 — Connectivity Map (CMap)
- Chen et al. (2017) Briefings in Bioinformatics — GPS method
- Bray et al. (2023) — L1000 CMap compound library
- GPS Docker: github.com/Bin-Chen-Lab/GPS

---

## Known limitations

1. **L1000 coverage**: Only ~1,000 landmark genes are measured in L1000, limiting sensitivity for rare disease programs with non-L1000 drivers.
2. **Cell-line mismatch**: Most L1000 compound profiles are from cancer cell lines (MCF7, PC3, A549). AMD RPE biology may not be well captured.
3. **BGRD permutation count = signature size**: This is conservative (fewer permutations than a full bootstrap) and may underestimate FDR for small signatures.
4. **No dose–response modelling**: RGES scoring is binary (treated vs. untreated) — the optimal dose is not identified.
