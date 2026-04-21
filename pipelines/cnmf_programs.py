"""
cnmf_programs.py — Cellular program definitions for the Ota framework.

Programs are cell-type-specific gene regulatory modules that mediate the
gene → program → trait causal chain.  This module provides three program
sources in decreasing order of biological specificity:

  1. cNMF from Perturb-seq h5ad (cell-type-specific, most informative)
     STUB — requires Figshare download (357 MB pseudo-bulk h5ad).

  2. MSigDB Hallmark gene sets (universal, free, no download)
     50 well-curated gene sets representing coherent biological processes.
     Available immediately via REST API.  Used as universal fallback and
     as the program vocabulary for S-LDSC γ estimation across all diseases.

  3. Disease-specific curated programs (hardcoded from literature)
     Used when neither cNMF nor MSigDB coverage is satisfactory.

MSigDB Hallmark is the recommended starting point for γ_{P→trait} estimation
because S-LDSC enrichment of Hallmark sets in GWAS heritability is a
well-established analysis in the literature (Finucane et al. 2018).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

# MSigDB REST API (Broad Institute) — no auth required
_MSIGDB_BASE = "https://www.gsea-msigdb.org/gsea/msigdb"
_MSIGDB_GENE_SET_URL = (
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
    "2024.1.Hs/h.all.v2024.1.Hs.json"
)

# Mapping from MSigDB Hallmark names to our pipeline's program vocabulary.
# The left side is MSigDB's canonical name; right side is our internal name.
HALLMARK_TO_PROGRAM: dict[str, str] = {
    "HALLMARK_INFLAMMATORY_RESPONSE":         "inflammatory_NF-kB",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB":       "inflammatory_NF-kB",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING":        "IL-6_signaling",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE":      "MHC_class_II_presentation",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE":      "interferon_alpha",
    "HALLMARK_CHOLESTEROL_HOMEOSTASIS":        "lipid_metabolism",
    "HALLMARK_FATTY_ACID_METABOLISM":          "lipid_metabolism",
    "HALLMARK_DNA_REPAIR":                     "DNA_methylation_maintenance",
    "HALLMARK_G2M_CHECKPOINT":                 "G2M_phase_program",
    "HALLMARK_MYC_TARGETS_V1":                 "proliferation_MYC",
    "HALLMARK_MYC_TARGETS_V2":                 "proliferation_MYC",
    "HALLMARK_E2F_TARGETS":                    "cell_cycle_E2F",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": "EMT_program",
    "HALLMARK_APOPTOSIS":                      "apoptosis_program",
    "HALLMARK_HYPOXIA":                        "hypoxia_program",
    "HALLMARK_MTORC1_SIGNALING":               "mTORC1_signaling",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING":        "PI3K_AKT_program",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION":      "oxidative_phosphorylation",
    "HALLMARK_GLYCOLYSIS":                     "glycolysis_program",
    "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY": "ROS_program",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE":      "UPR_program",
    "HALLMARK_P53_PATHWAY":                    "p53_program",
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING":     "Wnt_signaling",
    "HALLMARK_NOTCH_SIGNALING":                "Notch_signaling",
    "HALLMARK_HEDGEHOG_SIGNALING":             "Hedgehog_signaling",
    "HALLMARK_ANGIOGENESIS":                   "angiogenesis_program",
    "HALLMARK_COAGULATION":                    "coagulation_program",
    "HALLMARK_COMPLEMENT":                     "complement_program",
    "HALLMARK_IL2_STAT5_SIGNALING":            "IL-2_STAT5_signaling",
    "HALLMARK_KRAS_SIGNALING_UP":              "KRAS_signaling",
    "HALLMARK_TGF_BETA_SIGNALING":             "TGF_beta_signaling",
    "HALLMARK_ANDROGEN_RESPONSE":              "androgen_response",
    "HALLMARK_ESTROGEN_RESPONSE_EARLY":        "estrogen_response",
    "HALLMARK_UV_RESPONSE_UP":                 "UV_response",
    "HALLMARK_XENOBIOTIC_METABOLISM":          "xenobiotic_metabolism",
    "HALLMARK_BILE_ACID_METABOLISM":           "bile_acid_metabolism",
    "HALLMARK_HEME_METABOLISM":                "heme_metabolism",
    "HALLMARK_MITOTIC_SPINDLE":                "mitotic_spindle",
    "HALLMARK_SPERMATOGENESIS":                "spermatogenesis",
    "HALLMARK_PANCREAS_BETA_CELLS":            "pancreas_beta_cell_program",
    "HALLMARK_MYOGENESIS":                     "myogenesis_program",
    "HALLMARK_ADIPOGENESIS":                   "adipogenesis_program",
    "HALLMARK_PEROXISOME":                     "peroxisome_program",
    "HALLMARK_PROTEIN_SECRETION":              "protein_secretion",
    "HALLMARK_ALLOGRAFT_REJECTION":            "allograft_rejection",
    "HALLMARK_IMMUNE_EVASION":                 "immune_evasion",
}

# Reverse map: internal program_id → list of MSigDB Hallmark names that map to it.
# Used by GPS program screen to fetch the full Hallmark gene set for a program.
PROGRAM_TO_HALLMARKS: dict[str, list[str]] = {}
for _h, _p in HALLMARK_TO_PROGRAM.items():
    PROGRAM_TO_HALLMARKS.setdefault(_p, []).append(_h)

# In-process cache for the MSigDB gene set fetch (HTTP, avoid repeat calls per run)
_MSIGDB_FULL_CACHE: dict | None = None

# Disease → most relevant Hallmark programs for S-LDSC γ estimation
DISEASE_HALLMARK_PROGRAMS: dict[str, list[str]] = {
    "CAD":  [
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "HALLMARK_CHOLESTEROL_HOMEOSTASIS",
        "HALLMARK_COAGULATION",
        "HALLMARK_COMPLEMENT",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
        "HALLMARK_HYPOXIA",
    ],
    "RA": [
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "HALLMARK_IL2_STAT5_SIGNALING",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_ALLOGRAFT_REJECTION",
    ],
    "SLE": [
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "HALLMARK_COMPLEMENT",
        "HALLMARK_INFLAMMATORY_RESPONSE",
    ],
    "IBD": [
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "HALLMARK_HYPOXIA",
        "HALLMARK_IL2_STAT5_SIGNALING",
    ],
    "AD": [
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_COMPLEMENT",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
        "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
        "HALLMARK_APOPTOSIS",
    ],
    "T2D": [
        "HALLMARK_CHOLESTEROL_HOMEOSTASIS",
        "HALLMARK_FATTY_ACID_METABOLISM",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
        "HALLMARK_GLYCOLYSIS",
        "HALLMARK_MTORC1_SIGNALING",
        "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
        "HALLMARK_PANCREAS_BETA_CELLS",
    ],
    # AMD: RPE-centric programs. Complement intentionally excluded — complement proteins
    # (CFH, C3, CFB) are liver-secreted, absent from Perturb-seq, and cannot be scored
    # by β×γ. They are handled via OT direct-γ (genetic-only graph). All programs here
    # are expressed in RPE/Mueller glia and scorable from h5ad.
    "AMD": [
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",   # mitochondrial OXPHOS — GA/RPE metabolism
        "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY",  # RPE oxidative stress
        "HALLMARK_FATTY_ACID_METABOLISM",       # RPE lipid efflux (ABCA1, ABCA4, LIPC, APOE)
        "HALLMARK_CHOLESTEROL_HOMEOSTASIS",     # Bruch's membrane lipid accumulation
        "HALLMARK_ANGIOGENESIS",                # VEGF/neovascularization (wet AMD)
        "HALLMARK_HYPOXIA",                     # RPE hypoxia → VEGF upregulation
        "HALLMARK_APOPTOSIS",                   # RPE/photoreceptor cell death (GA)
        "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",   # drusen, RPE ER stress
        "HALLMARK_MTORC1_SIGNALING",            # RPE autophagy/mitophagy
        "HALLMARK_TGF_BETA_SIGNALING",          # subretinal fibrosis, CNV remodeling
        "HALLMARK_WNT_BETA_CATENIN_SIGNALING",  # RPE epithelial integrity
        "HALLMARK_INFLAMMATORY_RESPONSE",       # para-inflammation, microglial activation
    ],
}


# ---------------------------------------------------------------------------
# MSigDB Hallmark — live REST API
# ---------------------------------------------------------------------------

def get_msigdb_hallmark_programs(
    disease: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """
    Fetch MSigDB Hallmark gene sets from the Broad Institute REST API.

    Returns a dict of program definitions usable as input to S-LDSC enrichment
    analysis and as the universal program vocabulary for γ_{P→trait} estimation.

    Args:
        disease:  If provided, returns only the Hallmark sets most relevant to
                  that disease (from DISEASE_HALLMARK_PROGRAMS).  If None,
                  returns all 50 Hallmark sets.
        timeout:  HTTP timeout in seconds.

    Returns:
        {
            "programs": [{"program_id": str, "hallmark_name": str,
                          "gene_set": [str], "n_genes": int}],
            "n_programs": int,
            "source": "MSigDB_Hallmark_v2024.1",
            "disease_filtered": bool,
        }
    """
    global _MSIGDB_FULL_CACHE
    relevant = set(DISEASE_HALLMARK_PROGRAMS.get(disease or "", []))

    if _MSIGDB_FULL_CACHE is None:
        try:
            resp = httpx.get(_MSIGDB_GENE_SET_URL, timeout=timeout)
            resp.raise_for_status()
            _MSIGDB_FULL_CACHE = resp.json()
        except Exception as exc:
            return _hallmark_fallback(disease, error=str(exc))

    raw = _MSIGDB_FULL_CACHE
    programs = []
    for hallmark_name, info in raw.items():
        if relevant and hallmark_name not in relevant:
            continue
        gene_set = info.get("geneSymbols", info.get("genes", []))
        programs.append({
            "program_id":    HALLMARK_TO_PROGRAM.get(hallmark_name, hallmark_name.lower()),
            "hallmark_name": hallmark_name,
            "gene_set":      gene_set,
            "n_genes":       len(gene_set),
            "cell_type":     "universal",
            "source":        "MSigDB_Hallmark",
        })

    if not programs:
        return _hallmark_fallback(disease, error="No programs matched after filtering")

    return {
        "programs":         programs,
        "n_programs":       len(programs),
        "source":           "MSigDB_Hallmark_v2024.1",
        "disease_filtered": bool(relevant),
        "note": (
            "Hallmark gene sets are universal and not cell-type-specific. "
            "Suitable for S-LDSC γ estimation across all diseases. "
            "For β estimation, prefer cell-type-matched Perturb-seq (Tier 1) or eQTL-MR (Tier 2)."
        ),
    }


def _hallmark_fallback(disease: str | None, error: str = "") -> dict:
    """
    Return curated Hallmark program stubs when the MSigDB API is unreachable.
    These are the most disease-relevant programs with gene counts from published papers.
    """
    fallback_programs = [
        {"program_id": "inflammatory_NF-kB",    "hallmark_name": "HALLMARK_INFLAMMATORY_RESPONSE",    "gene_set": [], "n_genes": 200, "cell_type": "universal", "source": "MSigDB_Hallmark_stub"},
        {"program_id": "IL-6_signaling",         "hallmark_name": "HALLMARK_IL6_JAK_STAT3_SIGNALING",  "gene_set": [], "n_genes": 87,  "cell_type": "universal", "source": "MSigDB_Hallmark_stub"},
        {"program_id": "lipid_metabolism",       "hallmark_name": "HALLMARK_CHOLESTEROL_HOMEOSTASIS",  "gene_set": [], "n_genes": 74,  "cell_type": "universal", "source": "MSigDB_Hallmark_stub"},
        {"program_id": "MHC_class_II_presentation","hallmark_name": "HALLMARK_INTERFERON_GAMMA_RESPONSE","gene_set":[],"n_genes": 200, "cell_type": "universal", "source": "MSigDB_Hallmark_stub"},
        {"program_id": "interferon_alpha",        "hallmark_name": "HALLMARK_INTERFERON_ALPHA_RESPONSE","gene_set": [], "n_genes": 97,  "cell_type": "universal", "source": "MSigDB_Hallmark_stub"},
        {"program_id": "DNA_methylation_maintenance","hallmark_name":"HALLMARK_DNA_REPAIR",             "gene_set": [], "n_genes": 150, "cell_type": "universal", "source": "MSigDB_Hallmark_stub"},
        {"program_id": "G2M_phase_program",       "hallmark_name": "HALLMARK_G2M_CHECKPOINT",          "gene_set": [], "n_genes": 200, "cell_type": "universal", "source": "MSigDB_Hallmark_stub"},
        {"program_id": "oxidative_phosphorylation","hallmark_name":"HALLMARK_OXIDATIVE_PHOSPHORYLATION","gene_set": [], "n_genes": 200, "cell_type": "universal", "source": "MSigDB_Hallmark_stub"},
        {"program_id": "complement_program",      "hallmark_name": "HALLMARK_COMPLEMENT",               "gene_set": [], "n_genes": 200, "cell_type": "universal", "source": "MSigDB_Hallmark_stub"},
        {"program_id": "coagulation_program",     "hallmark_name": "HALLMARK_COAGULATION",              "gene_set": [], "n_genes": 138, "cell_type": "universal", "source": "MSigDB_Hallmark_stub"},
    ]

    relevant = set(DISEASE_HALLMARK_PROGRAMS.get(disease or "", []))
    if relevant:
        fallback_programs = [p for p in fallback_programs if p["hallmark_name"] in relevant]

    return {
        "programs":         fallback_programs,
        "n_programs":       len(fallback_programs),
        "source":           "MSigDB_Hallmark_stub",
        "disease_filtered": bool(relevant),
        "error":            error,
        "note":             "MSigDB API unreachable — using stub program list (gene_set lists are empty).",
    }


# ---------------------------------------------------------------------------
# Disease-aware program routing
# ---------------------------------------------------------------------------

def get_programs_for_disease(
    disease: str,
    cnmf_output_dir: str = "./data/cnmf_programs",
) -> dict:
    """
    Return the best available programs for a given disease.

    Priority:
      1. Pre-computed cNMF programs from cell-type-matched Perturb-seq h5ad
      2. MSigDB Hallmark gene sets (disease-filtered)
      3. Hardcoded provisional programs from burden_perturb_server

    Args:
        disease:          Short disease name (CAD, RA, IBD, AD, etc.)
        cnmf_output_dir:  Directory where cNMF outputs are saved.

    Returns:
        Program definition dict compatible with build_beta_matrix() and
        build_gamma_matrix().
    """
    from graph.schema import DISEASE_CELL_TYPE_MAP

    ctx = DISEASE_CELL_TYPE_MAP.get(disease, {})
    cell_type = ctx.get("perturb_seq_source", "unknown")

    # 1 — cNMF from h5ad (if computed)
    cnmf_path = Path(cnmf_output_dir) / f"{cell_type}_programs.json"
    if cnmf_path.exists():
        data = json.loads(cnmf_path.read_text())
        programs = data.get("programs", [])
        if programs:
            return {
                "programs":    programs,
                "n_programs":  len(programs),
                "source":      f"cNMF_{cell_type}",
                "cell_type":   cell_type,
                "disease":     disease,
            }

    # 2 — MSigDB Hallmark (universal, immediately available)
    hallmark = get_msigdb_hallmark_programs(disease=disease)
    if hallmark.get("n_programs", 0) > 0 and "stub" not in hallmark.get("source", ""):
        hallmark["cell_type"] = cell_type
        hallmark["disease"] = disease
        return hallmark

    # 3 — Hardcoded provisional fallback
    return load_cnmf_programs(cnmf_output_dir)


# ---------------------------------------------------------------------------
# cNMF pipeline stub (requires h5ad download)
# ---------------------------------------------------------------------------

def run_cnmf_pipeline(
    h5ad_path: str,
    k_programs: int = 12,
    n_iter: int = 50,
    output_dir: str = "./data/cnmf_programs",
    cell_type: str = "RPE",
    n_top_genes: int = 2000,
    n_top_marker_genes: int = 15,
    random_state: int = 42,
) -> dict:
    """
    Run NMF program extraction on a CELLxGENE h5ad.

    Uses sklearn NMF on highly variable genes (HVGs) to extract k_programs
    latent gene expression programs.  Compatible with AMD RPE h5ad from
    CELLxGENE (30k cells × 20k genes; disease + normal RPE cells).

    Args:
        h5ad_path:         Path to .h5ad file (anndata format).
        k_programs:        Number of NMF components (= programs).  Default 12
                           matches the 12 AMD Hallmark programs.
        n_iter:            NMF solver iterations (50 is fast; 200 for production).
        output_dir:        Directory to write program JSON cache.
        cell_type:         Label used in output program names.
        n_top_genes:       Number of HVGs selected before NMF.
        n_top_marker_genes: Top genes per program to report as markers.
        random_state:      Reproducibility seed.

    Returns:
        dict with programs, n_programs, source, program_gene_sets, and metadata.
        Compatible with get_programs_for_disease() output schema.
    """
    import os
    if not os.path.exists(h5ad_path):
        return {
            "status":    "error",
            "h5ad_path": h5ad_path,
            "message":   f"File not found: {h5ad_path}",
            "note":      (
                f"Download the {cell_type} h5ad from CELLxGENE. "
                "See data/cellxgene/{disease}/ for expected paths."
            ),
        }

    try:
        import anndata
        import numpy as np
        from sklearn.decomposition import NMF
        from scipy.sparse import issparse
    except ImportError as exc:
        return {
            "status": "error",
            "message": f"Missing dependency: {exc}. Run: pip install anndata scikit-learn scipy",
        }

    # --- Load h5ad --------------------------------------------------------
    adata = anndata.read_h5ad(h5ad_path)

    # --- Erythrocyte contamination filter ---------------------------------
    # RPE h5ads from CELLxGENE contain residual erythrocytes whose extreme
    # hemoglobin expression dominates HVG selection and produces uninformative
    # NMF programs (HBB, HBA2, ALAS2, AHSP, etc.).  These genes are removed
    # before HVG selection so NMF captures RPE biology, not red-cell noise.
    _ERYTHROCYTE_GENES = frozenset({
        # Hemoglobin subunits
        "HBB", "HBA1", "HBA2", "HBD", "HBE1", "HBG1", "HBG2",
        "HBZ", "HBM", "HBQ1",
        # Heme biosynthesis (erythrocyte-specific isoform)
        "ALAS2",
        # Alpha-hemoglobin stabilising protein
        "AHSP",
        # Erythrocyte membrane / cytoskeleton
        "SPTA1", "SPTB", "ANK1", "KCNN4",
        # Glycophorins
        "GYPA", "GYPB", "GYPC", "GYPE",
        # Other strong erythroid markers
        "CA1", "CA2", "BLVRB", "DMTN", "EPB42",
    })

    # Resolve gene symbols from var (use feature_name if available)
    if "feature_name" in adata.var.columns:
        all_gene_symbols = np.array(adata.var["feature_name"].tolist())
    else:
        all_gene_symbols = np.array(adata.var_names)

    erythrocyte_mask = np.array([g in _ERYTHROCYTE_GENES for g in all_gene_symbols])
    n_filtered = erythrocyte_mask.sum()

    # --- HVG selection via coefficient of variation -----------------------
    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    # Zero out erythrocyte columns so they never rank in top HVGs
    if n_filtered > 0:
        X[:, erythrocyte_mask] = 0.0

    gene_means = X.mean(axis=0)
    gene_stds  = X.std(axis=0)

    # Coefficient of variation (CV); avoid divide-by-zero for zero-mean genes
    gene_cv = np.where(gene_means > 0, gene_stds / gene_means, 0.0)

    # Keep top n_top_genes by CV
    n_hvg = min(n_top_genes, X.shape[1])
    hvg_idx = np.argsort(gene_cv)[-n_hvg:]

    X_hvg = X[:, hvg_idx]
    gene_names = all_gene_symbols[hvg_idx]

    # Log1p-normalise counts (total-count normalise then log1p)
    row_sums = X_hvg.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    X_norm = np.log1p(X_hvg / row_sums * 1e4)

    # --- NMF ---------------------------------------------------------------
    model = NMF(
        n_components=k_programs,
        init="nndsvda",
        max_iter=n_iter,
        random_state=random_state,
        l1_ratio=0.0,
        alpha_W=0.0,
        alpha_H="same",
    )
    W = model.fit_transform(X_norm)   # (n_cells, k_programs)
    H = model.components_             # (k_programs, n_hvg_genes)

    # --- Extract top marker genes per program ------------------------------
    # Programs stored as dicts with gene_set field so _get_gamma_estimates
    # can use them directly without a separate get_program_gene_loadings call.
    programs_as_dicts: list[dict] = []

    for k in range(k_programs):
        loadings = H[k]
        top_idx  = np.argsort(loadings)[-n_top_marker_genes:][::-1]
        top_genes = [str(gene_names[i]) for i in top_idx]
        prog_name = f"{cell_type}_NMF_P{k+1:02d}_{top_genes[0]}"
        programs_as_dicts.append({
            "program_id": prog_name,
            "gene_set":   top_genes,
            "n_genes":    len(top_genes),
            "cell_type":  cell_type,
            "source":     f"cNMF_{cell_type}",
        })

    # --- Persist to JSON cache -------------------------------------------
    # Filename matches what get_programs_for_disease looks for.
    out_path = Path(output_dir) / f"{cell_type}_programs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "cell_type":   cell_type,
        "h5ad_path":   h5ad_path,
        "k_programs":  k_programs,
        "n_cells":     int(adata.n_obs),
        "n_hvg":       n_hvg,
        "programs":    programs_as_dicts,
        "n_programs":  len(programs_as_dicts),
    }
    try:
        with open(out_path, "w") as fh:
            json.dump(cache, fh, indent=2)
    except Exception:
        pass  # non-fatal

    return {
        "status":      "success",
        "programs":    programs_as_dicts,
        "n_programs":  len(programs_as_dicts),
        "source":              f"sklearn_NMF_k{k_programs}_{cell_type}",
        "cell_type":           cell_type,
        "h5ad_path":           h5ad_path,
        "n_cells":             int(adata.n_obs),
        "n_hvg":               n_hvg,
        "n_erythrocyte_genes_filtered": int(n_filtered),
        "note":                (
            f"sklearn NMF k={k_programs}, {n_hvg} HVGs, {n_iter} iter. "
            f"{n_filtered} erythrocyte contamination genes zeroed before HVG selection. "
            "Top marker genes per program extracted by H-matrix loading magnitude."
        ),
    }


def load_cnmf_programs(output_dir: str = "./data/cnmf_programs") -> dict:
    """
    Load pre-computed cNMF programs, falling back to hardcoded provisional list.
    """
    from mcp_servers.burden_perturb_server import get_cnmf_program_info
    programs_info = get_cnmf_program_info()
    return {
        "programs":   programs_info["programs"],
        "n_programs": programs_info["n_programs"],
        "source":     "hardcoded_provisional",
        "note":       "Provisional programs. Use get_programs_for_disease() for disease-matched source.",
    }
