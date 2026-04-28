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

# Set of all internal program IDs derived from MSigDB Hallmark gene sets.
# Used to exclude Hallmark-derived programs from the OTA γ sum — only
# cNMF and state-space transition programs contribute to ota_gamma.
HALLMARK_PROGRAM_IDS: frozenset[str] = frozenset(HALLMARK_TO_PROGRAM.values())

# In-process cache for the MSigDB gene set fetch (HTTP, avoid repeat calls per run)
_MSIGDB_FULL_CACHE: dict | None = None



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
        disease:  Ignored (kept for API compatibility). Always returns all 50
                  Hallmark sets. Use for GPS annotation and GSEA, not program
                  selection (which is now data-driven via run_cnmf_pipeline).
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

    if _MSIGDB_FULL_CACHE is None:
        _disk_cache = Path(__file__).parent.parent / "data" / "msigdb_hallmark_2024.json"
        if _disk_cache.exists():
            try:
                import json as _json
                _MSIGDB_FULL_CACHE = _json.loads(_disk_cache.read_text())
            except Exception:
                pass
        if _MSIGDB_FULL_CACHE is None:
            try:
                resp = httpx.get(_MSIGDB_GENE_SET_URL, timeout=timeout)
                resp.raise_for_status()
                _MSIGDB_FULL_CACHE = resp.json()
                _disk_cache.parent.mkdir(parents=True, exist_ok=True)
                import json as _json
                _disk_cache.write_text(_json.dumps(_MSIGDB_FULL_CACHE))
            except Exception as exc:
                return _hallmark_fallback(disease, error=str(exc))

    raw = _MSIGDB_FULL_CACHE
    programs = []
    for hallmark_name, info in raw.items():
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
        return _hallmark_fallback(disease, error="No programs in MSigDB response")

    return {
        "programs":         programs,
        "n_programs":       len(programs),
        "source":           "MSigDB_Hallmark_v2024.1",
        "disease_filtered": False,
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

    return {
        "programs":         fallback_programs,
        "n_programs":       len(fallback_programs),
        "source":           "MSigDB_Hallmark_stub",
        "disease_filtered": False,
        "error":            error,
        "note":             "MSigDB API unreachable — using stub program list (gene_set lists are empty).",
    }


# ---------------------------------------------------------------------------
# Disease-aware program routing
# ---------------------------------------------------------------------------

def _resolve_ensembl_gene_sets(programs: list[dict]) -> list[dict]:
    """
    Resolve Ensembl IDs in program gene_set lists to HGNC symbols in-place.

    Programs from the disease-keyed NMF cache (AMD_programs.json, CAD_programs.json)
    store gene sets as Ensembl IDs because the h5ad var_names are Ensembl. This
    converts them to symbols so downstream beta/gamma matching (which uses symbols)
    works correctly. Genes that cannot be resolved are dropped from the set.
    """
    try:
        from pipelines.static_lookups import get_lookups
        lk = get_lookups()
    except Exception:
        return programs

    resolved: list[dict] = []
    for p in programs:
        gs = p.get("gene_set", [])
        if not gs or not gs[0].startswith("ENSG"):
            resolved.append(p)
            continue
        def _resolve_list(ids: list) -> list:
            out = []
            for g in ids:
                sym = lk.get_symbol_from_ensembl(g) if str(g).startswith("ENSG") else g
                if sym:
                    out.append(sym)
            return out

        syms = _resolve_list(gs)
        top_genes = _resolve_list(p.get("top_genes", []))
        gene_loadings_raw = p.get("gene_loadings", {})
        gene_loadings: dict = {}
        for ensg, val in gene_loadings_raw.items():
            sym = lk.get_symbol_from_ensembl(ensg) if str(ensg).startswith("ENSG") else ensg
            if sym:
                gene_loadings[sym] = val
        resolved.append({**p, "gene_set": syms, "top_genes": top_genes, "gene_loadings": gene_loadings})
    return resolved


def get_programs_for_disease(
    disease: str,
    cnmf_output_dir: str = "./data/cnmf_programs",
) -> dict:
    """
    Return NMF programs for a given disease from the on-disk cache.

    Lookup order:
      1. {disease}_programs.json  — disease-specific h5ad NMF (preferred: contains
         disease-relevant cell types with GWAS gene overlap for γ estimation)
      2. {cell_type}_programs.json — Perturb-seq cell-type NMF (used when no
         disease-specific cache exists)

    Gene sets stored as Ensembl IDs (disease h5ad caches) are resolved to HGNC
    symbols automatically so downstream beta/gamma matching works correctly.

    Args:
        disease:          Short disease name (CAD, AMD, RA, IBD, etc.)
        cnmf_output_dir:  Directory where cNMF outputs are cached.

    Returns:
        Program definition dict with 'programs', 'n_programs', 'source',
        'cell_type', 'disease'. Compatible with build_beta_matrix() and
        build_gamma_matrix(). Returns empty programs list (not an error) when
        no cache exists — caller should invoke run_cnmf_pipeline() first.
    """
    from graph.schema import DISEASE_CELL_TYPE_MAP

    ctx = DISEASE_CELL_TYPE_MAP.get(disease, {})
    cell_type = ctx.get("perturb_seq_source", "unknown")
    out_dir = Path(cnmf_output_dir)
    _frozen = Path(__file__).resolve().parent.parent / "frozen"

    # 1. Disease-keyed cache — frozen repo copy takes priority for exact reproduction,
    #    then locally computed NMF from disease-specific h5ad (cellxgene).
    frozen_path  = _frozen / f"{disease}_programs.json"
    disease_path = out_dir / f"{disease}_programs.json"
    if frozen_path.exists() and not disease_path.exists():
        disease_path = frozen_path
    if disease_path.exists():
        data = json.loads(disease_path.read_text())
        programs = data.get("programs", [])
        if programs:
            programs = _resolve_ensembl_gene_sets(programs)
            programs = [p for p in programs if p.get("gene_set")]
            if programs:
                return {
                    "programs":   programs,
                    "n_programs": len(programs),
                    "source":     f"cNMF_{disease}_h5ad",
                    "cell_type":  cell_type,
                    "disease":    disease,
                }

    # 2. Cell-type-keyed cache — NMF from Perturb-seq cell line h5ad.
    cell_type_path = out_dir / f"{cell_type}_programs.json"
    if cell_type_path.exists():
        data = json.loads(cell_type_path.read_text())
        programs = data.get("programs", [])
        if programs:
            return {
                "programs":   programs,
                "n_programs": len(programs),
                "source":     f"cNMF_{cell_type}",
                "cell_type":  cell_type,
                "disease":    disease,
            }

    return {
        "programs":   [],
        "n_programs": 0,
        "source":     "no_cache",
        "cell_type":  cell_type,
        "disease":    disease,
        "note":       (
            f"No NMF program cache found at {disease_path} or {cell_type_path}. "
            "Run run_cnmf_pipeline() to generate programs."
        ),
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Empirical selection helpers
# ---------------------------------------------------------------------------

def _select_k_empirically(
    X_norm: "np.ndarray",
    k_range: tuple[int, int] = (4, 20),
    n_iter: int = 50,
    random_state: int = 42,
) -> int:
    """
    Select NMF k by reconstruction error elbow (second-derivative method).

    Fits NMF for each k in k_range, records the Frobenius reconstruction error,
    then finds the k where the rate of error improvement slows most — the elbow.
    The second derivative of the error curve peaks at this elbow.
    """
    import numpy as np
    from sklearn.decomposition import NMF

    k_min, k_max = k_range
    k_values = list(range(k_min, min(k_max + 1, X_norm.shape[0], X_norm.shape[1] + 1)))
    if len(k_values) < 3:
        return k_values[0] if k_values else k_min

    errors: list[float] = []
    for k in k_values:
        model = NMF(n_components=k, init="nndsvda", max_iter=n_iter,
                    random_state=random_state, l1_ratio=0.0)
        model.fit(X_norm)
        errors.append(float(model.reconstruction_err_))

    err = np.array(errors)
    # Second differences: positive where improvement rate is decelerating.
    # Elbow = argmax(second derivative) + 1 (offset by 2 shorter array).
    diff2 = np.diff(np.diff(err))
    knee_idx = int(np.argmax(diff2)) + 1
    knee_idx = max(0, min(knee_idx, len(k_values) - 1))
    return int(k_values[knee_idx])


def _select_genes_by_loading(
    loadings: "np.ndarray",
    min_genes: int = 10,
    max_genes: int = 100,
) -> "np.ndarray":
    """
    Select top genes for one NMF component by loading score elbow (kneedle method).

    Returns indices of selected genes sorted by loading descending. The number
    of genes is the knee of the loading curve, bounded to [min_genes, max_genes].
    """
    import numpy as np

    sorted_idx = np.argsort(loadings)[::-1]
    sorted_load = loadings[sorted_idx]

    n = len(sorted_load)
    if n <= min_genes:
        return sorted_idx[:n]

    cap = min(n, max_genes)
    x = np.arange(cap, dtype=float) / max(cap - 1, 1)
    y_max, y_min = float(sorted_load[0]), float(sorted_load[cap - 1])
    if y_max <= y_min:
        return sorted_idx[:min_genes]

    y = (sorted_load[:cap] - y_min) / (y_max - y_min)
    # Kneedle for decreasing curve: distance from diagonal y = 1 - x
    dist = y + x - 1.0
    knee = int(np.argmax(dist)) + 1   # +1 to include the knee gene
    knee = max(min_genes, min(max_genes, knee))
    return sorted_idx[:knee]


# ---------------------------------------------------------------------------
# cNMF pipeline stub (requires h5ad download)
# ---------------------------------------------------------------------------

def run_cnmf_pipeline(
    h5ad_path: str,
    k_programs: int | None = None,
    k_range: tuple[int, int] = (4, 20),
    n_iter: int = 50,
    output_dir: str = "./data/cnmf_programs",
    cell_type: str = "RPE",
    n_top_genes: int = 2000,
    min_genes_per_program: int = 10,
    max_genes_per_program: int = 100,
    random_state: int = 42,
    min_disease_fraction: float = 0.20,
) -> dict:
    """
    Run NMF program extraction on a CELLxGENE h5ad.

    Uses sklearn NMF on highly variable genes (HVGs) to extract latent gene
    expression programs. Both the number of programs (k) and the number of
    genes per program are selected empirically from the data:

    - k: chosen by reconstruction error elbow (second derivative) over k_range.
      Pass k_programs=<int> to override with a fixed value.
    - genes per program: chosen by loading score elbow (kneedle) per component,
      bounded to [min_genes_per_program, max_genes_per_program].

    Two pre-processing steps improve program quality:

    1. Cell-type contamination gene exclusion: erythrocyte and photoreceptor
       identity genes are zeroed before HVG selection so NMF captures disease
       pathway biology rather than cell-type contamination signal.

    2. Disease-cell oversampling: when disease cells are underrepresented
       (e.g., AMD RPE h5ad has <1% AMD cells), disease rows are repeated up to
       min_disease_fraction (default 20%) so NMF extracts disease-state programs
       rather than exclusively normal-state biology.

    Args:
        h5ad_path:               Path to .h5ad file (anndata format).
        k_programs:              Number of NMF components; None = auto-select by elbow.
        k_range:                 (k_min, k_max) search range for empirical k selection.
        n_iter:                  NMF solver iterations (50 fast; 200 production).
        output_dir:              Directory to write program JSON cache.
        cell_type:               Label used in output filenames and program IDs.
        n_top_genes:             Number of HVGs selected before NMF.
        min_genes_per_program:   Minimum genes kept per program (loading elbow lower bound).
        max_genes_per_program:   Maximum genes kept per program (loading elbow upper bound).
        random_state:            Reproducibility seed.
        min_disease_fraction:    If disease cells < this fraction, subsample normal cells
                                 to balance at this ratio.

    Returns:
        dict with programs, n_programs, source, and metadata.
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

    # --- Contamination gene filters ---------------------------------------
    # Erythrocyte genes: dominate HVGs in RPE h5ads (HBB, ALAS2, etc.)
    _ERYTHROCYTE_GENES = frozenset({
        "HBB", "HBA1", "HBA2", "HBD", "HBE1", "HBG1", "HBG2",
        "HBZ", "HBM", "HBQ1", "ALAS2", "AHSP",
        "SPTA1", "SPTB", "ANK1", "KCNN4",
        "GYPA", "GYPB", "GYPC", "GYPE",
        "CA1", "CA2", "BLVRB", "DMTN", "EPB42",
    })

    # Photoreceptor identity genes: in AMD RPE h5ads these reflect RPE
    # phagocytosis of outer segments and photoreceptor death — a consequence
    # of disease, not an RPE-autonomous pathway. Excluding them lets NMF
    # surface the actual RPE disease programs (oxidative stress, lipid efflux,
    # UPR, VEGF) that are directly scorable by Replogle RPE1 Perturb-seq β.
    _PHOTORECEPTOR_GENES = frozenset({
        # Opsins
        "RHO", "OPN1LW", "OPN1MW", "OPN1SW", "OPN1MW2", "OPN1MW3",
        # Phototransduction cascade
        "PDE6A", "PDE6B", "PDE6C", "PDE6G", "PDE6H",
        "CNGA1", "CNGB1", "CNGA3", "CNGB3",
        "GUCA1A", "GUCA1B", "SAG",
        "GNGT1", "GNGT2", "GNB1",
        # Photoreceptor structural / specific
        "ROM1", "PRPH2", "RCVRN", "ARR3",
        # Olfactory receptors expressed in retina (non-informative for AMD)
        "OR51E1", "OR51E2",
    })

    _EXCLUDE_GENES = _ERYTHROCYTE_GENES | _PHOTORECEPTOR_GENES

    # Resolve gene symbols from var — try common symbol columns before falling back to var_names
    _sym_col = next(
        (c for c in ("feature_name", "gene_name", "gene_symbol", "symbol") if c in adata.var.columns),
        None,
    )
    if _sym_col:
        all_gene_symbols = np.array(adata.var[_sym_col].tolist())
    else:
        all_gene_symbols = np.array(adata.var_names)

    exclude_mask = np.array([g in _EXCLUDE_GENES for g in all_gene_symbols])
    n_erythrocyte_filtered = sum(1 for g in all_gene_symbols if g in _ERYTHROCYTE_GENES)
    n_photoreceptor_filtered = sum(1 for g in all_gene_symbols if g in _PHOTORECEPTOR_GENES)

    # --- Disease-balanced cell selection ----------------------------------
    # When disease cells are severely underrepresented (e.g., AMD has 162
    # disease cells in 30,000 RPE cells = 0.5%), NMF extracts only normal
    # RPE biology. Strategy: keep ALL disease cells + a stratified subsample
    # of normal cells so disease reaches min_disease_fraction (default 20%).
    # This is stratified subsampling, not oversampling — a balanced dataset
    # rather than inflated copies of the same few disease cells.
    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    # NMF requires non-negative input. Pseudo-bulk / log-FC h5ads (Replogle)
    # contain negative LFC values and ±inf where control expression is 0.
    # Clip: negatives → 0 (keep upregulation signal only), inf → 0.
    np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.clip(X, 0.0, None, out=X)

    n_disease_oversampled = 0
    n_normal_downsampled = 0
    if "disease" in adata.obs.columns and min_disease_fraction > 0:
        is_normal = adata.obs["disease"].str.lower().str.contains("normal").values
        disease_idx = np.where(~is_normal)[0]
        normal_idx  = np.where(is_normal)[0]
        n_dis = len(disease_idx)
        n_norm = len(normal_idx)
        current_frac = n_dis / max(n_dis + n_norm, 1)
        if n_dis > 0 and current_frac < min_disease_fraction:
            # Keep all disease cells + subsample normal cells
            target_n_norm = int(n_dis * (1 - min_disease_fraction) / min_disease_fraction)
            target_n_norm = max(target_n_norm, n_dis)  # at least 1:1 ratio
            rng = np.random.default_rng(random_state)
            if n_norm > target_n_norm:
                sampled_normal_idx = rng.choice(normal_idx, target_n_norm, replace=False)
                n_normal_downsampled = n_norm - target_n_norm
            else:
                sampled_normal_idx = normal_idx
            all_idx = np.concatenate([disease_idx, sampled_normal_idx])
            rng.shuffle(all_idx)
            X = X[all_idx]
            n_disease_oversampled = n_dis  # report actual disease count used

    # --- HVG selection via coefficient of variation -----------------------
    # Zero out contamination genes so they never rank in top HVGs
    if exclude_mask.any():
        X[:, exclude_mask] = 0.0

    gene_means = X.mean(axis=0)
    gene_stds  = X.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gene_cv = np.where(gene_means > 0, gene_stds / gene_means, 0.0)

    n_hvg = min(n_top_genes, X.shape[1])
    hvg_idx = np.argsort(gene_cv)[-n_hvg:]

    X_hvg = X[:, hvg_idx]
    gene_names = all_gene_symbols[hvg_idx]

    # Log1p-normalise counts (total-count normalise then log1p)
    row_sums = X_hvg.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    X_norm = np.log1p(X_hvg / row_sums * 1e4)
    np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    # --- Empirical k selection (if not fixed) ---------------------------------
    k_auto_selected = k_programs is None
    if k_auto_selected:
        k_programs = _select_k_empirically(
            X_norm, k_range=k_range, n_iter=n_iter, random_state=random_state)

    # --- NMF (final fit with chosen k) ------------------------------------
    model = NMF(
        n_components=k_programs,
        init="nndsvda",
        max_iter=n_iter,
        random_state=random_state,
        l1_ratio=0.0,
        alpha_W=0.0,
        alpha_H="same",
    )
    model.fit_transform(X_norm)        # W unused; H encodes program gene weights
    H = model.components_              # (k_programs, n_hvg_genes)

    # --- Extract genes per program by loading elbow -----------------------
    programs_as_dicts: list[dict] = []
    for k in range(k_programs):
        loadings = H[k]
        top_idx = _select_genes_by_loading(
            loadings,
            min_genes=min_genes_per_program,
            max_genes=max_genes_per_program,
        )
        top_genes = [str(gene_names[i]) for i in top_idx]
        # Normalise loadings to [0,1] so cross-program weights are comparable
        raw_loads = loadings[top_idx]
        max_load  = float(raw_loads.max()) if len(raw_loads) > 0 else 1.0
        norm_loads = (raw_loads / max_load).tolist() if max_load > 0 else [1.0] * len(top_genes)
        gene_loadings = {g: round(float(w), 6) for g, w in zip(top_genes, norm_loads)}

        prog_name = f"{cell_type}_NMF_P{k+1:02d}_{top_genes[0]}"
        programs_as_dicts.append({
            "program_id":    prog_name,
            "gene_set":      top_genes,
            "gene_loadings": gene_loadings,   # {gene: normalised_loading [0,1]}
            "n_genes":       len(top_genes),
            "cell_type":     cell_type,
            "source":        f"cNMF_{cell_type}",
        })

    # --- Persist to JSON cache -------------------------------------------
    out_path = Path(output_dir) / f"{cell_type}_programs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "cell_type":              cell_type,
        "h5ad_path":              h5ad_path,
        "k_programs":             k_programs,
        "k_auto_selected":        k_auto_selected,
        "k_range_searched":       list(k_range) if k_auto_selected else None,
        "min_genes_per_program":  min_genes_per_program,
        "max_genes_per_program":  max_genes_per_program,
        "n_cells":                int(adata.n_obs),
        "n_hvg":                  n_hvg,
        "programs":               programs_as_dicts,
        "n_programs":             len(programs_as_dicts),
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
        "source":      f"sklearn_NMF_k{k_programs}_{cell_type}",
        "k_programs":  k_programs,
        "k_auto_selected": k_auto_selected,
        "cell_type":   cell_type,
        "h5ad_path":   h5ad_path,
        "n_cells":     int(adata.n_obs),
        "n_hvg":       n_hvg,
        "n_erythrocyte_genes_filtered":    n_erythrocyte_filtered,
        "n_photoreceptor_genes_filtered":  n_photoreceptor_filtered,
        "n_disease_cells_used":            n_disease_oversampled,
        "n_normal_cells_downsampled":      n_normal_downsampled,
        "note": (
            f"sklearn NMF k={k_programs} ({'auto elbow' if k_auto_selected else 'fixed'}), "
            f"{n_hvg} HVGs, {n_iter} iter. "
            f"genes/program: elbow [{min_genes_per_program}–{max_genes_per_program}]. "
            f"{n_erythrocyte_filtered} erythrocyte + {n_photoreceptor_filtered} photoreceptor "
            f"genes zeroed. {n_disease_oversampled} disease cells kept; "
            f"{n_normal_downsampled} normal cells downsampled to balance at "
            f"min_disease_fraction={min_disease_fraction}."
        ),
    }


def run_scvi_program_extraction(
    h5ad_path: str,
    n_latent: int = 10,
    n_layers: int = 2,
    n_hidden: int = 128,
    max_epochs: int = 200,
    batch_size: int = 128,
    min_genes_per_program: int = 10,
    max_genes_per_program: int = 200,
    output_dir: str = "./data/cnmf_programs",
    cell_type: str = "scvi",
    random_state: int = 42,
    min_disease_fraction: float = 0.20,
) -> dict:
    """
    Extract gene expression programs using scVI latent factors.

    Replaces sklearn NMF with a probabilistic VAE (negative-binomial likelihood)
    that models scRNA-seq count noise explicitly. Gene programs are derived from
    the scVI decoder weight matrix: each latent dimension corresponds to one program,
    and decoder weights give per-gene loadings (analogous to NMF W matrix).

    Key difference vs NMF:
      - NMF: non-negative factorisation of log-normalised expression
      - scVI: VAE on raw counts; decoder weights can be positive or negative
              (absolute value used for gene ranking within each program)

    Output schema is identical to run_cnmf_pipeline() for drop-in compatibility
    with get_programs_for_disease() and the GWAS γ enrichment pipeline.

    Args:
        h5ad_path:             Path to h5ad with raw integer counts in .X.
        n_latent:              Number of latent dimensions (= number of programs).
        n_layers:              scVI encoder/decoder depth.
        n_hidden:              Hidden layer width.
        max_epochs:            Training epochs (200 CPU-feasible; 400+ for GPU).
        batch_size:            Mini-batch size.
        min_genes_per_program: Minimum genes per program (loading elbow lower bound).
        max_genes_per_program: Maximum genes per program (loading elbow upper bound).
        output_dir:            Where to write the program JSON cache.
        cell_type:             Label used in output filenames and program IDs.
        random_state:          Reproducibility seed.
        min_disease_fraction:  Disease cell oversampling fraction (same as NMF pipeline).

    Returns:
        dict compatible with get_programs_for_disease() output schema.
    """
    import os
    if not os.path.exists(h5ad_path):
        return {
            "status":    "error",
            "h5ad_path": h5ad_path,
            "message":   f"File not found: {h5ad_path}",
        }

    try:
        import anndata
        import numpy as np
        from scipy.sparse import issparse
        import scvi as scvi_lib
    except ImportError as exc:
        return {
            "status": "error",
            "message": f"Missing dependency: {exc}. Run: pip install scvi-tools anndata",
        }

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    adata = anndata.read_h5ad(h5ad_path)

    # --- Raw counts: scVI requires integers -----------------------------------
    X = adata.X
    if issparse(X):
        X = X.toarray()
    if X.dtype.kind == "f":
        X = np.round(X).astype(np.int32)
    else:
        X = X.astype(np.int32)

    # --- Disease cell oversampling (same logic as NMF pipeline) ---------------
    n_disease_oversampled = 0
    n_normal_downsampled  = 0
    obs = adata.obs.copy()
    disease_col = next(
        (c for c in ("disease", "disease_state", "condition") if c in obs.columns),
        None,
    )
    if disease_col is not None:
        disease_mask = obs[disease_col] != "normal"
        n_disease = disease_mask.sum()
        n_normal  = (~disease_mask).sum()
        if n_disease / max(n_disease + n_normal, 1) < min_disease_fraction:
            target_normal = int(n_disease / min_disease_fraction) - n_disease
            target_normal = max(0, min(target_normal, n_normal))
            normal_idx = np.where(~disease_mask)[0]
            keep_normal = np.random.RandomState(random_state).choice(
                normal_idx, size=target_normal, replace=False,
            )
            keep_idx = np.concatenate([np.where(disease_mask)[0], keep_normal])
            X   = X[keep_idx]
            obs = obs.iloc[keep_idx].copy()
            n_disease_oversampled = n_disease
            n_normal_downsampled  = target_normal

    var = adata.var.copy()
    adata_counts = anndata.AnnData(X, obs=obs, var=var)

    # --- HVG selection (top 2000) for scVI training ---------------------------
    try:
        import scanpy as sc
        adata_log = adata_counts.copy()
        adata_log.X = adata_log.X.astype(float)
        sc.pp.normalize_total(adata_log, target_sum=1e4)
        sc.pp.log1p(adata_log)
        n_hvg = min(2000, adata_counts.n_vars)
        sc.pp.highly_variable_genes(adata_log, n_top_genes=n_hvg, flavor="seurat_v3")
        hvg_mask = adata_log.var["highly_variable"].values
    except Exception:
        hvg_mask = np.ones(adata_counts.n_vars, dtype=bool)
    gene_names = adata_counts.var_names[hvg_mask].tolist()
    adata_hvg = adata_counts[:, hvg_mask].copy()

    # --- Train scVI -----------------------------------------------------------
    scvi_lib.settings.seed = random_state
    scvi_lib.model.SCVI.setup_anndata(adata_hvg)
    model = scvi_lib.model.SCVI(
        adata_hvg,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
    )
    model.train(
        max_epochs=max_epochs,
        batch_size=batch_size,
        progress_bar_refresh_rate=0,
    )

    # --- Extract decoder weights as gene loadings ----------------------------
    # scVI decoder: linear layer mapping latent z → hidden → gene params.
    # We extract the first linear weight (n_hidden × n_latent) and the final
    # gene output weight (n_genes × n_hidden), computing effective loading as
    # W_eff = |W_gene @ W_hidden| per latent dimension.
    try:
        import torch
        decoder = model.module.decoder
        # Access the mean decoder's first and last linear layers
        px_decoder = decoder.px_decoder      # Sequential: Linear(n_latent→n_hidden)...
        px_scale    = decoder.px_scale_decoder  # Linear(n_hidden→n_genes)

        with torch.no_grad():
            # First linear layer: (n_hidden × n_latent)
            W1 = px_decoder[0].weight.detach().cpu().numpy()  # (n_hidden, n_latent)
            # Output linear layer: (n_genes × n_hidden)
            W_out = px_scale.weight.detach().cpu().numpy()    # (n_genes, n_hidden)
        # Effective loading: (n_genes × n_latent)
        W_eff = np.abs(W_out @ W1)
    except Exception:
        # Fallback: use correlation between latent z and gene expression
        z = model.get_latent_representation()           # (n_cells, n_latent)
        X_norm = adata_hvg.X.astype(float)
        if issparse(X_norm):
            X_norm = X_norm.toarray()
        # Pearson correlation: (n_genes × n_latent)
        z_c    = z    - z.mean(axis=0)
        X_c    = X_norm - X_norm.mean(axis=0)
        z_std  = z_c.std(axis=0) + 1e-8
        X_std  = X_c.std(axis=0) + 1e-8
        W_eff  = np.abs((X_c.T @ z_c) / (X_c.shape[0] * X_std[:, None] * z_std[None, :]))

    # --- Build programs from loadings ----------------------------------------
    programs = []
    gene_name_idx = {g: i for i, g in enumerate(gene_names)}
    for dim_idx in range(n_latent):
        loadings = W_eff[:, dim_idx]
        top_idx = _select_genes_by_loading(
            loadings,
            min_genes=min_genes_per_program,
            max_genes=max_genes_per_program,
        )
        top_genes = [gene_names[i] for i in top_idx]
        if len(top_genes) < min_genes_per_program:
            continue

        prog_id = f"{cell_type}_scVI_Z{dim_idx:02d}"
        programs.append({
            "program_id":    prog_id,
            "cell_type":     cell_type,
            "gene_set":      top_genes,
            "top_genes":     top_genes[:20],
            "source":        "scvi_decoder_weights",
            "n_genes":       len(top_genes),
            "loading_scores": [round(float(loadings[i]), 4) for i in top_idx],
        })

    # --- Cache to JSON --------------------------------------------------------
    import json, pathlib
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"{cell_type}_scvi_programs.json"
    cache_obj = {
        "programs":   programs,
        "n_programs": len(programs),
        "source":     f"scVI_{cell_type}_h5ad",
        "n_latent":   n_latent,
        "n_epochs":   max_epochs,
        "n_hvg":      int(hvg_mask.sum()),
        "n_disease_cells_used":   n_disease_oversampled,
        "n_normal_cells_downsampled": n_normal_downsampled,
        "note": (
            f"scVI VAE n_latent={n_latent}, {n_layers} layers, "
            f"n_hidden={n_hidden}, {max_epochs} epochs. "
            f"{int(hvg_mask.sum())} HVGs. "
            f"Loadings from decoder weight matrix (|W_out @ W1|). "
            f"genes/program: elbow [{min_genes_per_program}–{max_genes_per_program}]."
        ),
    }
    cache_path.write_text(json.dumps(cache_obj, indent=2))

    return cache_obj


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
