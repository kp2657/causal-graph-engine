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
    relevant = set(DISEASE_HALLMARK_PROGRAMS.get(disease or "", []))

    try:
        resp = httpx.get(_MSIGDB_GENE_SET_URL, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        return _hallmark_fallback(disease, error=str(exc))

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
    k_programs: int = 20,
    n_iter: int = 200,
    output_dir: str = "./data/cnmf_programs",
    cell_type: str = "K562",
) -> dict:
    """
    Run cNMF program extraction on a Perturb-seq pseudo-bulk h5ad.

    STUB — requires the h5ad file and `pip install anndata cnmf`.

    Recommended inputs:
      K562 (blood/CAD):   357 MB — https://ndownloader.figshare.com/files/35773217
      K562 essential:      76 MB — https://ndownloader.figshare.com/files/35780870
      Papalexi PBMCs:     ~200 MB — https://zenodo.org/record/7041690 (scPerturb)
      Ursu iPSC neurons:  ~500 MB — https://zenodo.org/record/7041690 (scPerturb)
    """
    import os
    if not os.path.exists(h5ad_path):
        return {
            "status":   "error",
            "h5ad_path": h5ad_path,
            "message":  f"File not found: {h5ad_path}",
            "note":     (
                f"Download the {cell_type} pseudo-bulk h5ad from Figshare or scPerturb. "
                "See PERTURB_SEQ_SOURCES in graph/schema.py for URLs."
            ),
        }

    # When implemented:
    # import anndata, cnmf
    # adata = anndata.read_h5ad(h5ad_path)
    # model = cnmf.cNMF(output_dir=output_dir, name=cell_type)
    # model.prepare(counts_fn=h5ad_path, components=k_programs, n_iter=n_iter, ...)
    # model.factorize(worker_i=0, total_workers=1)
    # model.combine(components=k_programs, skip_missing_files=True)
    # usage, spectra, top_genes = model.load_results(K=k_programs, density_threshold=2.00)

    return {
        "status":    "stub",
        "h5ad_path": h5ad_path,
        "cell_type": cell_type,
        "note":      "STUB — implement after: pip install anndata cnmf",
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
