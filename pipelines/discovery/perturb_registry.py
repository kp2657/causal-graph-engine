"""
perturb_registry.py — Catalog of Perturb-seq datasets for β_{gene→program} estimation.

Each entry maps a disease/cell-type context to available Perturb-seq datasets,
ordered by biological relevance. This replaces ad-hoc decisions about which
Perturb-seq data to use for each disease.

Usage:
    from pipelines.discovery.perturb_registry import get_perturb_datasets_for_disease
    datasets = get_perturb_datasets_for_disease("AMD")
    best = datasets["recommended"]  # {"geo_id", "url", "cell_type", "n_cells", ...}
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Perturb-seq dataset catalog
# ---------------------------------------------------------------------------
# Each entry: {
#   "id":          unique dataset identifier
#   "geo_id":      GEO accession (GSExxx)
#   "zenodo_id":   Zenodo record ID (if available)
#   "figshare_url": direct download URL (if available)
#   "cell_type":   cell type (matches CELLxGENE / CL ontology label)
#   "n_cells":     approximate number of perturbed cells
#   "n_genes_kd":  number of unique gene knockdowns/knockouts
#   "perturbation_type": "CRISPRi" | "CRISPRa" | "CRISPRko" | "shRNA"
#   "diseases_relevant": list of disease short names where this is useful
#   "pmid":        PubMed ID
#   "note":        usage guidance
# }

PERTURB_SEQ_CATALOG: list[dict] = [
    # -----------------------------------------------------------------------
    # CAD vascular — disease-matched cell types (HCASMC, HAEC)
    # These supersede K562 for all CAD target emulation.
    # -----------------------------------------------------------------------
    {
        "id": "Schnitzler2023_CAD_vascular",
        "geo_id": "GSE210681",
        "cell_type": "HCASMC_HAEC",
        "n_cells": None,  # pre-computed log2FC matrix, not single-cell h5ad
        "n_genes_kd": 332,
        "perturbation_type": "CRISPRi",
        "diseases_relevant": ["CAD"],
        "pmid": None,  # Schnitzler et al., GEO GSE210681
        "local_path_pattern": "./data/perturbseq/schnitzler_cad_vascular/signatures.json.gz",
        "note": (
            "332 CAD GWAS risk genes perturbed in human coronary artery smooth muscle cells "
            "(HCASMC) and endothelial cells — the disease-relevant vascular cell types. "
            "Preprocessed log2FC signatures. Supersedes K562 for CAD vascular biology."
        ),
    },
    {
        "id": "Natsume2023_HAEC",
        "geo_id": None,  # GEO accession not confirmed; see doi:10.1371/journal.pgen.1010680
        "cell_type": "HAEC",
        "n_cells": None,
        "n_genes_kd": 2_285,
        "perturbation_type": "CRISPRko_CRISPRi_CRISPRa",
        "diseases_relevant": ["CAD"],
        "pmid": "37343252",  # Natsume et al., PLOS Genetics 2023
        "local_path_pattern": "./data/perturbseq/natsume_2023_haec/signatures.json.gz",
        "note": (
            "2,285 CAD GWAS loci × 3 perturbation modes in primary-like aortic endothelial cells. "
            "Secondary CAD dataset after Schnitzler."
        ),
    },

    # -----------------------------------------------------------------------
    # K562 (chronic myelogenous leukemia — large general coverage, fallback)
    # Use for CAD lipid/metabolic programs only when vascular data unavailable.
    # -----------------------------------------------------------------------
    {
        "id": "Replogle2022_K562_essential",
        "geo_id": "GSE246756",
        "figshare_url": "https://ndownloader.figshare.com/files/35780870",
        "cell_type": "K562",
        "n_cells": 2_586_340,
        "n_genes_kd": 9_065,
        "perturbation_type": "CRISPRi",
        "diseases_relevant": ["T2D", "AD"],
        "pmid": "35688146",
        "local_path_pattern": "./data/perturbseq/replogle_2022_k562/K562_essential*.h5ad",
        "note": (
            "Largest Perturb-seq dataset. Covers essential genes (metabolic, cell cycle, "
            "translation). CAD fallback only — prefer Schnitzler/Natsume for vascular biology."
        ),
    },
    {
        "id": "Replogle2022_K562_gwps",
        "geo_id": "GSE246756",
        "figshare_url": "https://ndownloader.figshare.com/files/35773217",
        "cell_type": "K562",
        "n_cells": 2_586_340,
        "n_genes_kd": 9_065,
        "perturbation_type": "CRISPRi",
        "diseases_relevant": [],  # superseded by Schnitzler for CAD; not the default for any disease
        "pmid": "35688146",
        "local_path_pattern": "./data/perturbseq/replogle_2022_k562/K562_gwps*.h5ad",
        "note": "Genome-wide screen. CAD fallback only — superseded by Schnitzler (GSE210681).",
    },

    # -----------------------------------------------------------------------
    # Immune cell Perturb-seq (IBD, RA, SLE relevant)
    # -----------------------------------------------------------------------
    {
        "id": "Papalexi2021_PBMC",
        "geo_id": "GSE164378",
        "zenodo_id": "7041690",
        "cell_type": "PBMC",
        "n_cells": 218_000,
        "n_genes_kd": 750,
        "perturbation_type": "CRISPRko",
        "diseases_relevant": ["IBD", "RA", "SLE"],
        "pmid": "33649592",
        "local_path_pattern": "./data/perturb_seq/papalexi2021/*.h5ad",
        "note": (
            "PBMC Perturb-seq with paired CITE-seq protein measurement. "
            "Covers 750 immune-relevant genes. Ideal for IBD/RA/SLE β estimation."
        ),
    },
    {
        "id": "Dixit2016_PBMC",
        "geo_id": "GSE90063",
        "cell_type": "PBMC",
        "n_cells": 65_000,
        "n_genes_kd": 96,
        "perturbation_type": "CRISPRko",
        "diseases_relevant": ["IBD", "RA"],
        "pmid": "27984732",
        "local_path_pattern": "./data/perturb_seq/dixit2016/*.h5ad",
        "note": "Pioneering in vivo Perturb-seq in mouse + PBMC. Limited to 96 TF knockouts.",
    },
    {
        "id": "Norman2019_K562_TF",
        "geo_id": "GSE133344",
        "zenodo_id": "7041690",
        "cell_type": "K562",
        "n_cells": 111_000,
        "n_genes_kd": 287,
        "perturbation_type": "CRISPRa",
        "diseases_relevant": ["IBD", "CAD", "RA"],
        "pmid": "31420545",
        "local_path_pattern": "./data/perturb_seq/norman2019/*.h5ad",
        "note": (
            "CRISPRa overexpression + combinatorial perturbations. "
            "Covers 287 TFs. Good for transcription factor β in inflammatory programs."
        ),
    },
    {
        "id": "Schmidt2022_T_cells",
        "geo_id": "GSE185045",
        "cell_type": "T cell",
        "n_cells": 150_000,
        "n_genes_kd": 111,
        "perturbation_type": "CRISPRko",
        "diseases_relevant": ["IBD", "RA", "SLE"],
        "pmid": "35549406",
        "local_path_pattern": "./data/perturb_seq/schmidt2022/*.h5ad",
        "note": "CD8+ T cell Perturb-seq focused on immune exhaustion and signaling genes.",
    },

    # -----------------------------------------------------------------------
    # Neurodegeneration
    # -----------------------------------------------------------------------
    {
        "id": "Ursu2022_iPSC_neurons",
        "geo_id": "GSE196862",
        "zenodo_id": "7041690",
        "cell_type": "neuron",
        "n_cells": 200_000,
        "n_genes_kd": 1_300,
        "perturbation_type": "CRISPRi",
        "diseases_relevant": ["AD"],
        "pmid": "35878621",
        "local_path_pattern": "./data/perturb_seq/ursu2022/*.h5ad",
        "note": "iPSC-derived neuron Perturb-seq. Best for AD-relevant neuronal programs.",
    },

    # -----------------------------------------------------------------------
    # Metabolic / T2D
    # -----------------------------------------------------------------------
    {
        "id": "Replogle2022_RPE1",
        "geo_id": "GSE246756",
        "figshare_url": "https://ndownloader.figshare.com/files/35780876",
        "cell_type": "RPE1",
        "n_cells": 1_194_982,
        "n_genes_kd": 7_975,
        "perturbation_type": "CRISPRi",
        "diseases_relevant": ["T2D", "CAD", "AMD"],
        "pmid": "35688146",
        "local_path_pattern": "./data/perturb_seq/replogle2022/RPE1*.h5ad",
        "note": "Diploid retinal epithelial. Better than K562 for metabolic gene essentials. Also used for AMD (retinal epithelial biology).",
    },
]


# ---------------------------------------------------------------------------
# Disease → priority-ordered dataset list
# ---------------------------------------------------------------------------

_DISEASE_PRIORITY: dict[str, list[str]] = {
    # CAD: vascular cell types first (HCASMC/HAEC) → K562 fallback only for lipid programs
    "CAD":  ["Schnitzler2023_CAD_vascular", "Natsume2023_HAEC", "Replogle2022_K562_essential"],
    "IBD":  ["Papalexi2021_PBMC",           "Schmidt2022_T_cells", "Norman2019_K562_TF"],
    "RA":   ["Papalexi2021_PBMC",           "Schmidt2022_T_cells", "Dixit2016_PBMC"],
    "SLE":  ["Papalexi2021_PBMC",           "Schmidt2022_T_cells"],
    "AD":   ["Ursu2022_iPSC_neurons",       "Replogle2022_K562_essential"],
    "T2D":  ["Replogle2022_RPE1",           "Replogle2022_K562_essential"],
    # AMD: RPE1 (retinal epithelial = tissue match) > K562 (generic fallback)
    "AMD":  ["Replogle2022_RPE1",           "Replogle2022_K562_essential"],
}

_CATALOG_INDEX: dict[str, dict] = {d["id"]: d for d in PERTURB_SEQ_CATALOG}


def get_perturb_datasets_for_disease(disease: str) -> dict:
    """
    Return prioritised Perturb-seq datasets for a disease, checking local availability.

    Args:
        disease: Short disease name (CAD, IBD, RA, SLE, AD, T2D)

    Returns:
        {
            "disease": str,
            "recommended": dict | None,   # best available local dataset
            "priority_list": list[dict],  # all datasets in priority order
            "local_available": list[dict],# datasets with h5ad files on disk
            "needs_download": list[dict], # top-priority not yet downloaded
        }
    """
    priority_ids = _DISEASE_PRIORITY.get(disease.upper(), [])
    if not priority_ids:
        # Fall back to all relevant datasets
        priority_ids = [
            d["id"] for d in PERTURB_SEQ_CATALOG
            if disease.upper() in d.get("diseases_relevant", [])
        ]

    priority_list = [_CATALOG_INDEX[did] for did in priority_ids if did in _CATALOG_INDEX]

    # Check which datasets are locally available
    local_available = []
    needs_download = []
    for ds in priority_list:
        pattern = ds.get("local_path_pattern", "")
        if pattern:
            found = list(Path(".").glob(pattern.lstrip("./")))
            if found:
                ds_copy = dict(ds)
                ds_copy["local_path"] = str(found[0])
                local_available.append(ds_copy)
                continue
        needs_download.append(ds)

    recommended = local_available[0] if local_available else None

    return {
        "disease":         disease.upper(),
        "recommended":     recommended,
        "priority_list":   priority_list,
        "local_available": local_available,
        "needs_download":  needs_download[:3],  # top 3 to download
        "n_available":     len(local_available),
    }


def get_download_commands(disease: str) -> list[str]:
    """Return shell commands to download the recommended Perturb-seq dataset."""
    result = get_perturb_datasets_for_disease(disease)
    if not result["needs_download"]:
        return ["# All datasets already downloaded"]

    ds = result["needs_download"][0]
    commands = [f"# Download {ds['id']} ({ds['cell_type']} Perturb-seq for {disease})"]
    out_dir = ds["local_path_pattern"].rsplit("/", 1)[0]
    commands.append(f"mkdir -p {out_dir}")

    if ds.get("figshare_url"):
        commands.append(f"wget -O {out_dir}/{ds['id']}.h5ad {ds['figshare_url']}")
    elif ds.get("geo_id"):
        commands.append(f"# GEO accession: {ds['geo_id']}")
        commands.append(
            f"# Download from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={ds['geo_id']}"
        )
    elif ds.get("zenodo_id"):
        commands.append(
            f"# Zenodo record: https://zenodo.org/record/{ds['zenodo_id']}"
        )

    return commands
