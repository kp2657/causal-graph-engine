"""
perturb_registry.py — Catalog of Perturb-seq datasets for β_{gene→program} estimation.

Each entry maps a disease/cell-type context to available Perturb-seq datasets,
ordered by biological relevance. This replaces ad-hoc decisions about which
Perturb-seq data to use for each disease.

Usage:
    from pipelines.discovery.perturb_registry import get_perturb_datasets_for_disease
    datasets = get_perturb_datasets_for_disease("IBD")
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
    # K562 (chronic myelogenous leukemia — general metabolic/essentials)
    # -----------------------------------------------------------------------
    {
        "id": "Replogle2022_K562_essential",
        "geo_id": "GSE246756",
        "figshare_url": "https://ndownloader.figshare.com/files/35780870",
        "cell_type": "K562",
        "n_cells": 2_586_340,
        "n_genes_kd": 9_065,
        "perturbation_type": "CRISPRi",
        "diseases_relevant": ["CAD", "T2D", "AD"],
        "pmid": "35688146",
        "local_path_pattern": "./data/perturb_seq/replogle2022/K562_essential*.h5ad",
        "note": (
            "Largest Perturb-seq dataset. Covers essential genes (metabolic, cell cycle, "
            "translation). Best for CAD lipid/CHIP programs. NOT suitable for immune diseases."
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
        "diseases_relevant": ["CAD"],
        "pmid": "35688146",
        "local_path_pattern": "./data/perturb_seq/replogle2022/K562_gwps*.h5ad",
        "note": "Genome-wide perturbation screen. ~357 MB. Best for broad CAD target coverage.",
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
        "diseases_relevant": ["T2D", "CAD"],
        "pmid": "35688146",
        "local_path_pattern": "./data/perturb_seq/replogle2022/RPE1*.h5ad",
        "note": "Diploid retinal epithelial. Better than K562 for metabolic gene essentials.",
    },
]


# ---------------------------------------------------------------------------
# Disease → priority-ordered dataset list
# ---------------------------------------------------------------------------

_DISEASE_PRIORITY: dict[str, list[str]] = {
    "CAD":  ["Replogle2022_K562_gwps",  "Replogle2022_K562_essential", "Norman2019_K562_TF"],
    "IBD":  ["Papalexi2021_PBMC",       "Schmidt2022_T_cells",         "Norman2019_K562_TF"],
    "RA":   ["Papalexi2021_PBMC",       "Schmidt2022_T_cells",         "Dixit2016_PBMC"],
    "SLE":  ["Papalexi2021_PBMC",       "Schmidt2022_T_cells"],
    "AD":   ["Ursu2022_iPSC_neurons",   "Replogle2022_K562_essential"],
    "T2D":  ["Replogle2022_RPE1",       "Replogle2022_K562_essential"],
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
