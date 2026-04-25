"""
pipelines/discovery — Data-driven program and γ discovery.

This package replaces pre-specified program registries and hardcoded γ values
with computations from real genomic data:

  1. cellxgene_downloader.py  — download sc-RNA from CELLxGENE census
  2. cnmf_runner.py           — run NMF on sc-RNA to discover programs de novo

Entry point for callers:
    from pipelines.discovery import run_discovery_pipeline
"""
from __future__ import annotations

from pipelines.discovery.cellxgene_downloader import download_disease_scrna
from pipelines.discovery.cnmf_runner import run_nmf_programs, load_computed_programs


def run_discovery_pipeline(
    disease: str,
    efo_id: str,
    n_programs: int = 20,
    force_recompute: bool = False,
) -> dict:
    """
    Full discovery pipeline: sc-RNA download → NMF programs.

    Steps:
      1. Download disease-matched sc-RNA from CELLxGENE census (if not cached)
      2. Run NMF to discover programs de novo (if not cached)
      3. Return programs ready for the Ota computation

    Args:
        disease:        Short disease name, e.g. "IBD", "CAD"
        efo_id:         EFO disease ID, e.g. "EFO_0003767"
        n_programs:     Number of NMF programs to extract (default 20)
        force_recompute: Rerun even if cached results exist

    Returns:
        {
            "programs": list[{program_id, gene_set, top_genes, gene_loadings}],
            "source": str,
            "n_programs": int,
        }
    """
    from pathlib import Path

    programs_path = Path(f"./data/cnmf_programs/{disease}_programs.json")

    if not programs_path.exists() or force_recompute:
        scrna = download_disease_scrna(disease)
        h5ad_path = scrna.get("h5ad_path")
        if h5ad_path and Path(h5ad_path).exists():
            programs_result = run_nmf_programs(
                h5ad_path=h5ad_path,
                disease=disease,
                n_programs=n_programs,
            )
        else:
            from pipelines.cnmf_programs import get_programs_for_disease
            programs_result = get_programs_for_disease(disease)
    else:
        programs_result = load_computed_programs(disease)

    programs = programs_result.get("programs", [])

    return {
        "programs":   programs,
        "source":     programs_result.get("source", "discovery_pipeline"),
        "n_programs": len(programs),
        "disease":    disease,
        "efo_id":     efo_id,
    }


def _disease_to_traits(disease: str) -> list[str]:
    """Map short disease name to relevant trait labels."""
    from graph.schema import DISEASE_TRAIT_MAP
    return DISEASE_TRAIT_MAP.get(disease, [disease])
