"""
pipelines/discovery — Data-driven program and γ discovery.

This package replaces pre-specified program registries and hardcoded γ values
with computations from real genomic data:

  1. cellxgene_downloader.py  — download sc-RNA from CELLxGENE census
  2. cnmf_runner.py           — run NMF on sc-RNA to discover programs de novo
  3. ldsc_pipeline.py         — estimate γ via GWAS heritability enrichment
  4. perturb_registry.py      — catalog of Perturb-seq datasets for β estimation

Entry point for callers:
    from pipelines.discovery import run_discovery_pipeline
"""
from __future__ import annotations

from pipelines.discovery.cellxgene_downloader import download_disease_scrna
from pipelines.discovery.cnmf_runner import run_nmf_programs, load_computed_programs
from pipelines.discovery.ldsc_pipeline import estimate_program_gamma_enrichment
from pipelines.discovery.perturb_registry import get_perturb_datasets_for_disease


def run_discovery_pipeline(
    disease: str,
    efo_id: str,
    n_programs: int = 20,
    force_recompute: bool = False,
) -> dict:
    """
    Full discovery pipeline: sc-RNA download → NMF programs → GWAS enrichment γ.

    Steps:
      1. Download disease-matched sc-RNA from CELLxGENE census (if not cached)
      2. Run NMF to discover programs de novo (if not cached)
      3. Compute GWAS heritability enrichment per program → γ estimates
      4. Return programs + γ matrix ready for the Ota computation

    Args:
        disease:        Short disease name, e.g. "IBD", "CAD"
        efo_id:         EFO disease ID, e.g. "EFO_0003767"
        n_programs:     Number of NMF programs to extract (default 20)
        force_recompute: Rerun even if cached results exist

    Returns:
        {
            "programs": list[{program_id, gene_set, top_genes, gene_loadings}],
            "gamma_matrix": {program_id: {trait: {gamma, gamma_se, evidence_tier}}},
            "source": str,
            "n_programs": int,
        }
    """
    from pathlib import Path
    import json

    programs_path = Path(f"./data/cnmf_programs/{disease}_programs.json")

    # --- Step 1+2: Download sc-RNA + run NMF ---
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
            # Fall through to MSigDB
            from pipelines.cnmf_programs import get_programs_for_disease
            programs_result = get_programs_for_disease(disease)
    else:
        programs_result = load_computed_programs(disease)

    programs = programs_result.get("programs", [])

    # --- Step 3: GWAS enrichment γ per program ---
    gamma_matrix: dict[str, dict] = {}
    traits = _disease_to_traits(disease)
    for prog in programs:
        pid = prog if isinstance(prog, str) else prog.get("program_id", "")
        gene_set = set() if isinstance(prog, str) else set(prog.get("gene_set", []))
        if not pid:
            continue
        gamma_matrix[pid] = {}
        for trait in traits:
            try:
                g = estimate_program_gamma_enrichment(
                    program_gene_set=gene_set,
                    efo_id=efo_id,
                    program_id=pid,
                    trait=trait,
                )
                gamma_matrix[pid][trait] = g
            except Exception:
                pass

    return {
        "programs":    programs,
        "gamma_matrix": gamma_matrix,
        "source":      programs_result.get("source", "discovery_pipeline"),
        "n_programs":  len(programs),
        "disease":     disease,
        "efo_id":      efo_id,
    }


def _disease_to_traits(disease: str) -> list[str]:
    """Map short disease name to relevant trait labels."""
    from graph.schema import DISEASE_TRAIT_MAP
    return DISEASE_TRAIT_MAP.get(disease, [disease])
