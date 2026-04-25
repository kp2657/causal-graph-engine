"""
scripts/run_track_a.py

Track A validation: run state-space pipeline on real sc-RNA h5ad data.
Run after CELLxGENE downloads complete.

Usage:
    conda run -n causal-graph python scripts/run_track_a.py AMD
    conda run -n causal-graph python scripts/run_track_a.py CAD
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_track_a(disease: str) -> None:
    from pipelines.discovery.cellxgene_downloader import download_disease_scrna
    from pipelines.state_space.latent_model import build_disease_latent_space
    from pipelines.state_space.state_definition import define_cell_states
    from pipelines.state_space.transition_graph import infer_state_transition_graph, get_basin_summary

    print(f"\n{'='*60}")
    print(f"Track A: {disease} state-space pipeline")
    print(f"{'='*60}")

    # ---- Step 1: Check / download h5ad ----
    print("\n[1/4] CELLxGENE download...")
    dl = download_disease_scrna(disease)
    print(f"  status: {dl['status']}")
    print(f"  n_cells: {dl.get('n_cells', 0)}")
    if dl.get("note"):
        print(f"  note: {dl['note']}")

    if dl["status"] == "unavailable" or not dl.get("h5ad_path"):
        print("ABORT: h5ad not available")
        return

    h5ad_path = dl["h5ad_path"]
    print(f"  h5ad: {h5ad_path}")

    # ---- Step 2: Latent embedding ----
    print("\n[2/4] Building latent embedding...")
    latent = build_disease_latent_space(disease, [h5ad_path])
    if latent.get("error"):
        print(f"ABORT: latent model error: {latent['error']}")
        return

    adata = latent["adata"]
    print(f"  backend: {latent['backend']}")
    print(f"  cells: {adata.n_obs}, genes: {adata.n_vars}")
    if latent["integration_warnings"]:
        print(f"  warnings: {latent['integration_warnings']}")

    # ---- Step 3: Cell states ----
    print("\n[3/4] Defining cell states...")
    states = define_cell_states(latent, disease)
    for res_name, slist in states.items():
        path_states = [s for s in slist if s.pathological_score and s.pathological_score >= 0.6]
        print(f"  {res_name}: {len(slist)} states, {len(path_states)} pathological")

    # ---- Step 4: Transition graph ----
    print("\n[4/4] Inferring transition graph...")
    trans = infer_state_transition_graph(latent, states, disease)
    print(f"  transitions: {len(trans['transitions'])}")
    print(f"  pathological basins: {len(trans['pathologic_basin_ids'])}")
    print(f"  healthy basins: {len(trans['healthy_basin_ids'])}")
    print(f"  escape basins: {len(trans['escape_basin_ids'])}")
    print(f"  confidence: {json.dumps(trans['confidence_summary'], indent=4)}")

    print(f"\n{'='*60}")
    print(f"Track A COMPLETE for {disease}")
    print(f"{'='*60}")


if __name__ == "__main__":
    disease = sys.argv[1] if len(sys.argv) > 1 else "coronary artery disease"
    run_track_a(disease)
