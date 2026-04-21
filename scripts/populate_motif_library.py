
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.state_space.latent_model import build_disease_latent_space, extract_disease_transition_motif

def populate_motif_library():
    data_dir = Path(__file__).parent.parent / "data"
    cellxgene_dir = data_dir / "cellxgene"
    library_path = data_dir / "motif_library.pkl"

    motif_library = {}

    diseases = {
        "IBD": {
            "h5ad": cellxgene_dir / "IBD" / "IBD_macrophage.h5ad",
            "json": data_dir / "analyze_inflammatory_bowel_disease.json"
        },
        "CAD": {
            "h5ad": cellxgene_dir / "CAD" / "CAD_macrophage.h5ad",
            "json": data_dir / "analyze_coronary_artery_disease.json"
        }
    }

    for name, paths in diseases.items():
        print(f"Processing {name}...")
        if not paths["h5ad"].exists():
            print(f"  Missing h5ad: {paths['h5ad']}")
            continue
        
        # Build latent space to get motif
        # We use cache if available, but force numpy_pca to avoid scanpy/tf crash
        from pipelines.state_space.latent_model import _NumpyPCABackend
        latent = build_disease_latent_space(
            disease=name,
            dataset_paths=[str(paths["h5ad"])],
            use_cache=True,
            backend=_NumpyPCABackend()
        )
        
        motif = latent.get("disease_transition_motif")
        if motif is None:
            print(f"  Failed to extract motif for {name}")
            continue
        
        # Load top genes and their ota_gamma as beta proxy for transfer
        betas = {}
        if paths["json"].exists():
            with open(paths["json"], "r") as f:
                res = json.load(f)
                for target in res.get("target_list", []):
                    # We use ota_gamma as the 'effect' to transfer
                    # In a real hijack, we'd use the raw beta matrix row
                    # but since we don't have it saved, this is the best proxy.
                    gene = target["target_gene"]
                    gamma = target.get("ota_gamma", 0.0)
                    if gamma != 0:
                        betas[gene] = gamma
        
        motif_library[name] = {
            "motif": motif,
            "betas": betas,
            "genes": list(latent["adata"].var_names)
        }
        print(f"  Extracted motif with {len(betas)} gene effects.")

    if motif_library:
        with open(library_path, "wb") as f:
            pickle.dump(motif_library, f)
        print(f"Saved motif library to {library_path}")
    else:
        print("No motifs extracted.")

if __name__ == "__main__":
    populate_motif_library()
