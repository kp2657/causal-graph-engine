#!/usr/bin/env bash
# Reproduce the v0.2.0 AMD target prioritisation results end-to-end.
# Expected runtime: ~3h with GPS Docker, ~45min without.
# See README.md and docs/RUNTIME.md for prerequisites.
set -euo pipefail

CONDA_ENV="causal-graph"
DISEASE="age-related macular degeneration"

echo "=== AMD Reproducibility Script (v0.2.0) ==="
echo "Disease: $DISEASE"
echo ""

# --- 1. Check environment ---
if ! conda run -n "$CONDA_ENV" python -c "import causal_graph_engine" 2>/dev/null; then
    echo "[setup] Installing package..."
    conda run -n "$CONDA_ENV" pip install -e . -q
fi

# --- 2. Optional: download Replogle RPE1 h5ad (~100 MB) ---
H5AD_PATH="data/perturb_seq/replogle2022/RPE1_essential_normalized_bulk_01.h5ad"
if [ ! -f "$H5AD_PATH" ]; then
    echo "[data] Downloading Replogle 2022 RPE1 h5ad (~100 MB)..."
    mkdir -p "$(dirname "$H5AD_PATH")"
    wget -q --show-progress \
        -O "$H5AD_PATH" \
        "https://ndownloader.figshare.com/files/35780876" \
        || echo "[data] WARNING: h5ad download failed — pipeline will fall back to eQTL-MR (Tier 2)"
fi

# --- 3. Optional: start GPS Docker ---
if docker info > /dev/null 2>&1; then
    if ! docker ps --filter "name=gps" --filter "status=running" | grep -q gps; then
        echo "[gps] Starting GPS Docker container..."
        docker pull binchengroup/gpsimage:latest -q
        docker run -d --name gps binchengroup/gpsimage:latest sleep infinity
    else
        echo "[gps] GPS Docker already running."
    fi
else
    echo "[gps] Docker not available — GPS screen will be skipped (targets still ranked from genetic evidence)."
fi

# --- 4. Run pipeline ---
echo ""
echo "[pipeline] Starting full AMD pipeline..."
conda run -n "$CONDA_ENV" python -m orchestrator.pi_orchestrator_v2 analyze_disease_v2 "$DISEASE"

# --- 5. Validate ---
echo ""
echo "[validate] Running anchor recovery check..."
conda run -n "$CONDA_ENV" python scripts/validate_results.py --disease amd

echo ""
echo "=== Done. Outputs: ==="
echo "  data/analyze_age_related_macular_degeneration.json"
echo "  data/analyze_age_related_macular_degeneration.md"
echo "  data/exports/age_related_macular_degeneration.ttl"
