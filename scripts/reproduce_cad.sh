#!/usr/bin/env bash
# Reproduce the v0.2.0 CAD target prioritisation results end-to-end.
# Expected runtime: ~4h with GPS Docker, ~1h without.
# See README.md and docs/RUNTIME.md for prerequisites.
set -euo pipefail

CONDA_ENV="causal-graph"
DISEASE="coronary artery disease"

echo "=== CAD Reproducibility Script (v0.2.0) ==="
echo "Disease: $DISEASE"
echo ""

# --- 1. Check environment ---
if ! conda run -n "$CONDA_ENV" python -c "import causal_graph_engine" 2>/dev/null; then
    echo "[setup] Installing package..."
    conda run -n "$CONDA_ENV" pip install -e . -q
fi

# --- 2a. Preferred: Schnitzler 2023 HCASMC/HAEC Perturb-seq (CAD-relevant vascular cells) ---
# 332 CAD GWAS risk genes perturbed in human coronary artery smooth muscle cells.
# This is the disease-relevant Tier1 data source for CAD; K562 (below) is a myeloid fallback.
# Download from GEO GSE210681 and preprocess before running the pipeline.
SCHNITZLER_DIR="data/perturbseq/schnitzler_cad_vascular"
SCHNITZLER_FILE="$SCHNITZLER_DIR/GSE210681_ALL_log2fcs.txt.gz"
if [ ! -f "$SCHNITZLER_FILE" ]; then
    echo "[data] Downloading Schnitzler 2023 HCASMC Perturb-seq (GSE210681, ~150 MB)..."
    mkdir -p "$SCHNITZLER_DIR"
    wget -q --show-progress \
        -O "$SCHNITZLER_FILE" \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE210nnn/GSE210681/suppl/GSE210681_ALL_log2fcs_dup4_s4n3.99x.txt.gz" \
        && conda run -n "$CONDA_ENV" python -m mcp_servers.perturbseq_server preprocess \
            schnitzler_cad_vascular "$SCHNITZLER_FILE" \
        || echo "[data] WARNING: Schnitzler download failed — pipeline will fall back to K562/eQTL-MR"
else
    echo "[data] Schnitzler HCASMC data found: $SCHNITZLER_FILE"
fi

# --- 2b. Fallback: Replogle K562 h5ad (~370 MB) ---
# K562 is a leukemia cell line — NOT vascular biology. The pipeline demotes K562 β
# for CAD from Tier1_Interventional → Tier3_Provisional automatically (cell-line mismatch).
# Still useful for GPS signature generation and as a generic fallback.
H5AD_PATH="data/perturb_seq/replogle_2022_k562/K562_gwps_normalized_bulk_01.h5ad"
if [ ! -f "$H5AD_PATH" ]; then
    echo "[data] Downloading Replogle 2022 K562 h5ad (~370 MB)..."
    mkdir -p "$(dirname "$H5AD_PATH")"
    wget -q --show-progress \
        -O "$H5AD_PATH" \
        "https://ndownloader.figshare.com/files/35773217" \
        || echo "[data] WARNING: K562 h5ad download failed — GPS will use OTA-derived signature"
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
echo "[pipeline] Starting full CAD pipeline..."
conda run -n "$CONDA_ENV" python -m orchestrator.pi_orchestrator_v2 analyze_disease_v2 "$DISEASE"

# --- 5. Validate ---
echo ""
echo "[validate] Running anchor recovery check..."
conda run -n "$CONDA_ENV" python scripts/validate_results.py --disease cad

echo ""
echo "=== Done. Outputs: ==="
echo "  data/analyze_coronary_artery_disease.json"
echo "  data/analyze_coronary_artery_disease.md"
echo "  data/exports/coronary_artery_disease.ttl"
