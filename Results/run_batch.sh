#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  run_batch.sh — Run headless simulation batch + data ingestion
# ═══════════════════════════════════════════════════════════════
#
#  Prerequisites:
#    1. Build your Unity project as a standalone Linux/Windows build
#    2. Place batch_configs.json alongside the build (or pass absolute path)
#    3. pip install pandas matplotlib   (for the ingestion script)
#
#  Usage:
#    chmod +x run_batch.sh
#    ./run_batch.sh

# ── Configuration ────────────────────────────────────────────
BUILD_PATH="./Builds/MySimulation.x86_64"      # adjust to your build
CONFIG_PATH="./batch_configs.json"
REPEATS=3
TIMESCALE=100    # speed up physics (100x wall-clock)
RESULTS_DIR="./Results"

# ── Run the headless simulation ──────────────────────────────
echo "=== Starting headless batch run ==="
echo "    Build:     $BUILD_PATH"
echo "    Config:    $CONFIG_PATH"
echo "    Repeats:   $REPEATS"
echo "    Timescale: ${TIMESCALE}x"
echo ""

"$BUILD_PATH" \
    -batchmode \
    -nographics \
    -timescale "$TIMESCALE" \
    -batchconfig "$CONFIG_PATH" \
    -repeats "$REPEATS" \
    -logFile ./batch_log.txt

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Simulation exited with code $EXIT_CODE"
    echo "Check batch_log.txt for details."
    exit $EXIT_CODE
fi

echo ""
echo "=== Simulation complete ==="
echo ""

# ── Run data ingestion ───────────────────────────────────────
echo "=== Running data ingestion ==="

python3 ingest_results.py \
    --results-dir "$RESULTS_DIR" \
    --output "${RESULTS_DIR}/summary.csv" \
    --plot \
    --plot-dir "${RESULTS_DIR}/plots"

echo ""
echo "=== All done ==="
echo "  Summary:  ${RESULTS_DIR}/summary.csv"
echo "  Plots:    ${RESULTS_DIR}/plots/"
echo "  Raw logs: ${RESULTS_DIR}/*.csv"