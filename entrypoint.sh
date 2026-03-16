#!/bin/bash
set -e

# Step 1: build cell hulls (skip if already exists — supports resume)
if [ ! -f /app/data/cell_hulls.csv ]; then
    echo "Building cell hulls..."
    python -m mobile_coverage.src.build_cell_hulls
else
    echo "Cell hulls already exist, skipping."
fi

# Step 2: run experiments
echo "Running experiments..."
python -m mobile_coverage.src.run_experiments
