#!/bin/bash
# Quick start: download data, train short runs, and launch the UI
set -e

source venv/bin/activate

echo "=== Step 1: Downloading and processing Stanford dataset ==="
python -m data_collection.run_collection --start-congress 110 --end-congress 114

echo ""
echo "=== Step 2: Training models (quick 1-epoch run) ==="
python -m model_training.train --both --epochs 1 --batch-size 4

echo ""
echo "=== Step 3: Launching comparison UI ==="
python -m web_ui.app
