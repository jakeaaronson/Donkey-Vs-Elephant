#!/bin/bash
# Quick start: collect a small sample, train short runs, and launch the UI
set -e

source venv/bin/activate

echo "=== Step 1: Collecting sample data (100 pages per party) ==="
python -m data_collection.run_collection --start-year 2020 --end-year 2024 --max-pages 100

echo ""
echo "=== Step 2: Training models (quick 1-epoch run) ==="
python -m model_training.train --both --epochs 1 --batch-size 4

echo ""
echo "=== Step 3: Launching comparison UI ==="
python -m web_ui.app
