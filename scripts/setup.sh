#!/bin/bash
# Setup script for Donkey vs Elephant
set -e

echo "🫏 Setting up Donkey vs Elephant... 🐘"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create data directories
mkdir -p data/raw data/processed models/donkey models/elephant logs

echo ""
echo "Setup complete! Activate the environment with:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Add your Congress.gov API key to data_collection/.env"
echo "  2. Run data collection:  python -m data_collection.run_collection"
echo "  3. Train models:         python -m model_training.train --both"
echo "  4. Launch the UI:        python -m web_ui.app"
