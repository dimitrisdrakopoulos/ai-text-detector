#!/usr/bin/env bash
# Sets up the ai-text-detector app, working around an upstream packaging bug:
# `pip install git+https://github.com/paulgp/econ-ai-detector` does NOT ship
# the models/ directory (only the econ_ai_detector/ package is declared in
# its pyproject.toml, so the sibling models/ folder is dropped from the
# wheel). An editable install from a clone keeps the package pointed at its
# original source tree, where models/ is still a sibling directory, so
# Detector() can find its weights.
set -euo pipefail
cd "$(dirname "$0")"

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

if [ ! -d vendor/econ-ai-detector ]; then
    mkdir -p vendor
    git clone https://github.com/paulgp/econ-ai-detector vendor/econ-ai-detector
fi
pip install -e vendor/econ-ai-detector

echo
echo "Setup complete. Run with:"
echo "  source .venv/bin/activate && streamlit run app.py"
