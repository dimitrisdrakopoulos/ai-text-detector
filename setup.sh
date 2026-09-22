#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo
echo "Setup complete. Run with:"
echo "  source .venv/bin/activate && streamlit run app.py"
