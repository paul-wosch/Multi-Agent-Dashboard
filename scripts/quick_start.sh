#!/usr/bin/env bash
set -euo pipefail
# Quick-start script (Linux / macOS)
# - Creates a venv, activates it for the session
# - Installs the package in editable mode
# - Copies .env.template -> .env if present
# - Starts the Streamlit app

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found; install Python 3.11+ and retry" >&2
  exit 1
fi

python -m venv .venv

# Activate for this script session
if [ -f ".venv/bin/activate" ]; then›
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

pip install -e .

if [ -f .env.template ] && [ ! -f .env ]; then
  cp .env.template .env
  echo "Copied .env.template → .env (edit .env to add secrets)"
elif [ ! -f .env ]; then
  echo "OPENAI_API_KEY=your_key_here" > .env
  echo "Created .env (edit to add real secrets)"
fi

echo "Starting Streamlit app..."
streamlit run src/multi_agent_dashboard/ui/app.py