# 🚀 Installation & Getting Started (full)

This doc expands the fast "Try in 5 minutes" steps from the main README with optional and troubleshooting notes.

## Requirements
- Python >= 3.11 (repo tested with CPython 3.14)
- Git
- Network access for OpenAI if you plan to use live models

## Quick install (macOS / Linux)
```bash
git clone <your-repo-url>
cd <repo-root>
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.template .env   # or create .env and set OPENAI_API_KEY
streamlit run src/multi_agent_dashboard/ui/app.py
```

### Windows PowerShell
```powershell
git clone <your-repo-url>
cd <repo-root>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.template .env
streamlit run src/multi_agent_dashboard/ui/app.py
```

### One-click scripts
- `scripts/quick_start.sh` — Linux/macOS quick-start (make executable: `chmod +x scripts/quick_start.sh`)
- `scripts/quick_start.ps1` — Windows PowerShell quick-start

## Notes
- Editable install (`pip install -e .`) is recommended for development to make the `src/` package importable.
- If you prefer direct invocation without installing, run helper scripts directly under `src/`, e.g.:
  ```bash
  python src/multi_agent_dashboard/db/infra/generate_migration.py --help
  ```