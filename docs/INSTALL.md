# 🚀 Installation & Getting Started (full)

This doc expands the fast "Try in 5 minutes" steps from the main README with additional details about configuration, environment variables, and first‑run behavior.

## Requirements

- **Python**: ≥3.10, <3.14 (tested with CPython 3.13). The project uses modern Python syntax and type hints.
- **Git**: To clone the repository.
- **Network access**: Required for downloading external provider data, fetching LLM models (OpenAI, DeepSeek), and optional Langfuse observability.

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

### One‑click convenience scripts

- `scripts/quick_start.sh` – creates venv, installs, copies `.env`, starts app (Linux/macOS; make executable: `chmod +x scripts/quick_start.sh`)
- `scripts/quick_start.ps1` – same for Windows PowerShell

## Environment Variables

Secrets (API keys) and user‑specific preferences are stored in `.env` (git‑ignored). The following variables are supported:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `DEEPSEEK_API_KEY` | Yes | DeepSeek API key |
| `OLLAMA_PROTOCOL` | Optional | Ollama server protocol (default: `http`) |
| `DB_FILE` | Optional | Override default database filename |
| `LOG_LEVEL` | Optional | Logging level (INFO, DEBUG, etc.); falls back to `logging.yaml` |
| `LANGFUSE_PUBLIC_KEY` | Optional | Langfuse public API key (enables observability) |
| `LANGFUSE_SECRET_KEY` | Optional | Langfuse secret API key (required if public key is set) |
| `LANGFUSE_BASE_URL` | Optional | Langfuse server URL (default: `https://cloud.langfuse.com`) |
| `LANGFUSE_ENABLED` | Optional | Explicitly disable Langfuse integration (set to `false` to disable even if keys are present) |
| `RAISE_ON_AGENT_FAIL` | Optional | Whether to raise exceptions on agent failure (default: `true`) |
| `AGENT_INPUT_CHAR_CAP` | Optional | Maximum input character count per agent (overrides `agents.yaml`) |
| `AGENT_OUTPUT_CHAR_CAP` | Optional | Maximum output character count per agent (overrides `agents.yaml`) |
| `AGENT_OUTPUT_TOKEN_CAP` | Optional | Maximum output token limit per agent (overrides `agents.yaml`; `0` = no limit) |
| `STRICT_OUTPUT_TOKEN_CAP_OVERRIDE` | Optional | If `true`, ignore per‑agent `max_output` and enforce `AGENT_OUTPUT_TOKEN_CAP` globally (default: `false`) |

**Note**: The variables `DB_FILE`, `LOG_LEVEL`, `AGENT_INPUT_CHAR_CAP`, `AGENT_OUTPUT_CHAR_CAP`, `AGENT_OUTPUT_TOKEN_CAP`, and `STRICT_OUTPUT_TOKEN_CAP_OVERRIDE` override YAML configuration defaults; all other constants are defined in YAML files and cannot be overridden by environment variables.

## Configuration (YAML‑based)

The project uses a centralized YAML‑based configuration system located in the `config/` directory:

- `paths.yaml` – directory and file names
- `agents.yaml` – agent limits and snapshot settings
- `providers.yaml` – provider‑data file names and URLs
- `ui.yaml` – UI colors and attachment file types
- `logging.yaml` – default log level

Each file is validated with Pydantic at import time; missing or malformed keys raise immediate errors. For a full configuration reference, see [CONFIG.md](CONFIG.md).

## Provider Data (Dynamic Capabilities & Pricing)

The dashboard loads dynamic provider capabilities and pricing per 1M tokens from external data (`provider_models.json`) with optional local overrides for Ollama models. This data is used for cost computation and advisory feature detection.

### Initialization
On first run, the system downloads external provider data and creates filtered copies in `data/provider_models/`. You can customize local Ollama models by copying the template:

```bash
cp data/provider_models/template_ollama_models.json data/provider_models/local_ollama_models.json
```

Edit `local_ollama_models.json` to add, remove, or modify Ollama model entries. Local entries take precedence over external definitions for the same `ollama|model` composite keys.

### Manual Updates
To refresh external provider data (OpenAI, DeepSeek, etc.):

- Delete `data/provider_models/provider_models_all.json` to trigger a fresh download.
- Delete `data/provider_models/provider_models.json` to re‑extract the canonical data.

Local Ollama models remain unaffected. For detailed workflows, see the **Dynamic Pricing & Capabilities** section in [AGENTS.md](../AGENTS.md).

## First‑Run Behavior

When you start the dashboard for the first time, it automatically:

- Ensures the `data/`, `data/db/`, `data/logs/`, and `data/migrations/` directories exist.
- Creates the SQLite database at `data/db/multi_agent_runs.db`.
- Applies SQL migrations from `data/migrations/` (if any).
- Seeds default agents (planner/solver/critic/finalizer‑style roles) if the `agents` table is empty.
- Initializes a rotating log file under `data/logs/`.

No manual database setup is required. For details on database migrations and schema changes, refer to [MIGRATIONS.md](MIGRATIONS.md).

## Additional Notes

- **Editable install (`pip install -e .`)** is recommended for development to make the `src/` package importable.
- If you prefer direct invocation without installing, you can run helper scripts directly under `src/`, e.g.:
  ```bash
  python src/multi_agent_dashboard/db/infra/generate_migration.py --help
  ```
- The project architecture is documented in [ARCHITECTURE.md](ARCHITECTURE.md); observability features (including optional Langfuse integration) are described in [OBSERVABILITY.md](OBSERVABILITY.md).
- For troubleshooting common issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).