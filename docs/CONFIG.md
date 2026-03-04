# ⚙️ Configuration Reference

This document lists the main configuration knobs and where they live.

## Primary config files / locations

- `.env` (project root) — environment variables for secrets and user-specific overrides (see `.env.template`)
- `config/` (project root) — YAML‑based configuration files with programmatic defaults:
  - `paths.yaml` — directory and file names
  - `agents.yaml` — agent limits and snapshot settings
  - `providers.yaml` — provider‑data file names and external URLs
  - `ui.yaml` — UI colors and attachment file types
  - `logging.yaml` — default log level

Configuration is centralized in the `src/multi_agent_dashboard/config/` package, which loads YAML files, validates them with Pydantic, and merges environment‑variable overrides.

## 🌱 Environment Variables (`.env` at project root)

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key used by the LLM client |
| `DEEPSEEK_API_KEY` | Yes | DeepSeek API key for DeepSeek provider |
| `OLLAMA_PROTOCOL` | Optional | Ollama server protocol (default: `http`) |
| `DB_FILE` | Optional | Override default database filename |
| `LOG_LEVEL` | Optional | Logging level (INFO, DEBUG, etc.) |
| `LANGFUSE_PUBLIC_KEY` | Optional | Langfuse public API key (enables observability) |
| `LANGFUSE_SECRET_KEY` | Optional | Langfuse secret API key (required if public key is set) |
| `LANGFUSE_BASE_URL` | Optional | Langfuse server URL (default: `https://cloud.langfuse.com`) |
| `LANGFUSE_ENABLED` | Optional | Explicitly disable Langfuse integration (set to `false` to disable even if keys are present) |
| `RAISE_ON_AGENT_FAIL` | Optional | Whether to raise exceptions on agent failure (default: `true`) |
| `AGENT_INPUT_CHAR_CAP` | Optional | Maximum input character count per agent (overrides `agents.yaml`) |
| `AGENT_OUTPUT_CHAR_CAP` | Optional | Maximum output character count per agent (overrides `agents.yaml`) |
| `AGENT_OUTPUT_TOKEN_CAP` | Optional | Maximum output token limit per agent (overrides `agents.yaml`; `0` = no limit) |
| `STRICT_OUTPUT_TOKEN_CAP_OVERRIDE` | Optional | If `true`, ignore per‑agent `max_output` and enforce `AGENT_OUTPUT_TOKEN_CAP` globally (default: `false`) |

If `OPENAI_API_KEY` is missing or invalid, LLM calls will fail at runtime; the UI may load but requests to the model will error.

- `DEEPSEEK_API_KEY` is required when using the DeepSeek provider.
- `OLLAMA_PROTOCOL` defaults to `http` (can be set to `https` for remote Ollama servers).
- `DB_FILE` can be used to specify a custom SQLite database filename (relative to the `data/db/` directory).
- `LOG_LEVEL` overrides the default log level defined in `logging.yaml`.
- `AGENT_INPUT_CHAR_CAP`, `AGENT_OUTPUT_CHAR_CAP`, and `AGENT_OUTPUT_TOKEN_CAP` override the corresponding YAML defaults.
- `STRICT_OUTPUT_TOKEN_CAP_OVERRIDE` changes the token‑limit precedence (see [Agent Caps](#agent-caps)).

## 🧱 YAML Configuration Files

All YAML files reside in the `config/` directory and are validated at import time via Pydantic models.

### `paths.yaml` – Directory and file names

```yaml
data_dir: "data"
db_dir: "db"
migrations_dir: "migrations"
logs_dir: "logs"
dotenv_file: ".env"
log_file: "application.log"
db_file: "multi_agent_runs.db"          # can be overridden by DB_FILE env
provider_data_dir: "data/provider_models"
```

These values are used to compute the derived paths listed below.

### `agents.yaml` – Agent limits and snapshot settings

```yaml
# Character caps per agent
agent_input_char_cap: 40000
agent_output_char_cap: 50000

# Token cap per agent (0 = no limit)
agent_output_token_cap: 0

# Snapshot behavior
agent_snapshots_auto: false
agent_snapshot_prune_auto: false
agent_snapshot_prune_keep: 100
```

### `providers.yaml` – Provider‑data file names and URLs

```yaml
# Provider‑data file names
provider_models_all_file: "provider_models_all.json"
provider_models_file: "provider_models.json"
template_ollama_models_file: "template_ollama_models.json"
local_ollama_models_file: "local_ollama_models.json"

# External data source
models_dev_url: "https://models.dev/api.json"
```

### `ui.yaml` – UI colors and attachment file types

```yaml
# File‑upload restrictions
attachment_file_types_restricted: false
attachment_file_types:
  - "txt"
  - "pdf"
  - "csv"
  - "md"
  - "json"
  - "log"
  - "py"
  - "sql"
  - "patch"
  - "LICENSE"
  - "toml"
  - "gitignore"

# Color palette & symbols
ui_colors:
  red:
    value: "#dc3545"
    symbol: "🔴"
  orange:
    value: "#fd7e14"
    symbol: "🟠"
  … (see the file for the complete palette)
```

### `logging.yaml` – Default log level

```yaml
# Default log level (can still be overridden by LOG_LEVEL env)
default_log_level: "INFO"
```

## 📁 Derived Paths & Constants

The configuration package computes the following absolute paths (exposed as module‑level constants):

| Constant | Description |
|----------|-------------|
| `DATA_PATH` | `PROJECT_ROOT / "data"` |
| `DB_PATH` | `DATA_PATH / "db"` |
| `MIGRATIONS_PATH` | `DATA_PATH / "migrations"` |
| `LOGS_PATH` | `DATA_PATH / "logs"` |
| `DB_FILE_PATH` | `DB_PATH / DB_FILE` (final SQLite DB file) |
| `LOG_FILE_PATH` | `LOGS_PATH / "application.log"` |
| `PROVIDER_DATA_PATH` | `DATA_PATH / "provider_models"` |

All constants are exported from `multi_agent_dashboard.config` (see [Using Configuration in Code](#using-configuration-in-code)).

## Agent Caps

- `AGENT_INPUT_CHAR_CAP = 40_000` – maximum input character count per agent (used for prompt value truncation)
- `AGENT_OUTPUT_CHAR_CAP = 50_000` – maximum output character count per agent (used for final prompt truncation)
- `AGENT_OUTPUT_TOKEN_CAP = 0` – maximum output token limit per agent (`0` means no limit)

These character caps can be overridden by environment variables `AGENT_INPUT_CHAR_CAP` and `AGENT_OUTPUT_CHAR_CAP`.

### Token‑limit precedence rules (implemented in `AgentSpec.effective_max_output()`)

1. If `STRICT_OUTPUT_TOKEN_CAP_OVERRIDE = true`: use `AGENT_OUTPUT_TOKEN_CAP` if `> 0`, else `None` (no limit).
2. Otherwise: take the smallest non‑zero value among `AGENT_OUTPUT_TOKEN_CAP` and the agent's own `max_output` field.
3. `0` in any source means no limit and is treated as `None`.
4. The effective limit is passed to the LLM provider as `max_tokens` (`None` indicates no provider‑side limit).

Per‑agent `max_output` can be set via the UI or agent configuration. The precedence ensures safe defaults while allowing granular overrides.

## Configuration Hierarchy & Overrides

Secrets (API keys) and user‑specific preferences remain in `.env` (git‑ignored). The following environment variables **override** YAML defaults:

- `DB_FILE` – override the database file name
- `LOG_LEVEL` – override the default log level (still falls back to YAML if not set)
- `AGENT_INPUT_CHAR_CAP` – override maximum input character count per agent
- `AGENT_OUTPUT_CHAR_CAP` – override maximum output character count per agent
- `AGENT_OUTPUT_TOKEN_CAP` – override maximum output token limit per agent (`0` = no limit)
- `STRICT_OUTPUT_TOKEN_CAP_OVERRIDE` – if `true`, ignore per‑agent `max_output` and enforce `AGENT_OUTPUT_TOKEN_CAP` globally (default: `false`)

All other constants are defined in YAML and cannot be overridden by environment variables.

## Adding a New Configuration Value

For developers who need to extend the configuration system:

1. Add the key‑value pair to the appropriate YAML file in `config/`.
2. Update the corresponding Pydantic model in `src/multi_agent_dashboard/config/loader.py`.
3. Add the constant to `src/multi_agent_dashboard/config/core.py` (read from `_yaml_config`).
4. Export the constant in `src/multi_agent_dashboard/config/__init__.py`.

See `AGENTS.md` for a detailed walkthrough.

## Using Configuration in Code

Import the constants directly from the configuration package:

```python
from multi_agent_dashboard.config import (
    AGENT_INPUT_CHAR_CAP,
    AGENT_OUTPUT_CHAR_CAP,
    UI_COLORS,
    DB_FILE_PATH,
    LOG_LEVEL,
    OPENAI_API_KEY,
    DEEPSEEK_API_KEY,
    # … any other constant listed in the package __init__.py
)
```

The import pattern is backward‑compatible with the previous monolithic `config.py`.

## 🔧 Logging

- Log path (rotating file): `data/logs/application.log`
- The app uses a `RotatingFileHandler` with these parameters:
  - maxBytes = 5 * 1024 * 1024 (5 MB) per file
  - backupCount = 3 (keeps up to 3 rotated backups)
- Logs also stream to stdout for easy Streamlit viewing.
- Logging configuration is centralized in `src/multi_agent_dashboard/config/core.py` (`configure_logging()`).

## How to override in dev

- Copy `.env.template` to `.env` and edit values.
- For temporary overrides, export env vars when launching Streamlit:

  ```bash
  OPENAI_API_KEY=your_key LOG_LEVEL=DEBUG streamlit run src/multi_agent_dashboard/ui/app.py
  ```

- To change YAML defaults, edit the appropriate file in `config/` (note that changes may be overwritten by future updates; consider using environment‑variable overrides for personal preferences).

## Secrets handling

- Do NOT commit secrets. Use `.env` for local dev and your preferred secrets manager for any hosted deployments.

---

If you need a compact programmatic reference, inspect `src/multi_agent_dashboard/config/__init__.py` for the list of exported constants, or run the configuration module directly:

```bash
python -m multi_agent_dashboard.config.core
```