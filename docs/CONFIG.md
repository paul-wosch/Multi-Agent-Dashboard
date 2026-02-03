# ⚙️ Configuration Reference

This document lists the main configuration knobs and where they live.

## Primary config files / locations

- `.env` (project root) — environment variables for local dev (see `.env.template`)
- `src/multi_agent_dashboard/config.py` — programmatic defaults & derived paths

Most configuration is centralized in `config.py` and `.env`.

## 🌱 Environment Variables (`.env` at project root)

| Name              | Required | Default | Description                                                |
| ----------------- | -------- | ------- | ---------------------------------------------------------- |
| `OPENAI_API_KEY`  | ✅        | None    | OpenAI API key used by the LLM client                      |
| `DEEPSEEK_API_KEY`| ❌        | None    | DeepSeek API key for DeepSeek provider                     |
| `USE_LITELLM`     | ❌        | `false` | Enable unified provider access via LiteLLM (true/false)    |
| `LOG_LEVEL`       | ❌        | `INFO`  | Global logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `DB_FILE`         | ❌        | `multi_agent_runs.db` | Custom SQLite database filename          |

If `OPENAI_API_KEY` is missing or invalid, LLM calls will fail at runtime; the UI may load but requests to the model will error.

- `DEEPSEEK_API_KEY` is required when using the DeepSeek provider
- `USE_LITELLM` enables unified provider access via LiteLLM (default `false` during phased rollout). When enabled, structured output uses LiteLLM's JSON Schema format for all providers.
- `DB_FILE` can be used to specify a custom SQLite database filename

## 🧱 Core Paths & Caps (from `config.py`)

| Setting            | Default                       | Description                                 |
| ------------------ | ----------------------------- | ------------------------------------------- |
| `PROJECT_ROOT`     | Repo root                     | Auto-detected project root                  |
| `DATA_PATH`        | `PROJECT_ROOT / "data"`       | Root for data and artifacts                 |
| `DB_PATH`          | `DATA_PATH / "db"`            | Directory for SQLite databases              |
| `DB_FILE_PATH`     | `data/db/multi_agent_runs.db` | Main SQLite DB file (auto-created)          |
| `MIGRATIONS_PATH`  | `data/migrations`             | Ordered SQL migrations                      |
| `LOGS_PATH`        | `data/logs`                   | Log directory for rotating app logs         |
| `AGENT_INPUT_CAP`  | defined in `config.py`        | Max characters per formatted input segment  |
| `AGENT_OUTPUT_CAP` | defined in `config.py`        | Max characters per rendered prompt / output |

Prompt formatting and outputs are passed through `utils.safe_format` using these caps to avoid unbounded prompt sizes.

## How to override in dev

- Copy `.env.template` to `.env` and edit values.
- For temporary overrides, export env vars when launching Streamlit:
  
  ```bash
  OPENAI_API_KEY=your_key LOG_LEVEL=DEBUG streamlit run src/multi_agent_dashboard/ui/app.py
  ```

## Secrets handling

- Do NOT commit secrets. Use `.env` for local dev and your preferred secrets manager for any hosted deployments.

---

## 🔧 Logging

- Log path (rotating file): `data/logs/application.log`
- The app uses a RotatingFileHandler with these parameters:
  - maxBytes = 5 * 1024 * 1024 (5 MB) per file
  - backupCount = 3 (keeps up to 3 rotated backups)
- Logs also stream to stdout for easy Streamlit viewing.
- Logging configuration is centralized in `src/multi_agent_dashboard/config.py` (or the logging setup referenced therein).

---

If you need a compact programmatic reference, inspect `src/multi_agent_dashboard/config.py` for derived paths, pricing tables, caps, colors, and emoji symbols used across the UI.