# 🩺 Troubleshooting & FAQ

## Common issues and how to resolve them.

### UI loads but model calls fail

- **Symptom**: Streamlit app loads, but requesting a run returns authentication errors or "LLMError".
- **Fix**: Ensure `OPENAI_API_KEY` and `DEEPSEEK_API_KEY` are set in `.env` and restart Streamlit. Check network connectivity. Verify the provider configuration in the agent editor matches your API keys.

### PermissionError when writing `data/`

- **Symptom**: Exceptions while creating `data/db` or `data/logs`.
- **Fix**: Ensure the current user has write permissions in the project directory. For containers, ensure volume mounts have correct UID/GID.

### Python version syntax errors

- **Symptom**: Syntax errors for `str | None` or other newer syntax.
- **Fix**: Use Python ≥ 3.10, < 3.14 (tested with CPython 3.13). Check your Python version with `python --version`. If using a virtual environment, ensure it uses a compatible Python interpreter.

### Graphviz export / rendering problems

- **Symptom**: Graph export to PNG/PDF fails with `dot: command not found`.
- **Fix**: Install system Graphviz (`apt install graphviz` or `brew install graphviz`) if you plan to render files. In-browser Streamlit `graphviz_chart` usually works without the binary.

### Migration issues / destructive changes

- **Symptom**: Migrations require table rebuilds; DB is non‑empty.
- **Fix**: Back up DB, then run the rebuild tool in dry‑run mode and then execute with backup enabled. **Important**: Run `sqlite_rebuild.py` as a direct script from the repository root, not as a module:

  ```bash
  # Preview rebuild plan
  python src/multi_agent_dashboard/db/infra/sqlite_rebuild.py --dry-run --all-with-diffs data/db/multi_agent_runs.db

  # Execute rebuild (creates backups automatically)
  python src/multi_agent_dashboard/db/infra/sqlite_rebuild.py --all-with-diffs data/db/multi_agent_runs.db
  ```

  For detailed migration workflow, see [MIGRATIONS.md](MIGRATIONS.md).

### App extremely slow or high latency

- **Symptom**: Long waits during runs.
- **Fix**: Check the model used by agents (larger models are slower). Inspect logs (`data/logs/`) for network retries or backoff. Consider running smaller models for local experiments. Enable Langfuse observability to break down latency per agent step.

### Tests/CI missing or failing

- **Symptom**: No tests or CI failures.
- **Fix**: Add unit tests for engine & DB behavior; verify CI config (not currently provided). Run `pytest` to see which tests fail.

### DeepSeek‑reasoner structured output error (400 tool_choice)

- **Symptom**: DeepSeek‑reasoner models may return a 400 error when tool‑choice is enabled.
- **Fix**: pending

### LangChain not available

- **Symptom**: Import errors or "LangChain not available" warnings.
- **Fix**: Ensure LangChain is installed (`pip install langchain`). The codebase gracefully handles missing LangChain via the `LANGCHAIN_AVAILABLE` flag, but agent creation will fail without it.

### Provider‑specific adapter errors

- **Symptom**: `NotImplementedError` for missing provider implementations.
- **Fix**: Supported providers are `openai`, `deepseek`, and `ollama`. Check that the provider name matches exactly. If adding a new provider, ensure its capabilities are in the external provider data and implement provider‑specific adapters.

### Missing provider models file

- **Symptom**: `FileNotFoundError` for `provider_models_all.json` or `provider_models.json`.
- **Fix**: On first run, the system downloads external provider data automatically. If the download fails, check network connectivity. You can manually delete `data/provider_models/provider_models_all.json` to trigger a fresh download.

### Structured schema validation fails / early exit

- **Symptom**: Pipeline exits early with `agent_schema_validation_failed` flag set.
- **Fix**: When `strict_schema_validation=True` and a schema is missing or empty, the run may early‑exit. Check the agent’s structured output configuration in the UI and ensure a valid JSON schema is provided. The UI displays badges (`[strict schema]`, `[schema failed]`) to indicate the status.

### Ollama unreachable

- **Symptom**: Errors connecting to local Ollama endpoint.
- **Fix**: Ensure Ollama is running (`ollama serve`). Check the `OLLAMA_PROTOCOL` environment variable (default `http`). Verify the endpoint is reachable from the dashboard.

### File upload size limits

- **Symptom**: "File size limits exceeded" error when uploading attachments.
- **Fix**: The UI enforces a 5MB per‑file limit and 20MB total limit. Reduce file sizes or split content across multiple agents.

### Tool dependency missing

- **Symptom**: `ImportError` for missing libraries (`beautifulsoup4`, `markdownify`, `requests`, etc.).
- **Fix**: Install required tool dependencies: `pip install beautifulsoup4 markdownify requests duckduckgo-search`. See `pyproject.toml` for optional dependencies.

### Langfuse initialization failures

- **Symptom**: Langfuse tracing not working despite setting `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`.
- **Fix**: Check network connectivity to Langfuse servers. Verify API keys are correct. You can disable Langfuse by removing the environment variables or setting `LANGFUSE_ENABLED=false`.

### Agent snapshot auto‑creation not working

- **Symptom**: Saving an agent does not automatically create a snapshot.
- **Fix**: By default, automatic snapshot creation is disabled (`AGENT_SNAPSHOTS_AUTO = false`). Enable it by setting `agent_snapshots_auto: true` in `config/agents.yaml`.

## If the above doesn't help

- Open an issue and include:
  - Platform (OS + Python version)
  - Steps to reproduce
  - Relevant logs from `data/logs/`
  - Small reproduction (pipeline JSON / steps)

See also:
- [README.md](../README.md) for quick start and essential configuration
- [docs/CONFIG.md](CONFIG.md) for full environment variable and YAML reference
- [docs/INSTALL.md](INSTALL.md) for detailed installation guide
- [docs/MIGRATIONS.md](MIGRATIONS.md) for database migration and rebuild guidance
- [docs/ARCHITECTURE.md](ARCHITECTURE.md) for architecture details