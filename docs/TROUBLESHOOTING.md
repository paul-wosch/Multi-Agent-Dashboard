# 🩺 Troubleshooting & FAQ

## Common issues and how to resolve them.

### UI loads but model calls fail

- Symptom: Streamlit app loads, requesting a run returns auth errors.
- Fix: Ensure `OPENAI_API_KEY` is set in `.env` and restart Streamlit. Check network connectivity.

### PermissionError when writing `data/`

- Symptom: Exceptions while creating `data/db` or `data/logs`.
- Fix: Ensure the current user has write permissions in the project directory. For containers, ensure volume mounts have correct UID/GID.

### Python version syntax errors

- Symptom: Syntax errors for `str | None` or other newer syntax.
- Fix: Use Python >= 3.11. (This repo is tested with CPython 3.14.)

### Graphviz export / rendering problems

- Symptom: Graph export to PNG/PDF fails with `dot: command not found`.
- Fix: Install system Graphviz (`apt install graphviz` or `brew install graphviz`) if you plan to render files. In-browser Streamlit graphviz_chart usually works without the binary.

### Migration issues / `_REQUIRES_REBUILD`

- Symptom: Migrations require table rebuilds; DB is non-empty.
- Fix: Back up DB, then run the rebuild tool in dry-run mode and then execute with backup enabled:
  
  ```bash
  python -m multi_agent_dashboard.db.infra.sqlite_rebuild --all-with-diffs data/db/multi_agent_runs.db
  ```

### App extremely slow or high latency

- Symptom: Long waits during runs.
- Fix: Check the model used by agents (larger models are slower). Inspect logs (`data/logs/`) for network retries or backoff. Consider running smaller models for local experiments.

### Tests/CI missing or failing

- Symptom: No tests or CI failures.
- Fix: Add unit tests for engine & DB behavior; verify CI config (not currently provided).

## If the above doesn't help

- Open an issue and include:
  - Platform (OS + Python version)
  - Steps to reproduce
  - Relevant logs from `data/logs/`
  - Small reproduction (pipeline JSON / steps)

See also: README troubleshooting short list and [docs/MIGRATIONS.md](MIGRATIONS.md) for migration-related issues.