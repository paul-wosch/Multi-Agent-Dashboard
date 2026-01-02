# 🧪 Development Notes

This document expands developer-focused guidance present in the README.

- 🧱 **`src/` layout**:
  
  - Avoids accidental imports from the working directory
  - Always develop with `pip install -e .` for module-style runs and imports

- 🖼️ **Engine/UI separation**:
  
  - Keep Streamlit-specific code in `multi_agent_dashboard/ui`
  - Engine (`engine.py`) and services (`services.py`) should remain UI-agnostic for reuse in scripts/tests

- 🗃️ **DB access layering**:
  
  - `db/infra`: low-level connections, schema, migrations
  - `db/*.py`: DAOs for agents, pipelines, runs, metrics
  - `db/services.py`: higher-level transactional APIs used by the UI and other frontends

- 🔐 **Prompt safety & caps**:
  
  - Use `utils.safe_format` for prompt formatting
  - Respect centralized caps (`AGENT_INPUT_CAP`, `AGENT_OUTPUT_CAP`) to avoid unbounded prompts

- 🎨 **UI theming & symbols**:
  
  - Shared color and emoji schemes live in `config.py`
  - Avoid hardcoding colors/symbols in UI components

- 🧪 Tests & CI:
  
  - Unit tests are not yet implemented (help wanted). Add or extend tests for new engine, DB, or migration behavior where applicable.

## 🔁 Typical Schema-Change Flow (developer checklist)

1. Update the canonical schema: Edit `schema.py`.
2. Preview changes (dry-run): `python -m multi_agent_dashboard.db.infra.generate_migration my_change --dry-run`
3. Generate migration SQL: `python -m multi_agent_dashboard.db.infra.generate_migration my_change`
4. Apply migrations: Start the app (or run a script which calls `init_db`) to apply migrations automatically.
5. If migration files end with `_REQUIRES_REBUILD` and your DB is non-empty:
   - Run `sqlite_rebuild.py` with `--all-with-diffs` after taking backups; use `--dry-run` first.

Note about `sqlite_rebuild.py` invocation and the internal generator subprocess: see docs/MIGRATIONS.md for details and caveats.

---

## 🔧 Status & Known Gaps

- Unit tests: not yet implemented.
- CHANGELOG: not currently maintained — a `CHANGELOG.md` would be a helpful addition for releases.
- CI: add checks for linting and tests once a test suite exists.

---

If you are working on a larger change touching schema, migrations, engine behavior, or persistence, please document the intended migration plan in your PR and include sample DB copies where appropriate for reviewer testing.
