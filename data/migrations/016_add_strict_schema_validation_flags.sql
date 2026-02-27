-- MIGRATION-META: {"id":"016_add_strict_schema_validation_flags","created_at":"2026-01-26T00:00:00Z","safe_sql":["ALTER TABLE runs ADD COLUMN strict_schema_exit INTEGER DEFAULT 0;","ALTER TABLE agent_outputs ADD COLUMN schema_validation_failed INTEGER DEFAULT 0;","ALTER TABLE agent_run_configs ADD COLUMN strict_schema_validation INTEGER DEFAULT 0;"],"generator_options":{"allow_drop_table":false,"allow_drop_column":false,"assume_rename_table":[],"assume_rename_column":[],"sqlite_version":"3.51.0"}}

-- Track early exit when strict schema validation halts the pipeline
ALTER TABLE runs ADD COLUMN strict_schema_exit INTEGER DEFAULT 0;

-- Track per-agent validation failure
ALTER TABLE agent_outputs ADD COLUMN schema_validation_failed INTEGER DEFAULT 0;

-- Capture run-level strict schema validation toggle in per-agent config snapshots
ALTER TABLE agent_run_configs ADD COLUMN strict_schema_validation INTEGER DEFAULT 0;
