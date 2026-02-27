-- MIGRATION-META: {"created_at":"2026-02-27T11:06:34.262806+00:00","diff":{"columns":{"agent_metrics":{"added":[],"removed":[],"renamed":[],"type_changed":[]},"agent_outputs":{"added":[],"removed":[],"renamed":[],"type_changed":[]},"agent_run_configs":{"added":["max_output","max_output_effective"],"removed":[],"renamed":[],"type_changed":[]},"agent_snapshots":{"added":[],"removed":[],"renamed":[],"type_changed":[]},"agents":{"added":["max_output"],"removed":[],"renamed":[],"type_changed":[]},"pipelines":{"added":[],"removed":[],"renamed":[],"type_changed":[]},"runs":{"added":[],"removed":[],"renamed":[],"type_changed":[]},"tool_usages":{"added":[],"removed":[],"renamed":[],"type_changed":[]}},"foreign_keys":{},"indexes":{},"tables":{"added":[],"removed":[],"renamed":[]},"triggers":{},"views":{"both":[],"db_only":[],"schema_only":[]}},"generator_options":{"allow_drop_column":false,"allow_drop_table":false,"assume_rename_column":[],"assume_rename_table":[],"sqlite_capabilities":{"added_constraint_testing":true,"drop_column":true,"rename_column":true},"sqlite_version":"3.51.0"},"id":"018_add_agent_max_output_columns","safe_sql":["ALTER TABLE \"agents\" ADD COLUMN \"max_output\" INTEGER DEFAULT 0;","ALTER TABLE \"agent_run_configs\" ADD COLUMN \"max_output\" INTEGER DEFAULT 0;","ALTER TABLE \"agent_run_configs\" ADD COLUMN \"max_output_effective\" INTEGER DEFAULT 0;"]}

-- auto-generated migration

ALTER TABLE "agents" ADD COLUMN "max_output" INTEGER DEFAULT 0;

ALTER TABLE "agent_run_configs" ADD COLUMN "max_output" INTEGER DEFAULT 0;

ALTER TABLE "agent_run_configs" ADD COLUMN "max_output_effective" INTEGER DEFAULT 0;

