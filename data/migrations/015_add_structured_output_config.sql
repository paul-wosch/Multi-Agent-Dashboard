-- MIGRATION-META: {"id":"015_add_structured_output_config","created_at":"2026-01-24T00:00:00Z","safe_sql":["ALTER TABLE \"agents\" ADD COLUMN \"structured_output_enabled\" INTEGER DEFAULT 0;","ALTER TABLE \"agents\" ADD COLUMN \"schema_json\" TEXT;","ALTER TABLE \"agents\" ADD COLUMN \"schema_name\" TEXT;","ALTER TABLE \"agents\" ADD COLUMN \"temperature\" REAL;","ALTER TABLE \"agent_run_configs\" ADD COLUMN \"structured_output_enabled\" INTEGER DEFAULT 0;","ALTER TABLE \"agent_run_configs\" ADD COLUMN \"schema_json\" TEXT;","ALTER TABLE \"agent_run_configs\" ADD COLUMN \"schema_name\" TEXT;","ALTER TABLE \"agent_run_configs\" ADD COLUMN \"temperature\" REAL;"],"generator_options":{"allow_drop_table":false,"allow_drop_column":false,"assume_rename_table":[],"assume_rename_column":[],"sqlite_version":"3.51.0"}}

-- auto-generated migration

ALTER TABLE "agents" ADD COLUMN "structured_output_enabled" INTEGER DEFAULT 0;
ALTER TABLE "agents" ADD COLUMN "schema_json" TEXT;
ALTER TABLE "agents" ADD COLUMN "schema_name" TEXT;
ALTER TABLE "agents" ADD COLUMN "temperature" REAL;

ALTER TABLE "agent_run_configs" ADD COLUMN "structured_output_enabled" INTEGER DEFAULT 0;
ALTER TABLE "agent_run_configs" ADD COLUMN "schema_json" TEXT;
ALTER TABLE "agent_run_configs" ADD COLUMN "schema_name" TEXT;
ALTER TABLE "agent_run_configs" ADD COLUMN "temperature" REAL;
