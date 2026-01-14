-- MIGRATION-META: {"created_at":"2026-01-14T15:15:13.099059+00:00","diff":{"columns":{"agent_metrics":{"added":[],"removed":[],"renamed":[],"type_changed":[]},"agent_outputs":{"added":[],"removed":[],"renamed":[],"type_changed":[]},"agent_run_configs":{"added":["endpoint","model_class","provider_features_json","provider_id","use_responses_api"],"removed":[],"renamed":[],"type_changed":[]},"agent_snapshots":{"added":[],"removed":[],"renamed":[],"type_changed":[]},"agents":{"added":["endpoint","model_class","provider_features_json","provider_id","use_responses_api"],"removed":[],"renamed":[],"type_changed":[]},"pipelines":{"added":[],"removed":[],"renamed":[],"type_changed":[]},"runs":{"added":[],"removed":[],"renamed":[],"type_changed":[]},"tool_usages":{"added":[],"removed":[],"renamed":[],"type_changed":[]}},"foreign_keys":{},"indexes":{},"tables":{"added":[],"removed":[],"renamed":[]},"triggers":{},"views":{"both":[],"db_only":[],"schema_only":[]}},"generator_options":{"allow_drop_column":false,"allow_drop_table":false,"assume_rename_column":[],"assume_rename_table":[],"sqlite_capabilities":{"added_constraint_testing":true,"drop_column":true,"rename_column":true},"sqlite_version":"3.51.0"},"id":"013_add_provider_metadata","safe_sql":["ALTER TABLE \"agents\" ADD COLUMN \"endpoint\" TEXT;","ALTER TABLE \"agents\" ADD COLUMN \"model_class\" TEXT;","ALTER TABLE \"agents\" ADD COLUMN \"provider_features_json\" TEXT;","ALTER TABLE \"agents\" ADD COLUMN \"provider_id\" TEXT;","ALTER TABLE \"agents\" ADD COLUMN \"use_responses_api\" INTEGER DEFAULT 0;","ALTER TABLE \"agent_run_configs\" ADD COLUMN \"endpoint\" TEXT;","ALTER TABLE \"agent_run_configs\" ADD COLUMN \"model_class\" TEXT;","ALTER TABLE \"agent_run_configs\" ADD COLUMN \"provider_features_json\" TEXT;","ALTER TABLE \"agent_run_configs\" ADD COLUMN \"provider_id\" TEXT;","ALTER TABLE \"agent_run_configs\" ADD COLUMN \"use_responses_api\" INTEGER DEFAULT 0;"]}

-- auto-generated migration

ALTER TABLE "agents" ADD COLUMN "provider_id" TEXT;
ALTER TABLE "agents" ADD COLUMN "model_class" TEXT;
ALTER TABLE "agents" ADD COLUMN "endpoint" TEXT;
ALTER TABLE "agents" ADD COLUMN "use_responses_api" INTEGER DEFAULT 1;
ALTER TABLE "agents" ADD COLUMN "provider_features_json" TEXT;

ALTER TABLE "agent_run_configs" ADD COLUMN "provider_id" TEXT;
ALTER TABLE "agent_run_configs" ADD COLUMN "model_class" TEXT;
ALTER TABLE "agent_run_configs" ADD COLUMN "endpoint" TEXT;
ALTER TABLE "agent_run_configs" ADD COLUMN "use_responses_api" INTEGER DEFAULT 1;
ALTER TABLE "agent_run_configs" ADD COLUMN "provider_features_json" TEXT;