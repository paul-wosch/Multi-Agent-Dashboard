-- MIGRATION-META: {"id":"014_backfill_legacy_agent_providers","created_at":"2026-01-17T00:00:00Z","safe_sql":["UPDATE agents SET provider_id = 'openai', use_responses_api = 1, provider_features_json = '{\"structured_output\": true, \"tool_calling\": true, \"reasoning\": true}' WHERE provider_id IS NULL OR TRIM(provider_id) = '';","UPDATE agent_run_configs SET provider_id = 'openai', use_responses_api = 1, provider_features_json = '{\"structured_output\": true, \"tool_calling\": true, \"reasoning\": true}' WHERE provider_id IS NULL OR TRIM(provider_id) = '';"],"generator_options":{"allow_drop_table":false,"allow_drop_column":false,"assume_rename_table":[],"assume_rename_column":[],"sqlite_version":"3.51.0"}}

-- auto-generated migration (hand-crafted to backfill provider metadata)

-- Backfill agents table with reasonable provider defaults for legacy agents.
-- This keeps existing agents working after removing legacy OpenAI-only paths.
UPDATE agents
SET provider_id = 'openai',
    use_responses_api = 1,
    provider_features_json = '{"structured_output": true, "tool_calling": true, "reasoning": true}'
WHERE provider_id IS NULL OR TRIM(provider_id) = '';

-- Backfill agent_run_configs so historical run snapshots are reproducible.
UPDATE agent_run_configs
SET provider_id = 'openai',
    use_responses_api = 1,
    provider_features_json = '{"structured_output": true, "tool_calling": true, "reasoning": true}'
WHERE provider_id IS NULL OR TRIM(provider_id) = '';