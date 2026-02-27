-- MIGRATION-META: {"id":"020_backfill_max_output_effective_null","created_at":"2026-02-27T16:15:00Z","safe_sql":["UPDATE agent_run_configs SET max_output_effective = NULL WHERE max_output_effective = 0;"],"generator_options":{"allow_drop_table":false,"allow_drop_column":false,"assume_rename_table":[],"assume_rename_column":[],"sqlite_version":"3.51.0"}}

-- auto-generated migration (hand-crafted to backfill max_output_effective NULL for historical runs)

-- Backfill agent_run_configs table: convert default 0 values to NULL for historical runs
-- (runs that existed before the per‑agent max‑output feature was implemented).
-- This distinguishes "no value recorded" (NULL) from "computed as no limit" (0).

UPDATE agent_run_configs
SET max_output_effective = NULL
WHERE max_output_effective IS 0;