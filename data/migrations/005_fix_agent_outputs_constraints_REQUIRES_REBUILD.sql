-- MIGRATION-META: {"author_notes":"Auto-annotated from legacy comments. PLEASE REVIEW and adjust rebuild_defs before applying.","created_at":"2026-01-09T16:41:35.982168+00:00","id":"005_fix_agent_outputs_constraints_REQUIRES_REBUILD","legacy_inferred":true,"rebuild":{"batch":true,"rebuild_defs":{"agent_outputs":{"columns":{"agent_name":"TEXT","id":"INTEGER PRIMARY KEY AUTOINCREMENT","is_json":"INTEGER DEFAULT 0","model":"TEXT","output":"TEXT","run_id":"INTEGER"},"constraints":{"foreign_keys":[{"column":"run_id","on_delete":"CASCADE","references":"runs(id)"}]}}},"requires_rebuild":["agent_outputs"]}}

-- auto-generated migration

-- NOTE: agent_outputs requires rebuild to change foreign keys
--   Foreign keys to CHANGE (DB -> schema):
--     FOREIGN KEY(run_id) REFERENCES runs(id)  ==>  FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
--   Use sqlite_rebuild.py to rebuild this table with constraints from schema.py
--   e.g.: python sqlite_rebuild.py --all-with-diffs