-- auto-generated migration

-- NOTE: agent_outputs requires rebuild to change foreign keys
--   Foreign keys to CHANGE (DB -> schema):
--     FOREIGN KEY(run_id) REFERENCES runs(id)  ==>  FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
--   Use sqlite_rebuild.py to rebuild this table with constraints from schema.py
--   e.g.: python sqlite_rebuild.py --all-with-diffs /Users/paul/Documents/PyCharmProjects/_AIE_TOOLS_MultiAgentDashboard/Multi-Agent-Dashboard/data/db/multi_agent_runs.db