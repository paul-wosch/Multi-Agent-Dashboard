-- auto-generated migration

ALTER TABLE agents ADD COLUMN tools_json TEXT;
ALTER TABLE agents ADD COLUMN reasoning_effort TEXT;
ALTER TABLE agents ADD COLUMN reasoning_summary TEXT;
CREATE TABLE tool_usages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    agent_name TEXT,
    tool_type TEXT,
    tool_call_id TEXT,
    args_json TEXT,
    result_summary TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);