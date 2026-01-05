-- auto-generated migration

CREATE TABLE agent_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT,
    version INTEGER,
    snapshot_json TEXT,
    metadata_json TEXT,
    is_auto INTEGER DEFAULT 0,
    created_at TEXT
);