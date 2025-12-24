-- auto-generated migration

CREATE TABLE agent_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    agent_name TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    latency REAL,
    cost REAL,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);