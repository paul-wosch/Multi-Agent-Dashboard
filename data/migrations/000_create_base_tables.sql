-- Base schema (applied once)

CREATE TABLE IF NOT EXISTS migrations (
    id TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    task_input TEXT,
    final_output TEXT
);

CREATE TABLE IF NOT EXISTS agent_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    agent_name TEXT,
    output TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS agent_prompt_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT,
    version INTEGER,
    prompt TEXT,
    metadata_json TEXT,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS agents (
    agent_name TEXT PRIMARY KEY,
    model TEXT,
    prompt_template TEXT,
    role TEXT,
    input_vars TEXT,
    output_vars TEXT
);

CREATE TABLE IF NOT EXISTS pipelines (
    pipeline_name TEXT PRIMARY KEY,
    steps_json TEXT,
    metadata_json TEXT,
    timestamp TEXT
);