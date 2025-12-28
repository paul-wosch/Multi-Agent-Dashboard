-- auto-generated migration

CREATE TABLE agent_run_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    agent_name TEXT,
    model TEXT,
    prompt_template TEXT,
    role TEXT,
    input_vars TEXT,
    output_vars TEXT,
    tools_json TEXT,
    tools_config_json TEXT,
    reasoning_effort TEXT,
    reasoning_summary TEXT,
    reasoning_config_json TEXT,
    extra_config_json TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
    FOREIGN KEY(agent_name) REFERENCES agents(agent_name)
);