# schema.py

SCHEMA = {
    "migrations": {
        "id": "TEXT PRIMARY KEY",
        "applied_at": "TEXT NOT NULL",
    },

    "runs": {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "timestamp": "TEXT",
        "task_input": "TEXT",
        "final_output": "TEXT",
    },

    "agent_outputs": {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "run_id": "INTEGER",
        "agent_name": "TEXT",
        "output": "TEXT",
    },

    "agent_prompt_versions": {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "agent_name": "TEXT",
        "version": "INTEGER",
        "prompt": "TEXT",
        "metadata_json": "TEXT",
        "timestamp": "TEXT",
    },

    "agents": {
        "agent_name": "TEXT PRIMARY KEY",
        "model": "TEXT",
        "temperature": "FLOAT",
        "prompt_template": "TEXT",
        "role": "TEXT",
        "input_vars": "TEXT",
        "output_vars": "TEXT",
    },

    "pipelines": {
        "pipeline_name": "TEXT PRIMARY KEY",
        "steps_json": "TEXT",
        "metadata_json": "TEXT",
        "timestamp": "TEXT",
    },
}