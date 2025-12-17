# schema.py

SCHEMA = {
    "migrations": {
        "columns": {
            "id": "TEXT PRIMARY KEY",
            "applied_at": "TEXT NOT NULL",
        },
    },

    "runs": {
        "columns": {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "timestamp": "TEXT",
            "task_input": "TEXT",
            "final_output": "TEXT",
            "final_is_json": "INTEGER DEFAULT 0",
            "final_model": "TEXT",
        },
    },

    "agent_outputs": {
        "columns": {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "run_id": "INTEGER",
            "agent_name": "TEXT",
            "output": "TEXT",
            "is_json": "INTEGER DEFAULT 0",
            "model": "TEXT",
        },
        "foreign_keys": [
            {
                "column": "run_id",
                "references": "runs(id)",
                "on_delete": "CASCADE",
            }
        ],
    },

    "agent_prompt_versions": {
        "columns": {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "agent_name": "TEXT",
            "version": "INTEGER",
            "prompt": "TEXT",
            "metadata_json": "TEXT",
            "timestamp": "TEXT",
        },
        # Optional but future-safe:
        # agent_name could reference agents(agent_name)
    },

    "agents": {
        "columns": {
            "agent_name": "TEXT PRIMARY KEY",
            "model": "TEXT",
            "prompt_template": "TEXT",
            "role": "TEXT",
            "input_vars": "TEXT",
            "output_vars": "TEXT",
        },
    },

    "pipelines": {
        "columns": {
            "pipeline_name": "TEXT PRIMARY KEY",
            "steps_json": "TEXT",
            "metadata_json": "TEXT",
            "timestamp": "TEXT",
        },
    },
}
