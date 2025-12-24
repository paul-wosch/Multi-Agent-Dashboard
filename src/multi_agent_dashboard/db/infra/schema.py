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
        # TODO: debug migration workflow
        #  1. add missing parent key "constraints"
        #  2. run generate_migrations with --dry-run --enable-constraints
        #  3. it should show table requires rebuild
        #     ...to add foreign keys even if FKs are already present in DB
        #  4. debug generate_migrations
        #  5. for cases which require table rebuild make sqlite_rebuild user friendly
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
        # TODO: Optional but future-safe
        #   agent_name could reference agents(agent_name)
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

    "agent_metrics": {
        "columns": {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "run_id": "INTEGER",
            "agent_name": "TEXT",
            "input_tokens": "INTEGER",
            "output_tokens": "INTEGER",
            "latency": "REAL",
            "cost": "REAL",
        },
        "constraints": {
            "foreign_keys": [
                {
                    "column": "run_id",
                    "references": "runs(id)",
                    "on_delete": "CASCADE",
                }
            ],
        }
    },
}
