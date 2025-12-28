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

        "constraints": {
            "foreign_keys": [
                {
                    "column": "run_id",
                    "references": "runs(id)",
                    "on_delete": "CASCADE",
                }
            ],
        },
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
            "color": "TEXT",
            "symbol": "TEXT",
            # tool & reasoning config (backward compatible: all nullable/text)
            # JSON string with e.g. {"enabled": true, "tools": ["web_search"]}
            "tools_json": "TEXT",
            # Reasoning effort: none|low|medium|high|xhigh
            "reasoning_effort": "TEXT",
            # Reasoning summary: auto|concise|detailed|none
            "reasoning_summary": "TEXT",
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
            "input_cost": "REAL",
            "output_cost": "REAL",
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

    # Snapshot of per-agent configuration for each run.
    # Keeps tool configuration and reasoning settings separate from individual
    # tool call rows in tool_usages.
    "agent_run_configs": {
        "columns": {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "run_id": "INTEGER",
            "agent_name": "TEXT",
            "model": "TEXT",
            "prompt_template": "TEXT",
            "role": "TEXT",
            # JSON-encoded lists of variable names at the time of the run
            "input_vars": "TEXT",
            "output_vars": "TEXT",
            # High-level tools overview copied from agents.tools_json
            "tools_json": "TEXT",
            # Low-level tools configuration actually sent to the LLM
            "tools_config_json": "TEXT",
            # Reasoning configuration at the time of the run
            "reasoning_effort": "TEXT",
            "reasoning_summary": "TEXT",
            "reasoning_config_json": "TEXT",
            # Future‑proof bag for additional options (temperature, top_p, etc.)
            "extra_config_json": "TEXT",
        },
        "constraints": {
            "foreign_keys": [
                {
                    "column": "run_id",
                    "references": "runs(id)",
                    "on_delete": "CASCADE",
                },
                # Optional FK to the agents table (nullable on purpose so
                # history still works even if an agent is deleted).
                {
                    "column": "agent_name",
                    "references": "agents(agent_name)",
                },
            ],
        },
    },

    # per-agent, per-run tool usage logging for history UI
    "tool_usages": {
        "columns": {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "run_id": "INTEGER",
            "agent_name": "TEXT",
            "tool_type": "TEXT",          # e.g. "web_search"
            "tool_call_id": "TEXT",       # ws_xxx if present
            # Per-call arguments and status only. Tool and reasoning config
            # are stored once per agent/run in agent_run_configs.
            "args_json": "TEXT",
            "result_summary": "TEXT",     # short summary or empty
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