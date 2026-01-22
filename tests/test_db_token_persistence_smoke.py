# tests/test_db_token_persistence_smoke.py
import sqlite3
from pathlib import Path

from multi_agent_dashboard.db.services import RunService


def test_db_persists_agent_token_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "token_persist.db"

    schema_sql = r"""
    CREATE TABLE migrations (
        id TEXT PRIMARY KEY,
        applied_at TEXT NOT NULL
    );

    CREATE TABLE runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        task_input TEXT,
        final_output TEXT,
        final_is_json INTEGER DEFAULT 0,
        final_model TEXT
    );

    CREATE TABLE agent_outputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        agent_name TEXT,
        output TEXT,
        is_json INTEGER DEFAULT 0,
        model TEXT,
        FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
    );

    CREATE TABLE agent_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        agent_name TEXT,
        input_tokens INTEGER,
        output_tokens INTEGER,
        latency REAL,
        input_cost REAL,
        output_cost REAL,
        cost REAL,
        FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
    );

    CREATE TABLE agent_run_configs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        agent_name TEXT,
        model TEXT,
        provider_id TEXT,
        model_class TEXT,
        endpoint TEXT,
        use_responses_api INTEGER DEFAULT 1,
        provider_features_json TEXT,
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
        system_prompt_template TEXT,
        FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
    );

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
    """

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()

    run_svc = RunService(str(db_path))

    agent_metrics = {
        "agent_one": {
            "agent_name": "agent_one",
            "input_tokens": 123,
            "output_tokens": 456,
            "latency": 0.1,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "cost": 0.0,
        }
    }

    run_id = run_svc.save_run(
        task_input="token test",
        final_output="ok",
        memory={"agent_one": "ok"},
        agent_models={"agent_one": "gpt-test"},
        agent_configs={
            "agent_one": {
                "model": "gpt-test",
                "provider_id": "openai",
                "model_class": "gpt-test",
                "endpoint": None,
                "use_responses_api": True,
                "provider_features": {},
                "prompt_template": "Do X",
                "role": "worker",
                "input_vars": [],
                "output_vars": [],
                "tools": {"enabled": False, "tools": []},
                "tools_config": None,
                "reasoning_config": None,
                "extra": {},
                "system_prompt_template": None,
            }
        },
        agent_metrics=agent_metrics,
        tool_usages={},
    )

    assert isinstance(run_id, int) and run_id > 0

    conn2 = sqlite3.connect(str(db_path))
    try:
        cur = conn2.cursor()
        cur.execute(
            "SELECT input_tokens, output_tokens FROM agent_metrics WHERE run_id = ? AND agent_name = ?",
            (run_id, "agent_one"),
        )
        row = cur.fetchone()
        assert row == (123, 456)
    finally:
        conn2.close()
