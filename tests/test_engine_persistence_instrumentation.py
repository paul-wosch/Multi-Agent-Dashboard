# tests/test_engine_persistence_instrumentation.py
import json
import sqlite3
from pathlib import Path

from multi_agent_dashboard.engine import MultiAgentEngine
from multi_agent_dashboard.models import AgentSpec
from multi_agent_dashboard.llm_client import TextResponse
from multi_agent_dashboard.db.services import AgentService, RunService
import pytest


class DummyLangChainLLMClientForPersistence:
    _langchain_available = True

    def create_agent_for_spec(self, spec, **kwargs):
        class FakeAgent:
            def __init__(self, system_prompt):
                self.system_prompt = system_prompt
        return FakeAgent(getattr(spec, "system_prompt_template", "") or "")

    def invoke_agent(self, agent, prompt, **kwargs):
        block = {
            "type": "web_search_call",
            "name": "web_search",
            "args": {"query": "persist-me"},
            "id": "call-persist-1",
            "status": "ok",
        }
        events = [
            {
                "content_blocks": [block],
                "structured_response": {"answer": "persisted"},
                "text": "persisted text",
                "ts": 1.2345,
            }
        ]

        raw = {
            "agent_response": {
                "_multi_agent_dashboard_events": events,
                "instrumentation_events": events,
                "content_blocks": [block],
                "output": [
                    {
                        "response": {
                            "usage": {"prompt_tokens": 10, "completion_tokens": 15},
                            "content_blocks": [block],
                            "structured_response": {"answer": "persisted"},
                        }
                    }
                ],
            },
            "detected_provider_profile": {
                "tool_calling": True,
                "structured_output": True,
                "max_input_tokens": 65536,
            },
            "structured_response": {"answer": "persisted"},
        }

        return TextResponse(
            text='{"answer":"persisted"}',
            raw=raw,
            input_tokens=10,
            output_tokens=15,
            latency=0.01,
        )

    def create_text_response(self, *args, **kwargs):
        raise RuntimeError("Legacy path should not be called in this test")


def test_engine_run_and_persists_content_blocks_and_instrumentation(tmp_path: Path):
    """
    End-to-end style test:
    - create a temporary SQLite DB with minimal schema
    - register an agent via AgentService
    - run the engine using a Dummy LangChain client that returns instrumentation
    - persist the EngineResult through RunService.save_run
    - verify agent_run_configs.extra_config_json contains content_blocks and instrumentation_events
    """

    db_path = tmp_path / "test_runs_engine.db"

    # Minimal set of tables required by RunDAO.save (same as other tests)
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

    CREATE TABLE agent_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_name TEXT,
        version INTEGER,
        snapshot_json TEXT,
        metadata_json TEXT,
        is_auto INTEGER DEFAULT 0,
        created_at TEXT
    );

    CREATE TABLE agents (
        agent_name TEXT PRIMARY KEY,
        model TEXT,
        prompt_template TEXT,
        role TEXT,
        input_vars TEXT,
        output_vars TEXT,
        color TEXT,
        symbol TEXT,
        provider_id TEXT,
        model_class TEXT,
        endpoint TEXT,
        use_responses_api INTEGER DEFAULT 1,
        provider_features_json TEXT,
        tools_json TEXT,
        reasoning_effort TEXT,
        reasoning_summary TEXT,
        system_prompt_template TEXT
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
        FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
        FOREIGN KEY(agent_name) REFERENCES agents(agent_name)
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

    # Services target the test DB
    agent_svc = AgentService(str(db_path))
    run_svc = RunService(str(db_path))

    # Save an agent so the FK in agent_run_configs (if enforced) is satisfied
    agent_svc.save_agent_atomic(
        name="persist_agent",
        model="gpt-test",
        prompt="Run X",
        role="tester",
        input_vars=["task"],
        output_vars=["answer"],
        provider_id="openai",
        model_class="gpt-test",
        endpoint=None,
        use_responses_api=True,
        provider_features={"structured_output": True},
    )

    # Prepare engine with Dummy LangChain-style client
    client = DummyLangChainLLMClientForPersistence()
    engine = MultiAgentEngine(llm_client=client)

    # AgentSpec used by engine must match the agent name we persisted above
    spec = AgentSpec(
        name="persist_agent",
        model="gpt-test",
        prompt_template="Do {task}",
        role="tester",
        input_vars=["task"],
        output_vars=["answer"],
        tools={"enabled": True, "tools": ["web_search"]},
        reasoning_effort="medium",
        reasoning_summary="auto",
        provider_id="openai",
        model_class="gpt-test",
        endpoint=None,
        use_responses_api=True,
        provider_features={"structured_output": True},
    )
    engine.add_agent(spec)

    # Run the engine sequence (LangChain only path will be used)
    result = engine.run_seq(steps=["persist_agent"], initial_input="capture", strict=False)

    # Persist the run via RunService (mirrors UI flow)
    run_id = run_svc.save_run(
        "capture",
        result.final_output or "",
        result.memory,
        agent_models={k: v.spec.model for k, v in engine.agents.items()},
        final_model=None,
        agent_configs=result.agent_configs,
        agent_metrics=result.agent_metrics,
        tool_usages=result.tool_usages,
    )

    assert isinstance(run_id, int) and run_id > 0

    # Query the persisted agent_run_configs.extra_config_json to ensure instrumentation persisted
    conn2 = sqlite3.connect(str(db_path))
    try:
        cur = conn2.cursor()
        cur.execute(
            "SELECT agent_name, extra_config_json, provider_features_json FROM agent_run_configs WHERE run_id = ?",
            (run_id,),
        )
        rows = cur.fetchall()
        assert rows, "agent_run_configs rows must be present"

        row = rows[0]
        extra_json = row[1]
        pf_json = row[2]

        extra = json.loads(extra_json or "{}")
        pf = json.loads(pf_json or "{}")

        # The 'extra' JSON should include content_blocks or instrumentation_events (or both)
        assert (
            "content_blocks" in extra or "instrumentation_events" in extra or "content_blocks_summary" in extra
        ), f"extra_config_json missing expected instrumentation keys: {extra}"

        # provider_features persisted or derived should be present
        assert isinstance(pf, dict)

        cur.execute("SELECT agent_name, input_tokens, output_tokens FROM agent_metrics WHERE run_id = ?", (run_id,))
        metrics_rows = cur.fetchall()
        assert metrics_rows, "agent_metrics rows should exist"
        for agent_name, input_tokens, output_tokens in metrics_rows:
            assert input_tokens is not None and output_tokens is not None

        cur.execute(
            "SELECT agent_name, tool_type, args_json FROM tool_usages WHERE run_id = ?",
            (run_id,),
        )
        tool_rows = cur.fetchall()
        assert tool_rows, "tool_usages rows should exist when tools are invoked"
        assert any(row[1] == "web_search" for row in tool_rows), "At least one web_search tool call should be logged"
    finally:
        conn2.close()
