# tests/test_provider_parity_and_persistence.py
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any

import pytest

from multi_agent_dashboard.engine import MultiAgentEngine
from multi_agent_dashboard.models import AgentSpec
from multi_agent_dashboard.llm_client import TextResponse
from multi_agent_dashboard.db.services import AgentService, RunService
from multi_agent_dashboard.config import OPENAI_PRICING


# -----------------------
# Helpers / Dummy clients
# -----------------------

class DummyLangChainLLMClient:
    """
    Simulates a LangChain-style client (agent.invoke path).
    The engine's AgentRuntime will take the LangChain path when llm_client._langchain_available is True.
    """
    _langchain_available = True

    def create_agent_for_spec(self, spec, **kwargs):
        # Return a very small fake agent object that carries system_prompt
        class FakeAgent:
            def __init__(self, sys):
                self.system_prompt = sys

        return FakeAgent(getattr(spec, "system_prompt_template", "") or "")

    def invoke_agent(self, agent, prompt, **kwargs):
        # Return content mimicking content_blocks + detected provider profile + structured_response
        return TextResponse(
            text='{"answer":"ok"}',
            raw={
                "_multi_agent_dashboard_events": [
                    {
                        "content_blocks": [
                            {
                                "type": "web_search_call",
                                "name": "web_search",
                                "args": {"query": "instrumentize"},
                                "id": "call-lc-1",
                                "status": "ok",
                            }
                        ],
                        "structured_response": {"answer": "ok"},
                    }
                ],
                "detected_provider_profile": {
                    "tool_calling": True,
                    "structured_output": True,
                    "max_input_tokens": 131072,
                },
                "structured_response": {"answer": "ok"},
            },
            input_tokens=4,
            output_tokens=6,
            latency=0.03,
        )

    def create_text_response(self, *args, **kwargs):
        raise RuntimeError("Legacy path should not be called for langchain-style client")


class DummyLegacyLLMClient:
    """
    Simulates the legacy OpenAI Responses-style client (create_text_response path).
    The engine's AgentRuntime will take the legacy path when llm_client._langchain_available is False.
    """
    _langchain_available = False

    def create_text_response(self, *args, **kwargs):
        # Return TextResponse shaped like a Responses API result (including instrumentation events)
        return TextResponse(
            text='{"answer":"ok"}',
            raw={
                "_multi_agent_dashboard_events": [
                    {
                        "content_blocks": [
                            {
                                "type": "web_search_call",
                                "name": "web_search",
                                "args": {"query": "instrumentize"},
                                "id": "call-legacy-1",
                                "status": "ok",
                            }
                        ],
                        "structured_response": {"answer": "ok"},
                    }
                ],
                "detected_provider_profile": {
                    "tool_calling": True,
                    "structured_output": True,
                    "max_input_tokens": 131072,
                },
                "structured_response": {"answer": "ok"},
            },
            input_tokens=3,
            output_tokens=5,
            latency=0.02,
        )

    # Not necessary for this test, but kept for completeness
    def create_agent_for_spec(self, *args, **kwargs):
        raise RuntimeError("LangChain agent creation should not be used for legacy client")


# -----------------------
# Tests
# -----------------------

def _make_test_agent_spec(name: str) -> AgentSpec:
    return AgentSpec(
        name=name,
        model="gpt-test",
        prompt_template="Do {task}",
        role="tester",
        input_vars=["task"],
        output_vars=["answer"],
        tools={"enabled": True, "tools": ["web_search"]},
        reasoning_effort="medium",
        reasoning_summary="auto",
    )


def test_legacy_and_langchain_parity_content_blocks_and_profile():
    """
    Ensure an engine run using the legacy create_text_response path and an engine
    run using the LangChain invoke path both produce instrumentation (content_blocks)
    and provider profile hints in the same places in the engine's agent_configs.
    """

    spec_name = "tester"

    # LangChain-style run
    client_lc = DummyLangChainLLMClient()
    engine_lc = MultiAgentEngine(llm_client=client_lc)
    spec = _make_test_agent_spec(spec_name)
    engine_lc.add_agent(spec)

    res_lc = engine_lc.run_seq(steps=[spec_name], initial_input="capture", strict=False)

    # Legacy (Responses API) run
    client_legacy = DummyLegacyLLMClient()
    engine_leg = MultiAgentEngine(llm_client=client_legacy)
    spec2 = _make_test_agent_spec(spec_name)  # same spec; name may be same across engines
    engine_leg.add_agent(spec2)

    res_leg = engine_leg.run_seq(steps=[spec_name], initial_input="capture", strict=False)

    # Sanity checks
    assert spec_name in res_lc.agent_configs, "LangChain run must record agent config"
    assert spec_name in res_leg.agent_configs, "Legacy run must record agent config"

    cfg_lc = res_lc.agent_configs[spec_name]
    cfg_leg = res_leg.agent_configs[spec_name]

    # Extra payload should include content_blocks (full) or content_blocks_summary in either case
    extra_lc = cfg_lc.get("extra", {}) or {}
    extra_leg = cfg_leg.get("extra", {}) or {}

    def has_content_info(extra: Dict[str, Any]) -> bool:
        return bool(extra.get("content_blocks")) or bool(extra.get("content_blocks_summary")) or bool(extra.get("instrumentation_events"))

    assert has_content_info(extra_lc), "LangChain run did not record content_blocks/instrumentation"
    assert has_content_info(extra_leg), "Legacy run did not record content_blocks/instrumentation"

    # Provider profile detection (engine attempts to derive provider_features when profile present)
    pf_lc = cfg_lc.get("provider_features") or {}
    pf_leg = cfg_leg.get("provider_features") or {}

    # Either derived hints (tool_calling/structured_output) or a detected_profile_present marker should be present
    def profile_has_hints(pf: Dict[str, Any]) -> bool:
        return (
            (pf.get("tool_calling") is True and pf.get("structured_output") is True)
            or ("detected_profile_present" in pf)
            or (pf.get("structured_output") is True)
        )

    assert profile_has_hints(pf_lc) or profile_has_hints(pf_leg), "At least one run should expose provider hints"

    # Cross-compare parity: both runs should have at least the same "presence" of content_blocks and profile detection
    assert has_content_info(extra_lc) == has_content_info(extra_leg), "Content capture presence should match between LangChain and legacy flows"
    assert bool(pf_lc) == bool(pf_leg), "Provider features presence should match across flows"


def test_persistence_and_cost_across_providers(tmp_path: Path):
    """
    Integration: create a temporary SQLite DB, register two agents (OpenAI and Ollama)
    and persist a run containing per-agent metrics. Verify:
      - agent_run_configs captured provider metadata
      - agent_metrics persisted cost matches expected (OpenAI cost per OPENAI_PRICING; Ollama cost = 0)
    """
    db_path = tmp_path / "test_runs.db"

    # Minimal SQL to create the tables used by the services/run persistence code.
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

    # Initialize DB file and create tables
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()

    # Services pointing at this test DB
    agent_svc = AgentService(str(db_path))
    run_svc = RunService(str(db_path))

    # Save two agents: one OpenAI-backed, one Ollama-backed
    # OpenAI agent (has pricing)
    agent_svc.save_agent_atomic(
        name="openai_agent",
        model="gpt-4.1-nano",
        prompt="Do X",
        role="worker",
        input_vars=["task"],
        output_vars=["answer"],
        color=None,
        symbol=None,
        provider_id="openai",
        model_class="gpt-4.1-nano",
        endpoint=None,
        use_responses_api=True,
        provider_features={"structured_output": True},
    )

    # Ollama agent (on-prem; costs should not be attributed using OPENAI_PRICING)
    agent_svc.save_agent_atomic(
        name="ollama_agent",
        model="llama3",
        prompt="Do Y",
        role="worker",
        input_vars=["task"],
        output_vars=["answer"],
        color=None,
        symbol=None,
        provider_id="ollama",
        model_class="ollama-mini",
        endpoint="http://localhost:11434",
        use_responses_api=False,
        provider_features={"structured_output": False},
    )

    # Construct a "run" to persist: per-agent metrics (input/output tokens)
    # We'll pick token counts and compute expected cost for OpenAI model using OPENAI_PRICING
    input_tokens_openai = 10000
    output_tokens_openai = 20000

    input_tokens_ollama = 10000
    output_tokens_ollama = 20000

    # Compute expected OpenAI cost using the same simple formula as engine._compute_cost
    # (price per 1M tokens -> cost = tokens/1_000_000 * price)
    pricing = OPENAI_PRICING.get("gpt-4.1-nano")
    assert pricing is not None, "Test assumes 'gpt-4.1-nano' pricing present in OPENAI_PRICING"

    expected_input_cost_openai = input_tokens_openai / 1_000_000.0 * pricing["input"]
    expected_output_cost_openai = output_tokens_openai / 1_000_000.0 * pricing["output"]
    expected_total_openai = expected_input_cost_openai + expected_output_cost_openai

    # Ollama costs are expected to be zero in our cost logic
    expected_total_ollama = 0.0

    # Build agent_metrics mapping expected by RunDAO.save
    agent_metrics = {
        "openai_agent": {
            "agent_name": "openai_agent",
            "input_tokens": input_tokens_openai,
            "output_tokens": output_tokens_openai,
            "latency": 0.5,
            "input_cost": expected_input_cost_openai,
            "output_cost": expected_output_cost_openai,
            "cost": expected_total_openai,
        },
        "ollama_agent": {
            "agent_name": "ollama_agent",
            "input_tokens": input_tokens_ollama,
            "output_tokens": output_tokens_ollama,
            "latency": 0.7,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "cost": expected_total_ollama,
        },
    }

    # Agent configs (snapshot) that will be persisted into agent_run_configs
    agent_configs = {
        "openai_agent": {
            "model": "gpt-4.1-nano",
            "provider_id": "openai",
            "model_class": "gpt-4.1-nano",
            "endpoint": None,
            "use_responses_api": True,
            "provider_features": {"structured_output": True},
            "prompt_template": "Do X",
            "role": "worker",
            "input_vars": ["task"],
            "output_vars": ["answer"],
            "tools": {"enabled": True, "tools": ["web_search"]},
            "tools_config": None,
            "reasoning_config": None,
            "extra": {},
            "system_prompt_template": None,
        },
        "ollama_agent": {
            "model": "llama3",
            "provider_id": "ollama",
            "model_class": "ollama-mini",
            "endpoint": "http://localhost:11434",
            "use_responses_api": False,
            "provider_features": {"structured_output": False},
            "prompt_template": "Do Y",
            "role": "worker",
            "input_vars": ["task"],
            "output_vars": ["answer"],
            "tools": {"enabled": False, "tools": []},
            "tools_config": None,
            "reasoning_config": None,
            "extra": {},
            "system_prompt_template": None,
        },
    }

    # Minimal memory dict for outputs
    memory = {"openai_agent": "ok", "ollama_agent": "ok"}

    # Persist run using RunService (this uses RunDAO.save under the hood)
    run_id = run_svc.save_run(
        task_input="integration test",
        final_output="ok",
        memory=memory,
        agent_models={"openai_agent": "gpt-4.1-nano", "ollama_agent": "llama3"},
        final_model=None,
        agent_configs=agent_configs,
        agent_metrics=agent_metrics,
        tool_usages={},
    )

    assert isinstance(run_id, int) and run_id > 0

    # Query persisted total cost via RunService (wraps RunDAO.get_cost_total_for_period)
    total_all_time = run_svc.get_cost_total_for_period(period="total")

    # Expected total cost equals OpenAI total only (ollama cost is 0)
    assert pytest.approx(total_all_time, rel=1e-9) == pytest.approx(expected_total_openai, rel=1e-9)

    # Also verify provider metadata persisted into agent_run_configs (DB-level)
    conn2 = sqlite3.connect(str(db_path))
    try:
        cur = conn2.cursor()
        cur.execute("SELECT agent_name, provider_id, model_class, endpoint, use_responses_api, provider_features_json FROM agent_run_configs WHERE run_id = ?", (run_id,))
        rows = cur.fetchall()
        assert rows, "agent_run_configs rows should be present for the run"

        by_agent = {r[0]: r for r in rows}
        assert "openai_agent" in by_agent and "ollama_agent" in by_agent

        # Validate openai_agent persisted provider_id and model_class
        oa = by_agent["openai_agent"]
        assert oa[1] == "openai"
        assert oa[2] == "gpt-4.1-nano"

        # Validate ollama_agent persisted provider_id and endpoint
        ol = by_agent["ollama_agent"]
        assert ol[1] == "ollama"
        assert ol[3] == "http://localhost:11434"

        # provider_features_json should be valid JSON
        pf_json_openai = oa[5]
        pf_openai = json.loads(pf_json_openai or "{}")
        assert pf_openai.get("structured_output") is True

    finally:
        conn2.close()
