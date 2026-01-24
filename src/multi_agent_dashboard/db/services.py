# src/multi_agent_dashboard/db/services.py
from typing import Dict, Any, List, Optional, Tuple
import json

from multi_agent_dashboard.db.runs import RunDAO, run_dao
from multi_agent_dashboard.db.agents import AgentDAO, agent_dao
from multi_agent_dashboard.db.pipelines import PipelineDAO, pipeline_dao
from multi_agent_dashboard.config import AGENT_SNAPSHOTS_AUTO, AGENT_SNAPSHOT_PRUNE_KEEP

# Runtime hooks (UI registers handlers; services call on changes)
from multi_agent_dashboard import runtime_hooks

# -----------------------
# Run Service
# -----------------------
class RunService:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def save_run(
            self,
            task_input: str,
            final_output: str,
            memory: Dict[str, Any],
            *,
            agent_models: Optional[Dict[str, str]] = None,
            final_model: Optional[str] = None,
            agent_configs: Optional[Dict[str, Dict[str, Any]]] = None,
            agent_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
            tool_usages: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> int:
        with run_dao(self.db_path) as dao:
            run_id = dao.save(
                task_input,
                final_output,
                memory,
                agent_models=agent_models,
                final_model=final_model,
                agent_configs=agent_configs,
                agent_metrics=agent_metrics,
                tool_usages=tool_usages,
            )
            return run_id

    def list_runs(self) -> List[dict]:
        return RunDAO(self.db_path).list()

    def get_run_details(
            self,
            run_id: int
    ) -> Tuple[dict | None, list[dict], list[dict], list[dict], list[dict]]:
        return RunDAO(self.db_path).get(run_id)

    # -----------------------
    # Cost helpers (exposed by service layer)
    # -----------------------
    def get_cost_total_for_period(self, period: str = "monthly") -> float:
        """
        Convenience wrapper: compute running total for the current period (monthly by default).
        """
        return RunDAO(self.db_path).get_cost_total_for_period(period=period)

    def get_costs_by_period(self, period: str = "monthly", limit: int = 12) -> List[Dict[str, Any]]:
        """
        Return a list of recent period breakdowns (period_value + total_cost).
        """
        return RunDAO(self.db_path).get_costs_by_period(period=period, limit=limit)


# -----------------------
# Agent Service
# -----------------------
class AgentService:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def list_agents(self) -> List[dict]:
        return AgentDAO(self.db_path).list()

    def save_agent(
            self,
            name: str,
            model: str,
            prompt_template: str,
            role: str = "",
            input_vars: Optional[List[str]] = None,
            output_vars: Optional[List[str]] = None,
            color: Optional[str] = None,
            symbol: Optional[str] = None,
            tools: Optional[dict] = None,
            reasoning_effort: Optional[str] = None,
            reasoning_summary: Optional[str] = None,
            system_prompt: Optional[str] = None,
            # Provider metadata
            provider_id: Optional[str] = None,
            model_class: Optional[str] = None,
            endpoint: Optional[str] = None,
            use_responses_api: bool = False,
            provider_features: Optional[dict] = None,
            # Structured output config
            structured_output_enabled: bool = False,
            schema_json: Optional[str] = None,
            schema_name: Optional[str] = None,
            temperature: Optional[float] = None,
    ) -> None:
        """
        Save an agent non-atomically. Detect provider metadata changes and, if
        changed, notify the runtime hooks so the UI/engine can reload runtimes.
        """
        dao = AgentDAO(self.db_path)
        prev = dao.get(name)

        dao.save(
            name,
            model,
            prompt_template,
            role,
            input_vars,
            output_vars,
            color,
            symbol,
            tools=tools,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            system_prompt_template=system_prompt,
            provider_id=provider_id,
            model_class=model_class,
            endpoint=endpoint,
            use_responses_api=use_responses_api,
            provider_features=provider_features,
            structured_output_enabled=structured_output_enabled,
            schema_json=schema_json,
            schema_name=schema_name,
            temperature=temperature,
        )

        # Decide whether provider metadata changed (or a new agent created).
        def _provider_key_from_row(row: Optional[dict]) -> Optional[tuple]:
            if not row:
                return None
            pf = row.get("provider_features") or {}
            try:
                pf_s = json.dumps(pf, sort_keys=True, separators=(",", ":"))
            except Exception:
                pf_s = repr(pf)
            return (
                row.get("provider_id"),
                row.get("model_class"),
                row.get("endpoint"),
                bool(row.get("use_responses_api")),
                pf_s,
            )

        new_row_like = {
            "provider_id": provider_id,
            "model_class": model_class,
            "endpoint": endpoint,
            "use_responses_api": bool(use_responses_api),
            "provider_features": provider_features or {},
        }

        prev_key = _provider_key_from_row(prev)
        new_key = _provider_key_from_row(new_row_like)

        if prev_key != new_key:
            # Notify runtime: provider metadata changed (or new agent).
            try:
                runtime_hooks.on_agent_change()
            except Exception:
                # Avoid bubbling into business logic on hook failures
                pass

    # Note: old prompt-versioning has been removed. If external code calls
    # save_prompt_version/load_prompt_versions, adapt to use snapshots instead.

    def rename_agent(self, old_name: str, new_name: str) -> None:
        AgentDAO(self.db_path).rename(old_name, new_name)

    def delete_agent(self, name: str) -> None:
        AgentDAO(self.db_path).delete(name)

    def save_agent_atomic(
            self,
            name: str,
            model: str,
            prompt: str,
            role: str,
            input_vars: list[str],
            output_vars: list[str],
            *,
            color: Optional[str] = None,
            symbol: Optional[str] = None,
            save_prompt_version: bool = True,
            metadata: dict | None = None,
            tools: Optional[dict] = None,
            reasoning_effort: Optional[str] = None,
            reasoning_summary: Optional[str] = None,
            system_prompt: Optional[str] = None,
            # Provider metadata
            provider_id: Optional[str] = None,
            model_class: Optional[str] = None,
            endpoint: Optional[str] = None,
            use_responses_api: bool = False,
            provider_features: Optional[dict] = None,
            # Structured output config
            structured_output_enabled: bool = False,
            schema_json: Optional[str] = None,
            schema_name: Optional[str] = None,
            temperature: Optional[float] = None,
    ) -> None:
        """Atomically save agent metadata. The legacy 'prompt version' system
        has been removed in favour of agent snapshots. The `save_prompt_version`
        parameter is retained for API compatibility but is ignored.
        """
        # Use transaction-scoped DAO so callers get atomicity
        with agent_dao(self.db_path) as dao:
            # Fetch previous row (if any) to detect changes
            prev = dao.get(name)

            dao.save(
                name,
                model,
                prompt,
                role,
                input_vars,
                output_vars,
                color,
                symbol,
                tools=tools,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                system_prompt_template=system_prompt,
            provider_id=provider_id,
            model_class=model_class,
            endpoint=endpoint,
            use_responses_api=use_responses_api,
            provider_features=provider_features,
            structured_output_enabled=structured_output_enabled,
            schema_json=schema_json,
            schema_name=schema_name,
            temperature=temperature,
        )

            # Optional automatic snapshot (configurable)
            if AGENT_SNAPSHOTS_AUTO:
                snapshot = {
                    "model": model,
                    "prompt_template": prompt,
                    "system_prompt_template": system_prompt,
                    "role": role,
                    "input_vars": input_vars,
                    "output_vars": output_vars,
                    "color": color,
                    "symbol": symbol,
                    "tools": tools,
                    "reasoning_effort": reasoning_effort,
                    "reasoning_summary": reasoning_summary,
                    "provider_id": provider_id,
                    "model_class": model_class,
                    "endpoint": endpoint,
                    "use_responses_api": use_responses_api,
                    "provider_features": provider_features,
                }
                try:
                    dao.save_snapshot(name, snapshot, metadata={"event": "auto_save"}, is_auto=True)
                except Exception:
                    # Non-fatal; transaction will still commit the agent save.
                    logger = __import__("logging").getLogger(__name__)
                    logger.exception("Failed to create automatic snapshot for %s", name)

        # Decide whether provider metadata changed (or new agent).
        def _provider_key_from_row(row: Optional[dict]) -> Optional[tuple]:
            if not row:
                return None
            pf = row.get("provider_features") or {}
            try:
                pf_s = json.dumps(pf, sort_keys=True, separators=(",", ":"))
            except Exception:
                pf_s = repr(pf)
            return (
                row.get("provider_id"),
                row.get("model_class"),
                row.get("endpoint"),
                bool(row.get("use_responses_api")),
                pf_s,
            )

        prev_key = _provider_key_from_row(prev)

        new_like = {
            "provider_id": provider_id,
            "model_class": model_class,
            "endpoint": endpoint,
            "use_responses_api": bool(use_responses_api),
            "provider_features": provider_features or {},
        }
        new_key = _provider_key_from_row(new_like)

        if prev_key != new_key:
            # Notify runtime: provider metadata changed (or new agent).
            try:
                runtime_hooks.on_agent_change()
            except Exception:
                pass

    def rename_agent_atomic(self, old_name: str, new_name: str) -> None:
        """Atomically rename an agent."""
        with agent_dao(self.db_path) as dao:
            dao.rename(old_name, new_name)

    def delete_agent_atomic(self, name: str) -> None:
        """Atomically delete an agent."""
        with agent_dao(self.db_path) as dao:
            dao.delete(name)

    # -----------------------
    # Snapshot wrappers
    # -----------------------
    def save_snapshot(self, agent_name: str, snapshot: dict, metadata: Optional[dict] = None, is_auto: bool = False) -> int:
        return AgentDAO(self.db_path).save_snapshot(agent_name, snapshot, metadata=metadata, is_auto=is_auto)

    def list_snapshots(self, agent_name: str) -> List[dict]:
        return AgentDAO(self.db_path).list_snapshots(agent_name)

    def get_snapshot(self, snapshot_id: int) -> Optional[dict]:
        return AgentDAO(self.db_path).get_snapshot_by_id(snapshot_id)

    def delete_snapshot(self, snapshot_id: int) -> None:
        return AgentDAO(self.db_path).delete_snapshot(snapshot_id)

    def prune_snapshots(self, agent_name: Optional[str] = None, keep: Optional[int] = None) -> int:
        """
        Prune agent snapshots using the DB maintenance helper.
        If keep is None, fall back to AGENT_SNAPSHOT_PRUNE_KEEP from config.
        Returns number of deleted rows.
        """
        from multi_agent_dashboard.db.infra.maintenance import prune_agent_snapshots
        if keep is None:
            keep = AGENT_SNAPSHOT_PRUNE_KEEP
        return prune_agent_snapshots(agent_name=agent_name, keep=keep)


# -----------------------
# Pipeline Service
# -----------------------
class PipelineService:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def list_pipelines(self) -> List[dict]:
        return PipelineDAO(self.db_path).list()

    def save_pipeline(self, pipeline_name: str, steps: List[str], metadata: Optional[dict] = None) -> None:
        PipelineDAO(self.db_path).save(pipeline_name, steps, metadata)

    def delete_pipeline(self, pipeline_name: str) -> None:
        PipelineDAO(self.db_path).delete(pipeline_name)
