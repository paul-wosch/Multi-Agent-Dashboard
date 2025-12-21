# src/multi_agent_dashboard/db/services.py
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Tuple

from multi_agent_dashboard.db.runs import RunDAO, run_dao
from multi_agent_dashboard.db.agents import AgentDAO, agent_dao
from multi_agent_dashboard.db.pipelines import PipelineDAO, pipeline_dao


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
        final_model: Optional[str] = None
    ) -> int:
        with run_dao(self.db_path) as dao:
            run_id = dao.save(
                task_input,
                final_output,
                memory,
                agent_models=agent_models,
                final_model=final_model,
            )
            return run_id

    def list_runs(self) -> List[dict]:
        return RunDAO(self.db_path).list()

    def get_run_details(self, run_id: int) -> Tuple[dict | None, List[dict]]:
        return RunDAO(self.db_path).get(run_id)


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
    ) -> None:
        AgentDAO(self.db_path).save(name, model, prompt_template, role, input_vars, output_vars)

    def save_prompt_version(self, agent_name: str, prompt_text: str, metadata: Optional[dict] = None) -> int:
        return AgentDAO(self.db_path).save_prompt_version(agent_name, prompt_text, metadata)

    def rename_agent(self, old_name: str, new_name: str) -> None:
        AgentDAO(self.db_path).rename(old_name, new_name)

    def delete_agent(self, name: str) -> None:
        AgentDAO(self.db_path).delete(name)

    def load_prompt_versions(self, agent_name: str) -> List[dict]:
        return AgentDAO(self.db_path).load_prompt_versions(agent_name)

    def save_agent_atomic(
            self,
            name: str,
            model: str,
            prompt: str,
            role: str,
            input_vars: list[str],
            output_vars: list[str],
            *,
            save_prompt_version: bool = True,
            metadata: dict | None = None,
    ) -> None:
        """Atomically save agent metadata and optionally create a prompt version."""
        with agent_dao(self.db_path) as dao:
            dao.save(
                name,
                model,
                prompt,
                role,
                input_vars,
                output_vars,
            )

            if save_prompt_version:
                dao.save_prompt_version(
                    name,
                    prompt,
                    metadata,
                )

    def rename_agent_atomic(self, old_name: str, new_name: str) -> None:
        """Atomically rename an agent."""
        with agent_dao(self.db_path) as dao:
            dao.rename(old_name, new_name)

    def delete_agent_atomic(self, name: str) -> None:
        """Atomically delete an agent."""
        with agent_dao(self.db_path) as dao:
            dao.delete(name)


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
