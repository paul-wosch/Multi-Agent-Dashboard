"""db/agents.py

Construction modes:

- AgentDAO(db_path="...")
- AgentDAO(conn=sqlite3.Connection)

Support for atomic multi-step operations:

with agent_dao(db_path) as dao:
    dao.save(...)
    dao.save_prompt_version(...)
    dao.rename(...)
"""
import json
import logging
import warnings
from datetime import datetime, UTC
from typing import List, Optional
from contextlib import contextmanager

from multi_agent_dashboard.db.infra.core import get_conn, safe_json_loads
from multi_agent_dashboard.config import UI_COLORS

DEFAULT_COLOR = UI_COLORS["default"]["value"]
DEFAULT_SYMBOL = UI_COLORS["default"]["symbol"]

logger = logging.getLogger(__name__)


class AgentDAO:
    def __init__(self, db_path: Optional[str] = None, conn=None):
        if conn is None and db_path is None:
            raise ValueError("AgentDAO requires either db_path or conn")

        self._db_path = db_path
        self._conn = conn

    # -----------------------
    # internal helpers
    # -----------------------

    @contextmanager
    def _connection(self):
        """
        Yield a connection.
        If DAO was constructed with a connection, reuse it.
        Otherwise, open a new one.
        """
        if self._conn is not None:
            yield self._conn
        else:
            with get_conn(self._db_path) as conn:
                yield conn

    # -----------------------
    # READ operations
    # -----------------------

    def list(self) -> list[dict]:
        logger.debug("Loading agents from DB")
        try:
            with self._connection() as conn:
                rows = conn.execute(
                    """
                    SELECT agent_name,
                           model,
                           prompt_template,
                           role,
                           input_vars,
                           output_vars,
                           color,
                           symbol,
                           tools_json,
                           reasoning_effort,
                           reasoning_summary
                    FROM agents
                    """
                ).fetchall()
        except Exception:
            logger.exception("Failed to load agents from DB")
            raise

        agents = []
        for row in rows:
            color = row["color"] or DEFAULT_COLOR
            symbol = row["symbol"] or DEFAULT_SYMBOL
            agents.append(
                {
                    "agent_name": row["agent_name"],
                    "model": row["model"],
                    "prompt_template": row["prompt_template"],
                    "role": row["role"],
                    "input_vars": safe_json_loads(row["input_vars"], []),
                    "output_vars": safe_json_loads(row["output_vars"], []),
                    "color": color,
                    "symbol": symbol,
                    "tools": safe_json_loads(row["tools_json"], {}),
                    "reasoning_effort": row["reasoning_effort"],
                    "reasoning_summary": row["reasoning_summary"],
                }
            )
        return agents

    def load_prompt_versions(self, agent_name: str) -> list[dict]:
        logger.debug("Loading prompt versions for %s from DB", agent_name)
        try:
            with self._connection() as conn:
                rows = conn.execute(
                    """
                    SELECT id, version, prompt, metadata_json, timestamp
                    FROM agent_prompt_versions
                    WHERE agent_name = ?
                    ORDER BY version DESC
                    """,
                    (agent_name,),
                ).fetchall()
        except Exception:
            logger.exception("Failed to load prompt versions for %s from DB", agent_name)
            raise

        versions = []
        for row in rows:
            versions.append(
                {
                    "id": row["id"],
                    "version": row["version"],
                    "prompt": row["prompt"],
                    "metadata": safe_json_loads(row["metadata_json"], {}),
                    "created_at": row["timestamp"],
                }
            )
        return versions

    # -----------------------
    # WRITE operations
    # -----------------------

    def save(
            self,
            agent_name: str,
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
    ) -> None:
        input_json = json.dumps(input_vars or [])
        output_json = json.dumps(output_vars or [])

        # Backwards-compatible defaults
        color = color or DEFAULT_COLOR
        symbol = symbol or DEFAULT_SYMBOL

        tools_json = json.dumps(tools or {})

        logger.info("Saving agent %s to DB", agent_name)
        try:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO agents
                        (agent_name,
                         model,
                         prompt_template,
                         role,
                         input_vars,
                         output_vars,
                         color,
                         symbol,
                         tools_json,
                         reasoning_effort,
                         reasoning_summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        agent_name,
                        model,
                        prompt_template,
                        role,
                        input_json,
                        output_json,
                        color,
                        symbol,
                        tools_json,
                        reasoning_effort,
                        reasoning_summary,
                    ),
                )
        except Exception:
            logger.exception("Failed to save %s to DB", agent_name)
            raise

    def save_prompt_version(
        self,
        agent_name: str,
        prompt_text: str,
        metadata: Optional[dict] = None,
    ) -> int:
        logger.info("Saving prompt version for %s to DB", agent_name)
        try:
            with self._connection() as conn:
                row = conn.execute(
                    "SELECT MAX(version) FROM agent_prompt_versions WHERE agent_name = ?",
                    (agent_name,),
                ).fetchone()

                new_version = 1 if row[0] is None else row[0] + 1

                ts = datetime.now(UTC).isoformat()
                metadata_json = json.dumps(metadata or {})

                conn.execute(
                    """
                    INSERT INTO agent_prompt_versions
                        (agent_name, version, prompt, metadata_json, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        agent_name,
                        new_version,
                        prompt_text,
                        metadata_json,
                        ts,
                    ),
                )
        except Exception:
            logger.exception("Failed to save prompt version for %s to DB", agent_name)
            raise

        return new_version

    def rename(self, old_name: str, new_name: str) -> None:
        if old_name == new_name:
            return

        logger.info("Renaming agent %s in DB", old_name)
        try:
            with self._connection() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM agents WHERE agent_name = ?",
                    (new_name,),
                ).fetchone()
                if exists:
                    raise ValueError(f"Agent '{new_name}' already exists")

                conn.execute(
                    "UPDATE agents SET agent_name = ? WHERE agent_name = ?",
                    (new_name, old_name),
                )

                conn.execute(
                    "UPDATE agent_prompt_versions SET agent_name = ? WHERE agent_name = ?",
                    (new_name, old_name),
                )

                conn.execute(
                    "UPDATE agent_outputs SET agent_name = ? WHERE agent_name = ?",
                    (new_name, old_name),
                )

                # Keep metrics, tool usages, and per-run configs in sync so
                # history queries stay coherent.
                conn.execute(
                    "UPDATE agent_metrics SET agent_name = ? WHERE agent_name = ?",
                    (new_name, old_name),
                )
                conn.execute(
                    "UPDATE tool_usages SET agent_name = ? WHERE agent_name = ?",
                    (new_name, old_name),
                )
                conn.execute(
                    "UPDATE agent_run_configs SET agent_name = ? WHERE agent_name = ?",
                    (new_name, old_name),
                )

                rows = conn.execute(
                    "SELECT pipeline_name, steps_json FROM pipelines"
                ).fetchall()

                for pipeline_name, steps_json in rows:
                    if not steps_json:
                        continue

                    steps = json.loads(steps_json)
                    updated = False

                    new_steps = []
                    for step in steps:
                        if step == old_name:
                            new_steps.append(new_name)
                            updated = True
                        else:
                            new_steps.append(step)

                    if updated:
                        conn.execute(
                            "UPDATE pipelines SET steps_json = ? WHERE pipeline_name = ?",
                            (json.dumps(new_steps), pipeline_name),
                        )
        except Exception:
            logger.exception("Failed to rename agent %s in DB", old_name)
            raise

    def delete(self, agent_name: str) -> None:
        logger.info("Deleting agent %s from DB", agent_name)
        try:
            with self._connection() as conn:
                conn.execute(
                    "DELETE FROM agents WHERE agent_name = ?",
                    (agent_name,),
                )
        except Exception:
            logger.exception("Failed to delete %s from DB", agent_name)
            raise


# -----------------------
# Transaction-scoped DAO
# -----------------------

@contextmanager
def agent_dao(db_path: str):
    """
    Yield an AgentDAO bound to a single transaction.
    """
    with get_conn(db_path) as conn:
        yield AgentDAO(conn=conn)


# -----------------------
# Compatibility wrappers
# -----------------------

def load_agents_from_db(db_path: str) -> list[dict]:
    warnings.warn(
        "load_agents_from_db is deprecated; use AgentDAO.list",
        DeprecationWarning,
    )
    return AgentDAO(db_path=db_path).list()


def load_prompt_versions(db_path: str, agent_name: str):
    warnings.warn(
        "load_prompt_versions is deprecated; use AgentDAO.load_prompt_versions",
        DeprecationWarning,
    )
    return AgentDAO(db_path=db_path).load_prompt_versions(agent_name)


def save_agent_to_db(db_path: str, *args, **kwargs):
    warnings.warn(
        "save_agent_to_db is deprecated; use AgentDAO.save",
        DeprecationWarning,
    )
    return AgentDAO(db_path=db_path).save(*args, **kwargs)


def save_prompt_version(db_path: str, *args, **kwargs):
    warnings.warn(
        "save_prompt_version is deprecated; use AgentDAO.save_prompt_version",
        DeprecationWarning,
    )
    return AgentDAO(db_path=db_path).save_prompt_version(*args, **kwargs)


def rename_agent_in_db(db_path: str, *args, **kwargs):
    warnings.warn(
        "rename_agent_in_db is deprecated; use AgentDAO.rename",
        DeprecationWarning,
    )
    return AgentDAO(db_path=db_path).rename(*args, **kwargs)


def delete_agent(db_path: str, *args, **kwargs):
    warnings.warn(
        "delete_agent is deprecated; use AgentDAO.delete",
        DeprecationWarning,
    )
    return AgentDAO(db_path=db_path).delete(*args, **kwargs)
