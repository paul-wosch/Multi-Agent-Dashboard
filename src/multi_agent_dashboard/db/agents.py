# db/agents.py
import json
import logging

from datetime import datetime, UTC
from typing import List, Dict, Any, Optional, Tuple

from multi_agent_dashboard.db.infra.core import get_conn, safe_json_loads

logger = logging.getLogger(__name__)


# -----------------------
# READ operations
# -----------------------

def load_agents_from_db(db_path: str) -> list[dict]:
    """Return rows: agent_name, model, prompt_template, role, input_vars_json, output_vars_json"""
    logger.debug("Loading agents from DB")
    try:
        with get_conn(db_path) as conn:
            rows = conn.execute(
                "SELECT agent_name, model, prompt_template, role, input_vars, output_vars FROM agents"
            ).fetchall()
    except Exception:
        logger.exception("Failed to load agents from DB")
        raise

    agents = []
    for row in rows:
        agents.append({
            "agent_name": row["agent_name"],
            "model": row["model"],
            "prompt_template": row["prompt_template"],
            "role": row["role"],
            "input_vars": safe_json_loads(row["input_vars"], []),
            "output_vars": safe_json_loads(row["output_vars"], []),
        })
    return agents


def load_prompt_versions(db_path: str, agent_name: str):
    logger.debug("Loading prompt versions for %s from DB", agent_name)
    try:
        with get_conn(db_path) as conn:
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
        versions.append({
            "id": row["id"],
            "version": row["version"],
            "prompt": row["prompt"],
            "metadata": safe_json_loads(row["metadata_json"], {}),
            "created_at": row["timestamp"],
        })

    return versions


# -----------------------
# WRITE operations
# -----------------------

def save_agent_to_db(
        db_path: str,
        agent_name: str,
        model: str,
        prompt_template: str,
        role: str = "",
        input_vars: Optional[List[str]] = None,
        output_vars: Optional[List[str]] = None
    ):
    """Save agent metadata. input_vars/output_vars are stored as JSON arrays (strings) for flexibility."""

    input_json = json.dumps(input_vars or [])
    output_json = json.dumps(output_vars or [])

    logger.info("Saving agent %s to DB", agent_name)
    try:
        with get_conn(db_path) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT OR REPLACE INTO agents (agent_name, model, prompt_template, role, input_vars, output_vars)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (agent_name, model, prompt_template, role, input_json, output_json))
    except Exception:
        logger.exception("Failed to save %s to DB", agent_name)
        raise


def save_prompt_version(
        db_path: str,
        agent_name: str,
        prompt_text: str,
        metadata: Optional[dict] = None
    ) -> int:
    logger.info("Saving prompt version for %s to DB", agent_name)
    try:
        with get_conn(db_path) as conn:
            c = conn.cursor()

            c.execute("SELECT MAX(version) FROM agent_prompt_versions WHERE agent_name = ?", (agent_name,))
            result = c.fetchone()[0]
            new_version = 1 if result is None else result + 1

            ts = datetime.now(UTC).isoformat()
            metadata_json = json.dumps(metadata or {})

            c.execute("""
                INSERT INTO agent_prompt_versions (agent_name, version, prompt, metadata_json, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (agent_name, new_version, prompt_text, metadata_json, ts))
    except Exception:
        logger.exception("Failed to save prompt version for %s to DB", agent_name)
        raise

    return new_version


def rename_agent_in_db(db_path: str, old_name: str, new_name: str):
    """
    Safely rename an agent across all related tables in a single transaction.
    """
    if old_name == new_name:
        return
    logger.info("Renaming agent %s in DB", old_name)
    try:
        with get_conn(db_path) as conn:
            try:
                # Ensure target name does not already exist
                exists = conn.execute(
                    "SELECT 1 FROM agents WHERE agent_name = ?",
                    (new_name,)
                ).fetchone()
                if exists:
                    raise ValueError(f"Agent '{new_name}' already exists")

                # Update agents table
                conn.execute(
                    "UPDATE agents SET agent_name = ? WHERE agent_name = ?",
                    (new_name, old_name),
                )

                # Update prompt versions
                conn.execute(
                    "UPDATE agent_prompt_versions SET agent_name = ? WHERE agent_name = ?",
                    (new_name, old_name),
                )

                # Update historical outputs
                conn.execute(
                    "UPDATE agent_outputs SET agent_name = ? WHERE agent_name = ?",
                    (new_name, old_name),
                )

                # Update pipelines (JSON steps)
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

                conn.commit()

            except Exception:
                conn.rollback()
                raise
    except Exception:
        logger.exception("Failed to rename agent %s in DB", old_name)
        raise


def delete_agent(db_path: str, agent_name: str):
    logger.info("Deleting agent %s from DB", agent_name)
    try:
        with get_conn(db_path) as conn:
            conn.execute(
                "DELETE FROM agents WHERE agent_name = ?",
                (agent_name,)
            )
    except Exception:
        logger.exception("Failed to delete %s from DB", agent_name)
        raise
