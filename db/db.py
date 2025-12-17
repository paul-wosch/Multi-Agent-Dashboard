# db/db.py
import json
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, UTC
from contextlib import contextmanager
import sqlite3
import json

from db.migrations import apply_migrations

import sys
from pathlib import Path
# Get the parent directory using pathlib
parent_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))
# Import the module
from config import MIGRATIONS_PATH


def init_db(db_path: str):
    """
    Initialize the database:
    - open connection
    - apply migrations
    """
    with get_conn(db_path) as conn:
        apply_migrations(conn, MIGRATIONS_PATH)


def safe_json_loads(value: str | None, default):
    """
    Safely load JSON from DB fields.
    Returns default if value is None, empty, or invalid.
    """
    if not value or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


# -----------------------
# Connection helper
# -----------------------

@contextmanager
def get_conn(path):
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# -----------------------
# Cached READ operations
# -----------------------

@st.cache_data(ttl=60)
def load_agents_from_db(db_path: str):
    """Return rows: agent_name, model, prompt_template, role, input_vars_json, output_vars_json"""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT agent_name, model, prompt_template, role, input_vars, output_vars FROM agents"
        ).fetchall()

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


@st.cache_data(ttl=60)
def load_pipelines_from_db(db_path: str):
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT pipeline_name, steps_json, metadata_json, timestamp
            FROM pipelines
            ORDER BY pipeline_name
            """
        ).fetchall()

    result = []
    for row in rows:
        result.append({
            "pipeline_name": row["pipeline_name"],
            "steps": safe_json_loads(row["steps_json"], []),
            "metadata": safe_json_loads(row["metadata_json"], {}),
            "timestamp": row["timestamp"],
        })
    return result


@st.cache_data(ttl=30)
def load_runs(db_path: str):
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT id, timestamp, task_input FROM runs ORDER BY id DESC"
        ).fetchall()

    return [dict(row) for row in rows]


@st.cache_data(ttl=30)
def load_run_details(db_path: str, run_id: int):
    with get_conn(db_path) as conn:
        run = conn.execute(
            """
            SELECT timestamp, task_input, final_output, final_is_json, final_model
            FROM runs WHERE id = ?
            """,
            (run_id,)
        ).fetchone()

        agents = conn.execute(
            """
            SELECT agent_name, output, is_json, model
            FROM agent_outputs WHERE run_id = ?
            """,
            (run_id,)
        ).fetchall()

    return (
        dict(run) if run else None,
        [dict(a) for a in agents],
    )


@st.cache_data(ttl=60)
def load_prompt_versions(db_path: str, agent_name: str):
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
# WRITE operations (uncached)
# -----------------------

def save_run_to_db(
        db_path: str,
        task_input: str,
        final_output: str,
        memory_dict: Dict[str, Any]
    ) -> int:
    with get_conn(db_path) as conn:
        c = conn.cursor()

        ts = datetime.now(UTC).isoformat()

        # ---- detect final output type ----
        if isinstance(final_output, str):
            final_text = final_output
            final_is_json = 0
            try:
                json.loads(final_output)
                final_is_json = 1
            except Exception:
                pass
        else:
            final_text = json.dumps(final_output)
            final_is_json = 1

        # ---- best-effort final model ----
        final_model = None
        engine = getattr(st.session_state, "engine", None)
        if engine:
            # final output comes from 'final' if present, else last agent
            if "final" in engine.state:
                final_agent = "finalizer"
            else:
                final_agent = next(reversed(engine.memory.keys()), None)

            if final_agent and final_agent in engine.agents:
                final_model = engine.agents[final_agent].model

        c.execute("""
                  INSERT INTO runs (timestamp,
                                    task_input,
                                    final_output,
                                    final_is_json,
                                    final_model)
                  VALUES (?, ?, ?, ?, ?)
                  """, (
                      ts,
                      task_input,
                      final_text,
                      final_is_json,
                      final_model
                  ))

        run_id = c.lastrowid

        for agent, output in memory_dict.items():
            # Normalize output to string
            if isinstance(output, str):
                raw_text = output
                is_json = 0
            else:
                raw_text = json.dumps(output)
                is_json = 1

            # Defensive JSON detection for string outputs
            if isinstance(output, str):
                try:
                    json.loads(output)
                    is_json = 1
                except Exception:
                    pass

            # Best-effort model lookup (non-breaking)
            model = None
            engine = getattr(st.session_state, "engine", None)
            if engine and agent in engine.agents:
                model = engine.agents[agent].model

            c.execute("""
                      INSERT INTO agent_outputs (run_id,
                                                 agent_name,
                                                 output,
                                                 is_json,
                                                 model)
                      VALUES (?, ?, ?, ?, ?)
                      """, (run_id, agent, raw_text, is_json, model))

    return run_id


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

    with get_conn(db_path) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO agents (agent_name, model, prompt_template, role, input_vars, output_vars)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (agent_name, model, prompt_template, role, input_json, output_json))


def save_prompt_version(
        db_path: str,
        agent_name: str,
        prompt_text: str,
        metadata: Optional[dict] = None
    ) -> int:
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

    return new_version


def save_pipeline_to_db(
        db_path: str,
        pipeline_name: str,
        steps: List[str],
        metadata: Optional[dict] = None
    ):
    with get_conn(db_path) as conn:
        c = conn.cursor()

        ts = datetime.now(UTC).isoformat()
        steps_json = json.dumps(steps)
        metadata_json = json.dumps(metadata or {})

        c.execute("""
            INSERT OR REPLACE INTO pipelines (pipeline_name, steps_json, metadata_json, timestamp)
            VALUES (?, ?, ?, ?)
        """, (pipeline_name, steps_json, metadata_json, ts))


def delete_pipeline_from_db(db_path: str, pipeline_name: str):
    with get_conn(db_path) as conn:
        c = conn.cursor()
        c.execute("DELETE FROM pipelines WHERE pipeline_name = ?", (pipeline_name,))


def rename_agent_in_db(db_path: str, old_name: str, new_name: str):
    """
    Safely rename an agent across all related tables in a single transaction.
    """
    if old_name == new_name:
        return

    with get_conn(db_path) as conn:
        try:
            conn.execute("BEGIN")

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


def delete_agent(db_path: str, agent_name: str):
    with get_conn(db_path) as conn:
        conn.execute(
            "DELETE FROM agents WHERE agent_name = ?",
            (agent_name,)
        )
