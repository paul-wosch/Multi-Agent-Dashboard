# db/runs.py
import json
import logging

from datetime import datetime, UTC
from typing import List, Dict, Any, Optional, Tuple

from multi_agent_dashboard.db.infra.core import get_conn

logger = logging.getLogger(__name__)


class RunDAO:
    def __init__(self, db_path: str):
        self.db_path = db_path

    # ---- reads ----

    def list(self):
        return load_runs(self.db_path)

    def get(self, run_id: int):
        return load_run_details(self.db_path, run_id)

    # ---- writes ----

    def save(self, *args, **kwargs):
        return save_run_to_db(self.db_path, *args, **kwargs)


# -----------------------
# READ operations
# -----------------------

def load_runs(db_path: str) -> list[dict]:
    logger.debug("Loading runs from DB")
    try:
        with get_conn(db_path) as conn:
            rows = conn.execute(
                "SELECT id, timestamp, task_input FROM runs ORDER BY id DESC"
            ).fetchall()
    except Exception:
        logger.exception("Failed to load runs from DB")
        raise

    return [dict(row) for row in rows]


def load_run_details(db_path: str, run_id: int) -> Tuple[dict | None, list[dict]]:
    logger.debug("Loading details for run %s from DB", run_id)
    try:
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
    except Exception:
        logger.exception("Failed to load details for run %s from DB", run_id)
        raise

    return (
        dict(run) if run else None,
        [dict(a) for a in agents],
    )


# -----------------------
# WRITE operations
# -----------------------

def save_run_to_db(
    db_path: str,
    task_input: str,
    final_output: str,
    memory_dict: Dict[str, Any],
    *,
    agent_models: Dict[str, str] | None = None,
    final_model: str | None = None,
) -> int:
    ts = datetime.now(UTC).isoformat()
    agent_models = agent_models or {}

    if isinstance(final_output, str):
        final_text = final_output
        try:
            json.loads(final_output)
            final_is_json = 1
        except Exception:
            final_is_json = 0
    else:
        final_text = json.dumps(final_output)
        final_is_json = 1

    logger.info("Saving run to DB")
    try:
        with get_conn(db_path) as conn:
            c = conn.cursor()

            c.execute(
                """
                INSERT INTO runs (timestamp, task_input, final_output, final_is_json, final_model)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ts, task_input, final_text, final_is_json, final_model),
            )
            run_id = c.lastrowid

            for agent, output in memory_dict.items():
                if isinstance(output, str):
                    raw = output
                    try:
                        json.loads(output)
                        is_json = 1
                    except Exception:
                        is_json = 0
                else:
                    raw = json.dumps(output)
                    is_json = 1

                c.execute(
                    """
                    INSERT INTO agent_outputs (run_id, agent_name, output, is_json, model)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        agent,
                        raw,
                        is_json,
                        agent_models.get(agent),
                    ),
                )
    except Exception:
        logger.exception("Failed to save run to DB")
        raise

    return run_id
