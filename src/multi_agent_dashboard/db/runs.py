"""db/runs.py

Construction modes:

- RunDAO(db_path="...")
- RunDAO(conn=sqlite3.Connection)

Support for atomic multi-step operations:

with run_dao(db_path) as dao:
    run_id = dao.save(...)
    dao.add_tags(run_id, ...)
    dao.attach_metadata(run_id, ...)
"""
import json
import logging
import warnings
from datetime import datetime, UTC
from typing import Dict, Any, Tuple, Optional
from contextlib import contextmanager

from multi_agent_dashboard.db.infra.core import get_conn

logger = logging.getLogger(__name__)


class RunDAO:
    def __init__(self, db_path: Optional[str] = None, conn=None):
        if conn is None and db_path is None:
            raise ValueError("RunDAO requires either db_path or conn")

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
        logger.debug("Loading runs from DB")
        try:
            with self._connection() as conn:
                rows = conn.execute(
                    "SELECT id, timestamp, task_input FROM runs ORDER BY id DESC"
                ).fetchall()
        except Exception:
            logger.exception("Failed to load runs from DB")
            raise

        return [dict(row) for row in rows]

    def get(self, run_id: int) -> Tuple[dict | None, list[dict], list[dict]]:
        logger.debug("Loading details for run %s from DB", run_id)
        try:
            with self._connection() as conn:
                run = conn.execute(
                    """
                    SELECT timestamp, task_input, final_output, final_is_json, final_model
                    FROM runs WHERE id = ?
                    """,
                    (run_id,),
                ).fetchone()

                agents = conn.execute(
                    """
                    SELECT agent_name, output, is_json, model
                    FROM agent_outputs WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchall()

                metrics = conn.execute(
                    """
                    SELECT agent_name,
                           input_tokens,
                           output_tokens,
                           latency,
                           input_cost,
                           output_cost,
                           cost
                    FROM agent_metrics
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchall()
        except Exception:
            logger.exception("Failed to load details for run %s from DB", run_id)
            raise

        return (
            dict(run) if run else None,
            [dict(a) for a in agents],
            [dict(m) for m in metrics],
        )

    # -----------------------
    # WRITE operations
    # -----------------------

    def save(
        self,
        task_input: str,
        final_output: str,
        memory_dict: Dict[str, Any],
        *,
        agent_models: Dict[str, str] | None = None,
        final_model: str | None = None,
        agent_metrics: Dict[str, Dict[str, Any]] | None = None,
    ) -> int:
        ts = datetime.now(UTC).isoformat()
        agent_models = agent_models or {}
        agent_metrics = agent_metrics or {}

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
            with self._connection() as conn:
                c = conn.cursor()

                c.execute(
                    """
                    INSERT INTO runs
                        (timestamp, task_input, final_output, final_is_json, final_model)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (ts, task_input, final_text, final_is_json, final_model),
                )
                run_id = c.lastrowid

                # Agent outputs
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
                        INSERT INTO agent_outputs
                            (run_id, agent_name, output, is_json, model)
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

                    # Per-Agent metrics
                    for agent_name, m in agent_metrics.items():
                        c.execute(
                            """
                            INSERT INTO agent_metrics
                            (run_id,
                             agent_name,
                             input_tokens,
                             output_tokens,
                             latency,
                             input_cost,
                             output_cost,
                             cost)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                run_id,
                                agent_name,
                                m.get("input_tokens"),
                                m.get("output_tokens"),
                                m.get("latency"),
                                m.get("input_cost"),
                                m.get("output_cost"),
                                m.get("cost"),
                            ),
                        )
        except Exception:
            logger.exception("Failed to save run to DB")
            raise

        return run_id


# -----------------------
# Transaction-scoped DAO
# -----------------------

@contextmanager
def run_dao(db_path: str):
    """
    Yield a RunDAO bound to a single transaction.
    """
    with get_conn(db_path) as conn:
        yield RunDAO(conn=conn)


# -----------------------
# Compatibility wrappers
# -----------------------

def load_runs(db_path: str) -> list[dict]:
    warnings.warn(
        "load_runs is deprecated; use RunDAO.list",
        DeprecationWarning,
    )
    return RunDAO(db_path=db_path).list()


def load_run_details(db_path: str, run_id: int):
    warnings.warn(
        "load_run_details is deprecated; use RunDAO.get",
        DeprecationWarning,
    )
    return RunDAO(db_path=db_path).get(run_id)


def save_run_to_db(db_path: str, *args, **kwargs) -> int:
    warnings.warn(
        "save_run_to_db is deprecated; use RunDAO.save",
        DeprecationWarning,
    )
    return RunDAO(db_path=db_path).save(*args, **kwargs)
