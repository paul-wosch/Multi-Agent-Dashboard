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
import calendar
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple, Optional
from contextlib import contextmanager

from multi_agent_dashboard.db.infra.core import get_conn

logger = logging.getLogger(__name__)


def _align_to_period_start(reference: datetime, period: str) -> datetime:
    period = period.lower()
    if period == "daily":
        return reference.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "weekly":
        week_start = reference - timedelta(days=reference.weekday())
        return week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "monthly":
        return reference.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if period == "yearly":
        return reference.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"Unsupported period: {period}")


def _add_months(base: datetime, months: int) -> datetime:
    total_months = base.month - 1 + months
    year = base.year + total_months // 12
    month = total_months % 12 + 1
    day = min(base.day, calendar.monthrange(year, month)[1])
    return base.replace(year=year, month=month, day=day)


def _shift_period_start(start: datetime, period: str, steps: int = 1) -> datetime:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    period = period.lower()
    if period == "daily":
        return start - timedelta(days=steps)
    if period == "weekly":
        return start - timedelta(weeks=steps)
    if period == "monthly":
        return _add_months(start, -steps)
    if period == "yearly":
        return start.replace(year=start.year - steps)
    raise ValueError(f"Unsupported period: {period}")


def _format_period_value(start: datetime, period: str) -> str:
    period = period.lower()
    if period == "daily":
        return start.strftime("%Y-%m-%d")
    if period == "weekly":
        return start.strftime("%Y-%W")
    if period == "monthly":
        return start.strftime("%Y-%m")
    if period == "yearly":
        return start.strftime("%Y")
    raise ValueError(f"Unsupported period: {period}")


def _build_recent_period_keys(period: str, limit: int, reference: datetime) -> List[str]:
    period = period.lower()
    if period not in {"daily", "weekly", "monthly", "yearly"}:
        raise ValueError(f"Unsupported period: {period}")
    current = _align_to_period_start(reference, period)
    keys: List[str] = []
    for _ in range(limit):
        keys.append(_format_period_value(current, period))
        current = _shift_period_start(current, period)
    return keys


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

    def get(
            self,
            run_id: int
    ) -> Tuple[dict | None, list[dict], list[dict], list[dict], list[dict]]:
        logger.debug("Loading details for run %s from DB", run_id)
        try:
            with self._connection() as conn:
                run = conn.execute(
                    """
                    SELECT timestamp, task_input, final_output, final_is_json, final_model, strict_schema_exit
                    FROM runs
                    WHERE id = ?
                    """,
                    (run_id,),
                ).fetchone()

                agents = conn.execute(
                    """
                    SELECT agent_name, output, is_json, model, schema_validation_failed
                    FROM agent_outputs
                    WHERE run_id = ?
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

                tool_usages = conn.execute(
                    """
                    SELECT agent_name,
                           tool_type,
                           tool_call_id,
                           args_json,
                           result_summary
                    FROM tool_usages
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchall()

                agent_run_configs = conn.execute(
                    """
                    SELECT agent_name,
                           model,
                           prompt_template,
                           role,
                           input_vars,
                           output_vars,
                           tools_json,
                           tools_config_json,
                           reasoning_effort,
                           reasoning_summary,
                           reasoning_config_json,
                           extra_config_json,
                           system_prompt_template,
                           provider_id,
                           model_class,
                           endpoint,
                           use_responses_api,
                           provider_features_json,
                           structured_output_enabled,
                           schema_json,
                           schema_name,
                           temperature,
                           strict_schema_validation
                    FROM agent_run_configs
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
            [dict(t) for t in tool_usages],
            [dict(c) for c in agent_run_configs],
        )

    # -----------------------
    # Cost summary helpers (SQL-backed, independent of UI)
    # -----------------------

    def get_cost_total_for_period(self, period: str = "monthly") -> float:
        """
        Return the total cost for the 'current' period.

        Supported periods:
          - 'daily'   -> today (YYYY-MM-DD)
          - 'weekly'  -> ISO-like week key (YYYY-WW using %W)
          - 'monthly' -> current month (YYYY-MM)
          - 'yearly'  -> current year (YYYY)
          - 'total'   -> all time total (no timestamp filtering)

        The implementation uses SQLite expressions to extract a period key from
        runs.timestamp (stored as ISO-like strings). For safety against differing
        timestamp formats, substr() is used for daily/monthly/yearly extraction,
        while weekly uses strftime on the date portion.

        Returns 0.0 when no rows found.
        """
        period = (period or "monthly").lower()
        now = datetime.now(timezone.utc)

        if period == "total":
            try:
                with self._connection() as conn:
                    row = conn.execute(
                        "SELECT SUM(cost) AS total_cost FROM agent_metrics"
                    ).fetchone()
                    total = row[0] if row and row[0] is not None else 0.0
                    return float(total)
            except Exception:
                logger.exception("Failed to compute total cost from DB")
                raise

        # Mapping: SQL expression to extract a period key from runs.timestamp
        period_expr_map = {
            "daily": "substr(r.timestamp,1,10)",     # YYYY-MM-DD
            "monthly": "substr(r.timestamp,1,7)",    # YYYY-MM
            "yearly": "substr(r.timestamp,1,4)",     # YYYY
            "weekly": "strftime('%Y-%W', substr(r.timestamp,1,10))",  # YYYY-WW
        }

        if period not in period_expr_map:
            raise ValueError(f"Unsupported period: {period}")

        if period == "daily":
            period_value = now.strftime("%Y-%m-%d")
        elif period == "monthly":
            period_value = now.strftime("%Y-%m")
        elif period == "yearly":
            period_value = now.strftime("%Y")
        elif period == "weekly":
            period_value = now.strftime("%Y-%W")
        else:
            period_value = now.strftime("%Y-%m")  # fallback

        expr = period_expr_map[period]

        sql = f"""
            SELECT SUM(am.cost) AS total_cost
            FROM agent_metrics am
            JOIN runs r ON r.id = am.run_id
            WHERE r.timestamp IS NOT NULL AND ({expr}) = ?
        """

        try:
            with self._connection() as conn:
                row = conn.execute(sql, (period_value,)).fetchone()
                total = row[0] if row and row[0] is not None else 0.0
                return float(total)
        except Exception:
            logger.exception("Failed to compute cost for period=%s", period)
            raise

    def get_costs_by_period(self, period: str = "monthly", limit: int = 12) -> List[Dict[str, Any]]:
        """
        Return a list of cost totals keyed by period_value for recent periods.

        Example returned list:
          [
            {"period": "2026-01", "total_cost": 1.23},
            {"period": "2025-12", "total_cost": 4.56},
            ...
          ]

        This is useful for historical trend displays. The query groups agent_metrics
        by the chosen period expression and orders by period DESC.

        Limit is applied to number of returned periods.
        """
        period = (period or "monthly").lower()

        if period == "total":
            try:
                with self._connection() as conn:
                    row = conn.execute("SELECT SUM(cost) AS total_cost FROM agent_metrics").fetchone()
                    total = row[0] if row and row[0] is not None else 0.0
                    return [{"period": "all", "total_cost": float(total)}]
            except Exception:
                logger.exception("Failed to compute all-time cost breakdown")
                raise

        period_expr_map = {
            "daily": "substr(r.timestamp,1,10)",
            "monthly": "substr(r.timestamp,1,7)",
            "yearly": "substr(r.timestamp,1,4)",
            "weekly": "strftime('%Y-%W', substr(r.timestamp,1,10))",
        }

        if period not in period_expr_map:
            raise ValueError(f"Unsupported period: {period}")

        limit = max(1, int(limit))
        expr = period_expr_map[period]
        now = datetime.now(timezone.utc)
        period_keys = _build_recent_period_keys(period, limit, now)

        sql = f"""
            SELECT ({expr}) AS period_value, SUM(am.cost) AS total_cost
            FROM agent_metrics am
            JOIN runs r ON r.id = am.run_id
            WHERE r.timestamp IS NOT NULL
            GROUP BY period_value
            ORDER BY period_value DESC
            LIMIT ?
        """

        try:
            with self._connection() as conn:
                rows = conn.execute(sql, (limit,)).fetchall()
                period_totals: Dict[str, float] = {}
                for row in rows:
                    row_dict = dict(row)
                    period_totals[row_dict["period_value"]] = float(row_dict["total_cost"] or 0.0)
                return [
                    {"period": key, "total_cost": period_totals.get(key, 0.0)}
                    for key in period_keys
                ]
        except Exception:
            logger.exception("Failed to compute costs by period=%s", period)
            raise

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
            agent_configs: Dict[str, Dict[str, Any]] | None = None,
            agent_metrics: Dict[str, Dict[str, Any]] | None = None,
            tool_usages: Dict[str, List[Dict[str, Any]]] | None = None,
            strict_schema_exit: bool = False,
            agent_schema_validation_failed: Dict[str, bool] | None = None,
    ) -> int:
        ts = datetime.now(timezone.utc).isoformat()
        agent_models = agent_models or {}
        agent_configs = agent_configs or {}
        agent_metrics = agent_metrics or {}
        tool_usages = tool_usages or {}
        agent_schema_validation_failed = agent_schema_validation_failed or {}

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
                        (timestamp, task_input, final_output, final_is_json, final_model, strict_schema_exit)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (ts, task_input, final_text, final_is_json, final_model, 1 if strict_schema_exit else 0),
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
                            (run_id, agent_name, output, is_json, model, schema_validation_failed)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            agent,
                            raw,
                            is_json,
                            agent_models.get(agent),
                            1 if agent_schema_validation_failed.get(agent) else 0,
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

                # Per-agent configuration snapshot (per run).
                # NOTE: schema is intentionally redundant with agents table so
                # that past runs remain reproducible even if agents change.
                for agent_name, cfg in agent_configs.items():
                    c.execute(
                        """
                        INSERT INTO agent_run_configs
                            (run_id,
                             agent_name,
                             model,
                             provider_id,
                             model_class,
                             endpoint,
                             use_responses_api,
                             provider_features_json,
                             structured_output_enabled,
                             schema_json,
                             schema_name,
                             temperature,
                             strict_schema_validation,
                             prompt_template,
                             role,
                             input_vars,
                             output_vars,
                             tools_json,
                             tools_config_json,
                             reasoning_effort,
                             reasoning_summary,
                             reasoning_config_json,
                             extra_config_json,
                             system_prompt_template)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            agent_name,
                            cfg.get("model"),
                            cfg.get("provider_id"),
                            cfg.get("model_class"),
                            cfg.get("endpoint"),
                            1 if cfg.get("use_responses_api") else 0,
                            json.dumps(cfg.get("provider_features") or {}),
                            1 if cfg.get("structured_output_enabled") else 0,
                            cfg.get("schema_json"),
                            cfg.get("schema_name"),
                            cfg.get("temperature"),
                            1 if cfg.get("strict_schema_validation") else 0,
                            cfg.get("prompt_template"),
                            cfg.get("role"),
                            json.dumps(cfg.get("input_vars") or []),
                            json.dumps(cfg.get("output_vars") or []),
                            json.dumps(cfg.get("tools") or {}),
                            json.dumps(cfg.get("tools_config") or {}),
                            cfg.get("reasoning_effort"),
                            cfg.get("reasoning_summary"),
                            json.dumps(cfg.get("reasoning_config") or {}),
                            # Reserved for future options such as temperature
                            json.dumps(cfg.get("extra") or {}),
                            cfg.get("system_prompt_template"),
                        ),
                    )

                # tool usages per agent
                logger.debug("Saving tool usages for run %s: %r", run_id, tool_usages)
                for agent_name, entries in tool_usages.items():
                    for entry in entries or []:
                        action = entry.get("action") or {}
                        # args_json is now scoped to the tool call itself
                        # (arguments + status). Tool and reasoning config are
                        # stored once per agent/run in agent_run_configs.
                        payload = {"action": action}

                        status = entry.get("status")
                        if status is not None:
                            payload["status"] = status

                        c.execute(
                            """
                            INSERT INTO tool_usages
                            (run_id,
                             agent_name,
                             tool_type,
                             tool_call_id,
                             args_json,
                             result_summary)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                run_id,
                                agent_name,
                                entry.get("tool_type"),
                                entry.get("id"),
                                json.dumps(payload),
                                entry.get("result_summary") or "",
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
