# ui/view_models.py
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

import pandas as pd

from multi_agent_dashboard.ui.utils import parse_json_field


class AgentMetricsView(NamedTuple):
    agent_name: str
    model: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    latency: Optional[float]
    input_cost: Optional[float]
    output_cost: Optional[float]
    total_cost: Optional[float]


class AgentConfigView(NamedTuple):
    agent_name: str
    model: str
    role: str
    tools_enabled: bool
    tools: List[str]
    web_search_allowed_domains: Optional[List[str]]
    reasoning_effort: str
    reasoning_summary: str
    raw_tools_config: Optional[dict] = None
    raw_reasoning_config: Optional[dict] = None
    raw_extra_config: Optional[dict] = None


def summarize_agent_metrics(
    metrics: Dict[str, dict] | List[dict],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Normalize metrics to a summary + per-agent DataFrame.

    Accepts either:
      - mapping agent_name -> metrics dict, or
      - list of DB rows where each row has 'agent_name'.
    """
    if isinstance(metrics, dict):
        metrics_iter: List[dict] = [
            {"agent_name": name, **(m or {})} for name, m in metrics.items()
        ]
    else:
        metrics_iter = list(metrics or [])

    total_latency = sum((m.get("latency") or 0.0) for m in metrics_iter)
    total_input_cost = sum((m.get("input_cost") or 0.0) for m in metrics_iter)
    total_output_cost = sum((m.get("output_cost") or 0.0) for m in metrics_iter)
    total_cost = sum((m.get("cost") or 0.0) for m in metrics_iter)

    rows = []
    for m in metrics_iter:
        rows.append(
            {
                "Agent": m.get("agent_name", ""),
                "Model": m.get("model", ""),
                "Input Tokens": m.get("input_tokens"),
                "Output Tokens": m.get("output_tokens"),
                "Latency (s)": None
                if m.get("latency") is None
                else round(m["latency"], 3),
                "Input Cost ($)": None
                if m.get("input_cost") is None
                else round(m["input_cost"], 6),
                "Output Cost ($)": None
                if m.get("output_cost") is None
                else round(m["output_cost"], 6),
                "Total Cost ($)": None
                if m.get("cost") is None
                else round(m["cost"], 6),
            }
        )

    df = pd.DataFrame(rows)

    return (
        {
            "total_latency": total_latency,
            "total_input_cost": total_input_cost,
            "total_output_cost": total_output_cost,
            "total_cost": total_cost,
        },
        df,
    )


def df_replace_none(df: pd.DataFrame, none_value: str = "–") -> pd.DataFrame:
    """
    Replace None/NaN values in a DataFrame with a readable placeholder.

    Uses DataFrame.where with pd.notnull to reliably replace missing values
    across pandas versions (avoids relying on DataFrame.map/applymap
    semantics which have changed across releases).

    Consider re-adding `return df.map(lambda v: none_value if v is None else v)`
    and removing the conditional after verifying correctness
    of the expression above.
    """
    # Ensure we have a DataFrame (if a different structure is passed)
    if not isinstance(df, pd.DataFrame):
        return df

    return df.where(pd.notnull(df), none_value)


def metrics_view_from_engine_result(
    result,
    steps: List[str],
) -> List[AgentMetricsView]:
    """
    Build AgentMetricsView list from an EngineResult and explicit step order.
    """
    views: List[AgentMetricsView] = []
    metrics = result.agent_metrics or {}
    for name in steps:
        m = metrics.get(name, {}) or {}
        views.append(
            AgentMetricsView(
                agent_name=name,
                model=m.get("model", ""),
                input_tokens=m.get("input_tokens"),
                output_tokens=m.get("output_tokens"),
                latency=m.get("latency"),
                input_cost=m.get("input_cost"),
                output_cost=m.get("output_cost"),
                total_cost=m.get("cost"),
            )
        )
    return views


def metrics_view_from_db_rows(metrics_rows: List[dict]) -> List[AgentMetricsView]:
    """
    Build AgentMetricsView list from DB metric rows.
    """
    views: List[AgentMetricsView] = []
    for m in metrics_rows or []:
        views.append(
            AgentMetricsView(
                agent_name=m["agent_name"],
                model=m.get("model", ""),
                input_tokens=m.get("input_tokens"),
                output_tokens=m.get("output_tokens"),
                latency=m.get("latency"),
                input_cost=m.get("input_cost"),
                output_cost=m.get("output_cost"),
                total_cost=m.get("cost"),
            )
        )
    return views


def parse_agent_run_config_row(cfg: dict) -> dict:
    """
    Normalize a historic agent_run_config row from the DB into a dict
    that is easier to consume in the UI.

    This is used only for historical runs and does NOT change storage.
    """
    tools_json = parse_json_field(cfg.get("tools_json"), {})
    tools_cfg_json = parse_json_field(cfg.get("tools_config_json"), {})
    reasoning_cfg_json = parse_json_field(cfg.get("reasoning_config_json"), {})
    extra_cfg_json = parse_json_field(cfg.get("extra_config_json"), {})

    tools_enabled = bool(tools_json.get("enabled"))
    enabled_tools = tools_json.get("tools") or []

    return {
        "model": cfg.get("model"),
        "role": cfg.get("role") or "–",
        "tools_enabled": tools_enabled,
        "enabled_tools": enabled_tools,
        "tools_json": tools_json,
        "tools_config_json": tools_cfg_json,
        "reasoning_config_json": reasoning_cfg_json,
        "extra_config_json": extra_cfg_json,
        "reasoning_effort": cfg.get("reasoning_effort") or "default",
        "reasoning_summary": cfg.get("reasoning_summary") or "none",
    }


def extract_web_search_allowed_domains_from_tools_cfg(
    tools_cfg_json: dict,
) -> Optional[List[str]]:
    """
    Extract web_search allowed_domains from the low-level tools_config_json.

    Returns a flat list of string domains, or None if not specified.
    """
    if not isinstance(tools_cfg_json, dict):
        return None
    tools_low = tools_cfg_json.get("tools")
    if not isinstance(tools_low, list):
        return None

    for tcfg in tools_low:
        if not isinstance(tcfg, dict):
            continue
        if tcfg.get("type") != "web_search":
            continue
        filters = tcfg.get("filters") or {}
        if isinstance(filters, dict) and "allowed_domains" in filters:
            allowed = filters["allowed_domains"]
            if isinstance(allowed, list):
                return [str(d) for d in allowed]
            return [str(allowed)]
    return None


def config_view_from_db_rows(
    agents: List[dict],
    agent_run_configs: List[dict],
) -> List[AgentConfigView]:
    cfg_by_name = {c["agent_name"]: c for c in (agent_run_configs or [])}
    views: List[AgentConfigView] = []

    for a in agents:
        name = a["agent_name"]
        cfg = cfg_by_name.get(name, {})
        parsed = parse_agent_run_config_row(cfg)
        allowed_domains = extract_web_search_allowed_domains_from_tools_cfg(
            parsed["tools_config_json"]
        )

        views.append(
            AgentConfigView(
                agent_name=name,
                model=parsed["model"] or a.get("model") or "unknown",
                role=parsed["role"],
                tools_enabled=parsed["tools_enabled"],
                tools=parsed["enabled_tools"],
                web_search_allowed_domains=allowed_domains,
                reasoning_effort=parsed["reasoning_effort"],
                reasoning_summary=parsed["reasoning_summary"],
                raw_tools_config=parsed["tools_config_json"] or None,
                raw_reasoning_config=parsed["reasoning_config_json"] or None,
                raw_extra_config=parsed["extra_config_json"] or None,
            )
        )
    return views
