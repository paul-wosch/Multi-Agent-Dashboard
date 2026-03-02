"""
Structured output validation against JSON schemas.

This module provides the SchemaValidator class which validates agent outputs against
configured JSON schemas. It supports schema resolution (from schema_json or schema_name),
JSON validation using jsonschema, and strict‑validation‑exit detection for early
pipeline termination when validation fails.

The validator handles various validation states including missing schemas, invalid
JSON, empty schemas, and validation errors.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from jsonschema import validate as jsonschema_validate, ValidationError  # type: ignore

from multi_agent_dashboard.llm_client import LLMClient
from multi_agent_dashboard.models import AgentSpec
from multi_agent_dashboard.shared.structured_schemas import resolve_schema_json

logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Validates an agent's output against its configured JSON schema.

    Supports:
      - Schema resolution (from schema_json or schema_name)
      - JSON validation using jsonschema
      - Strict‑validation‑exit detection (caller must handle early exit)
    """

    def _schema_resolution_state(self, spec: AgentSpec) -> Dict[str, Any]:
        """Determine whether a schema is configured and can be resolved."""
        cfg_schema_json = getattr(spec, "schema_json", None)
        cfg_schema_name = getattr(spec, "schema_name", None)
        configured = bool(cfg_schema_json) or bool(cfg_schema_name)
        if not configured:
            return {"status": "missing", "schema": None, "error": "Schema not configured"}
        try:
            schema = resolve_schema_json(cfg_schema_json, cfg_schema_name)
        except Exception as e:
            return {"status": "invalid_json", "schema": None, "error": str(e)}
        if schema is None:
            return {"status": "invalid_json", "schema": None, "error": "Schema could not be resolved"}
        if isinstance(schema, dict) and len(schema) == 0:
            return {"status": "empty", "schema": schema, "error": "Schema resolved to empty object"}
        return {"status": "resolved", "schema": schema, "error": None}

    def validate(
        self,
        agent_spec: AgentSpec,
        parsed: Optional[Any],
        raw_output: Any,
        strict_schema_validation: bool,
    ) -> Tuple[str, str]:
        """
        Validate the agent's output against its configured schema.

        Args:
            agent_spec: The agent specification (contains schema_json/schema_name).
            parsed: The parsed output (dict, list, or None). If None, raw_output will be used.
            raw_output: The raw output (string or any) as fallback.
            strict_schema_validation: Whether strict validation is enabled (affects error message).

        Returns:
            (status, reason)
            - status: one of "ok", "missing", "invalid_json", "empty", "validation_error"
            - reason: human‑readable explanation (empty string if status == "ok")
        """
        if not getattr(agent_spec, "structured_output_enabled", False):
            # No validation required
            return "ok", ""

        res = self._schema_resolution_state(agent_spec)
        if res["status"] != "resolved":
            # status could be "missing", "invalid_json", "empty"
            return res["status"], res.get("error") or res["status"]

        schema = res["schema"]
        candidate = parsed if isinstance(parsed, dict) else LLMClient.safe_json(raw_output) if isinstance(raw_output, str) else None
        if candidate is None:
            return "validation_error", "No JSON payload to validate"

        try:
            jsonschema_validate(candidate, schema)
            return "ok", ""
        except ValidationError as ve:
            return "validation_error", str(ve).splitlines()[0]
        except Exception as ve:
            return "validation_error", str(ve)