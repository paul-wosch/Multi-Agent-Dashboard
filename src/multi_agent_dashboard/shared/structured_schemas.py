# structured_schemas.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional

# Registry of reusable schemas (Pydantic BaseModel classes or JSON schema dicts)
SCHEMA_REGISTRY: Dict[str, Any] = {}


def register_schema(name: str, schema: Any) -> None:
    """
    Register a schema by name for reuse across agents.
    The schema can be a Pydantic BaseModel class or a JSON schema dict.
    """
    if not name:
        raise ValueError("Schema name must be non-empty")
    SCHEMA_REGISTRY[name] = schema


def get_schema(name: str) -> Any:
    """
    Lookup a schema by name from the registry. Returns None if not found.
    """
    return SCHEMA_REGISTRY.get(name)


def schema_to_json(schema: Any) -> Optional[Dict[str, Any]]:
    """
    Convert a Pydantic model class (v1/v2) or JSON schema dict into JSON schema dict.
    Returns None if conversion fails.
    """
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    # Pydantic v2
    if hasattr(schema, "model_json_schema"):
        try:
            return schema.model_json_schema()
        except Exception:
            return None
    # Pydantic v1
    if hasattr(schema, "schema") and callable(getattr(schema, "schema")):
        try:
            return schema.schema()
        except Exception:
            return None
    return None


def resolve_schema_json(schema_json: Optional[str], schema_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Resolve a JSON schema dict from either an explicit JSON schema string
    or a registered schema name.
    """
    if schema_json:
        try:
            parsed = json.loads(schema_json)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    if schema_name:
        return schema_to_json(get_schema(schema_name))
    return None
