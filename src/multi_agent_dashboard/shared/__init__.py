"""
Shared utilities and cross-cutting concerns for the multi-agent dashboard.

This package contains functionality shared between the engine, runtime, and UI
packages, providing consistent patterns for instrumentation, capability detection,
runtime coordination, and schema management.

Key modules:
- `instrumentation.py`: Helper functions for extracting and processing
  instrumentation events from LLM responses
- `provider_capabilities.py`: Advisory capability mapping for LLM providers
  (used for warnings and UI defaults only)
- `runtime_hooks.py`: Hook registry for agent-change notifications between
  database/services and the UI runtime
- `structured_schemas.py`: Schema registry and JSON schema resolution for
  structured output validation

These modules enable loose coupling between different parts of the system while
maintaining consistent behavior across the execution pipeline.
"""

from .provider_capabilities import (
    get_capabilities,
    supports_feature,
)

from .runtime_hooks import (
    register_agent_change_handlers,
    clear_agent_change_handlers,
    on_agent_change,
)

from .structured_schemas import (
    SCHEMA_REGISTRY,
    register_schema,
    get_schema,
    schema_to_json,
    resolve_schema_json,
)

__all__ = [
    "get_capabilities",
    "supports_feature",
    "register_agent_change_handlers",
    "clear_agent_change_handlers",
    "on_agent_change",
    "SCHEMA_REGISTRY",
    "register_schema",
    "get_schema",
    "schema_to_json",
    "resolve_schema_json",
]