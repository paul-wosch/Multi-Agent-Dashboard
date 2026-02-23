# Shared utilities for multi-agent dashboard

from .provider_capabilities import (
    get_capabilities,
    supports_feature,
    PROVIDER_DEFAULT_CAPABILITIES,
    MODEL_CAPABILITIES,
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
    "PROVIDER_DEFAULT_CAPABILITIES",
    "MODEL_CAPABILITIES",
    "register_agent_change_handlers",
    "clear_agent_change_handlers",
    "on_agent_change",
    "SCHEMA_REGISTRY",
    "register_schema",
    "get_schema",
    "schema_to_json",
    "resolve_schema_json",
]