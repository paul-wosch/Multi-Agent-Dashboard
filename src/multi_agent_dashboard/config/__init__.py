"""
Configuration package for the Multi-Agent Dashboard.

This package provides centralized configuration management with YAML-based
configuration files, environment variable overrides, and derived path resolution.
It replaces the previous monolithic config.py with a modular, validated approach.

Modules:
- core: Loads .env, YAML, computes derived paths, and exposes global constants
- loader: YAML validation with Pydantic models

The public API exposes exactly the same names as the previous config.py module
for backward compatibility.
"""
from .core import (
    UI_COLORS,
    DB_FILE_PATH,
    AGENT_INPUT_CHAR_CAP,
    AGENT_OUTPUT_CHAR_CAP,
    AGENT_OUTPUT_TOKEN_CAP,
    AGENT_SNAPSHOTS_AUTO,
    AGENT_SNAPSHOT_PRUNE_AUTO,
    AGENT_SNAPSHOT_PRUNE_KEEP,
    ATTACHMENT_FILE_TYPES_RESTRICTED,
    ATTACHMENT_FILE_TYPES,
    OPENAI_API_KEY,
    DEEPSEEK_API_KEY,
    LOG_LEVEL,
    RAISE_ON_AGENT_FAIL,
    STRICT_OUTPUT_TOKEN_CAP_OVERRIDE,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_BASE_URL,
    LANGFUSE_ENABLED,
    configure_logging,
    # Path constants
    DATA_DIR,
    DB_DIR,
    MIGRATIONS_DIR,
    LOGS_DIR,
    DOTENV_FILE,
    LOG_FILE,
    DB_FILE,
    PROVIDER_DATA_DIR,
    DATA_PATH,
    DB_PATH,
    MIGRATIONS_PATH,
    LOGS_PATH,
    DOTENV_FILE_PATH,
    PROVIDER_DATA_PATH,
    LOG_FILE_PATH,
    # Provider‑data constants
    PROVIDER_MODELS_ALL_FILE,
    PROVIDER_MODELS_FILE,
    TEMPLATE_OLLAMA_MODELS_FILE,
    LOCAL_OLLAMA_MODELS_FILE,
    MODELS_DEV_URL,
)

__all__ = [
    "UI_COLORS",
    "DB_FILE_PATH",
    "AGENT_INPUT_CHAR_CAP",
    "AGENT_OUTPUT_CHAR_CAP",
    "AGENT_OUTPUT_TOKEN_CAP",
    "AGENT_SNAPSHOTS_AUTO",
    "AGENT_SNAPSHOT_PRUNE_AUTO",
    "AGENT_SNAPSHOT_PRUNE_KEEP",
    "ATTACHMENT_FILE_TYPES_RESTRICTED",
    "ATTACHMENT_FILE_TYPES",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "LOG_LEVEL",
    "RAISE_ON_AGENT_FAIL",
    "STRICT_OUTPUT_TOKEN_CAP_OVERRIDE",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_ENABLED",
    "configure_logging",
    "DATA_DIR",
    "DB_DIR",
    "MIGRATIONS_DIR",
    "LOGS_DIR",
    "DOTENV_FILE",
    "LOG_FILE",
    "DB_FILE",
    "PROVIDER_DATA_DIR",
    "DATA_PATH",
    "DB_PATH",
    "MIGRATIONS_PATH",
    "LOGS_PATH",
    "DOTENV_FILE_PATH",
    "PROVIDER_DATA_PATH",
    "LOG_FILE_PATH",
    "PROVIDER_MODELS_ALL_FILE",
    "PROVIDER_MODELS_FILE",
    "TEMPLATE_OLLAMA_MODELS_FILE",
    "LOCAL_OLLAMA_MODELS_FILE",
    "MODELS_DEV_URL",
]