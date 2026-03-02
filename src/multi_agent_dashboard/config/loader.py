"""
YAML configuration loader with Pydantic validation.

This module provides structured loading and validation of YAML configuration
files using Pydantic models. It defines schema classes for each configuration
domain and validates them at load time.

Configuration domains:
- Paths: Directory and file names for data storage
- Agents: Agent-specific limits and snapshot settings
- Providers: Provider data file names and external URLs
- UI: User interface settings (colors, file types)
- Logging: Default log level and logging configuration

The module ensures configuration consistency and provides early error detection
for malformed or missing configuration values.
"""
import yaml
from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel, ValidationError


class PathsConfig(BaseModel):
    data_dir: str
    db_dir: str
    migrations_dir: str
    logs_dir: str
    dotenv_file: str
    log_file: str
    db_file: str
    provider_data_dir: str


class AgentsConfig(BaseModel):
    agent_input_char_cap: int
    agent_output_char_cap: int
    agent_output_token_cap: int
    agent_snapshots_auto: bool
    agent_snapshot_prune_auto: bool
    agent_snapshot_prune_keep: int


class ProvidersConfig(BaseModel):
    provider_models_all_file: str
    provider_models_file: str
    template_ollama_models_file: str
    local_ollama_models_file: str
    models_dev_url: str


class UIConfig(BaseModel):
    attachment_file_types_restricted: bool
    attachment_file_types: list[str]
    ui_colors: Dict[str, Dict[str, str]]


class LoggingConfig(BaseModel):
    default_log_level: str


def load_yaml_config(config_root: Path) -> Dict[str, Any]:
    """Load all YAML files from config_root and return validated dict."""
    config = {}
    
    # Helper to load and validate a single file
    def load_and_validate(filename: str, model_cls: type[BaseModel]) -> Dict[str, Any]:
        path = config_root / filename
        if not path.exists():
            raise FileNotFoundError(f"Config file {path} not found")
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return model_cls(**data).model_dump()
    
    try:
        config['paths'] = load_and_validate('paths.yaml', PathsConfig)
        config['agents'] = load_and_validate('agents.yaml', AgentsConfig)
        config['providers'] = load_and_validate('providers.yaml', ProvidersConfig)
        config['ui'] = load_and_validate('ui.yaml', UIConfig)
        config['logging'] = load_and_validate('logging.yaml', LoggingConfig)
    except ValidationError as e:
        raise ValueError(f"YAML validation error: {e}") from e
    
    return config