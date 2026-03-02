"""
Extract and filter provider model data from raw downloaded JSON.

This module processes the raw provider_models_all.json file, filtering for
supported providers (currently OpenAI and DeepSeek). It preserves the complete
nested structure of provider entries and writes the filtered data to
provider_models.json, but only if the output file doesn't already exist
(idempotent operation).

Key Features:
- Idempotent extraction: won't overwrite existing provider_models.json
- Preserves complete nested structure of provider entries
- Filters only for supported providers (configurable list)
- No schema mapping at this stage (deferred to loader)

Workflow:
    1. Check if provider_models.json already exists → return existing data
    2. Read raw provider_models_all.json
    3. Filter for supported providers (openai, deepseek)
    4. Copy entire provider entry including all fields
    5. Write filtered dictionary to provider_models.json
    6. Return filtered data

File State Management:
    - If provider_models.json exists: return it, skip extraction
    - If provider_models_all.json missing: raise FileNotFoundError
    - If output file missing: create it from filtered data

Supported Providers:
    Currently: 'openai', 'deepseek'
    (Extensible by modifying the `supported_providers` tuple)

Usage:
    from multi_agent_dashboard.provider_data.extractor import extract_provider_models
    
    filtered_data = extract_provider_models()
    print(f"Extracted {len(filtered_data)} providers")
    
    # Access OpenAI models
    openai_models = filtered_data.get('openai', {}).get('models', {})

Note: This module performs minimal processing - no field mapping or schema
validation. The raw structure is preserved for the loader module to parse
into ProviderModel instances with proper schema mapping.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from multi_agent_dashboard.config import (
    DATA_PATH,
    PROVIDER_DATA_DIR,
    PROVIDER_MODELS_ALL_FILE,
    PROVIDER_MODELS_FILE,
)
from .schemas import ProviderModel

logger = logging.getLogger(__name__)


def extract_provider_models() -> Dict[str, Dict[str, Any]]:
    """
    Filter provider entries from raw JSON and return filtered copy.

    Reads the raw provider_models_all.json, processes only the 'openai' and
    'deepseek' top‑level keys, copies the entire provider entry (including all
    fields) preserving the original nested structure.

    Returns:
        Dictionary with provider keys mapping to the complete provider entry
        (same structure as in api.json).

    Raises:
        FileNotFoundError: If provider_models_all.json does not exist.
        ValueError: If JSON is malformed.
    """
    # Compute file paths
    provider_data_dir = DATA_PATH.parent / PROVIDER_DATA_DIR
    raw_file = provider_data_dir / PROVIDER_MODELS_ALL_FILE
    output_file = provider_data_dir / PROVIDER_MODELS_FILE

    # Check if output already exists – if it does, we should not overwrite
    if output_file.exists():
        logger.info(
            f"{PROVIDER_MODELS_FILE} already exists, extraction skipped. "
            f"Delete the file to force re‑extraction."
        )
        # Still load and return the existing extracted data for consistency
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw provider models file not found: {raw_file}. "
            f"Run download_provider_models_all() first."
        )

    logger.info(f"Extracting provider models from {raw_file}")

    with open(raw_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Filter for openai and deepseek (the only providers we currently support)
    supported_providers = ("openai", "deepseek")
    extracted: Dict[str, Dict[str, Any]] = {}

    for provider_key in supported_providers:
        if provider_key not in raw_data:
            logger.warning(f"Provider '{provider_key}' not found in raw data")
            continue

        provider_entry = raw_data[provider_key]
        if not isinstance(provider_entry, dict):
            raise ValueError(
                f"Provider '{provider_key}' entry is not a dictionary"
            )

        # Copy the entire provider entry
        extracted[provider_key] = provider_entry

        # Count models for logging
        models_dict = provider_entry.get("models", {})
        if not isinstance(models_dict, dict):
            logger.warning(
                f"Provider '{provider_key}' entry 'models' field is not a dictionary"
            )
            model_count = 0
        else:
            model_count = len(models_dict)

        logger.debug(
            f"Copied {model_count} models from provider '{provider_key}'"
        )

    total_models = sum(
        len(provider.get("models", {}))
        for provider in extracted.values()
        if isinstance(provider.get("models"), dict)
    )
    logger.info(
        f"Copied {total_models} total models from raw data"
    )

    # Write the filtered dictionary to disk (only if we created new data)
    if extracted:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote filtered provider models to {output_file}")

    return extracted
