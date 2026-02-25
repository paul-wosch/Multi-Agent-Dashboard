"""
Load provider model data from disk.

Handles the file‑state machine and parses raw JSON into ProviderModel instances.
"""
import json
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from multi_agent_dashboard.config import (
    DATA_PATH,
    PROVIDER_DATA_DIR,
    PROVIDER_MODELS_ALL_FILE,
    PROVIDER_MODELS_FILE,
)
from .downloader import download_provider_models_all
from .extractor import extract_provider_models
from .schemas import ProviderModel

logger = logging.getLogger(__name__)
_file_lock = threading.Lock()


def _get_provider_data_dir() -> Path:
    """Return the absolute path to the provider data directory."""
    # Consistent with downloader.py and extractor.py
    return DATA_PATH.parent / PROVIDER_DATA_DIR


def _get_provider_models_all_path() -> Path:
    """Return absolute path of provider_models_all.json."""
    return _get_provider_data_dir() / PROVIDER_MODELS_ALL_FILE


def _get_provider_models_path() -> Path:
    """Return absolute path of provider_models.json."""
    return _get_provider_data_dir() / PROVIDER_MODELS_FILE


def _ensure_data_files() -> Path:
    """
    Ensure provider_models_all.json and provider_models.json exist.

    Implements the file‑state machine:
      - both missing → download → extract
      - provider_models_all.json missing, provider_models.json present → download (warn)
      - provider_models_all.json present, provider_models.json missing → extract
      - both present → no action

    Returns the path to provider_models.json (guaranteed to exist after this call).
    """
    with _file_lock:
        all_path = _get_provider_models_all_path()
        models_path = _get_provider_models_path()

        all_exists = all_path.exists()
        models_exists = models_path.exists()

        if not all_exists and not models_exists:
            logger.info("Both provider data files missing, downloading and extracting")
            download_provider_models_all()
            extract_provider_models()
        elif not all_exists and models_exists:
            logger.warning(
                f"{PROVIDER_MODELS_ALL_FILE} missing but {PROVIDER_MODELS_FILE} "
                f"present, re‑downloading raw data (extracted file kept unchanged)"
            )
            download_provider_models_all()
        elif all_exists and not models_exists:
            logger.info(
                f"{PROVIDER_MODELS_ALL_FILE} present but {PROVIDER_MODELS_FILE} "
                f"missing, extracting"
            )
            extract_provider_models()
        else:
            logger.debug("Both provider data files present, using existing")
        return models_path


def load_provider_models() -> Dict[str, ProviderModel]:
    """
    Load provider_models.json, parse each model into a ProviderModel,
    and return a dictionary mapping composite key 'provider|model_id' → ProviderModel.

    This function does not cache; the caller is responsible for caching.
    """
    models_path = _ensure_data_files()

    logger.debug(f"Loading provider models from {models_path}")
    with open(models_path, "r", encoding="utf-8") as f:
        provider_data = json.load(f)

    cache: Dict[str, ProviderModel] = {}
    for provider_key, provider_entry in provider_data.items():
        if not isinstance(provider_entry, dict):
            logger.warning(
                f"Provider entry '{provider_key}' is not a dictionary, skipping"
            )
            continue
        models_dict = provider_entry.get("models")
        if not isinstance(models_dict, dict):
            logger.warning(
                f"Provider '{provider_key}' has no 'models' dictionary, skipping"
            )
            continue

        for model_id, raw_model in models_dict.items():
            if not isinstance(raw_model, dict):
                logger.warning(
                    f"Model '{model_id}' in provider '{provider_key}' is not a "
                    f"dictionary, skipping"
                )
                continue
            try:
                provider_model = ProviderModel.from_raw_json(
                    provider=provider_key,
                    model_id=model_id,
                    raw=raw_model,
                )
                cache[f"{provider_key}|{model_id}"] = provider_model
            except Exception as e:
                logger.error(
                    f"Failed to parse model '{model_id}' from provider "
                    f"'{provider_key}': {e}"
                )
                # Continue with other models

    logger.info(f"Loaded {len(cache)} models")
    return cache


@lru_cache(maxsize=1)
def get_all_models() -> List[str]:
    """
    Return a list of all known model IDs.

    The list is sorted alphabetically. Duplicate model IDs across providers
    are deduplicated (only the first occurrence is kept).
    """
    from .cache import get_model_cache
    cache = get_model_cache()
    model_ids = set()
    for composite_key in cache.keys():
        # Split 'provider|model_id', keep the model_id part
        if "|" in composite_key:
            _, model_id = composite_key.split("|", 1)
        else:
            # fallback for backward compatibility (should not happen)
            model_id = composite_key
        model_ids.add(model_id)
    return sorted(model_ids)


def _find_provider_models(model_id: str) -> List["ProviderModel"]:
    """
    Find all ProviderModel instances matching a given model_id.

    Returns a (possibly empty) list of ProviderModel objects.
    """
    from .cache import get_model_cache
    from .schemas import ProviderModel
    cache = get_model_cache()
    matches = []
    for composite_key, provider_model in cache.items():
        # Split composite key 'provider|model_id'
        if "|" in composite_key:
            _, key_model_id = composite_key.split("|", 1)
        else:
            key_model_id = composite_key
        if key_model_id == model_id:
            matches.append(provider_model)
    return matches


@lru_cache(maxsize=None)  # unlimited cache per model
def get_capabilities(model_id: str) -> Dict[str, Any]:
    """
    Return the capability dictionary for a given model.

    If the model is unknown, logs a WARNING and returns an empty dictionary
    (all capabilities default to False, no token limits, empty knowledge).
    """
    matches = _find_provider_models(model_id)
    if not matches:
        logger.warning(f"Unknown model '{model_id}', returning empty capabilities")
        return {}
    if len(matches) > 1:
        logger.warning(
            f"Ambiguous model ID '{model_id}' matches {len(matches)} providers: "
            f"{[m.provider for m in matches]}. Using first match."
        )
    return matches[0].to_capability_dict()


@lru_cache(maxsize=None)
def get_pricing(model_id: str) -> Tuple[float, float]:
    """
    Return (input_price, output_price) in USD per 1M tokens.

    If the model is unknown, logs a WARNING and returns (0.0, 0.0).
    """
    matches = _find_provider_models(model_id)
    if not matches:
        logger.warning(f"Unknown model '{model_id}', returning zero pricing")
        return (0.0, 0.0)
    if len(matches) > 1:
        logger.warning(
            f"Ambiguous model ID '{model_id}' matches {len(matches)} providers: "
            f"{[m.provider for m in matches]}. Using first match."
        )
    return (matches[0].input_price, matches[0].output_price)


def get_capabilities_for_provider(provider_id: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Return capability dictionary for a specific provider and model.

    If model is None, returns an empty dictionary (no provider‑level defaults).
    If the provider/model combination is unknown, logs a WARNING and returns empty dict.
    """
    if model is None:
        return {}
    composite_key = f"{provider_id}|{model}"
    from .cache import get_model_cache
    cache = get_model_cache()
    provider_model = cache.get(composite_key)
    if provider_model is None:
        logger.warning(
            f"Unknown provider|model combination '{provider_id}|{model}', "
            "returning empty capabilities"
        )
        return {}
    return provider_model.to_capability_dict()


def get_pricing_for_provider(provider_id: str, model: Optional[str] = None) -> Tuple[float, float]:
    """
    Return (input_price, output_price) for a specific provider and model.

    If model is None, returns (0.0, 0.0) (no provider‑level defaults).
    If the provider/model combination is unknown, logs a WARNING and returns (0.0, 0.0).
    """
    if model is None:
        return (0.0, 0.0)
    composite_key = f"{provider_id}|{model}"
    from .cache import get_model_cache
    cache = get_model_cache()
    provider_model = cache.get(composite_key)
    if provider_model is None:
        logger.warning(
            f"Unknown provider|model combination '{provider_id}|{model}', "
            "returning zero pricing"
        )
        return (0.0, 0.0)
    return (provider_model.input_price, provider_model.output_price)


# Re‑export for convenience
__all__ = [
    "download_provider_models_all",
    "extract_provider_models",
    "load_provider_models",
    "get_all_models",
    "get_capabilities",
    "get_pricing",
    "get_capabilities_for_provider",
    "get_pricing_for_provider",
]