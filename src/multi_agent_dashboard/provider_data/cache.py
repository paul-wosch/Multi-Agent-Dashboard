"""
In‑memory caching for provider model data.

Provides thread‑safe access to the parsed ProviderModel dictionary.
"""
import logging
import threading
from typing import Dict, Optional

from .schemas import ProviderModel
from .loader import load_provider_models

logger = logging.getLogger(__name__)

# Module‑level lock for thread‑safe cache initialization
_cache_lock = threading.Lock()

# In‑memory cache of model_id → ProviderModel instances
_model_cache: Optional[Dict[str, ProviderModel]] = None


def get_model_cache() -> Dict[str, ProviderModel]:
    """
    Return the model cache, initializing it if necessary.

    Thread‑safe: uses double‑checked locking with _cache_lock.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    with _cache_lock:
        if _model_cache is None:
            logger.debug("Loading provider models into cache")
            _model_cache = load_provider_models()
            logger.info(f"Cached {len(_model_cache)} provider models")
        return _model_cache


def clear_cache() -> None:
    """Clear the in‑memory cache (mostly for testing)."""
    global _model_cache
    with _cache_lock:
        _model_cache = None
    logger.debug("Provider model cache cleared")