"""
In‑memory caching for provider model data with thread‑safe initialization.

This module provides a singleton cache for ProviderModel instances loaded from
external data sources. It uses double‑checked locking for thread‑safe lazy
initialization and composite keys ('provider|model_id') to disambiguate
duplicate model IDs across different providers.

Key Features:
- Thread‑safe singleton pattern with double‑checked locking
- Composite cache keys: 'provider|model_id' (e.g., 'openai|gpt-4o')
- Lazy initialization on first access
- Cache clearing for testing purposes

Architecture:
    ┌─────────────────┐
    │   get_model_cache()  │ ← Thread‑safe lazy initialization
    └─────────┬───────┘
              │
    ┌─────────▼───────┐
    │   _model_cache   │ ← Dict['provider|model_id' → ProviderModel]
    └─────────────────┘

Usage:
    from multi_agent_dashboard.provider_data.cache import get_model_cache
    
    cache = get_model_cache()
    model = cache.get('openai|gpt-4o')
    if model:
        print(f"Model supports tool calling: {model.tool_calling}")

Note: The cache is populated by calling load_provider_models() from the
loader module. This happens automatically on first access.
"""
import logging
import threading
from typing import Dict, Optional

from .schemas import ProviderModel
from .loader import load_provider_models

logger = logging.getLogger(__name__)

# Module‑level lock for thread‑safe cache initialization
_cache_lock = threading.Lock()

# In‑memory cache of composite key 'provider|model_id' → ProviderModel instances
_model_cache: Optional[Dict[str, ProviderModel]] = None


def get_model_cache() -> Dict[str, ProviderModel]:
    """
    Return the model cache, initializing it if necessary.

    The cache keys are composite strings 'provider|model_id' to disambiguate
    duplicate model IDs across providers.

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