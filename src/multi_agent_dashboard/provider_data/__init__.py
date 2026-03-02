"""
Provider Data Package - Dynamic pricing and capability data management.

This package manages dynamic provider model data loaded from external sources
(models.dev) with optional local overrides for Ollama models. It provides
advisory capability detection and pricing information for LLM providers.

Key Features:
- Dynamic loading of provider capabilities (tool calling, structured output, vision, etc.)
- Pricing data (USD per 1M tokens) for cost calculation
- Composite cache keys ('provider|model') for disambiguation
- Local Ollama model overrides (git-ignored for user customization)
- Thread-safe caching with lazy initialization

Architecture:
1. Downloader: Fetches raw JSON from external source with retry logic
2. Extractor: Filters for supported providers (OpenAI, DeepSeek)
3. Loader: Parses JSON into ProviderModel instances, handles file-state machine
4. Cache: Thread-safe in-memory cache with composite keys
5. Schemas: Pydantic-like dataclasses for structured data

Usage:
    from multi_agent_dashboard.provider_data import (
        get_capabilities,
        get_pricing,
        get_capabilities_for_provider,
    )
    
    # Get advisory capabilities for a model
    caps = get_capabilities("gpt-4o")
    if caps.get("tool_calling"):
        print("Model supports tool calling")
    
    # Get pricing for cost calculation
    input_price, output_price = get_pricing("gpt-4o")

File Management:
    data/provider_models/
    ├── provider_models_all.json    # Raw downloaded data
    ├── provider_models.json        # Filtered copy (OpenAI, DeepSeek)
    ├── template_ollama_models.json # Template (git-tracked)
    └── local_ollama_models.json    # User overrides (git-ignored)

Note: Capability data is advisory only. Agent configuration remains the
primary source of truth for features like tool calling and structured output.
"""

from .loader import (
    download_provider_models_all,
    extract_provider_models,
    load_provider_models,
    get_all_models,
    get_capabilities,
    get_pricing,
    get_capabilities_for_provider,
    get_pricing_for_provider,
)

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