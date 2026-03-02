"""
Multimodal file handling sub-package for LLM providers.

This sub-package provides functionality for handling multimodal file inputs
(images, PDFs, text files) across different LLM providers. It converts
binary files into provider-appropriate formats and provides fallback
mechanisms for providers without native multimodal support.

Key functionality:
- File type detection and MIME type classification
- Provider-specific file format conversion (base64 for vision, text extraction)
- Vision capability detection for different providers
- Fallback to text concatenation for non-vision providers
- PDF text extraction when pypdf is available

The multimodal sub-package enables consistent file handling across
different LLM providers while respecting each provider's capabilities
and limitations.
"""
# Multimodal file handling for LLM providers

from .multimodal_handler import (
    prepare_multimodal_content,
    build_message_parts_for_provider,
    provider_supports_vision,
    provider_supports_tools,
    get_supported_mime_types,
    VISION_MIME_TYPES,
    TEXT_MIME_TYPES,
)

__all__ = [
    "prepare_multimodal_content",
    "build_message_parts_for_provider",
    "provider_supports_vision",
    "provider_supports_tools",
    "get_supported_mime_types",
    "VISION_MIME_TYPES",
    "TEXT_MIME_TYPES",
]