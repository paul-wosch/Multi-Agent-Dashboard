"""
Multimodal file handler for provider-agnostic file uploads.

This module handles conversion of binary files (images, PDFs, etc.) into
provider-appropriate formats (e.g., base64 for vision-capable providers).
It falls back to plain text concatenation when the provider does not support
multimodal attachments.
"""

import base64
import logging
import mimetypes
from functools import lru_cache
from typing import Dict, List, Any, Optional, Tuple, Union

from multi_agent_dashboard import litellm_config

logger = logging.getLogger(__name__)

# Supported image MIME types for vision providers
VISION_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
}

# MIME types that can be safely decoded as text
TEXT_MIME_TYPES = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "text/html",
    "text/xml",
}

# Maximum file size for base64 encoding (10 MB) to avoid memory issues
MAX_BASE64_SIZE = 10 * 1024 * 1024


@lru_cache(maxsize=128)
def provider_supports_vision(provider_id: str, model: str) -> bool:
    """
    Cached check whether a provider/model supports vision (image inputs).
    """
    return litellm_config.supports_feature(provider_id, "vision")


@lru_cache(maxsize=128)
def provider_supports_tools(provider_id: str, model: str) -> bool:
    """
    Cached check whether a provider/model supports tool attachments.
    """
    return litellm_config.supports_feature(provider_id, "tools")


def prepare_multimodal_content(
    provider_id: str,
    model: str,
    files: List[Dict[str, Any]],
    profile: Optional[Dict[str, Any]] = None,
    prompt: str = "",
) -> Tuple[Union[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Prepare files for multimodal input if provider supports it.

    Args:
        provider_id: Provider identifier (openai, ollama, deepseek)
        model: Model name (e.g., "gpt-4o", "llama3")
        files: List of file dicts with keys "filename", "content" (bytes), "mime_type"
        profile: Optional provider features dict (e.g., {"image_inputs": True})
        prompt: Optional text prompt to include before files.

    Returns:
        tuple (content, processed_files):
        - content: Either a string (plain text prompt) or a list of content parts
          following OpenAI's multimodal message format.
        - processed_files: List of files that have been processed as attachments
          (base64 encoded). Empty list if fallback to text concatenation.
    """
    if not files:
        return prompt, []

    # Determine if provider supports vision based on profile or cached detection
    vision_supported = False
    if profile and profile.get("image_inputs"):
        vision_supported = True
    else:
        vision_supported = provider_supports_vision(provider_id, model)

    # If provider doesn't support vision, fall back to text concatenation
    if not vision_supported:
        logger.debug(
            "Provider %s/%s does not support vision; falling back to text concatenation",
            provider_id,
            model,
        )
        combined = prompt + _fallback_text_concat(files)
        return combined, []

    # Process files: encode images as base64, keep text files as text
    content_parts = []
    processed_files = []
    if prompt:
        content_parts.append({"type": "text", "text": prompt})
    for f in files:
        filename = f.get("filename", "file")
        mime_type = f.get("mime_type", "")
        content = f.get("content")
        if not isinstance(content, (bytes, bytearray)):
            # Already text? treat as string
            text = str(content)
            content_parts.append({"type": "text", "text": text})
            continue

        # Binary content
        if mime_type in VISION_MIME_TYPES:
            # Image file: encode as base64 if size limit allows
            if len(content) > MAX_BASE64_SIZE:
                logger.warning(
                    "Image file %s exceeds size limit (%d > %d), skipping attachment",
                    filename,
                    len(content),
                    MAX_BASE64_SIZE,
                )
                content_parts.append(
                    {"type": "text", "text": f"--- FILE: {filename} (image too large) ---"}
                )
                continue
            try:
                b64_data = base64.b64encode(content).decode("utf-8")
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
                    }
                )
                processed_files.append(
                    {
                        "filename": filename,
                        "mime_type": mime_type,
                        "content": b64_data,
                        "base64": True,
                    }
                )
                continue
            except Exception as e:
                logger.warning("Failed to base64 encode %s: %s", filename, e)

        # Non-image binary or unsupported MIME type: fallback to text placeholder
        content_parts.append(
            {"type": "text", "text": f"--- FILE: {filename} (binary not attached) ---"}
        )

    # If no image parts were added, we could still have text parts.
    # For now, return content_parts as list (OpenAI format).
    # If content_parts contains only one text part, we could flatten to string.
    # However, keep as list for consistency.
    return content_parts, processed_files


def _fallback_text_concat(files: List[Dict[str, Any]]) -> str:
    """
    Fallback method: concatenate text files as plain text with headers.
    Mimics the existing behavior in llm_client.py.
    """
    combined = ""
    for f in files:
        filename = f.get("filename", "file")
        content = f.get("content")
        try:
            if isinstance(content, (bytes, bytearray)):
                text = content.decode("utf-8", errors="replace")
            else:
                text = str(content)
            combined += f"\n\n--- FILE: {filename} ---\n{text}"
        except Exception:
            combined += f"\n\n--- FILE: {filename} (binary not attached) ---\n"
    return combined


def get_supported_mime_types(provider_id: str, model: str) -> List[str]:
    """
    Return a list of MIME types supported by the provider for multimodal input.
    """
    supported = []
    if provider_supports_vision(provider_id, model):
        supported.extend(VISION_MIME_TYPES)
    # Could add PDF, audio, etc. in the future
    return supported


# Export public interface
__all__ = [
    "prepare_multimodal_content",
    "provider_supports_vision",
    "provider_supports_tools",
    "get_supported_mime_types",
    "VISION_MIME_TYPES",
    "TEXT_MIME_TYPES",
]