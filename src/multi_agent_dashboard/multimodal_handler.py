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

from multi_agent_dashboard import provider_capabilities

# Optional PDF extraction library
try:
    import pypdf
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    pypdf = None
    PDF_EXTRACTION_AVAILABLE = False

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

# PDF MIME types
PDF_MIME_TYPES = {
    "application/pdf",
}

# Maximum file size for base64 encoding (10 MB) to avoid memory issues
MAX_BASE64_SIZE = 10 * 1024 * 1024


@lru_cache(maxsize=128)
def provider_supports_vision(provider_id: str, model: str) -> bool:
    """
    Cached check whether a provider/model supports vision (image inputs).
    """
    logger.debug(f"Checking vision support for provider={provider_id}, model={model}")
    result = provider_capabilities.supports_feature(provider_id, "vision", model)
    logger.info(f"Vision support for provider={provider_id}, model={model}: {result}")
    return result


@lru_cache(maxsize=128)
def provider_supports_tools(provider_id: str, model: str) -> bool:
    """
    Cached check whether a provider/model supports tool attachments.
    """
    logger.debug(f"Checking tool support for provider={provider_id}, model={model}")
    result = provider_capabilities.supports_feature(provider_id, "tools", model)
    logger.info(f"Tool support for provider={provider_id}, model={model}: {result}")
    return result


def build_message_parts_for_provider(
    files: List[Dict[str, Any]],
    provider_id: str,
    model: str,
    profile: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build LangChain-compatible message parts from files for a given provider.

    Args:
        files: List of file dicts with keys "filename", "content" (bytes), "mime_type"
        provider_id: Provider identifier (openai, ollama, deepseek)
        model: Model name (e.g., "gpt-4o", "llama3")
        profile: Optional provider features dict (e.g., {"image_inputs": True})

    Returns:
        tuple (content_parts, processed_files):
        - content_parts: List of content parts in OpenAI-style format
          ({"type": "text", "text": ...} or {"type": "image_url", "image_url": {...}})
        - processed_files: List of files that have been processed as attachments
          (base64 encoded). Empty list if fallback to text concatenation.
    """
    # Delegate to prepare_multimodal_content with empty prompt
    content, processed_files = prepare_multimodal_content(
        provider_id=provider_id,
        model=model,
        files=files,
        profile=profile,
        prompt="",
    )
    # Ensure content is a list (if no files, content is empty string)
    if isinstance(content, str):
        # If empty string, return empty list; otherwise wrap as text part
        if content:
            return [{"type": "text", "text": content}], processed_files
        else:
            return [], processed_files
    return content, processed_files


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
    logger.debug(f"Determining vision support for provider={provider_id}, model={model}")
    vision_supported = False
    if profile is not None and "image_inputs" in profile:
        vision_supported = bool(profile["image_inputs"])
        logger.info(f"Vision support from profile: {vision_supported}")
    else:
        vision_supported = provider_supports_vision(provider_id, model)
        logger.info(f"Vision support from detection: {vision_supported}")



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
            if vision_supported:
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
            else:
                # Vision not supported: add placeholder text
                content_parts.append(
                    {"type": "text", "text": f"--- IMAGE: {filename} (vision not supported) ---"}
                )
                continue
        
        elif mime_type in TEXT_MIME_TYPES:
            # Text file: decode to UTF-8 and include as text part
            try:
                text = content.decode("utf-8", errors="replace")
                content_parts.append(
                    {"type": "text", "text": f"--- FILE: {filename} ---\n{text}"}
                )
                continue
            except Exception as e:
                logger.warning("Failed to decode text file %s: %s", filename, e)
                # fall through to placeholder

        # Attempt to extract text from PDFs or decode binary as text
        if mime_type in PDF_MIME_TYPES and PDF_EXTRACTION_AVAILABLE:
            try:
                import io
                pdf_file = io.BytesIO(content)
                reader = pypdf.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                content_parts.append(
                    {"type": "text", "text": f"--- FILE: {filename} ---\n{text}"}
                )
                continue
            except Exception as e:
                logger.warning("Failed to extract text from PDF %s: %s", filename, e)
                # fall through to generic binary handling
        
        # Generic binary fallback: attempt to decode as UTF-8 with replacement
        try:
            text = content.decode("utf-8", errors="replace")
            content_parts.append(
                {"type": "text", "text": f"--- FILE: {filename} ---\n{text}"}
            )
            continue
        except Exception as e:
            logger.warning("Failed to decode binary file %s: %s", filename, e)
            # Ultimate fallback: placeholder
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
    "build_message_parts_for_provider",
    "provider_supports_vision",
    "provider_supports_tools",
    "get_supported_mime_types",
    "VISION_MIME_TYPES",
    "TEXT_MIME_TYPES",
]