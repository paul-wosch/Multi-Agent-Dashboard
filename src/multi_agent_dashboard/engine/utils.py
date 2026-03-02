"""
Utility functions for the engine package.

Consolidated helper functions extracted from engine.py and models.py for
content block normalization and provider feature extraction.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _normalize_content_blocks(blocks: List[Any]) -> List[Dict[str, Any]]:
    """
    Ensure each content block is a serializable dict (best-effort).
    """
    out_blocks: List[Dict[str, Any]] = []
    if not isinstance(blocks, list):
        return []
    for b in blocks:
        if isinstance(b, dict):
            out_blocks.append(b)
            continue
        try:
            if hasattr(b, "model_dump"):
                out_blocks.append(b.model_dump())
            elif hasattr(b, "to_dict"):
                out_blocks.append(b.to_dict())
            elif hasattr(b, "__dict__"):
                out_blocks.append(dict(b.__dict__))
            else:
                out_blocks.append({"__repr": repr(b)})
        except Exception:
            out_blocks.append({"__repr": repr(b)})
    return out_blocks


def _extract_provider_features_from_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a LangChain model 'profile' into a compact provider_features mapping.

    This is intentionally conservative: only expose a few well-known capability hints
    used by the UI (structured_output, tool_calling, reasoning, image_inputs, max_input_tokens).
    """

    features: Dict[str, Any] = {}

    if not isinstance(profile, dict):
        return features

    # Normalize profile keys to handle camelCase, snake_case, and lower-case variants.
    def _normalize_key(k: str) -> str:
        # Convert camelCase / PascalCase to snake_case
        s = re.sub(r'(?<!^)(?=[A-Z])', '_', str(k)).lower()
        s = s.replace('-', '_')
        return s

    normalized: Dict[str, Any] = {}
    for k, v in profile.items():
        try:
            normalized[str(k)] = v
        except Exception:
            normalized[k] = v
        try:
            normalized[str(k).lower()] = v
        except Exception:
            pass
        try:
            nk = _normalize_key(str(k))
            normalized[nk] = v
        except Exception:
            pass

    # Structured output related hints
    if normalized.get("structured_output") or normalized.get("structuredoutput") or normalized.get("reasoning_output") or normalized.get("structured"):
        features["structured_output"] = True

    # Tool calling hints
    if normalized.get("tool_calling") or normalized.get("toolcalling") or normalized.get("tool_calls") or normalized.get("toolcalls") or normalized.get("tool_call"):
        features["tool_calling"] = True

    # Reasoning hints
    if normalized.get("reasoning") or normalized.get("reasoning_output") or normalized.get("reasoningoutput") or normalized.get("supports_reasoning"):
        features["reasoning"] = True

    # Image / multimodal hints
    if "image_inputs" in normalized or "imageinputs" in normalized:
        try:
            features["image_inputs"] = bool(normalized.get("image_inputs") or normalized.get("imageinputs"))
        except Exception:
            features["image_inputs"] = True

    # Max input tokens (context window) — try variants
    max_tokens_candidates = [
        normalized.get("max_input_tokens"),
        normalized.get("maxinputtokens"),
        normalized.get("max_input_token"),
        normalized.get("max_input"),
        normalized.get("maxInputTokens"),
    ]
    for candidate in max_tokens_candidates:
        if candidate is not None:
            try:
                features["max_input_tokens"] = int(candidate)
            except Exception:
                features["max_input_tokens"] = candidate
            break

    # If nothing obvious matched, expose a shallow copy for auditing
    if not features and profile:
        # Keep only a small subset to avoid clobbering DB with huge dicts
        keys_to_copy = ["tool_calling", "structured_output", "reasoning", "image_inputs", "max_input_tokens", "maxInputTokens", "structuredOutput", "toolCalling"]
        for k in keys_to_copy:
            if k in profile:
                features[k if "_" in k else _normalize_key(k)] = profile[k]

    return features
