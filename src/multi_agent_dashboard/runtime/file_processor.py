# File processing logic for agent runtime
# This module will contain file-type detection and content decoding functions.

from __future__ import annotations

from typing import Dict, List, Any, Tuple


def process_files(all_files: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Separate files into text and binary categories based on MIME type.
    
    Text files (plain text, markdown, CSV, JSON) are decoded to UTF‑8 and re‑encoded
    to ensure they can be safely inlined in LLM prompts. Binary files (PDF, images, audio)
    are passed through unchanged.
    
    Args:
        all_files: List of file dicts with keys 'filename', 'mime_type', 'content' (bytes).
    
    Returns:
        Tuple of (text_files, binary_files) where each list contains the same dicts
        with text files having their content re‑encoded as UTF‑8 bytes.
    """
    text_files: List[Dict[str, Any]] = []
    binary_files: List[Dict[str, Any]] = []
    
    for f in all_files:
        mime = f.get("mime_type", "")
        if mime in {"text/plain", "text/markdown", "text/csv", "application/json"}:
            # Inline small text files
            try:
                content = f["content"].decode("utf-8", errors="replace")
            except Exception:
                content = ""
            text_files.append({
                "filename": f["filename"],
                "mime_type": mime,
                "content": content.encode("utf-8"),
            })
        else:
            # Treat as binary (PDF, images, audio)
            binary_files.append(f)
    
    return text_files, binary_files