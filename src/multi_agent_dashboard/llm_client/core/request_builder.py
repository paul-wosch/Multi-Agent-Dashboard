"""
Request building for LLM client.

This module encapsulates the logic for constructing agent invocation states,
including multimodal file handling and message formatting.
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class RequestBuilder:
    """
    Builds the request state for agent.invoke calls.
    
    Handles multimodal file processing and message construction,
    with fallback to text concatenation when the multimodal module is unavailable.
    """
    
    def __init__(
        self,
        langchain_available: bool,
        SystemMessage: Optional[type],
        HumanMessage: Optional[type],
    ):
        """
        Args:
            langchain_available: Whether LangChain is available.
            SystemMessage: The SystemMessage class (or None if not available).
            HumanMessage: The HumanMessage class (or None if not available).
        """
        self._langchain_available = langchain_available
        self._SystemMessage = SystemMessage
        self._HumanMessage = HumanMessage
    
    def build(
        self,
        agent: Any,
        prompt: str,
        *,
        files: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build input with files, apply multimodal handling.
        
        Args:
            agent: LangChain agent instance
            prompt: Text prompt to send to the agent
            files: Optional list of file attachments (dicts with filename, content, mime_type)
            context: Optional context dictionary (currently unused)
            
        Returns:
            State dictionary for agent.invoke containing messages array
            
        Raises:
            RuntimeError: If LangChain is not available
        """
        if not self._langchain_available:
            raise RuntimeError("LangChain invoke_agent not available")

        combined_prompt = str(prompt or "")
        multimodal_content_parts = None  # If list, use this instead of combined_prompt
        processed_files = []

        if files:
            files_processed = False
            # Try to use multimodal handler regardless of provider
            try:
                from multi_agent_dashboard.llm_client.multimodal import prepare_multimodal_content
            except ImportError:
                logger.warning("multimodal_handler not available, falling back to text concatenation")
                prepare_multimodal_content = None

            if prepare_multimodal_content:
                provider_id = getattr(agent, "_provider_id", None)
                model = getattr(agent, "_model", None)
                provider_features = getattr(agent, "_provider_features", None)
                content, processed_files = prepare_multimodal_content(
                    provider_id=provider_id,
                    model=model,
                    files=files,
                    profile=provider_features,
                    prompt=combined_prompt,
                )
                if isinstance(content, list):
                    multimodal_content_parts = content
                    combined_prompt = ""  # not used
                else:
                    combined_prompt = content  # string
                files_processed = True
            # If prepare_multimodal_content is None, fall through to legacy concatenation

            if not files_processed:
                # Legacy concatenation (original logic)
                for f in files:
                    filename = f.get("filename", "file")
                    content = f.get("content")
                    try:
                        if isinstance(content, (bytes, bytearray)):
                            text = content.decode("utf-8", errors="replace")
                            combined_prompt += f"\n\n--- FILE: {filename} ---\n{text}"
                        else:
                            combined_prompt += f"\n\n--- FILE: {filename} ---\n{str(content)}"
                    except Exception:
                        combined_prompt += f"\n\n--- FILE: {filename} (binary not attached) ---\n"

        # Build the state expected by agent.invoke (messages array)
        # Determine user message content based on multimodal handling
        if multimodal_content_parts is not None:
            user_content = multimodal_content_parts
        else:
            user_content = combined_prompt

        state = {
            "messages": [
                self._SystemMessage(getattr(agent, "system_prompt", "") or "") if (
                            getattr(agent, "system_prompt", None) and self._SystemMessage) else None,
                self._HumanMessage(user_content) if self._HumanMessage else {"role": "user", "content": user_content},
            ]
        }
        # Clean None if we didn't build a SystemMessage instance
        state["messages"] = [m for m in state["messages"] if m is not None]
        
        return state