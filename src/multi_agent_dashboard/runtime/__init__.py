"""
Runtime subpackage for agent execution logic and orchestration.

This package provides the core execution engine for running individual agents
and orchestrating multi-agent pipelines. It handles the complete lifecycle
from input processing through LLM invocation to output parsing and metrics
extraction.

Key modules:
- `agent_runtime.py`: Main AgentRuntime class for executing agents with
  instrumentation, tool calling, and structured output support
- `file_processor.py`: Text/binary file separation and content decoding for
  multimodal inputs
- `tool_converter.py`: Tool configuration merging and provider-specific
  adaptation
- `metrics_extractor.py`: Token usage, cost, and provider profile extraction
  from LLM responses
- `structured_output_detector.py`: 4-path detection and JSON schema validation
  for structured output parsing
- `utils.py`: Utility functions for safe template formatting and error handling

The runtime package works closely with the `engine` package for orchestration
and the `llm_client` package for provider-specific LLM integration.
"""

from .agent_runtime import AgentRuntime
from .utils import safe_format, SafeTemplate

__all__ = ["AgentRuntime", "safe_format", "SafeTemplate"]