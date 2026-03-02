"""
Engine package for modular multi-agent orchestration.

This package provides the core orchestration engine for multi-agent LLM pipelines,
separated from UI concerns for reuse in CLI tools, tests, and batch jobs.

Maintains backward compatibility with existing imports.
"""

from .engine_orchestrator import MultiAgentEngine, EngineResult

__all__ = ["MultiAgentEngine", "EngineResult"]