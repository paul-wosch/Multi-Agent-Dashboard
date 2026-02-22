# engine.py - backward compatibility wrapper for the engine/ package
from __future__ import annotations

from .engine.engine_orchestrator import MultiAgentEngine, EngineResult

__all__ = ["MultiAgentEngine", "EngineResult"]