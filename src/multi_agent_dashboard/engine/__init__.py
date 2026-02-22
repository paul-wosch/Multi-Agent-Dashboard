# Engine package for modular multi-agent orchestration
# Maintains backward compatibility with existing imports

from .engine_orchestrator import MultiAgentEngine, EngineResult

__all__ = ["MultiAgentEngine", "EngineResult"]