# Langfuse Observability Integration: Implementation Strategy

**Date**: 2026‑02‑27 (created); 2026-02-28 (completed)
**Based on**: `docs/implementation‑strategies/260227‑2100_integrate_langfuse_feasibility_report.md`  
**Confidence**: 99/100 (no technical blockers, optional integration, follows existing patterns)

## Executive Summary

This document provides a **progressive, incremental, step‑by‑step strategy** to integrate Langfuse v3.x.x observability into the Multi‑Agent Dashboard. The integration is **optional** (activated only when `LANGFUSE_*` environment variables are set) and **non‑invasive** (preserves all existing instrumentation and metrics). The approach follows the existing LangChain architecture, adding a single callback at the agent‑invocation level.

### Goals

- Add Langfuse tracing for all agent executions **without breaking existing functionality**
- Keep the integration **optional** – no required configuration for existing users
- Maintain **full compatibility** with existing provider‑agnostic LLM client, tool integration, and structured output
- Complement (not replace) the dashboard’s native metrics collection (`_DashboardInstrumentationMiddleware`)
- Provide a clear path for future expansion (model‑level tracing, custom metadata, UI toggles)

### Non‑Goals

- Replacing the existing instrumentation middleware
- Adding mandatory dependencies (Langfuse is an optional extra)
- Changing the public API of any module
- Supporting Langfuse v2.x.x (target v3.x.x only)

## Prerequisites

### 1. Langfuse Python SDK

The only new dependency is `langfuse>=3.0.0`. It will be added as a **regular dependency** with graceful import‑error handling (matching the existing LangChain pattern). The dashboard already uses a similar pattern for optional dependencies.

### 2. Environment Variables

The following environment variables must be defined to activate Langfuse:

| Variable              | Purpose                  | Required                                         |
| --------------------- | ------------------------ | ------------------------------------------------ |
| `LANGFUSE_PUBLIC_KEY` | Public API key           | Yes (if enabled)                                 |
| `LANGFUSE_SECRET_KEY` | Secret API key           | Yes (if enabled)                                 |
| `LANGFUSE_BASE_URL`   | Cloud or self‑hosted URL | Optional (default: `https://cloud.langfuse.com`) |
| `LANGFUSE_ENABLED`    | Master switch            | Optional (can be inferred from keys)             |

These will be added to `.env.template` and loaded via the existing configuration system (`config` package).

### 3. Familiarity with Current Architecture

- **LLMClient** (`src/multi_agent_dashboard/llm_client/core.py`): The main integration point is the `_execute_with_retry` method (lines 424‑443).
- **AgentRuntime** (`src/multi_agent_dashboard/runtime/agent_runtime.py`): Already captures metrics (`last_metrics`); Langfuse will run in parallel.
- **Context Metadata**: The `context` dict passed to agent invocations contains `run_id`, `pipeline_name`, and other metadata that maps directly to Langfuse tags/session IDs.

## Architecture Decisions

### Integration Point: Agent‑Invocation Level

**Location**: `LLMClient._execute_with_retry()`  
**Mechanism**: Pass a `config` dict with `callbacks: [langfuse_handler]` and `metadata` to `agent.invoke()`.

**Why this option**:

- Minimal change (single location).
- Captures entire agent execution (tools, reasoning, model calls).
- Metadata easily attached via existing `context`.
- Works with any LangChain agent regardless of underlying model.
- Aligns with the dashboard’s primary execution pattern.

### Optional Activation Pattern

Langfuse integration will **only** be activated when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. If either is missing, the code path behaves exactly as before (no callbacks, no extra overhead). This follows the existing pattern for optional dependencies (see `_LANGCHAIN_AVAILABLE`).

### Singleton Langfuse Client

Use `langfuse.get_client()` as recommended by Langfuse v3.x.x. The client is thread‑safe and manages its own connection pool. We will create a lightweight wrapper module (`observability/langfuse.py`) that initializes the singleton only when enabled.

### Metadata Mapping

| Dashboard Concept | Langfuse Field        | Source                                            |
| ----------------- | --------------------- | ------------------------------------------------- |
| Agent name        | `langfuse_tags`       | `agent._name` or `spec.name`                      |
| Pipeline name     | `langfuse_tags`       | `context.get("pipeline_name", "unknown")`         |
| Run ID            | `langfuse_session_id` | `context.get("run_id", "unknown")`                |
| User identifier   | `langfuse_user_id`    | Static (`"multi_agent_dashboard"`) or future auth |
| Custom fields     | `metadata`            | Tool config, reasoning config, provider profile   |

**Initial implementation** will use tags for agent, pipeline and session ID for run grouping. The user ID can be static (`"multi_agent_dashboard"`) for now.

### Complementary Observability

Langfuse traces will **coexist** with the existing `_DashboardInstrumentationMiddleware`. The two systems are complementary:

- **Dashboard middleware**: Extracts content blocks, token counts, provider profiles, tool usage for internal display.
- **Langfuse**: Provides external tracing, latency breakdown, cost tracking, and advanced analytics.

No existing instrumentation will be removed or altered.

## ✅ Phase 1: Foundation (Non‑Breaking)

**Objective**: Add dependency, extend configuration, create Langfuse client module. No functional changes yet.

### ✅ Step 1.1 – Add Langfuse Dependency

**File**: `pyproject.toml`  
**Action**: Add `langfuse>=3.0.0` as a regular dependency with graceful import handling.

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "langfuse>=3.0.0",
]
```

**Note**: The integration uses graceful import handling (matching the existing LangChain pattern). If Langfuse is not installed, the observability module will be inactive.

### ✅ Step 1.2 – Update `.env.template`

**File**: `.env.template` (at project root)  
**Action**: Append Langfuse environment variables with placeholder values.

```env
# Langfuse Observability (optional)
# LANGFUSE_PUBLIC_KEY=your_public_key_here
# LANGFUSE_SECRET_KEY=your_secret_key_here
# LANGFUSE_BASE_URL=https://cloud.langfuse.com  # optional, default cloud
```

**Note**: These lines are commented out by default; users uncomment and fill them to enable.

### ✅ Step 1.3 – Extend Configuration Package

**Files**:

- `src/multi_agent_dashboard/config/core.py`
- `src/multi_agent_dashboard/config/__init__.py`

**Action**: Add Langfuse constants that read from environment variables.

**`core.py`** (add near other environment‑variable constants):

```python
# Langfuse observability (optional)
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)
```

**`__init__.py`** (re‑export the new constants):

```python
from .core import (
    # ... existing imports ...
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_BASE_URL,
    LANGFUSE_ENABLED,
)
```

**Note**: No changes to YAML config files; Langfuse configuration is environment‑variable driven.

### ✅ Step 1.4 – Create Langfuse Client Module

**File**: `src/multi_agent_dashboard/observability/langfuse.py`  
**Purpose**: Encapsulate Langfuse initialization, callback creation, and error handling.

```python
"""
Langfuse observability integration (optional).

This module is only active when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.
"""

import atexit
import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Singleton Langfuse client
_langfuse_client = None
_langfuse_handler_class = None

def _try_import_langfuse():
    """Conditionally import Langfuse components."""
    global _langfuse_client, _langfuse_handler_class
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler  # v3.x.x import path
        _langfuse_client = get_client()
        # Register automatic flush on program exit
        atexit.register(flush_langfuse)
        _langfuse_handler_class = CallbackHandler
        return True
    except ImportError as e:
        logger.debug("Langfuse not available: %s", e)
        return False

def is_langfuse_enabled() -> bool:
    """Return True if Langfuse environment variables are set and SDK is importable."""
    from multi_agent_dashboard.config import LANGFUSE_ENABLED
    if not LANGFUSE_ENABLED:
        return False
    return _try_import_langfuse()

def get_langfuse_handler(
    session_id: Optional[str] = None,
    user_id: str = "multi_agent_dashboard",
    tags: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Create a Langfuse CallbackHandler for a specific invocation.

    Args:
        session_id: Langfuse session ID (maps to run_id).
        user_id: Static user identifier (default: "multi_agent_dashboard").
        tags: List of tags (e.g., ["agent:my_agent", "pipeline:my_pipeline"]).
        metadata: Additional custom metadata.

    Returns:
        A CallbackHandler instance, or None if Langfuse is disabled.
    """
    if not is_langfuse_enabled():
        return None

    if _langfuse_handler_class is None:
        logger.warning("Langfuse handler class not available")
        return None

    try:
        # Langfuse v3.x.x expects metadata via config, not handler constructor
        # We'll attach metadata later in the invocation config
        handler = _langfuse_handler_class()
        # Note: session_id, user_id, tags are passed via config["metadata"]
        return handler
    except Exception as e:
        logger.error("Failed to create Langfuse handler: %s", e)
        return None

def flush_langfuse():
    """Ensure all pending traces are sent to Langfuse (called automatically on exit)."""
    if is_langfuse_enabled() and _langfuse_client is not None:
        try:
            _langfuse_client.flush()
        except Exception as e:
            logger.warning("Failed to flush Langfuse client: %s", e)
```

**Key points**:

- Graceful import handling (mirrors `_LANGCHAIN_AVAILABLE` pattern).
- Singleton client via `get_client()`.
- Handler creation separated from metadata attachment (Langfuse v3 expects metadata in invocation config, not handler constructor).
- All errors are logged but never propagate to break agent execution.

### ✅ Step 1.5 – Update `__init__.py` for Observability Module

**File**: `src/multi_agent_dashboard/observability/__init__.py` (create if missing)  
**Content**:

```python
"""
Observability integrations (Langfuse, future tools).
"""

from .langfuse import (
    is_langfuse_enabled,
    get_langfuse_handler,
    flush_langfuse,
)

__all__ = [
    "is_langfuse_enabled",
    "get_langfuse_handler",
    "flush_langfuse",
]
```
**Note**: Create the `observability` directory if it doesn’t exist.

### ✅ Step 1.6 – Resolve remaining langfuse client issues

1. Guard `atexi` registration by `_atexit_registered` flag
2. Prevent repeated import attempts by `_import_attempted` flag
3. Cache result of `is_langfuse_enabled()` in `_langfuse_enabled_cache` so `_ensure_langfuse_initialized()` only called once
4. Set `_langfuse_handler_class` only after all imports and client creation succeed

**End of Phase 1**: At this point, the Langfuse SDK is available, configuration is loaded, and a client module exists, but no tracing is yet active. All existing tests should pass.

## ✅ Phase 2: Core Integration

**Objective**: Modify `LLMClient` to pass Langfuse callbacks during agent invocation. Add metadata extraction from context/agent spec.

### ✅ Step 2.1 – Update `LLMClient.__init__()`

**File**: `src/multi_agent_dashboard/llm_client/core.py`  
**Action**: Add private attributes `_langfuse_enabled` and `_langfuse_handler_factory` (or import the module functions directly).

**Location**: Inside `__init__` after other attribute initializations.

```python
# Near other imports at top of file
from multi_agent_dashboard.observability import is_langfuse_enabled, get_langfuse_handler

# Inside __init__ method:
self._langfuse_enabled = is_langfuse_enabled()
```

**Rationale**: Compute once per LLMClient instance; environment variables are static for the lifetime of the process.

### ✅ Step 2.2 – Modify `_execute_with_retry()`

**Location**: Same file, method `_execute_with_retry` (lines 424‑443).  
**Action**: Build invocation config with Langfuse callback and metadata when enabled.

**Current code snippet (for reference)**:

```python
def _execute_with_retry(self, agent, state, context=None):
    start_ts = time.perf_counter()

    try:
        if context is not None:
            result = agent.invoke(state, context=context)
        else:
            result = agent.invoke(state)
    # ... retry logic and error handling
```

**New version**:

```python
def _execute_with_retry(self, agent, state, context=None):
    start_ts = time.perf_counter()

    # Build invocation config with Langfuse callback if enabled
    invoke_config = {}
    if self._langfuse_enabled:
        # Extract metadata from context and agent spec
        session_id = None
        tags = []
        metadata = {}

        if context is not None:
            session_id = context.get("run_id")
            pipeline_name = context.get("pipeline_name")
            if pipeline_name:
                tags.append(f"pipeline:{pipeline_name}")

        # Agent name (from agent._name or spec)
        agent_name = getattr(agent, "_name", None)
        if agent_name:
            tags.append(f"agent:{agent_name}")
        else:
            # Fallback: try to get from agent spec if available
            agent_spec = getattr(agent, "_agent_spec", None)
            if agent_spec and hasattr(agent_spec, "name"):
                tags.append(f"agent:{agent_spec.name}")
            else:
                tags.append("agent:unknown")

        # Create Langfuse handler (no metadata in constructor for v3)
        handler = get_langfuse_handler()
        if handler is not None:
            invoke_config["callbacks"] = [handler]
            invoke_config["metadata"] = {
                "langfuse_user_id": "multi_agent_dashboard",
                "langfuse_session_id": session_id or "unknown",
                "langfuse_tags": tags,
                **metadata,
            }

    try:
        if context is not None:
            result = agent.invoke(state, context=context, config=invoke_config)
        else:
            result = agent.invoke(state, config=invoke_config)
    # ... rest of method unchanged
```

**Important notes**:

- The `metadata` dict uses Langfuse’s expected keys (`langfuse_user_id`, `langfuse_session_id`, `langfuse_tags`).
- The handler is created fresh for each invocation (lightweight, Langfuse manages internal state).
- If Langfuse is disabled, `invoke_config` remains an empty dict and the call proceeds exactly as before.
- All errors in handler creation are caught inside `get_langfuse_handler()` and logged; `handler` will be `None`, leaving `invoke_config` empty.

### ✅ Step 2.3 – Add Optional Flush for Short Scripts

**File**: `src/multi_agent_dashboard/llm_client/core.py` (optional)  
**Action**: Add a `flush_langfuse` method to `LLMClient` that delegates to the observability module.

```python
def flush_langfuse(self):
    """Flush any pending Langfuse traces (useful for short scripts)."""
    from multi_agent_dashboard.observability import flush_langfuse
    flush_langfuse()
```

This method can be called by scripts that run a few agent invocations and exit quickly, ensuring traces are sent before process termination.

**Note**: An `atexit` hook is automatically registered when Langfuse is enabled (see Step 1.4), ensuring traces are flushed on normal program exit. This manual method is still available for explicit flushing before exit.

### ✅ Step 2.4 – Allow optional LANGFUSE_ENABLED global override from .env

- Add LANGFUSE_ENABLED to .env.template (optional; existing API keys presence takes precedence)
- LANGFUSE_ENABLED=false in .env takes precedence and overrides LANGFUSE_ENABLED independently of existing API keys

### ✅ Step 2.5 – Verify No Breaking Changes

**Checklist**:

- The `context` parameter still works unchanged.
- The `agent.invoke` signature accepts `config` (LangChain standard).
- If `config` is empty, behavior is identical.
- Existing instrumentation middleware still attaches its events.

**Run existing tests** to confirm nothing broke.

## Phase 3: Validation & Documentation

**Objective**: Test the integration, update documentation, and provide an example.

### Step 3.0 - Critical Issues Resolved

1. Python 3.14 Incompatibility: Langfuse v3.14.5 has Pydantic v1 model issues with Python 3.14. Solution: Downgrade to Python 3.13.
2. `get_client()` Parameter Error: `get_client()` doesn't accept `secret_key` parameter; Solution: Set environment variables and call without parameters.

### ✅ Step 3.1 – Manual Validation

**Procedure**:

1. Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`.
2. Start the Streamlit dashboard (`streamlit run src/multi_agent_dashboard/ui/app.py`).
3. Run an existing agent pipeline.
4. Check the Langfuse dashboard for incoming traces.
5. Verify that the agent still works and existing metrics are collected.

**Expected outcome**:

- Traces appear in Langfuse with correct tags (agent name, pipeline name, run ID).
- No errors in the dashboard logs related to Langfuse.
- Existing functionality (token counts, tool usage, structured output) unchanged.

**Fix meta data population**:
- Update `get_langfuse_handler()` signature
- Update `_execute_with_retry()` logic**
- Update Field mapping logic

**Current mapping** → **Desired mapping**:
- `Name: Langfuse` → `Name: Agent Name` (via `name` parameter)
- `Session: unknown` → `Session: Pipeline Name` (or "Ad‑Hoc" for ad‑hoc pipelines)
- `User ID: nulti‑agent‑dashboard` → Keep as is (fix typo: "multi‑agent‑dashboard")
- `agent: unknown` → Keep as custom tag, populate with actual Agent Name
- `pipeline: unknown` → Keep as custom tag, populate with actual Pipeline Name or Ad-Hoc

### Step 3.2 – Update `AGENTS.md`

**File**: `AGENTS.md`  
**Action**: Add a new section “Observability with Langfuse” near the end of the document.

````markdown
## Observability with Langfuse

The dashboard supports optional integration with [Langfuse](https://langfuse.com) for advanced tracing, latency breakdown, and cost tracking.

### Setup
1. Install Langfuse SDK (already included as a dependency).
2. Add your Langfuse credentials to `.env`:
```

   LANGFUSE_PUBLIC_KEY=your_public_key
   LANGFUSE_SECRET_KEY=your_secret_key

# Optional: LANGFUSE_BASE_URL=https://self‑hosted.langfuse.com

```
3. Restart the dashboard.

### What Gets Traced
- Every agent invocation (including tool calls, reasoning steps, and model interactions).
- Metadata: agent name, pipeline name, run ID.
- Latency, token usage, provider costs (if available).

### Disabling Langfuse
Remove or comment out the `LANGFUSE_*` environment variables. The integration will be inactive with zero overhead.

### Manual Flush
For explicit control, call `LLMClient.flush_langfuse()` before exit. An automatic `atexit` flush is already registered when Langfuse is enabled.
````

### ❌ Step 3.3 – Create Example Script (Optional)

**File**: `examples/langfuse_observability.py`  
**Purpose**: Demonstrate standalone usage of Langfuse integration.

```python
"""
Example: Run an agent with Langfuse observability.
"""

import asyncio
import os
from dotenv import load_dotenv
from multi_agent_dashboard.llm_client import LLMClient
from multi_agent_dashboard.models import AgentSpec

load_dotenv()

async def main():
    # Create a simple agent spec
    spec = AgentSpec(
        name="langfuse_demo_agent",
        provider="openai",
        model="gpt-4o-mini",
        prompt="You are a helpful assistant. Answer the user's question.",
        tools=[],
        structured_output=False,
    )

    # Initialize LLM client (Langfuse will auto‑enable if keys are set)
    client = LLMClient()

    # Create agent
    agent = client.create_agent(spec)

    # Invoke with context (provides run_id and pipeline_name for tagging)
    state = {"input": "What is the capital of France?"}
    context = {
        "run_id": "demo_run_001",
        "pipeline_name": "demo_pipeline",
    }

    result = client.execute_agent(agent, state, context=context)
    print(f"Result: {result}")

    # Ensure traces are flushed before script ends
    client.flush_langfuse()

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 3.4 – Verify Rollback Safety

**Procedure**:

1. Remove `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` from `.env`.
2. Restart the dashboard and run the same agent.
3. Confirm that no Langfuse‑related errors appear and that the agent works normally.

**Expected**: The integration is completely inert; the code paths for building `invoke_config` are skipped.

## Decisions Made

- **Dependency**: `langfuse>=3.0.0` added as a regular dependency with graceful import handling (matches LangChain pattern).
- **Handler storage**: Create a new `CallbackHandler` per agent invocation (lightweight, ensures fresh metadata).
- **Flush strategy**: Automatic flush via `atexit` hook in `langfuse.py` client module; manual `flush_langfuse()` method retained for explicit control.
- **UI toggle**: Deferred to future iteration; activation remains environment‑variable driven.
- **Error handling**: Catch and log exceptions in handler creation; return `None` to prevent breaking agent execution.

## Risks & Mitigations

| Risk                              | Likelihood | Impact | Mitigation                                                                            |
| --------------------------------- | ---------- | ------ | ------------------------------------------------------------------------------------- |
| Langfuse dependency conflicts     | Low        | Medium | Graceful import handling; dependency version pinned to `>=3.0.0`.                     |
| Performance overhead              | Low        | Low    | Callbacks are async; no synchronous network calls during agent execution.             |
| Missing metadata (agent name)     | Medium     | Low    | Fallback to `"agent:unknown"`; still creates trace.                                   |
| Breaking existing instrumentation | Very Low   | High   | Integration is additive; existing middleware untouched.                               |
| Network failures to Langfuse      | Medium     | Low    | Langfuse client queues and retries; failures logged but don’t affect agent execution. |
| Langfuse v3.x.x API changes       | Low        | Medium | Pin to `langfuse>=3.0.0,<4.0.0`; monitor release notes.                               |

## Rollback Instructions

If the integration causes issues, revert to previous behavior by:

1. **Remove environment variables**: Delete or comment out `LANGFUSE_*` lines in `.env`.
2. **Optional code removal**: The changes are non‑breaking, but if you want to completely remove the code:
   - Delete `src/multi_agent_dashboard/observability/` directory.
   - Remove Langfuse constants from `config/core.py` and `config/__init__.py`.
   - Revert changes to `LLMClient._execute_with_retry` and `LLMClient.__init__`.
   - Remove `langfuse` dependency from `pyproject.toml`.

Because the integration is optional and guarded by environment variables, simply removing the keys effectively disables it with zero runtime impact.

## Next Steps

1. **User approval**: Review this strategy and confirm the approach.
2. **Implementation**: Execute the three phases in order, verifying each step with existing tests.
3. **Validation**: Manual test with Langfuse Cloud account to confirm traces appear.
4. **Documentation**: Update `AGENTS.md` and ensure the example script works.

## References

- [Langfuse LangChain Integration Guide](https://langfuse.com/integrations/frameworks/langchain)
- [Langfuse Python SDK Documentation](https://langfuse.com/docs/sdk/python)
- [Multi‑Agent Dashboard Architecture](./AGENTS.md)
- [Feasibility Report](./260227‑2100_integrate_langfuse_feasibility_report.md)

---

*This implementation strategy is based on the feasibility report dated 2026‑02‑27. All steps are designed to be incremental and reversible. Final implementation decisions require user approval.*