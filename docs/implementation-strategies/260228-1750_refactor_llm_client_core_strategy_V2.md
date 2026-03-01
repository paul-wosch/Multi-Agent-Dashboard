# Refactoring LLMClient Core – Implementation Strategy

**Date**: 2026‑02‑28 (created); 2026‑03‑01 (completed); 
**Goal**: Decompose `src/multi_agent_dashboard/llm_client/core.py` (1002 lines) into cohesive modules while preserving the public API (`LLMClient`, `TextResponse`, `LLMError`, `INSTRUMENTATION_MIDDLEWARE`, `create_agent_for_spec`, `invoke_agent`).  

**Guiding Principles**:

1. **Keep the public API unchanged** – no external behavior differences.
2. **Incremental, low‑risk phases** – each phase is self‑contained, verifiable, and reversible.
3. **DRY/KISS** – avoid unnecessary complexity; follow existing patterns.
4. **Separation of concerns** – each new module has a single, clear responsibility.
5. **Test as you go** – after each phase, run existing tests and manually verify agent execution.

## ✅ Phase 1: Extract Low‑Risk Support Modules

**Objective**: Move conditional imports and well‑encapsulated classes out of `core.py` with minimal impact.

### ✅ Step 1.1 – Create `availability.py`

**Actions**:

1. Create `src/multi_agent_dashboard/llm_client/availability.py`.
2. Move the following from `core.py`:
   - `_LANGCHAIN_AVAILABLE`, `_LANGFUSE_AVAILABLE`, `_DUCKDUCKGO_TOOL_AVAILABLE` flags.
   - Lazy‑loaded module references (`_SystemMessage`, `_HumanMessage`, `_AIMessage`, `_init_chat_model`, `_create_agent`, `_AgentMiddleware`).
   - LangChain, Langfuse, and DuckDuckGoSearchTool conditional import blocks.
3. Export a clean public interface:
   ```python
   # availability.py
   LANGCHAIN_AVAILABLE: bool
   LANGFUSE_AVAILABLE: bool
   DUCKDUCKGO_TOOL_AVAILABLE: bool
   
   # Lazy references (callable to defer import errors)
   def get_SystemMessage(): ...
   def get_HumanMessage(): ...
   def get_AIMessage(): ...
   def get_init_chat_model(): ...
   def get_create_agent(): ...
   def get_AgentMiddleware(): ...
   ```
4. Update `core.py` to import from `availability` instead of defining its own flags.

**Verification**:
- Run `pytest tests/` – no new failures.
- Start the Streamlit UI, create and run a simple agent (no Langfuse).

**Rollback**: Revert changes to `core.py` and delete `availability.py`.

### ✅ Step 1.2 – Create `agent_creation.py`

**Actions**:

1. Create `src/multi_agent_dashboard/llm_client/agent_creation.py`.
2. Move the entire `AgentCreationFacade` class (lines 87‑252) to the new file.
3. Keep the same `__init__` signature (`def __init__(self, client):`).
4. In `core.py`, import `AgentCreationFacade` from `.agent_creation`.
5. Update `LLMClient.create_agent_for_spec` to instantiate the imported facade (line 313).

**Verification**:
- Run `pytest tests/` – ensure agent‑creation tests still pass.
- Create an agent via UI and verify it works.

### ✅ Step 1.3 – Create `observability/langfuse_integration.py`

**Actions**:

1. Create `src/multi_agent_dashboard/llm_client/observability/langfuse_integration.py`.
2. Extract the Langfuse‑config building logic from `_execute_with_retry` (lines 451‑504) into a standalone function:
   ```python
   def build_langfuse_config(agent, context=None) -> Dict[str, Any]:
       # Returns invoke_config dict (empty if Langfuse disabled)
   ```
3. Move Langfuse‑specific imports (`is_langfuse_enabled`, `get_langfuse_handler`) into this module.
4. Update `_execute_with_retry` to call `build_langfuse_config` and merge the result.

**Verification**:
- Run tests that involve Langfuse (if any).
- Start UI with `LANGFUSE_ENABLED=True` (or default) and verify observability still works.

**Rollback**: Revert changes to `core.py` and delete the new module.

---

## ✅ Phase 2: Extract Response Processing

**Objective**: Isolate the 430‑line `_process_response` method and its nested helpers into a dedicated `ResponseProcessor`.

### ✅ Step 2.1 – Create `response_processor.py`

**Actions**:

1. Create `src/multi_agent_dashboard/llm_client/response_processor.py`.
2. Define `ResponseProcessor` class with static methods:
   - `extract_usage_from_candidate`
   - `extract_usage_from_messages`
   - `extract_tool_info_from_messages`
   - `extract_text_from_messages`
   - `process` (main entry point)
3. Copy the exact logic from the nested helpers in `_process_response` (lines 557‑640, etc.) into the appropriate static methods.
4. Keep the same signatures and return types.

### ✅ Step 2.2 – Update `LLMClient._process_response`

**Actions**:

1. In `core.py`, import `ResponseProcessor`.
2. Replace the body of `_process_response` with a delegation:
   ```python
   def _process_response(self, result, latency, agent):
       return ResponseProcessor.process(result, latency, agent)
   ```
3. **Important**: Ensure `ResponseProcessor.process` receives all necessary context (e.g., `agent` for any agent‑specific logic). `ResponseProcessor` will import `ResponseNormalizer` directly and call `to_dict` (per decision 1), removing the dependency on `LLMClient`.

### ✅ Step 2.3 – Verify Response Processing

**Verification**:
- Run all existing tests, especially those that validate token extraction, instrumentation events, and content‑blocks merging.
- Manually run an agent with files, structured output, and tool calls; verify `TextResponse` fields are identical to before.

**Rollback**: Revert changes to `core.py` and delete `response_processor.py`.

---

## Phase 3: Extract Request Building and Execution

**Objective**: Separate request construction (`_prepare_request`) and agent invocation (`_execute_with_retry`) into `RequestBuilder` and `ExecutionEngine`.

### ✅ Step 3.1 – Create `request_builder.py`

**Actions**:

1. Create `src/multi_agent_dashboard/llm_client/request_builder.py`.
2. Define `RequestBuilder` class:
   ```python
   class RequestBuilder:
       def __init__(self, langchain_available: bool, SystemMessage, HumanMessage):
           ...
       def build(self, agent, prompt: str, files=None, context=None) -> Dict[str, Any]:
           # Returns the `state` dict (messages array)
   ```
3. Move the logic from `_prepare_request` (lines 415‑442) into `RequestBuilder.build`.
4. Handle the fallback concatenation path for missing multimodal module inside `RequestBuilder`. (This logic will later be moved into `prepare_multimodal_content` in the `multimodal/handler.py` module per decision 3.)
5. Update `core.py` to instantiate a `RequestBuilder` in `LLMClient.__init__` (or create on‑the‑fly) and call `builder.build` from `_prepare_request`.

### ✅ Step 3.2 – Create `execution_engine.py`

**Actions**:

1. Create `src/multi_agent_dashboard/llm_client/execution_engine.py`.
2. Define `ExecutionEngine` class:
   ```python
   class ExecutionEngine:
       def __init__(self, langfuse_enabled: bool, max_retries: int, backoff_base: float, on_rate_limit: Callable):
           ...
       def execute(self, agent, state: Dict[str, Any], context=None) -> Tuple[Any, float]:
           # Returns (result, latency)
   ```
3. Move `_execute_with_retry` logic (lines 444‑525) into `ExecutionEngine.execute`.
4. Integrate the `build_langfuse_config` function from Phase 1.
5. Update `core.py` to instantiate `ExecutionEngine` in `LLMClient.__init__` and delegate `_execute_with_retry` to `engine.execute`.

### ✅ Step 3.3 – Update `LLMClient` Integration

**Actions**:

1. Add `self._request_builder` and `self._execution_engine` attributes in `LLMClient.__init__`.
2. Replace `_prepare_request` and `_execute_with_retry` with stubs that delegate to the respective components.
3. Ensure `invoke_agent` (line 960) still works unchanged.

### ✅ Step 3.4 – Verify Request/Execution Flow

**Verification**:
- Run all agent‑invocation tests.
- Manually test agents with files, different providers, and Langfuse enabled.
- Verify that retry logic works (simulate a rate‑limit error with a mock agent).

**Rollback**: Revert changes to `core.py` and delete `request_builder.py`, `execution_engine.py`.

---

## ✅ Phase 4: Clean Up Core

**Objective**: Remove the now‑empty method stubs from `core.py` and consolidate imports.

### ✅ Step 4.1 – Remove Delegating Stubs

**Actions**:

1. Delete `_prepare_request`, `_execute_with_retry`, `_process_response` from `core.py` (or keep them as one‑line delegators for safety).
2. Update `invoke_agent` to call the components directly:
   ```python
   state = self._request_builder.build(agent, prompt, files=files, context=context)
   result, latency = self._execution_engine.execute(agent, state, context=context)
   return self._response_processor.process(result, latency, agent)
   ```
3. Remove any unused imports.

### ✅ Step 4.2 – Update Imports Across Codebase

**Actions**:

1. Search for imports from `multi_agent_dashboard.llm_client.core` that might rely on internal classes (e.g., `AgentCreationFacade`). Update them to import from the new modules.
2. Verify no external code depends on private methods (unlikely, but check).

**Verification**:
- Run full test suite (`pytest`).
- Start UI, create a pipeline, run agents with various configurations (structured output, tools, files).

**Rollback**: Revert deletions and restore stubs.

### ✅ Step 4.3 – Refactor to core subpackage
- Create `llm_client/core/` package with `__init__.py` re-exporting the public API (`LLMClient`, `TextResponse`, `LLMError`, `INSTRUMENTATION_MIDDLEWARE`, `create_agent_for_spec`, `invoke_agent`), keeping the public surface stable.
- Move current modules into the subpackage for cohesion:
  - `core/availability.py`, `core/agent_creation.py`, `core/request_builder.py`, `core/execution_engine.py`, `core/response_processor.py`, `core/langfuse_integration.py` (observability).
  - Leave provider-agnostic utilities (e.g., `wrappers.py`, `structured_output.py`, `instrumentation.py`) at top-level unless they are only used by `LLMClient`; if so, relocate to `core/`.
- Replace `llm_client/core.py` with `core/__init__.py` that constructs `LLMClient` from submodules; remove the old monolith file after updating imports.
- Resolve circulars by dependency direction:
  - `LLMClient` → `core.*` helpers only; helpers should not import `LLMClient`.
  - Observability and multimodal helpers should depend downward (no back-imports).
  - Keep provider adapters/tool binder/structured output as separate leaves to avoid cycles.
- Migration steps:
  1) Introduce `llm_client/core/` and move modules (update relative imports).
  2) Add re-exports in `llm_client/core/__init__.py` and adjust `llm_client/__init__.py`.
  3) Update internal imports to point to the new paths; run import check.
  4) Delete old `core.py` once everything resolves.

---

## ✅ Phase 5: Final Polish and Documentation

**Objective**: Ensure the new modules are well‑documented and follow project conventions.

### ✅ Step 5.1 – Add Docstrings and Type Hints

**Actions**:

1. Add/update module docstrings and docstrings for each new class and method.
2. Ensure type hints are consistent with the existing codebase (use `from typing import ...`).

### ✅ Step 5.2 – Update `AGENTS.md`

**Actions**:

1. Update project file tree to reflect new package structure
2. Add a brief note about the new module structure under the “LLM Provider Integration” section.
3. Mention that `LLMClient` internals have been modularized for maintainability.

### ✅ Step 5.3 – Verify No Regressions

**Actions**:

1. Run a full integration test: create a pipeline with multiple agents, tools, structured output, and file attachments.
2. Check that logging, cost tracking, and observability (Langfuse) still work.

---

## Decisions Made

1. **`_to_dict` delegation**: `ResponseProcessor` will import `ResponseNormalizer` and call `to_dict` directly, removing the dependency on `LLMClient`.
2. **Schema‑resolution duplication**: `_build_structured_output_adapter` will be refactored to delegate to `StructuredOutputBinder.extract_schema` (separate cleanup after core extraction).
3. **Multimodal fallback location**: The fallback concatenation logic will be moved into `prepare_multimodal_content` in the `multimodal/handler.py` module.
4. **Observability abstraction**: No abstraction for now; keep the simple `build_langfuse_config` function. An `ObservabilityBackend` interface can be added later if needed.

All inline decision points in the implementation steps have been resolved per the recommendations above (e.g., `AgentCreationFacade` remains internal, `ResponseProcessor` is static, `RequestBuilder`/`ExecutionEngine` are internal and not configurable, etc.).

---

## Expected Outcomes

- `core.py` reduced from ~1002 lines to ~300‑400 lines (mainly `LLMClient` boilerplate and public methods).
- Six new focused modules:
  1. `availability.py`
  2. `agent_creation.py`
  3. `observability/langfuse_integration.py`
  4. `response_processor.py`
  5. `request_builder.py`
  6. `execution_engine.py`
- Clear separation of concerns, easier unit testing, and simpler future extensions.
- **Zero changes** to the public API – existing UI and engine code continues to work unchanged.
