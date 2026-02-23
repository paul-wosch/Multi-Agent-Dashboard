# Models Analysis, Refactoring & Packaging Strategy

## **Analysis of `models.py` (539 lines)**

**Current Issues**:

1. **Circular import**: `models.py` imports `engine.utils` (line 17‑24) while `engine/engine_orchestrator.py` imports `AgentSpec`, `AgentRuntime` (line 10). This creates a hard circular dependency that currently causes import failures after removing the wrapper.

2. **Single‑class megalith**: `AgentRuntime.run` is 357 lines long and violates SRP—it handles:
   - File type detection & content decoding
   - Prompt variable injection
   - Tool configuration merging & provider‑specific conversion
   - LangChain agent creation & invocation
   - Token/metrics extraction from raw responses
   - Tool‑usage extraction from content blocks
   - Structured‑output detection (4 fallback paths)
   - State writeback with validation

3. **Deeply nested helper logic**: `_build_tools_config`, `_get_allowed_domains`, `_build_reasoning_config` are class‑local but have no dependency on instance state beyond `self.spec` and `state`.

4. **Shared utility entanglement**: The six helper functions imported from `engine.utils` are used only within `AgentRuntime.run` (metrics/instrumentation extraction). They are unrelated to the engine’s core orchestration.

**Refactoring Recommendations**:

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| **Create `runtime/` subpackage** | Move `AgentRuntime` to `runtime/agent_runtime.py`; extract helpers to focused modules (`runtime/metrics_extractor.py`, `runtime/tool_converter.py`, `runtime/file_processor.py`, `runtime/structured_output_detector.py`). | Clean separation of concerns; eliminates circular import (runtime depends on engine.utils, engine depends only on models.AgentSpec); each module ≤150 lines. | Adds 5‑6 new files; requires updating imports in UI/tests. |

**New runtime subpackage structure**:

```
src/multi_agent_dashboard/
├── models.py                # AgentSpec, PipelineSpec (pure dataclasses)
├── runtime/                 # New subpackage
│   ├── __init__.py          # Export AgentRuntime
│   ├── agent_runtime.py     # AgentRuntime class (calls delegates)
│   ├── file_processor.py    # text/binary file separation
│   ├── tool_converter.py    # _build_tools_config, _get_allowed_domains, provider tool merging
│   ├── metrics_extractor.py # token extraction, instrumentation, tool‑usage collection
│   └── structured_output_detector.py # 4‑path detection & state writeback
└── shared/                  # New package for engine/models common utilities
    ├── __init__.py
    └── instrumentation.py   # _extract_instrumentation_events, _collect_content_blocks, etc.
```

**Key Benefits**:
- **Circular import eliminated**: `engine.utils` can move shared helpers to `shared.instrumentation`; `runtime` imports from `shared`, engine imports from `shared`.
- **SRP compliance**: Each module has a single, testable responsibility.
- **Reusability**: `metrics_extractor` can be used by engine and standalone runtime.
- **Maintainability**: Each file ≤150 lines, clear boundaries.

**Migration steps**
1. Create `shared/instrumentation.py` with the six helper functions from `engine.utils`.
2. Update `engine/utils.py` and `models.py` to import from `shared.instrumentation`.
3. Create `runtime/` subpackage with extracted modules.
4. Move `AgentRuntime` class to `runtime/agent_runtime.py`, delegating to new modules.
5. Update imports in UI (`bootstrap.py`, `run_mode.py`, etc.) and tests.

**Risk**: Medium (structural changes), but all existing interfaces remain identical; only internal delegation changes.

---

## **Refactoring Strategy (Runtime Subpackage + Shared Utilities)**

COMPLETED: 1, 2, 3, 4, 5, 6, 7, 8, 9 (23 Feb 2026)

**Objective**: Break circular imports, modularize `AgentRuntime.run`, and create a clean separation of concerns without breaking existing functionality.

### ✅ **Phase 1: Create Shared Instrumentation Package**
**Goal**: Move helper functions used by both `engine/` and `models.py` to a neutral location.

1. **Create `shared/` package**:
   - Create directory `src/multi_agent_dashboard/shared/`
   - Create `src/multi_agent_dashboard/shared/__init__.py` (empty)

2. **Create `shared/instrumentation.py`**:
   - Move six helper functions from `engine/utils.py`:
     - `_extract_instrumentation_events`
     - `_value_to_dict`
     - `_collect_content_blocks`
     - `_structured_from_instrumentation`
     - `_collect_tool_calls`
     - `_tool_usage_entry_from_payload`
   - Keep function signatures and logic identical.

3. **Update `engine/utils.py`**:
   - Replace moved functions with imports from `shared.instrumentation`
   - Keep remaining functions (`_normalize_content_blocks`, `_extract_provider_features_from_profile`)

4. **Update `models.py`**:
   - Change import from `.engine.utils` to `shared.instrumentation`
   - Remove unused `_value_to_dict` import

5. **Update engine modules** (`agent_executor.py`, `snapshot_builder.py`):
   - Change imports from `.utils` to `shared.instrumentation` for the six functions

**Verification**: Run a quick import test to ensure no circular import errors.

### ✅ **Phase 2: Create Runtime Subpackage Skeleton**
**Goal**: Establish the `runtime/` package structure without moving logic yet.

1. **Create `runtime/` directory**:
   - Create `src/multi_agent_dashboard/runtime/`
   - Create `src/multi_agent_dashboard/runtime/__init__.py` with placeholder `__all__ = ["AgentRuntime"]`

2. **Create empty module files**:
   - `agent_runtime.py` (placeholder class)
   - `file_processor.py`
   - `tool_converter.py`
   - `metrics_extractor.py`
   - `structured_output_detector.py`

**Verification**: Ensure the package is importable.

### ✅ **Phase 3: Extract File Processing Logic**
**Goal**: Move file-type detection and content decoding to a dedicated module.

1. **Implement `file_processor.py`**:
   - Function `process_files(all_files: List[Dict]) -> Tuple[List[Dict], List[Dict]]`
   - Extract logic from `AgentRuntime.run` lines 95-114
   - Returns `(text_files, binary_files)`

2. **Update `AgentRuntime.run`**:
   - Replace file processing block with call to `process_files()`
   - Keep the rest of the method unchanged

**Verification**: Manual agent run to confirm file handling works.

### ✅ **Phase 4: Extract Tool Configuration Logic**
**Goal**: Move tool-building and domain-filter logic to `tool_converter.py`.

1. **Implement `tool_converter.py`**:
   - `get_allowed_domains(spec, state)` (from `_get_allowed_domains`)
   - `build_tools_config(spec, state)` (from `_build_tools_config`)
   - `build_reasoning_config(spec)` (from `_build_reasoning_config`)
   - `prepare_tools_for_agent(spec, state, provider_id, model, use_responses_api, provider_features)` – combines tool conversion and allowed-domain merging (lines 162-230)

2. **Update `AgentRuntime.run`**:
   - Replace `_get_allowed_domains`, `_build_tools_config`, `_build_reasoning_config` calls
   - Replace tool conversion block with `prepare_tools_for_agent()`

**Verification**: Manual agent run with tools enabled.

### ✅ **Phase 5: Extract Metrics Extraction Logic**
**Goal**: Move token extraction, tool-usage collection, and provider-profile detection.

1. **Implement `metrics_extractor.py`**:
   - `extract_tokens_from_raw(raw, response)` (lines 291-307)
   - `collect_tool_usage(raw, content_blocks)` (lines 344-370, uses `shared.instrumentation`)
   - `extract_detected_provider_profile(agent_obj_for_invoke, raw)` (lines 328-339)

2. **Update `AgentRuntime.run`**:
   - Replace token extraction block
   - Replace tool-usage collection block
   - Replace provider-profile detection block

**Verification**: Manual agent run; check metrics in UI.

### ✅ **Phase 6: Extract Structured Output Detection**
**Goal**: Move the four-path structured output detection and state writeback.

1. **Implement `structured_output_detector.py`**:
   - `detect_structured_output(raw, content_blocks, raw_output)` (lines 377-410)
   - `writeback_to_state(spec, state, parsed, raw_output)` (lines 412-441)

2. **Update `AgentRuntime.run`**:
   - Replace structured output detection block
   - Replace state writeback block

**Verification**: Manual agent run with structured output enabled.

### ✅ **Phase 7: Move AgentRuntime Class**
**Goal**: Relocate `AgentRuntime` to the new package and update imports.

1. **Move class to `runtime/agent_runtime.py`**:
   - Copy entire `AgentRuntime` class from `models.py`
   - Update imports to use new runtime modules and `shared.instrumentation`
   - Keep the class API identical

2. **Update `models.py`**:
   - Remove `AgentRuntime` class
   - Remove `shared.instrumentation` import (no longer needed)
   - Keep `AgentSpec` and `PipelineSpec`

3. **Update `runtime/__init__.py`**:
   - Export `AgentRuntime` from `agent_runtime`

**Verification**: Import test for `from multi_agent_dashboard.runtime import AgentRuntime`.

### ✅ **Phase 8: Update External Imports**
**Goal**: Update all references to `AgentRuntime` throughout the codebase.

1. **Find all imports**:
   - `engine/engine_orchestrator.py`
   - UI modules (`bootstrap.py`, `run_mode.py`, `exports.py`, `graph_view.py`)
   - Test files (`test_schema_validation_flow.py`)
   - Any other references

2. **Change imports**:
   - From `multi_agent_dashboard.models import AgentRuntime`
   - To `multi_agent_dashboard.runtime import AgentRuntime`

**Verification**: Full application import test.

### ✅ **Phase 9: Final Cleanup**
**Goal**: Remove any leftover dependencies and verify circular import is resolved.

1. **Check `models.py`**:
   - Ensure it only contains dataclasses
   - No engine or runtime imports

2. **Check `engine/utils.py`**:
   - Remove any unused imports
   - Verify it only contains engine-specific utilities

3. **Run manual verification**:
   - Start the Streamlit UI
   - Run a simple agent pipeline
   - Test with tools, structured output, files

### **Risk Mitigation**:
- Each phase is atomic and verifiable
- No changes to public APIs (classes remain identical)
- Import paths updated gradually
- Backward compatibility maintained throughout