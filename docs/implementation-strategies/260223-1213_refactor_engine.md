# Unified Engine Refactoring & Packaging Strategy

## Goal
Refactor the monolithic `engine.py` (470+ lines) into a modular `engine/` package that:
- Respects the existing modular architecture (`llm_client/`, `tool_integration/`, `db/`, `ui/`)
- Eliminates duplicated helper functions
- Encapsulates distinct responsibilities in single‑purpose modules
- Preserves **100% backward compatibility** for all existing imports and API calls
- Enables future execution modes (parallel, conditional) without modifying the core orchestration loop

## Proposed Architecture
```
MultiAgentEngine (facade/orchestrator)
├── AgentExecutor          # per‑agent pipeline (input validation → invocation → metrics extraction)
├── PipelineState          # state, memory, warnings, tool_usages, agent_configs
├── RunSnapshotBuilder     # constructs agent_configs from AgentSpec + runtime metrics
├── SchemaValidator        # schema resolution, validation, strict‑exit logic
├── MetricsAggregator      # token/cost computation, aggregation across agents
├── ProgressReporter       # progress‑callback wrapper with tick calculations
└── InstrumentationUtils   # shared helpers (content‑blocks, instrumentation events)
```

## Package Structure
```
src/multi_agent_dashboard/engine/
├── __init__.py            # re‑export MultiAgentEngine, EngineResult
├── engine_orchestrator.py # MultiAgentEngine class (main orchestrator)
├── agent_executor.py      # AgentExecutor
├── state_manager.py       # PipelineState
├── snapshot_builder.py    # RunSnapshotBuilder
├── schema_validator.py    # SchemaValidator
├── metrics_aggregator.py  # MetricsAggregator
├── progress_reporter.py   # ProgressReporter
├── utils.py               # shared helpers (instrumentation, content blocks)
└── types.py               # data containers (AgentRunResult, RunMetrics, etc.)
```

**Note**: The top‑level `engine.py` will remain as a backward‑compatibility wrapper that imports from `engine.engine_orchestrator`.

## Step‑by‑Step Migration Plan (Incremental, Low‑Risk)

COMPLETED: 1, 2, 3, 4, 5 (23 Feb 2026)

### ✅ Phase 1 – Create package & consolidate helpers
1. Create `src/multi_agent_dashboard/engine/` directory with `__init__.py`.
2. Move all duplicate helper functions (6 functions: `_extract_instrumentation_events`, `_value_to_dict`, `_collect_content_blocks`, `_structured_from_instrumentation`, `_collect_tool_calls`, `_tool_usage_entry_from_payload`) from `engine.py` and `models.py` into `engine/utils.py`. Add `_normalize_content_blocks` as a new utility.
3. Update both `engine.py` and `models.py` to import from `engine.utils`.
4. Verify no regression via existing tests (`pytest tests/`).

### ✅ Phase 2 – Data containers & AgentExecutor
1. Create `engine/types.py` with dataclasses:
   - `AgentRunResult` (raw_output, metrics, parsed, tool_usages, config_snapshot, cost_breakdown)
   - `PipelineState` (state, memory, warnings, tool_usages, agent_configs, strict_schema flags)
   - `RunMetrics` (input_tokens, output_tokens, latency, input_cost, output_cost, total_cost)
2. Build `AgentExecutor` in `engine/agent_executor.py`; migrate per‑agent logic from `run_seq()` (the main loop from agent execution through output parsing and writeback).
3. Update `run_seq()` to call `AgentExecutor.execute_agent()`.

### ✅ Phase 3 – Extract remaining components
1. Create `RunSnapshotBuilder` (`engine/snapshot_builder.py`) (agent configuration snapshot logic).
2. Create `SchemaValidator` (`engine/schema_validator.py`) (structured output validation and strict‑exit logic).
3. Create `MetricsAggregator` (`engine/metrics_aggregator.py`) wrapping `_compute_cost` and aggregation logic.
4. Create `ProgressReporter` (`engine/progress_reporter.py`) for tick‑based progress updates.

### ✅ Phase 4 – State management & orchestration
1. Implement `PipelineState` (`engine/state_manager.py`) to encapsulate `self.state`, `self.memory`, `self._warnings`, `self.agent_metrics`.
2. Move `MultiAgentEngine` into `engine/engine_orchestrator.py`, delegating to the new components.
3. Keep top‑level `engine.py` as a backward‑compatibility wrapper that imports `MultiAgentEngine` from `engine.engine_orchestrator` and re‑exports it. (Already implemented in Phase 1.)

### ✅ Phase 5 – Final cleanup & validation
1. Run full test suite (`pytest`), verify zero behavioral changes.
2. Optionally update internal imports in the codebase to use `engine.*` directly (not required for compatibility).
3. Document the new package structure in `AGENTS.md`.

## Backward Compatibility Guarantees
- **Import path unchanged**: `from multi_agent_dashboard.engine import MultiAgentEngine, EngineResult` continues to work.
- **API unchanged**: `add_agent()`, `remove_agent()`, `run_seq()` signatures identical.
- **Result fields identical**: `EngineResult` dataclass fields remain exactly the same.
- **No breaking changes**: All existing UI, tests, and scripts continue to run without modification.

## Benefits
- **Architectural consistency**: Follows established patterns (`llm_client/`, `tool_integration/`, `db/`).
- **Maintainability**: Each module ≤150 lines, single responsibility.
- **Testability**: Isolated components enable targeted unit tests.
- **Reusability**: `AgentExecutor` can be used outside sequential pipelines.
- **DRY**: Eliminated duplicate helpers; clear dependency graph.

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Breaking existing imports | Keep `engine.py` as compatibility wrapper; never rename/remove public classes. |
| Circular imports | Design dependency graph top‑down; use `from . import …` within package. |
| Behavioral regression | Incremental migration; run existing tests after each phase. |
| Performance overhead | Additional object creation negligible compared to LLM calls. |
