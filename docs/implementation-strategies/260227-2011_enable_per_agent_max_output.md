# Implementation Strategy: Enable Per-Agent Max Output Tokens Configuration

**Date:** 2026-02-27 (created); 2026-02-27 (completed)
**Feature Request:** FR-260227-1018_enable_per_agent_max_output.md  
**Author:** AI Assistant (Crush)  
**Version:** 2 (updated per feedback)

## Overview

This document outlines a progressive, incremental strategy for implementing per‑agent maximum output token configuration. The goal is to allow each agent to have its own `max_output` token limit, which is persisted in the database, exposed in the UI Agent Editor, and wired into the agent execution flow. The implementation respects the existing precedence rules and adds a strict‑override option via `.env`.

## Core Requirements

1. **Database schema:** Add `max_output` and `max_output_effective` columns to `agents` and `agent_run_configs` tables (INTEGER DEFAULT 0).
2. **Agent model:** Extend `AgentSpec` with `max_output: int = 0` field.
3. **UI changes:** Add input for `max_output` in the Agent Editor’s advanced tab; display configured value (and temperature) in agent cards and history views.
4. **Execution flow:** Compute effective token cap using the precedence rule “.env overrides YAML; smallest non‑zero wins (except 0)”, with an optional strict‑override flag that always gives priority to the `.env` cap.
5. **Back‑compatibility:** Existing agents automatically receive `max_output = 0` (no limit) after migration (NULL treated as 0).
6. **Observability:** Store configured and effective cap used for each run in `agent_run_configs.max_output` / `agent_run_configs.max_output_effective`.

## Precedence & Override Logic

Given three sources of token limits:

- **Global YAML cap** (`AGENT_OUTPUT_TOKEN_CAP` from `agents.yaml`).
- **Global .env cap** (`AGENT_OUTPUT_TOKEN_CAP` from `.env`) – overrides YAML **regardless of amount**.
- **Per‑agent cap** (`max_output`) – stored in the agent’s DB record.
- **Strict‑override flag** (`STRICT_OUTPUT_TOKEN_CAP_OVERRIDE`) – boolean in `.env`.

The effective cap is determined as follows:

```python
def effective_max_tokens(env_cap: int, agent_cap: int, strict_override: bool) -> int | None:
    # 0 means “no limit” → treat as infinite
    if strict_override:
        return env_cap if env_cap > 0 else None
    # .env already overrides YAML (env_cap is the .env value or YAML default)
    # smallest non‑zero wins; if one is zero, take the other; if both zero → no limit
    candidates = [c for c in (env_cap, agent_cap) if c > 0]
    if not candidates:
        return None  # no limit
    return min(candidates)
```

**Important:** The `.env` value **always** replaces the YAML default, irrespective of which is larger. The precedence between `.env` and per‑agent config is “smallest non‑zero wins”, except when `STRICT_OUTPUT_TOKEN_CAP_OVERRIDE=true`, which forces the `.env` cap to be used (or no limit if `.env` cap is 0).

The same logic applies to **character caps** (`AGENT_INPUT_CHAR_CAP`, `AGENT_OUTPUT_CHAR_CAP`), but this FR only concerns the token cap. Character‑cap precedence remains unchanged (they have no per‑agent configuration).

## Step‑by‑Step Implementation Plan

### ✅ Phase 1: Database & Model Changes

#### **✅ Step 1.1 – Update canonical schema** (`src/multi_agent_dashboard/db/infra/schema.py`)

- Add `max_output` column to the `agents` table: `"max_output": "INTEGER DEFAULT 0"`
- Add `max_output` column to the `agent_run_configs` table: `"max_output": "INTEGER DEFAULT 0"`
- Add `max_output_effective` column to the `agent_run_configs` table: `"max_output_effective": "INTEGER DEFAULT 0"`

#### **✅ Step 1.2 – Generate migration SQL**

- Run `python -m multi_agent_dashboard.db.infra.generate_migration add_agent_max_output_columns`
- Verify the generated SQL file in `data/migrations/` includes all three column additions.
- No back‑population needed; existing rows will receive the default value 0.

#### **✅ Step 1.3 – Update AgentDAO** (`src/multi_agent_dashboard/db/agents.py`)

- Extend `SELECT` queries to include `max_output`
- Update the `save()` method signature and INSERT statement to store `max_output`
- Ensure `list()` and `get()` return the new field in the agent dict.
- No changes needed for `max_output_effective` (it is written only during run recording).

#### **✅ Step 1.4 – Extend AgentSpec** (`src/multi_agent_dashboard/models.py`)

- Add `max_output: int = 0` field to the `AgentSpec` dataclass (not optional).
- Update any `AgentSpec` constructor calls that may now need the new parameter (search for `AgentSpec(`).

### ✅ Phase 2: Configuration & Environment Variables

#### **✅ Step 2.1 – Add strict‑override flag to config** (`src/multi_agent_dashboard/config/core.py`)

- Read `STRICT_OUTPUT_TOKEN_CAP_OVERRIDE` from `_env` (default `"false"`).
- Convert to boolean: `STRICT_OUTPUT_TOKEN_CAP_OVERRIDE = _env.get("STRICT_OUTPUT_TOKEN_CAP_OVERRIDE", "false").lower() == "true"`
- Export the constant in `src/multi_agent_dashboard/config/__init__.py`.

#### **✅ Step 2.2 – Ensure global token cap is already overridable via `.env`**

- This was completed in the previous session (AGENT_OUTPUT_TOKEN_CAP already reads from `.env` with precedence over YAML).

### ✅ Phase 3: UI Updates

#### **✅ Step 3.1 – Agent Editor** (`src/multi_agent_dashboard/ui/agent_editor_mode.py`)

- Locate the “Advanced” tab (around line 632 where temperature is set).
- Add a number‑input field for `max_output` below temperature, with label “Max output tokens (0 = no limit)”.
- Store the value in `state["max_output"]` and persist it when saving the agent.

#### **✅ Step 3.2 – Agent cards / config views**

- Identify where agent configuration is displayed (likely in `run_mode.py`, `history_mode.py`).
- Extend the displayed fields to include `max_output` and `temperature`.
- For runs, show both the configured value and the actually used effective cap as `configured (effective)` e.g., “50 (30)” when they differ.

#### **✅ Step 3.3 – Exports**

- Verify that exported agent JSON (all export possibilities across the UI) includes the new `max_output` / `max_output_effective` and `temperature` fields.

### ✅ Phase 4: Execution Flow Integration

#### **✅ Step 4.0**(see APPENDIX A section for details)
- Use `NULL` as the **database default** for `max_output_effective` (not as "no limit"), indicating "no value was recorded".
- Backfill existing rows to `NULL`.
- For each new run, compute the effective cap via precedence rules and store the actual value (0 for "no limit", >0 for actual cap).
- This preserves historical clarity.

#### **✅ Step 4.1 – Implement precedence helper** (`src/multi_agent_dashboard/llm_client/core.py`)

- Create a helper `_effective_max_tokens(agent_spec: AgentSpec) -> int | None` that considers:
  1. `config.AGENT_OUTPUT_TOKEN_CAP` (already reflects .env > YAML precedence)
  2. `agent_spec.max_output`
  3. `config.STRICT_OUTPUT_TOKEN_CAP_OVERRIDE`
- Implement the logic described in the “Precedence & Override Logic” section.
- IMPORTANT: Check for edge case, where .env sets AGENT_OUTPUT_TOKEN_CAP=0 and STRICT_OUTPUT_TOKEN_CAP_OVERRIDE=true (must result in max_output_effective=0; currently there is null stored in the db, but for this special case should it be 0)

**IMPLEMENTATION NOTE:**
- Precedence logic location: Currently in `AgentSpec.effective_max_output()` method, not in separate helper in `llm_client/core.py`.
- Current implementation accepted by user decision

#### **✅ Step 4.2 – Wire effective cap into agent creation**

- In `AgentCreationFacade.create_agent()`, replace the current line that reads `config.AGENT_OUTPUT_TOKEN_CAP` with a call to the new helper.
- Pass the computed effective cap as `max_tokens` to `_model_factory.get_model()`.

#### **✅ Step 4.3 – Record effective cap in agent_run_configs**

- When storing the run configuration, write the **effective cap** (the value actually sent to the LLM) into `max_output_effective`.
- Write the **configured per‑agent cap** (`agent_spec.max_output`) into `max_output`.

### Phase 5: Testing & Verification

#### **✅ Step 5.1 – Manual verification steps**

- Create a new agent, set `max_output = 50`. Run the agent and confirm the LLM receives `max_tokens=50`.
- Set `.env` `AGENT_OUTPUT_TOKEN_CAP=30` (and `STRICT_OUTPUT_TOKEN_CAP_OVERRIDE=false`). Re‑run the same agent; the effective cap should be `min(30,50)=30`.
- Set `STRICT_OUTPUT_TOKEN_CAP_OVERRIDE=true`; the effective cap should be `30` (global cap) regardless of the agent’s setting.
- Set `max_output = 0` (or global cap = 0) and verify no `max_tokens` limit is applied (i.e., `max_tokens=None`).
- Verify the UI displays both the configured and actually used values in history views as `configured (effective)`.

## Decision Points (Resolved per Feedback)

1. **Naming:** DB columns named `max_output` and `max_output_effective`; field in `AgentSpec` named `max_output`.
2. **Character caps:** Precedence logic unchanged for character caps (they remain global‑only).
3. **UI placement:** `max_output` input placed below temperature in the Advanced tab with a clear label.
4. **History display:** Show as `configured (effective)` e.g., “50 (30)” when values differ.
5. **Optional vs int:** Use `int` with default `0` (0 means no limit); not optional.

## Risk Mitigation

- **Schema migration:** Use the existing `generate_migration.py` tool, which creates reversible SQL. Test the migration on a copy of the production DB before applying.
- **Back‑compatibility:** The `max_output` column defaults to 0, so existing agents continue to behave as before (no limit). NULL values are treated as 0.
- **UI changes:** Keep the new input optional; if the field is missing in old UI states, treat it as 0.
- **Execution flow:** The precedence helper will be thoroughly tested with the manual verification steps above.
- **Observability:** Storing `max_output_effective` ensures consistent debugging and audit trails.

## Dependencies & Order

The phases are designed to be independent and can be executed sequentially. Each phase produces a working system:

1. After Phase 1: DB can store `max_output`, but UI and execution ignore it.
2. After Phase 2: Configuration flags are available.
3. After Phase 3: UI can edit and display the value, but execution still uses the global cap.
4. After Phase 4: Full feature is operational.

This allows incremental validation and reduces the risk of breaking existing functionality.

## Estimated Effort

- **Schema & DAO changes:** 1‑2 hours (including migration generation)
- **Model & config updates:** 1 hour
- **UI changes:** 2‑3 hours (editing, display, exports)
- **Execution flow integration:** 2‑3 hours (helper, precedence logic, testing)
- **Manual verification:** 1‑2 hours

Total: ~8‑10 hours of focused development.

## Next Steps

1. Review this updated strategy with the product owner.
2. Begin implementation with Phase 1.

## APPENDIX A - Analysis of the Clarification: Use `NULL` as the **database default** for `max_output_effective`

**Your clarification**: Use `NULL` as the **database default** for `max_output_effective` (not as "no limit"), indicating "no value was recorded". Backfill existing rows to `NULL`. For each new run, compute the effective cap via precedence rules and store the actual value (0 for "no limit", >0 for actual cap). This preserves historical clarity.

### Current State vs. Proposed Change

**Current (`INTEGER DEFAULT 0`)**:
- `0` could mean: "no limit" OR "historical run before feature existed"
- All existing rows have `0` (default from migration)
- New runs store `0` when effective cap is "no limit"

**Proposed (`INTEGER DEFAULT NULL`)**:
- `NULL` = "no value recorded" (historical runs before feature)
- `0` = "effective cap computed as 0 (no limit)"
- `>0` = actual cap (when actual cap comes from agent configuration it's equal to max_output)
- Requires: schema migration + backfill + DAO NULL handling

### Impact Assessment

**Minimal code changes needed**:
1. **Schema**: Change `max_output_effective` default to `NULL` (new migration)
2. **Backfill**: Convert existing `0` → `NULL` for historical runs (via migration see 014_backfill_legacy_agent_providers.sql for reference)
3. **Snapshot builder**: Keep `effective_max_output() or 0` (stores 0 for "no limit")
4. **UI/reads**: Already handles `None` (`cfg.get("max_output_effective")` returns `None`)

**No changes required**:
- `AgentSpec.effective_max_output()` logic (still returns `None` for "no limit")
- Precedence rules
- `max_output` column (per‑agent config, stays `DEFAULT 0`)

### Recommendation

**Implement the change** - Benefits:
- **Clear audit trail**: Distinguishes "unrecorded" from "no limit"
- **Historical accuracy**: NULL correctly marks runs before feature existed
- **Future‑proof**: No ambiguity in analytics/debugging
- **Low risk**: UI already handles NULL, precedence logic unchanged

---

*This document is self‑contained and can be used as a blueprint for the implementation team.*