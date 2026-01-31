# PLAN.md — DeepSeek Provider Integration

## Context
- Add DeepSeek as a provider while keeping the existing provider-agnostic architecture.
- Support structured output (schema-based) via LangChain `with_structured_output(method="function_calling")` where available.
- Preserve token accounting and pricing for cost tracking.

## Steps

### Step 1 — Configuration & Pricing
- Add DeepSeek env wiring in `config.py` (API key, optional base URL).
- Extend pricing table for `deepseek-chat` and `deepseek-reasoner` at $0.28/1M input tokens and $0.42/1M output tokens.
- Verification: unit/assert check that config exposes pricing and env keys; lint passes.

### Step 2 — Provider Adapter in LLMClient/Factory
- Add DeepSeek to `ChatModelFactory`/`LLMClient` init path (`model_provider="deepseek"`), supporting endpoint override and timeouts.
- Implement structured-output handling via `with_structured_output(method="function_calling")`, preserving usage/token metadata; ensure token mapping for DeepSeek response fields.
- Verification: targeted unit tests for DeepSeek client adapter (usage/token extraction), no regressions for existing providers.

### Step 3 — UI / Agent Spec plumbing
- Surface DeepSeek in provider selection and persist provider metadata using existing fields; no new schema columns.
- Ensure structured-output controls work unchanged for DeepSeek.
- Verification: manual smoke (set provider to deepseek in editor, save/reload), snapshot/state round-trip test if available.

### Step 4 — Tests & Docs
- Add integration-like tests (mocked) for deepseek-chat and deepseek-reasoner: structured output on/off, usage tokens present.
- Update docs/README snippet for DeepSeek setup (.env, models, pricing, structured-output caveat for reasoner).
- Verification: `pytest` targeted suite passes; docs lint (if applicable).

