# PLAN.md — DeepSeek Provider Integration

## Context
- Add DeepSeek as a provider while keeping the existing provider-agnostic architecture.
- Support structured output (schema-based) via LangChain `with_structured_output(method="function_calling")` where available.
- Preserve token accounting and pricing for cost tracking.
- Unit tests are optional and should be added only after a feature works as expected to lock in behavior; smoke/manual validation is acceptable earlier in the cycle.

## Steps

### Step 1 — Configuration & Pricing ✅
- Add DeepSeek env wiring in `config.py` (API key).
- Add DEEPSEEK_PRICING for `deepseek-chat` and `deepseek-reasoner` at $0.28/1M input tokens and $0.42/1M output tokens.
- Verification: unit/assert check that config exposes pricing and env keys; lint passes.

### Step 2 — Provider Adapter in LLMClient/Factory ✅
- Add DeepSeek to `ChatModelFactory`/`LLMClient` init path (`model_provider="deepseek"`), supporting endpoint override and timeouts.
- Implement structured-output handling via `with_structured_output(method="function_calling")`, preserving usage/token metadata; ensure token mapping for DeepSeek response fields.
- Encapsulate DeepSeek-specific logic so OpenAI/Ollama paths remain untouched (no behavior changes for existing providers).
- Verification: targeted unit tests for DeepSeek client adapter (usage/token extraction), no regressions for existing providers. ✅

### Step 3 — UI / Agent Spec plumbing ✅
- Surface DeepSeek in provider selection and persist provider metadata using existing fields; no new schema columns.
- Ensure structured-output controls work unchanged for DeepSeek.
- Update _compute_cost to recognize provider_id == "deepseek" and fetch pricing from config.DEEPSEEK_PRICING
- In _build_structured_output_adapter, DeepSeek branch wraps a plain JSON schema into an OpenAI-style tool/function schema ({name, description, parameters}) so that  
  with_structured_output(method="function_calling") can enforce structured output for DeepSeek 
- For deepseek-reasoner: on 400 tool_choice error, retry json_mode to enforce JSON formatting without tool invocation.  
- Verification: manual smoke (set provider to deepseek in editor, save/reload), snapshot/state round-trip test if available.


### Step 4 — Tests & Docs ❌
- Add integration-like tests (mocked) for deepseek-chat and deepseek-reasoner: structured output on/off, usage tokens present.
- Update docs/README snippet for DeepSeek setup (.env, models, pricing, structured-output caveat for reasoner).
- Verification: `pytest` targeted suite passes; docs lint (if applicable).
