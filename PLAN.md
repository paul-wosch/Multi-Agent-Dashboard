# PLAN.md — LangChain + LiteLLM Integration for Provider Normalization

## Context

The Multi-Agent Dashboard currently supports three LLM providers:
- **OpenAI** (via `langchain-openai`)
- **Ollama** (local, via `langchain-ollama`)  
- **DeepSeek** (via `langchain-deepseek`)

Each provider requires custom handling for:
1. **Token counting** – different response fields (`usage`, `prompt_eval_count`, `completion_eval_count`)
2. **Structured output** – varying schema injection methods (`json_schema`, `function_calling`, `json_mode`)
3. **File uploads** – only OpenAI’s Responses API supports native multimodal attachments
4. **Tool calling** – only OpenAI path currently supports web search; custom tools are provider‑specific
5. **Feature support** – temperature, streaming, reasoning settings differ per provider

This fragmentation increases code complexity, hinders adding new providers, and creates inconsistent user experiences.



## Current Status & Regression Analysis (2026-02-01)

**Integration Progress**: LiteLLM integration is partially implemented with a `USE_LITELLM` flag that switches between:
- `USE_LITELLM=true`: Full LiteLLM translation layer using `langchain-litellm` 
- `USE_LITELLM=false`: Original provider-specific implementations (OpenAI, Ollama, DeepSeek)

**Critical Regressions Identified** (✅ All Resolved):
1. ✅ **Unified JSON Schema Format Incompatibility**: Fixed by dual‑path adapter in `_build_structured_output_adapter` that returns provider‑specific formats when `USE_LITELLM=false` and LiteLLM JSON‑Schema format only when `USE_LITELLM=true`.
2. ✅ **Provider‑Specific Logic Still Present But Overridden**: Restored provider‑specific `with_structured_output` wrapping loops in `create_agent_for_spec` with `if not self._use_litellm` guard. The `USE_LITELLM` flag now controls both model initialization and structured‑output logic.
3. ✅ **LiteLLM Configuration Issues**: Verified `litellm_config.py` correctly maps `OLLAMA_HOST` → `base_url`. DeepSeek authentication fixed via explicit `os.environ["DEEPSEEK_API_KEY"]` setting. Ollama endpoint propagation issue resolved (custom agent endpoints now properly override environment defaults). Universal os.environ strategy implemented for all providers.
4. ✅ **Temperature Handling for GPT‑5**: Enabled `litellm.drop_params = True` globally in LiteLLM import block, allowing graceful parameter dropping for unsupported parameters.

**Root Cause (Now Resolved)**: The integration attempted to unify structured output handling before LiteLLM path was fully validated, creating a single code path that broke backward compatibility. The `USE_LITELLM` flag initially only switched model initialization, not complete logic paths. **Fix applied**: Dual‑path logic now fully separates `USE_LITELLM=true` (LiteLLM JSON‑Schema) and `USE_LITELLM=false` (provider‑specific formats).

**Latest Dependencies**: `langchain-litellm` from GitHub main (Jan 30, 2026) includes PR #62 fixes for JSON Schema support, `tool_choice="any"` crashes, and proper `tool_calls` population.

**Clarified Constraints**:
1. **Goal Alignment**: Integration uses `langchain-litellm` to maintain LangChain agent architecture while building new path; removal of old code comes **later** after stable rollout.
2. **Dual‑Path Requirement**: Must maintain fully separate logic for `USE_LITELLM=true` (full LiteLLM) and `USE_LITELLM=false` (original implementations) until explicit removal.
3. **Parameter Handling**: Use LiteLLM's `drop_params` feature for graceful handling of unsupported parameters (e.g., GPT‑5 temperature), not hardcoded exceptions.
4. **Leverage LiteLLM Features**: Use `supports_response_schema()`, built‑in fallbacks, and provider detection instead of custom logic.
5. **Environment Variable Strategy**: Set all provider API keys in `os.environ` for consistent LiteLLM compatibility, avoiding provider‑specific exceptions.

## Goal

Replace provider‑specific adapters with **LiteLLM** as a universal translation layer, leveraging its:
- **Unified OpenAI‑compatible interface** for 100+ LLM APIs
- **Consistent usage/token tracking** across all providers
- **Built‑in structured output** (JSON mode, function calling)
- **Multimodal support** (file uploads, images, audio)
- **Tool/function calling** with automatic fallback strategies
- **Cost tracking** and **retry/fallback** logic
- **Streaming** and **reasoning** configuration

Integration will use the `langchain‑litellm` package to keep the existing LangChain agent architecture, with provider‑specific code removed **only after** LiteLLM path is fully validated and stable.

## Prerequisites

- Install `litellm>=1.81.6` and `langchain‑litellm` from GitHub main (includes PR #62 fixes for JSON Schema support, `tool_choice="any"` mapping, and proper `tool_calls` population).
- Keep existing provider‑specific packages (`langchain‑openai`, `langchain‑ollama`, `langchain‑deepseek`) for fallback and dual‑path operation.
- Ensure environment variables for each provider are still read (OPENAI_API_KEY, DEEPSEEK_API_KEY, OLLAMA_HOST) and correctly mapped to LiteLLM (`OLLAMA_HOST` → `base_url`, `DEEPSEEK_API_KEY` → `api_key`).
- Enable `litellm.drop_params = True` globally for graceful handling of unsupported parameters (e.g., GPT‑5 temperature).
- No breaking changes to the UI, database schema, or agent specs.

## Architectural Guardrails

**Critical Isolation Principle**: New (`USE_LITELLM=true`) and old (`USE_LITELLM=false`) paths must be as isolated as possible:

1. **Prefer new modules/functions/classes** for the `langchain‑litellm` path over modifications/conditionals inside existing structures.
2. **Minimal coupling**: Keep new and old logic separate and uncoupled. Example patterns:
   - Add `LiteLLMClient` class instead of modifying `LLMClient` with conditional branches.
   - Create `litellm_config.py` module instead of adding LiteLLM mappings to existing `config.py`.
   - Use composition/wrapper patterns rather than inheritance or mixed logic.
3. **Clear separation**: When changes are unavoidable in shared code, isolate them behind the `USE_LITELLM` flag with minimal footprint and maximal clarity.
4. **Dual‑path validation**: Both paths must remain independently testable and functional until explicit removal of old adapters.

This isolation ensures:
- **Safety**: Old path remains stable while new path is developed.
- **Testability**: Each path can be tested independently.
- **Removability**: Old adapters can be deleted cleanly when migration is complete.
- **Debugging**: Issues are clearly attributable to one path or the other.

## Implementation Steps

### Step 1 – Install and Configure LiteLLM ✅ COMPLETED

**Sub‑steps:**

1. Add dependencies to `pyproject.toml` (install `langchain‑litellm` from GitHub main for latest fixes):
   ```toml
   dependencies = [
       "litellm>=1.81.6",
       "langchain-litellm @ git+https://github.com/langchain-ai/langchain-litellm.git",
       # keep existing langchain‑openai, langchain‑ollama, langchain‑deepseek for dual‑path support
   ]
   ```

2. Create a new configuration module `src/multi_agent_dashboard/litellm_config.py` that:
   - Maps provider IDs (`openai`, `ollama`, `deepseek`) to LiteLLM model strings (`openai/gpt-4o`, `ollama/llama3`, `deepseek/deepseek-chat`)
   - Reads provider‑specific environment variables and passes them to LiteLLM’s `api_key`/`base_url` settings
   - Defines a global LiteLLM `api_base` fallback for custom endpoints

3. Add a `LiteLLMClient` class in `llm_client.py` that wraps `litellm.completion` (or `ChatLiteLLM` from `langchain‑litellm`) and unifies:
   - Request formatting (messages, tools, response_format)
   - Response parsing (content, usage, finish_reason)
   - Error handling (retry, fallback, unsupported feature detection)

**Verification:**
- ✅ `pip install -e .` succeeds
- ✅ Import `litellm` and `langchain_litellm` without errors  
- ✅ Unit test that `LiteLLMClient` can be instantiated with each provider

**Completed on 2026‑02‑01:** All sub‑steps implemented and verified. `LiteLLMClient` class added to `llm_client.py`, provider mapping in `litellm_config.py`, dependencies updated in `pyproject.toml`.

---

### Step 2 – Replace Provider‑Specific Token Counting ⚠️ PARTIALLY COMPLETED

**Current problem:** `llm_client.py` and `engine.py` contain separate logic to extract `usage` from OpenAI, `prompt_eval_count`/`completion_eval_count` from Ollama, and `usage_metadata` from DeepSeek.

**Sub‑steps:**

1. ✅ In `LiteLLMClient.invoke`, call `litellm.completion(…, extra_body={"metadata": true})` to guarantee a unified `usage` object in the response (LiteLLM normalizes this across providers).  
   *Implemented in `llm_client.py:527-533`; verified with unit tests.*

2. ⏳ Remove all provider‑specific usage‑extraction helpers (`_extract_usage_from_candidate`, `_extract_usage_from_ollama`, `_extract_usage_from_deepseek`).  
   *Deferred to Phase 4: these helpers are still used by the legacy `LLMClient`. Will be removed when `LiteLLMClient` becomes the primary client.*

3. ✅ Update `engine.py`’s `_compute_cost` to read the normalized `usage` object and map provider‑ID to LiteLLM’s pricing dictionary (LiteLLM provides per‑model pricing; we can keep our own `*_PRICING` tables for consistency).  
   *Implemented in `engine.py:258-301`; added parsing of LiteLLM‑style `provider/model` strings and extraction of provider prefix for pricing lookup.*

4. ✅ Ensure token counts are stored in the database exactly as before (no schema change).  
   *Token counting via normalized `usage` object preserves existing database fields.*

**Verification:**
- ✅ Existing cost‑tracking tests pass (no regressions)
- ✅ Unit tests for `LiteLLMClient` token extraction (`tests/test_litellm_client.py`) pass
- ✅ Engine cost calculation handles both old model strings and LiteLLM `provider/model` formats
- ⚠️ Manual integration with each provider pending (requires full LiteLLM rollout)

**Status:** Core token‑counting normalization implemented for `LiteLLMClient`. Legacy `LLMClient` still uses provider‑specific helpers (will be removed in Phase 4).

---

### Step 3 – Implement Unified Structured Output Handling ⚠️ PARTIALLY COMPLETED (REGRESSIONS)

**Current problem:** `_build_structured_output_adapter` branches per provider (OpenAI `json_schema`, Ollama `json_schema`, DeepSeek `function_calling`/`json_mode`). The initial unification introduced regressions because it forced a single JSON‑Schema format (LiteLLM) across both `USE_LITELLM` paths, breaking provider‑specific APIs.

**Note**: A regression occurred during the initial implementation of Step 3 when the unified JSON‑Schema format was applied to both `USE_LITELLM` paths. This has been resolved by implementing true dual‑path structured output (sub‑step 2). Guardrails have been added to PLAN.md (see "Clarified Constraints" section) to prevent similar regressions in future integration steps.

**Sub‑steps:**

1. ✅ Replace `_build_structured_output_adapter` with a single code path that calls LiteLLM’s `response_format` parameter (supports `json_object`, `json_schema`, `function_calling`).  
   *Implemented but causes regressions for `USE_LITELLM=false` (legacy providers).*

2. ⚠️ **Fix: Implement true dual‑path structured output**:
   - When `USE_LITELLM=true`: Use LiteLLM’s `response_format` with JSON Schema format and leverage `langchain‑litellm` PR #62 fixes for proper `tool_calls` population.
   - When `USE_LITELLM=false`: Restore provider‑specific `with_structured_output` wrapping in `create_agent_for_spec` (OpenAI `json_schema`, Ollama `json_schema`, DeepSeek `function_calling`/`json_mode`).
   - Keep both paths fully separate until LiteLLM path is validated and explicitly switched to default.

3. ⚠️ Use LiteLLM’s `strict=True` flag to enforce schema validation where the provider supports it; for unsupported providers, LiteLLM will fall back to prompt‑based JSON enforcement.  
   *Requires checking provider support via `litellm.supports_response_schema()`.*

4. ⚠️ Update `structured_schemas.py` to produce JSON‑Schema dictionaries compatible with LiteLLM’s `response_format` **only** for the LiteLLM path; keep existing schemas for legacy providers.

5. ⚠️ Add graceful fallback for unsupported structured‑output features using LiteLLM’s `drop_params=True` to log a warning about dropping unsupported parameters and name them when possible (e.g., GPT‑5 temperature) instead of raising errors.

6. ⚠️ **Fix LiteLLM Configuration Issues**:
   - ✅ `OLLAMA_HOST` properly maps to `base_url` with `http://` prefix; custom agent endpoints override environment defaults (implemented in `_init_chat_model_with_litellm()` with comprehensive logging)
   - ✅ **DeepSeek authentication fixed**: Added explicit `os.environ["DEEPSEEK_API_KEY"] = api_key` in `_init_chat_model_with_litellm()` to meet LiteLLM's provider requirements.
   - ✅ **Adopt universal `os.environ` strategy**: Set all provider API keys in environment variables for consistent LiteLLM compatibility, eliminating provider‑specific exceptions.
   - ⚠️ Add validation that required environment variables are present (optional enhancement).

**Verification:**
- Structured output works for all three providers with the same UI toggle **in both `USE_LITELLM` modes**
- JSON validation errors are reported consistently
- Token counts include the structured‑output overhead
- All existing structured‑output tests pass (23/23) **without regressions**

**Status:** Unified structured output adapter implemented with dual‑path handling restored. The regression affecting legacy providers has been resolved. Configuration issues fully resolved (Ollama endpoint propagation fixed, DeepSeek authentication fixed via explicit `os.environ` setting, universal environment variable strategy implemented). Remaining work: strict validation, schema compatibility, and graceful fallback (sub‑steps 3‑5).

---

### Step 4 – Enable Provider‑Agnostic File Uploads

**Current problem:** File attachments are only processed for OpenAI’s Responses API (`use_responses_api=True`); other providers receive a plain‑text concatenation.

**Sub‑steps:**

1. Leverage LiteLLM’s native multimodal support: convert file attachments into `content` blocks with `type: "image_url"`, `type: "document_url"`, or `type: "text"`.

2. Add a preprocessing function in `llm_client.py` that:
   - Detects file MIME types
   - Encodes small files inline (base64)
   - For large files, uploads to a temporary store and passes a URL (if the provider supports URL‑based content)

3. Modify the `files` parameter in `invoke_agent` to pass the processed content blocks directly to LiteLLM.

4. Ensure fallback for providers without multimodal support: LiteLLM will automatically strip unsupported content types and keep only the text parts.

5. **Dual‑path handling**: Keep existing file‑attachment logic for `USE_LITELLM=false` (legacy providers) until LiteLLM path is fully validated.

**Verification:**
- Upload a `.txt`, `.pdf`, `.csv` file with each provider; the agent receives the file content
- Image files are accepted for vision‑capable models (e.g., GPT‑4o, Claude)
- No breaking changes to the UI file‑upload component

---

### Step 5 – Implement Tool Calling Across All Providers

**Current problem:** Tool calling is only wired for OpenAI’s Responses API (web search). Custom tools cannot be used with Ollama or DeepSeek.

**Sub‑steps:**

1. Use LiteLLM’s `tools` parameter (OpenAI‑style tool definitions) which LiteLLM translates to the provider’s native tool‑calling format.

2. Update `models.py`’s `_build_tools_args` to produce tool schemas compatible with LiteLLM (already OpenAI‑style, so minimal change).

3. In `llm_client.py`, pass the tools list to `litellm.completion(tools=…)` and parse the `tool_calls` response.

4. Add a middleware that executes tool calls and returns results to the LLM (re‑use existing LangChain agent tool‑execution logic).

5. For providers that do not support tool calling (e.g., some local Ollama models), LiteLLM will automatically fall back to a description‑based prompt.

6. **Dual‑path handling**: Keep existing tool‑calling logic (OpenAI Responses API) for `USE_LITELLM=false` path until LiteLLM path is validated.

**Verification:**
- Web search tool works with all three providers
- Custom tools (e.g., calculator, database query) can be added and invoked regardless of provider
- Tool‑call results are correctly injected into the conversation

---

### Step 6 – Standardize Temperature and Model Parameters

**Current problem:** Temperature, max_tokens, top_p, etc., are passed via provider‑specific kwargs. Some models have specific requirements (e.g., GPT‑5 requires temperature=1.0) that cause errors when unsupported values are passed.

**Sub‑steps:**

1. ✅ Consolidate all model‑parameter handling in `LiteLLMClient` using LiteLLM’s unified parameter set (`temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `seed`).  
   *Implemented for `USE_LITELLM=true` path.*

2. ✅ Map agent‑spec fields (`temperature`, `max_output_tokens`, `stop_sequences`) to these parameters.  
   *Mapping exists but needs graceful handling for unsupported values.*

3. ⚠️ **Fix: Use LiteLLM’s `drop_params` feature for graceful parameter handling**:
   - Set `litellm.drop_params = True` globally to log warnings about dropping unsupported parameters instead of raising errors.
   - This solves the GPT‑5 temperature issue and other provider‑specific constraints without hardcoded exceptions.
   - Keep provider‑specific parameter translation in `ChatModelFactory.get_model` for the `USE_LITELLM=false` path until full migration.

4. ⚠️ Add validation for critical parameters: ensure temperature is within provider‑supported range (0‑2 for most providers) and max_tokens is within model context limits.

**Verification:**
- Temperature slider in the UI affects all providers equally **without crashing on unsupported values**
- Max output tokens limit is respected
- Random seed produces deterministic outputs where supported
- GPT‑5 and other restrictive models work with default parameters (unsupported values are dropped with warnings logged)

---

### Step 7 – Add Streaming Support

**Current problem:** Streaming is not implemented; responses are returned only after completion.

**Sub‑steps:**

1. Extend `LiteLLMClient.invoke` with a `stream=True` parameter that yields chunks via `litellm.completion(stream=True)`.

2. Propagate streaming chunks through the existing LangChain agent middleware (if needed) or directly to the UI.

3. Update the UI’s `run_mode.py` to display streaming tokens as they arrive (optional future enhancement).

4. **Dual‑path handling**: Streaming will only be available for `USE_LITELLM=true` path initially; legacy providers continue to use non‑streaming responses.

**Verification:**
- Streaming can be enabled via a UI toggle (later phase)
- Each provider delivers token streams where supported (LiteLLM handles translation)

---

### Step 8 – Graceful Handling of Unsupported Features

**Current problem:** The code must manually detect and work around provider limitations (e.g., DeepSeek‑reasoner rejects `tool_choice`). The initial integration removed provider‑specific fallback loops, causing regressions.

**Sub‑steps:**

1. ✅ **Rely on LiteLLM’s built‑in fallback system**: Enable `litellm.drop_params = True` globally to log warnings about dropping unsupported parameters (e.g., `tool_choice`, `response_format`) instead of raising errors.  
   *This provides graceful degradation for unsupported features across all providers.*

2. ⚠️ **Keep provider‑specific fallback loops temporarily**: For `USE_LITELLM=false` path, maintain the `methods` loop for DeepSeek and other provider‑specific workarounds in `create_agent_for_spec`. Remove only after LiteLLM path is fully validated and default.

3. ⚠️ **Use LiteLLM’s feature detection**: Query `litellm.supports_response_schema(provider)` and `litellm.get_supported_openai_params(provider)` to dynamically disable UI toggles for unsupported features.  
   *Replace hardcoded provider checks with runtime detection.*

4. ⚠️ **Leverage `langchain‑litellm` PR #62 fixes**: Ensure `tool_choice="any"` is mapped to `"required"` for unsupported providers, preventing crashes.

**Verification:**
- Attempting to use tool calling with a model that doesn’t support it yields a clear user‑friendly message **or falls back with a warning logged**
- The system does not crash when an unsupported parameter is passed (parameters are dropped with warning)
- Provider‑specific fallbacks continue to work in legacy mode (`USE_LITELLM=false`)

---

### Step 9 – Update UI and Configuration

**Sub‑steps:**

1. Update provider dropdowns to reflect that LiteLLM is the underlying engine (keep same provider IDs for backward compatibility).

2. Add a hidden advanced setting `use_litellm` (default `True`) that can be turned off to revert to the old provider‑specific adapters (safety switch).

3. Extend the provider‑features detection in `ui/agent_editor_mode.py` to query LiteLLM for supported capabilities (streaming, tools, vision, JSON mode) and grey out unsupported options.

4. Update documentation (README, CONFIG.md) to reflect the new LiteLLM‑based setup.

**Verification:**
- Existing agent specs load without modification
- Provider selection still shows “OpenAI”, “Ollama (local)”, “DeepSeek” but uses LiteLLM under the hood
- Advanced settings page contains the LiteLLM toggle

---

### Step 10 – Testing and Validation

**Sub‑steps:**

1. Write unit tests for `LiteLLMClient` covering:
   - Token counting normalization
   - Structured output with each provider
   - File‑attachment preprocessing
   - Tool‑call parameter translation

2. Integration tests that run a real agent with each provider (using mocked API calls where appropriate).

3. End‑to‑end smoke tests:
   - Create an agent with each provider, enable structured output, run a prompt
   - Attach a file, verify the agent receives the content
   - Enable a custom tool, invoke it, check results

4. Performance regression test: compare latency and token counts before/after the change (should be within 5‑10%).

**Verification:**
- All existing tests pass (`pytest` runs green)
- New integration tests pass
- Manual smoke test with each provider succeeds

---

## Rollout Phases

**Phase 1 (Steps 1–3)** – Core LiteLLM integration and token‑counting normalization.  
- Implement `LiteLLMClient` and configuration mapping (already completed).
- **Critical**: Maintain dual‑path architecture—keep legacy provider‑specific code fully functional for `USE_LITELLM=false`.
- Set `use_litellm` flag default to `False` (already set).

**Phase 2 (Steps 4–6)** – File uploads, tool calling, parameter standardization.  
- Implement file‑upload preprocessing using LiteLLM’s multimodal support.
- Add tool‑calling via LiteLLM’s `tools` parameter.
- Keep provider‑specific fallback loops for legacy path.

**Phase 3 (Steps 7–9)** – Streaming, UI updates, configuration.  
- Add streaming support via LiteLLM’s `stream=True`.
- Update UI to query LiteLLM for supported features (`supports_response_schema`, `get_supported_openai_params`).
- Update documentation and configuration examples.

**Phase 4 (Step 10)** – Testing, validation, and documentation.  
- Comprehensive unit and integration tests for both `USE_LITELLM` paths.
- End‑to‑end smoke tests with all three providers.
- Performance benchmarking and token‑counting validation.
- Switch `use_litellm` default to `True` after validation passes.

**Phase 5 (Post‑validation)** – Remove old adapters and provider‑specific code.  
- Delete provider‑specific usage extraction helpers (`_extract_usage_from_*`).
- Remove provider‑specific parameter translation from `ChatModelFactory.get_model`.
- Remove provider‑specific fallback loops (DeepSeek `methods` loop, etc.).
- Final code reduction goal: ~300 lines removed.

Each phase should be merged separately, with the `use_litellm` flag defaulting to `False` until Phase 4. Removal of old adapters (Phase 5) is **optional** and can be deferred if continued dual‑path operation is preferred.

## Success Metrics

- **Code reduction (Phase 5):** Remove at least 300 lines of provider‑specific branching **after** LiteLLM path is validated; dual‑path operation may keep some provider‑specific code longer.
- **Feature parity:** All existing features work identically across providers in both `USE_LITELLM` modes.
- **Extensibility:** Adding a new provider requires only adding its model string to the LiteLLM mapping (once LiteLLM path is default).
- **Performance:** No significant increase in latency or token‑counting errors compared to native provider implementations.
- **User experience:** No visible change for existing users except improved consistency; advanced `use_litellm` toggle allows fallback to legacy adapters if needed.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| LiteLLM introduces breaking changes | Pin to a specific minor version (`litellm~=1.81.6`) and test upgrades thoroughly |
| **Regression from unified code paths** | **Maintain dual‑path architecture (`USE_LITELLM` flag) until LiteLLM path is fully validated; keep provider‑specific logic separate** |
| LiteLLM does not support a required feature | Keep the old provider‑specific adapters as fallback paths; contribute missing feature upstream |
| Token counting discrepancies | Compare LiteLLM’s `usage` with provider‑native counts during testing; adjust mapping if needed |
| Increased latency due to translation layer | Benchmark and cache LiteLLM model objects; use LiteLLM’s native provider clients where possible |
| Unsupported parameters cause errors (e.g., GPT‑5 temperature) | Enable `litellm.drop_params = True` to log warnings about dropping unsupported parameters |

## Future Expansion

Once LiteLLM is integrated, adding a new provider (e.g., Anthropic, Gemini, Groq) becomes a configuration change:

1. Add the provider’s API key environment variable
2. Map its provider ID to a LiteLLM model string (`anthropic/claude‑3‑opus`, `gemini/gemini‑pro`, `groq/llama3‑70b‑8192`)
3. (Optional) add pricing information to `config.py`

No code changes in `llm_client.py`, `engine.py`, or the UI are required.

---

## References

- [LiteLLM Documentation](https://docs.litellm.ai)
- [LangChain‑LiteLLM Integration](https://python.langchain.com/docs/integrations/chat/litellm)
- [LiteLLM Supported Providers](https://docs.litellm.ai/docs/providers)
- `langchain‑litellm` PR #62 (Jan 30, 2026) – JSON Schema support, `tool_choice="any"` mapping, proper `tool_calls` population
- Existing provider‑specific code in `src/multi_agent_dashboard/llm_client.py`, `engine.py`, `config.py`

---

*Plan generated on 2026‑02‑01.*
*Target completion: 2026‑02‑15.*