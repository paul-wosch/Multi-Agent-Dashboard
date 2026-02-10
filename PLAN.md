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

## Current OpenAI Model Landscape (2026)

**Key Insight**: The codebase references outdated OpenAI models (`gpt‑4o‑search‑preview`) while current frontier models (GPT‑5 series) use different naming conventions and capabilities. Integration must adapt to model deprecations and evolving web‑search support.

### Model Categories and Web‑Search Support

**Frontier Models (Current)**:
- `gpt‑5.2` / `gpt‑5.2 pro` – Latest flagship (400K context, web‑search support)
- `gpt‑5.1` – Previous flagship (configurable reasoning, web‑search support)
- `gpt‑5` / `gpt‑5 mini` / `gpt‑5 nano` – Intelligent reasoning models

**GPT‑4 Series (Legacy, being deprecated)**:
- `gpt‑4.1` / `gpt‑4.1 mini` / `gpt‑4.1 nano` – Smart non‑reasoning models
- `gpt‑4o` / `gpt‑4o mini` – Fast models (deprecated 2025‑2026)

**Specialized Search Models**:
- `gpt‑5‑search‑api` – Dedicated search model (Chat Completions API)
- `gpt‑4o‑search‑preview` – Legacy search model (deprecated)

### Web‑Search Evolution

- **Responses API**: Most current models support `{"type": "web_search"}` tool (GA version)
- **Completions API**: Specialized search models use `web_search_options` parameter
- **Domain filtering**: Available in GA `web_search` (allow‑lists up to 100 domains)
- **Deprecation Timeline**: GPT‑4 series models are being replaced by GPT‑5 series (2025‑2026)

### Integration Implications

1. **Dynamic detection**: Use LiteLLM’s `supports_web_search()` and `get_supported_openai_params()` instead of hardcoded model lists.
2. **API‑switching**: Respect the `use_responses_api` flag to select between Responses API (`tools=[{"type": "web_search"}]`) and Completions API (`web_search_options`).
3. **Model‑awareness**: Update provider‑model mapping in `litellm_config.py` to include current frontier models.


## Current Status & Regression Analysis (2026-02-05)

**Integration Progress**: LiteLLM integration is partially implemented with a `USE_LITELLM` flag that switches between:
- `USE_LITELLM=true`: Full LiteLLM translation layer using `langchain-litellm` 
- `USE_LITELLM=false`: Original provider-specific implementations (OpenAI, Ollama, DeepSeek)

**Critical Regressions Identified** (✅ All Resolved):
1. ✅ **Unified JSON Schema Format Incompatibility**: Fixed by dual‑path adapter in `_build_structured_output_adapter` that returns provider‑specific formats when `USE_LITELLM=false` and LiteLLM JSON‑Schema format only when `USE_LITELLM=true`.
2. ✅ **Provider‑Specific Logic Still Present But Overridden**: Restored provider‑specific `with_structured_output` wrapping loops in `create_agent_for_spec` with `if not self._use_litellm` guard. The `USE_LITELLM` flag now controls both model initialization and structured‑output logic.
3. ✅ **LiteLLM Configuration Issues**: Verified `litellm_config.py` correctly maps `OLLAMA_HOST` → `base_url`. DeepSeek authentication fixed via explicit `os.environ["DEEPSEEK_API_KEY"]` setting. Ollama endpoint propagation issue resolved (custom agent endpoints now properly override environment defaults). Universal os.environ strategy implemented for all providers.
4. ✅ **Temperature Handling for GPT‑5**: Enabled `litellm.drop_params = True` globally in LiteLLM import block, allowing graceful parameter dropping for unsupported parameters.
5. ✅ **DeepSeek‑reasoner Structured Output Regression**: Fixed by detecting reasoner models in LiteLLM path and using `json_mode` before `function_calling`, ensuring DeepSeek‑reasoner agents support structured output with `USE_LITELLM=true`.

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

### Step 3 – Implement Unified Structured Output Handling ✅ COMPLETED

**Current problem:** `_build_structured_output_adapter` branches per provider (OpenAI `json_schema`, Ollama `json_schema`, DeepSeek `function_calling`/`json_mode`). The initial unification introduced regressions because it forced a single JSON‑Schema format (LiteLLM) across both `USE_LITELLM` paths, breaking provider‑specific APIs.

**Note**: A regression occurred during the initial implementation of Step 3 when the unified JSON‑Schema format was applied to both `USE_LITELLM` paths. This has been resolved by implementing true dual‑path structured output (sub‑step 2). Guardrails have been added to PLAN.md (see "Clarified Constraints" section) to prevent similar regressions in future integration steps.

**Sub‑steps:**

1. ✅ Replace `_build_structured_output_adapter` with a single code path that calls LiteLLM’s `response_format` parameter (supports `json_object`, `json_schema`, `function_calling`).  
   *Implemented but causes regressions for `USE_LITELLM=false` (legacy providers).*

2. ✅ **Fix: Implement true dual‑path structured output**:
   - When `USE_LITELLM=true`: Use LiteLLM’s `response_format` with JSON Schema format and leverage `langchain‑litellm` PR #62 fixes for proper `tool_calls` population.
   - When `USE_LITELLM=false`: Restore provider‑specific `with_structured_output` wrapping in `create_agent_for_spec` (OpenAI `json_schema`, Ollama `json_schema`, DeepSeek `function_calling`/`json_mode`).
   - Keep both paths fully separate until LiteLLM path is validated and explicitly switched to default.
   - **DeepSeek‑reasoner compatibility**: Added model detection to try `json_mode` before `function_calling` for reasoner models, fixing the regression for DeepSeek agents using structured output.

3. ✅ Use LiteLLM’s `strict=True` flag to enforce schema validation where the provider supports it; for unsupported providers, LiteLLM will fall back to prompt‑based JSON enforcement.  
   *Implemented with provider‑specific strict parameter handling in `_build_structured_output_adapter`.*

4. ✅ Update `structured_schemas.py` to produce JSON‑Schema dictionaries compatible with LiteLLM’s `response_format` **only** for the LiteLLM path; keep existing schemas for legacy providers.
   *Schema compatibility maintained through unified JSON Schema format in LiteLLM path.*

5. ✅ Add graceful fallback for unsupported structured‑output features using LiteLLM’s `drop_params=True` to log a warning about dropping unsupported parameters and name them when possible (e.g., GPT‑5 temperature) instead of raising errors.
   *Global `litellm.drop_params = True` enabled in `litellm_config.py`.*

6. ✅ **Fix LiteLLM Configuration Issues**:
   - ✅ `OLLAMA_HOST` properly maps to `base_url` with `http://` prefix; custom agent endpoints override environment defaults (implemented in `_init_chat_model_with_litellm()` with comprehensive logging)
   - ✅ **DeepSeek authentication fixed**: Added explicit `os.environ["DEEPSEEK_API_KEY"] = api_key` in `_init_chat_model_with_litellm()` to meet LiteLLM's provider requirements.
   - ✅ **Adopt universal `os.environ` strategy**: Set all provider API keys in environment variables for consistent LiteLLM compatibility, eliminating provider‑specific exceptions.
   - ✅ Environment validation handled by LiteLLM's error reporting (optional enhancement implemented).

**Verification:**
- Structured output works for all three providers with the same UI toggle **in both `USE_LITELLM` modes**
- JSON validation errors are reported consistently
- Token counts include the structured‑output overhead
- All existing structured‑output tests pass (23/23) **without regressions**

**Status:** Unified structured output adapter implemented with dual‑path handling fully functional. All three providers (OpenAI, Ollama, DeepSeek) support structured output with `USE_LITELLM=true`. The DeepSeek‑reasoner regression has been fixed with model detection logic. Configuration issues resolved, strict parameter handling implemented, and graceful fallback enabled via `drop_params`. Step 3 completed successfully.

---

### Step 4 – Enable Provider‑Agnostic File Uploads ✅ COMPLETED

**Current problem:** File attachments are only processed for OpenAI’s Responses API (`use_responses_api=True`); other providers receive a plain‑text concatenation.

**Updated constraints:**
- **Legacy path locked**: `USE_LITELLM=false` path must stay unchanged – no modifications to existing file‑handling logic.
- **Isolation**: All new logic must be isolated from legacy code in dedicated module(s).
- **Separation of concerns**: New logic must not touch legacy branches; reuse existing UI, engine, and agent‑runtime layers unchanged.
- **Non‑persistent storage**: Files remain in pipeline state only (no disk/database writes).
- **Dynamic capability detection**: Use `litellm.supports_feature()` or equivalent runtime detection instead of static mapping.

**Sub‑steps:**

1. ✅ **Research LiteLLM capability APIs** (prerequisite):
   - Verified existence of `litellm.supports_response_schema()` and `litellm.get_supported_openai_params()` (though they may return `None` or incorrect values).
   - Identified that `litellm.supports_feature()` does **not** exist in the installed version.
   - Updated `litellm_config.supports_feature` to attempt dynamic detection first, then fall back to static mapping.

2. ✅ **Create `multimodal_handler.py` module** with responsibilities:
   - Defined interface: `prepare_multimodal_content(provider_id, model, files, profile, prompt)` returning either a string (text concatenation) or a list of content parts (OpenAI‑style).
   - Detects provider capabilities via `litellm_config.supports_feature` (cached).
   - Converts image files to base64 `image_url` content blocks when provider supports vision.
   - Implements fallback routing: unsupported file types → text placeholder.
   - Enforces size limits (10 MB for base64 encoding).
   - Caches capability detection per (provider, model) using `lru_cache`.

3. ✅ **Extend `llm_client.py` with isolated LiteLLM‑path logic**:
   - Added conditional branch in `invoke_agent` that activates when `self._use_litellm` is True.
   - Inside the branch, lazily imports `multimodal_handler` and calls `prepare_multimodal_content`.
   - Keeps the `else` branch **byte‑for‑byte identical** to current text‑concatenation logic.
   - Propagates provider info (`_provider_id`, `_model`, `_provider_features`) to the agent instance in `create_agent_for_spec`.
   - Builds the user message content as either a list (multimodal parts) or a string (text concatenation).
   - Ensures any failure in multimodal preparation falls back to text‑concatenation (matching legacy behavior).

4. ✅ **Implement capability‑aware file processing with fallbacks** (completed):
   - ✅ **Images** (jpg, png, gif, webp): Convert to base64 `image_url` content blocks if provider supports vision.
   - ✅ **Text files** (txt, md, csv, json): Decode to UTF‑8 and append as `{"type": "text", "text": …}` content parts.
   - ✅ **PDFs** (pdf, docx): Fixed with three‑stage fallback: (1) PDF text extraction via `pypdf` (if installed), (2) UTF‑8 decode with replacement characters for metadata, (3) placeholder for truly binary files. PDF inspection now works on LiteLLM path.
   - ✅ **Binary unsupported**: Mention filename only (placeholder).
   - ✅ **Base64 encoding lazily**: Only performed when provider supports vision and file is an image.
   - ✅ **Fallback function**: `provider_supports_vision()` uses cached detection.
   - ✅ **Model‑aware detection**: `supports_feature()` now accepts optional `model` parameter for model‑specific capability detection (e.g., Ollama `llava` vs `llama3`).

5. ✅ **Preserve existing layers unchanged**:
   - **UI** (`run_mode.py`): File uploader, size limits, metadata collection – no changes.
   - **Engine** (`engine.py`): Files stored in `engine.state["files"]` and injected into agents – no changes.
   - **Agent Runtime** (`models.py`): Text/binary splitting, UTF‑8 decoding, `llm_files_payload` building – no changes.

6. ✅ **Testing and documentation** (completed):
   - ✅ Run existing file‑attachment tests with `USE_LITELLM=false` (pass unchanged).
   - ✅ Smoke tests with mocked agents confirm dual‑path isolation works.
   - ✅ **Manual testing** – text files and PDFs fixed on LiteLLM path, all file types processed correctly.
   - ✅ Integration tests added for `USE_LITELLM=true` with image, PDF, and text uploads (tests pass).
   - ✅ `AGENTS.md` updated with file‑upload architecture and usage notes.

**Verification** (current state):
- ✅ **Isolation**: Legacy path (`USE_LITELLM=false`) works exactly as before – text‑concatenation only.
- ✅ **Image handling**: PNG files are base64‑encoded for vision‑capable providers (OpenAI) and fall back to text concatenation for non‑vision providers (DeepSeek, Ollama).
- ✅ **Text files**: Fixed on LiteLLM path for all providers – text files are decoded and attached as text content parts.
- ✅ **PDF files**: Fixed on LiteLLM path for all providers – PDFs are extracted via pypdf (if installed) or fallback to UTF‑8 decode.
- ✅ **No UI changes**: File‑upload component unchanged.
- ✅ **Caching**: Capability detection cached via `lru_cache`.
- ✅ **Model‑aware detection**: Vision/tool detection now uses model‑specific checks via `supports_feature(provider_id, feature, model)`.
- ✅ **Fallback**: Non‑vision providers automatically fall back to text concatenation.

**Status:** Step 4 fully implemented and validated. All file types (images, text, PDFs) are correctly processed on the LiteLLM path with proper fallbacks. Dual‑path isolation preserved, integration tests passing.

**Completion Report**:
- **Date**: 2026‑02‑05
- **Status**: Fully implemented and validated
- **Testing**: All integration tests pass; manual validation confirms text files, images, and PDFs work correctly on LiteLLM path
- **Documentation**: Updated `AGENTS.md` with file‑upload architecture and usage notes
- **Dependencies**: Added `pypdf` optional dependency for PDF text extraction

---

### Step 5 – Implement Tool Calling Across All Providers

**Current problem:** Tool calling is only wired for OpenAI’s Responses API (web search) and uses a provider‑specific `web_search` tool format that is incompatible with LiteLLM. Custom tools cannot be used with Ollama or DeepSeek, and the current LiteLLM path fails for OpenAI agents with web search enabled due to the incompatible tool format.

**Three Architectural Issues Identified** (from REPORT_LITELLM_INTEGRATION.md):

1. **Web‑search detection fails for OpenAI agents** – Models like `gpt‑5.1` (current frontier model) that support web search are incorrectly flagged as unsupported because detection uses outdated hardcoded model names (`gpt‑4o‑search‑preview`) and may call `litellm.supports_web_search()` with wrong model‑string format.
2. **No INFO logging for successful tool binds** – DeepSeek/Ollama agents with tools work silently without user‑facing confirmation; only web‑search detection logs INFO.
3. **LiteLLM doesn't switch between Completions/Responses APIs** – The `use_responses_api` flag is filtered out at `llm_client.py:147`, preventing proper API selection and web‑search configuration.

**Detailed Root Cause Analysis:**

**Overall Architectural Mismatch:** The existing tool configuration (`AgentSpec.tools`) is passed as a list of dictionaries with `type: "web_search"`. LiteLLM expects OpenAI‑compatible tool definitions with `type: "function"` and a JSON Schema `parameters` object. The `web_search` tool type is specific to OpenAI’s Responses API and cannot be translated by LiteLLM. Additionally, the integration relies on static provider‑feature mapping and outdated model references instead of LiteLLM’s dynamic detection methods.

**Issue 1 – Web‑search detection fails for OpenAI agents:**
- **Outdated model list**: Detection only checks `gpt‑4o‑search‑preview` (deprecated), missing current models like `gpt‑5.1`, `gpt‑5‑search‑api`.
- **Wrong model‑string format**: `litellm.supports_web_search()` expects `provider/model` format but detection may pass raw model ID.
- **No dynamic detection**: Hardcoded patterns instead of using `litellm.supports_web_search(model)` and `litellm.get_supported_openai_params(model)`.
- **Code location**: `llm_client.py:1105‑1124`.

**Issue 2 – No INFO logging for successful tool binds:**
- **Tool binding occurs silently**: `ChatLiteLLM.bind_tools()` success not logged at INFO level.
- **Only debug‑level logging**: Engine middleware logs debug messages, not user‑facing confirmation.
- **Impact**: Users cannot see confirmation that tools are successfully bound for DeepSeek/Ollama agents.

**Issue 3 – LiteLLM doesn't switch between Completions/Responses APIs:**
- **Flag filtered out**: `use_responses_api` is removed from kwargs in `_init_chat_model_with_litellm()` (`llm_client.py:147`).
- **No API translation**: LiteLLM has no explicit `use_responses_api` parameter; our flag doesn't map to LiteLLM concepts.
- **Different abstraction**: LiteLLM automatically selects API based on model `mode` in catalog, but our flag should influence selection between `web_search_options` (Completions) and `tools=[{"type": "web_search"}]` (Responses).

**Current OpenAI Model Landscape (2026) – Context for Implementation:**

- **Frontier Models:** `gpt‑5.2` / `gpt‑5.2 pro`, `gpt‑5.1`, `gpt‑5` / `gpt‑5 mini` / `gpt‑5 nano` – all support web search via GA `web_search` tool (Responses API) or `web_search_options` (Completions API).
- **GPT‑4 Series (Legacy, being deprecated):** `gpt‑4.1`, `gpt‑4o` – avoid hardcoding these; prefer current models.
- **Specialized Search Models:** `gpt‑5‑search‑api` (dedicated search model, Completions API), `gpt‑4o‑search‑preview` (deprecated).
- **Web Search Evolution:** Generally Available (GA) `web_search` tool (`{"type": "web_search"}`) replaces the preview version; supports domain filtering (allow‑lists up to 100 domains), user location, and `external_web_access` control.

**Objective:** Create a provider‑agnostic tool‑calling system that works across all providers via LiteLLM while maintaining backward compatibility with the legacy (`USE_LITELLM=false`) path. Fix the regression where OpenAI agents with web search fail on the LiteLLM path, address the three architectural issues, and leverage dynamic feature detection for current and future models.

**LiteLLM API Research Findings (from REPORT_LITELLM_INTEGRATION.md):**

**Completions vs Responses APIs in LiteLLM:**
- **`litellm.completion()` (Completions API)**: Standard chat completions; uses `web_search_options` parameter for web search. This is what `ChatLiteLLM` uses internally.
- **`litellm.responses()` (Responses API)**: For extended reasoning models (o‑series, GPT‑4o with reasoning); uses `tools=[{"type": "web_search"}]` for web search. Automatically selected by LiteLLM based on model `mode` in catalog.
- **Key Insight**: LiteLLM automatically selects the appropriate API based on model metadata. Our `use_responses_api` flag needs translation to LiteLLM parameters.

**Feature Detection Methods:**
- **`litellm.get_supported_openai_params(model)`**: Returns dynamic list of supported parameters per model (includes `web_search_options`, `tools`, `response_format`, etc.). Use to determine API compatibility and validate agent‑spec parameters.
- **`litellm.supports_response_schema(model)`**: Checks if model supports JSON Schema response format.
- **`litellm.supports_web_search(model)`**: Checks if model supports native web search capability. Expects `provider/model` format.
- **`litellm.supports_feature(provider_id, feature, model)`**: (If available) Generic feature detection; fallback to static mapping if method unavailable.

**Integration Guidance:**
- Use `get_supported_openai_params()` to decide between `web_search_options` (Completions) and `tools=[{"type": "web_search"}]` (Responses).
- Combine with `supports_web_search()` for native web‑search capability confirmation.
- Filter out unsupported parameters using `drop_params=True` (already enabled globally) or explicit validation.

**Immediate Actions (from Report):**

1. **Stop filtering `use_responses_api`** – allow the flag to reach the LiteLLM tool adapter.
2. **Create `litellm_tool_adapter.py`** with:
   - `convert_tools_for_litellm(tools, provider_id, model, use_responses_api)`.
   - Dynamic API selection using `litellm.get_supported_openai_params()` and `litellm.supports_web_search()`.
   - Use `{"type": "web_search"}` (GA) for Responses API.
3. **Enhanced logging** – INFO for successful binds, WARNING for dropped tools.

**API‑Switching Logic:**

- **If `use_responses_api=True` and model supports `/responses`** → Use `tools=[{"type": "web_search", "search_context_size": …}]`.
- **If `use_responses_api=False` or model lacks `/responses` support** → Use `web_search_options={…}` (if model supports Completions‑API web search).
- **Fallback** to LiteLLM's automatic bridging (it already routes based on model `mode`).

**Long‑Term Considerations:**

- **Dual‑client approach:** Keep `ChatLiteLLM` for agent orchestration; add optional `LiteLLMResponsesClient` for reasoning‑intensive tasks.
- **Dynamic feature detection:** Replace static `SUPPORTED_FEATURES` mapping with `litellm.get_supported_openai_params()` and `litellm.supports_feature()`.
- **Web‑search detection:** Move to adapter layer; use `supports_web_search()` with proper model string (`provider/model`).
- **Model naming updates:** Update hardcoded references to current models; use `litellm_config.py` for provider‑model mapping.

**Constraints / Clarifications:**

1. **Tools to implement**:
   - **Native web search capability** (`web_search`): A provider‑native capability (OpenAI Responses API, LiteLLM `web_search_options`) that enables model‑side web search. Selectable via UI agent editor, includes per‑run domain filter, restores integrated web search for OpenAI models (still works with `USE_LITELLM=false`), and can enhance other provider models (DeepSeek, Ollama) where supported. Use `litellm.supports_web_search(model="provider/model")` to detect support. **Note:** This is not a LangChain `BaseTool`; the incorrectly implemented `src/multi_agent_dashboard/tool_integration/web_search.py` file should be removed.
   - **Custom DuckDuckGo search alternative** (`web_search_ddg`): A regular LangChain tool (`BaseTool` subclass) that wraps DuckDuckGo search. Selectable via UI agent editor as a separate tool, uses same per‑run domain filter as native web search, serves as an alternative for models/providers without native web search support.

2. **Differentiation**:
   - Native web search (a provider capability) and DuckDuckGo search (a regular tool) are separate options that can be enabled independently.
   - **No automatic fallbacks** from `web_search` to `web_search_ddg` to ensure maximum transparency and reduce complexity.

3. **Log behavior**:
   - Log successful tool application at INFO level (user‑facing messages).
   - Log WARNING when tools are dropped due to lack of provider support.
   - Log ERROR for unexpected tool‑use behavior and handled exceptions.

**Architecture Design – Three‑Layer Separation:**

1. **Tool Registry** (`tool_registry.py`): Central registry for registering LangChain‑compatible tool implementations (e.g., DuckDuckGo search). Uses a decorator pattern for easy extension. Each tool is defined as a LangChain `BaseTool` subclass with a JSON Schema description. **Native capabilities like `web_search` are not registered here**; they are handled by the LiteLLM adapter as provider‑specific configurations.

2. **LiteLLM Tool Adapter** (`litellm_tool_adapter.py`): Converts tool configurations into LiteLLM‑compatible format. Handles provider‑specific transformations:
   - For `web_search`: Uses dynamic detection (`litellm.supports_web_search(model)`, `litellm.get_supported_openai_params(model)`) to decide between `web_search_options` (Completions) and `tools=[{"type": "web_search"}]` (Responses). Logs a warning and excludes if unsupported.
   - For `web_search_ddg`: Converts to a standard function‑calling tool (DuckDuckGo wrapper) with `type: "function"`.
   - Detects general tool‑calling support via `litellm.supports_feature(provider_id, "tool_calling", model)` (fallback to static mapping if method unavailable).
   - Incorporates `use_responses_api` flag translation.

3. **Provider‑Agnostic Execution Layer**: Integrated into `LLMClient.create_agent_for_spec()` for the `USE_LITELLM=true` branch. Uses `ChatLiteLLM.bind_tools()` to automatically bind tools to the chat model, leveraging LiteLLM’s built‑in translation for each provider. Adds INFO‑level logging after successful bind.

**Implementation Sub‑steps:**

**Note:** Implementation must consult recent official LiteLLM documentation for library‑specific details (see Sources section below). Verify API usage, function signatures, and support detection before implementing.

1. ✅ **Create Tool Registry Module** (`src/multi_agent_dashboard/tool_integration/registry.py`):
   - Implement `ToolRegistry` class with `register`, `get_tool`, `list_tools` methods.
   - Support decorator `@register_tool(name, description, schema)` for easy registration.
   - Maintain mapping from tool name to LangChain tool instance and metadata.
   - **Note:** Native web search is not a regular tool but a provider capability.

2. ✅ **Stop filtering `use_responses_api` flag**:
   - Remove the line that strips `use_responses_api` from the kwargs in `_init_chat_model_with_litellm()` (`llm_client.py:147`).
   - Ensure the flag is passed to `convert_tools_for_litellm()`.

3. **Create LiteLLM Tool Adapter** (`src/multi_agent_dashboard/tool_integration/litellm_tool_adapter.py`):
   - Remove the existing web‑search detection logic at `llm_client.py:1105‑1124` (mixing detection and conversion) and replace it with a call to `convert_tools_for_litellm()`.
   - Function `convert_tools_for_litellm(tool_configs, provider_id, model, use_responses_api)`: accepts existing `AgentSpec.tools` configuration.
   - **Dynamic API selection**:
     - Call `litellm.get_supported_openai_params(model)` to see if `"tools"` (Responses API) or `"web_search_options"` (Completions API) is supported.
     - Use `litellm.supports_web_search(model)` to confirm native web‑search capability.
     - Apply API‑switching logic based on `use_responses_api` flag and detected support.
   - For `web_search`: Returns either `web_search_options` dict or `tools` list with `{"type": "web_search"}` (GA). Logs a warning and excludes if unsupported.
   - For `web_search_ddg`: Converts to a standard function‑calling tool with `type: "function"` and JSON Schema parameters.
   - Uses `litellm.supports_feature(provider_id, "tool_calling", model)` (or fallback) to detect general tool‑calling support; if unsupported, returns empty list (LiteLLM will fall back to description‑based prompting).
   - Caches configurations per provider‑model‑flag combination.

4. **Implement DuckDuckGo Search Tool** (`src/multi_agent_dashboard/tool_integration/duckduckgo_search.py`):
   - Create a LangChain `DuckDuckGoSearchRun` wrapper with optional `domain_filter` parameter.
   - Register as `web_search_ddg` tool with JSON Schema describing `query` and `domain_filter` parameters.

5. **Enhance `LLMClient.create_agent_for_spec()` to use the adapter**:
   - Pass transformed tools to `ChatLiteLLM.bind_tools()` (already using `ChatLiteLLM` via `_init_chat_model_with_litellm`).
   - **Add INFO logging** after successful bind: `logger.info(f"Successfully bound tools: {tool_names}")`.
   - Keep existing LangChain agent tool‑execution middleware unchanged (re‑uses `_extract_tool_info_from_messages` and `_collect_content_blocks`).

6. **Maintain Legacy Path Unchanged**:
   - For `USE_LITELLM=false`, the existing `_build_tools_config()` in `models.py` continues to produce OpenAI Responses API‑style `web_search` tool definitions.
   - No modifications to the legacy tool‑calling logic (OpenAI Responses API integration).

7. **Update UI Integration**:
   - The UI (`tools_view.py`) already displays tool usage from `content_blocks`; no changes required.
   - Agent editor’s tool configuration must be extended to support both `web_search` and `web_search_ddg` as separate selectable tool types.

**Verification:**
- **Regression Fix**: OpenAI agents with web search work on the LiteLLM path (using native web search where supported; DuckDuckGo search is available as a separate tool for unsupported models).
- **Three Issues Resolved**:
  1. Web‑search detection works for current OpenAI models (e.g., `gpt‑5.1`) via dynamic detection.
  2. INFO logging shows confirmation of successful tool binding for DeepSeek/Ollama agents.
  3. `use_responses_api` flag influences API selection (observable in logs).
- **Provider‑Agnostic**: Both native web search and DuckDuckGo search tools work with all three providers (OpenAI, DeepSeek, Ollama) via LiteLLM translation where supported.
- **Backward Compatibility**: Existing agent specs with `web_search` continue to work in legacy mode (`USE_LITELLM=false`).
- **Domain Filtering**: Optional `domain_filter` parameter preserved as a tool parameter for both search tools.
- **Graceful Fallback**: Providers without native tool calling receive a description‑based prompt (LiteLLM automatic fallback).
- **Transparent Logging**: Successful tool application logged at INFO, warnings logged when tools are dropped, errors logged for unexpected behavior.
- **Extensibility**: New tools can be added by registering a LangChain tool in the registry; no provider‑specific code required.

**Success Metrics:**
- No regression in legacy tool‑calling (`USE_LITELLM=false`)
- OpenAI web‑search regression fixed in LiteLLM path (native web search works for supported models)
- DuckDuckGo search returns comparable results to native web search
- Tool execution results appear in `content_blocks` and UI as before
- Domain filtering still functional (as tool parameter)
- Clear logging of tool support status (INFO/WARNING/ERROR)
- **Three architectural issues resolved** (web‑search detection, INFO logging, API switching)

**Risks & Mitigations:**
- **DuckDuckGo API rate limits** – Implement caching and rate‑limiting in the tool wrapper.
- **LiteLLM tool‑calling bugs for specific providers** – Keep legacy path as fallback; test extensively before enabling `USE_LITELLM=true` by default.
- **Native web search unsupported for some models** – Log clear warning and exclude the tool (no automatic fallback to DuckDuckGo).
- **Performance impact of additional abstraction** – Benchmark tool‑calling latency; use caching for tool definitions.
- **Dynamic detection may fail for new/unlisted models** – Fall back to static mapping with warning; encourage contribution to LiteLLM model catalog.

**Sources / Examples:**

**LiteLLM Official Documentation (Critical for Implementation):**
- [LiteLLM - Web Search](https://docs.litellm.ai/docs/completion/web_search)
- [LiteLLM - Function Calling / Checking if a model supports function calling](https://docs.litellm.ai/docs/completion/function_call)
- [LiteLLM - Tool Calling and Function Integration](https://deepwiki.com/BerriAI/litellm/8.1-tool-calling-and-function-integration)
- [LiteLLM - `get_supported_openai_params()`](https://docs.litellm.ai/docs/completion/parameters#get_supported_openai_params) (verify exact URL)
- [LiteLLM - `supports_web_search()`](https://docs.litellm.ai/docs/completion/web_search#supports_web_search) (verify exact URL)
- [LiteLLM - `supports_feature()`](https://docs.litellm.ai/docs/completion/features#supports_feature) (verify exact URL)
- [LiteLLM supports all models from Ollama](https://docs.litellm.ai/docs/providers/ollama)

**OpenAI Model Updates (2026):**
- [OpenAI API Documentation – Models](https://platform.openai.com/docs/models) (check for current model list and web‑search support)
- [OpenAI API Changelog](https://platform.openai.com/docs/changelog) (monitor deprecations and new features)

**Example Code & Community Resources:**
- [Tool Calling Example with LiteLLM](https://gist.github.com/RahulDas-dev/3cbfc73b89cc5f33c295e8e03d2a3360/raw/baf715ca831bef813bc847154e369f860c542355/litellm_tool_calling.py)
- [OllamaException - 405 method not allowed](https://github.com/agent0ai/agent-zero/issues/819)
- [Ollama - Tool calling](https://docs.ollama.com/capabilities/tool-calling)

**Implementation Guidance:**
- Always verify the latest LiteLLM version and method signatures before implementing.
- Use `litellm.drop_params = True` globally to avoid errors from unsupported parameters.
- Test with current frontier models (e.g., `gpt‑5.1` or latest available) to ensure compatibility.

---

### Step 6 – Standardize Temperature and Model Parameters

**Current problem:** Temperature, max_tokens, top_p, etc., are passed via provider‑specific kwargs. Some models have specific requirements (e.g., GPT‑5 requires temperature=1.0) that cause errors when unsupported values are passed.

**Sub‑steps:**

1. ✅ Consolidate all model‑parameter handling in `LiteLLMClient` using LiteLLM’s unified parameter set (`temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `seed`).  
   *Implemented for `USE_LITELLM=true` path.*

2. ✅ Map agent‑spec fields (`temperature`, `max_output_tokens`, `stop_sequences`) to these parameters.  
   *Mapping implemented; unsupported values are logged via `drop_params`.*

3. ✅ **Fix: Use LiteLLM’s `drop_params` feature for graceful parameter handling**:
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

3. ⚠️ **Use LiteLLM’s dynamic feature detection**: Query `litellm.get_supported_openai_params(model)` to detect supported parameters per model, `litellm.supports_response_schema(model)` for JSON Schema support, `litellm.supports_web_search(model)` for native web‑search capability, and `litellm.supports_feature(provider_id, feature, model)` for generic feature detection (if available).  
   *Replace hardcoded provider checks with runtime detection. Use `provider/model` format for model strings.*

4. ⚠️ **Leverage `langchain‑litellm` PR #62 fixes**: Ensure `tool_choice="any"` is mapped to `"required"` for unsupported providers, preventing crashes.

**Verification:**
- Attempting to use tool calling with a model that doesn’t support it yields a clear user‑friendly message **or falls back with a warning logged**
- The system does not crash when an unsupported parameter is passed (parameters are dropped with warning)
- Provider‑specific fallbacks continue to work in legacy mode (`USE_LITELLM=false`)

---

### Step 9 – Update UI and Configuration

**Sub‑steps:**

1. Update provider dropdowns to reflect that LiteLLM is the underlying engine (keep same provider IDs for backward compatibility).

2. Add a hidden advanced setting `USE_LITELLM` (default `False`) that can be turned off to revert to the old provider‑specific adapters (safety switch).

3. Extend the provider‑features detection in `ui/agent_editor_mode.py` to query LiteLLM for supported capabilities using `litellm.get_supported_openai_params(model)` (streaming, tools, vision, JSON mode) and `litellm.supports_feature(provider_id, feature, model)` (if available). Grey out unsupported options in the UI.

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
- Set `USE_LITELLM` flag default to `False` (already set).

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
- Switch `USE_LITELLM` default to `True` after validation passes.

**Phase 5 (Post‑validation)** – Remove old adapters and provider‑specific code.  
- Delete provider‑specific usage extraction helpers (`_extract_usage_from_*`).
- Remove provider‑specific parameter translation from `ChatModelFactory.get_model`.
- Remove provider‑specific fallback loops (DeepSeek `methods` loop, etc.).
- Final code reduction goal: ~300 lines removed.

Each phase should be merged separately, with the `USE_LITELLM` flag defaulting to `False` until Phase 4. Removal of old adapters (Phase 5) is **optional** and can be deferred if continued dual‑path operation is preferred.

## Success Metrics

- **Code reduction (Phase 5):** Remove at least 300 lines of provider‑specific branching **after** LiteLLM path is validated; dual‑path operation may keep some provider‑specific code longer.
- **Feature parity:** All existing features work identically across providers in both `USE_LITELLM` modes.
- **Extensibility:** Adding a new provider requires only adding its model string to the LiteLLM mapping (once LiteLLM path is default).
- **Performance:** No significant increase in latency or token‑counting errors compared to native provider implementations.
- **User experience:** No visible change for existing users except improved consistency; advanced `USE_LITELLM` toggle allows fallback to legacy adapters if needed.

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
- [LiteLLM - Web Search](https://docs.litellm.ai/docs/completion/web_search)
- [LiteLLM - Function Calling / Checking if a model supports function calling](https://docs.litellm.ai/docs/completion/function_call)
- [LiteLLM - Tool Calling and Function Integration](https://deepwiki.com/BerriAI/litellm/8.1-tool-calling-and-function-integration)
- [LiteLLM - `get_supported_openai_params()`](https://docs.litellm.ai/docs/completion/parameters#get_supported_openai_params)
- [LiteLLM - `supports_web_search()`](https://docs.litellm.ai/docs/completion/web_search#supports_web_search)
- [LangChain‑LiteLLM Integration](https://python.langchain.com/docs/integrations/chat/litellm)
- [LiteLLM Supported Providers](https://docs.litellm.ai/docs/providers)
- `langchain‑litellm` PR #62 (Jan 30, 2026) – JSON Schema support, `tool_choice="any"` mapping, proper `tool_calls` population
- Existing provider‑specific code in `src/multi_agent_dashboard/llm_client.py`, `engine.py`, `config.py`

---

*Plan generated on 2026‑02‑05.*
*Target completion: 2026‑02‑15.*