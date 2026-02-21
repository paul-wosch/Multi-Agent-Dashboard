# Strategy: Dropping LiteLLM & Migrating Enhancements to a Clean LangChain‑Only Implementation

**Goal:** Completely remove the LiteLLM integration, lock in support for **OpenAI, DeepSeek, and Ollama**, and merge all its enhancements (structured output, tool calling, web search, user‑controlled endpoints, multimodal file handling) back into a unified, provider‑specific LangChain‑only code path. The result will be a simpler, more predictable codebase with full control over the three supported providers.

---

## 1. Executive Summary

The current dual‑path architecture (`USE_LITELLM=true/false`) introduces complexity and opaque detection logic. LiteLLM’s hardcoded model‑specification files cause unreliable capability detection, and its abstraction layer obscures provider‑specific behavior. By removing LiteLLM entirely, we can:

- **Eliminate** ~1,500 lines of LiteLLM‑specific code (configuration, adapter, client wrapper).
- **Retain** all current features (structured output, tool calling, web search, endpoint control, multimodal file handling) by migrating them to provider‑specific LangChain implementations.
- **Improve predictability** – capabilities are determined by explicit provider‑feature mapping and runtime detection based on LangChain’s `init_chat_model` and provider‑specific APIs.
- **Keep the UI unchanged** – the same checkboxes and toggles (`provider_features`, `tools`, `structured_output_enabled`) continue to work.
- **Reduce dependencies** – drop optional `langchain‑litellm` and rely only on `langchain‑openai`, `langchain‑deepseek`, `langchain‑ollama`.

**Scope:** The migration is a **refactor, not a rewrite**. The existing non‑LiteLLM path (`USE_LITELLM=false`) already provides structured output across all three providers, passes the use_responses_api flag for model initialization but does not handle tool conversion between Responses/Completions APIs, and supports basic file attachments (text files for OpenAI; vision currently broken). It lacks generic tool‑handling and robust multimodal file‑processing logic, which currently reside in the LiteLLM path. We will migrate those missing capabilities to create a unified provider‑specific implementation.

### 1.1 Design Principles

To enhance maintainability, readability, and separation of concerns in the refactored code, prefer a **highly modularized approach** over nested branches (helpers, modules, packages). This means:

- **Split existing code** affected by the refactor into separate modules/functions where appropriate
- **Create dedicated helper functions** for provider‑specific logic (e.g., `_build_openai_tools`, `_build_deepseek_tools`)
- **Avoid deep nesting** of provider‑specific branches within large functions
- **Keep modules focused** on a single responsibility (e.g., tool adaptation, file handling, capability mapping)
- **Preserve existing interfaces** while refactoring internals

This modular design will make the codebase easier to test, extend, and maintain.

---

## 2. Component Inventory – What to Remove

| File / Component                                                                                                | Purpose                                                                  | Removal Action                                                               |
| --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| `src/multi_agent_dashboard/litellm_config.py`                                                                   | Model‑string conversion, static capability mapping, `supports_feature()` | Delete entire file.                                                          |
| `src/multi_agent_dashboard/tool_integration/litellm_tool_adapter.py`                                            | Converts tool configs to LiteLLM‑compatible `tools`/`web_search_options` | Delete entire file.                                                          |
| `src/multi_agent_dashboard/llm_client.py` – `_init_chat_model_with_litellm` (line 79)                           | Factory for `ChatLiteLLM` instances                                      | Remove function.                                                             |
| `src/multi_agent_dashboard/llm_client.py` – `LiteLLMClient` class (line 515)                                    | Wrapper around `litellm.completion()`                                    | Remove class.                                                                |
| `src/multi_agent_dashboard/llm_client.py` – `_build_structured_output_adapter` – LiteLLM branch (lines 879‑996) | Workaround for LiteLLM JSON‑Schema bug                                   | Delete branch, keep provider‑specific branches.                              |
| `src/multi_agent_dashboard/llm_client.py` – `create_agent_for_spec` – LiteLLM tool conversion (lines 1140‑1166) | Calls `convert_tools_for_litellm` and binds tools                        | Replace with new provider‑specific tool‑binding logic.                       |
| `src/multi_agent_dashboard/llm_client.py` – `USE_LITELLM` flag references                                       | Guard condition throughout client                                        | Remove all `self._use_litellm` checks, keep only the LangChain‑only logic.   |
| `src/multi_agent_dashboard/config.py` – `USE_LITELLM` environment variable                                      | Global switch                                                            | Remove variable; default to `False` (or remove flag entirely).               |
| `src/multi_agent_dashboard/multimodal_handler.py` – LiteLLM‑specific file‑processing branches                   | File encoding for vision‑capable providers via LiteLLM                   | Keep the module but adapt to provider‑specific LangChain file‑handling APIs. |
| `tests/test_litellm_client.py`                                                                                  | Tests for LiteLLM integration                                            | Delete or convert to provider‑specific tests.                                |
| `requirements.txt` / `pyproject.toml` – `langchain‑litellm` dependency                                          | Optional dependency                                                      | Remove.                                                                      |

**Note:** The `ChatModelFactory` and `_init_chat_model` (LangChain’s built‑in initializer) remain unchanged; they already support the three providers via LangChain’s provider detection.

**Important non‑LiteLLM implementations to preserve:**

- `AgentRuntime.run` lines 300‑319: OpenAI‑specific `web_search` binding (already works in the non‑LiteLLM path). This code must be **extended** to DeepSeek and Ollama using the new capability mapping, not replaced.
- `llm_client.py` function `_extract_tool_info_from_messages`: Tool‑call and content‑block extraction (already works). Keep this function.
- `AgentRuntime._build_tools_config()` and `_build_reasoning_config()` methods for OpenAI‑specific tool and reasoning configuration (already works). Extend to other providers.
- `LLMClient._build_input_with_files()` method for OpenAI‑specific file handling with image support (already works). Extend to other providers.
- `_build_structured_output_adapter` provider‑specific branches (OpenAI, DeepSeek, Ollama) already work; only the LiteLLM branch should be deleted.
- Token‑usage extraction functions restored in commits 053ed39 and b7f119b must remain intact.

---

## Execution Guidelines

**Critical: The executing agent MUST follow these guidelines:**

1. **Case‑by‑case evaluation** – Do **not** rely solely on the suggestions in this document. For each code change, examine the actual old and new code paths, understand their behavior, and verify the proposed changes are correct for the specific context.

2. **Stop for ambiguity** – If any requirement, behavior, or code path is ambiguous or unclear, **stop work immediately** and wait for user approval/decision. Do not guess or make assumptions about unclear requirements.

3. **Fetch latest documentation** – Before implementing provider‑specific logic, fetch the latest online documentation for:
   
   - LangChain provider libraries (`langchain‑openai`, `langchain‑deepseek`, `langchain‑ollama`)
   - Provider API specifications (OpenAI Responses/Completions API, DeepSeek API, Ollama API)
   - Model‑specific capabilities (vision support, tool‑calling, structured‑output formats)
   
   Use the `agentic_fetch` tool to search for and analyze current documentation. Verify parameter names, supported features, and any recent changes that may affect the implementation.

4. **Test before deletion** – Always test the new implementation thoroughly before removing any LiteLLM code. Maintain the dual‑path architecture (`USE_LITELLM=false`) until the new implementation is validated.

5. **Preserve working code** – When extending existing implementations to new providers, preserve the working logic for the original provider. Add new branches or adapters rather than rewriting.

6. **Prefer modularization over nested branches** – Split code into separate helper functions, modules, and packages. Avoid deep nesting of provider‑specific branches; instead create focused modules (e.g., `tool_integration/provider_tool_adapter.py`, `multimodal_handler.py`). Move existing code affected by the refactor into dedicated modules where appropriate.

---

## 3. Feature Migration Plan

## 3. Feature Migration Plan

### 3.1 Structured Output

| Aspect                | LiteLLM Path                                                                             | Non‑LiteLLM Path (Current)                                                                                                                                  | New Unified Path                                                                                    |
| --------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Entry point**       | `_build_structured_output_adapter` (returns `{"type": "json_schema", "json_schema": …}`) | Same function, but provider‑specific branches (OpenAI JSON Schema, DeepSeek function‑calling, Ollama raw schema)                                            | Keep provider‑specific branches; delete LiteLLM branch.                                             |
| **Provider handling** | LiteLLM normalizes to `response_format` parameter.                                       | OpenAI: `{"type": "json_schema", "json_schema": …}`<br>DeepSeek: `method="function_calling"` or `"json_mode"`<br>Ollama: raw schema, `method="json_schema"` | No change – the existing provider‑specific logic already works.                                     |
| **Workarounds**       | JSON‑Schema bug workaround (lines 879‑996)                                               | None                                                                                                                                                        | Delete bug workaround; rely on LangChain’s `with_structured_output`.                                |
| **Integration**       | Passed as `response_format` to `litellm.completion`.                                     | Passed as `response_format` to `create_agent_for_spec` (OpenAI) or via `with_structured_output` (DeepSeek, Ollama).                                         | Keep existing integration; ensure `response_format` is correctly passed to `create_agent_for_spec`. |

**Action:** Remove the LiteLLM branch in `_build_structured_output_adapter` and keep the provider‑specific branches. Ensure `response_format` is correctly passed through `create_agent_for_spec` (already done).

### 3.2 Tool Calling & Web Search

| Aspect                   | LiteLLM Path                                                                                                                                                                           | Non‑LiteLLM Path (Current)                                                                                                                                                                                                                                                       | New Unified Path                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Configuration format** | `AgentSpec.tools = {"enabled": True, "tools": ["web_search", "web_search_ddg"]}`                                                                                                       | Same format, but only a simple `web_search` dict is created for OpenAI in `AgentRuntime.run`.                                                                                                                                                                                    | Keep the same configuration format; build a **new provider‑specific tool adapter** that replaces `litellm_tool_adapter`.                                                                                                                                                                                                                                                                                                |
| **Conversion logic**     | `convert_tools_for_litellm` uses `litellm.get_supported_openai_params()` and static mapping to decide between `tools` list (Responses API) and `web_search_options` (Completions API). | OpenAI‑only `web_search` conversion via `AgentRuntime._build_tools_config()` and `_build_reasoning_config()` (called from `AgentRuntime.run` lines 300‑319); no generic conversion for other providers/tools. Tool‑call extraction via `_extract_tool_info_from_messages` works. | Implement a new `convert_tools_for_provider` that:<br>1. Uses provider‑specific capabilities (see Capability Detection).<br>2. For `web_search`: binds `web_search_options` for OpenAI Completions API when `use_responses_api=False`; otherwise uses `{"type": "web_search"}` tool.<br>3. For `web_search_ddg`: creates a DuckDuckGo function‑calling tool (JSON Schema) for providers that support `tools` parameter. |
| **Binding**              | `model_instance.bind_tools(tools_list)` or `model_instance.bind(web_search_options=…)`.                                                                                                | `model_instance.bind_tools` (if tools list) or `model_instance.bind(web_search_options=…)` (if web‑search dict).                                                                                                                                                                 | Same binding API (LangChain’s `ChatOpenAI` methods).                                                                                                                                                                                                                                                                                                                                                                    |
| **Capability detection** | Dynamic via LiteLLM + static fallback.                                                                                                                                                 | Limited detection via `provider_features` (UI checkboxes). No automatic detection of tool‑calling capabilities.                                                                                                                                                                  | Implement a simple rule‑based detector (see section 5).                                                                                                                                                                                                                                                                                                                                                                 |

**Action:** Create `tool_integration/provider_tool_adapter.py` with `convert_tools_for_provider`. Replace the LiteLLM tool‑conversion call in `create_agent_for_spec` with this new adapter. Update `AgentRuntime.run` to call the adapter and produce the proper tool list for all three providers.

### 3.3 Multimodal File Handling

| Aspect                   | LiteLLM Path                                                                      | Non‑LiteLLM Path (Current)                                                                                                                                                    | New Unified Path                                                                                                                                                                                                   |
| ------------------------ | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Entry point**          | `llm_client.py` references `multimodal_handler.py` for LiteLLM path.              | OpenAI‑specific file handling via `LLMClient._build_input_with_files()` (called from `llm_client.py`); images encoded as data URLs, binary files uploaded or base64‑embedded. | Keep `multimodal_handler.py` but adapt to provider‑specific LangChain message‑building (e.g., `HumanMessage` with `content` list of text/images).                                                                  |
| **Processing**           | Base64‑encodes images for vision‑capable providers, extracts text from PDFs, etc. | Sophisticated file handling: detects MIME types, inlines text, uploads binaries, encodes images as data URLs, embeds base64 for small files (OpenAI‑specific format).         | Reuse the same processing logic, but instead of passing to LiteLLM, construct LangChain message parts according to provider capabilities (OpenAI: `image_url`; Ollama: base64‑encoded image; DeepSeek: text‑only). |
| **Capability detection** | Uses `litellm.supports_feature(…, "image_inputs")`.                               | None.                                                                                                                                                                         | Use the same rule‑based detector (section 5) to know if a provider supports images.                                                                                                                                |

**Action:** Extend `multimodal_handler.py` to support provider‑specific message construction. Integrate with the capability detector to decide whether to encode images or fall back to text descriptions.

### 3.4 Endpoint & Configuration Control

| Aspect                     | LiteLLM Path                                              | Non‑LiteLLM Path (Current)                                                               | New Unified Path                                                     |
| -------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Endpoint override**      | `endpoint` passed to `litellm.completion` via `api_base`. | `endpoint` mapped to `base_url` in `ChatModelFactory`.                                   | Keep existing mapping (`base_url`).                                  |
| **API‑version preference** | `use_responses_api` flag passed to tool adapter.          | `use_responses_api` passed to `ChatModelFactory` (sets `output_version="responses/v1"`). | Keep existing handling; ensure tool adapter respects the flag.       |
| **Provider‑feature hints** | `provider_features` passed to LiteLLM as `profile`.       | `provider_features` passed to LangChain as `profile` (some providers ignore).            | Keep passing `profile`; use it for capability detection (section 5). |

**Action:** No changes needed; the existing LangChain path already supports these overrides.

---

## 4. Provider‑Specific Implementation Guide

### OpenAI

- **Model initialization:** `init_chat_model` with `model="openai/..."` (or `provider_id="openai"`).
- **Structured output:** `{"type": "json_schema", "json_schema": …}` passed as `response_format`. Use `with_structured_output(method="json_schema")`.
- **Tools:**  
  - `web_search` → if `use_responses_api=True`: `{"type": "web_search"}` tool; else `web_search_options` dict bound via `bind(web_search_options=…)`.  
  - `web_search_ddg` → function‑calling tool (JSON Schema) bound via `bind_tools`.  
- **File handling:** Vision‑capable models (gpt‑4o) accept `image_url` in message content. Use `HumanMessage(content=[…])` with image parts.

### DeepSeek

- **Model initialization:** `init_chat_model` with `model="deepseek/..."` (or `provider_id="deepseek"`).
- **Structured output:** Prefer `method="function_calling"`; for reasoner models use `method="json_mode"`. Use `with_structured_output`.
- **Tools:** Function‑calling tools (`tools` parameter) are supported. Native `web_search` **may be supported** (manual agent runs via LiteLLM path have worked, though possibly relying on cached results internally with mixed outcomes). If enabled, the system will attempt to bind it as a function‑calling tool; runtime behavior needs validation. `web_search_ddg` creates a DuckDuckGo function‑calling tool (JSON Schema) bound via `bind_tools`.
- **File handling:** Text‑only; encode files as text snippets.

### Ollama

- **Model initialization:** `init_chat_model` with `model="ollama/..."` (or `provider_id="ollama"`), `base_url` for local endpoint.
- **Structured output:** `method="json_schema"` (raw schema). Use `with_structured_output`.
- **Tools:** Some Ollama models support tool/function calling (needs further research during implementation). If `web_search` or `web_search_ddg` are enabled, the system will attempt to bind them; runtime behavior needs validation.
- **File handling:** Vision‑capable models (e.g., llava) support images via base64 encoding. Use dynamic capability detection (based on provider API, not hardcoded model names) to decide.

---

## 5. Capability Detection Philosophy

Agent configuration (via `provider_features` UI checkboxes and `tools` settings) is the primary source of truth. The system should respect user choices and attempt to bind requested features even if they may not be supported by the provider, allowing runtime errors to surface naturally.

A minimal static capability mapping may be maintained for advisory purposes:

- Warn users when enabled features are unlikely to be supported by the provider
- Guide UI defaults (pre‑checking capability checkboxes)
- Provide fallback behavior for file handling (e.g., text‑only concatenation when vision unsupported)

However, static mapping **must not**:

- Automatically convert between tool types (e.g., `web_search` to `web_search_ddg`)
- Disable features that the user has explicitly enabled
- Override `provider_features` settings
- Rely on heuristics or hardcoded model names for runtime capability detection
- Implement prompt‑based tool calling emulation (tool invocation remains the user’s responsibility via manual agent prompt configuration)

**Implementation approach:**

1. **Primary source:** Use `provider_features` from agent configuration
2. **Advisory mapping:** Optional `provider_capabilities.py` with best‑knowledge capabilities for known models (purely advisory, not for runtime decisions)
3. **Runtime validation:** Attempt to bind requested tools; log warnings if capabilities mapping suggests they will fail
4. **File handling:** Use provider‑specific LangChain APIs to determine supported features dynamically. If capability detection is needed, it must be based on actual API‑supported methods, not heuristics or hardcoded model names.

**Integration points:**

- In `ChatModelFactory`, pass `provider_features` as `profile`
- In `engine._extract_provider_features_from_profile`, derive UI‑friendly hints from the profile (already works)
- In the new tool adapter, respect the exact tool configuration; use capability mapping only for logging warnings

---

## 6. Action Plan (Phased Implementation)

1. **Phase 1 – Create advisory capability mapping & tool adapter**  
   
   - Write `provider_capabilities.py` with static advisory maps for OpenAI, DeepSeek, Ollama (used only for warnings and UI defaults).  
   - Write `tool_integration/provider_tool_adapter.py` that respects exact tool configuration from `AgentSpec.tools` and uses capability mapping only for logging warnings (no automatic conversions).  
   - Update `AgentRuntime.run` to use the new adapter (still behind `USE_LITELLM=false`).  
   - Test with each provider (smoke tests).

2. **Phase 2 – Remove LiteLLM tool‑conversion & structured‑output branches**  
   
   - Delete `litellm_tool_adapter.py`.  
   - Replace the LiteLLM tool‑conversion call in `create_agent_for_spec` with the new adapter.  
   - Remove the LiteLLM branch in `_build_structured_output_adapter`.  
   - Ensure `response_format` flows correctly for all providers.

3. **Phase 3 – Update multimodal file handling**  
   
   - Adapt `multimodal_handler.py` to use provider‑specific message building.  
   - Integrate with dynamic capability detection for image support (using provider APIs, not hardcoded model names).  
   - Update `llm_client.py` file‑processing logic to use the updated handler.

4. **Phase 4 – Delete remaining LiteLLM components**  
   
   - Remove `litellm_config.py`.  
   - Remove `LiteLLMClient` class and `_init_chat_model_with_litellm`.  
   - Remove all `self._use_litellm` conditionals, defaulting to LangChain‑only path.  
   - Remove `USE_LITELLM` environment variable from config.

5. **Phase 5 – Update tests & documentation**  
   
   - Delete LiteLLM‑specific tests.  
   - Add unit tests for the new capability mapping and tool adapter.  
   - Update `AGENTS.md` and `PLAN.md` to reflect the new architecture.

**Note:** Each phase must keep the existing non‑LiteLLM path (`USE_LITELLM=false`) fully functional. After Phase 4, the flag can be removed entirely.

---

## 7. Validation Checklist

- [ ] **OpenAI**  
  
  - [ ] Basic chat works.  
  - [ ] Structured output (JSON Schema) works.  
  - [ ] Web search via Responses API (`use_responses_api=true`) works.  
  - [ ] Web search via Completions API (`use_responses_api=false`) works.  
  - [ ] DuckDuckGo function‑calling tool works.  
  - [ ] Image file upload works (vision‑capable models).  

- [ ] **DeepSeek**  
  
  - [ ] Basic chat works.  
  - [ ] Structured output (`function_calling`/`json_mode`) works.  
  - [ ] DuckDuckGo function‑calling tool works (no native web search).  
  - [ ] File upload (text‑only) works.  

- [ ] **Ollama**  
  
  - [ ] Basic chat works (local endpoint).  
  - [ ] Structured output (`json_schema`) works.  
  - [ ] Tool‑calling (some models may support – attempts binding, logs warning if unsupported).  
  - [ ] Image file upload works for vision‑capable models (using dynamic capability detection).  

- [ ] **UI**  
  
  - [ ] Provider‑feature checkboxes still reflect capabilities.  
  - [ ] Tool checkboxes enable/disable correctly.  
  - [ ] Structured‑output toggle works.  
  - [ ] Endpoint override works.  

- [ ] **Database & Engine**  
  
  - [ ] `provider_features` are stored in agent snapshots.  
  - [ ] Cost/latency instrumentation remains accurate.  
  - [ ] Strict‑schema validation flags work as before.  

---

## 8. Risks & Mitigations

| Risk                                                      | Mitigation                                                                                                   |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Provider‑specific bugs** in LangChain integrations.     | Keep the existing provider‑specific branches (already tested). Add more unit tests for edge cases.           |
| **Capability mapping** becomes outdated as models evolve. | Treat mapping as advisory only; rely on user configuration (`provider_features`) as primary source of truth. |
| **Regression in tool‑handling** for existing agents.      | Thoroughly test each provider with the new adapter before deleting LiteLLM path.                             |
| **File‑handling breakage** for multimodal use cases.      | Keep fallback to text‑only concatenation when provider lacks image support.                                  |
| **Performance impact** from removing LiteLLM caching.     | LangChain’s `ChatModelFactory` already caches model instances; no loss.                                      |

---

**Final Outcome:** A cleaner, more maintainable codebase with **zero LiteLLM dependencies**, full feature parity for the three supported providers, and a straightforward path to add new providers (by extending the advisory capability map and provider‑specific tool bindings).