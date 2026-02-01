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

## Goal

Replace provider‑specific adapters with **LiteLLM** as a universal translation layer, leveraging its:
- **Unified OpenAI‑compatible interface** for 100+ LLM APIs
- **Consistent usage/token tracking** across all providers
- **Built‑in structured output** (JSON mode, function calling)
- **Multimodal support** (file uploads, images, audio)
- **Tool/function calling** with automatic fallback strategies
- **Cost tracking** and **retry/fallback** logic
- **Streaming** and **reasoning** configuration

Integration will use the `langchain‑litellm` package to keep the existing LangChain agent architecture while removing provider‑specific code.

## Prerequisites

- Install `litellm>=1.81.6` and `langchain‑litellm>=0.3.5` (add to `pyproject.toml`); consider bumping `langchain` from 1.2.4 to 1.2.7 for latest fixes
- Keep existing provider‑specific packages for fallback or direct access where needed
- Ensure environment variables for each provider are still read (OPENAI_API_KEY, DEEPSEEK_API_KEY, OLLAMA_HOST)
- No breaking changes to the UI, database schema, or agent specs

## Implementation Steps

### Step 1 – Install and Configure LiteLLM

**Sub‑steps:**

1. Add dependencies to `pyproject.toml`:
   ```toml
   dependencies = [
       "litellm>=1.81.6",
       "langchain-litellm>=0.3.5",
       # keep existing langchain‑openai, langchain‑ollama, langchain‑deepseek for now
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
- `pip install -e .` succeeds
- Import `litellm` and `langchain_litellm` without errors
- Unit test that `LiteLLMClient` can be instantiated with each provider

---

### Step 2 – Replace Provider‑Specific Token Counting

**Current problem:** `llm_client.py` and `engine.py` contain separate logic to extract `usage` from OpenAI, `prompt_eval_count`/`completion_eval_count` from Ollama, and `usage_metadata` from DeepSeek.

**Sub‑steps:**

1. In `LiteLLMClient.invoke`, call `litellm.completion(…, extra_body={"metadata": true})` to guarantee a unified `usage` object in the response (LiteLLM normalizes this across providers).

2. Remove all provider‑specific usage‑extraction helpers (`_extract_usage_from_candidate`, `_extract_usage_from_ollama`, `_extract_usage_from_deepseek`).

3. Update `engine.py`’s `_compute_cost` to read the normalized `usage` object and map provider‑ID to LiteLLM’s pricing dictionary (LiteLLM provides per‑model pricing; we can keep our own `*_PRICING` tables for consistency).

4. Ensure token counts are stored in the database exactly as before (no schema change).

**Verification:**
- Existing cost‑tracking tests pass
- Manual test with each provider shows correct token counts in the UI
- No regression in pricing calculations

---

### Step 3 – Implement Unified Structured Output Handling

**Current problem:** `_build_structured_output_adapter` branches per provider (OpenAI `json_schema`, Ollama `json_schema`, DeepSeek `function_calling`/`json_mode`).

**Sub‑steps:**

1. Replace `_build_structured_output_adapter` with a single code path that calls LiteLLM’s `response_format` parameter (supports `json_object`, `json_schema`, `function_calling`).

2. Use LiteLLM’s `strict=True` flag to enforce schema validation where the provider supports it; for unsupported providers, LiteLLM will fall back to prompt‑based JSON enforcement.

3. Remove provider‑specific `with_structured_output` wrapping in `create_agent_for_spec`.

4. Update `structured_schemas.py` to produce JSON‑Schema dictionaries compatible with LiteLLM’s `response_format`.

**Verification:**
- Structured output works for all three providers with the same UI toggle
- JSON validation errors are reported consistently
- Token counts include the structured‑output overhead

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

**Verification:**
- Web search tool works with all three providers
- Custom tools (e.g., calculator, database query) can be added and invoked regardless of provider
- Tool‑call results are correctly injected into the conversation

---

### Step 6 – Standardize Temperature and Model Parameters

**Current problem:** Temperature, max_tokens, top_p, etc., are passed via provider‑specific kwargs.

**Sub‑steps:**

1. Consolidate all model‑parameter handling in `LiteLLMClient` using LiteLLM’s unified parameter set (`temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `seed`).

2. Map agent‑spec fields (`temperature`, `max_output_tokens`, `stop_sequences`) to these parameters.

3. Remove provider‑specific parameter translation from `ChatModelFactory.get_model`.

**Verification:**
- Temperature slider in the UI affects all providers equally
- Max output tokens limit is respected
- Random seed produces deterministic outputs where supported

---

### Step 7 – Add Streaming Support

**Current problem:** Streaming is not implemented; responses are returned only after completion.

**Sub‑steps:**

1. Extend `LiteLLMClient.invoke` with a `stream=True` parameter that yields chunks via `litellm.completion(stream=True)`.

2. Propagate streaming chunks through the existing LangChain agent middleware (if needed) or directly to the UI.

3. Update the UI’s `run_mode.py` to display streaming tokens as they arrive (optional future enhancement).

**Verification:**
- Streaming can be enabled via a UI toggle (later phase)
- Each provider delivers token streams where supported (LiteLLM handles translation)

---

### Step 8 – Graceful Handling of Unsupported Features

**Current problem:** The code must manually detect and work around provider limitations (e.g., DeepSeek‑reasoner rejects `tool_choice`).

**Sub‑steps:**

1. Rely on LiteLLM’s built‑in fallback system: when a feature is unsupported, LiteLLM will either:
   - Strip the unsupported parameter and issue a warning
   - Use an alternative method (e.g., `json_mode` instead of `function_calling`)
   - Raise a clear error that can be caught and displayed in the UI

2. Remove all provider‑specific fallback loops (e.g., the `methods` loop for DeepSeek in `create_agent_for_spec`).

3. Add a feature‑detection helper that queries LiteLLM’s `provider_supports` (or similar) to disable UI toggles for unsupported features.

**Verification:**
- Attempting to use tool calling with a model that doesn’t support it yields a clear user‑friendly message
- The system does not crash when an unsupported parameter is passed

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
**Phase 2 (Steps 4–6)** – File uploads, tool calling, parameter standardization.  
**Phase 3 (Steps 7–9)** – Streaming, UI updates, configuration.  
**Phase 4 (Step 10)** – Testing, validation, and documentation.

Each phase should be merged separately, with the `use_litellm` flag defaulting to `False` until Phase 4, when we switch it to `True` and remove the old adapters.

## Success Metrics

- **Code reduction:** Remove at least 300 lines of provider‑specific branching
- **Feature parity:** All existing features work identically across providers
- **Extensibility:** Adding a new provider requires only adding its model string to the LiteLLM mapping
- **Performance:** No significant increase in latency or token‑counting errors
- **User experience:** No visible change for existing users except improved consistency

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| LiteLLM introduces breaking changes | Pin to a specific minor version (`litellm~=1.81.6`) and test upgrades thoroughly |
| LiteLLM does not support a required feature | Keep the old provider‑specific adapters as fallback paths; contribute missing feature upstream |
| Token counting discrepancies | Compare LiteLLM’s `usage` with provider‑native counts during testing; adjust mapping if needed |
| Increased latency due to translation layer | Benchmark and cache LiteLLM model objects; use LiteLLM’s native provider clients where possible |

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
- Existing provider‑specific code in `src/multi_agent_dashboard/llm_client.py`, `engine.py`, `config.py`

---

*Plan generated on 2026‑02‑01.*
*Target completion: 2026‑02‑15.*