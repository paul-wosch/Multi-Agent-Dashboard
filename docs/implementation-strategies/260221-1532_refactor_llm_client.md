**Analysis of `llm_client.py` (1,832 lines)**

## Current Problems
1. **Monolithic functions** – `create_agent_for_spec` (346 lines), `invoke_agent` (527 lines), `_to_dict` (186 lines)
2. **Deep nesting** – Up to 5–6 levels in binding logic; duplicated provider‑specific branches
3. **Mixed responsibilities** – Agent creation also handles middleware instrumentation, tool conversion, structured‑output binding, and tool‑instance preparation
4. **Scattered provider logic** – OpenAI/DeepSeek/Ollama differences appear in schema extraction, structured‑output method selection, and unified‑binding attempts
5. **Hard‑to‑test inner helpers** – Many nested `def` inside methods (e.g., `_wrap_structured_output_model` contains 8 inner helpers)

## Refactoring Recommendations

COMPLETED: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (26 Feb 2026)

NOTE:
- during the refactor line numbers will change
- commit `4407ece fix(llm): drop strict in unified binding` was the basis for this plan

### ✅ 1. **Extract Middleware Instrumentation** (`~lines 497‑581`)
- **New class**: `InstrumentationManager`
- **Responsibilities**: detect existing instrumentation, instantiate/attach middleware, log errors
- **Benefit**: removes 85 lines of tangential logic from `create_agent_for_spec`

### ✅ 2. **Extract Tool Binding & Conversion** (`lines 589‑707`)
- **New class**: `ToolBinder`
- **Responsibilities**: call `convert_tools_for_provider`, handle unified vs sequential binding, retrieve tool instances from registry, manage `web_search_options` vs `tools` separation
- **Benefit**: isolates tool‑related logic, eliminates duplication of tool‑filter mapping

### ✅ 3. **Extract Structured‑Output Binding** (`lines 604‑757`)
- **New class**: `StructuredOutputBinder`
- **Responsibilities**: provider‑specific schema extraction, unified‑binding attempt, fallback to sequential binding, `strict=True/False` provider‑awareness
- **Benefit**: removes duplicated schema‑extraction code (OpenAI/DeepSeek/Ollama branches appear twice)

### ✅ 4. **Create Provider Adapters** (strategy pattern)
- **Interface**: `ProviderAdapter` with methods `extract_schema(response_format)`, `get_structured_output_method()`, `should_use_strict()`
- **Concrete adapters**: `OpenAIAdapter`, `DeepSeekAdapter`, `OllamaAdapter`
- **Benefit**: centralizes provider‑specific logic, makes adding new providers trivial

### ✅ 5. **Split `create_agent_for_spec` into Facade**
```python
def create_agent_for_spec(self, spec, ...):
    middleware = self._instrumentation.prepare(middleware)
    model = self._model_factory.get_model(...)
    model = self._structured_output_binder.bind(model, spec, response_format)
    model, tools = self._tool_binder.bind(model, spec, converted_tools)
    return self._create_agent(model, tools, ...)
```

### ✅ 6. **Move `_wrap_structured_output_model` to a Helper Class**
- **New class**: `StructuredOutputWrapper`
- **Responsibilities**: normalize payloads, extract usage/token counts, wrap results for agent pipeline
- **Benefit**: removes 191‑line nested function; enables standalone testing

### ✅ 7. **Split `invoke_agent`**
- `_prepare_request(spec, messages, files)` – build input with files, apply multimodal handling
- `_execute_with_retry(model, request)` – retry/backoff logic
- `_process_response(raw_response, provider_id)` – cost computation, normalization
- **Benefit**: separates I/O, retry management, and response processing

### ✅ 8. **Extract Response Normalization** (`_to_dict` and helpers)
- **New class**: `ResponseNormalizer`
- **Responsibilities**: convert LangChain/AIMessage objects to uniform dict, merge content blocks, extract tool calls
- **Benefit**: removes 186‑line method with 5 nested helpers

### ✅ 9. **Move `ChatModelFactory` to Its Own Module**
- **File**: `chat_model_factory.py`
- **Benefit**: reduces line count of main file, clearer separation of model‑caching concerns

### ✅ 10. **Create `llm_client/` Subpackage**
```
llm_client/
├── __init__.py
├── core.py                  # LLMClient facade
├── instrumentation.py
├── tool_binder.py
├── structured_output.py
├── provider_adapters.py
├── response_normalizer.py
├── chat_model_factory.py
├── wrappers.py              # StructuredOutputWrapper
└── ...                      # additional modules related to LLMClient
```

## Expected Benefits
- **Readability**: Each module ≤300 lines, single responsibility
- **Testability**: Isolated components can be unit‑tested without mocking entire LLMClient
- **Maintainability**: Adding a new provider requires only a new adapter class
- **Risk mitigation**: Incremental extraction; original public API remains unchanged

## Risk Assessment
- **Low risk** if extraction is done gradually with delegation from existing methods
- **No changes to external interface** – `create_agent_for_spec` and `invoke_agent` signatures stay identical
- **Potential hidden dependencies** on closure variables (need careful parameter passing)

## Next Steps
1. **Approval**: Do you want to proceed with any of these refactorings?
2. **Priority**: Which area is most painful (agent creation, response normalization, tool binding)?
3. **Incremental plan**: Start with extracting `ChatModelFactory` and `InstrumentationManager` (lowest risk).

**Recommendation**: Begin with steps 1, 2, and 9 to immediately reduce `create_agent_for_spec` by ~200 lines while keeping the change safe and reversible.