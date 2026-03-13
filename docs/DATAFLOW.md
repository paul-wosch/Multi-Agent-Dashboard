# Data Flow and Data Transformations

This document explains the actual data flow through the pipeline, focusing on the **data transformations** at each layer rather than database interactions.

---
## Pipeline Data Flow: Actual Data Transformations

### High-Level Data Flow

```
UI/CLI Input → PipelineState Creation → Agent Execution Loop → LLM Client → Provider Adapter → LangChain → LLM API
                                    ↑                                      ↓
State Updates ←──── Metrics Extraction ←──── Response Processing ←──── Raw LLM Response
```

### 1. Pipeline Initialization (`engine_orchestrator.py:run_seq()`)

**Input Data**:
- `steps: List[str]` - Agent names to execute sequentially
- `initial_input: Any` - Initial task/input from UI
- `files: Optional[List[Dict]]` - Uploaded files with `filename`, `content`, `mime_type`
- `allowed_domains: Optional[Any]` - Domain filtering for web search

**Data Transformation**:
```python
# Creates the central PipelineState container
pipeline_state = PipelineState(
    pipeline_name=pipeline_name,
    state={
        "task": initial_input,      # Primary task
        "input": initial_input,     # Backward compatibility
    },
    memory={},                      # Per-agent raw outputs
    warnings=[],
    tool_usages={},                 # Tool call records
    agent_configs={},               # Configuration snapshots
    strict_schema_exit=False,
    agent_schema_validation_failed={},
    agent_metrics={},               # Token/cost metrics
)

# File injection into state
if files:
    pipeline_state.state["files"] = files  # All agents can access

# Domain filtering configuration
if allowed_domains:
    if isinstance(allowed_domains, dict):
        pipeline_state.state["allowed_domains_by_agent"] = filtered
    else:
        pipeline_state.state["allowed_domains"] = allowed_domains
```

**Key Transformation**: Raw UI inputs → Structured `PipelineState` with initial context.

### 2. Per-Agent Execution Loop (`engine_orchestrator.py`)

For each agent in `steps`:
1. **Agent Retrieval**: Get `AgentRuntime` instance by name
2. **Execution Delegation**: Call `executor.execute_agent(agent_name, agent, pipeline_state)`
3. **State Propagation**: `pipeline_state` is mutated in-place by the executor
4. **Early Exit Check**: Break if `strict_schema_exit` triggered

**Data Flow Between Loop Iterations**: `pipeline_state.state` accumulates outputs from previous agents, becoming inputs for subsequent agents.

### 3. Agent Execution (`agent_executor.py:execute_agent()`)

**Input Validation**:
- Checks `agent.spec.input_vars` against `pipeline_state.state`
- Special handling for `"files"` variable (injected once)
- Warns or raises based on `strict` mode

**Agent Invocation**:
```python
# Pass files if agent.run() accepts them
run_kwargs = {}
if "files" in inspect.signature(agent.run).parameters:
    run_kwargs["files"] = pipeline_state.state.get("files")
    
# Actual agent execution
raw_output = agent.run(pipeline_state.state, **run_kwargs)
```

**Metrics Extraction**:
1. **Raw Metrics**: From `agent.last_metrics` (populated by `AgentRuntime`)
2. **Token Extraction**: Multiple fallback paths from `raw_metrics`
3. **Cost Computation**: Using `MetricsAggregator.compute_cost()` with model/provider rates
4. **Tool Usage**: Collected from `metrics.get("tools")` with config attachments

**Structured Output Detection**:
Four-path detection hierarchy:
1. **Canonical 'structured' key** in `raw_metrics` (from LLM client)
2. **'structured_response' key** in `raw_metrics` (LangChain agents)
3. **Content blocks search** for `structured`, `structured_response`, `tool_call` types
4. **JSON parsing fallback** of `raw_output` text

**State Writeback**:
```python
if agent.spec.output_vars:
    if isinstance(parsed, dict):
        # Map parsed dict keys to output_vars
        for var in agent.spec.output_vars:
            if var in parsed:
                pipeline_state.state[var] = parsed[var]
    else:
        if len(agent.spec.output_vars) == 1:
            # Single output var gets raw output
            pipeline_state.state[agent.spec.output_vars[0]] = raw_output
else:
    # No output_vars: store under agent name
    pipeline_state.state[agent_name] = raw_output
```

**Schema Validation**:
- Validates `parsed` output against `agent.spec.structured_output_schema`
- Sets `pipeline_state.strict_schema_exit` if validation fails in strict mode
- Records failures in `pipeline_state.agent_schema_validation_failed`

### 4. Agent Runtime Execution (`agent_runtime.py:run()`)

**File Processing**:
```python
# Split files into text and binary
text_files, binary_files = process_files(all_files)

# Build LLM payload
llm_files_payload = []
for f in text_files:
    llm_files_payload.append({
        "filename": f["filename"],
        "content": f["content"],      # Text content
        "mime_type": f["mime_type"],
    })
for f in binary_files:
    llm_files_payload.append(f)       # Binary files for upload
```

**Prompt Formatting**:
```python
# Extract variables from state
variables = {k: state.get(k, "") for k in self.spec.input_vars}

# Inject file summaries if "files" in input_vars
if "files" in self.spec.input_vars and all_files:
    variables["files"] = "\n".join(
        f"- {f['filename']} ({f['mime_type']})"
        for f in all_files
    )

# Safe template substitution
prompt = safe_format(self.spec.prompt_template, variables)
```

**Tool Preparation**:
```python
# Build configurations
tc = build_tools_config(self.spec, state)      # Tools config
rc = build_reasoning_config(self.spec)         # Reasoning config
allowed_domains = get_allowed_domains(self.spec, state)

# Convert to provider-specific tools
langchain_tools = prepare_tools_for_agent(
    self.spec,
    state,
    provider_id,
    model,
    use_responses_api,
    provider_features,
)
```

**LLM Client Invocation**:
```python
# Create LangChain agent with tools
agent = self.llm_client.create_agent_for_spec(
    self.spec,
    tools=langchain_tools,
    middleware=None,
    response_format=structured_schema,
)

# Build context for middleware
context = {}
if tc: context["tools_config"] = tc
if rc: context["reasoning_config"] = rc
if "allowed_domains" in state: context["allowed_domains"] = state["allowed_domains"]

# Invoke via LLMClient
response = self.llm_client.invoke_agent(
    agent,
    prompt,
    files=llm_files_payload if llm_files_payload else None,
    response_format=structured_schema,
    stream=stream,
    context=context if context else None,
)
```

**Response Processing**:
```python
# Extract from TextResponse
raw = response.raw or {}
input_tokens, output_tokens = extract_tokens_from_raw(raw, response)

# Collect instrumentation data
instrumentation_events = _extract_instrumentation_events(raw)
content_blocks = _collect_content_blocks(raw)

# Store metrics for engine
self.last_metrics = {
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "latency": response.latency,
    "raw": raw,
    "content_blocks": content_blocks,
    "instrumentation_events": instrumentation_events,
    "tools_config": tc,
    "reasoning_config": rc,
    "used_langchain_agent": True,
}

# Extract tool usage
used_tools = collect_tool_usage(raw, content_blocks)
if used_tools:
    self.last_metrics["tools"] = used_tools

# Structured output detection
raw_output = response.text
parsed = detect_structured_output(raw, content_blocks, raw_output)
writeback_to_state(self.spec, state, parsed, raw_output)

return response.text
```

### 5. LLM Client Layer (`llm_client/core/client.py`)

**Agent Creation**:
- Converts `AgentSpec` to LangChain agent with appropriate runtime
- Attaches instrumentation middleware for metrics collection
- Configures provider-specific adaptations (OpenAI, Anthropic, Ollama, etc.)

**Agent Invocation**:
- Handles provider-specific file upload mechanisms
- Manages streaming vs. non-streaming responses
- Applies response normalization via `ResponseNormalizer`

**Response Normalization**:
```python
# Returns consistent TextResponse
TextResponse(
    text=normalized_text,      # Extracted response text
    raw=raw_data,              # Raw provider response
    latency=calculated_latency,
    # Plus structured, files, etc.
)
```

### 6. Final Aggregation (`engine_orchestrator.py`)

**State Synchronization**:
```python
# Update engine instance attributes
self.state_manager.update_from_pipeline_state(pipeline_state)
```

**Output Determination**:
```python
# Prefer "final" key, fallback to last agent's output
final_output = self.state.get("final", last_output)
```

**Metrics Aggregation**:
```python
total_cost, total_input_cost, total_output_cost, total_latency = MetricsAggregator.aggregate_totals(
    list(pipeline_state.agent_metrics.values())
)
```

**EngineResult Construction**:
```python
return EngineResult(
    final_output=final_output,
    state=dict(self.state),                    # Final state dict
    memory=dict(self.memory),                  # Per-agent raw outputs
    warnings=list(self._warnings),
    errors=list(self._errors),
    final_agent=last_agent,
    agent_metrics=dict(self.agent_metrics),    # Per-agent token/cost metrics
    agent_configs=pipeline_state.agent_configs, # Configuration snapshots
    total_cost=total_cost,
    total_latency=total_latency,
    total_input_cost=total_input_cost,
    total_output_cost=total_output_cost,
    tool_usages=pipeline_state.tool_usages,    # Tool call records
    strict_schema_exit=pipeline_state.strict_schema_exit,
    agent_schema_validation_failed=pipeline_state.agent_schema_validation_failed,
)
```

### 7. Instrumentation & Metrics Flow

**Real-time Collection**:
- LangChain middleware captures `content_blocks`, `tool_calls`, `usage_metadata`
- Provider adapters extract token counts from raw responses
- Latency measured from invocation start to response receipt

**Aggregation Points**:
1. **Per-agent**: `agent_runtime.py` collects into `last_metrics`
2. **Per-execution**: `agent_executor.py` extracts and computes costs
3. **Pipeline-level**: `engine_orchestrator.py` aggregates totals

**Database Persistence** (not focus, but for completeness):
- `EngineResult` passed to `RunService.save_run()`
- Metrics stored in `agent_runs` and `agent_run_configs` tables
- Tool usage records in `tool_usage_events`

---

## ASCII Diagram: Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                PIPELINE DATA FLOW (Actual Transformations)                  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────┐    ┌─────────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│   UI/CLI   │    │   MultiAgentEngine  │    │    AgentExecutor     │    │  AgentRuntime    │
│   Input    │    │   (engine_orchestra-│    │  (agent_executor.py) │    │(agent_runtime.py)│
└─────┬──────┘    │      tor.py)        │    │                      │    └────────┬─────────┘
      │           └──────────┬──────────┘    └──────────┬───────────┘             │
      │ initial_input,       │                          │                         │
      │ steps, files,        │ create PipelineState     │ execute_agent()         │ run()
      │ allowed_domains      │                          │                         │
      ▼                      ▼                          ▼                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                               DATA TRANSFORMATION STEPS                                    │
├────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                            │
│ 1. INITIALIZATION:                                                                         │
│    Inputs → PipelineState {                                                                │
│      state: {task, input, files?, allowed_domains?},                                       │
│      memory: {}, warnings: [], tool_usages: {}, ...                                        │
│    }                                                                                       │
│                                                                                            │
│ 2. PER-AGENT LOOP (for each agent_name in steps):                                          │
│    │                                                                                       │
│    ├─A. INPUT VALIDATION:                                                                  │
│    │   Check agent.spec.input_vars against pipeline_state.state                            │
│    │   → Warn/error if missing (strict mode)                                               │
│    │                                                                                       │
│    ├─B. AGENT RUNTIME EXECUTION:                                                           │
│    │   │                                                                                   │
│    │   ├─i. FILE PROCESSING:                                                               │
│    │   │   all_files → text_files (content) + binary_files (upload)                        │
│    │   │   → llm_files_payload [{filename, content, mime_type}, ...]                       │
│    │   │                                                                                   │
│    │   ├─ii. PROMPT FORMATTING:                                                            │
│    │   │   variables = {k: state[k] for k in input_vars}                                   │
│    │   │   + file summaries if "files" in input_vars                                       │
│    │   │   → prompt = safe_format(template, variables)                                     │
│    │   │                                                                                   │
│    │   ├─iii. TOOL PREPARATION:                                                            │
│    │   │   tc = build_tools_config(spec, state)                                            │
│    │   │   rc = build_reasoning_config(spec)                                               │
│    │   │   allowed_domains = get_allowed_domains(spec, state)                              │
│    │   │   → langchain_tools = prepare_tools_for_agent(...)                                │
│    │   │                                                                                   │
│    │   ├─iv. LLM INVOCATION:                                                               │
│    │   │   agent = llm_client.create_agent_for_spec(spec, tools, response_format)          │
│    │   │   context = {tools_config: tc, reasoning_config: rc, ...}                         │
│    │   │   → response = llm_client.invoke_agent(agent, prompt, files, context)             │
│    │   │                                                                                   │
│    │   └─v. RESPONSE PROCESSING:                                                           │
│    │       raw = response.raw                                                              │
│    │       input_tokens, output_tokens = extract_tokens_from_raw(raw, response)            │
│    │       content_blocks = _collect_content_blocks(raw)                                   │
│    │       used_tools = collect_tool_usage(raw, content_blocks)                            │
│    │       raw_output = response.text                                                      │
│    │       parsed = detect_structured_output(raw, content_blocks, raw_output)              │
│    │       writeback_to_state(spec, state, parsed, raw_output)                             │
│    │       → last_metrics = {input_tokens, output_tokens, raw, content_blocks, tools, ...} │
│    │                                                                                       │
│    ├─C. METRICS EXTRACTION:                                                                │
│    │   metrics = agent.last_metrics                                                        │
│    │   input_tokens, output_tokens = extract_with_fallbacks(metrics)                       │
│    │   total_cost, input_cost, output_cost = compute_cost(model, tokens, provider)         │
│    │   → run_metrics = RunMetrics(...)                                                     │
│    │   → pipeline_state.agent_metrics[agent_name] = run_metrics                            │
│    │                                                                                       │
│    ├─D. STRUCTURED OUTPUT DETECTION:                                                       │
│    │   1. raw_metrics["structured"]                                                        │
│    │   2. raw_metrics["structured_response"]                                               │
│    │   3. content_blocks search (type="structured"/"tool_call")                            │
│    │   4. JSON parse fallback of raw_output                                                │
│    │   → parsed = detected_output                                                          │
│    │                                                                                       │
│    ├─E. STATE WRITEBACK:                                                                   │
│    │   if output_vars:                                                                     │
│    │     if parsed is dict: state[var] = parsed[var] for var in output_vars                │
│    │     else if single output_var: state[output_var] = raw_output                         │
│    │     else: state[f"{agent_name}__raw"] = raw_output                                    │
│    │   else: state[agent_name] = raw_output                                                │
│    │                                                                                       │
│    ├─F. SCHEMA VALIDATION:                                                                 │
│    │   status = validate(agent_spec, parsed, raw_output)                                   │
│    │   if validation_failed and strict_schema_validation:                                  │
│    │     pipeline_state.strict_schema_exit = True                                          │
│    │     → BREAK LOOP (early exit)                                                         │
│    │                                                                                       │
│    └─G. MEMORY STORAGE:                                                                    │
│        pipeline_state.memory[agent_name] = raw_output                                      │
│        pipeline_state.tool_usages[agent_name] = used_tools (if any)                        │
│        pipeline_state.agent_configs[agent_name] = agent_config snapshot                    │
│                                                                                            │
│ 3. FINAL AGGREGATION:                                                                      │
│    final_output = state.get("final", last_output)                                          │
│    total_cost, total_input_cost, total_output_cost, total_latency =                        │
│        aggregate_totals(pipeline_state.agent_metrics.values())                             │
│                                                                                            │
│    → EngineResult {                                                                        │
│        final_output,                                                                       │
│        state, memory, warnings, errors,                                                    │
│        agent_metrics, agent_configs,                                                       │
│        total_cost, total_latency, total_input_cost, total_output_cost,                     │
│        tool_usages, strict_schema_exit, agent_schema_validation_failed                     │
│      }                                                                                     │
│                                                                                            │
└────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   LLM Client │     │  Provider    │     │   LangChain  │
│   (Client)   │     │   Adapter    │     │    Agent     │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │ create_agent_for_  │ provider-specific  │ agent.invoke()
       │ _spec()            │ tool/file adapters │ with middleware
       │ invoke_agent()     │                    │
       ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│                    LLM API CALL                         │
│  - Files uploaded (binary) or injected (text)           │
│  - Tools converted to provider format                   │
│  - Structured output schema applied                     │
│  → Raw provider response + usage metadata               │
└─────────────────────────────────────────────────────────┘
       │                    │                    │
       │ TextResponse       │ metrics extraction │ instrumentation
       │ normalization      │                    │ events
       ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│              INSTRUMENTATION & METRICS FLOW             │
│                                                         │
│  Real-time:                                             │
│  - content_blocks capture                               │
│  - tool_calls recording                                 │
│  - token usage extraction                               │
│  - latency measurement                                  │
│                                                         │
│  Aggregation:                                           │
│  1. AgentRuntime.last_metrics                           │
│  2. AgentExecutor RunMetrics + cost computation         │
│  3. Engine totals aggregation                           │
│  4. DB persistence (RunService.save_run)                │
└─────────────────────────────────────────────────────────┘
```

### Key Data Transformation Points

1. **Files → Processed Payload**: Raw file uploads → categorized text/binary → LLM-ready format
2. **State Variables → Prompt**: `state[k]` values → formatted prompt with template substitution
3. **Tool Config → Provider Tools**: Abstract tool definitions → provider-specific implementations
4. **Raw LLM Response → Normalized Text**: Multi-format provider responses → consistent `TextResponse`
5. **Response → Structured Output**: Raw text/content blocks → parsed dictionary via 4-path detection
6. **Per-Agent Metrics → Pipeline Totals**: Individual token/cost metrics → aggregated pipeline totals
7. **PipelineState Evolution**: Initial state → accumulates outputs → final state with all results

### State Propagation Through Layers

```
UI Input
    │
    ▼
PipelineState (initial)           ← Created once at pipeline start
    │
    ├─ Agent 1 ────────────────────┐
    │   │                          │
    │   ├─ input_vars validation   │
    │   ├─ prompt formatting       │
    │   ├─ LLM invocation          │
    │   ├─ response processing     │
    │   └─ state[output_vars] = ...│
    │                              │
    ├─ Agent 2 ────────────────────┤
    │   │ (uses updated state)     │
    │   └─ state[output_vars] = ...│
    │                              │
    └─ Agent N ────────────────────┤
        │                          │
        └─ Final state aggregation │
                                   │
EngineResult ←─────────────────────┘
    │
    ▼
UI Output / DB Storage
```
