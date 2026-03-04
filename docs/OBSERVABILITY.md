# 🔍 Observability with Langfuse

The Multi-Agent Dashboard includes rich built-in observability features for monitoring agent executions, including cost computation, token usage tracking, latency measurement, tool invocation logging, and historical run storage. These features are always available and provide comprehensive visibility into pipeline performance. For details, see [ARCHITECTURE.md](ARCHITECTURE.md) and [USAGE.md](USAGE.md).

The dashboard also offers optional integration with [Langfuse](https://langfuse.com) for advanced distributed tracing, detailed latency breakdown, and enhanced visualization of LLM calls, tool invocations, and agent reasoning steps. Langfuse **augments** the existing observability stack—it does not replace or interfere with built-in metrics, logs, or history. Both systems work side‑by‑side, giving you the flexibility to use either or both according to your needs.

## Quick Start

To enable Langfuse observability:

1. Add your Langfuse credentials to `.env`:

   ```bash
   LANGFUSE_PUBLIC_KEY=your_public_key
   LANGFUSE_SECRET_KEY=your_secret_key
   # Optional: LANGFUSE_BASE_URL=https://self-hosted.langfuse.com
   ```

2. Restart the dashboard.

This document provides detailed technical information about the Langfuse integration architecture, configuration, and usage.

## Architecture Overview

The Langfuse integration follows a modular, optional-dependency design:

```
src/multi_agent_dashboard/
├── observability/
│   └── langfuse.py                    # Singleton client & handler management
└── llm_client/
    ├── observability/
    │   └── langfuse_integration.py    # Invocation config builder
    └── core/
        ├── execution_engine.py        # LLM execution with tracing
        └── availability.py            # Conditional imports
```

### Key Components

1. **`langfuse.py`** – Singleton client with lazy initialization, automatic `atexit` flush registration, and environment variable setup.
2. **`langfuse_integration.py`** – Builds invocation configurations with metadata (agent tags, pipeline context).
3. **`ExecutionEngine`** – Injects Langfuse callback handlers into LangChain agent invocations.
4. **`LLMClient`** – Exposes `flush_langfuse()` method for manual control.

## Configuration

### Environment Variables

Langfuse integration is configured via environment variables in `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `LANGFUSE_PUBLIC_KEY` | Yes* | Public API key from Langfuse dashboard |
| `LANGFUSE_SECRET_KEY` | Yes* | Secret API key from Langfuse dashboard |
| `LANGFUSE_BASE_URL` | No | Langfuse server URL (default: `https://cloud.langfuse.com`) |
| `LANGFUSE_ENABLED` | No | Explicitly disable Langfuse (set to `false` to disable even if keys are present) |

*\* Both keys must be present for auto-enablement, unless `LANGFUSE_ENABLED=false`.*

### Auto-Enablement Logic

Langfuse is automatically enabled when:
1. Both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set in `.env`
2. `LANGFUSE_ENABLED` is not explicitly set to `"false"`

The integration remains **completely inactive** with **zero overhead** when disabled.

### Dependency

Langfuse v3.x.x SDK is included as a dependency in `pyproject.toml`:

```toml
dependencies = [
    # ...
    "langfuse==3.14.5",
]
```

The SDK is only imported when Langfuse is enabled, ensuring no runtime overhead for users who don't need observability.

## How the Integration Works

### 1. Client Initialization

On first use, `is_langfuse_enabled()` checks configuration and imports the SDK:

```python
def _ensure_langfuse_initialized() -> bool:
    # Sets environment variables for Langfuse SDK
    # Initializes singleton client via langfuse.get_client()
    # Registers atexit flush handler
```

### 2. Handler Creation

For each agent invocation, `get_langfuse_handler()` creates a `CallbackHandler` instance:

```python
def get_langfuse_handler(
    session_id: Optional[str] = None,
    user_id: str = "multi_agent_dashboard",
    tags: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
```

### 3. Invocation Configuration

`build_langfuse_config()` constructs the configuration dictionary for LangChain:

```python
def build_langfuse_config(
    agent: Any,
    context: Optional[Dict[str, Any]] = None,
    langfuse_enabled: bool = False,
) -> Dict[str, Any]:
```

The configuration includes:
- `callbacks`: List containing the Langfuse handler
- `run_name`: Agent name (appears as "Name" in Langfuse UI)
- `metadata`: Session ID, user ID, tags for correlation

### 4. Execution Integration

The `ExecutionEngine` injects the configuration into agent invocations:

```python
invoke_config = build_langfuse_config(
    agent,
    context=context,
    langfuse_enabled=self._langfuse_enabled,
)
result = agent.invoke(inputs, config=invoke_config)
```

## What Gets Traced

### Metadata & Tags

Each trace includes the following metadata for correlation:

| Field | Source | Example |
|-------|--------|---------|
| **Session ID** | Pipeline name or `"Ad‑Hoc"` | `my_pipeline` |
| **User ID** | Static identifier | `"multi_agent_dashboard"` |
| **Run Name** | Agent name | `"planner"` |
| **Tags** | Context-derived labels | `["agent:planner", "pipeline:my_pipeline"]` |

### Trace Content

- **LLM Calls**: Model interactions, prompts, responses
- **Tool Invocations**: Tool calls with inputs/outputs
- **Agent Reasoning**: Intermediate reasoning steps
- **Latency**: Per-step timing breakdown
- **Token Usage**: Input/output token counts (when available)
- **Provider Costs**: Estimated costs based on dynamic pricing data

### Context Propagation

The integration automatically propagates context through the execution chain:

```python
# From pipeline execution
context = {
    "pipeline_name": "my_pipeline",
}

# Automatically added to tags
tags = ["pipeline:my_pipeline", "agent:planner"]
```

## Usage Examples

### Basic Setup

1. Add credentials to `.env`:

```bash
echo 'LANGFUSE_PUBLIC_KEY=your_public_key' >> .env
echo 'LANGFUSE_SECRET_KEY=your_secret_key' >> .env
```

2. Restart the dashboard:

```bash
streamlit run src/multi_agent_dashboard/ui/app.py
```

### Self-Hosted Langfuse

For self-hosted Langfuse instances:

```bash
# .env
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL=https://langfuse.your-domain.com
```

### Explicit Disablement

To explicitly disable Langfuse even when credentials are present:

```bash
# .env
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_secret
LANGFUSE_ENABLED=false  # Force disable
```

### Manual Flush

For explicit control (e.g., in scripts):

```python
from multi_agent_dashboard.llm_client import LLMClient

client = LLMClient()
# ... run agents ...
client.flush_langfuse()  # Ensure all traces are sent
```

The automatic `atexit` flush ensures traces are sent on normal program exit.

## Troubleshooting

### No Traces Appearing in Langfuse

1. **Check environment variables**: Verify `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set correctly.
2. **Verify Langfuse URL**: Ensure `LANGFUSE_BASE_URL` points to the correct instance.
3. **Check network connectivity**: Ensure the dashboard can reach the Langfuse server.
4. **Review logs**: Look for Langfuse-related warnings or errors in application logs.

### Import Errors

If you encounter `ImportError: cannot import name 'CallbackHandler'`:

- Ensure you have Langfuse v3.x.x installed (`langfuse>=3.0.0`)
- The dashboard uses the v3 import path: `from langfuse.langchain import CallbackHandler`

### Performance Considerations

- **Zero overhead when disabled**: No SDK imports, no runtime checks.
- **Minimal overhead when enabled**: Singleton client, lazy initialization.
- **Network calls asynchronous**: Tracing data is sent asynchronously to avoid blocking agent execution.

## Integration Points

### For Developers

To add Langfuse tracing to custom components:

```python
from multi_agent_dashboard.observability import get_langfuse_handler

# Create handler for custom execution
handler = get_langfuse_handler(
    session_id="custom_session",
    tags=["custom:component"],
    metadata={"custom_field": "value"}
)

# Use with LangChain components
result = chain.invoke(inputs, config={"callbacks": [handler]})
```

### Extending the Integration

The modular design allows easy extension:

1. **Additional metadata**: Extend `build_langfuse_config()` to include custom fields.
2. **Custom tags**: Modify context propagation to include additional tagging logic.

## Related Documentation


- [CONFIG.md](CONFIG.md) – Full configuration reference
- [ARCHITECTURE.md](ARCHITECTURE.md) – System architecture overview
- [DEVELOPMENT.md](DEVELOPMENT.md) – Developer workflow and contribution guidelines

## Implementation Notes

- **Thread-safe singleton**: The Langfuse client uses lazy initialization with thread safety.
- **Graceful degradation**: Import errors are caught and logged, not raised.
- **Backward compatibility**: The integration maintains compatibility with existing agent configurations.
- **No configuration changes required**: Existing pipelines work unchanged with observability enabled.
