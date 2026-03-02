"""
Multi-Agent Dashboard: A Streamlit-based platform for building, managing, and running multi-agent LLM pipelines.

This package provides a comprehensive framework for orchestrating AI agents with persistent storage,
rich observability, tool calling, and provider-agnostic LLM integration. It features a clean separation
between UI components and the execution engine, enabling reuse in scripts and tests.

Package Structure:
- **config/**: YAML-based configuration with Pydantic validation
- **engine/**: Modular multi-agent orchestration engine (UI-agnostic)
- **runtime/**: AgentRuntime class and execution logic
- **llm_client/**: Provider-agnostic LLM client with factory pattern
- **observability/**: Langfuse integration for tracing and monitoring
- **provider_data/**: Dynamic provider capabilities & pricing data loading
- **tool_integration/**: Tool registry and provider-specific tool adapter
- **ui/**: Streamlit-based web interface components
- **db/**: Database layer with SQLite storage and automatic migrations
- **shared/**: Shared utilities between engine and runtime
- **models.py**: Core data classes (AgentSpec, PipelineSpec)

Key Features:
- **UI-Agnostic Engine**: Reusable agent orchestration in `engine/` and `runtime/` packages
- **Persistent Storage**: SQLite database with automatic migrations
- **Rich Observability**: Cost, latency, logs, and history tracking
- **Tool Calling**: Per-agent controls with provider-specific adapters
- **Provider-Agnostic LLM**: OpenAI, DeepSeek, Ollama via LangChain's unified interface
- **Structured Output**: JSON schema validation with provider-specific methods
- **Dynamic Capabilities**: Provider model data loaded from external sources with local overrides

Usage:
    from multi_agent_dashboard.models import AgentSpec, PipelineSpec
    from multi_agent_dashboard.runtime import AgentRuntime
    from multi_agent_dashboard.engine import MultiAgentEngine
    
    # Create agent specification
    agent = AgentSpec(
        name="researcher",
        model="gpt-4",
        prompt_template="Research topic: {topic}",
        provider_id="openai"
    )
    
    # Run pipeline via UI
    # streamlit run src/multi_agent_dashboard/ui/app.py

Dependencies:
- streamlit: Web interface framework
- sqlite3: Database storage (Python standard library)
- langchain: LLM provider integration (optional)
- pydantic: Configuration validation
- langfuse: Observability and tracing (optional)

Configuration:
- Environment variables in `.env` file (API keys, database path, logging level)
- YAML configuration files in `config/` directory (paths, agents, providers, UI, logging)
- Dynamic provider data in `data/provider_models/` (capabilities, pricing)

See Also:
- `multi_agent_dashboard.ui.app`: Main Streamlit application entry point
- `multi_agent_dashboard.engine.MultiAgentEngine`: Core orchestration engine
- `multi_agent_dashboard.runtime.AgentRuntime`: Individual agent execution
- `multi_agent_dashboard.llm_client.LLMClient`: Provider-agnostic LLM client
"""