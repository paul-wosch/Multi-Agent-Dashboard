# 🏗️ Architecture (expanded)

This document expands the short architecture overview in the README with responsibilities and common extension points.

## High-level layers

The Multi-Agent Dashboard follows a layered, modular architecture that separates concerns and enables clear extension points. Each layer has distinct responsibilities and interacts with adjacent layers through well-defined interfaces.

### 🎛️ UI Layer (`ui/`)

**Purpose**: Provides the user interface for building, managing, and executing multi-agent pipelines via Streamlit.

**Responsibilities**:
- Presents interactive dashboards for agent editing, pipeline configuration, and run execution
- Visualizes run history, metrics, logs, and pipeline graphs
- Handles user inputs and file uploads, translating them into internal data structures

**Integration**: Communicates exclusively with the Service Layer (`db/services.py`) and Engine (`engine/`) for business logic—never directly accessing database internals. This ensures the UI remains agnostic to persistence details and can be replaced or extended independently.

**Design rationale**: Streamlit was chosen for rapid prototyping and rich interactive capabilities while maintaining Python‑native development. The UI is intentionally isolated from core engine logic to support future CLI or API frontends.

### 🧰 Service Layer (`db/services.py`)

**Purpose**: Provides high-level, transactional APIs that coordinate persistence and engine operations.

**Responsibilities**:
- Orchestrates complex operations involving multiple DAOs (agents, pipelines, runs)
- Manages transaction boundaries and error handling for UI and script clients
- Coordinates between persistence operations and engine executions

**Integration**: Sits between the UI and both the Persistence Layer (DAOs) and Engine Layer. Services call DAOs for CRUD operations and invoke the Engine for agent/pipeline execution, returning consolidated results.

**Design rationale**: Services encapsulate business logic that spans multiple entities, ensuring data consistency and providing a clean API for UI consumers.

### 🗃️ Persistence Layer (`db/*.py`)

**Purpose**: Manages persistent storage of agents, pipelines, runs, and related metadata in SQLite.

**Responsibilities**:
- Defines Data Access Objects (DAOs) for each entity type (`agents.py`, `pipelines.py`, `runs.py`)
- Handles low‑level SQL operations, parameter binding, and result mapping
- Provides backward‑compatible entry points via `db.py` (re‑exports DAOs and `init_db`)

**Integration**: Used by the Service Layer for all database operations. The DAOs are simple, focused classes that abstract table‑specific SQL away from business logic.

**Design rationale**: SQLite offers zero‑configuration, file‑based storage suitable for local development and lightweight deployments. DAOs keep SQL isolated and make schema evolution manageable through migrations.

### 🧱 DB Infrastructure (`db/infra/`)

**Purpose**: Provides database connectivity, schema management, migration tooling, and maintenance utilities.

**Responsibilities**:
- Bootstraps database connections and applies migrations (`core.py`, `migrations.py`)
- Defines the canonical schema (`schema.py`) and generates migration scripts (`generate_migration.py`)
- Supports safe schema evolution via rebuild helpers (`sqlite_rebuild.py`) for destructive changes
- Offers maintenance utilities: backups, snapshot pruning, constraint diffing, etc.

**Integration**: Used by the Persistence Layer during initialization and by maintenance scripts. The infrastructure is transparent to normal application code.

**Design rationale**: SQLite lacks built‑in migration facilities; this layer provides a robust, version‑controlled schema‑evolution workflow with safety nets for production data.

### 🧠 Engine & Runtime (`engine/`, `runtime/`)

**Purpose**: Executes multi‑agent pipelines with instrumentation, tool calling, structured output, and provider‑agnostic LLM integration.

**Responsibilities**:
- **Engine** (`engine/`): Orchestrates multi‑agent workflows, manages agent state, aggregates metrics, validates schemas, and reports progress.
- **Runtime** (`runtime/`): Executes individual agents with tool invocation, file processing, structured‑output detection, and metrics extraction.

**Integration**: The Engine is invoked by the Service Layer for pipeline runs. It delegates individual agent execution to the Runtime, which in turn uses the LLM Client for model calls. The Engine and Runtime share utilities via the `shared/` package.

**Design rationale**: Separating orchestration (Engine) from per‑agent execution (Runtime) allows reuse of the engine in non‑UI contexts (scripts, tests). The Runtime encapsulates all provider‑specific adaptations, keeping the engine provider‑agnostic.

### 🔌 LLM Client & Provider Integration (`llm_client/`, `tool_integration/`, `provider_data/`)

**Purpose**: Abstracts LLM provider differences, handles tool calling, structured output, multimodal inputs, and loads dynamic capability data.

**Responsibilities**:
- **LLM Client** (`llm_client/`): Provider‑agnostic facade with modular core implementation; handles request building, execution with retries, response normalization, instrumentation, and observability.
- **Tool Integration** (`tool_integration/`): Manages tool registry and provider‑specific tool‑calling adapters.
- **Provider Data** (`provider_data/`): Downloads, caches, and filters dynamic provider capabilities and pricing data.

**Integration**: The Runtime calls the LLM Client for model interactions. The client uses provider‑specific adapters and tool integration to bridge between unified agent configuration and provider‑native APIs.

**Design rationale**: LangChain's unified `init_chat_model` interface is used internally, but the client adds a consistent abstraction layer with instrumentation, cost tracking, and structured‑output support across all providers. Dynamic capability data ensures UI warnings and defaults stay up‑to‑date without code changes.

### 🧩 Shared Utilities & Configuration (`shared/`, `config/`, `observability/`)

**Purpose**: Provides cross‑cutting utilities, configuration management, and observability integrations.

**Responsibilities**:
- **Shared** (`shared/`): Utilities used by both Engine and Runtime (instrumentation helpers, capability mapping, runtime hooks, schema resolution).
- **Configuration** (`config/`): Loads environment variables, validates YAML configuration files, and exposes global constants.
- **Observability** (`observability/`): Optional Langfuse integration for distributed tracing of LLM calls and agent executions.

**Integration**: Used throughout the codebase via imports. Configuration is loaded once at startup; observability hooks are injected into the LLM Client middleware.

**Design rationale**: Centralizing configuration and shared utilities reduces duplication and ensures consistent behavior. Observability is optional and non‑invasive—enabled only when credentials are present.

### 📦 Models & Result DTOs (`models.py` + `engine/`)

**Purpose**: Defines the core data structures and result containers used across all layers.

**Responsibilities**:
- **Core domain models** (`models.py`): Immutable dataclasses for `AgentSpec`, `PipelineSpec`.
- **Engine result DTOs** (`engine/engine_orchestrator.py`, `engine/types.py`): `EngineResult`, `AgentRunResult`, `PipelineState`, `RunMetrics`.
- Provides validation, serialization, and convenience methods.

**Integration**: Imported and used by UI, Services, Engine, Runtime, and Persistence layers. Serve as the common language for data exchange.

**Design rationale**: Immutable dataclasses ensure thread‑safe data passing and clear contracts between layers. They also enable easy serialization for storage and UI display.

### A typical flow:

```text
UI (Streamlit)
  → Services (AgentService / RunService / PipelineService)
    → DAOs (agents / pipelines / runs)
      → DB (SQLite)
  → Engine (orchestrates agents & LLM calls)
    → Runtime (executes individual agents with tool calling, structured output, instrumentation)
      → LLM Client (provider‑agnostic interface)
        → Provider‑specific adapter (OpenAI, DeepSeek, Ollama)
          → LangChain's unified init_chat_model
```

### Extension points

- Add new tools: implement tool adapters and register them in the tool registry (`tool_integration/registry.py`).
- Add new persistence backends: replace DAO internals while keeping service contracts.
- Add integrations (e.g., telemetry): use a pluggable logging/metrics interface in `config.configure_logging()` (from the `config` package).
- Add new LLM providers: ensure capabilities are included in provider data and implement provider‑specific adapters in `llm_client/provider_adapters.py` and `tool_integration/provider_tool_adapter.py`.

### Read the code

Important files to review:

- `src/multi_agent_dashboard/ui/app.py`
- `src/multi_agent_dashboard/engine/engine_orchestrator.py`
- `src/multi_agent_dashboard/runtime/agent_runtime.py`
- `src/multi_agent_dashboard/llm_client/core/client.py`
- `src/multi_agent_dashboard/db/infra/schema.py`

---

## 🗂️ Repository Structure

```text
repo_root/
├── .env                                        # Environment variables (API keys, secrets; untracked)
├── .env.template                               # Environment variable template
├── .gitignore                                  # Git ignore patterns
├── LICENSE                                     # MIT License
├── README.md                                   # Project overview and quick start guide
├── pyproject.toml                              # Python project configuration and dependencies
├── AGENTS.md                                   # Agent guidelines and project documentation (this file)
│
└── src/                                        # Source code directory
    └── multi_agent_dashboard/                  # Main Python package
        ├── __init__.py                         # Package exports and version
        ├── models.py                           # Core data classes (AgentSpec, PipelineSpec) – immutable dataclasses
        │
        ├── config/                             # YAML-based configuration package
        │   ├── __init__.py                     # Public API (same constants as before)
        │   ├── core.py                         # Core configuration loading
        │   └── loader.py                       # YAML validation with Pydantic
        │
        ├── engine/                             # Modular multi-agent orchestration engine
        │   ├── __init__.py                     # Engine package exports
        │   ├── agent_executor.py               # Individual agent execution logic
        │   ├── engine_orchestrator.py          # Multi-agent orchestration and coordination
        │   ├── metrics_aggregator.py           # Metrics collection and aggregation
        │   ├── progress_reporter.py            # Progress reporting and status updates
        │   ├── schema_validator.py             # JSON schema validation for structured output
        │   ├── snapshot_builder.py             # Agent state snapshot creation and management
        │   ├── state_manager.py                # Agent state persistence and retrieval
        │   ├── types.py                        # Engine type definitions and data structures
        │   └── utils.py                        # Engine utility functions
        │
        ├── runtime/                            # AgentRuntime class and execution logic
        │   ├── __init__.py
        │   ├── agent_runtime.py                # Main AgentRuntime class
        │   ├── file_processor.py               # File type detection & content decoding
        │   ├── tool_converter.py               # Tool configuration merging & provider conversion
        │   ├── metrics_extractor.py            # Token extraction & provider profile detection
        │   ├── structured_output_detector.py   # 4‑path detection & state writeback
        │   └── utils.py                        # Utility functions (safe_format, etc.)
        │
        ├── shared/                             # Shared utilities between engine and runtime
        │   ├── __init__.py                     # Shared utilities package exports
        │   ├── instrumentation.py              # Helper functions for metrics/instrumentation extraction
        │   ├── provider_capabilities.py        # Static advisory capability mapping (warnings/UI defaults)
        │   ├── runtime_hooks.py                # Runtime hooks for agent execution
        │   └── structured_schemas.py           # JSON schema resolution for structured output
        │
        ├── llm_client/                         # Modular LLM provider integration subpackage
        │   ├── __init__.py                     # LLM client package exports
        │   ├── chat_model_factory.py           # Factory for creating LangChain chat models
        │   ├── instrumentation.py              # LLM call instrumentation and metrics
        │   ├── provider_adapters.py            # Provider-specific adapter implementations
        │   ├── response_normalizer.py          # Response normalization across providers
        │   ├── structured_output.py            # Structured output configuration
        │   ├── tool_binder.py                  # Tool binding and invocation
        │   ├── wrappers.py                     # LLM client wrapper utilities
        │   │
        │   ├── core/                           # Modular LLMClient core implementation
        │   │   ├── __init__.py                 # Public API (LLMClient, TextResponse, etc.)
        │   │   ├── availability.py             # Conditional import flags and lazy references
        │   │   ├── agent_creation.py           # AgentCreationFacade for agent creation
        │   │   ├── request_builder.py          # RequestBuilder for constructing agent inputs
        │   │   ├── execution_engine.py         # ExecutionEngine for agent invocation with retries
        │   │   ├── response_processor.py       # ResponseProcessor for normalizing responses
        │   │   └── client.py                   # Main LLMClient class implementation
        │   │
        │   ├── multimodal/                     # Multimodal file handling
        │   │   ├── __init__.py                 # Multimodal package exports
        │   │   └── multimodal_handler.py       # File type detection and content processing
        │   │
        │   └── observability/                  # Langfuse integration for LLM tracing
        │       ├── __init__.py                 # Observability package exports
        │       └── langfuse_integration.py     # Langfuse tracing integration
        │
        ├── provider_data/                      # Dynamic provider capabilities & pricing data loading
        │   ├── __init__.py                     # Provider data package exports
        │   ├── cache.py                        # Caching for provider model data
        │   ├── downloader.py                   # Download external provider data
        │   ├── extractor.py                    # Extract and filter provider data
        │   ├── loader.py                       # Load provider data into memory
        │   └── schemas.py                      # Pydantic schemas for provider data
        │
        ├── tool_integration/                   # Tool registry and provider-specific tool adapter
        │   ├── __init__.py                     # Tool integration package exports
        │   ├── provider_tool_adapter.py        # Provider-specific tool calling adapter
        │   ├── registry.py                     # Tool registry and management
        │   ├── web_fetch_tool.py               # Web content fetching tool implementation
        │   │
        │   └── search/                         # Web search tools
        │       ├── __init__.py                 # Search tools package exports
        │       ├── duckduckgo_base.py          # Base DuckDuckGo search functionality
        │       └── duckduckgo_tool.py          # DuckDuckGo search tool implementation
        │
        ├── ui/                                 # Streamlit UI components
        │   ├── app.py                          # Main Streamlit application entry point
        │   ├── bootstrap.py                    # UI initialization and session state setup
        │   ├── agent_editor_mode.py            # Agent creation and editing interface
        │   ├── history_mode.py                 # Run history and results viewer
        │   ├── run_mode.py                     # Pipeline execution interface
        │   ├── cache.py                        # UI caching utilities for performance
        │   ├── exports.py                      # Data export functionality (JSON, CSV)
        │   ├── graph_view.py                   # Pipeline graph visualization component
        │   ├── logging_ui.py                   # Logging UI components and log viewer
        │   ├── metrics_view.py                 # Metrics display components and charts
        │   ├── styles.py                       # UI styling and themes (CSS, colors)
        │   ├── tools_view.py                   # Tools management UI and configuration
        │   ├── utils.py                        # UI utility functions and helpers
        │   ├── view_models.py                  # UI data models and view state
        │   ├── static/                         # Static assets (fonts)
        │   └── .streamlit/                     # Streamlit theme configuration        
        │    
        ├── observability/                      # Observability and tracing integrations
        │   ├── __init__.py                     # Observability package exports
        │   └── langfuse.py                     # Langfuse integration for distributed tracing
        │
        └── db/                                 # Database layer
            ├── __init__.py                     # Database package exports
            ├── agents.py                       # Agent DAO (Data Access Object)
            ├── db.py                           # Low-level DB connection and re‑exports
            ├── pipelines.py                    # Pipeline DAO
            ├── runs.py                         # Run DAO
            ├── services.py                     # High-level transactional APIs
            │
            └── infra/                          # Low-level DB infrastructure
                ├── __init__.py                 # DB infrastructure package exports
                ├── backup_utils.py             # Database backup utilities
                ├── cli_utils.py                # CLI utility functions for database operations
                ├── core.py                     # Core DB infrastructure and connection management
                ├── generate_migration.py       # Migration generation from schema changes
                ├── maintenance.py              # Database maintenance utilities
                ├── migration_meta.py           # Migration metadata management and tracking
                ├── migrations.py               # Migration application and version control
                ├── prune_snapshots.py          # Agent snapshot pruning and cleanup
                ├── schema.py                   # Canonical SQL schema definitions
                ├── schema_diff.py              # Schema comparison utilities for migrations
                ├── schema_diff_constraints.py  # Constraint comparison utilities
                ├── sql_utils.py                # SQL utility functions and helpers
                ├── sqlite_features.py          # SQLite feature detection and compatibility
                └── sqlite_rebuild.py           # SQLite database rebuild for destructive changes

data/                                   # Runtime data (created on first run)
├── db/                                 # SQLite database files (multi_agent_runs.db, etc.)
├── migrations/                         # Generated migration SQL files (20+ migrations)
├── provider_models/                    # Dynamic provider capabilities & pricing data
│   ├── local_ollama_models.json        # Local Ollama model customization (untracked)
│   ├── provider_models.json            # Provider Data for OpenAI & DeepSeek (untracked)
│   ├── provider_models_all.json        # Provider Data fetched (untracked)
│   └── template_ollama_models.json     # Template for local Ollama model customization
└── logs/                               # Application logs (rotating log files)

tests/                                  # Unit tests (pytest)

docs/                                   # Project documentation
├── ARCHITECTURE.md                     # System architecture overview
├── CONFIG.md                           # Configuration reference and YAML format
├── DEVELOPMENT.md                      # Developer guide and workflow
├── INSTALL.md                          # Installation instructions
├── MIGRATIONS.md                       # Database migration guide
├── TROUBLESHOOTING.md                  # Troubleshooting common issues
├── USAGE.md                            # User guide and features
├── implementation-strategies/          # Implementation strategy documents
└── archive/                            # Archived planning documents

config/                                 # Centralized YAML‑based configuration
├── agents.yaml                         # Agent limits and snapshot settings
├── logging.yaml                        # Default log level configuration
├── paths.yaml                          # Directory and file names
├── providers.yaml                      # Provider‑data file names and URLs
└── ui.yaml                             # UI colors and attachment file types

scripts/                                # Utility scripts
├── quick_start.sh                      # Quick start script for Linux/macOS (venv setup)
├── quick_start.ps1                     # Quick start script for Windows PowerShell
└── verification/                       # Verification scripts for development

tools/                                  # Development tools
└── annotate_old_migrations.py          # Migration annotation utility

.github/                                # GitHub configuration
└── ISSUE_TEMPLATE/
    └── quickstart_feedback.md          # Issue template for quickstart feedback
```

> 📝 Note: The `data/` directory and its contents are typically created automatically at runtime. The exact set of migration files will evolve over time; see `data/migrations/` in your clone.

## APPENDIX A: Architecture Diagrams

This appendix provides visual representations of the Multi‑Agent Dashboard architecture using Mermaid diagrams. These diagrams complement the textual descriptions in the main document and help visualize layer interactions, data flows, and interface contracts.

### Diagram 1: Basic Layered Architecture

```mermaid
flowchart TB

subgraph UI["UI Layer"]
    Streamlit["Streamlit UI"]
end

subgraph ServiceLayer["Service Layer"]
    Services["RunService • AgentService • PipelineService"]
end

subgraph PersistenceLayer["Persistence Layer"]
    DAO["Data Access Objects"]
    DBInfraCore["Schema • Migrations • DB Core"]
    SQLite[(SQLite)]
end

subgraph EngineRuntimeLayer["Engine & Runtime Layer"]
    EngineCore["MultiAgentEngine"]
    RuntimeCore["AgentRuntime"]
end

subgraph LLMClientLayer["LLM Client Layer"]
    LLMClientCore["LLM Client"]
end

subgraph Shared["Shared Utilities"]
    SharedUtils["Config • Instrumentation • Hooks"]
end

subgraph ModelsLayer["Models & Result DTOs"]
    ModelsCore["AgentSpec • PipelineSpec • Results"]
end

UI --> ServiceLayer
ServiceLayer ---> PersistenceLayer
DAO --> DBInfraCore --> SQLite
ServiceLayer --> EngineRuntimeLayer
EngineCore --> RuntimeCore
EngineRuntimeLayer --> LLMClientLayer
Shared -.-> EngineRuntimeLayer
LLMClientLayer -.-> ModelsLayer
EngineRuntimeLayer -.-> ModelsLayer
```

*Shows the seven core layers and their directional dependencies. Dashed lines indicate advisory/utility relationships.*

### Diagram 2: Compact Flow Chart

```mermaid
flowchart TB
User["User starts pipeline"] --> Engine["MultiAgentEngine.run_seq()"]
Engine --> Runtime["AgentRuntime.run()"]
Runtime --> LLMClient["LLMClient.invoke_agent()"]
LLMClient --> Provider["Provider-specific adapter"]
Provider --> LangChain["LangChain init_chat_model"]
LangChain --> LLM["LLM API call (tools if configured, Langfuse)"]
LLM --> Normalize["Normalize response"]
Normalize --> State["Update pipeline state"]
State --> Next["Next agent step"]
Next --> Engine
Engine --> Result["Engine aggregates results"]
Result --> Save["RunService.save_run()"]
Save --> UI["UI displays results"]
```

*High‑level flow of a pipeline execution from user action to result display.*

### Diagram 3: Compact Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant U as UI
    participant S as RunService
    participant E as Engine
    participant R as Runtime
    participant L as LLM Client
    participant P as Provider
    
    U->>E: run_seq(steps, initial_input)
    loop For each agent
        E->>R: run(state)
        R->>L: invoke_agent(agent, prompt)
        L->>P: LLM API call with tools/schema (Langfuse)
        P-->>L: Raw response
        L-->>R: TextResponse
        R-->>E: AgentRunResult
    end
    E-->>U: EngineResult
    U->>S: save_run(task_input, final_output, memory, ...)
    S-->>U: Run ID
```

*Simplified sequence of method calls across layers during a pipeline run.*

### Diagram 4: Compact Interface Spec

```mermaid
classDiagram
    class RunService {
        +save_run(task_input, final_output, memory, **kwargs) int
        +list_runs() List[dict]
        +get_run_details(run_id) tuple
    }

    class MultiAgentEngine {
        +run_seq(steps, initial_input, **kwargs) EngineResult
    }
    
    class AgentRuntime {
        +run(state, **kwargs) str
    }
    
    class LLMClient {
        +invoke_agent(agent, prompt, **kwargs) TextResponse
    }
    
    class EngineResult {
        +final_output: Any
        +state: dict
        +memory: dict
        +agent_metrics: dict
        +total_cost: float
    }
    
    class TextResponse {
        +text: str
        +input_tokens: int
        +output_tokens: int
        +latency: float
    }
    
    RunService --> MultiAgentEngine
    MultiAgentEngine --> AgentRuntime
    AgentRuntime --> LLMClient
    MultiAgentEngine ..> EngineResult
    LLMClient ..> TextResponse
```

*Core interface contracts between Service, Engine, Runtime, and LLM Client layers.*


## APPENDIX B: Extended Architecture Diagrams

This appendix contains extended versions of the architecture diagrams for readers who want more detail. The compact versions remain in APPENDIX A for quick reference.

### Diagram 5: Extended Layered Architecture

```mermaid
graph TD
    subgraph UILayer["UI"]
        UI["Streamlit UI<br/>ui/app.py"]
        UIModes["Run Mode • Agent Editor • History Mode"]
    end
    
    subgraph ServiceLayer["Services"]
        RunService[RunService]
        AgentService[AgentService]
        PipelineService[PipelineService]
    end
    
    subgraph PersistenceLayer["Persistence"]
        subgraph DAOs["Data Access Objects"]
            AgentsDAO[AgentDAO]
            RunsDAO[RunDAO]
            PipelinesDAO[PipelineDAO]
        end
        subgraph DBInfraLayer["DB Infrastructure"]
            Schema["Schema & Migrations<br/>db/infra/schema.py"]
            Core["Core DB<br/>db/infra/core.py"]
        end
    end
    
    SQLite[(SQLite)]

    
    subgraph EngineLayer["Engine"]
        MultiAgentEngine["MultiAgentEngine.run_seq()"]
        Orchestrator[Orchestrator]
        StateManager[StateManager]
    end
    
    subgraph RuntimeLayer["Runtime"]
        AgentRuntime["AgentRuntime.run()"]
        FileProcessor[FileProcessor]
        ToolConverter[ToolConverter]
    end
    
    subgraph LLMClientLayer["LLM Client"]
        LLMClient["LLMClient.invoke_agent()"]
        ProviderAdapters[Provider Adapters]
        ToolIntegration[Tool Integration]
    end
    
    subgraph ModelsResultsLayer["Models & Result DTOs"]
        AgentSpec[AgentSpec]
        PipelineSpec[PipelineSpec]
        EngineResult[EngineResult]
    end
    
    subgraph SharedLayer["Shared Utilities"]
        SharedUtils["Config • Instrumentation • Hooks"]
    end

    subgraph Observability["Observability"]
        Langfuse[Langfuse]
    end
    
    UILayer --> ServiceLayer
    UI --> UIModes
    RunService --> RunsDAO
    RunService --> MultiAgentEngine
    
    AgentService --> AgentsDAO
    PipelineService --> PipelinesDAO
    PersistenceLayer --> SQLite
    MultiAgentEngine --> AgentRuntime
    AgentRuntime --> LLMClient
    LLMClient --> ProviderAdapters
    LLMClient --> ToolIntegration
    SharedUtils -.-> EngineLayer
    SharedUtils -.-> RuntimeLayer
    SharedUtils -.-> LLMClientLayer
    Observability -.-> LLMClient
    AgentSpec -.-> MultiAgentEngine
    PipelineSpec -.-> MultiAgentEngine
    EngineResult -.-> RunService
```

*Expands each layer into its key components and shows concrete class dependencies. Dashed lines indicate configuration/data‑flow relationships.*

### Diagram 6: Extended Flow Chart

```mermaid
flowchart TD
    A["UI: User triggers pipeline run<br/>ui/run_mode.py"] --> C["Engine: MultiAgentEngine.run_seq()<br/>engine/engine_orchestrator.py"]
    C --> D[For each agent step]
    D --> E["Runtime: AgentRuntime.run()<br/>runtime/agent_runtime.py"]
    E --> F{Has files?}
    F -->|Yes| G["FileProcessor.decode_files()"]
    F -->|No| H[Build prompt]
    G --> H
    H -->|tools if configured| I["LLM Client: LLMClient.invoke_agent()<br/>llm_client/core/client.py"]
    I --> J{Structured output?}
    J -->|Yes| K[Apply JSON schema]
    J -->|No| L[Standard call]
    K --> M[Provider adapter builds request with tools/schema]
    L --> M
    M --> N[LangChain init_chat_model]
    N --> O["LLM API call with instrumentation (Langfuse)"]
    O --> P[Extract tokens, latency, cost]
    P --> Q[Response normalized to TextResponse]
    Q --> R["Runtime: detect structured output"]
    R --> S[Write back to state]
    S --> T["Engine: update PipelineState"]
    T --> D
    D --> U[All steps complete]
    U --> V[Engine returns EngineResult]
    V --> W["Service: RunService.save_run()<br/>db/services.py"]
    W --> X[UI displays results & metrics]
```

*Detailed flow including file processing, structured‑output detection, and instrumentation. Decision points reflect runtime adaptations.*

### Diagram 7: Extended Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant U as UI (run_mode.py)
    participant S as RunService
    participant D as RunDAO
    participant E as MultiAgentEngine
    participant R as AgentRuntime
    participant L as LLMClient
    participant PA as Provider Adapter
    participant LC as LangChain
    participant LLM as LLM API
    participant M as Metrics
    participant LF as Langfuse
    
    U->>E: run_seq(steps, initial_input, **kwargs)
    
    loop For each agent step
        E->>R: run(state, **kwargs)
        R->>L: invoke_agent(agent, prompt, files, tools, structured_output)
        L->>PA: build_request_with_files()
        PA->>LC: init_chat_model()
        LC->>LLM: API call with tools/schema
        LC->>LF: trace invocation (async)
        LLM-->>LC: Raw response
        LC-->>PA: LangChain message
        PA-->>L: Normalized message
        L->>M: extract_tokens_latency()
        M-->>L: tokens, latency, cost
        L-->>R: TextResponse(content, tokens, latency, cost)
        R->>R: detect_structured_output()
        R-->>E: AgentRunResult(output, metrics, state_updates)
        E->>E: update PipelineState
    end
    
    E-->>U: EngineResult(final_output, state, memory, metrics)
    U->>S: save_run(task_input, final_output, memory, **kwargs)
    S->>D: save(task_input, final_output, memory, ...)
    D-->>S: run_id
    S-->>U: run_id
```

*Detailed sequence showing database interactions, provider‑adapter delegation, and metrics extraction. Reflects the actual code flow.*

### Diagram 8: Extended Interface Spec

```mermaid
classDiagram
    class RunService {
        +save_run(task_input: str, final_output: str, memory: dict, **kwargs) int
        +list_runs() List[dict]
        +get_run_details(run_id: int) tuple
    }
    
    class MultiAgentEngine {
        +run_seq(steps: List[str], pipeline_name: Optional[str] = None, initial_input: Any, strict: bool = False, strict_schema_validation: bool = False, last_agent: Optional[str] = None, files: Optional[List[Dict[str, Any]]] = None, allowed_domains: Optional[Any] = None) EngineResult
    }
    
    class AgentRuntime {
        +run(state: Dict[str, Any], *, files: Optional[List[Dict[str, Any]]] = None, structured_schema: Optional[Dict[str, Any]] = None, stream: bool = False) str
    }
    
    class LLMClient {
        +invoke_agent(agent: Any, prompt: str, *, files: Optional[List[Dict[str, Any]]] = None, response_format: Optional[Dict[str, Any]] = None, stream: bool = False, context: Optional[Dict[str, Any]] = None) TextResponse
    }
    
    class EngineResult {
        +final_output: Any
        +state: Dict[str, Any]
        +memory: Dict[str, Any]
        +warnings: List[str]
        +errors: List[str]
        +final_agent: Optional[str]
        +agent_metrics: Dict[str, Dict[str, Any]]
        +agent_configs: Dict[str, Dict[str, Any]]
        +total_cost: float
        +total_latency: float
        +total_input_cost: float
        +total_output_cost: float
        +tool_usages: Dict[str, List[Dict[str, Any]]]
        +strict_schema_exit: bool
        +agent_schema_validation_failed: Dict[str, bool]
    }
    
    class TextResponse {
        +text: str
        +raw: Dict[str, Any]
        +input_tokens: Optional[int]
        +output_tokens: Optional[int]
        +latency: Optional[float]
    }
    
    class AgentSpec {
        +name: str
        +model: str
        +prompt_template: str
        +role: str
        +input_vars: List[str]
        +output_vars: List[str]
        +color: Optional[str]
        +symbol: Optional[str]
        +tools: Dict[str, Any]
        +reasoning_effort: Optional[str]
        +reasoning_summary: Optional[str]
        +system_prompt_template: Optional[str]
        +provider_id: Optional[str]
        +model_class: Optional[str]
        +endpoint: Optional[str]
        +use_responses_api: bool
        +provider_features: Dict[str, Any]
        +structured_output_enabled: bool
        +schema_json: Optional[str]
        +schema_name: Optional[str]
        +temperature: Optional[float]
        +max_output: int
    }
    
    class PipelineSpec {
        +name: str
        +steps: List[str]
        +metadata: Dict[str, Any]
    }
    
    class PipelineState {
        +pipeline_name: Optional[str]
        +state: Dict[str, Any]
        +memory: Dict[str, Any]
        +warnings: List[str]
        +errors: List[str]
        +tool_usages: Dict[str, List[Dict[str, Any]]]
        +agent_configs: Dict[str, Dict[str, Any]]
        +strict_schema_exit: bool
        +agent_schema_validation_failed: Dict[str, bool]
        +agent_metrics: Dict[str, RunMetrics]
    }
    
    class AgentRunResult {
        +raw_output: Any
        +metrics: RunMetrics
        +parsed: Optional[Any]
        +tool_usages: List[Dict[str, Any]]
        +config_snapshot: Dict[str, Any]
        +cost_breakdown: Dict[str, float]
    }
    
    class RunMetrics {
        +input_tokens: Optional[int]
        +output_tokens: Optional[int]
        +latency: Optional[float]
        +input_cost: Optional[float]
        +output_cost: Optional[float]
        +total_cost: Optional[float]
        +model: Optional[str]
    }
    
    RunService --> MultiAgentEngine
    MultiAgentEngine --> AgentRuntime
    AgentRuntime --> LLMClient
    AgentRuntime --> AgentSpec
    MultiAgentEngine ..> EngineResult
    LLMClient ..> TextResponse
    MultiAgentEngine ..> PipelineState
    MultiAgentEngine ..> AgentRunResult
    AgentRunResult --> RunMetrics
    PipelineState ..> RunMetrics
    MultiAgentEngine ..> AgentSpec
    MultiAgentEngine ..> PipelineSpec
```

*Complete interface specification with data‑transfer objects (DTOs) and their fields. Shows the full type signature of cross‑layer calls. Note: `AgentRuntime` already holds an `AgentSpec` instance; `LLMClient.invoke_agent`'s `agent` parameter is a LangChain agent instance, not an `AgentSpec`. The signatures reflect the actual code interfaces.*

### Diagram 9: Package Dependency graph

```mermaid
flowchart LR
    db --> config
    db --> shared
    engine --> config
    engine --> llm_client
    engine --> models
    engine --> provider_data
    engine --> runtime
    engine --> shared
    llm_client --> models
    llm_client --> observability
    llm_client --> shared
    llm_client --> tool_integration
    models --> db
    models --> engine
    models --> runtime
    observability --> config
    provider_data --> config
    runtime --> config
    runtime --> llm_client
    runtime --> tool_integration
    shared --> provider_data
    tool_integration --> shared
    ui --> config
    ui --> db
    ui --> engine
    ui --> llm_client
    ui --> models
    ui --> shared
```

*Auto-generated diagram showing true architectural relationships for all packages*