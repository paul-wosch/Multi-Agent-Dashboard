# 📘 Usage (expanded)

This document expands the brief usage summary in the [README](../README.md) with a comprehensive overview of the Streamlit UI modes and their capabilities. For technical architecture, configuration details, and provider integration, refer to the linked documentation.

🛠 **Configuration**: The dashboard uses YAML‑based configuration (`config/` directory) and environment variables (`.env`) for API keys, agent limits, and UI settings. See [docs/CONFIG.md](CONFIG.md) for the complete reference.

🔌 **Provider Integration**: The UI supports multiple LLM providers (OpenAI, DeepSeek, Ollama) with dynamically loaded capability data. Model selection, tool configuration, and structured output options adapt based on provider‑specific features—see [docs/ARCHITECTURE.md](ARCHITECTURE.md) for technical details.

👁️‍🗨 **Observability**: Built‑in logging, cost tracking, and historical run storage are always active. Optional [Langfuse](https://langfuse.com) integration provides advanced distributed tracing, latency breakdown, and enhanced visualization—see [docs/OBSERVABILITY.md](OBSERVABILITY.md) for setup.

---

## 🚀 Run Pipeline

Execute multi‑agent pipelines with real‑time monitoring and rich result inspection.

### 🏗️ Pipeline Construction
- **Named Pipelines** – load saved pipeline definitions from the database  
- **Ad‑hoc Pipelines** – build pipelines on‑the‑fly by selecting agents in execution order  
- **Agent Selection** – visual picker with agent‑specific colors and symbols  

### 📎 File Attachments
- Upload files (images, PDFs, text) to augment agent inputs  
- Automatic MIME‑type detection and size enforcement  
- Provider‑specific multimodal handling for vision‑capable models  

### 🎛️ Execution Control
- **Web Search Domain Filtering** – restrict allowed domains for web search tools  
- **Strict Mode** – enforce structured output validation and early exit on schema failure  
- **Real‑time Progress** – progress bar, elapsed timer, and per‑agent status updates  

### 🔍 Results Inspection (Tabbed Interface)
- **Final Output** – consolidated pipeline output with Code/Markdown rendering toggle  
- **Warnings** – execution warnings, contract violations, and provider‑specific errors  
- **Agent Outputs** – inspection of each agent's output
- **Graph** – Graphviz visualization of the pipeline flow with cost and latency annotations  
- **Compare** – unified diff between agent outputs for quick comparison  
- **Cost & Latency** – token usage, latency breakdown, and provider‑cost computation per agent  
- **Tools & Advanced** – tool‑call details, domain filtering analysis, and configuration snapshots  

### 📤 Export
- Export pipeline definitions and associated agent configurations as JSON  
- Download full run results (including metrics, tool usage, and agent configurations) for offline analysis  

---

## 🤖 Manage Agents

Create, edit, and configure reusable agent specifications with versioned snapshots.

### 🗂️ Tabbed Editor
- **Basics** – agent name, model, role, provider selection, endpoint customization and visual customization (color and symbol)
- **Prompt** – system and user prompt templates with variable substitution  
- **Inputs/Outputs** – input / output variables
- **Advanced** – tool configuration, structured output schemas, reasoning behavior, temperature, max output tokens
- **Snapshots** – versioned prompt history with manual/auto‑snapshot creation and rollback  

### ⚙️ Provider Configuration
- Support for OpenAI, DeepSeek, Ollama
- Provider‑specific settings
- Ollama endpoint health checking and reachability verification  

### 🛠️ Tool Integration
- Enable/disable tool calling per agent  
- Select specific tool types (web_search, web_search_ddg, web_fetch) from available options

### 📸 Snapshot Management
- Automatic or manual snapshot creation on agent save  
- Browse historical prompt versions and configurations  
- Rollback to any previous snapshot with one click  

### 🔄 Import / Duplicate
- Import agent configurations from JSON templates
- Duplicate agents with a single action  

---

## 🕒 History

Browse, analyze, and export historical pipeline executions with complete runtime snapshots.

### 📝 Run Selection
- Chronological list of past runs with abbreviated task descriptions and agent names

### 🔍 Run Inspection
- **Task Input** – original task description  
- **Final Output** – pipeline result with Code/Markdown rendering toggle  
- **Per‑Agent Outputs** – individual agent outputs, models, and JSON flags  
- **Strict Schema Exit Indicators** – visual badge for runs that exited early due to validation failure  

### 📈 Metrics & Cost Analysis
- **Cost & Latency Summary** – total cost, input/output token costs, and total latency  
- **Per‑Agent Breakdown** – token counts, provider costs, and latency per agent  
- **Stored Metrics** – historical cost computation using provider pricing data at execution time  

### 🔧 Tool Usage Analysis
- **Configuration Snapshots** – agent configurations as they existed at run time  
- **Tool Call Details** – per‑call arguments, domains, and results  
- **Tool Usage Overview** – tabular summary of tool calls across all agents  

### 📤 Export
- Export full run records as JSON, including agent configurations, metrics, tool usage, and agent configurations  
- Download structured data for offline analysis or integration with external tools  

---

## 📜 Logs

Monitor real‑time application logs and agent execution traces with search and filtering.

### 📝 Real‑time Log Capture
- Custom Streamlit log handler captures application logs in session state  
- Logs appear immediately in the UI as they are emitted  
- Rolling buffer (default 500 lines) prevents memory exhaustion  

### 🖥️ Log Viewer Features
- **Color‑coded Levels** – DEBUG, INFO, WARNING, ERROR, CRITICAL with distinct colors  
- **Level Filtering** – show/hide log levels with checkbox controls  
- **Text Search** – filter logs by message content  
- **Live Updates** – automatic refresh as new logs arrive  
- **Export** – download filtered logs as a plain text file  
- **Clear Buffer** – remove all logs from the in‑memory buffer  

### 🗄️ Historic Log Loading
- On startup, loads existing logs from the application log file at `data/logs/application.log` for continuity  
- Combines real‑time capture with persistent log file contents  

### 🔗 Integration
- Logs are written via centralized configuration (`config/logging.yaml`) to both stdout and the rotating log file  
- Optional Langfuse tracing augments built‑in logs with detailed LLM call and tool invocation traces  

---

## 🧷 Cross‑cutting Features

### 🗺️ Graph Visualization
- Pipeline graphs rendered with Graphviz, showing agent flow and state transitions  
- Nodes styled with agent‑specific colors and symbols  
- Annotated with cost and latency metrics on nodes and edges  
- Available in Run Pipeline mode for visualizing active pipeline execution  

### 📊 Metrics Display
- Consistent cost and latency visualization across live and historical runs  
- Currency formatting, latency formatting, and null‑value handling  
- Per‑agent breakdown tables with token counts and provider costs  

### 🔬 Tool Configuration Analysis
- Detailed panels showing agent tool settings and web search domain filters  
- Per‑call tool usage with arguments and domain filtering analysis  
- Unified presentation for both current run results and historical data  

### 📦 Export & Serialization
- JSON export of pipeline definitions, agent configurations, and run results  
- Structured export formats that mirror historical run storage

---

## 📚 Further Reading

- [docs/ARCHITECTURE.md](ARCHITECTURE.md) – system architecture and engine design
- [docs/CONFIG.md](CONFIG.md) – complete configuration reference
- [docs/OBSERVABILITY.md](OBSERVABILITY.md) – Langfuse integration and advanced tracing
- [README.md](../README.md) – quick start and project overview