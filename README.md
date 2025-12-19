# 🤖 Multi-Agent Dashboard

## 📖 Description

**Multi-Agent Dashboard** is a Streamlit-based Python application for building, managing, and running multi-agent pipelines with persistent storage. It provides a UI for configuring agents, executing pipelines, inspecting outputs, comparing results, and reviewing historical runs stored in SQLite.

The project is structured as a **standard Python package using a `src/` layout**, ensuring clean imports, testability, and long-term maintainability.

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd <repo-root>
```

---

### 2️⃣ Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

---

### 3️⃣ Install the project in editable mode

This step is **required** when using a `src/` layout.

```bash
pip install -e .
```

This makes the `multi_agent` package importable in Streamlit, tests, and scripts.

---

### 4️⃣ Configure environment variables

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

Optional:

```text
LOG_LEVEL=INFO
```

---

### 5️⃣ Run the dashboard

From the project root:

```bash
streamlit run src/multi_agent/ui/app.py
```

Then open your browser at:

👉 [http://localhost:8501](http://localhost:8501)

---

## 🧭 Usage

* 🎛️ Create, edit, duplicate, and delete agents via the UI
* 🔗 Build and execute multi-agent pipelines
* 👀 Inspect agent outputs and intermediate state
* 🔍 Compare outputs between agents
* 🕒 Review historical runs stored in SQLite
* 📦 Export runs and pipeline definitions as JSON

---

## 🛠️ Tech Stack

* **Python** – core application logic
* **Streamlit** – web-based UI
* **SQLite** – persistent storage for agents, prompts, and runs
* **OpenAI API** – LLM execution via an abstracted client
* **Graphviz** – agent pipeline visualization

---

## ✨ Features

* 🤖 Multi-agent pipeline execution with a unified dashboard UI
* 🧩 Dynamic agent configuration (model, role, inputs, outputs)
* 💾 SQLite-backed persistence for agents, prompt versions, and runs
* 🕒 Versioned agent prompts with history and inspection
* 🗺️ Agent graph visualization for pipeline flow clarity
* 🧪 Output comparison and diff tooling
* 🧱 Centralized SQLite migration system
* 🔌 Decoupled LLM client abstraction with retries and normalization
* 📤 JSON export for pipelines and historical runs
* 🧠 Strict vs permissive execution modes

---

## 🗂️ Repository Structure

```
repo_root/
├── .env                         # Environment variables (API keys, log level; not committed)
├── .gitignore                   # Ignore sensitive/generated files
├── LICENSE                      # Project license
├── pyproject.toml               # Project metadata, dependencies, packaging config
├── README.md                    # Project documentation
├── src/
│   └── multi_agent/             # Main Python package (src layout)
│       ├── __init__.py          # Package marker
│       ├── config.py            # Global configuration, paths, logging setup
│       ├── engine.py            # Core multi-agent orchestration engine (UI-agnostic)
│       ├── llm_client.py        # LLM client abstraction (retries, normalization, backoff)
│       ├── models.py            # Domain models (AgentSpec, AgentRuntime, PipelineSpec)
│       ├── utils.py             # Shared utilities (safe prompt formatting, helpers)
│       ├── db/
│       │   ├── __init__.py      # Database subpackage
│       │   └── db.py            # SQLite access layer (CRUD, migrations, persistence)
│       └── ui/
│           ├── __init__.py      # UI subpackage
│           └── app.py           # Streamlit application (presentation + orchestration only)
├── data/
│   ├── db/
│   │   └── multi_agent_runs.db  # Auto-created SQLite database (not tracked)
│   └── migrations/
│       ├── 000_create_base_tables.sql          # Initial schema
│       ├── 001_add_agent_output_metadata.sql   # Schema evolution
│       ├── 002_add_runs_metadata.sql
│       ├── 003_normalize_agent_json.sql
│       └── ...                                 # Future migrations
└── tests/                      # Tests (optional / future expansion)
```

---

## 🧪 Development Notes

* The project uses a **`src/` layout** to avoid accidental imports from the working directory.
* Always install the project with `pip install -e .` during development.
* UI code is isolated under `multi_agent/ui`.
* Core logic (engine, models, DB) is UI-agnostic and safe for reuse in CLI tools or tests.

---

## 🤝 Contributing

Contributions are welcome.

Please:

* Keep UI changes confined to `multi_agent/ui`
* Avoid introducing `sys.path` manipulation
* Include database migrations for schema changes
* Preserve engine/UI separation

---

## 📄 License

This project is licensed under the terms described in the `LICENSE` file.

---

## 📝 Project History

The project evolved from a single-file Streamlit dashboard into a modular, package-based architecture featuring:

* A decoupled multi-agent execution engine
* Versioned prompt management
* Persistent execution history
* A clean `src/`-based layout for long-term maintainability