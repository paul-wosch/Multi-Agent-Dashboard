# 🤖 Multi-Agent-Dashboard

## 📖 Description

Multi-Agent-Dashboard is a Streamlit-based, Python dashboard for building, managing, and running multi-agent pipelines with persistent storage. It provides a UI for configuring agents, executing pipelines, inspecting outputs, comparing results, and reviewing historical runs stored in SQLite.

## 🚀 Getting Started

1. Clone the repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables by creating a `.env` file with your OpenAI API key:

   ```text
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the dashboard:

   ```bash
   streamlit run dashboard.py
   ```
5. Open your browser and view the app at:
   [http://localhost:8501](http://localhost:8501)

## 🧭 Usage

* 🎛️ Use the dashboard UI to create, edit, duplicate, or delete agents.
* 🔗 Build and execute multi-agent pipelines.
* 👀 View agent outputs, compare results, and inspect historical runs.
* 📦 Download past runs as JSON for offline analysis.

## 🛠️ Tech Stack

* **Python** – core application logic
* **Streamlit** – web-based dashboard UI
* **SQLite** – persistent storage for agents, prompts, and runs
* **OpenAI API** – large language model execution via an abstracted client

## ✨ Features

* 🤖 Multi-agent pipeline execution with a unified dashboard UI.
* 🧩 Dynamic agent configuration with editable metadata, inputs, and outputs.
* 💾 SQLite-backed persistence for agents, prompt versions, and execution runs.
* 🕒 Versioned agent prompts with history, revert, and diff tooling.
* 🗺️ Agent graph visualization for pipeline flow clarity.
* 🧪 Output comparison tools and code-view rendering for easy copying.
* 🧱 Centralized SQLite migration system with ordered migrations and dry-run support.
* 🔌 Decoupled LLM client abstraction for normalized responses, retries, and rate-limit handling.
* 📤 JSON export of historical runs.

## 🗂️ Repository Structure

```
.
├── .gitignore                # Ignore sensitive/generated files
├── LICENSE
├── README.md
├── config.py                 # Global configuration and constants
├── dashboard.py              # Main dashboard application
├── data
│   ├── db
│   │   └── multi_agent_runs.db   # Auto-created SQLite database (not tracked)
│   └── migrations
│       └── 000_create_base_tables.sql
├── db
│   ├── db.py                 # Database connection helpers
│   ├── generate_migration.py # Migration generator
│   ├── migrations.py         # Migration application logic
│   ├── schema.py             # Current schema definitions
│   └── schema_diff.py        # Schema diff utilities
├── llm_client.py             # LLM client abstraction
└── requirements.txt          # Project dependencies
```

## 🤝 Contributing

Contributions are welcome. Please keep changes focused, follow the existing project structure, and ensure database or schema changes are accompanied by appropriate migrations.

## 📄 License

This project is licensed under the terms described in the `LICENSE` file.

## 📝 GITLOG Summary

The project evolved from an initial dashboard scaffold into a full-featured multi-agent system with SQLite persistence, versioned prompts, CRUD management, visualization tools, and a robust migration system. Recent updates focused on UI refinements, improved agent editing workflows, restored run exports, code-view outputs, and architectural refactors to decouple the LLM client and harden database handling.
