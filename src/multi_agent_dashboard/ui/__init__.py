"""
Streamlit UI components for the Multi-Agent Dashboard.

This package provides the complete web interface for managing and running
multi-agent LLM pipelines. It includes application modes, caching, view models,
and visualization components built on Streamlit.

Key application modes:
- `app.py`: Main application entry point with mode routing
- `run_mode.py`: Pipeline execution interface with file uploads and real-time metrics
- `agent_editor_mode.py`: Agent configuration editor with snapshot management
- `history_mode.py`: Run history viewer with detailed metrics and tool usage analysis

Supporting components:
- `bootstrap.py`: Application initialization and engine setup
- `cache.py`: Data caching and service management
- `view_models.py`: Data structures for UI presentation
- `metrics_view.py`: Cost and latency visualization components
- `tools_view.py`: Tool configuration and usage analysis
- `graph_view.py`: Pipeline visualization with Graphviz
- `logging_ui.py`: Real-time log viewer with color-coded levels
- `exports.py`: Data export functionality for runs and pipelines
- `styles.py`: CSS styling and UI theme components
- `utils.py`: Shared utility functions for UI operations

The UI package works with the engine and runtime packages to provide a complete
dashboard experience while maintaining separation from business logic.
"""