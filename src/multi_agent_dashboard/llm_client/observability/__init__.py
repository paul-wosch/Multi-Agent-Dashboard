"""
Observability integration sub-package for LLM client.

This sub-package provides observability and tracing integration for the LLM client,
primarily through Langfuse for distributed tracing, latency monitoring, and
cost tracking. It enables comprehensive monitoring of LLM interactions across
the multi-agent dashboard.

Key components:
- Langfuse integration for distributed tracing
- Invocation configuration building for observability
- Context propagation for pipeline and run tracking

The observability layer is designed to be transparent to the LLM client
while providing detailed insights into agent execution performance and costs.
"""