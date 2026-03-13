#!/usr/bin/env python3
# Quick test script (run in Python interpreter)
from multi_agent_dashboard.llm_client.core.response_processor import ResponseProcessor

# Test multi‑message accumulation
test_messages = [
    {"type": "ai", "usage_metadata": {"input_tokens": 100, "output_tokens": 50}},
    {"type": "human", "content": "Hello"},
    {"type": "ai", "usage_metadata": {"input_tokens": 200, "output_tokens": 150}},
]
result = ResponseProcessor.extract_usage_from_messages(test_messages)
print(f"Accumulated: {result}")  # Should be {"input_tokens": 300, "output_tokens": 200}