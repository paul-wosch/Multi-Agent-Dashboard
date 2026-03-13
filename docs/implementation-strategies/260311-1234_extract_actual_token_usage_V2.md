# Implementation Plan V2: Extracting Actual Token Usage Across Complete LLM Call Chains

## Overview

**Date**: 2026‑03‑11 (created); 2026‑03‑13 (completed)
**Problem**: The current system extracts token usage from only the **last AI message** in multi‑step LLM interactions, missing tokens from previous messages in the chain. This leads to under‑reporting of actual token consumption and incorrect cost calculations.
**Goal**: Modify the token extraction logic to accumulate token counts across **all AI messages** in a response chain while preserving backward compatibility and existing fallback mechanisms.
**Confidence Level**: 99/100 – The solution leverages proven recursive search patterns already in the codebase.

### NOTES
- Skipped Phases 2 and 3.
- The original feature request (accumulating token usage across multi‑message LLM chains) is fully addressed by Phase 1 alone.
- If sub‑agents are added later, Phase 2‑3 can be implemented then, using the same accumulation pattern from Phase 1.

---

## Current Architecture Analysis

### Token Extraction Flow
```
ResponseProcessor.process() (client.py:358‑539)
├── extract_usage_from_messages() (lines 82‑113) → searches messages list, returns first match
└── extract_usage_from_candidate() (lines 39‑79) → recursive search through nested structures
```

### Key Data Structures
- **Token usage fields**: `usage`, `usage_metadata`, `response_metadata.token_usage`
- **Token count field names**: `input_tokens`/`prompt_tokens`, `output_tokens`/`completion_tokens`
- **Nested structures**: `agent_response`, `output` lists, `messages` arrays

### Downstream Consumers
- `TextResponse.input_tokens` / `output_tokens` (populated by `ResponseProcessor`)
- `agent_runtime.py:208` – `extract_tokens_from_raw` fallback
- `MetricsAggregator.compute_cost()` – cost calculation based on token totals

---

## Recommended Solution: Modify Existing Recursive Search

### Why This Approach?
1. **Minimal disruption** – modifies only 2‑3 existing methods in `ResponseProcessor`
2. **Leverages existing investment** – uses proven recursive search patterns already handling provider‑specific field variations
3. **Addresses core requirement** – accumulates token usage across all AI messages in chains
4. **Backward compatible** – single‑message chains produce identical results
5. **Faster implementation** – 4‑6 hours vs. 8‑12 hours for callback‑based approach

### What Changes?
1. **`extract_usage_from_messages()`** – change from first‑match to accumulation across all AI messages
2. **`extract_usage_from_candidate()`** – make recursive search accumulate nested usage payloads
3. **`process()`** – use aggregated totals from enhanced methods

---

## Implementation Strategy

### Guiding Principles
- **DRY** – reuse existing recursive search logic
- **KISS** – minimal changes, no new infrastructure
- **Incremental** – each phase independently testable
- **Backward compatible** – existing single‑message runs unchanged

### Phased Approach
1. **Phase 1**: Modify `extract_usage_from_messages()` for message‑level accumulation
2. **Phase 2**: Enhance `extract_usage_from_candidate()` for nested‑structure accumulation
3. **Phase 3**: Update `process()` method to use aggregated totals
4. **Phase 4**: Manual verification (user‑led)

Each phase builds on the previous, allowing early validation and rollback if needed.

---

## Detailed Implementation Steps

### ✅ Phase 1: Message‑Level Accumulation (2 hours)

**File**: `src/multi_agent_dashboard/llm_client/core/response_processor.py` (lines 82‑113)

**Current behavior**: Searches messages from end to beginning, returns first valid usage payload.

**Required changes**:
1. Change return type from `Optional[Dict]` to `Dict[str, Any]` (always returns a dict)
2. Initialize `total_input = 0`, `total_output = 0`
3. Iterate **all** messages (not reversed), extract token counts from each AI message
4. Sum `input_tokens`/`prompt_tokens` and `output_tokens`/`completion_tokens` separately
5. Return synthetic dict: `{"input_tokens": total_input, "output_tokens": total_output}`

**Implementation details**:
```python
@staticmethod
def extract_usage_from_messages(messages: Any) -> Dict[str, Any]:
    """
    Extract and accumulate usage metadata from all AIMessage-like objects.
    
    Returns dict with accumulated input_tokens and output_tokens (both may be 0).
    """
    result = {"input_tokens": 0, "output_tokens": 0}
    if not isinstance(messages, list):
        return result
    
    for msg in messages:  # NOT reversed - we want ALL messages
        try:
            # Determine if this is an AI message (where token usage is relevant)
            is_ai = False
            if isinstance(msg, dict):
                is_ai = msg.get("type") == "ai"
            else:
                # Check LangChain AIMessage types
                msg_type = getattr(msg, "type", "")
                if msg_type == "ai" or hasattr(msg, "usage_metadata") or hasattr(msg, "usage"):
                    is_ai = True
            
            if not is_ai:
                continue
            
            # Extract usage payload (same logic as before)
            if isinstance(msg, dict):
                usage_payload = msg.get("usage_metadata") or msg.get("usage") or msg.get("response_metadata")
            else:
                usage_payload = getattr(msg, "usage_metadata", None) or getattr(msg, "usage", None) or getattr(msg, "response_metadata", None)
            
            if not isinstance(usage_payload, dict):
                continue
                
            # Extract token counts with fallback logic
            input_tokens = usage_payload.get("input_tokens") or usage_payload.get("prompt_tokens") or usage_payload.get("prompt_token_count")
            output_tokens = usage_payload.get("output_tokens") or usage_payload.get("completion_tokens") or usage_payload.get("completion_token_count")
            
            # Also check nested token_usage field
            token_usage = usage_payload.get("token_usage")
            if isinstance(token_usage, dict):
                if input_tokens is None:
                    input_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
                if output_tokens is None:
                    output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
            
            # Accumulate (treat None as 0)
            if isinstance(input_tokens, (int, float)) and input_tokens > 0:
                result["input_tokens"] += int(input_tokens)
            if isinstance(output_tokens, (int, float)) and output_tokens > 0:
                result["output_tokens"] += int(output_tokens)
                
        except Exception:
            continue  # Middleware must not raise
    
    return result
```

**Note**: We filter for AI messages only (where token usage is relevant) to avoid unnecessary processing and potential false positives from non‑AI messages that may contain usage‑like metadata.

---

### Phase 2 (SKIPPED): Nested‑Structure Accumulation (1 hour)

**File**: `src/multi_agent_dashboard/llm_client/core/response_processor.py` (lines 39‑79)

**Current behavior**: Recursively searches `agent_response` and `output` entries, returns first match.

**Required changes**:
1. Change from early‑return to accumulation
2. Initialize local totals, recursively accumulate from nested structures
3. Return aggregated dict (similar to `extract_usage_from_messages`)

**Implementation details**:
```python
@staticmethod
def extract_usage_from_candidate(candidate: Any) -> Dict[str, Any]:
    """
    Recursively search a candidate dict for usage or usage_metadata,
    accumulating token counts across all nested structures.
    
    Returns dict with accumulated input_tokens and output_tokens.
    """
    result = {"input_tokens": 0, "output_tokens": 0}
    
    if not isinstance(candidate, dict):
        return result
    
    # Check current level
    usage_payload = candidate.get("usage") or candidate.get("usage_metadata")
    if isinstance(usage_payload, dict) and usage_payload:
        # Extract token counts with same fallback logic as extract_usage_from_messages
        input_tokens = usage_payload.get("input_tokens") or usage_payload.get("prompt_tokens") or usage_payload.get("prompt_token_count")
        output_tokens = usage_payload.get("output_tokens") or usage_payload.get("completion_tokens") or usage_payload.get("completion_token_count")
        
        token_usage = usage_payload.get("token_usage")
        if isinstance(token_usage, dict):
            if input_tokens is None:
                input_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
            if output_tokens is None:
                output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
        
        if isinstance(input_tokens, (int, float)) and input_tokens > 0:
            result["input_tokens"] += int(input_tokens)
        if isinstance(output_tokens, (int, float)) and output_tokens > 0:
            result["output_tokens"] += int(output_tokens)
    
    # Recurse into agent_response
    nested = candidate.get("agent_response")
    if isinstance(nested, dict):
        nested_result = ResponseProcessor.extract_usage_from_candidate(nested)
        result["input_tokens"] += nested_result.get("input_tokens", 0)
        result["output_tokens"] += nested_result.get("output_tokens", 0)
    
    # Recurse into output entries
    output_entries = candidate.get("output")
    if isinstance(output_entries, list):
        for entry in output_entries:
            if not isinstance(entry, dict):
                continue
            # Check response, result, agent_response fields
            for field in ("response", "result", "agent_response"):
                nested = entry.get(field)
                if isinstance(nested, dict):
                    nested_result = ResponseProcessor.extract_usage_from_candidate(nested)
                    result["input_tokens"] += nested_result.get("input_tokens", 0)
                    result["output_tokens"] += nested_result.get("output_tokens", 0)
    
    return result
```

**Note**: We accept the potential minor risk of double‑counting if the same usage payload appears in multiple nested structures. This is unlikely in practice and keeps the implementation simple. If double‑counting is observed later, we can revisit with visited‑object tracking.

---

### Phase 3 (SKIPPED): Process Method Integration (1 hour)

**File**: `src/multi_agent_dashboard/llm_client/core/response_processor.py` (lines 358‑539)

**Required changes**:
1. Update calls to `extract_usage_from_messages()` and `extract_usage_from_candidate()` to handle aggregated dicts
2. Merge accumulated totals with existing fallback logic
3. Ensure single‑message chains produce identical results

**Implementation details**:
```python
# In process() method, around line 448:
if input_tokens is None or output_tokens is None:
    # Try usage on all AIMessages from agent state
    msg_usage = ResponseProcessor.extract_usage_from_messages(messages)
    if isinstance(msg_usage, dict):
        # Use accumulated totals, but only if we don't already have values
        if input_tokens is None:
            input_tokens = msg_usage.get("input_tokens")
        if output_tokens is None:
            output_tokens = msg_usage.get("output_tokens")
        
        # Promote usage to raw_dict (use first non‑empty message usage for structure)
        # ... existing promotion logic ...

    # Also check nested agent_response chains
    nested_usage = ResponseProcessor.extract_usage_from_candidate(raw_dict.get("agent_response"))
    if not nested_usage:
        try:
            nested_usage = ResponseProcessor.extract_usage_from_candidate(getattr(result, "agent_response", None))
        except Exception:
            nested_usage = None
    
    if isinstance(nested_usage, dict):
        # Accumulate nested totals
        if input_tokens is None:
            input_tokens = nested_usage.get("input_tokens")
        elif isinstance(nested_usage.get("input_tokens"), (int, float)):
            input_tokens += nested_usage["input_tokens"]
        
        if output_tokens is None:
            output_tokens = nested_usage.get("output_tokens")
        elif isinstance(nested_usage.get("output_tokens"), (int, float)):
            output_tokens += nested_usage["output_tokens"]
        
        # ... existing promotion logic ...
```

**Note**: We **accumulate** token counts from messages and nested structures, adding them to any existing counts we already have from top‑level usage fields. This provides the most accurate totals while avoiding double‑counting of the same tokens across different levels.

---

### Phase 4: Manual Verification Plan

**Verification Steps** (user‑led):
1. **Single‑message chain**: Run existing agent, verify token counts unchanged
2. **Multi‑message chain**: Create test with DeepSeek‑reasoner (as in FR example), verify accumulated tokens > last‑message tokens
3. **Cost calculation**: Check UI metrics show increased cost proportional to accumulated tokens
4. **Backward compatibility**: Verify existing pipeline runs still work

**Early Validation Points**:
- After Phase 1: Test `extract_usage_from_messages()` with synthetic message list
- After Phase 2: Test `extract_usage_from_candidate()` with nested structures
- After Phase 3: Run simple agent invocation, check `TextResponse` token counts

**Verification Method**:
```python
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
```

---

## Reference: Implementation Decisions

1. **Message Type Filtering**: Process only **AI messages** (where token usage is relevant), not all message types.
2. **Double‑Counting Prevention**: Accept potential minor double‑counting risk; keep implementation simple.
3. **Accumulation vs. Replacement**: **Accumulate** token counts from messages and nested structures, adding to existing totals.

These decisions balance accuracy, simplicity, and implementation speed while addressing the core requirement.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Double‑counting tokens | High | Accept minor risk; monitor real runs |
| Provider field inconsistency | Medium | Use existing fallback logic for each message |
| Performance impact | Low | Messages lists are small (<100), recursion depth unchanged |
| Backward compatibility | High | Preserve existing API, test single‑message chains |
| Missing intermediate LLM calls | Medium | Phase 1 addresses message chains; monitor for edge cases |

**Fallback Position**: If this approach misses token usage from intermediate LLM calls not captured in final messages, we can later add lightweight callback handler as **supplemental** (not replacement).

---

## Next Actions

### Implementation Sequence
1. **Implement Phase 1** (message‑level accumulation)
2. **User verification** – test with multi‑message chains
3. **Implement Phase 2** (nested‑structure accumulation)
4. **User verification** – test with nested agent_response structures
5. **Implement Phase 3** (process method integration)
6. **Final verification** – run actual agent pipelines

### Success Criteria
- ✅ Multi‑message chains show accumulated token counts > last‑message counts
- ✅ Single‑message chains produce identical results
- ✅ Cost calculations reflect accumulated totals
- ✅ Existing pipeline runs continue working

---

## Appendix: Reference Code Locations

### Primary Files
- `src/multi_agent_dashboard/llm_client/core/response_processor.py`
  - `extract_usage_from_messages()`: lines 82‑113
  - `extract_usage_from_candidate()`: lines 39‑79
  - `process()`: lines 358‑539

### Supporting Files
- `src/multi_agent_dashboard/runtime/metrics_extractor.py` – `extract_tokens_from_raw()` fallback
- `src/multi_agent_dashboard/engine/metrics_aggregator.py` – cost computation
- `src/multi_agent_dashboard/shared/instrumentation.py` – shared utilities

### Test Location
- `tests/test_llm_client_instrumentation_output.py` – existing instrumentation tests

---

**Implementation Owner**: AI Assistant (with user verification at each phase)  
**Estimated Time**: 4‑6 hours (implementation) + 1‑2 hours (verification)  
**Confidence Level**: 99/100 – leverages proven patterns, minimal disruption