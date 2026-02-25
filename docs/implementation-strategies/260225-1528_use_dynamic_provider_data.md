# Dynamic Pricing & Capabilities Implementation Strategy (Completed: 25 Feb 2026)

## Overview
Replace hardcoded pricing (`OPENAI_PRICING`, `DEEPSEEK_PRICING`) and static capability mappings (`provider_capabilities.py`) with dynamic data fetched from `https://models.dev/api.json`. The new system will also support user‑configurable local Ollama models.

**Confidence**: 99/100 – architecture clean, advisory‑only usage, minimal integration points.

## Key Design Decisions

1. **Data Source**: External `api.json` from models.dev (top‑level keys `openai`, `deepseek`).
2. **File Management**: Four JSON files:
   - `provider_models_all.json` – raw downloaded `api.json`
   - `provider_models.json` – filtered copy of raw provider data (same nested structure as api.json, only openai/deepseek providers) (never overwritten)
   - `template_ollama_models.json` – static template with top‑level `"ollama"` key (git‑tracked)
   - `local_ollama_models.json` – user‑customizable with top‑level `"ollama"` key (git‑ignored)
3. **Update Strategy**: Manual only – delete files to trigger re‑download.
4. **Error Handling**: No fallback; log WARNING for missing data, assume `False` for capabilities and `0.0` for pricing.
5. **Advisory‑Only Usage**: Existing `supports_feature()` and `get_capabilities()` remain advisory (warnings/UI defaults).
6. **Composite Cache Keys**: Internal cache uses `"provider|model"` as unique identifier to disambiguate duplicate model IDs across providers. Provider‑aware lookup functions (`get_capabilities_for_provider`, `get_pricing_for_provider`) accept `provider_id` and optional `model`; `model=None` returns empty dict (no provider defaults).

## Schema Mapping (External → Internal)

| External Field | Internal Key | Type | Notes |
|----------------|--------------|------|-------|
| `id` | model identifier | string | Lookup key |
| `cost.input` | input price | float | USD per 1M tokens |
| `cost.output` | output price | float | USD per 1M tokens |
| `attachment` | `image_inputs` | boolean | Vision support |
| `tool_call` | `tool_calling` | boolean | Tool‑calling support |
| `structured_output` | `structured_output` | boolean | Structured output support |
| `reasoning` | `reasoning` | boolean | Reasoning/chain‑of‑thought |
| `temperature` | `temperature` | boolean | Temperature parameter support |
| `knowledge` | `knowledge` | string | Knowledge cutoff date (YYYY‑MM) |
| `modalities.input` contains “image” | `image_inputs` | boolean | Duplicate mapping for consistency |

## Progressive Implementation Phases

### ✅ Phase 0: Preparation & Parallel Setup
**Goal**: Set up new modules without breaking existing functionality.

1. **Create provider data directory**
   ```bash
   mkdir -p data/provider_models
   ```

2. **Add new configuration constants** (`config.py`)
   - Keep existing `OPENAI_PRICING`/`DEEPSEEK_PRICING` (will be removed later)
   - Add:
     ```python
     PROVIDER_DATA_DIR = "data/provider_models"
     PROVIDER_MODELS_ALL_FILE = "provider_models_all.json"
     PROVIDER_MODELS_FILE = "provider_models.json"
     TEMPLATE_OLLAMA_MODELS_FILE = "template_ollama_models.json"
     LOCAL_OLLAMA_MODELS_FILE = "local_ollama_models.json"
     MODELS_DEV_URL = "https://models.dev/api.json"
     ```

3. **Create empty `provider_data` package**
   ```
   src/multi_agent_dashboard/provider_data/
   ├── __init__.py
   ├── downloader.py
   ├── extractor.py
   ├── cache.py
   ├── loader.py
   └── schemas.py
   ```

### ✅ Phase 1: Data Layer Implementation
**Goal**: Implement core data loading, downloading, and extraction logic.

#### ✅ Step 1.1: Schemas & Types (`schemas.py`)
Define `ProviderModel` dataclass with fields mapping to external schema.

#### ✅ Step 1.2: Downloader (`downloader.py`)
- Fetch `api.json` from `MODELS_DEV_URL` with retries/timeouts
- Save raw JSON to `PROVIDER_MODELS_ALL_FILE`
- Handle network errors (raise exception, log ERROR)

#### ✅ Step 1.3: Extractor (`extractor.py`)
- Parse `provider_models_all.json`, filter for `openai` and `deepseek` providers
- Copy entire provider entry (including all fields) preserving the original nested structure
- No schema mapping at this stage – mapping deferred to loader/cache layer
- Write filtered copy to `provider_models.json` (only if file missing)

#### ✅ Step 1.4: Cache & Loader (`cache.py`, `loader.py`)
- **File‑state machine**:
  ```
  IF both provider_models_all.json and provider_models.json missing:
      download → provider_models_all.json
      extract → provider_models.json
  ELSE IF provider_models_all.json missing BUT provider_models.json present:
      download → provider_models_all.json
      log WARNING: “provider_models_all.json re‑downloaded; review for updates”
      (keep provider_models.json unchanged)
  ELSE IF provider_models_all.json present BUT provider_models.json missing:
      extract → provider_models.json
  ELSE (both present):
      use provider_models.json (no action)
  ```
- **Lazy loading**: Load data on first `get_capabilities()` or `get_pricing()` call
- **In‑memory cache**: `@lru_cache` on loader functions; cache dictionary maps composite key `"provider|model"` → `ProviderModel`
- **Thread safety**: `threading.Lock` for concurrent file access
- **Composite cache keys**: Cache uses `"provider|model"` as unique key to disambiguate duplicate model IDs across providers

#### ✅ Step 1.5: Public Interface (`loader.py`)
- `get_capabilities_for_provider(provider_id: str, model: Optional[str]) -> Dict[str, Any]`: returns capability dict for provider|model (composite cache key `"provider|model"`); `model=None` returns empty dict
- `get_pricing_for_provider(provider_id: str, model: Optional[str]) -> Tuple[float, float]`: returns (input_price, output_price) for provider|model
- `get_all_models() -> List[str]`: returns list of known composite model IDs (`"provider|model"`)
- (Legacy) `get_capabilities(model_id: str) -> Dict[str, Any]`: single‑argument version for unique model IDs (deprecated)
- (Legacy) `get_pricing(model_id: str) -> Tuple[float, float]`: single‑argument version (deprecated)

#### ✅ Step 1.6: Phase 1 Alignment (Introduce composite provider model key)

Phase 1 **needs alignment** before proceeding to Phase 2. Current implementation uses simple `model_id` as cache key, which cannot disambiguate duplicate model IDs across providers. Must update:

1. **`cache.py`**: Change `_model_cache` to use composite key `f"{provider}|{model_id}"`
2. **`loader.py`**: 
   - `load_provider_models()` → store with composite key
   - Add `get_capabilities_for_provider()` and `get_pricing_for_provider()` that construct composite key
   - Keep legacy single‑argument functions (deprecated) for backward compatibility
3. **`schemas.py`**: Already stores `provider` field; no change needed

Without this update, Phase 2’s integration will either fail (missing functions) or produce incorrect lookups for duplicate model IDs.

### ✅ Phase 2: Integration with Existing Code
**Goal**: Replace static lookups with dynamic data while maintaining backward compatibility.

#### ✅ Step 2.1: Update `provider_capabilities.py`
- Remove `PROVIDER_DEFAULT_CAPABILITIES` and `MODEL_CAPABILITIES`
- Implement `get_capabilities()` using `provider_data.loader.get_capabilities_for_provider()`
- Keep `supports_feature()` signature unchanged
- Add WARNING logging for missing model data (when `model=None` returns empty dict)
- Extend capability keys to include `temperature` and `knowledge`
- **Composite cache key**: Internal cache uses `"provider|model"` as unique identifier; duplicate model IDs across providers are disambiguated by provider

#### ✅ Step 2.2: Update `metrics_aggregator.py`
- Replace `OPENAI_PRICING`/`DEEPSEEK_PRICING` lookups with `provider_data.loader.get_pricing_for_provider()`
- Handle missing pricing (return `0.0`, log WARNING)
- Keep `_compute_cost()` signature unchanged (provider and model passed as arguments)

#### ✅ Step 2.3: Verify Advisory Usage
- `provider_tool_adapter.py` – already uses `supports_feature()` (no change)
- `multimodal_handler.py` – already uses `supports_feature()` (no change)
- Run existing tests to ensure no regression

#### ✅ Step 2.4: Align test suite
- Update existing tests with dynamic provider data usage 

### ✅ Phase 3: Ollama Support
**Goal**: Allow user‑configurable local Ollama models with same schema.

#### ✅ Step 3.1: Create Template File
- Create `template_ollama_models.json` in `data/provider_models/`
- Include example entries for common Ollama models (llama3, llava, etc.)

#### ✅ Step 3.2: Integration in Loader
- Extend `loader.py` to also read `local_ollama_models.json` (if exists)
- Merge with external provider data (local entries take precedence)
- Same schema validation as external data

#### ✅ Step 3.3: Documentation
- Update `AGENTS.md` with instructions for customizing local Ollama models
- Explain manual update process (delete files to refresh)

### ✅ Phase 4: Cleanup & Finalization
**Goal**: Remove old hardcoded data and finalize transition.

#### ✅ Step 4.1: Remove Hardcoded Pricing
- Delete `OPENAI_PRICING` and `DEEPSEEK_PRICING` from `config.py`
- Update any imports that reference them (none expected)

#### ✅ Step 4.2: Verify Full Functionality
- Start application, verify pricing/capability lookups work
- Test with missing model (should log WARNING)
- Test with local Ollama model (if configured)

#### ✅ Step 4.3: Update Documentation
- Add section in `AGENTS.md` about dynamic pricing/capabilities
- Document file‑state behavior and manual update procedure

## File‑State Management Details

### Initialization Flow (First Run)
1. Check `data/provider_models/` for `provider_models_all.json` and `provider_models.json`
2. Both missing → download `api.json`, save as `provider_models_all.json`
3. Extract to `provider_models.json`
4. Load `provider_models.json` into memory cache

### Update Flow (Manual)
- User deletes `provider_models_all.json` → re‑download on next run
- User deletes `provider_models.json` → re‑extract from `provider_models_all.json`
- User modifies `provider_models.json` manually → changes persist (never overwritten)

### Error Recovery
- **Network failure**: Log ERROR, keep existing files, raise exception
- **Malformed JSON**: Log ERROR, fallback to existing `provider_models.json` if present
- **Missing fields**: Log WARNING, treat missing capability as `False`, missing pricing as `0.0`

## Logging Strategy

| Level | When to Use |
|-------|-------------|
| ERROR | Network failure, malformed JSON, unrecoverable errors |
| WARNING | Missing pricing/capability data for a model |
| INFO | Using existing `provider_models.json` (no download needed) |
| DEBUG | Detailed extraction steps, cache hits/misses |

## Testing Strategy (Manual Verification)

1. **Initial download**: Delete both JSON files, start app → files created
2. **Missing model**: Use unknown model ID → WARNING logged, default capabilities
3. **Pricing lookup**: Run agent with known model → cost computed correctly
4. **Capability warnings**: Configure agent with unsupported feature → UI shows warning
5. **Local Ollama**: Create `local_ollama_models.json` → custom models appear in capabilities

## Rollback Plan
- **Parallel operation**: Old hardcoded data remains until Phase 4
- **If issues arise**: Revert changes to `provider_capabilities.py` and `metrics_aggregator.py`
- **Manual rollback**: Restore `provider_models.json` from backup

## Success Criteria
1. All existing tests pass
2. Pricing calculations match hardcoded values for known models
3. Capability warnings appear as before (advisory unchanged)
4. Local Ollama models can be added via JSON file
5. Manual update process works (delete → re‑download)

## Implementation Order Checklist (Completed: 25 Feb 2026)

- [x] Phase 0: Configuration & directory setup
- [x] Phase 1.1: Schemas (`schemas.py`)
- [x] Phase 1.2: Downloader (`downloader.py`)
- [x] Phase 1.3: Extractor (`extractor.py`)
- [x] Phase 1.4: Cache & loader (`cache.py`, `loader.py`)
- [x] Phase 1.5: Public Interface (`loader.py`)
- [x] Phase 1.6: Phase 1 Alignment (Introduce composite provider model key)
- [x] Phase 2.1: Update `provider_capabilities.py`
- [x] Phase 2.2: Update `metrics_aggregator.py`
- [x] Phase 2.3: Verify advisory usage
- [x] Phase 2.4: Align test suite
- [x] Phase 3.1: Create Ollama template
- [x] Phase 3.2: Integrate local Ollama
- [x] Phase 3.3: Update documentation
- [x] Phase 4.1: Remove hardcoded pricing
- [x] Phase 4.2: Verify full functionality
- [x] Phase 4.3: Final documentation updates

Each step is atomic and can be verified independently before proceeding to the next.

---

## APPENDIX A: **Clarified Extraction Approach**

**Current understanding flaw**: The extractor still transforms/flattens data. Instead, it must preserve the **exact nested structure** of `api.json` for supported providers (`openai`, `deepseek`), copying all fields unchanged.

**Revised flow**:
1. **Extractor** (`extractor.py`) → filters `openai`/`deepseek` keys, copies entire provider entry (including `id`, `env`, `npm`, `api`, `name`, `doc`, `models` dict with all model metadata)
2. **No schema mapping at extraction** → mapping (`tool_call`→`tool_calling`, `attachment`→`image_inputs`, etc.) deferred to loader/cache layer (`ProviderModel.from_raw_json`)
3. **Output**: `provider_models.json` = filtered copy of raw `api.json` (same nesting, all fields)
4. **Ollama consistency**: `template_ollama_models.json` / `local_ollama_models.json` will have top‑level `"ollama"` key with same nested pattern

**Benefits**:
- **KISS/DRY**: Extraction = filter‑and‑copy; mapping logic stays in `schemas.py`
- **Future‑proof**: All original metadata (`release_date`, `modalities`, `open_weights`, etc.) preserved for debugging/future features
- **Architectural fit**: Matches existing advisory system’s provider‑grouped structure (`provider_capabilities.py`)