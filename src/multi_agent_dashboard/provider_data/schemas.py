"""
Schema definitions for provider model data using immutable dataclasses.

This module defines the structured representation of provider model data
extracted from external sources (models.dev). It provides mapping from
external JSON schema fields to internal capability and pricing fields used
by the advisory system.

Key Components:
    ProviderModel: Immutable dataclass representing a single provider model
    - Capability flags (tool_calling, structured_output, image_inputs, etc.)
    - Pricing information (input_price, output_price per 1M tokens)
    - Metadata (knowledge cutoff, token limits)
    - Raw JSON data for debugging

External-to-Internal Field Mapping:
    External JSON field → ProviderModel field
    ──────────────────────────────────────────────
    cost.input           → input_price
    cost.output          → output_price
    attachment           → (not directly mapped)
    tool_call            → tool_calling
    structured_output    → structured_output
    reasoning            → reasoning
    temperature          → temperature
    modalities.input     → image_inputs (if 'image' in list)
    knowledge            → knowledge
    limit.context        → max_input_tokens
    limit.output         → max_output_tokens

Immutable Design:
    - All fields are frozen (immutable after creation)
    - Default values reflect missing data (False, 0.0, empty string)
    - Raw JSON stored for debugging but not used in normal operations

Factory Method:
    ProviderModel.from_raw_json(): Creates instance from raw JSON with
    proper field mapping and type conversion.

Capability Dictionary:
    to_capability_dict(): Converts to dictionary format expected by
    advisory capability system, omitting missing fields.

Usage:
    from multi_agent_dashboard.provider_data.schemas import ProviderModel
    
    # Create from raw JSON (typically done by loader)
    model = ProviderModel.from_raw_json(
        provider='openai',
        model_id='gpt-4o',
        raw={'cost': {'input': 5.0, 'output': 15.0}, 'tool_call': True}
    )
    
    # Access capabilities
    if model.tool_calling:
        print('Model supports tool calling')
    
    # Convert to capability dictionary
    caps = model.to_capability_dict()

Note: This schema represents advisory data only. Actual agent capabilities
are determined by agent configuration, not by these advisory flags.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass(frozen=True)
class ProviderModel:
    """
    Internal representation of a provider model's capabilities and pricing.

    This dataclass maps fields from external JSON schema (models.dev/api.json)
    to internal keys used by the advisory capability system.

    Fields are immutable; defaults reflect missing data (False for booleans,
    0.0 for pricing, empty string for knowledge, 0 for token limits).
    """
    # Model identifier (unique across all providers)
    model_id: str
    # Provider identifier (e.g., "openai", "deepseek")
    provider: str

    # Pricing (USD per 1M tokens)
    input_price: float = 0.0
    output_price: float = 0.0

    # Capability flags
    image_inputs: bool = False
    tool_calling: bool = False
    structured_output: bool = False
    reasoning: bool = False
    temperature: bool = False

    # Metadata
    knowledge: str = ""          # Knowledge cutoff date (YYYY‑MM)
    max_input_tokens: int = 0    # Context window size
    max_output_tokens: int = 0   # Output token limit

    # Raw JSON data (optional, for debugging)
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_raw_json(
        cls,
        provider: str,
        model_id: str,
        raw: Dict[str, Any]
    ) -> "ProviderModel":
        """
        Create a ProviderModel from raw JSON model entry.

        Applies the external‑to‑internal field mapping described in the
        dynamic pricing & capabilities strategy document.

        Args:
            provider: Provider identifier ("openai", "deepseek")
            model_id: Model identifier (key in the models dict)
            raw: Raw model dictionary from external JSON

        Returns:
            ProviderModel instance with mapped fields.
        """
        # Extract pricing
        cost = raw.get("cost", {})
        input_price = float(cost.get("input", 0.0))
        output_price = float(cost.get("output", 0.0))

        # Capability flags
        attachment = raw.get("attachment", False)
        tool_call = raw.get("tool_call", False)
        structured_output = raw.get("structured_output", False)
        reasoning = raw.get("reasoning", False)
        temperature = raw.get("temperature", False)

        # Determine image_inputs from modalities.input
        modalities = raw.get("modalities", {})
        input_modalities = modalities.get("input", [])
        image_inputs = "image" in input_modalities

        # Knowledge cutoff
        knowledge = raw.get("knowledge", "")
        if knowledge is None:
            knowledge = ""

        # Token limits
        limit = raw.get("limit", {})
        max_input_tokens = int(limit.get("context", 0))
        max_output_tokens = int(limit.get("output", 0))

        return cls(
            model_id=model_id,
            provider=provider,
            input_price=input_price,
            output_price=output_price,
            image_inputs=image_inputs,
            tool_calling=tool_call,
            structured_output=structured_output,
            reasoning=reasoning,
            temperature=temperature,
            knowledge=knowledge,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            raw_data=raw,
        )

    def to_capability_dict(self) -> Dict[str, Any]:
        """
        Convert to the capability dictionary format expected by
        `get_capabilities()` and `supports_feature()`.

        Returns a dictionary with keys matching the advisory capability system.
        Missing fields are omitted; the loader layer will add defaults.
        """
        caps: Dict[str, Any] = {}

        # Core capability flags (present in external schema)
        caps["image_inputs"] = self.image_inputs
        caps["tool_calling"] = self.tool_calling
        caps["structured_output"] = self.structured_output
        caps["reasoning"] = self.reasoning
        caps["temperature"] = self.temperature

        # Token limits
        if self.max_input_tokens > 0:
            caps["max_input_tokens"] = self.max_input_tokens
        if self.max_output_tokens > 0:
            caps["max_output_tokens"] = self.max_output_tokens

        # Knowledge cutoff (string)
        if self.knowledge:
            caps["knowledge"] = self.knowledge

        # Pricing fields (not part of capabilities, stored separately)
        # These are accessed via get_pricing()

        return caps