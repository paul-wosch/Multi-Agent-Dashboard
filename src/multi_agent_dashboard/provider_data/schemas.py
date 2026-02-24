"""
Schema definitions for provider models.

Defines the internal representation of provider model data extracted from external sources.
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

        # Determine image_inputs from both attachment and modalities.input
        modalities = raw.get("modalities", {})
        input_modalities = modalities.get("input", [])
        image_inputs = attachment or ("image" in input_modalities)

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