"""
MetricsAggregator – cost computation and aggregation across agent runs.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from multi_agent_dashboard.config import OPENAI_PRICING, DEEPSEEK_PRICING
from .types import RunMetrics

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """
    Provides static methods for computing token costs and aggregating run metrics.
    """

    @staticmethod
    def compute_cost(
        model: str,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        provider_id: Optional[str] = None,
    ) -> Tuple[float, float, float]:
        """Compute approximate cost for a single LLM call.

        Return (total_cost, input_cost, output_cost).

        Prices are per 1M tokens.

        This helper is provider-aware: for OpenAI-family providers
        (provider_id is None/'' or 'openai' or 'azure_openai') it uses
        OPENAI_PRICING. For other providers we currently return zero so that
        non-OpenAI calls do not get mis-attributed OpenAI prices.
        """
        if input_tokens is None and output_tokens is None:
            return 0.0, 0.0, 0.0

        # Parse provider/model string format
        model_for_pricing = model
        if "/" in model:
            # Split only on first slash
            maybe_provider, model_name = model.split("/", 1)
            model_for_pricing = model_name
            # If provider_id not specified, use extracted provider
            if provider_id is None:
                provider_id = maybe_provider
            # If provider_id specified but differs, log warning but use extracted provider
            elif provider_id != maybe_provider:
                logger.debug(
                    f"Provider mismatch in compute_cost: model suggests '{maybe_provider}', "
                    f"but provider_id is '{provider_id}'. Using '{maybe_provider}' for pricing."
                )
                provider_id = maybe_provider

        provider = (provider_id or "").strip().lower()

        # Backwards-compatible default: treat missing provider_id as OpenAI.
        is_openai_family = (not provider) or provider in ("openai", "azure_openai")

        pricing = None
        if is_openai_family:
            pricing = OPENAI_PRICING.get(model_for_pricing)
        elif provider == "deepseek":
            pricing = DEEPSEEK_PRICING.get(model_for_pricing)
        if not pricing:
            return 0.0, 0.0, 0.0

        inp = input_tokens or 0
        out = output_tokens or 0

        input_cost = inp / 1_000_000.0 * pricing.get("input", 0.0)
        output_cost = out / 1_000_000.0 * pricing.get("output", 0.0)
        return input_cost + output_cost, input_cost, output_cost

    @staticmethod
    def aggregate_totals(metrics_list: List[RunMetrics]) -> Tuple[float, float, float, float]:
        """
        Sum up total cost, input cost, output cost, and latency across multiple agent runs.

        Returns (total_cost, total_input_cost, total_output_cost, total_latency).
        """
        total_cost = sum((m.total_cost or 0.0) for m in metrics_list)
        total_input_cost = sum((m.input_cost or 0.0) for m in metrics_list)
        total_output_cost = sum((m.output_cost or 0.0) for m in metrics_list)
        total_latency = sum((m.latency or 0.0) for m in metrics_list)
        return total_cost, total_input_cost, total_output_cost, total_latency