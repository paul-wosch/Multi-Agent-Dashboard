# multi_agent_dashboard/chat_model_factory.py
import logging
import json
from typing import Any, Dict, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

# Try to import LangChain init_chat_model (optional)
_LANGCHAIN_AVAILABLE = False
_init_chat_model = None

try:
    from langchain.chat_models import init_chat_model  # type: ignore
    _LANGCHAIN_AVAILABLE = True
    _init_chat_model = init_chat_model
except Exception:
    # Keep resilience when LangChain is not installed or partial environments.
    _LANGCHAIN_AVAILABLE = False
    _init_chat_model = None


class ChatModelFactory:
    """
    Lightweight factory and cache for LangChain chat models created via init_chat_model.

    Keyed by (model, provider_id, endpoint, use_responses_api, model_class, provider_features_fingerprint, timeout).
    """

    def __init__(self, init_fn: Optional[Callable[..., Any]] = None):
        if init_fn is None and not _LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain not available; cannot create ChatModelFactory without init function.")
        self._init_fn = init_fn or _init_chat_model
        # include timeout as final component in key tuple (Optional[float])
        self._cache: Dict[
            Tuple[str, str, Optional[str], bool, Optional[str], str, Optional[float], Optional[float]], Any] = {}

    def _key(
            self,
            model: str,
            provider_id: Optional[str],
            endpoint: Optional[str],
            use_responses_api: bool,
            model_class: Optional[str],
            provider_features: Optional[Dict[str, Any]] = None,
            timeout: Optional[float] = None,
            temperature: Optional[float] = None,
    ) -> Tuple[str, str, Optional[str], bool, Optional[str], str, Optional[float], Optional[float]]:
        """
        Build a stable cache key for a chat model, including a fingerprint of provider_features
        and the timeout so that changes to capability hints or per-call timeout cause a fresh
        model instance to be created.
        """
        features_key = ""
        if provider_features:
            try:
                # Stable, order-independent JSON fingerprint
                features_key = json.dumps(
                    provider_features,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except Exception:
                # Fallback: use repr of sorted items when JSON encoding fails
                try:
                    features_key = repr(sorted(provider_features.items()))
                except Exception:
                    features_key = repr(provider_features)
        # Normalize timeout to a float (or None)
        timeout_val: Optional[float] = None
        if timeout is not None:
            try:
                timeout_val = float(timeout)
            except Exception:
                timeout_val = None

        temp_val: Optional[float] = None
        if temperature is not None:
            try:
                temp_val = float(temperature)
            except Exception:
                temp_val = None

        return (
            model or "",
            provider_id or "",
            endpoint or None,
            bool(use_responses_api),
            model_class or "",
            features_key,
            timeout_val,
            temp_val,
        )

    def get_model(
            self,
            model: str,
            *,
            provider_id: Optional[str] = None,
            endpoint: Optional[str] = None,
            use_responses_api: bool = None,
            model_class: Optional[str] = None,
            provider_features: Optional[Dict[str, Any]] = None,
            timeout: Optional[float] = None,
            temperature: Optional[float] = None,
    ):
        """
        Return a LangChain chat model instance for the provided metadata.
        Caches instances to avoid repeated init costs.
        """
        # Provider name normalization for provider-specific fallback logic
        provider_norm = (provider_id or "").strip().lower()

        # Normalize endpoint: if user provided a host:port without scheme, add a default scheme.
        if endpoint and "://" not in endpoint:
            endpoint = f"http://{endpoint}"

        key = self._key(
            model,
            provider_id,
            endpoint,
            bool(use_responses_api),
            model_class,
            provider_features,
            timeout=timeout,
            temperature=temperature,
        )
        if key in self._cache:
            return self._cache[key]

        init_kwargs: Dict[str, Any] = {}
        # map endpoint to base_url (popular param across providers)
        if endpoint:
            # Some providers expect 'base_url' or 'base_url' like param; init_chat_model passes kwargs to concrete impl
            init_kwargs["base_url"] = endpoint
            # Some integrations (openai) call this 'base_url' or 'api_base' - provider integration will accept 'base_url'
            # DeepSeek integration expects 'api_base' parameter
            if provider_norm == "deepseek":
                init_kwargs["api_base"] = endpoint
        # Propagate timeout to underlying LangChain model integrations:
        # Use 'request_timeout' (preferred alias used by ChatOpenAI) and set 'timeout' as fallback,
        # so providers reading either name will receive the configured numeric timeout value.
        if timeout is not None:
            try:
                timeout_val = float(timeout)
                init_kwargs["request_timeout"] = timeout_val
                # some integrations accept 'timeout' as the canonical kwarg; ensure it is present as well
                init_kwargs.setdefault("timeout", timeout_val)
            except Exception:
                # Fallback: pass raw value under 'timeout'
                init_kwargs["timeout"] = timeout

        # Convey Responses API preference for providers that support it (e.g., OpenAI)
        if use_responses_api:
            init_kwargs["use_responses_api"] = True
            # Recommend the responses output_version for consistent content_blocks formatting when available
            # many integrations accept 'output_version' (e.g., ChatOpenAI)
            init_kwargs["output_version"] = "responses/v1"

        # Provider profile / features may be passed through where supported
        if provider_features:
            # Some providers accept a 'profile' or 'model_profile' kwarg; pass under 'profile' and let integration ignore unknown keys.
            init_kwargs["profile"] = provider_features

        if temperature is not None:
            try:
                init_kwargs["temperature"] = float(temperature)
            except Exception:
                init_kwargs["temperature"] = temperature

        model_provider = provider_id or None

        # OpenAI API key is sourced from config (.env) and only applied when provider_id == "openai".
        if provider_norm == "openai":
            try:
                from multi_agent_dashboard import config as _cfg_key
                openai_key = getattr(_cfg_key, "OPENAI_API_KEY", None)
                if openai_key:
                    init_kwargs.setdefault("api_key", openai_key)
            except Exception:
                pass

        if provider_norm == "deepseek":
            try:
                from multi_agent_dashboard import config as _cfg_ds
                ds_key = getattr(_cfg_ds, "DEEPSEEK_API_KEY", None)
                if ds_key:
                    init_kwargs.setdefault("api_key", ds_key)
            except Exception:
                pass

        # Attempt to initialize via the unified helper
        try:
            chat_model = self._init_fn(
                model,
                model_provider=model_provider,
                **init_kwargs,
            )
            # Attach effective_request_timeout attribute for observability
            try:
                eff_to = init_kwargs.get("request_timeout", init_kwargs.get("timeout", None))
                if eff_to is not None:
                    try:
                        setattr(chat_model, "_effective_request_timeout", float(eff_to))
                    except Exception:
                        setattr(chat_model, "_effective_request_timeout", eff_to)
            except Exception:
                logger.debug("Failed to set _effective_request_timeout on chat_model", exc_info=True)

            # cache and return
            self._cache[key] = chat_model
            logger.debug(
                "ChatModelFactory: created model=%s provider=%s endpoint=%s request_timeout=%s",
                model, provider_id, endpoint, getattr(chat_model, "_effective_request_timeout", None)
            )
            return chat_model
        except Exception as e:
            logger.debug(
                "ChatModelFactory: init_chat_model failed for model=%s provider=%s endpoint=%s; error=%s",
                model,
                provider_id,
                endpoint,
                e,
                exc_info=True,
            )
            raise