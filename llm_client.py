# llm_client.py
import time
import logging
import json
import inspect
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "LLMClient",
    "TextResponse",
    "LLMError",
]


# =========================
# Public data structures
# =========================

@dataclass
class TextResponse:
    """
    Normalized text response returned by LLMClient.
    """
    text: str
    raw: Dict[str, Any]


class LLMError(RuntimeError):
    """
    Typed exception raised for LLM failures.
    Keeps LLM concerns isolated from orchestration logic.
    """
    pass


# =========================
# LLM Client
# =========================

class LLMClient:
    """
    Thin, reusable wrapper around an LLM SDK client.

    Responsibilities:
    - Request execution
    - Retry / backoff
    - Capability detection
    - Response normalization

    Non-responsibilities:
    - No agent knowledge
    - No pipeline knowledge
    - No state orchestration
    """

    def __init__(
        self,
        client: Any,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_base: float = 1.5,
        on_rate_limit: Optional[Callable[[int], None]] = None,
    ):
        self._client = client
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._on_rate_limit = on_rate_limit

        # SDK capability detection (best-effort)
        self._capabilities = self._detect_capabilities()

    # -------------------------
    # Public API
    # -------------------------

    def create_text_response(
        self,
        model: str,
        prompt: str,
        *,
        response_format: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> TextResponse:
        """
        Execute a text-generation request and return a normalized response.
        """
        last_err: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                kwargs = {
                    "model": model,
                    "input": prompt,
                }

                if stream and "stream" in self._capabilities:
                    kwargs["stream"] = True

                if response_format is not None and "response_format" in self._capabilities:
                    kwargs["response_format"] = response_format

                logger.debug(
                    "LLM request: model=%s, prompt_len=%d, stream=%s, structured=%s",
                    model,
                    len(prompt),
                    stream,
                    bool(response_format),
                )

                response = self._client.responses.create(**kwargs)

                return TextResponse(
                    text=self._extract_text(response),
                    raw=self._to_dict(response),
                )

            except TypeError:
                # Programming / integration error → fail fast
                raise

            except Exception as e:
                last_err = e
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    e,
                )

                # Best-effort rate-limit signal
                if self._on_rate_limit and self._looks_like_rate_limit(e):
                    try:
                        self._on_rate_limit(attempt)
                    except Exception:
                        pass  # metrics hooks must never break execution

                if attempt < self._max_retries:
                    time.sleep(self._backoff_base ** attempt)

        raise LLMError("LLM request failed after retries") from last_err

    # -------------------------
    # Capability detection
    # -------------------------

    def _detect_capabilities(self) -> set[str]:
        """
        Introspect the SDK to see which kwargs are supported.
        """
        try:
            fn = self._client.responses.create
            sig = inspect.signature(fn)
            return set(sig.parameters.keys())
        except Exception:
            return set()

    # -------------------------
    # Response normalization
    # -------------------------

    def _extract_text(self, response: Any) -> str:
        """
        Best-effort text extraction across SDK versions.
        """
        try:
            if hasattr(response, "output_text") and response.output_text:
                return response.output_text

            chunks: list[str] = []

            if hasattr(response, "output") and isinstance(response.output, list):
                for block in response.output:
                    if not isinstance(block, dict):
                        continue
                    content = block.get("content")
                    if isinstance(content, list):
                        for c in content:
                            if (
                                isinstance(c, dict)
                                and c.get("type") == "output_text"
                            ):
                                chunks.append(c.get("text", ""))
        except Exception:
            logger.exception("Failed to extract text from LLM response")
            raise

        if chunks:
            return "".join(chunks)

        if hasattr(response, "text"):
            return str(response.text)

        return str(response)

    def _to_dict(self, response: Any) -> Dict[str, Any]:
        """
        Convert SDK response into a serializable dict (best-effort).
        """
        try:
            if hasattr(response, "model_dump"):
                return response.model_dump()
            if hasattr(response, "dict"):
                return response.dict()
        except Exception:
            pass
        return {"repr": repr(response)}

    # -------------------------
    # Utilities
    # -------------------------

    @staticmethod
    def safe_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON safely; return None on failure.
        """
        try:
            return json.loads(text)
        except Exception:
            return None

    @staticmethod
    def _looks_like_rate_limit(exc: Exception) -> bool:
        """
        Heuristic check to detect rate-limit-like failures
        without importing provider-specific exception types.
        """
        msg = str(exc).lower()
        return any(
            token in msg
            for token in ("rate limit", "too many requests", "429")
        )
