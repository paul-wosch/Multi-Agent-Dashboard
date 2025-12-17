# llm_client.py
import time
import logging
import json
import inspect
from typing import Any, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextResponse:
    text: str
    raw: Dict[str, Any]


class LLMClient:
    def __init__(
        self,
        client: Any,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_base: float = 1.5,
    ):
        self._client = client
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base

        # --- SDK capability detection ---
        self._capabilities = self._detect_capabilities()

    def _detect_capabilities(self) -> set[str]:
        """
        Introspect responses.create() to see which kwargs are supported.
        """
        try:
            fn = self._client.responses.create
            sig = inspect.signature(fn)
            return set(sig.parameters.keys())
        except Exception:
            # Conservative fallback
            return set()

    def create_text_response(
            self,
            model: str,
            prompt: str,
            *,
            response_format: Optional[Dict[str, Any]] = None,
            stream: bool = False,
    ) -> TextResponse:
        last_err = None

        for attempt in range(1, self._max_retries + 1):
            try:
                # Build kwargs dynamically to avoid unsupported args
                kwargs = {
                    "model": model,
                    "input": prompt,
                }

                # Only include stream if explicitly requested
                if stream and "stream" in self._capabilities:
                    kwargs["stream"] = True

                # Only include response_format if provided AND supported
                if response_format is not None and "response_format" in self._capabilities:
                    kwargs["response_format"] = response_format

                logger.debug(
                    "LLM request: model=%s, prompt_len=%d, stream=%s, response_format=%s",
                    model,
                    len(prompt),
                    stream,
                    bool(response_format),
                )

                resp = self._client.responses.create(**kwargs)

                text = self._extract_text(resp)
                raw = self._to_dict(resp)

                return TextResponse(text=text, raw=raw)

            except TypeError:
                # Programming error → fail fast
                raise

            except Exception as e:
                last_err = e
                logger.warning(
                    "LLM call failed (attempt %s/%s): %s",
                    attempt,
                    self._max_retries,
                    e,
                )
                if attempt < self._max_retries:
                    time.sleep(self._backoff_base ** attempt)

        raise RuntimeError("LLM request failed after retries") from last_err

    # -------------------------
    # Internal helpers
    # -------------------------

    def _extract_text(self, response: Any) -> str:
        """
        Best-effort extraction across SDK versions.
        """
        try:
            if hasattr(response, "output_text") and response.output_text:
                return response.output_text

            chunks = []

            if hasattr(response, "output") and isinstance(response.output, list):
                for block in response.output:
                    content = block.get("content") if isinstance(block, dict) else None
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "output_text":
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
        try:
            if hasattr(response, "model_dump"):
                return response.model_dump()
            if hasattr(response, "dict"):
                return response.dict()
        except Exception:
            pass
        return {"repr": repr(response)}

    # -------------------------
    # Convenience parsing
    # -------------------------

    @staticmethod
    def safe_json(text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except Exception:
            return None
