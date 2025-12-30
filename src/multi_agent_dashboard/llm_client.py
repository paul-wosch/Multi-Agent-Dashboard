# llm_client.py
import io
import time
import logging
import json
import inspect
from typing import Any, Dict, List, Optional, Callable
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
    # Optional usage metadata
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency: Optional[float] = None  # seconds

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
        files: Optional[List[Dict[str, Any]]] = None,
        tools_config: Optional[Dict[str, Any]] = None,
        reasoning_config: Optional[Dict[str, Any]] = None,
    ) -> TextResponse:
        """
        Execute a text-generation request and return a normalized response.
        """
        last_err: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                kwargs = {
                    "model": model,
                }

                if files:
                    kwargs["input"] = self._build_input_with_files(prompt, files)
                else:
                    kwargs["input"] = prompt

                if stream and "stream" in self._capabilities:
                    kwargs["stream"] = True

                # 1) response_format (if supported)
                if response_format is not None and "response_format" in self._capabilities:
                    kwargs["response_format"] = response_format

                # 2) tools / tool_choice / include (independent of response_format)
                if tools_config:
                    tools = tools_config.get("tools")
                    if tools and "tools" in self._capabilities:
                        kwargs["tools"] = tools
                    tool_choice = tools_config.get("tool_choice")
                    if tool_choice and "tool_choice" in self._capabilities:
                        kwargs["tool_choice"] = tool_choice
                    include = tools_config.get("include")
                    if include and "include" in self._capabilities:
                        kwargs["include"] = include

                # 3) reasoning config (independent of response_format)
                if reasoning_config and "reasoning" in self._capabilities:
                    kwargs["reasoning"] = reasoning_config

                logger.debug(
                    "LLM request: model=%s, prompt_len=%d, stream=%s, structured=%s, tools=%s, reasoning=%s",
                    model,
                    len(prompt),
                    stream,
                    bool(response_format),
                    bool(tools_config),
                    bool(reasoning_config),
                )

                start_ts = time.perf_counter()
                response = self._client.responses.create(**kwargs)
                end_ts = time.perf_counter()
                latency = end_ts - start_ts

                raw_dict = self._to_dict(response)

                # Best-effort usage extraction (OpenAI Responses API)
                input_tokens = None
                output_tokens = None
                try:
                    # Typical structure: {"usage": {"input_tokens": ..., "output_tokens": ...}}
                    usage = raw_dict.get("usage") or {}
                    input_tokens = usage.get("input_tokens")
                    output_tokens = usage.get("output_tokens")
                except Exception:
                    logger.debug("No usage metadata found in LLM response", exc_info=True)

                return TextResponse(
                    text=self._extract_text(response),
                    raw=raw_dict,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency=latency,
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
    # FILE HANDLING HELPERS
    # -------------------------

    def _upload_file(self, f: Dict[str, Any]) -> str:
        """
        Upload a file to OpenAI and return its file_id.
        """
        file_obj = io.BytesIO(f["content"])
        file_obj.name = f["filename"]  # IMPORTANT: SDK reads filename from here

        uploaded = self._client.files.create(
            file=file_obj,
            purpose="assistants",
        )

        return uploaded.id

    def _build_input_with_files(
            self,
            prompt: str,
            files: List[Dict[str, Any]],
    ) -> list[dict]:
        """
        Build OpenAI Responses API-compatible input payload.
        Automatically inlines any text file and uploads binary files.
        """
        content: list[dict] = [
            {"type": "input_text", "text": prompt}
        ]

        for f in files:
            file_content = f["content"]
            filename = f["filename"]

            # Attempt to decode as UTF-8 to determine if it's text
            try:
                text = file_content.decode("utf-8")
                is_text = True
            except UnicodeDecodeError:
                # Not valid UTF-8 → treat as binary
                text = ""
                is_text = False

            # Inline text files
            if is_text:
                content.append({
                    "type": "input_text",
                    "text": f"\n\n--- FILE: {filename} ---\n{text}",
                })
            else:
                # Upload binary files
                file_obj = io.BytesIO(file_content)
                file_obj.name = filename
                uploaded = self._client.files.create(file=file_obj, purpose="assistants")
                content.append({
                    "type": "input_file",
                    "file_id": uploaded.id
                })

        return [{"role": "user", "content": content}]

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
        Convert SDK response into a serializable dict (best-effort),
        avoiding noisy Pydantic serializer warnings from the SDK's
        internal model graph.
        """
        try:
            # Prefer the SDK's own helper, if present
            if hasattr(response, "to_dict") and callable(getattr(response, "to_dict")):
                return response.to_dict()

            # Fall back to Pydantic's model_dump, but silence its
            # "PydanticSerializationUnexpectedValue" warnings, which
            # are expected with some of the internal tool-call types.
            if hasattr(response, "model_dump"):
                try:
                    return response.model_dump(warnings="none")
                except TypeError:
                    # Older Pydantic / SDK versions may not accept warnings=
                    return response.model_dump()

            if hasattr(response, "dict"):
                return response.dict()
        except Exception:
            logger.exception("Failed to convert LLM response to dict")

        # Last-resort fallback
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
