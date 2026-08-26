"""OpenAI-compatible JSON chat transport shared by RAG agents."""

from __future__ import annotations

import http.client
import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Type

from rag.core.io_utils import atomic_write_json, read_json


class ChatCompletionResult(str):
    """String response carrying non-scientific transport metadata."""

    response_metadata: dict[str, Any]

    def __new__(
        cls,
        content: object,
        response_metadata: dict[str, Any] | None = None,
    ) -> "ChatCompletionResult":
        instance = str.__new__(cls, "" if content is None else str(content))
        instance.response_metadata = dict(response_metadata or {})
        return instance


@dataclass(frozen=True)
class TransportConfig:
    timeout_seconds: float = 120.0
    retries: int = 1
    retry_backoff_seconds: float = 0.0
    temperature: float = 0.0
    thinking_mode: str | None = None
    max_output_tokens: int | None = None
    return_metadata: bool = True


@dataclass(frozen=True)
class CachedResponse:
    cache_key: str
    payload: dict[str, Any]
    path: Path


class LLMResponseCache:
    """Store only schema-validated JSON responses by complete request content."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    @staticmethod
    def build_key(
        *,
        call_type: str,
        messages: Sequence[dict[str, str]],
        model: str,
        base_url: str,
    ) -> str:
        material = {
            "call_type": call_type,
            "messages": list(messages),
            "model": model,
            "base_url": base_url.rstrip("/"),
        }
        return hashlib.sha256(
            json.dumps(
                material,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def lookup(
        self,
        *,
        call_type: str,
        messages: Sequence[dict[str, str]],
        model: str,
        base_url: str,
    ) -> CachedResponse | None:
        cache_key = self.build_key(
            call_type=call_type,
            messages=messages,
            model=model,
            base_url=base_url,
        )
        path = self.root / f"{cache_key}.json"
        if not path.is_file():
            return None
        try:
            value = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict) or value.get("cache_key") != cache_key:
            return None
        payload = value.get("payload")
        if not isinstance(payload, dict):
            return None
        return CachedResponse(cache_key, payload, path)

    def store_validated(
        self,
        *,
        call_type: str,
        messages: Sequence[dict[str, str]],
        model: str,
        base_url: str,
        payload: dict[str, Any],
    ) -> Path:
        cache_key = self.build_key(
            call_type=call_type,
            messages=messages,
            model=model,
            base_url=base_url,
        )
        path = self.root / f"{cache_key}.json"
        atomic_write_json(
            path,
            {
                "call_type": call_type,
                "model": model,
                "base_url": base_url.rstrip("/"),
                "cache_key": cache_key,
                "payload": payload,
            },
        )
        return path


def request_chat_completion(
    messages: Sequence[dict[str, str]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    config: TransportConfig | None = None,
    error_type: Type[RuntimeError] = RuntimeError,
    missing_context: str = " for LLM calls.",
    response_error_message: str = (
        "API response did not contain choices[0].message.content"
    ),
) -> str:
    """Send one JSON-object chat request with bounded transient retries."""

    settings = config or TransportConfig()
    if not base_url:
        raise error_type("OPENAI_BASE_URL is required" + missing_context)
    if not api_key:
        raise error_type("OPENAI_API_KEY is required" + missing_context)
    if not model:
        raise error_type("OPENAI_MODEL is required" + missing_context)

    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": settings.temperature,
        "response_format": {"type": "json_object"},
    }
    if settings.thinking_mode is not None:
        payload["thinking"] = {"type": settings.thinking_mode}
    if settings.max_output_tokens is not None:
        payload["max_tokens"] = settings.max_output_tokens
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    retries = max(1, int(settings.retries))
    response_payload: dict[str, Any] | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=float(settings.timeout_seconds)
            ) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            if not isinstance(decoded, dict):
                raise error_type("OpenAI-compatible API returned a non-object response")
            response_payload = decoded
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise error_type(
                f"OpenAI-compatible API HTTP {exc.code}: {body}"
            ) from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
        ) as exc:
            if attempt >= retries:
                if retries == 1:
                    message = f"OpenAI-compatible API request failed: {exc}"
                else:
                    message = (
                        "OpenAI-compatible API request failed after "
                        f"{retries} attempts: {exc}"
                    )
                raise error_type(message) from exc
            time.sleep(settings.retry_backoff_seconds * (2 ** (attempt - 1)))

    if response_payload is None:
        raise error_type("OpenAI-compatible API returned no response payload")
    try:
        choice = response_payload["choices"][0]
        message = choice["message"]
        content = message["content"] if "content" in message else None
    except (KeyError, IndexError, TypeError) as exc:
        raise error_type(response_error_message) from exc
    if not settings.return_metadata:
        return "" if content is None else str(content)
    reasoning_content = message.get("reasoning_content") or ""
    return ChatCompletionResult(
        content,
        {
            "finish_reason": str(choice.get("finish_reason") or ""),
            "usage": response_payload.get("usage") or {},
            "reasoning_content_chars": len(str(reasoning_content)),
            "thinking_mode": settings.thinking_mode or "",
        },
    )
