"""Target adapters for different LLM backends (spec §4.7)."""

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .domain import TargetSpec
from .enums import TargetKind
from .errors import (
    CapabilityUnavailable,
    StreamMalformedError,
    TargetUnreachableError,
)
from .errors import (
    TimeoutError as AlectoTimeout,
)
from .streaming import StreamEvent, iter_sse_events


def _resolve_api_key(spec: TargetSpec) -> str:
    """Resolve the API key: prefer ``spec.api_key_env`` from the environment,
    fall back to ``spec.extra['api_key']``."""
    if spec.api_key_env:
        return os.environ.get(spec.api_key_env, "")
    return spec.extra.get("api_key", "") or ""


def _parse_openai_chunk(
    data: dict[str, Any],
    *,
    seq: int,
    t_client_ns: int,
    raw: str,
) -> StreamEvent:
    """Map one OpenAI ``chat/completions`` SSE chunk to a :class:`StreamEvent`.

    Handles content deltas (possibly several tokens), ``reasoning_content``
    deltas, tool-call deltas, role-only openers, finish events and the final
    usage-only event emitted because of ``stream_options.include_usage``.
    """
    choices = data.get("choices") or []
    choice: dict[str, Any] = choices[0] if choices else {}
    delta: dict[str, Any] = choice.get("delta") or {}

    content = delta.get("content")
    reasoning = delta.get("reasoning_content")
    tool_calls = delta.get("tool_calls")
    finish_reason = choice.get("finish_reason")
    usage = data.get("usage")

    content_delta = content if content else None
    reasoning_delta = reasoning if reasoning else None

    if content_delta is not None:
        kind = "content"
    elif reasoning_delta is not None:
        kind = "reasoning"
    elif tool_calls:
        kind = "tool"
    elif finish_reason:
        kind = "finish"
    elif usage is not None:
        kind = "usage"
    elif delta.get("role"):
        kind = "role"
    else:
        kind = "content"

    return StreamEvent(
        seq=seq,
        t_client_ns=t_client_ns,
        kind=kind,  # type: ignore[arg-type]
        content_delta=content_delta,
        reasoning_delta=reasoning_delta,
        usage=usage,
        finish_reason=finish_reason,
        raw=raw,
    )


class TargetAdapter(ABC):
    """Base adapter for LLM target backends."""

    def __init__(self, spec: TargetSpec):
        self.spec = spec

    @abstractmethod
    async def call(self, step: dict[str, Any]) -> dict[str, Any]:
        """Execute a step against the target."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the target is reachable."""
        ...

    async def stream(self, step: dict[str, Any]) -> AsyncIterator[StreamEvent]:
        """Stream a step from the target (spec §5.2–§5.3).

        Streaming is an *optional* capability. The default implementation
        raises :class:`~alecto.errors.CapabilityUnavailable` when iterated,
        so callers must probe rather than invent native timing estimates.
        """
        raise CapabilityUnavailable(
            f"{type(self).__name__} does not support streaming"
        )
        yield  # pragma: no cover - makes this an async generator

    async def aclose(self) -> None:
        """Close the underlying client. Default: nothing to close."""
        return None

    async def close(self) -> None:
        """Alias for :meth:`aclose` (sync-style close)."""
        await self.aclose()


class OpenAIAdapter(TargetAdapter):
    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        api_key = _resolve_api_key(spec)
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Ensure base_url ends with '/' so httpx joins relative paths correctly
        # (e.g. http://host:8081/v1 + /chat/completions ->
        #  http://host:8081/v1/chat/completions).
        base_url = spec.endpoint.rstrip("/") + "/"
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=spec.timeout_s,
            headers=headers,
        )

    def _build_payload(self, step: dict[str, Any], *, stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": step.get("messages", []),
            "temperature": step.get("temperature", 0.7),
        }
        max_tokens = step.get("max_tokens")
        if max_tokens is None:
            max_tokens = self.spec.extra.get("max_tokens")
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stream:
            payload["stream"] = True
            # Ask the endpoint to emit a final usage-only event (§9.4).
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def call(self, step: dict[str, Any]) -> dict[str, Any]:
        payload = self._build_payload(step)
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return {
                "text": data["choices"][0]["message"]["content"],
                "tokens": data.get("usage", {}).get("total_tokens", 0),
                "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
                "raw": data,
            }
        except httpx.TimeoutException:
            raise AlectoTimeout(f"Timeout calling {self.spec.endpoint}")
        except httpx.HTTPStatusError as e:
            raise TargetUnreachableError(f"HTTP {e.response.status_code}")
        except httpx.TransportError as e:
            raise TargetUnreachableError(f"Connection error: {e}")

    async def stream(self, step: dict[str, Any]) -> AsyncIterator[StreamEvent]:
        """Stream a chat completion as typed :class:`StreamEvent` records.

        Uses the OpenAI ``chat/completions`` streaming protocol with
        ``stream_options.include_usage`` and timestamps every event with
        :func:`time.perf_counter_ns` so TTFT/TTFB are real client
        observations rather than fabricated estimates (spec §9.4).
        """
        payload = self._build_payload(step, stream=True)
        seq = 0
        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for raw in iter_sse_events(response):
                    t_ns = time.perf_counter_ns()
                    try:
                        data = json.loads(raw)
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise StreamMalformedError(
                            f"malformed SSE JSON: {exc}",
                            raw=raw[:4096],
                            raw_length=len(raw),
                        ) from exc
                    yield _parse_openai_chunk(data, seq=seq, t_client_ns=t_ns, raw=raw)
                    seq += 1
        except httpx.TimeoutException:
            raise AlectoTimeout(f"Timeout streaming {self.spec.endpoint}")
        except httpx.HTTPStatusError as e:
            raise TargetUnreachableError(f"HTTP {e.response.status_code}")
        except httpx.TransportError as e:
            raise TargetUnreachableError(f"Connection error: {e}")

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/models")
            return response.status_code < 400
        except httpx.TimeoutException:
            return False
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


class OllamaAdapter(TargetAdapter):
    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        base_url = spec.endpoint.rstrip("/") + "/"
        self._client = httpx.AsyncClient(base_url=base_url, timeout=spec.timeout_s)

    async def call(self, step: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.spec.model or "llama3",
            "messages": step.get("messages", []),
        }
        max_tokens = step.get("max_tokens")
        if max_tokens is None:
            max_tokens = self.spec.extra.get("max_tokens")
        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}
        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return {
                "text": data["message"]["content"],
                "tokens": data.get("count", 0),
                "finish_reason": data.get("done", None) and "stop" or None,
                "raw": data,
            }
        except httpx.TimeoutException:
            raise AlectoTimeout(f"Timeout calling {self.spec.endpoint}")
        except httpx.HTTPStatusError as e:
            raise TargetUnreachableError(f"HTTP {e.response.status_code}")
        except httpx.TransportError as e:
            raise TargetUnreachableError(f"Connection error: {e}")

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/api/tags")
            return response.status_code < 400
        except httpx.TimeoutException:
            return False
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


class LLMStudioAdapter(TargetAdapter):
    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        base_url = spec.endpoint.rstrip("/") + "/"
        self._client = httpx.AsyncClient(base_url=base_url, timeout=spec.timeout_s)

    async def call(self, step: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "input": step.get("input", ""),
        }
        max_tokens = step.get("max_tokens")
        if max_tokens is None:
            max_tokens = self.spec.extra.get("max_tokens")
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        try:
            response = await self._client.post("/v1/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return {
                "text": data.get("output", ""),
                "tokens": data.get("tokens", 0),
                "finish_reason": data.get("finish_reason"),
                "raw": data,
            }
        except httpx.TimeoutException:
            raise AlectoTimeout(f"Timeout calling {self.spec.endpoint}")
        except httpx.HTTPStatusError as e:
            raise TargetUnreachableError(f"HTTP {e.response.status_code}")
        except httpx.TransportError as e:
            raise TargetUnreachableError(f"Connection error: {e}")

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/v1/models")
            return response.status_code < 400
        except httpx.TimeoutException:
            return False
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


class MockAdapter(TargetAdapter):
    """Mock adapter for testing without a real backend."""

    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        self._call_count = 0

    async def call(self, step: dict[str, Any]) -> dict[str, Any]:
        self._call_count += 1
        return {
            "text": f"mock-response-{self._call_count}",
            "tokens": 10,
            "finish_reason": "stop",
            "raw": {"mock": True, "step": step},
        }

    async def health_check(self) -> bool:
        return True


def get_adapter(spec: TargetSpec) -> TargetAdapter:
    """Factory function to get the appropriate adapter for a target spec.

    Pure (synchronous) factory: no I/O is performed, so callers do not need
    to ``await`` it.
    """
    if spec.kind == TargetKind.MOCK or spec.extra.get("mock", False):
        return MockAdapter(spec)
    elif spec.kind in (TargetKind.OPENAI, TargetKind.OPENAI_COMPATIBLE):
        return OpenAIAdapter(spec)
    elif spec.kind == TargetKind.OLLAMA:
        return OllamaAdapter(spec)
    elif spec.kind in (TargetKind.LLM, TargetKind.LLMSTUDIO):
        return LLMStudioAdapter(spec)
    else:
        # Default to mock for unknown kinds
        return MockAdapter(spec)
