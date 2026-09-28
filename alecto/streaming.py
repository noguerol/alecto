"""Server-Sent Events (SSE) streaming primitives (spec §5.2, §5.3, §9.4).

This module holds the transport-agnostic pieces shared by every streaming
adapter:

* :class:`StreamEvent` — the typed, timestamped event record required by
  spec §5.2 (event sequence, client monotonic timestamp, content/reasoning/
  tool delta, usage, finish reason).
* :func:`iter_sse_events` — a robust SSE parser that tolerates arbitrary
  TCP fragmentation, multi-line ``data:`` fields, comment/heartbeat lines,
  multiple events per chunk, the OpenAI ``[DONE]`` sentinel, and oversized
  or malicious chunks. Malformed JSON raises
  :class:`~alecto.errors.StreamMalformedError` (``alecto.stream.malformed``)
  with a bounded raw excerpt preserved as evidence (spec §19).

Nothing here knows about httpx specifically: any response object exposing
``aiter_bytes()`` (httpx streaming responses included) can be parsed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .errors import StreamMalformedError

# Bounds required by spec §19: 64 KiB SSE event, bounded raw evidence.
MAX_EVENT_BYTES: int = 64 * 1024
RAW_EXCERPT_LIMIT: int = 4096

EventKind = Literal["role", "content", "reasoning", "tool", "usage", "finish"]


@dataclass
class StreamEvent:
    """A single parsed streaming event (spec §5.2).

    ``t_client_ns`` is a client monotonic timestamp taken with
    :func:`time.perf_counter_ns` the moment the event was received/parsed.
    ``raw`` keeps the bounded raw JSON payload for evidence.
    """

    seq: int
    t_client_ns: int
    kind: EventKind
    content_delta: str | None = None
    reasoning_delta: str | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    raw: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _raw_excerpt(raw: str) -> str:
    """Bounded raw evidence excerpt (spec §19)."""
    excerpt = raw[:RAW_EXCERPT_LIMIT]
    if len(raw) > RAW_EXCERPT_LIMIT:
        excerpt += f"...[truncated {len(raw) - RAW_EXCERPT_LIMIT} chars]"
    return excerpt


def _malformed(message: str, raw: str) -> StreamMalformedError:
    return StreamMalformedError(
        message,
        raw=_raw_excerpt(raw),
        raw_length=len(raw),
    )


def _validate_payload(payload: str) -> None:
    """Validate that a ``data:`` payload is well-formed JSON.

    Raises :class:`StreamMalformedError` (never ``json.JSONDecodeError``) so
    callers have a single typed failure mode.
    """
    try:
        json.loads(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise _malformed(f"malformed SSE JSON: {exc}", payload) from exc


def _iter_lines(buffer: bytes):
    """Yield complete lines from *buffer*, returning the unconsumed tail."""
    while True:
        idx = buffer.find(b"\n")
        if idx < 0:
            return buffer
        line = buffer[:idx]
        buffer = buffer[idx + 1 :]
        yield line


async def iter_sse_events(response: Any) -> AsyncIterator[str]:
    """Parse an SSE byte stream and yield each event's raw ``data`` payload.

    The parser is intentionally aggressive about robustness:

    * arbitrary TCP fragmentation (a ``data:`` line may arrive one byte at a
      time) is accumulated until a newline terminator arrives;
    * multiple events in a single chunk are all emitted;
    * multi-line ``data:`` fields are joined with ``"\\n"`` as per the SSE
      specification;
    * blank lines terminate an event, comment lines (``:``) are ignored;
    * the OpenAI ``[DONE]`` sentinel ends iteration without yielding;
    * events larger than :data:`MAX_EVENT_BYTES` raise
      :class:`~alecto.errors.StreamMalformedError` instead of exhausting
      memory (malicious/oversized chunk defence, spec §19).

    Yields the raw JSON string for each event. Malformed JSON raises
    :class:`~alecto.errors.StreamMalformedError` with a bounded raw excerpt.
    """
    buffer = b""
    data_lines: list[str] = []
    data_bytes = 0

    def flush() -> str | None:
        nonlocal data_lines, data_bytes
        if not data_lines:
            return None
        payload = "\n".join(data_lines)
        data_lines = []
        data_bytes = 0
        return payload

    async for chunk in response.aiter_bytes():
        if not chunk:
            continue
        if isinstance(chunk, str):  # tolerate text-mode fakes/adapters
            chunk = chunk.encode("utf-8")
        buffer += chunk

        while True:
            idx = buffer.find(b"\n")
            if idx < 0:
                break
            line_bytes = buffer[:idx]
            buffer = buffer[idx + 1 :]
            line = line_bytes.rstrip(b"\r").decode("utf-8", errors="replace")

            if line == "":
                payload = flush()
                if payload is None:
                    continue
                if payload.strip() == "[DONE]":
                    return
                _validate_payload(payload)
                yield payload
            elif line.startswith(":"):
                # Comment / heartbeat line — ignore.
                continue
            elif line.startswith("data:"):
                value = line[5:]
                if value.startswith(" "):
                    value = value[1:]
                data_lines.append(value)
                data_bytes += len(value)
                if data_bytes > MAX_EVENT_BYTES:
                    raise _malformed(
                        f"SSE event exceeds {MAX_EVENT_BYTES} byte bound",
                        "\n".join(data_lines),
                    )
            # Other SSE fields (event:, id:, retry:) do not affect the
            # OpenAI-style payload and are ignored.

        # Any unterminated partial line must itself respect the event bound.
        if len(buffer) > MAX_EVENT_BYTES:
            raise _malformed(
                f"SSE line exceeds {MAX_EVENT_BYTES} byte bound",
                buffer.decode("utf-8", errors="replace"),
            )

    # Stream ended without a trailing blank line: process any final partial
    # line, then flush the accumulated event.
    if buffer:
        line = buffer.rstrip(b"\r").decode("utf-8", errors="replace")
        buffer = b""
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)
            data_bytes += len(value)
            if data_bytes > MAX_EVENT_BYTES:
                raise _malformed(
                    f"SSE event exceeds {MAX_EVENT_BYTES} byte bound",
                    "\n".join(data_lines),
                )
    payload = flush()
    if payload is not None and payload.strip() != "[DONE]":
        _validate_payload(payload)
        yield payload
