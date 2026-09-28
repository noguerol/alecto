"""Unit tests for SSE streaming (spec §5.2, §5.3, §9.4, AC-05/AC-06).

Golden tests for :func:`alecto.streaming.iter_sse_events` plus the
streaming-aware performance request path.
"""

import asyncio
import json
import time

import httpx
import pytest

from alecto.adapters import MockAdapter, OpenAIAdapter
from alecto.domain import TargetSpec
from alecto.enums import TargetKind
from alecto.errors import CapabilityUnavailable, StreamMalformedError
from alecto.performance import (
    PerformanceCell,
    compute_timing_metrics,
    make_streaming_request_fn,
    run_performance_cell,
    stream_request_observation,
)
from alecto.streaming import MAX_EVENT_BYTES, StreamEvent, iter_sse_events


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class FakeResponse:
    """Minimal response duck-type exposing ``aiter_bytes``."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


def sse_bytes(*payloads: str, done: bool = True) -> bytes:
    body = "".join(f"data: {p}\n\n" for p in payloads)
    if done:
        body += "data: [DONE]\n\n"
    return body.encode("utf-8")


async def collect(response: FakeResponse) -> list[str]:
    return [event async for event in iter_sse_events(response)]


def openai_spec(**overrides) -> TargetSpec:
    kwargs = {
        "kind": TargetKind.OPENAI_COMPATIBLE,
        "endpoint": "http://test/v1",
        "model": "test-model",
    }
    kwargs.update(overrides)
    return TargetSpec(**kwargs)


def adapter_with(handler) -> OpenAIAdapter:
    spec = openai_spec()
    adapter = OpenAIAdapter(spec)
    # Swap in a mock transport client (the real client is never used).
    adapter._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://test/v1/",
    )
    return adapter


# ---------------------------------------------------------------------------
# StreamEvent
# ---------------------------------------------------------------------------


class TestStreamEvent:
    def test_required_fields_and_defaults(self):
        event = StreamEvent(seq=0, t_client_ns=123, kind="role")
        assert event.seq == 0
        assert event.t_client_ns == 123
        assert event.kind == "role"
        assert event.content_delta is None
        assert event.reasoning_delta is None
        assert event.usage is None
        assert event.finish_reason is None
        assert event.raw is None

    def test_as_dict_roundtrip(self):
        event = StreamEvent(
            seq=1,
            t_client_ns=9,
            kind="content",
            content_delta="hi",
            raw='{"x":1}',
        )
        d = event.as_dict()
        assert d["kind"] == "content"
        assert d["content_delta"] == "hi"
        assert json.dumps(d)  # JSON-serializable


# ---------------------------------------------------------------------------
# iter_sse_events golden tests (AC-05)
# ---------------------------------------------------------------------------


class TestSSEParser:
    @pytest.mark.asyncio
    async def test_arbitrary_tcp_fragmentation_every_byte(self):
        body = sse_bytes('{"a": 1}', '{"b": 2}', '{"c": 3}')
        # Split at *every* byte boundary — the parser must reassemble.
        chunks = [body[i : i + 1] for i in range(len(body))]
        events = await collect(FakeResponse(chunks))
        assert [json.loads(e) for e in events] == [{"a": 1}, {"b": 2}, {"c": 3}]

    @pytest.mark.asyncio
    async def test_fragmentation_in_data_field_prefix(self):
        # The literal "data:" prefix itself is split across chunks.
        body = sse_bytes('{"ok": true}')
        events = await collect(FakeResponse([body[:2], body[2:6], body[6:]]))
        assert [json.loads(e) for e in events] == [{"ok": True}]

    @pytest.mark.asyncio
    async def test_multiple_tokens_per_chunk(self):
        # A single content delta may contain many tokens; it is one event.
        payload = '{"choices":[{"delta":{"content":"one two three four five"}}]}'
        events = await collect(FakeResponse([sse_bytes(payload)]))
        assert len(events) == 1
        assert json.loads(events[0])["choices"][0]["delta"]["content"] == (
            "one two three four five"
        )

    @pytest.mark.asyncio
    async def test_multiple_events_in_one_chunk(self):
        events = await collect(FakeResponse([sse_bytes('{"i": 1}', '{"i": 2}', '{"i": 3}')]))
        assert [json.loads(e)["i"] for e in events] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_multiline_data_field(self):
        raw = b'data: {"a":\ndata: 1}\n\n'
        events = await collect(FakeResponse([raw]))
        assert len(events) == 1
        assert json.loads(events[0]) == {"a": 1}

    @pytest.mark.asyncio
    async def test_comment_and_heartbeat_lines_ignored(self):
        raw = b": keep-alive\n\ndata: {\"x\": 1}\n\n: another\n\n"
        events = await collect(FakeResponse([raw]))
        assert [json.loads(e) for e in events] == [{"x": 1}]

    @pytest.mark.asyncio
    async def test_reasoning_only_event(self):
        payload = '{"choices":[{"delta":{"reasoning_content":"thinking"}}]}'
        events = await collect(FakeResponse([sse_bytes(payload)]))
        assert len(events) == 1
        assert json.loads(events[0])["choices"][0]["delta"]["reasoning_content"] == "thinking"

    @pytest.mark.asyncio
    async def test_usage_only_final_event(self):
        usage = '{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}'
        events = await collect(FakeResponse([sse_bytes('{"choices":[{"delta":{"content":"x"}}]}', usage)]))
        assert len(events) == 2
        assert json.loads(events[1])["usage"]["total_tokens"] == 7

    @pytest.mark.asyncio
    async def test_done_sentinel_terminates(self):
        body = sse_bytes('{"a": 1}', '{"b": 2}') + b'data: not-json-after-done\n\n'
        events = await collect(FakeResponse([body]))
        assert [json.loads(e) for e in events] == [{"a": 1}, {"b": 2}]

    @pytest.mark.asyncio
    async def test_done_sentinel_without_trailing_blank_line(self):
        body = b'data: {"a": 1}\n\ndata: [DONE]'
        events = await collect(FakeResponse([body]))
        assert [json.loads(e) for e in events] == [{"a": 1}]

    @pytest.mark.asyncio
    async def test_trailing_event_without_blank_line_flushed(self):
        events = await collect(FakeResponse([b'data: {"a": 1}']))
        assert [json.loads(e) for e in events] == [{"a": 1}]

    @pytest.mark.asyncio
    async def test_crlf_line_endings(self):
        raw = b'data: {"a": 1}\r\n\r\ndata: [DONE]\r\n\r\n'
        events = await collect(FakeResponse([raw]))
        assert [json.loads(e) for e in events] == [{"a": 1}]

    @pytest.mark.asyncio
    async def test_malformed_json_raises_typed_error(self):
        with pytest.raises(StreamMalformedError) as excinfo:
            await collect(FakeResponse([sse_bytes("{not valid json}")]))
        err = excinfo.value
        assert err.code == "alecto.stream.malformed"
        # Bounded raw evidence is preserved.
        assert "not valid json" in err.context["raw"]
        assert err.context["raw_length"] > 0

    @pytest.mark.asyncio
    async def test_malformed_raw_excerpt_is_bounded(self):
        payload = "x" * (MAX_EVENT_BYTES - 1) + "{"  # invalid JSON, huge
        with pytest.raises(StreamMalformedError) as excinfo:
            await collect(FakeResponse([sse_bytes(payload)]))
        raw = excinfo.value.context["raw"]
        assert len(raw) <= MAX_EVENT_BYTES + 64

    @pytest.mark.asyncio
    async def test_oversized_event_raises(self):
        huge = b"data: " + b"x" * (MAX_EVENT_BYTES + 1) + b"\n\n"
        with pytest.raises(StreamMalformedError) as excinfo:
            await collect(FakeResponse([huge]))
        assert excinfo.value.code == "alecto.stream.malformed"

    @pytest.mark.asyncio
    async def test_oversized_unterminated_line_raises(self):
        huge = b"data: " + b"x" * (MAX_EVENT_BYTES + 1)
        with pytest.raises(StreamMalformedError):
            await collect(FakeResponse([huge]))


# ---------------------------------------------------------------------------
# OpenAIAdapter.stream
# ---------------------------------------------------------------------------


class TestOpenAIAdapterStream:
    @pytest.mark.asyncio
    async def test_stream_parses_all_event_kinds(self):
        payloads = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": ", "}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "t1"}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"total_tokens": 9}},
        ]

        def handler(request):
            return httpx.Response(200, content=sse_bytes(*[json.dumps(p) for p in payloads]))

        adapter = adapter_with(handler)
        try:
            events = [e async for e in adapter.stream({"messages": []})]
        finally:
            await adapter.aclose()

        assert [e.kind for e in events] == [
            "role",
            "reasoning",
            "content",
            "content",
            "tool",
            "finish",
            "usage",
        ]
        assert [e.seq for e in events] == list(range(len(events)))
        assert events[1].reasoning_delta == "think"
        assert events[2].content_delta == "Hello"
        assert events[5].finish_reason == "stop"
        assert events[6].usage == {"total_tokens": 9}
        assert all(e.t_client_ns > 0 for e in events)
        assert all(e.raw is not None for e in events)

    @pytest.mark.asyncio
    async def test_stream_requests_usage_and_stream_flags(self):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=sse_bytes('{"choices":[]}'))

        adapter = adapter_with(handler)
        try:
            _ = [e async for e in adapter.stream({"messages": [], "temperature": 0.3})]
        finally:
            await adapter.aclose()

        assert captured["body"]["stream"] is True
        assert captured["body"]["stream_options"] == {"include_usage": True}
        assert captured["body"]["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_stream_malformed_json_raises(self):
        def handler(request):
            return httpx.Response(200, content=b"data: {bad}\n\n")

        adapter = adapter_with(handler)
        try:
            with pytest.raises(StreamMalformedError):
                _ = [e async for e in adapter.stream({"messages": []})]
        finally:
            await adapter.aclose()

    @pytest.mark.asyncio
    async def test_base_adapter_stream_is_capability_unavailable(self):
        adapter = MockAdapter(openai_spec(kind=TargetKind.MOCK, endpoint="mock://x"))
        with pytest.raises(CapabilityUnavailable) as excinfo:
            _ = [e async for e in adapter.stream({"messages": []})]
        assert excinfo.value.code == "alecto.backend.capability_missing"


# ---------------------------------------------------------------------------
# Streaming-aware performance path (AC-06)
# ---------------------------------------------------------------------------


class FakeStreamingAdapter:
    """Duck-typed adapter yielding deterministic, real-timestamp events."""

    def __init__(self, gap_s: float = 0.01, usage_tokens: int = 5):
        self.gap_s = gap_s
        self.usage_tokens = usage_tokens
        self.closed = False

    async def stream(self, step):
        yield StreamEvent(seq=0, t_client_ns=time.perf_counter_ns(), kind="role")
        await asyncio.sleep(self.gap_s)
        yield StreamEvent(
            seq=1,
            t_client_ns=time.perf_counter_ns(),
            kind="content",
            content_delta="hello world",
        )
        yield StreamEvent(
            seq=2,
            t_client_ns=time.perf_counter_ns(),
            kind="usage",
            usage={"total_tokens": self.usage_tokens},
        )

    async def aclose(self):
        self.closed = True


class TestStreamingPerformancePath:
    @pytest.mark.asyncio
    async def test_observation_divides_ttfb_and_ttft_from_real_events(self):
        # Deterministic gap between the role opener and the first content.
        gap_ns = 7_000_000  # 7 ms
        base = time.perf_counter_ns()

        class DeterministicAdapter:
            async def stream(self, step):
                yield StreamEvent(seq=0, t_client_ns=base, kind="role")
                yield StreamEvent(
                    seq=1, t_client_ns=base + gap_ns, kind="content", content_delta="x"
                )
                yield StreamEvent(
                    seq=2,
                    t_client_ns=base + gap_ns + 1,
                    kind="usage",
                    usage={"total_tokens": 2},
                )

        obs = await stream_request_observation(DeterministicAdapter(), {"messages": []})
        assert obs.t_first_byte is not None
        assert obs.t_first is not None
        # Both timestamps come from the event stream; role-only does not count
        # as first output, so TTFB and TTFT are genuinely distinct.
        assert obs.t_first - obs.t_first_byte == pytest.approx(0.007, abs=0.0005)
        assert obs.t_first > obs.t_first_byte
        assert obs.output_tokens == 2

    @pytest.mark.asyncio
    async def test_reasoning_only_counts_as_first_output(self):
        gap_ns = 4_000_000  # 4 ms
        base = time.perf_counter_ns()

        class A:
            async def stream(self, step):
                yield StreamEvent(seq=0, t_client_ns=base, kind="role")
                yield StreamEvent(
                    seq=1,
                    t_client_ns=base + gap_ns,
                    kind="reasoning",
                    reasoning_delta="think",
                )

        obs = await stream_request_observation(A(), {})
        assert obs.t_first_byte is not None and obs.t_first is not None
        # Reasoning is a supported first-output channel (§9.4).
        assert obs.t_first - obs.t_first_byte == pytest.approx(0.004, abs=0.0005)

    @pytest.mark.asyncio
    async def test_cell_streaming_yields_ttft_not_equal_ttfb(self):
        cell = PerformanceCell(
            cell_id="stream-1", samples=3, concurrency=1, warmup=0, streaming=True
        )
        fn = make_streaming_request_fn(lambda: FakeStreamingAdapter(gap_s=0.015), {})
        result = await run_performance_cell(cell, fn)
        m = result.metrics
        assert m.n == 3
        assert m.ttfb_ms is not None and m.ttft_ms is not None
        assert m.ttft_ms > m.ttfb_ms
        assert m.ttft_ms >= 5.0  # real 15 ms gap, not a fabricated ~0
        assert m.ttft_status is None and m.ttfb_status is None
        assert m.e2e_ms is not None and m.e2e_ms >= m.ttft_ms

    @pytest.mark.asyncio
    async def test_mocked_httpx_stream_end_to_end(self):
        payloads = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "done"}}]},
            {"choices": [], "usage": {"total_tokens": 4}},
        ]

        async def gen():
            for i, p in enumerate(payloads):
                if i == 1:
                    await asyncio.sleep(0.02)  # gap: role -> first content
                yield f"data: {json.dumps(p)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        def handler(request):
            return httpx.Response(200, content=gen())

        def factory():
            return adapter_with(handler)

        cell = PerformanceCell(
            cell_id="stream-2", samples=2, concurrency=1, warmup=0, streaming=True
        )
        result = await run_performance_cell(cell, make_streaming_request_fn(factory, {}))
        m = result.metrics
        assert m.n == 2
        assert m.ttfb_ms is not None and m.ttft_ms is not None
        assert m.ttft_ms > m.ttfb_ms
        assert m.ttft_ms >= 5.0

    @pytest.mark.asyncio
    async def test_make_streaming_request_fn_closes_adapter(self):
        created: list[FakeStreamingAdapter] = []

        def factory():
            adapter = FakeStreamingAdapter(gap_s=0.0)
            created.append(adapter)
            return adapter

        fn = make_streaming_request_fn(factory, {})
        await fn("req-1")
        assert len(created) == 1
        assert created[0].closed is True
        assert created[0].gap_s == 0.0

    @pytest.mark.asyncio
    async def test_make_streaming_request_fn_propagates_capability_error(self):
        adapter = MockAdapter(openai_spec(kind=TargetKind.MOCK, endpoint="mock://x"))
        fn = make_streaming_request_fn(lambda: adapter, {})
        with pytest.raises(CapabilityUnavailable):
            await fn("req-1")


class TestUnsupportedTTFT:
    def test_compute_timing_metrics_none_first_event(self):
        m = compute_timing_metrics(
            [0.0, 0.1],
            None,
            [0.1, 0.2],
            [10, 20],
            wall_clock_s=0.2,
        )
        assert m.n == 2
        assert m.ttft_ms is None
        assert m.ttft_status == "unsupported"
        assert m.ttfb_ms is None
        assert m.ttfb_status == "unsupported"
        assert m.e2e_ms == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_non_streaming_cell_marks_ttft_unsupported(self):
        # A non-streaming request path does not report t_first: TTFT must be
        # null + status, never a fabricated ~0 value.
        async def request_fn(request_id: str) -> dict:
            await asyncio.sleep(0.005)
            return {"output_tokens": 3}

        cell = PerformanceCell(cell_id="nostream", samples=4, concurrency=1, warmup=0)
        result = await run_performance_cell(cell, request_fn)
        assert result.metrics.n == 4
        assert result.metrics.ttft_ms is None
        assert result.metrics.ttft_status == "unsupported"
        assert result.metrics.ttfb_ms is None
        assert result.metrics.e2e_ms is not None
        assert result.metrics.e2e_ms > 0
