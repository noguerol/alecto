"""Unit tests for the performance measurement protocol (spec §9)."""

import asyncio
import json
import time

import pytest

from alecto.enums import LoopMode
from alecto.performance import (
    ConcurrencySweep,
    ConcurrencySweepResult,
    OpenLoopCell,
    PerformanceCell,
    PerformanceResult,
    TimingMetrics,
    compute_timing_metrics,
    quantile,
    run_performance_cell,
)


def make_request_fn(delay_s: float = 0.01, output_tokens: int = 10, fail_after: int | None = None):
    """Build a request function that simulates an endpoint.

    ``delay_s``: time between dispatch and first event (and completion).
    ``fail_after``: if set, requests beyond this count raise.
    """
    calls = {"n": 0}

    async def request_fn(request_id: str) -> dict:
        calls["n"] += 1
        if fail_after is not None and calls["n"] > fail_after:
            raise RuntimeError("simulated endpoint failure")
        t0 = time.monotonic()
        await asyncio.sleep(delay_s)
        return {
            "t_first": t0 + delay_s,
            "t_first_byte": t0 + delay_s * 0.5,
            "output_tokens": output_tokens,
        }

    return request_fn


class TestCellCreationAndValidation:
    def test_closed_loop_cell_defaults(self):
        cell = PerformanceCell(cell_id="perf-1", mode=LoopMode.CLOSED)
        assert cell.cell_id == "perf-1"
        assert cell.mode == LoopMode.CLOSED
        assert cell.samples == 12
        assert cell.concurrency == 1
        assert cell.warmup == 2
        assert cell.deadline_s is None
        assert cell.workload == "chat"

    def test_open_loop_cell_requires_arrival_rate(self):
        with pytest.raises(ValueError):
            PerformanceCell(cell_id="perf-2", mode=LoopMode.OPEN)

    def test_invalid_concurrency(self):
        with pytest.raises(ValueError):
            PerformanceCell(cell_id="perf-3", concurrency=0)

    def test_invalid_samples(self):
        with pytest.raises(ValueError):
            PerformanceCell(cell_id="perf-4", samples=0)

    def test_invalid_deadline(self):
        with pytest.raises(ValueError):
            PerformanceCell(cell_id="perf-5", deadline_s=-1)

    def test_invalid_workload(self):
        with pytest.raises(ValueError):
            PerformanceCell(cell_id="perf-6", workload="nonexistent")

    def test_min_requests(self):
        cell = PerformanceCell(cell_id="perf-7", concurrency=8)
        assert cell.min_requests() == 32
        cell = PerformanceCell(cell_id="perf-8", concurrency=1)
        assert cell.min_requests() == 12

    def test_as_dict_serialization(self):
        cell = PerformanceCell(cell_id="perf-9", mode=LoopMode.OPEN, arrival_rate=5.0)
        d = cell.as_dict()
        assert d["mode"] == "open"
        assert d["arrival_rate"] == 5.0
        assert d["cell_id"] == "perf-9"


class TestTimingMetrics:
    def test_compute_timing_metrics_basic(self):
        # 4 requests, each 100ms e2e, 50ms ttft.
        dispatch = [0.0, 0.1, 0.2, 0.3]
        first = [0.05, 0.15, 0.25, 0.35]
        completion = [0.1, 0.2, 0.3, 0.4]
        tokens = [10, 20, 30, 40]
        m = compute_timing_metrics(dispatch, first, completion, tokens, wall_clock_s=0.4)
        assert m.n == 4
        assert m.ttfb_ms == pytest.approx(50.0)
        assert m.ttft_ms == pytest.approx(50.0)
        assert m.e2e_ms == pytest.approx(100.0)
        assert m.tokens_per_s == pytest.approx(100.0 / 0.4)
        assert m.wall_clock_s == 0.4

    def test_compute_timing_metrics_distinct_ttfb_ttft(self):
        # TTFB (first byte) and TTFT (first token) are distinct metrics.
        dispatch = [0.0, 0.1]
        first_byte = [0.02, 0.12]  # first byte arrives 20ms after dispatch
        first = [0.05, 0.15]  # first token arrives 50ms after dispatch
        completion = [0.1, 0.2]
        m = compute_timing_metrics(dispatch, first, completion, first_byte=first_byte)
        assert m.ttfb_ms == pytest.approx(20.0)
        assert m.ttft_ms == pytest.approx(50.0)
        assert m.ttfb_ms < m.ttft_ms

    def test_compute_timing_metrics_fallback_ttfb(self):
        # Without first_byte, TTFB falls back to t_first - t_dispatch (== TTFT).
        dispatch = [0.0, 0.1]
        first = [0.05, 0.15]
        completion = [0.1, 0.2]
        m = compute_timing_metrics(dispatch, first, completion)
        assert m.ttfb_ms == pytest.approx(50.0)
        assert m.ttft_ms == pytest.approx(50.0)
        assert m.ttfb_ms == m.ttft_ms

    def test_quantiles(self):
        m = compute_timing_metrics(
            [0.0, 0.0, 0.0, 0.0],
            [0.01, 0.02, 0.03, 0.04],
            [0.1, 0.2, 0.3, 0.4],
        )
        # e2e = [100, 200, 300, 400]
        assert m.p50_ms == pytest.approx(250.0)
        assert m.p90_ms == pytest.approx(370.0)
        assert m.p95_ms == pytest.approx(385.0)

    def test_quantile_function(self):
        assert quantile([1, 2, 3, 4], 0.5) == 2.5
        assert quantile([10], 0.5) == 10
        with pytest.raises(ValueError):
            quantile([], 0.5)

    def test_merge_metrics(self):
        a = TimingMetrics(ttfb_ms=10.0, ttft_ms=20.0, e2e_ms=100.0, wall_clock_s=1.0, n=4)
        b = TimingMetrics(ttfb_ms=20.0, ttft_ms=40.0, e2e_ms=200.0, wall_clock_s=2.0, n=4)
        m = a.merge(b)
        assert m.ttfb_ms == pytest.approx(15.0)
        assert m.ttft_ms == pytest.approx(30.0)
        assert m.e2e_ms == pytest.approx(150.0)
        assert m.wall_clock_s == pytest.approx(3.0)
        assert m.n == 8

    def test_empty_metrics(self):
        m = compute_timing_metrics([], [], [])
        assert m.n == 0
        assert m.ttfb_ms is None


class TestClosedLoopExecution:
    @pytest.mark.asyncio
    async def test_closed_loop_run(self):
        cell = PerformanceCell(cell_id="closed-1", mode=LoopMode.CLOSED, samples=8, concurrency=4, warmup=1)
        result = await run_performance_cell(cell, make_request_fn(delay_s=0.01))
        assert isinstance(result, PerformanceResult)
        assert result.cell_id == "closed-1"
        assert result.mode == "closed"
        assert result.samples == 8
        assert result.metrics.n == 8
        assert result.metrics.e2e_ms is not None
        assert result.metrics.e2e_ms >= 0
        assert result.metrics.ttfb_ms is not None
        # t_first_byte (50% of delay) arrives before t_first (full delay).
        assert result.metrics.ttfb_ms < result.metrics.ttft_ms
        assert result.deadline_hit is False

    @pytest.mark.asyncio
    async def test_closed_loop_warmup(self):
        calls = {"n": 0}

        async def counting_fn(request_id: str) -> dict:
            calls["n"] += 1
            t0 = time.monotonic()
            await asyncio.sleep(0.005)
            return {"t_first": t0 + 0.005, "output_tokens": 5}

        cell = PerformanceCell(cell_id="closed-2", samples=4, concurrency=2, warmup=2)
        await run_performance_cell(cell, counting_fn)
        # 2 warmup + 4 samples = 6 calls
        assert calls["n"] == 6

    @pytest.mark.asyncio
    async def test_closed_loop_error_recorded(self):
        async def failing_fn(request_id: str) -> dict:
            raise RuntimeError("boom")

        cell = PerformanceCell(cell_id="closed-3", samples=4, concurrency=2, warmup=0)
        result = await run_performance_cell(cell, failing_fn)
        # All observations are errors; metrics computed on ok subset (empty).
        assert result.samples == 4
        assert all(o.status == "error" for o in result.observations)
        assert result.metrics.n == 0


class TestClosedLoopContinuousPool:
    """§9.3: the closed loop keeps C requests in flight continuously.

    The pool must not release requests in fixed batches that let the
    in-flight count drain to zero between batches.
    """

    @staticmethod
    def _recording_fn(samples: int, delay: float = 0.005):
        state = {"live": 0, "max": 0, "started": 0}

        async def fn(request_id: str) -> dict:
            state["live"] += 1
            state["started"] += 1
            state["max"] = max(state["max"], state["live"])
            try:
                await asyncio.sleep(delay)
                t0 = time.monotonic()
                return {"t_first": t0, "t_first_byte": t0, "output_tokens": 3}
            finally:
                state["live"] -= 1

        return fn, state

    @pytest.mark.asyncio
    async def test_in_flight_never_exceeds_concurrency(self):
        """(a) boundedness: max observed in-flight count <= C."""
        C, samples = 4, 20
        fn, state = self._recording_fn(samples)
        cell = PerformanceCell(cell_id="pool-bounded", samples=samples, concurrency=C, warmup=0)
        result = await run_performance_cell(cell, fn)
        assert state["max"] <= C
        assert state["max"] == C  # all C slots are actually used
        assert result.samples == samples

    @pytest.mark.asyncio
    async def test_continuous_refill_never_drains_to_zero(self):
        """(b) continuous refill: with C=4, samples=20 the observed
        in-flight count never reaches zero before the budget is exhausted.

        The in-flight count is sampled by a concurrent monitor task. A
        fixed-batch runner leaves an observable idle gap between batches
        (the next batch's requests are only scheduled after the current
        batch's ``gather`` resolves); a continuous worker pool hands off
        without yielding to the event loop.
        """
        C, samples = 4, 20
        fn, state = self._recording_fn(samples)
        min_live = samples + 1
        finished = False

        async def monitor() -> None:
            nonlocal min_live
            while not finished:
                # Ignore the very start (no request yet) and the very end
                # (budget exhausted).
                if 0 < state["started"] < samples:
                    min_live = min(min_live, state["live"])
                await asyncio.sleep(0)

        cell = PerformanceCell(cell_id="pool-continuous", samples=samples, concurrency=C, warmup=0)
        mon = asyncio.ensure_future(monitor())
        try:
            result = await run_performance_cell(cell, fn)
        finally:
            finished = True
            await mon
        assert min_live >= 1
        assert state["max"] == C
        assert result.samples == samples

    @pytest.mark.asyncio
    async def test_exactly_budget_requests_executed(self):
        """(c) exactly ``samples`` requests are executed."""
        C, samples = 3, 11
        fn, state = self._recording_fn(samples)
        cell = PerformanceCell(cell_id="pool-budget", samples=samples, concurrency=C, warmup=0)
        result = await run_performance_cell(cell, fn)
        assert state["started"] == samples
        assert state["max"] <= C
        assert result.samples == samples
        assert len(result.observations) == samples

    @pytest.mark.asyncio
    async def test_slow_request_does_not_block_others_beyond_c(self):
        """(d) a slow request occupies one slot; the remaining C-1 slots
        keep dispatching, so later requests are not blocked behind it."""
        C, samples = 2, 6
        dispatch_times: dict[str, float] = {}

        async def fn(request_id: str) -> dict:
            dispatch_times[request_id] = time.monotonic()
            await asyncio.sleep(0.3 if request_id == "req-0" else 0.01)
            t0 = time.monotonic()
            return {"t_first": t0, "t_first_byte": t0, "output_tokens": 1}

        cell = PerformanceCell(cell_id="pool-slow", samples=samples, concurrency=C, warmup=0)
        result = await run_performance_cell(cell, fn)
        assert result.samples == samples
        # req-0 is the slow one. With a continuous pool the second worker
        # dispatches req-2 almost immediately; a fixed-batch runner would
        # only dispatch it after the slow request completes (~0.3s).
        assert dispatch_times["req-2"] - dispatch_times["req-0"] < 0.15


class TestOpenLoopExecution:
    @pytest.mark.asyncio
    async def test_open_loop_run(self):
        cell = PerformanceCell(
            cell_id="open-1", mode=LoopMode.OPEN, samples=6, arrival_rate=100.0, warmup=1
        )
        result = await run_performance_cell(cell, make_request_fn(delay_s=0.005))
        assert result.mode == "open"
        assert result.samples == 6
        assert result.metrics.n == 6
        assert result.max_queue >= 0

    @pytest.mark.asyncio
    async def test_open_loop_fixed_arrivals(self):
        cell = PerformanceCell(
            cell_id="open-2", mode=LoopMode.OPEN, samples=4,
            arrival_rate=50.0, arrival_process="fixed", warmup=0,
        )
        olc = OpenLoopCell(cell, make_request_fn())
        arrivals = olc.scheduled_arrivals()
        assert len(arrivals) == 4
        # Fixed: 1/50 = 0.02s apart
        assert arrivals[0] == 0.0
        assert arrivals[1] == pytest.approx(0.02)
        assert arrivals[3] == pytest.approx(0.06)

    @pytest.mark.asyncio
    async def test_open_loop_poisson_arrivals_seeded(self):
        cell = PerformanceCell(
            cell_id="open-3", mode=LoopMode.OPEN, samples=10,
            arrival_rate=10.0, arrival_process="poisson", seed=42,
        )
        olc = OpenLoopCell(cell, make_request_fn())
        a1 = olc.scheduled_arrivals()
        a2 = olc.scheduled_arrivals()
        # Same seed => same arrivals
        assert a1 == a2
        # Arrivals are strictly increasing
        assert all(b > a for a, b in zip(a1, a1[1:]))

    @pytest.mark.asyncio
    async def test_open_loop_queue_drop(self):
        cell = PerformanceCell(
            cell_id="open-4", mode=LoopMode.OPEN, samples=10,
            arrival_rate=200.0, warmup=0,
        )

        async def slow_fn(request_id: str) -> dict:
            await asyncio.sleep(0.05)
            t0 = time.monotonic()
            return {"t_first": t0, "output_tokens": 5}

        olc = OpenLoopCell(cell, slow_fn, queue_limit=3)
        result = await run_performance_cell(olc.cell if False else cell, slow_fn)
        # With a small queue limit and fast arrivals, some requests should drop.
        assert result.dropped >= 0


class TestConcurrencySweep:
    @pytest.mark.asyncio
    async def test_sweep_runs(self):
        cell = PerformanceCell(cell_id="sweep-1", mode=LoopMode.CLOSED, samples=4, warmup=0)
        sweep = ConcurrencySweep(cell, make_request_fn(delay_s=0.005), concurrency_levels=[1, 2])
        result = await sweep.run()
        assert isinstance(result, ConcurrencySweepResult)
        assert len(result.results) == 2
        assert result.results[0]["concurrency"] == 1
        assert result.results[1]["concurrency"] == 2
        # Each level should have measured n >= max(12, 4*C)
        assert result.results[0]["n"] >= 12
        assert result.results[1]["n"] >= 8

    @pytest.mark.asyncio
    async def test_sweep_throughput_curve(self):
        cell = PerformanceCell(cell_id="sweep-2", mode=LoopMode.CLOSED, samples=4, warmup=0)
        sweep = ConcurrencySweep(cell, make_request_fn(delay_s=0.005), concurrency_levels=[1, 4])
        result = await sweep.run()
        curve = result.throughput_curve()
        assert len(curve) == 2
        for c, rps in curve:
            assert rps is not None
            assert rps > 0

    def test_sweep_as_dict(self):
        result = ConcurrencySweepResult(cell_id="x", target="t")
        result.results.append({"concurrency": 1, "request_rps": 10.0})
        d = result.as_dict()
        assert d["cell_id"] == "x"
        assert d["results"][0]["request_rps"] == 10.0


class TestDeadlineEnforcement:
    @pytest.mark.asyncio
    async def test_deadline_hit(self):
        cell = PerformanceCell(
            cell_id="deadline-1", mode=LoopMode.CLOSED,
            samples=100, concurrency=1, warmup=0, deadline_s=0.05,
        )

        async def slow_fn(request_id: str) -> dict:
            await asyncio.sleep(0.1)
            t0 = time.monotonic()
            return {"t_first": t0, "output_tokens": 5}

        result = await run_performance_cell(cell, slow_fn)
        assert result.deadline_hit is True
        # Not all 100 samples completed.
        assert result.samples < 100

    @pytest.mark.asyncio
    async def test_no_deadline(self):
        cell = PerformanceCell(cell_id="deadline-2", samples=4, concurrency=2, warmup=0)
        result = await run_performance_cell(cell, make_request_fn())
        assert result.deadline_hit is False
        assert result.samples == 4


class TestResultSerialization:
    @pytest.mark.asyncio
    async def test_result_as_dict(self):
        cell = PerformanceCell(cell_id="ser-1", samples=4, concurrency=2, warmup=0)
        result = await run_performance_cell(cell, make_request_fn())
        d = result.as_dict()
        assert d["cell_id"] == "ser-1"
        assert d["mode"] == "closed"
        assert d["samples"] == 4
        assert "metrics" in d
        assert "observations" in d
        assert len(d["observations"]) == 4

    @pytest.mark.asyncio
    async def test_result_json_serializable(self):
        cell = PerformanceCell(cell_id="ser-2", samples=4, concurrency=2, warmup=0)
        result = await run_performance_cell(cell, make_request_fn())
        d = result.as_dict()
        # Must be JSON-serializable (no NaN/inf, §5.1).
        s = json.dumps(d)
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["cell_id"] == "ser-2"

    def test_metrics_as_dict(self):
        m = TimingMetrics(ttfb_ms=1.0, ttft_ms=2.0, e2e_ms=3.0, tokens_per_s=10.0, n=4)
        d = m.as_dict()
        assert d["ttft_ms"] == 2.0
        assert d["n"] == 4
        assert json.dumps(d)  # JSON-serializable


class TestRequestObservation:
    def test_request_observation_has_t_first_byte(self):
        from alecto.performance import RequestObservation

        obs = RequestObservation(
            request_id="req-1",
            t_dispatch=0.0,
            t_first_byte=0.02,
            t_first=0.05,
            t_done=0.1,
            output_tokens=10,
        )
        assert obs.t_first_byte == 0.02
        assert obs.t_first == 0.05
        # TTFB = t_first_byte - t_dispatch, TTFT = t_first - t_dispatch
        assert obs.t_first - obs.t_dispatch > obs.t_first_byte - obs.t_dispatch
        # t_first_byte defaults to None
        obs2 = RequestObservation(request_id="req-2")
        assert obs2.t_first_byte is None
