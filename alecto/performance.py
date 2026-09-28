"""Performance measurement protocol (spec §9).

Closed-loop and open-loop cells, timing metrics, concurrency sweeps,
and an async cell runner with deadline enforcement.

Metric IDs follow §9.4: ``alecto.perf.ttft_ms``, ``alecto.perf.e2e_ms``,
``alecto.perf.request_rps``, ``alecto.perf.output_tps``.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from .enums import LoopMode

# ---------------------------------------------------------------------------
# Workload definitions (§9.1)
# ---------------------------------------------------------------------------

WORKLOADS: dict[str, dict[str, int]] = {
    "short": {"input_tokens": 128, "output_tokens": 32},
    "chat": {"input_tokens": 1024, "output_tokens": 256},
    "decode": {"input_tokens": 256, "output_tokens": 1024},
    "prefill": {"input_tokens": 8192, "output_tokens": 128},
    "long_context": {"input_tokens": 32768, "output_tokens": 128},
    "occupied_context": {"input_tokens": 1024, "output_tokens": 256},
    "prefix_reuse": {"input_tokens": 8192, "output_tokens": 256},
    "mixed": {"input_tokens": 1024, "output_tokens": 256},
    "multiturn": {"input_tokens": 1024, "output_tokens": 128},
}


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


@dataclass
class PerformanceCell:
    """A single performance cell (spec §9.1–§9.3)."""

    cell_id: str
    mode: LoopMode = LoopMode.CLOSED
    target: str = ""
    samples: int = 12
    concurrency: int = 1
    warmup: int = 2
    iterations: int = 1
    deadline_s: float | None = None
    workload: str = "chat"
    arrival_rate: float | None = None  # requests/s, open loop only
    arrival_process: str = "poisson"  # 'poisson' | 'fixed'
    seed: int = 0
    cache_mode: str = "best_effort_unique"
    # When True the request path is expected to stream and to report real
    # TTFB/TTFT from SSE event timestamps (spec §5.3, §9.4). Non-streaming
    # cells leave TTFT unsupported rather than fabricating a value.
    streaming: bool = False

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ValueError("cell_id must be non-empty")
        if self.mode not in (LoopMode.CLOSED, LoopMode.OPEN):
            raise ValueError(f"mode must be closed_loop or open_loop, got {self.mode!r}")
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError("deadline_s must be positive when set")
        if self.workload not in WORKLOADS:
            raise ValueError(f"unknown workload {self.workload!r}")
        if self.arrival_process not in ("poisson", "fixed"):
            raise ValueError("arrival_process must be 'poisson' or 'fixed'")
        if self.mode == LoopMode.OPEN and self.arrival_rate is None:
            raise ValueError("open-loop cells require an arrival_rate")
        if self.arrival_rate is not None and self.arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")

    def min_requests(self) -> int:
        """Minimum diagnostic cell size (§9.3): max(12, 4*C)."""
        return max(12, 4 * self.concurrency)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        return d


# ---------------------------------------------------------------------------
# Timing metrics (§9.4)
# ---------------------------------------------------------------------------


@dataclass
class TimingMetrics:
    """Per-cell timing metrics.

    All timing is client-observed service performance (§9.5).
    Missing values are ``None`` (never invented estimates, §5.1).
    """

    ttfb_ms: float | None = None  # t_first_byte - t_dispatch (first byte/event)
    ttft_ms: float | None = None  # t_first - t_dispatch (first generated content)
    e2e_ms: float | None = None  # dispatch -> terminal completion
    tokens_per_s: float | None = None
    wall_clock_s: float | None = None
    p50_ms: float | None = None
    p90_ms: float | None = None
    p95_ms: float | None = None
    n: int = 0
    # Explanatory status for unavailable optional metrics (spec §5.1, §5.3):
    # None means measured; "unsupported" means the endpoint/path cannot
    # produce the metric (null), never a fabricated estimate.
    ttft_status: str | None = None
    ttfb_status: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def merge(self, other: TimingMetrics) -> TimingMetrics:
        """Merge two metric sets (e.g. across iterations)."""

        def _avg(a: float | None, b: float | None) -> float | None:
            vals = [v for v in (a, b) if v is not None]
            return sum(vals) / len(vals) if vals else None

        wc = (
            (self.wall_clock_s or 0.0) + (other.wall_clock_s or 0.0)
            if self.wall_clock_s is not None and other.wall_clock_s is not None
            else max(self.wall_clock_s, other.wall_clock_s)
        )
        ttft_status = None
        if self.ttft_ms is None or other.ttft_ms is None:
            ttft_status = self.ttft_status or other.ttft_status or "unsupported"
        ttfb_status = None
        if self.ttfb_ms is None or other.ttfb_ms is None:
            ttfb_status = self.ttfb_status or other.ttfb_status or "unsupported"
        return TimingMetrics(
            ttfb_ms=_avg(self.ttfb_ms, other.ttfb_ms),
            ttft_ms=_avg(self.ttft_ms, other.ttft_ms),
            e2e_ms=_avg(self.e2e_ms, other.e2e_ms),
            tokens_per_s=_avg(self.tokens_per_s, other.tokens_per_s),
            wall_clock_s=wc,
            p50_ms=_avg(self.p50_ms, other.p50_ms),
            p90_ms=_avg(self.p90_ms, other.p90_ms),
            p95_ms=_avg(self.p95_ms, other.p95_ms),
            n=self.n + other.n,
            ttft_status=ttft_status,
            ttfb_status=ttfb_status,
        )


def quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile (numpy-style, stdlib only)."""
    if not values:
        raise ValueError("quantile of empty list")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def compute_timing_metrics(
    dispatch: list[float],
    first_event: list[float] | None,
    completion: list[float],
    output_tokens: list[int] | None = None,
    wall_clock_s: float | None = None,
    first_byte: list[float] | None = None,
) -> TimingMetrics:
    """Compute timing metrics from per-request timestamps.

    All timestamps are ``time.monotonic()`` values on one host (§9.4).
    ``dispatch[i]`` is t_dispatch, ``first_event[i]`` is t_first (first
    non-empty generated content/reasoning/tool event), ``completion[i]`` is
    t_done. ``first_byte[i]`` is t_first_byte (first byte/event) if provided.

    TTFB = t_first_byte - t_dispatch (time to first byte/event).
    TTFT = t_first - t_dispatch (time to first generated content).
    When ``first_byte`` is not provided but ``first_event`` is, TTFB falls
    back to t_first - t_dispatch (the two metrics are then identical).

    When ``first_event`` is ``None`` the path cannot observe TTFT (for
    example a non-streaming endpoint). TTFT is then reported as ``null`` with
    ``ttft_status='unsupported'`` instead of a fabricated zero (spec §5.3).
    """
    if first_event is not None:
        n = min(len(dispatch), len(first_event), len(completion))
    else:
        n = min(len(dispatch), len(completion))

    if n == 0:
        return TimingMetrics(
            n=0,
            ttft_status="unsupported" if first_event is None else None,
        )

    ttft: list[float] | None = None
    ttft_status: str | None = None
    if first_event is not None:
        ttft = [(f - d) * 1000.0 for d, f in zip(dispatch[:n], first_event[:n])]
    else:
        ttft_status = "unsupported"

    # TTFB: time to first byte/event (first SSE chunk).
    ttfb: list[float] | None
    ttfb_status: str | None = None
    if first_byte is not None:
        ttfb = [(fb - d) * 1000.0 for d, fb in zip(dispatch[:n], first_byte[:n])]
    elif first_event is not None:
        # Fallback: use first_event if no separate first_byte timestamp
        ttfb = [(f - d) * 1000.0 for d, f in zip(dispatch[:n], first_event[:n])]
    else:
        ttfb = None
        ttfb_status = "unsupported"

    e2e = [(c - d) * 1000.0 for d, c in zip(dispatch[:n], completion[:n])]

    tps: float | None = None
    if output_tokens is not None:
        total_tokens = sum(output_tokens[:n])
        interval = max(completion[:n]) - min(dispatch[:n])
        if interval > 0:
            tps = total_tokens / interval

    return TimingMetrics(
        ttfb_ms=sum(ttfb) / n if ttfb is not None else None,
        ttft_ms=sum(ttft) / n if ttft is not None else None,
        e2e_ms=sum(e2e) / n,
        tokens_per_s=tps,
        wall_clock_s=wall_clock_s,
        p50_ms=quantile(e2e, 0.50),
        p90_ms=quantile(e2e, 0.90),
        p95_ms=quantile(e2e, 0.95),
        n=n,
        ttft_status=ttft_status,
        ttfb_status=ttfb_status,
    )


# ---------------------------------------------------------------------------
# Request observation
# ---------------------------------------------------------------------------


@dataclass
class RequestObservation:
    """Per-request timing observation (spec §5.1)."""

    request_id: str
    t_scheduled: float | None = None
    t_dispatch: float | None = None
    t_first_byte: float | None = None  # first byte/event (TTFB)
    t_first: float | None = None  # first generated content (TTFT)
    t_done: float | None = None
    output_tokens: int = 0
    status: str = "ok"  # ok | error | timeout | dropped
    error: str | None = None


# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------

# A request function takes a request id and returns a dict with keys
# ``t_first`` (monotonic timestamp of first generated content), optional
# ``t_first_byte`` (monotonic timestamp of the first byte/event received,
# used for TTFB), and ``output_tokens`` (int).
RequestFn = Callable[[str], Awaitable[dict[str, Any]]]


def _to_monotonic_seconds(t_client_ns: int, mono0: float, perf0_ns: int) -> float:
    """Translate a ``perf_counter_ns`` event timestamp onto the monotonic
    clock used by the cell runner without assuming the two clocks share an
    epoch. Deltas are preserved either way; the baseline only removes the
    epoch offset."""
    return mono0 + (t_client_ns - perf0_ns) / 1_000_000_000.0


async def stream_request_observation(
    adapter: Any,
    step: dict[str, Any],
    request_id: str = "stream",
    *,
    t_dispatch: float | None = None,
) -> RequestObservation:
    """Consume ``adapter.stream(step)`` and build a real-timestamp observation.

    TTFB is the arrival of the first SSE event (the first raw byte/event) and
    TTFT is the arrival of the first non-role content, reasoning or tool
    delta — role-only openers do not count (spec §9.4). ``t_done`` is the
    terminal completion of the stream. Output tokens are taken from the
    final ``usage`` event when present; usage is never invented.

    This is the streaming-aware request path for performance cells: the
    returned observation feeds :func:`_build_result` with distinct
    ``t_first_byte`` and ``t_first`` values instead of a fabricated TTFT.
    """
    if t_dispatch is None:
        t_dispatch = time.monotonic()
    mono0 = time.monotonic()
    perf0 = time.perf_counter_ns()

    t_first_byte: float | None = None
    t_first: float | None = None
    output_tokens = 0

    async for event in adapter.stream(step):
        t_sec = _to_monotonic_seconds(event.t_client_ns, mono0, perf0)
        if t_first_byte is None:
            t_first_byte = t_sec
        if event.kind in ("content", "reasoning", "tool") and t_first is None:
            t_first = t_sec
        if event.usage:
            total = (
                event.usage.get("total_tokens")
                or event.usage.get("completion_tokens")
                or 0
            )
            if total:
                output_tokens = int(total)

    return RequestObservation(
        request_id=request_id,
        t_dispatch=t_dispatch,
        t_first_byte=t_first_byte,
        t_first=t_first,
        t_done=time.monotonic(),
        output_tokens=output_tokens,
        status="ok",
    )


def make_streaming_request_fn(
    adapter_factory: Callable[[], Any],
    step: dict[str, Any],
) -> RequestFn:
    """Build a :data:`RequestFn` that streams through a fresh adapter.

    ``adapter_factory`` returns an adapter exposing ``stream()`` (it is
    closed after each request). The returned dict carries real
    ``t_first_byte``/``t_first`` timestamps so the cells can measure distinct
    TTFB and TTFT; if the adapter cannot stream, ``CapabilityUnavailable``
    propagates and the observation is recorded as an error rather than
    scored with a fake TTFT.
    """

    async def request_fn(request_id: str) -> dict[str, Any]:
        adapter = adapter_factory()
        try:
            obs = await stream_request_observation(adapter, step, request_id)
        finally:
            closer = getattr(adapter, "aclose", None) or getattr(adapter, "close", None)
            if closer is not None:
                result = closer()
                if inspect.isawaitable(result):
                    await result
        return {
            "t_first_byte": obs.t_first_byte,
            "t_first": obs.t_first,
            "t_done": obs.t_done,
            "output_tokens": obs.output_tokens,
            "streaming": True,
        }

    return request_fn


class ClosedLoopCell:
    """Closed-loop cell: maintain C active requests until the request
    budget is consumed (§9.3)."""

    def __init__(self, cell: PerformanceCell, request_fn: RequestFn):
        self.cell = cell
        self.request_fn = request_fn
        self.observations: list[RequestObservation] = []
        self.max_queue = 0
        self.dropped = 0

    async def _warmup(self) -> None:
        for i in range(self.cell.warmup):
            await self.request_fn(f"warmup-{i}")

    async def _execute(
        self, request_id: str, t_dispatch: float
    ) -> RequestObservation:
        """Run one request and turn it into an observation (never raises)."""
        try:
            res = await self.request_fn(request_id)
            t_done = time.monotonic()
            return RequestObservation(
                request_id=request_id,
                t_dispatch=t_dispatch,
                t_first_byte=res.get("t_first_byte"),
                t_first=res.get("t_first"),
                t_done=t_done,
                output_tokens=res.get("output_tokens", 0),
                status="ok",
            )
        except Exception as e:  # noqa: BLE001 - record, don't crash the cell
            return RequestObservation(
                request_id=request_id,
                t_dispatch=t_dispatch,
                t_done=time.monotonic(),
                status="error",
                error=str(e),
            )

    async def run(self) -> list[RequestObservation]:
        """Continuous worker-pool closed loop (spec §9.3).

        Exactly ``min(C, samples)`` workers are started. Each worker pulls
        the next pending request as soon as its current request finishes,
        so the in-flight count stays at ``C`` (never above it) and, while
        work remains, never idles at zero. Requests are therefore not
        released in fixed batches that let the pool drain between batches.
        """
        await self._warmup()
        observations = self.observations
        total = self.cell.samples
        if total <= 0:
            return observations

        next_index = 0

        async def worker() -> None:
            nonlocal next_index
            while True:
                index = next_index
                if index >= total:
                    return
                next_index += 1
                request_id = f"req-{index}"
                t_dispatch = time.monotonic()
                observations.append(await self._execute(request_id, t_dispatch))

        workers = min(self.cell.concurrency, total)
        await asyncio.gather(*(worker() for _ in range(workers)))
        return observations


class OpenLoopCell:
    """Open-loop cell: fixed arrival rate, measure queue length and
    latency distribution (§9.3).

    Scheduled arrival times are precomputed (fixed intervals or seeded
    Poisson). A bounded client queue is allowed; queue delay and admission
    drops remain part of the end-to-end load results.
    """

    def __init__(self, cell: PerformanceCell, request_fn: RequestFn, queue_limit: int = 100):
        self.cell = cell
        self.request_fn = request_fn
        self.queue_limit = queue_limit
        self.max_queue = 0
        self.dropped = 0
        self.observations: list[RequestObservation] = []

    def scheduled_arrivals(self) -> list[float]:
        """Precompute scheduled arrival times relative to dispatch start."""
        rate = self.cell.arrival_rate
        assert rate is not None
        scheduled: list[float] = [0.0]
        t = 0.0
        rng = random.Random(self.cell.seed)
        for _ in range(self.cell.samples - 1):
            if self.cell.arrival_process == "poisson":
                t += rng.expovariate(rate)
            else:
                t += 1.0 / rate
            scheduled.append(t)
        return scheduled

    async def _warmup(self) -> None:
        for i in range(self.cell.warmup):
            await self.request_fn(f"warmup-{i}")

    async def run(self) -> list[RequestObservation]:
        await self._warmup()
        scheduled = self.scheduled_arrivals()
        observations = self.observations
        start = time.monotonic()
        in_flight: set[asyncio.Task] = set()

        for i, sched in enumerate(scheduled):
            # Wait until the scheduled arrival time.
            wait = (start + sched) - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)

            t_dispatch = time.monotonic()
            queue_len = len(in_flight)
            self.max_queue = max(self.max_queue, queue_len)

            if queue_len >= self.queue_limit:
                self.dropped += 1
                observations.append(
                    RequestObservation(
                        request_id=f"req-{i}",
                        t_scheduled=start + sched,
                        t_dispatch=t_dispatch,
                        t_done=t_dispatch,
                        status="dropped",
                        error="queue limit reached",
                    )
                )
                continue

            async def _one(idx: int) -> RequestObservation:
                t0 = time.monotonic()
                try:
                    res = await self.request_fn(f"req-{idx}")
                    t_done = time.monotonic()
                    return RequestObservation(
                        request_id=f"req-{idx}",
                        t_scheduled=start + scheduled[idx],
                        t_dispatch=t0,
                        t_first_byte=res.get("t_first_byte"),
                        t_first=res.get("t_first"),
                        t_done=t_done,
                        output_tokens=res.get("output_tokens", 0),
                        status="ok",
                    )
                except Exception as e:  # noqa: BLE001
                    return RequestObservation(
                        request_id=f"req-{idx}",
                        t_scheduled=start + scheduled[idx],
                        t_dispatch=t0,
                        t_done=time.monotonic(),
                        status="error",
                        error=str(e),
                    )

            task = asyncio.ensure_future(_one(i))
            in_flight.add(task)

            # Reap completed tasks.
            for t in [t for t in in_flight if t.done()]:
                observations.append(await t)
                in_flight.discard(t)

        # Wait for remaining in-flight.
        for t in list(in_flight):
            observations.append(await t)
            in_flight.discard(t)

        return observations


# ---------------------------------------------------------------------------
# Concurrency sweep (§9.3)
# ---------------------------------------------------------------------------


@dataclass
class ConcurrencySweepResult:
    """Throughput vs concurrency across a sweep."""

    cell_id: str
    target: str
    results: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def throughput_curve(self) -> list[tuple[int, float | None]]:
        """(concurrency, request_rps) pairs."""
        return [(r["concurrency"], r["request_rps"]) for r in self.results]


class ConcurrencySweep:
    """Run cells at different concurrency levels, report throughput vs
    concurrency (§9.3)."""

    def __init__(
        self,
        cell: PerformanceCell,
        request_fn: RequestFn,
        concurrency_levels: list[int] | None = None,
    ):
        self.cell = cell
        self.request_fn = request_fn
        self.concurrency_levels = concurrency_levels or [1, 2, 4, 8]

    async def run(self) -> ConcurrencySweepResult:
        result = ConcurrencySweepResult(cell_id=self.cell.cell_id, target=self.cell.target)
        for c in self.concurrency_levels:
            cell = PerformanceCell(
                cell_id=f"{self.cell.cell_id}-c{c}",
                mode=self.cell.mode,
                target=self.cell.target,
                samples=max(self.cell.samples, max(12, 4 * c)),
                concurrency=c,
                warmup=self.cell.warmup,
                iterations=1,
                deadline_s=self.cell.deadline_s,
                workload=self.cell.workload,
                arrival_rate=self.cell.arrival_rate,
                arrival_process=self.cell.arrival_process,
                seed=self.cell.seed,
                cache_mode=self.cell.cache_mode,
            )
            perf = await run_performance_cell(cell, self.request_fn)
            m = perf.metrics
            rps = m.n / m.wall_clock_s if m.wall_clock_s and m.wall_clock_s > 0 else None
            result.results.append({
                "concurrency": c,
                "request_rps": rps,
                "e2e_ms": m.e2e_ms,
                "ttft_ms": m.ttft_ms,
                "wall_clock_s": m.wall_clock_s,
                "n": m.n,
            })
        return result


# ---------------------------------------------------------------------------
# Performance result
# ---------------------------------------------------------------------------


@dataclass
class PerformanceResult:
    """Result of a performance cell run."""

    cell_id: str
    mode: str
    target: str
    workload: str
    concurrency: int
    samples: int
    metrics: TimingMetrics
    observations: list[RequestObservation]
    deadline_s: float | None = None
    deadline_hit: bool = False
    warmup: int = 0
    max_queue: int = 0
    dropped: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "mode": self.mode,
            "target": self.target,
            "workload": self.workload,
            "concurrency": self.concurrency,
            "samples": self.samples,
            "metrics": self.metrics.as_dict(),
            "deadline_s": self.deadline_s,
            "deadline_hit": self.deadline_hit,
            "valid": self.valid,
            "warmup": self.warmup,
            "max_queue": self.max_queue,
            "dropped": self.dropped,
            "observations": [asdict(o) for o in self.observations],
        }

    @property
    def valid(self) -> bool:
        """Diagnostic validity: cell size >= max(12, 4*C) (§9.3)."""
        return self.samples >= max(12, 4 * self.concurrency)


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------


def _build_result(
    cell: PerformanceCell,
    observations: list[RequestObservation],
    wall_clock_s: float,
    deadline_hit: bool,
    max_queue: int = 0,
    dropped: int = 0,
) -> PerformanceResult:
    ok_obs = [o for o in observations if o.status == "ok"]
    # Every successfully completed request contributes to E2E and throughput.
    aligned = [
        o for o in ok_obs
        if o.t_dispatch is not None and o.t_done is not None
    ]
    # TTFT/TTFB require every aligned observation to carry its timestamp;
    # otherwise the metric is unsupported (null + status), never faked.
    first_event = (
        [o.t_first for o in aligned]
        if aligned and all(o.t_first is not None for o in aligned)
        else None
    )
    first_byte = (
        [o.t_first_byte for o in aligned]
        if aligned and all(o.t_first_byte is not None for o in aligned)
        else None
    )
    metrics = compute_timing_metrics(
        [o.t_dispatch for o in aligned],
        first_event,
        [o.t_done for o in aligned],
        [o.output_tokens for o in aligned],
        wall_clock_s=wall_clock_s,
        first_byte=first_byte,
    )
    return PerformanceResult(
        cell_id=cell.cell_id,
        mode=cell.mode.value,
        target=cell.target,
        workload=cell.workload,
        concurrency=cell.concurrency,
        samples=len(observations),
        metrics=metrics,
        observations=observations,
        deadline_s=cell.deadline_s,
        deadline_hit=deadline_hit,
        warmup=cell.warmup,
        max_queue=max_queue,
        dropped=dropped,
    )


async def run_performance_cell(cell: PerformanceCell, request_fn: RequestFn) -> PerformanceResult:
    """Run a performance cell with deadline enforcement (§8.1, §9.3).

    ``request_fn`` is an async callable taking a request id and returning
    a dict with keys ``t_first`` (monotonic timestamp of first generated
    content), optional ``t_first_byte`` (monotonic timestamp of the first
    byte/event received, used for TTFB), and ``output_tokens`` (int).

    The deadline is enforced by cancelling the runner task when the
    monotonic deadline is reached; in-flight requests are cancelled and
    the cell returns a partial result with ``deadline_hit=True``.
    """
    if cell.mode == LoopMode.CLOSED:
        runner: ClosedLoopCell | OpenLoopCell = ClosedLoopCell(cell, request_fn)
    else:
        runner = OpenLoopCell(cell, request_fn)

    start = time.monotonic()
    deadline = cell.deadline_s

    if deadline is None:
        observations = await runner.run()
        wall = time.monotonic() - start
        return _build_result(
            cell, observations, wall, False,
            max_queue=getattr(runner, "max_queue", 0),
            dropped=getattr(runner, "dropped", 0),
        )

    # Deadline enforcement: run the cell as a task and cancel it when the
    # monotonic deadline is reached (§8.1: active work is cancelled).
    runner_task = asyncio.ensure_future(runner.run())
    while True:
        remaining = deadline - (time.monotonic() - start)
        if remaining <= 0:
            runner_task.cancel()
            break
        try:
            observations = await asyncio.wait_for(
                asyncio.shield(runner_task), timeout=remaining
            )
            wall = time.monotonic() - start
            return _build_result(
                cell, observations, wall, False,
                max_queue=getattr(runner, "max_queue", 0),
                dropped=getattr(runner, "dropped", 0),
            )
        except TimeoutError:
            runner_task.cancel()
            break

    # Give the cancellation a chance to propagate.
    try:
        await runner_task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass

    wall = time.monotonic() - start
    # Partial observations collected by the runner before cancellation.
    partial = runner.observations
    return _build_result(
        cell, partial, wall, True,
        max_queue=getattr(runner, "max_queue", 0),
        dropped=getattr(runner, "dropped", 0),
    )
