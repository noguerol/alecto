"""Domain records for the alecto system (spec §4.1)."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .enums import (
    BenchmarkCategory,
    ComparisonMode,
    EvidenceKind,
    LoopMode,
    LoopStrategy,
    RefusalReason,
    RetryPolicy,
    TargetKind,
    TaskStatus,
    Verdict,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid4())


@dataclass
class TargetSpec:
    kind: TargetKind
    endpoint: str
    model: str | None = None
    api_key_env: str | None = None
    auth_mode: str = "bearer"
    generation_path: str = "chat/completions"
    timeout_s: float = 30.0
    max_retries: int = 2
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.auth_mode not in ("none", "bearer", "header"):
            raise ValueError("auth_mode must be one of: none, bearer, header")


@dataclass
class Plan:
    """A structured execution plan (spec §2.4)."""
    steps: list[dict[str, Any]] = field(default_factory=list)
    target: TargetSpec | None = None
    loop_mode: LoopMode = LoopMode.CLOSED
    loop_strategy: LoopStrategy = LoopStrategy.PARALLEL
    deadline_s: float | None = None
    retry_policy: RetryPolicy = RetryPolicy.ONCE
    notes: list[str] = field(default_factory=list)


@dataclass
class Task:
    id: str = field(default_factory=_new_id)
    plan: Plan | None = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    target: TargetSpec | None = None
    deadline_s: float | None = None
    retry_count: int = 0
    error: str | None = None
    evidence: list["Evidence"] = field(default_factory=list)
    verdict: Verdict | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    id: str = field(default_factory=_new_id)
    kind: EvidenceKind = EvidenceKind.TEXT
    path: str | None = None
    text: str | None = None
    metric_name: str | None = None
    value: float | None = None
    created_at: datetime = field(default_factory=_now)


@dataclass
class BenchmarkResult:
    task_id: str
    benchmark: str
    category: BenchmarkCategory
    score: float
    max_score: float
    duration_ms: int
    pass_: bool
    details: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class ComparisonResult:
    mode: ComparisonMode
    results: list[BenchmarkResult]
    winner: str | None = None
    margin: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class Refusal:
    reason: RefusalReason
    message: str
    task_id: str | None = None
    alternatives: list[str] = field(default_factory=list)


@dataclass
class LoopState:
    mode: LoopMode = LoopMode.CLOSED
    strategy: LoopStrategy = LoopStrategy.PARALLEL
    iteration: int = 0
    max_iterations: int = 10
    converged: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Report:
    task_id: str
    title: str
    sections: list[dict[str, Any]] = field(default_factory=list)
    verdict: Verdict | None = None
    created_at: datetime = field(default_factory=_now)
