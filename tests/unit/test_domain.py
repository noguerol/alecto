"""Unit tests for domain records."""

import pytest

from alecto.domain import (
    BenchmarkResult,
    ComparisonResult,
    Evidence,
    LoopState,
    Plan,
    Refusal,
    TargetSpec,
    Task,
)
from alecto.enums import (
    BenchmarkCategory,
    ComparisonMode,
    EvidenceKind,
    RefusalReason,
    TargetKind,
    TaskStatus,
)


class TestTargetSpec:
    def test_basic(self):
        spec = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test")
        assert spec.kind == TargetKind.MOCK
        assert spec.endpoint == "mock://test"
        assert spec.timeout_s == 30.0

    def test_invalid_timeout(self):
        with pytest.raises(ValueError):
            TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test", timeout_s=-1)

    def test_with_model(self):
        spec = TargetSpec(kind=TargetKind.OPENAI, endpoint="https://api.openai.com", model="gpt-4")
        assert spec.model == "gpt-4"


class TestPlan:
    def test_basic(self):
        plan = Plan(steps=[{"type": "noop", "name": "test"}])
        assert len(plan.steps) == 1
        assert plan.steps[0]["type"] == "noop"

    def test_empty(self):
        plan = Plan()
        assert plan.steps == []


class TestTask:
    def test_default_status(self):
        task = Task()
        assert task.status == TaskStatus.PENDING
        assert task.id  # auto-generated

    def test_with_plan(self):
        plan = Plan(steps=[{"type": "noop"}])
        task = Task(plan=plan)
        assert task.plan is not None
        assert task.plan.steps == [{"type": "noop"}]


class TestEvidence:
    def test_text_evidence(self):
        ev = Evidence(kind=EvidenceKind.TEXT, text="hello")
        assert ev.kind == EvidenceKind.TEXT
        assert ev.text == "hello"

    def test_metric_evidence(self):
        ev = Evidence(kind=EvidenceKind.METRIC, metric_name="score", value=9.5)
        assert ev.metric_name == "score"
        assert ev.value == 9.5


class TestBenchmarkResult:
    def test_basic(self):
        result = BenchmarkResult(
            task_id="task-1",
            benchmark="math_basic",
            category=BenchmarkCategory.MATH,
            score=10.0,
            max_score=10.0,
            duration_ms=100,
            pass_=True,
        )
        assert result.pass_ is True
        assert result.score == result.max_score


class TestComparisonResult:
    def test_pairwise(self):
        result = ComparisonResult(
            mode=ComparisonMode.PAIRWISE,
            results=[],
            winner="A",
            margin=2.0,
        )
        assert result.winner == "A"


class TestRefusal:
    def test_basic(self):
        refusal = Refusal(
            reason=RefusalReason.OUT_OF_SCOPE,
            message="Out of scope",
            alternatives=["benchmark", "compare"],
        )
        assert refusal.reason == RefusalReason.OUT_OF_SCOPE
        assert len(refusal.alternatives) == 2


class TestLoopState:
    def test_default(self):
        state = LoopState()
        assert state.iteration == 0
        assert state.converged is False
