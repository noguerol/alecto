"""Unit tests for benchmarks."""

import pytest

from alecto.benchmarks import MathBenchmark, CodingBenchmark, ReasoningBenchmark, LanguageBenchmark, run_benchmark, BENCHMARKS
from alecto.domain import TargetSpec, Task
from alecto.enums import TargetKind, BenchmarkCategory


@pytest.fixture
def mock_target():
    return TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test")


@pytest.fixture
def task():
    return Task(target=TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test"))


class TestMathBenchmark:
    @pytest.mark.asyncio
    async def test_execute(self, task, mock_target):
        result = await run_benchmark("math_basic", task, mock_target)
        assert result.benchmark == "math_basic"
        assert result.category == BenchmarkCategory.MATH
        assert result.max_score == 10.0

    @pytest.mark.asyncio
    async def test_execute_forwards_generation_params(self, monkeypatch):
        captured = {}

        class FakeAdapter:
            async def call(self, step):
                captured.update(step)
                return {"text": "4"}

        monkeypatch.setattr("alecto.adapters.get_adapter", lambda target: FakeAdapter())
        target = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test")
        target.extra["temperature"] = 0.3
        target.extra["max_tokens"] = 64
        task = Task(target=target)
        await MathBenchmark().execute(task, target)
        assert captured["temperature"] == 0.3
        assert captured["max_tokens"] == 64
        assert captured["messages"][0]["content"]

    def test_score_correct(self):
        bench = MathBenchmark()
        assert bench.score({"text": "4"}) == 10.0
        assert bench.score({"text": "wrong"}) == 0.0


class TestCodingBenchmark:
    @pytest.mark.asyncio
    async def test_execute(self, task, mock_target):
        result = await run_benchmark("coding_basic", task, mock_target)
        assert result.benchmark == "coding_basic"
        assert result.category == BenchmarkCategory.CODING

    def test_score(self):
        bench = CodingBenchmark()
        assert bench.score({"text": "def add(a, b):\n    return a + b"}) == 10.0
        assert bench.score({"text": "not code"}) == 0.0


class TestReasoningBenchmark:
    @pytest.mark.asyncio
    async def test_execute(self, task, mock_target):
        result = await run_benchmark("reasoning_basic", task, mock_target)
        assert result.benchmark == "reasoning_basic"
        assert result.category == BenchmarkCategory.REASONING

    def test_score(self):
        bench = ReasoningBenchmark()
        assert bench.score({"text": "Yes"}) == 10.0
        assert bench.score({"text": "No"}) == 0.0


class TestLanguageBenchmark:
    @pytest.mark.asyncio
    async def test_execute(self, task, mock_target):
        result = await run_benchmark("language_basic", task, mock_target)
        assert result.benchmark == "language_basic"
        assert result.category == BenchmarkCategory.LANGUAGE

    def test_score(self):
        bench = LanguageBenchmark()
        assert bench.score({"text": "Hola mundo"}) == 10.0
        assert bench.score({"text": "wrong"}) == 0.0


class TestRunBenchmark:
    @pytest.mark.asyncio
    async def test_unknown_benchmark(self, task, mock_target):
        with pytest.raises(ValueError):
            await run_benchmark("nonexistent", task, mock_target)

    def test_all_benchmarks_registered(self):
        assert len(BENCHMARKS) == 4
        assert "math_basic" in BENCHMARKS
        assert "coding_basic" in BENCHMARKS
        assert "reasoning_basic" in BENCHMARKS
        assert "language_basic" in BENCHMARKS
