"""Unit tests for comparison engine."""


from alecto.comparison import compare_group, compare_pairwise
from alecto.domain import BenchmarkResult
from alecto.enums import BenchmarkCategory, ComparisonMode


def make_result(benchmark: str, score: float) -> BenchmarkResult:
    return BenchmarkResult(
        task_id="task-1",
        benchmark=benchmark,
        category=BenchmarkCategory.MATH,
        score=score,
        max_score=10.0,
        duration_ms=100,
        pass_=score >= 5.0,
    )


class TestComparePairwise:
    def test_a_wins(self):
        results_a = [make_result("math", 8.0)]
        results_b = [make_result("math", 5.0)]
        result = compare_pairwise(results_a, results_b)
        assert result.winner == "A"
        assert result.margin == 3.0
        assert result.mode == ComparisonMode.PAIRWISE

    def test_b_wins(self):
        results_a = [make_result("math", 5.0)]
        results_b = [make_result("math", 8.0)]
        result = compare_pairwise(results_a, results_b)
        assert result.winner == "B"
        assert result.margin == 3.0

    def test_tie(self):
        results_a = [make_result("math", 5.0)]
        results_b = [make_result("math", 5.0)]
        result = compare_pairwise(results_a, results_b)
        assert result.winner is None
        assert result.margin == 0.0

    def test_multiple_results(self):
        results_a = [make_result("math", 8.0), make_result("coding", 7.0)]
        results_b = [make_result("math", 5.0), make_result("coding", 4.0)]
        result = compare_pairwise(results_a, results_b)
        assert result.winner == "A"
        assert result.margin == 6.0


class TestCompareGroup:
    def test_basic(self):
        results = [make_result("math", 8.0), make_result("coding", 5.0), make_result("reasoning", 10.0)]
        result = compare_group(results)
        assert result.winner == "reasoning"
        assert result.mode == ComparisonMode.GROUP
        assert result.margin == 5.0

    def test_empty(self):
        result = compare_group([])
        assert result.winner is None
        assert result.results == []
