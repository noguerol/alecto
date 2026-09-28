"""End-to-end acceptance tests."""

import json

import pytest

from alecto.benchmarks import run_benchmark
from alecto.cli import run_self_test
from alecto.comparison import compare_pairwise
from alecto.config import AlectoConfig
from alecto.domain import TargetSpec, Task
from alecto.enums import TargetKind
from alecto.refusal import check_refusal
from alecto.reporting import generate_report, save_report
from alecto.storage import Storage


@pytest.fixture
def config(tmp_path):
    config = AlectoConfig(data_dir=tmp_path)
    config.ensure_dirs()
    return config


@pytest.fixture
def storage(config):
    s = Storage(config.data_dir / "e2e.db")
    yield s
    s.close()


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_benchmark_pipeline(self, config, storage):
        """Full pipeline: create task -> run benchmark -> save results -> generate report."""
        target = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test")
        task = Task(target=target)

        # Run benchmark
        result = await run_benchmark("math_basic", task, target)
        assert result.benchmark == "math_basic"

        # Save to storage
        storage.create_task(task)
        storage.add_result(task.id, result)

        # Generate report
        report = generate_report(task, [result])
        path = save_report(report, config.report_dir)
        assert path.exists()

        # Verify report content
        data = json.loads(path.read_text())
        assert data["task_id"] == task.id
        assert data["verdict"] in ("pass", "fail", None)

    @pytest.mark.asyncio
    async def test_comparison_pipeline(self, config, storage):
        """Compare two benchmark runs."""
        target_a = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://a")
        target_b = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://b")

        task_a = Task(target=target_a)
        task_b = Task(target=target_b)

        result_a = await run_benchmark("math_basic", task_a, target_a)
        result_b = await run_benchmark("math_basic", task_b, target_b)

        comparison = compare_pairwise([result_a], [result_b])
        assert comparison.mode.value == "pairwise"

    def test_refusal_pipeline(self):
        """Test refusal for out-of-scope requests."""
        refusal = check_refusal({"task_type": "deploy"})
        assert refusal is not None
        assert "out_of_scope" in refusal.reason.value

    def test_self_test(self):
        """Run the CLI self-test."""
        run_self_test()

    @pytest.mark.asyncio
    async def test_multiple_benchmarks(self, config, storage):
        """Run multiple benchmarks on the same task."""
        target = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test")
        task = Task(target=target)

        results = []
        for name in ["math_basic", "coding_basic", "reasoning_basic", "language_basic"]:
            result = await run_benchmark(name, task, target)
            results.append(result)

        assert len(results) == 4
        for r in results:
            assert r.duration_ms >= 0
