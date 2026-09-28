"""Unit tests for reporting."""

import json

import pytest

from alecto.domain import BenchmarkResult, ComparisonResult, Task
from alecto.enums import BenchmarkCategory, ComparisonMode, Verdict
from alecto.reporting import format_report, generate_report, save_report


@pytest.fixture
def task():
    task = Task()
    task.verdict = Verdict.PASS
    task.metrics = {"duration_ms": 100}
    return task


@pytest.fixture
def results():
    return [
        BenchmarkResult(
            task_id="task-1",
            benchmark="math_basic",
            category=BenchmarkCategory.MATH,
            score=10.0,
            max_score=10.0,
            duration_ms=100,
            pass_=True,
        ),
    ]


class TestGenerateReport:
    def test_basic(self, task, results):
        report = generate_report(task, results)
        assert report.task_id == task.id
        assert len(report.sections) >= 2
        assert report.verdict == Verdict.PASS

    def test_with_comparison(self, task, results):
        comparison = ComparisonResult(
            mode=ComparisonMode.PAIRWISE,
            results=results,
            winner="A",
            margin=2.0,
        )
        report = generate_report(task, results, comparison)
        section_titles = [s["title"] for s in report.sections]
        assert "Comparison" in section_titles


class TestSaveReport:
    def test_save(self, task, results, tmp_path):
        report = generate_report(task, results)
        path = save_report(report, tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["task_id"] == task.id
        assert data["verdict"] == "pass"


class TestFormatReport:
    def test_format(self, task, results):
        report = generate_report(task, results)
        text = format_report(report)
        assert "Alecto Report" in text
        assert "Benchmark Results" in text
