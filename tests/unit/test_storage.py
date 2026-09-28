"""Unit tests for storage."""

import pytest
from alecto.domain import TargetSpec, Task
from alecto.enums import TargetKind, TaskStatus
from alecto.storage import Storage


@pytest.fixture
def storage(tmp_path):
    s = Storage(tmp_path / "test.db")
    yield s
    s.close()


class TestStorage:
    def test_create_and_get_task(self, storage):
        task = Task(target=TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test"))
        storage.create_task(task)
        fetched = storage.get_task(task.id)
        assert fetched is not None
        assert fetched.id == task.id
        assert fetched.status == TaskStatus.PENDING

    def test_update_status(self, storage):
        task = Task()
        storage.create_task(task)
        storage.update_task_status(task.id, TaskStatus.RUNNING)
        fetched = storage.get_task(task.id)
        assert fetched.status == TaskStatus.RUNNING

    def test_list_tasks(self, storage):
        for _ in range(3):
            storage.create_task(Task())
        tasks = storage.list_tasks()
        assert len(tasks) == 3

    def test_list_by_status(self, storage):
        t1 = Task()
        t2 = Task()
        storage.create_task(t1)
        storage.create_task(t2)
        storage.update_task_status(t1.id, TaskStatus.COMPLETED)
        completed = storage.list_tasks(TaskStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].id == t1.id

    def test_add_evidence(self, storage):
        from alecto.domain import Evidence
        from alecto.enums import EvidenceKind

        task = Task()
        storage.create_task(task)
        ev = Evidence(kind=EvidenceKind.TEXT, text="test evidence")
        storage.add_evidence(task.id, ev)

    def test_add_result(self, storage):
        from alecto.domain import BenchmarkResult
        from alecto.enums import BenchmarkCategory

        task = Task()
        storage.create_task(task)
        result = BenchmarkResult(
            task_id=task.id,
            benchmark="math_basic",
            category=BenchmarkCategory.MATH,
            score=10.0,
            max_score=10.0,
            duration_ms=100,
            pass_=True,
        )
        storage.add_result(task.id, result)
        results = storage.get_results(task.id)
        assert len(results) == 1
        assert results[0].score == 10.0
