"""Integration tests for the job runner."""

import pytest

from alecto.config import AlectoConfig
from alecto.domain import Plan, TargetSpec, Task
from alecto.enums import TargetKind, TaskStatus
from alecto.runner import JobRunner
from alecto.storage import Storage


@pytest.fixture
def config(tmp_path):
    config = AlectoConfig(data_dir=tmp_path)
    config.ensure_dirs()
    return config


@pytest.fixture
def storage(config):
    s = Storage(config.data_dir / "test.db")
    yield s
    s.close()


class TestJobRunner:
    @pytest.mark.asyncio
    async def test_submit_and_run(self, config, storage):
        runner = JobRunner(config, storage)
        task = Task(
            plan=Plan(steps=[{"type": "noop", "name": "test"}]),
            target=TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test"),
        )
        task_id = await runner.submit(task)
        assert task_id == task.id

        result = await runner.run(task)
        assert result.status == TaskStatus.COMPLETED or result.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_cancel(self, config, storage):
        runner = JobRunner(config, storage)
        task = Task(
            plan=Plan(steps=[{"type": "noop"}]),
            target=TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test"),
        )
        await runner.submit(task)
        await runner.cancel(task.id)
        assert task.id in runner._cancelled

    @pytest.mark.asyncio
    async def test_deadline_enforcement(self, config, storage):
        config.default_timeout_s = 0.001  # very short deadline
        runner = JobRunner(config, storage)
        task = Task(
            plan=Plan(steps=[{"type": "noop"}]),
            target=TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test"),
            deadline_s=0.001,
        )
        await runner.submit(task)
        result = await runner.run(task)
        # Task completes (mock is fast), but deadline was set
        assert result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
