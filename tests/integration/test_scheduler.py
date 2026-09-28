"""Integration tests for the scheduler."""

import pytest

from alecto.config import AlectoConfig
from alecto.domain import Plan, TargetSpec, Task
from alecto.enums import LoopMode, LoopStrategy, TargetKind, TaskStatus
from alecto.scheduler import Scheduler
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


class TestScheduler:
    @pytest.mark.asyncio
    async def test_closed_loop(self, config, storage):
        scheduler = Scheduler(config, storage)
        scheduler.set_loop_mode(LoopMode.CLOSED, LoopStrategy.PARALLEL)
        task = Task(
            plan=Plan(steps=[{"type": "noop", "name": "step1"}, {"type": "noop", "name": "step2"}]),
            target=TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test"),
        )
        task_id = await scheduler.schedule(task)
        assert task_id == task.id
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_open_loop(self, config, storage):
        scheduler = Scheduler(config, storage)
        scheduler.set_loop_mode(LoopMode.OPEN, LoopStrategy.ADAPTIVE)
        task = Task(
            plan=Plan(steps=[{"type": "noop"}]),
            target=TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test"),
        )
        task_id = await scheduler.schedule(task)
        assert task_id == task.id
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_loop_state(self, config, storage):
        scheduler = Scheduler(config, storage)
        scheduler.set_loop_mode(LoopMode.OPEN, LoopStrategy.ADAPTIVE)
        assert scheduler.loop_state.mode == LoopMode.OPEN
        assert scheduler.loop_state.strategy == LoopStrategy.ADAPTIVE
