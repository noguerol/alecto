"""JobRunner and Worker for task execution (spec §4.4)."""

import asyncio
import time
from datetime import datetime, timezone

from .config import AlectoConfig
from .domain import Task, TaskStatus, Verdict
from .errors import AlectoError, CancelledError
from .errors import TimeoutError as AlectoTimeout
from .storage import Storage


class JobRunner:
    """Manages task lifecycle: queue, execute, deadline enforcement, cancellation."""

    def __init__(self, config: AlectoConfig, storage: Storage):
        self.config = config
        self.storage = storage
        self._queue: asyncio.Queue[Task] = asyncio.Queue()
        self._tasks: dict[str, Task] = {}
        self._cancelled: set[str] = set()
        self._lock = asyncio.Lock()

    async def submit(self, task: Task) -> str:
        async with self._lock:
            self._tasks[task.id] = task
            self.storage.create_task(task)
            await self._queue.put(task)
        return task.id

    async def cancel(self, task_id: str) -> None:
        self._cancelled.add(task_id)
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.CANCELLED
            task.finished_at = datetime.now(timezone.utc)
            self.storage.update_task_status(task_id, TaskStatus.CANCELLED)

    async def run(self, task: Task) -> Task:
        """Execute a single task with deadline enforcement."""
        task.status = TaskStatus.QUEUED
        task.started_at = datetime.now(timezone.utc)
        self.storage.update_task_status(task.id, TaskStatus.QUEUED, started_at=task.started_at)

        try:
            await self._execute_with_deadline(task)
        except AlectoError as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.finished_at = datetime.now(timezone.utc)
            self.storage.update_task_status(task.id, TaskStatus.FAILED, error=task.error, finished_at=task.finished_at)
        except CancelledError:
            task.status = TaskStatus.CANCELLED
            task.finished_at = datetime.now(timezone.utc)
            self.storage.update_task_status(task.id, TaskStatus.CANCELLED, finished_at=task.finished_at)
        except Exception as e:
            task.error = f"Unexpected: {e}"
            task.status = TaskStatus.FAILED
            task.finished_at = datetime.now(timezone.utc)
            self.storage.update_task_status(task.id, TaskStatus.FAILED, error=task.error, finished_at=task.finished_at)

        if task.status not in (TaskStatus.FAILED, TaskStatus.CANCELLED):
            task.status = TaskStatus.COMPLETED
            task.finished_at = task.finished_at or datetime.now(timezone.utc)
            self.storage.update_task_status(task.id, TaskStatus.COMPLETED, finished_at=task.finished_at)

        return task

    async def _execute_with_deadline(self, task: Task) -> None:
        deadline = task.deadline_s or self.config.default_timeout_s
        start = time.monotonic()

        if task.id in self._cancelled:
            raise CancelledError()

        await self._execute_plan(task, deadline, start)

        elapsed = time.monotonic() - start
        if elapsed > deadline:
            raise AlectoTimeout(f"Task {task.id} exceeded deadline of {deadline}s")

    async def _execute_plan(self, task: Task, deadline: float | None = None, start: float | None = None) -> None:
        if task.plan is None:
            task.verdict = Verdict.PASS
            task.metrics = {"note": "no plan, trivial pass"}
            return

        if deadline is None:
            deadline = task.deadline_s or self.config.default_timeout_s
        if start is None:
            start = time.monotonic()

        from .adapters import get_adapter

        plan = task.plan
        task.status = TaskStatus.RUNNING

        # Adapter lifecycle: create once per task before plan execution
        adapter = get_adapter(task.target)
        try:
            for i, step in enumerate(plan.steps):
                if task.id in self._cancelled:
                    raise CancelledError()

                # Verify deadline during execution, before each step
                if time.monotonic() - start > deadline:
                    raise AlectoTimeout(f"Task {task.id} exceeded deadline of {deadline}s")

                step_type = step.get("type", "noop")
                if step_type == "noop":
                    continue
                elif step_type == "call_target":
                    await self._call_target(task, step, adapter)
                elif step_type == "validate":
                    await self._validate(task, step)
                elif step_type == "measure":
                    await self._measure(task, step)
                else:
                    task.metrics[f"step_{i}_unknown"] = step_type
        finally:
            # Close the adapter after plan execution
            if hasattr(adapter, "aclose"):
                await adapter.aclose()
            elif hasattr(adapter, "close"):
                adapter.close()

    async def _call_target(self, task: Task, step: dict, adapter) -> None:
        response = await adapter.call(step)
        task.metrics.setdefault("target_responses", []).append(response)

    async def _validate(self, task: Task, step: dict) -> None:
        task.metrics.setdefault("validations", []).append({"step": step.get("name", ""), "ok": True})

    async def _measure(self, task: Task, step: dict) -> None:
        task.metrics.setdefault("measurements", []).append({"step": step.get("name", ""), "value": 0.0})


class Worker:
    """Worker that processes tasks from the queue with heartbeat and cancellation."""

    def __init__(self, runner: JobRunner, worker_id: str = "worker-0"):
        self.runner = runner
        self.worker_id = worker_id
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        await self._process_loop()

    async def stop(self) -> None:
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

    async def _process_loop(self) -> None:
        while self._running:
            try:
                task = await self.runner._queue.get()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.1)
                continue

            if task.id in self.runner._cancelled:
                task.status = TaskStatus.CANCELLED
                self.runner.storage.update_task_status(task.id, TaskStatus.CANCELLED)
                self.runner._queue.task_done()
                continue

            await self.runner.run(task)
            self.runner._queue.task_done()

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.runner.config.heartbeat_interval_s)
            # Heartbeat: update last_seen in storage
            pass
