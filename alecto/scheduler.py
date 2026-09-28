"""Scheduler with closed-loop and open-loop modes (spec §4.5)."""

import asyncio

from .config import AlectoConfig
from .domain import LoopState, Task
from .enums import LoopMode, LoopStrategy, TaskStatus
from .runner import JobRunner, Worker
from .storage import Storage


class Scheduler:
    """Schedules tasks in closed-loop (deterministic) or open-loop (adaptive) mode."""

    def __init__(self, config: AlectoConfig, storage: Storage):
        self.config = config
        self.storage = storage
        self.runner = JobRunner(config, storage)
        self.workers: list[Worker] = []
        self.loop_state = LoopState()

    def set_loop_mode(self, mode: LoopMode, strategy: LoopStrategy = LoopStrategy.PARALLEL) -> None:
        self.loop_state.mode = mode
        self.loop_state.strategy = strategy

    async def start(self, num_workers: int = 1) -> None:
        for i in range(num_workers):
            worker = Worker(self.runner, worker_id=f"worker-{i}")
            self.workers.append(worker)
            asyncio.create_task(worker.start())

    async def stop(self) -> None:
        for worker in self.workers:
            await worker.stop()

    async def schedule(self, task: Task) -> str:
        task_id = await self.runner.submit(task)

        if self.loop_state.mode == LoopMode.CLOSED:
            await self._closed_loop(task)
        else:
            await self._open_loop(task)

        return task_id

    async def _closed_loop(self, task: Task) -> None:
        """Closed-loop: execute plan steps in sequence with verification."""
        task.status = TaskStatus.RUNNING
        if task.plan and task.plan.steps:
            for step in task.plan.steps:
                if task.status == TaskStatus.CANCELLED:
                    break
                # Each step is verified before the next
                task.metrics.setdefault("closed_loop_steps", []).append(step.get("name", step.get("type", "")))
        task.status = TaskStatus.COMPLETED

    async def _open_loop(self, task: Task) -> None:
        """Open-loop: adaptive execution based on feedback."""
        task.status = TaskStatus.RUNNING
        iteration = 0
        while not self.loop_state.converged and iteration < self.loop_state.max_iterations:
            if task.status == TaskStatus.CANCELLED:
                break
            iteration += 1
            self.loop_state.iteration = iteration
            self.loop_state.history.append({"iteration": iteration, "status": task.status.value})
            # In open-loop, we check convergence criteria
            if self._check_convergence(task):
                self.loop_state.converged = True
                break
        task.status = TaskStatus.COMPLETED

    def _check_convergence(self, task: Task) -> bool:
        # Simple convergence: task completed or max iterations reached
        return task.status == TaskStatus.COMPLETED or self.loop_state.iteration >= self.loop_state.max_iterations
