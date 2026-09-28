"""CLI entry point for alecto."""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import default_config
from .domain import Plan, TargetSpec, Task
from .enums import TargetKind, TaskStatus, Verdict
from .storage import Storage


def main():
    parser = argparse.ArgumentParser(description="Alecto - LLM benchmarking engine")
    subparsers = parser.add_subparsers(dest="command")

    # self-test command
    subparsers.add_parser("self-test", help="Run self-test")

    # run command
    run = subparsers.add_parser("run", help="Run a benchmark task")
    run.add_argument("--target", default="mock", help="Target kind")
    run.add_argument("--benchmark", default="math_basic", help="Benchmark name")
    run.add_argument("--data-dir", default=None, help="Data directory")

    # list command
    list_cmd = subparsers.add_parser("list", help="List tasks")
    list_cmd.add_argument("--status", default=None, help="Filter by status")
    list_cmd.add_argument("--data-dir", default=None, help="Data directory")

    # report command
    report = subparsers.add_parser("report", help="Generate report for a task")
    report.add_argument("--task-id", required=True, help="Task ID")
    report.add_argument("--data-dir", default=None, help="Data directory")

    args = parser.parse_args()

    if args.command == "self-test":
        run_self_test()
    elif args.command == "run":
        run_task(args)
    elif args.command == "list":
        list_tasks(args)
    elif args.command == "report":
        generate_report_cmd(args)
    else:
        parser.print_help()


def _resolve_config(args):
    """Build the runtime config, honouring an explicit ``--data-dir``.

    Every command that touches stored tasks must resolve its configuration the
    same way; otherwise ``--data-dir`` is accepted and silently ignored, and
    later commands cannot find the tasks an earlier one just wrote.
    """
    config = default_config()
    data_dir = getattr(args, "data_dir", None)
    if data_dir:
        config.data_dir = Path(data_dir).expanduser()
        config.evidence_dir = config.data_dir / "evidence"
        config.report_dir = config.data_dir / "reports"
    config.ensure_dirs()
    return config


def run_self_test():
    """Run a basic self-test to verify the installation."""
    from .adapters import MockAdapter
    from .config import default_config
    from .domain import TargetSpec, Task
    from .enums import TargetKind
    from .storage import Storage

    config = default_config()
    config.ensure_dirs()
    storage = Storage(config.data_dir / "alecto.db")

    # Test 1: Domain records
    spec = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test")
    plan = Plan(steps=[{"type": "noop", "name": "test"}])
    task = Task(plan=plan, target=spec)
    assert task.id, "Task must have an ID"
    assert task.status.value == "pending"
    print("✓ Domain records")

    # Test 2: Storage
    storage.create_task(task)
    fetched = storage.get_task(task.id)
    assert fetched is not None, "Storage must persist tasks"
    assert fetched.id == task.id
    print("✓ Storage")

    # Test 3: Mock adapter
    async def test_adapter():
        adapter = MockAdapter(spec)
        result = await adapter.call({"messages": [{"role": "user", "content": "test"}]})
        assert "text" in result
        return result

    result = asyncio.run(test_adapter())
    assert result["text"].startswith("mock-response")
    print("✓ Mock adapter")

    # Test 4: Config
    assert config.data_dir.exists()
    assert config.evidence_dir.exists()
    assert config.report_dir.exists()
    print("✓ Config")

    storage.close()
    print("\nAll self-tests passed!")


def run_task(args):
    """Run a benchmark task."""
    from .benchmarks import run_benchmark
    from .reporting import generate_report, save_report

    config = _resolve_config(args)
    storage = Storage(config.data_dir / "alecto.db")

    target = TargetSpec(kind=TargetKind(args.target), endpoint="mock://test")
    task = Task(target=target)

    async def execute():
        result = await run_benchmark(args.benchmark, task, target)
        return result

    result = asyncio.run(execute())
    task.status = TaskStatus.COMPLETED
    task.finished_at = datetime.now(timezone.utc)
    task.verdict = Verdict.PASS if result.pass_ else Verdict.FAIL
    task.metrics = {"duration_ms": result.duration_ms}

    storage.create_task(task)
    report = generate_report(task, [result])
    save_report(report, config.report_dir)
    storage.close()

    print(f"Task {task.id}: {result.benchmark} = {result.score}/{result.max_score} ({'PASS' if result.pass_ else 'FAIL'})")


def list_tasks(args):
    """List tasks."""
    from .enums import TaskStatus

    config = _resolve_config(args)
    storage = Storage(config.data_dir / "alecto.db")
    status = TaskStatus(args.status) if args.status else None
    tasks = storage.list_tasks(status)
    for t in tasks:
        print(f"{t.id}: {t.status.value} (created {t.created_at.isoformat()})")
    storage.close()


def generate_report_cmd(args):
    """Generate a report for a task."""
    from .reporting import format_report, generate_report

    config = _resolve_config(args)
    storage = Storage(config.data_dir / "alecto.db")
    task = storage.get_task(args.task_id)
    if not task:
        print(f"Task {args.task_id} not found", file=sys.stderr)
        sys.exit(1)
    results = storage.get_results(task.id)
    report = generate_report(task, results)
    print(format_report(report))
    storage.close()


if __name__ == "__main__":
    main()
