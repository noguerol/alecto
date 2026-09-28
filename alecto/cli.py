"""CLI entry point for alecto."""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import default_config
from .domain import Plan, TargetSpec, Task
from .enums import TargetKind, TaskStatus, Verdict
from .storage import Storage


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (exposed so tests can assert the documented surface)."""
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

    # catalogue command
    catalogue = subparsers.add_parser(
        "catalogue", help="Print the agent operation catalogue (JSON)"
    )
    catalogue.add_argument(
        "--format",
        default="canonical",
        choices=("canonical", "mcp", "openai", "anthropic", "doc"),
        help="Tool-description flavour to emit",
    )

    # mcp command
    mcp = subparsers.add_parser(
        "mcp", help="Serve the agent operations over MCP stdio"
    )
    mcp.add_argument("--data-dir", default=None, help="Data directory")

    # skill command
    skill = subparsers.add_parser(
        "skill", help="Emit the bundled agent skill (SKILL.md)"
    )
    skill.add_argument(
        "--path", action="store_true",
        help="Print only the filesystem path to SKILL.md",
    )
    skill.add_argument(
        "--install-dir", default=None,
        help="Copy SKILL.md into <dir>/alecto/SKILL.md",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "self-test":
        run_self_test()
    elif args.command == "run":
        run_task(args)
    elif args.command == "list":
        list_tasks(args)
    elif args.command == "report":
        generate_report_cmd(args)
    elif args.command == "catalogue":
        print_catalogue_cmd(args)
    elif args.command == "mcp":
        serve_mcp_cmd(args)
    elif args.command == "skill":
        skill_cmd(args)
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


def self_test_report() -> list[str]:
    """Run the offline installation self-test and return the checks that passed.

    Deliberately silent: this is also the implementation behind the
    ``alecto_self_test`` agent operation, and a handler must never write to
    stdout because stdout is the MCP protocol channel.
    """
    from .adapters import MockAdapter
    from .config import default_config
    from .domain import TargetSpec, Task
    from .enums import TargetKind
    from .storage import Storage

    checks: list[str] = []
    config = default_config()
    config.ensure_dirs()
    storage = Storage(config.data_dir / "alecto.db")

    # Test 1: Domain records
    spec = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test")
    plan = Plan(steps=[{"type": "noop", "name": "test"}])
    task = Task(plan=plan, target=spec)
    assert task.id, "Task must have an ID"
    assert task.status.value == "pending"
    checks.append("Domain records")

    # Test 2: Storage
    storage.create_task(task)
    fetched = storage.get_task(task.id)
    assert fetched is not None, "Storage must persist tasks"
    assert fetched.id == task.id
    checks.append("Storage")

    # Test 3: Mock adapter
    async def test_adapter():
        adapter = MockAdapter(spec)
        return await adapter.call({"messages": [{"role": "user", "content": "test"}]})

    result = asyncio.run(test_adapter())
    assert result["text"].startswith("mock-response")
    checks.append("Mock adapter")

    # Test 4: Config
    assert config.data_dir.exists()
    assert config.evidence_dir.exists()
    assert config.report_dir.exists()
    checks.append("Config")

    storage.close()
    return checks


def run_self_test():
    """Run a basic self-test to verify the installation (CLI entry point)."""
    for name in self_test_report():
        print(f"✓ {name}")
    print("\nAll self-tests passed!")
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


def print_catalogue_cmd(args):
    """Emit the shared operation catalogue for an agent harness.

    ``canonical`` is pure JSON with no surrounding text, so it can be parsed
    directly; the other flavours match the tool-description shapes used by
    MCP, OpenAI and Anthropic function calling.
    """
    from .service import build_catalogue

    catalogue = build_catalogue()
    flavour = args.format
    if flavour == "canonical":
        print(catalogue.to_canonical_json())
        return
    tools = {
        "mcp": catalogue.to_mcp_tools,
        "openai": catalogue.to_openai_tools,
        "anthropic": catalogue.to_anthropic_tools,
    }[flavour]()
    print(json.dumps(tools, indent=2, sort_keys=True))


def skill_cmd(args):
    """Emit the bundled agent skill so a harness can discover and install it.

    The skill lives inside the package, so it ships with the wheel and the
    installed path is the source of truth.
    """
    import shutil
    from pathlib import Path as _Path

    source = _Path(__file__).resolve().parent / "SKILL.md"
    if not source.is_file():  # pragma: no cover - broken install
        print(f"SKILL.md not found next to the package: {source}", file=sys.stderr)
        sys.exit(1)

    if args.install_dir:
        target = _Path(args.install_dir).expanduser() / "alecto" / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        print(str(target))
        return
    if args.path:
        print(str(source))
        return
    sys.stdout.write(source.read_text(encoding="utf-8"))


def serve_mcp_cmd(args):
    """Serve the agent operations over MCP stdio until EOF.

    Stdout is the protocol channel, so it is held aside for the transport and
    ``sys.stdout`` is redirected to stderr while handlers run. A handler that
    prints (now or in future) therefore cannot corrupt the JSON-RPC stream.
    """
    import contextlib

    from .agent import MCPStdioAdapter
    from .service import build_catalogue, build_dispatcher

    config = _resolve_config(args)
    adapter = MCPStdioAdapter(build_catalogue(), build_dispatcher(config))
    protocol_out = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        adapter.run(sys.stdin, protocol_out)


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
