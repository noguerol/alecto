"""CLI command tests.

These cover the ``run`` → ``list`` → ``report`` round-trip, which had two
release-blocking defects: the report command called an unimported name, and the
run command stored a verdict as a raw string where the storage contract requires
the ``Verdict`` enum, so no task could ever be persisted.
"""

from __future__ import annotations

import argparse

import pytest

from alecto.cli import _resolve_config, generate_report_cmd, list_tasks, run_task
from alecto.enums import TaskStatus, Verdict
from alecto.storage import Storage


def _run(tmp_path):
    return run_task(argparse.Namespace(
        target="mock", benchmark="math_basic", data_dir=str(tmp_path)
    ))


def test_run_persists_a_task(tmp_path):
    _run(tmp_path)
    storage = Storage(tmp_path / "alecto.db")
    tasks = storage.list_tasks()
    storage.close()
    assert len(tasks) == 1


def test_run_stores_verdict_as_enum(tmp_path):
    """A raw string verdict used to break ``storage.create_task``."""
    _run(tmp_path)
    storage = Storage(tmp_path / "alecto.db")
    task = storage.list_tasks()[0]
    storage.close()
    assert isinstance(task.verdict, Verdict)
    assert task.verdict in (Verdict.PASS, Verdict.FAIL)


def test_run_leaves_task_in_terminal_status(tmp_path):
    _run(tmp_path)
    storage = Storage(tmp_path / "alecto.db")
    task = storage.list_tasks()[0]
    storage.close()
    assert task.status is TaskStatus.COMPLETED
    assert task.finished_at is not None


def test_data_dir_is_honoured_by_every_command(tmp_path):
    """``--data-dir`` was accepted but silently ignored, so later commands
    could not find what an earlier one had written."""
    other = tmp_path / "nested" / "data"
    run_task(argparse.Namespace(
        target="mock", benchmark="math_basic", data_dir=str(other)
    ))
    assert (other / "alecto.db").exists()

    storage = Storage(other / "alecto.db")
    tasks = storage.list_tasks()
    storage.close()
    assert len(tasks) == 1


def test_resolve_config_without_data_dir_uses_default(tmp_path, monkeypatch):
    monkeypatch.setenv("ALECTO_DATA", str(tmp_path / "envdir"))
    config = _resolve_config(argparse.Namespace(data_dir=None))
    assert config.data_dir == tmp_path / "envdir"


def test_resolve_config_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ALECTO_DATA", str(tmp_path / "envdir"))
    config = _resolve_config(argparse.Namespace(data_dir=str(tmp_path / "flag")))
    assert config.data_dir == tmp_path / "flag"
    assert config.report_dir == tmp_path / "flag" / "reports"


def test_list_tasks_with_no_tasks_exits_cleanly(tmp_path, capsys):
    list_tasks(argparse.Namespace(status=None, data_dir=str(tmp_path)))
    assert capsys.readouterr().out.strip() == ""


def test_report_command_renders_a_report(tmp_path, capsys):
    """``generate_report`` was referenced but never imported."""
    _run(tmp_path)
    storage = Storage(tmp_path / "alecto.db")
    task_id = storage.list_tasks()[0].id
    storage.close()

    generate_report_cmd(argparse.Namespace(task_id=task_id, data_dir=str(tmp_path)))
    out = capsys.readouterr().out
    assert "Alecto Report" in out
    assert task_id in out


def test_report_command_missing_task_exits_nonzero(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        generate_report_cmd(argparse.Namespace(task_id="does-not-exist", data_dir=str(tmp_path)))
    assert exc.value.code == 1
    assert "not found" in capsys.readouterr().err
