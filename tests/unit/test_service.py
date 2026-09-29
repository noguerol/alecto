"""Tests for the agent service layer and the MCP server path.

These cover the contract an external harness depends on: the catalogue and the
handler inventory must agree, every catalogued operation must answer, and
nothing may write to stdout while serving MCP because stdout *is* the protocol
channel.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from alecto.agent import MCPStdioAdapter
from alecto.config import AlectoConfig
from alecto.service import (
    HANDLERS,
    AlectoService,
    UnsupportedOperation,
    build_catalogue,
    build_dispatcher,
)


@pytest.fixture
def service(tmp_path):
    return AlectoService(AlectoConfig(data_dir=tmp_path))


@pytest.fixture
def mock_target(service):
    service.configure_target(
        {"target": "mock1", "mode": "mock", "endpoint": "mock://local", "model": "m1"}
    )
    return "mock1"


def _run(service, target_id="mock1", suite="mmlu_pro", key="k1"):
    plan = service.create_plan(
        {"target_ids": [target_id], "profile": "smoke", "suite": suite}
    )
    return service.start_run({"plan_id": plan["plan_id"], "idempotency_key": key})


# ---------------------------------------------------------------------------
# Catalogue / handler agreement
# ---------------------------------------------------------------------------


class TestCatalogueContract:
    def test_inventory_matches_handlers(self):
        catalogue = build_catalogue()
        dispatcher = build_dispatcher()
        assert catalogue.inventory_matches_handlers(dispatcher)

    def test_every_operation_has_a_handler(self):
        catalogue = build_catalogue()
        for name in catalogue.names():
            assert name in HANDLERS, f"{name} has no service handler"
            assert catalogue.get(name).handler, f"{name} has no handler marker"

    def test_default_catalogue_still_has_fifteen_operations(self):
        assert len(build_catalogue()) == 15


# ---------------------------------------------------------------------------
# Operations, end to end against the offline mock target
# ---------------------------------------------------------------------------


class TestWorkflow:
    def test_self_test_passes_offline(self, service):
        payload = service.self_test({})
        assert payload["outcome"] == "pass"
        assert payload["checks"]

    def test_list_suites_reports_bundled_fixtures(self, service):
        suites = service.list_suites({})["suites"]
        names = {s["name"] for s in suites}
        assert {"mmlu_pro", "gsm8k", "humaneval", "ifeval"} <= names
        assert all(s["installed"] for s in suites)

    def test_configure_and_list_target(self, service):
        record = service.configure_target(
            {"target": "t1", "mode": "mock", "endpoint": "mock://x", "model": "m"}
        )
        assert record["target_id"] == "t1"
        assert record["revision"]
        assert service.list_targets({})["targets"][0]["target_id"] == "t1"

    def test_configure_target_is_a_new_revision_when_changed(self, service):
        a = service.configure_target({"target": "t", "mode": "mock", "endpoint": "mock://a", "model": "m"})
        b = service.configure_target({"target": "t", "mode": "mock", "endpoint": "mock://b", "model": "m"})
        assert a["revision"] != b["revision"]

    def test_unknown_mode_is_rejected(self, service):
        with pytest.raises(ValueError, match="unknown mode"):
            service.configure_target({"target": "t", "mode": "carrier-pigeon"})

    def test_plan_run_status_results(self, service, mock_target):
        plan = service.create_plan(
            {"target_ids": [mock_target], "profile": "smoke", "suite": "mmlu_pro"}
        )
        assert plan["coverage"]["samples"] > 0
        assert plan["throughput_source"] == "assumed"

        run = service.start_run({"plan_id": plan["plan_id"], "idempotency_key": "k"})
        assert run["state"] == "completed"
        assert Path(run["result_path"]).is_file()

        status = service.run_status({"run_id": run["run_id"]})
        assert status["state"] == "completed"

        results = service.get_results({"run_id": run["run_id"]})
        assert results["metrics"][0]["suite"] == "mmlu_pro"
        assert results["metrics"][0]["n_samples"] is not None
        assert results["evidence_refs"]

    def test_start_run_is_idempotent(self, service, mock_target):
        plan = service.create_plan(
            {"target_ids": [mock_target], "profile": "smoke", "suite": "gsm8k"}
        )
        first = service.start_run({"plan_id": plan["plan_id"], "idempotency_key": "same"})
        second = service.start_run({"plan_id": plan["plan_id"], "idempotency_key": "same"})
        assert first["run_id"] == second["run_id"]
        assert second["idempotent_replay"] is True

    def test_export_writes_markdown(self, service, mock_target, tmp_path):
        run = _run(service, mock_target)
        out = tmp_path / "report.md"
        result = service.export_report(
            {"run_id": run["run_id"], "format": "markdown", "output_path": str(out)}
        )
        assert out.is_file()
        assert result["bytes"] > 0
        assert len(result["digest"]) == 64

    def test_export_rejects_unknown_format(self, service, mock_target):
        run = _run(service, mock_target)
        with pytest.raises(ValueError, match="unsupported format"):
            service.export_report(
                {"run_id": run["run_id"], "format": "pdf", "output_path": "x.pdf"}
            )

    def test_judge_run_produces_a_report(self, service, mock_target):
        run = _run(service, mock_target)
        judged = service.judge_run({"run_id": run["run_id"]})
        assert judged["judge_job_id"]
        assert judged["results"]


class TestRefusalsAreExplicit:
    """A harness must be able to tell "not supported" from "it failed"."""

    def test_resume_of_completed_run_is_unsupported(self, service, mock_target):
        run = _run(service, mock_target)
        with pytest.raises(UnsupportedOperation) as exc:
            service.resume_run({"run_id": run["run_id"], "idempotency_key": "again"})
        assert exc.value.code == "alecto.tool.unsupported"

    def test_unknown_experiment_points_at_the_fix(self, service):
        with pytest.raises(ValueError, match="alecto_compare"):
            service.analyse_experiment({"experiment_id": "missing"})

    def test_compare_needs_two_runs(self, service, mock_target):
        run = _run(service, mock_target)
        with pytest.raises(ValueError, match="at least two"):
            service.compare({"run_ids": [run["run_id"]], "mode": "pairwise"})

    def test_unknown_target_is_reported(self, service):
        with pytest.raises(ValueError, match="unknown target"):
            service.create_plan(
                {"target_ids": ["nope"], "profile": "smoke", "suite": "mmlu_pro"}
            )

    def test_unknown_suite_is_reported(self, service, mock_target):
        with pytest.raises(ValueError, match="unknown suite"):
            service.create_plan(
                {"target_ids": [mock_target], "profile": "smoke", "suite": "bogus"}
            )


# ---------------------------------------------------------------------------
# MCP transport integrity — the bug this suite exists to prevent
# ---------------------------------------------------------------------------


def _rpc(adapter, method, params=None, rid=1):
    """Send one JSON-RPC request and return the decoded response."""
    raw = adapter.handle_request(
        json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
    )
    assert raw is not None, f"no response for {method}"
    return json.loads(raw)


class TestMcpTransport:
    def test_self_test_handler_does_not_write_to_stdout(self, service, capsys):
        """Printing in a handler corrupts the JSON-RPC channel.

        ``alecto_self_test`` originally called the CLI implementation, which
        prints four lines, so every MCP response after it was unparseable.
        """
        service.self_test({})
        assert capsys.readouterr().out == ""

    def test_run_loop_emits_only_json_lines(self, service, tmp_path):
        adapter = MCPStdioAdapter(build_catalogue(), build_dispatcher(service=service))
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "alecto_self_test", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
             "params": {"name": "alecto_list_suites", "arguments": {}}},
        ]
        stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
        stdout = io.StringIO()
        adapter.run(stdin, stdout)

        lines = [ln for ln in stdout.getvalue().splitlines() if ln.strip()]
        assert len(lines) == len(requests)
        for line in lines:
            json.loads(line)  # raises if a handler polluted the stream

    def test_tools_call_returns_a_parseable_envelope(self, service):
        adapter = MCPStdioAdapter(build_catalogue(), build_dispatcher(service=service))
        response = _rpc(adapter, "tools/call",
                        {"name": "alecto_self_test", "arguments": {}}, rid=7)
        payload = json.loads(response["result"]["content"][0]["text"])
        assert payload["ok"] is True
        assert payload["operation"] == "alecto_self_test"

    def test_unknown_field_is_rejected_before_dispatch(self, service):
        adapter = MCPStdioAdapter(build_catalogue(), build_dispatcher(service=service))
        response = _rpc(adapter, "tools/call",
                        {"name": "alecto_list_targets", "arguments": {"bogus": 1}}, rid=8)
        payload = json.loads(response["result"]["content"][0]["text"])
        assert payload["ok"] is False
        assert payload["error"]["code"] == "alecto.validation.failed"

    def test_unknown_operation_is_reported(self, service):
        adapter = MCPStdioAdapter(build_catalogue(), build_dispatcher(service=service))
        response = _rpc(adapter, "tools/call",
                        {"name": "alecto_does_not_exist", "arguments": {}}, rid=9)
        payload = json.loads(response["result"]["content"][0]["text"])
        assert payload["error"]["code"] == "alecto.tool.unknown_operation"
