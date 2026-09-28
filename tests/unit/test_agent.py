"""Unit tests for alecto.agent (WP-09 agent surfaces)."""

import io
import json

import pytest

from alecto.agent import (
    AgentDispatcher,
    MCPStdioAdapter,
    ManifestFormat,
    Operation,
    default_catalogue,
)
from alecto.errors import ValidationError

EXPECTED_OPS = [
    "alecto_configure_target",
    "alecto_list_targets",
    "alecto_capabilities",
    "alecto_list_suites",
    "alecto_create_plan",
    "alecto_start_run",
    "alecto_run_status",
    "alecto_cancel_run",
    "alecto_resume_run",
    "alecto_get_results",
    "alecto_compare",
    "alecto_analyse_experiment",
    "alecto_export_report",
    "alecto_judge_run",
    "alecto_self_test",
]


@pytest.fixture
def catalogue():
    return default_catalogue()


@pytest.fixture
def dispatcher(catalogue):
    d = AgentDispatcher(catalogue)
    d.register_handler("alecto_self_test", lambda args: {"outcome": "pass", "fixture_counts": {"unit": 42}})
    return d


class TestOperation:
    def test_to_dict(self):
        op = Operation(
            name="x",
            description="does x",
            input_schema={"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            output_schema={"type": "object"},
        )
        d = op.to_dict()
        assert d["name"] == "x"
        assert d["description"] == "does x"
        assert d["input_schema"]["required"] == ["a"]
        assert d["side_effects"] == []
        assert d["api_version"] == "1.0"


class TestAgentCatalogue:
    def test_default_has_15_ops(self, catalogue):
        assert len(catalogue) == 15
        assert set(catalogue.names()) == set(EXPECTED_OPS)

    def test_get(self, catalogue):
        op = catalogue.get("alecto_start_run")
        assert op.name == "alecto_start_run"
        assert op.description

    def test_get_unknown_raises(self, catalogue):
        with pytest.raises(KeyError):
            catalogue.get("nope")

    def test_contains_and_iter(self, catalogue):
        assert "alecto_self_test" in catalogue
        assert "nope" not in catalogue
        assert {op.name for op in catalogue} == set(EXPECTED_OPS)

    def test_duplicate_register_raises(self, catalogue):
        with pytest.raises(Exception):
            catalogue.register(catalogue.get("alecto_self_test"))


class TestValidateInput:
    def test_missing_required(self, catalogue):
        problems = catalogue.validate_input("alecto_start_run", {"plan_id": "p1"})
        assert any("idempotency_key" in p for p in problems)

    def test_unknown_field(self, catalogue):
        problems = catalogue.validate_input(
            "alecto_start_run",
            {"plan_id": "p1", "idempotency_key": "k", "bogus": 1},
        )
        assert any("bogus" in p for p in problems)

    def test_valid_args(self, catalogue):
        assert catalogue.validate_input(
            "alecto_start_run", {"plan_id": "p1", "idempotency_key": "k"}
        ) == []

    def test_wrong_type(self, catalogue):
        problems = catalogue.validate_input("alecto_list_suites", {"installed_only": "yes"})
        assert any("installed_only" in p for p in problems)


class TestCatalogueExports:
    def test_canonical_json_pure(self, catalogue):
        text = catalogue.to_canonical_json()
        data = json.loads(text)
        assert data["schema_version"] == "1.0"
        assert len(data["operations"]) == 15
        assert "```" not in text

    def test_mcp_tools(self, catalogue):
        tools = catalogue.to_mcp_tools()
        assert len(tools) == 15
        for tool in tools:
            assert set(tool) >= {"name", "description", "inputSchema"}

    def test_openai_tools(self, catalogue):
        tools = catalogue.to_openai_tools()
        assert len(tools) == 15
        for tool in tools:
            assert tool["type"] == "function"
            assert set(tool["function"]) == {"name", "description", "parameters"}

    def test_anthropic_tools(self, catalogue):
        tools = catalogue.to_anthropic_tools()
        assert len(tools) == 15
        for tool in tools:
            assert set(tool) == {"name", "description", "input_schema"}


class TestAgentDispatcher:
    def test_dispatch_ok(self, dispatcher):
        env = dispatcher.dispatch("alecto_self_test", {"profile": "quick"})
        assert env["ok"] is True
        assert env["operation"] == "alecto_self_test"
        assert env["data"]["outcome"] == "pass"
        assert env["error"] is None
        assert env["diagnostics"] == []
        assert env["schema_version"] == "1.0"

    def test_unknown_operation(self, dispatcher):
        env = dispatcher.dispatch("alecto_nope", {})
        assert env["ok"] is False
        assert env["error"]["code"] == "alecto.tool.unknown_operation"

    def test_validation_failure_does_not_call_handler(self, dispatcher):
        env = dispatcher.dispatch("alecto_start_run", {"plan_id": "p1"})
        assert env["ok"] is False
        assert env["error"]["code"] == "alecto.validation.failed"

    def test_handler_error(self, dispatcher):
        def boom(args):
            raise RuntimeError("kaput")

        dispatcher.register_handler("alecto_list_targets", boom)
        env = dispatcher.dispatch("alecto_list_targets", {})
        assert env["ok"] is False
        assert env["error"]["code"] == "alecto.tool.error"
        assert "kaput" in env["error"]["message"]

    def test_dispatch_all(self, dispatcher):
        envs = dispatcher.dispatch_all([
            ("alecto_self_test", {"profile": "quick"}),
            ("alecto_self_test", {}),
        ])
        assert len(envs) == 2
        assert envs[0]["ok"] is True
        assert envs[1]["ok"] is True


class TestMCPStdioAdapter:
    def _adapter(self, dispatcher):
        return MCPStdioAdapter(default_catalogue(), dispatcher)

    def test_initialize(self, dispatcher):
        adapter = self._adapter(dispatcher)
        resp = json.loads(adapter.handle_request(adapter.encode_request("initialize", {}, 1)))
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert resp["result"]["capabilities"] == {"tools": {}}

    def test_tools_list_matches_catalogue(self, dispatcher):
        adapter = self._adapter(dispatcher)
        resp = json.loads(adapter.handle_request(adapter.encode_request("tools/list", {}, 2)))
        tools = resp["result"]["tools"]
        assert len(tools) == 15
        assert {t["name"] for t in tools} == set(EXPECTED_OPS)

    def test_tools_call(self, dispatcher):
        adapter = self._adapter(dispatcher)
        resp = json.loads(
            adapter.handle_request(
                adapter.encode_request("tools/call", {"name": "alecto_self_test", "arguments": {}}, 3)
            )
        )
        content = resp["result"]["content"]
        assert content[0]["type"] == "text"
        envelope = json.loads(content[0]["text"])
        assert envelope["ok"] is True
        assert envelope["operation"] == "alecto_self_test"

    def test_unknown_method(self, dispatcher):
        adapter = self._adapter(dispatcher)
        resp = json.loads(adapter.handle_request(adapter.encode_request("nope", {}, 4)))
        assert resp["error"]["code"] == -32601

    def test_malformed_json(self, dispatcher):
        adapter = self._adapter(dispatcher)
        resp = json.loads(adapter.handle_request("{not json"))
        assert resp["error"]["code"] == -32700

    def test_notification_no_response(self, dispatcher):
        adapter = self._adapter(dispatcher)
        assert adapter.handle_request(adapter.encode_request("notifications/initialized")) is None

    def test_run_loop(self, dispatcher):
        adapter = self._adapter(dispatcher)
        incoming = io.StringIO(adapter.encode_request("initialize", {}, 1) + "\n")
        outgoing = io.StringIO()
        adapter.run(incoming, outgoing)
        resp = json.loads(outgoing.getvalue().strip())
        assert resp["result"]["protocolVersion"] == "2024-11-05"


class TestManifestFormat:
    def test_parse_valid_json(self):
        manifest = {"schema_version": "1.0", "targets": [{"target_id": "t1"}], "tasks": []}
        assert ManifestFormat().parse(json.dumps(manifest)) == manifest

    def test_parse_rejects_surrounding_text(self):
        with pytest.raises(ValidationError):
            ManifestFormat().parse("here is the manifest:\n" + json.dumps({"schema_version": "1.0"}))

    def test_parse_doc_flavor(self):
        manifest = {"schema_version": "1.0", "targets": []}
        doc = "```json\n" + json.dumps(manifest) + "\n```"
        assert ManifestFormat().parse(doc, flavor="doc") == manifest

    def test_render_canonical(self):
        manifest = {"schema_version": "1.0", "targets": []}
        assert json.loads(ManifestFormat().render(manifest)) == manifest
        assert "```" not in ManifestFormat().render(manifest)

    def test_render_doc(self):
        text = ManifestFormat().render({"a": 1}, flavor="doc")
        assert "```json" in text

    def test_validate_missing_keys(self):
        problems = ManifestFormat().validate({"targets": []})
        assert any("schema_version" in p for p in problems)

    def test_validate_ok(self):
        manifest = {"schema_version": "1.0", "targets": [{"target_id": "t1"}], "tasks": []}
        assert ManifestFormat().validate(manifest) == []
