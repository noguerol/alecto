"""Agent surfaces for alecto (spec §7.1, §22 WP-09).

Shared operation catalogue, dispatcher, MCP stdio adapter and manifest
format. All three interfaces (CLI, Python dispatch, MCP stdio) are
first-class and expose equivalent semantics (AC-02/AC-03/AC-04).

Stdlib-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .errors import ValidationError


SCHEMA_VERSION = "1.0"
MCP_PROTOCOL_VERSION = "2024-11-05"


@dataclass
class Operation:
    """A single agent operation definition (spec §7.1)."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    side_effects: list[str] = field(default_factory=list)
    handler: str = ""
    api_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "side_effects": list(self.side_effects),
            "handler": self.handler,
            "api_version": self.api_version,
        }


def _props(*names: str, types: str = "string", required: bool = True) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {n: {"type": types} for n in names},
        "required": list(names) if required else [],
    }


def _default_operations() -> list[Operation]:
    return [
        Operation(
            name="alecto_configure_target",
            description="Create or update a named target endpoint specification.",
            input_schema=_props("target", "mode"),
            output_schema={"type": "object", "properties": {"target_id": {"type": "string"}, "target_title": {"type": "string"}, "revision": {"type": "string"}}},
            side_effects=["mutates target configuration"],
        ),
        Operation(
            name="alecto_list_targets",
            description="List configured named targets, optionally filtered by ID.",
            input_schema=_props("target_id_filter", required=False),
            output_schema={"type": "object", "properties": {"targets": {"type": "array"}}},
        ),
        Operation(
            name="alecto_capabilities",
            description="Probe a target for capability information at a given probe level.",
            input_schema=_props("target_id", "probe_level"),
            output_schema={"type": "object", "properties": {"capabilities": {"type": "object"}}},
        ),
        Operation(
            name="alecto_list_suites",
            description="List installed benchmark suites and their prerequisites.",
            input_schema=_props("installed_only", types="boolean", required=False),
            output_schema={"type": "object", "properties": {"suites": {"type": "array"}}},
        ),
        Operation(
            name="alecto_create_plan",
            description="Create a bounded benchmark plan for the given targets and profile.",
            input_schema=_props("target_ids", "profile", "budget_s", required=False),
            output_schema={"type": "object", "properties": {"plan_id": {"type": "string"}, "plan_hash": {"type": "string"}, "coverage": {"type": "object"}}},
            side_effects=["creates plan artifact"],
        ),
        Operation(
            name="alecto_start_run",
            description="Atomically commit a durable job for the plan and spawn a worker.",
            input_schema=_props("plan_id", "idempotency_key"),
            output_schema={"type": "object", "properties": {"run_id": {"type": "string"}, "state": {"type": "string"}}},
            side_effects=["creates durable job"],
        ),
        Operation(
            name="alecto_run_status",
            description="Poll structured progress for a run with an event cursor.",
            input_schema=_props("run_id", required=False),
            output_schema={"type": "object", "properties": {"state": {"type": "string"}, "progress": {"type": "object"}, "next_cursor": {"type": "integer"}}},
        ),
        Operation(
            name="alecto_cancel_run",
            description="Cancel a run; stops new requests and performs bounded cleanup.",
            input_schema=_props("run_id"),
            output_schema={"type": "object", "properties": {"state": {"type": "string"}}},
            side_effects=["cancels durable job"],
        ),
        Operation(
            name="alecto_resume_run",
            description="Resume an interrupted run under a new attempt identity.",
            input_schema=_props("run_id", "idempotency_key"),
            output_schema={"type": "object", "properties": {"run_id": {"type": "string"}, "remaining_work": {"type": "object"}}},
            side_effects=["resumes durable job"],
        ),
        Operation(
            name="alecto_get_results",
            description="Read metrics and paginated evidence references for a run.",
            input_schema=_props("run_id", "suite", "cursor", "limit", required=False),
            output_schema={"type": "object", "properties": {"metrics": {"type": "array"}, "evidence_refs": {"type": "array"}}},
        ),
        Operation(
            name="alecto_compare",
            description="Compare runs in a given comparison mode.",
            input_schema=_props("run_ids", "mode"),
            output_schema={"type": "object", "properties": {"comparison_id": {"type": "string"}, "summary": {"type": "object"}}},
        ),
        Operation(
            name="alecto_analyse_experiment",
            description="Compute main and interaction contrasts for an experiment.",
            input_schema=_props("experiment_id"),
            output_schema={"type": "object", "properties": {"contrasts": {"type": "array"}}},
        ),
        Operation(
            name="alecto_export_report",
            description="Export a report for a run or comparison in the given format.",
            input_schema=_props("run_id", "format", "output_path"),
            output_schema={"type": "object", "properties": {"path": {"type": "string"}, "digest": {"type": "string"}}},
            side_effects=["writes report artifact"],
        ),
        Operation(
            name="alecto_judge_run",
            description="Start a separate endpoint-judge job for a run.",
            input_schema=_props("run_id", "judge_target", "budget_s", required=False),
            output_schema={"type": "object", "properties": {"judge_job_id": {"type": "string"}}},
            side_effects=["creates judge job"],
        ),
        Operation(
            name="alecto_self_test",
            description="Run the offline self-test for the given profile.",
            input_schema=_props("profile", required=False),
            output_schema={"type": "object", "properties": {"fixture_counts": {"type": "object"}, "outcome": {"type": "string"}}},
        ),
    ]


class AgentCatalogue:
    """Shared catalogue of agent surfaces (spec §7.1).

    One authoritative operation registry for CLI, Python invocation and MCP.
    """

    def __init__(self) -> None:
        self._ops: dict[str, Operation] = {}

    def register(self, op: Operation) -> None:
        if op.name in self._ops:
            raise ValueError(f"duplicate operation: {op.name}")
        self._ops[op.name] = op

    def get(self, name: str) -> Operation:
        if name not in self._ops:
            raise KeyError(f"unknown operation: {name}")
        return self._ops[name]

    def names(self) -> list[str]:
        return list(self._ops.keys())

    def __len__(self) -> int:
        return len(self._ops)

    def __contains__(self, name: str) -> bool:
        return name in self._ops

    def __iter__(self) -> Iterable[Operation]:
        return iter(self._ops.values())

    def validate_input(self, name: str, arguments: dict[str, Any]) -> list[str]:
        """Validate arguments against the operation input schema.

        Returns a list of problems (empty when valid). Unknown fields and
        invalid types/enums fail validation (spec §7.1).
        """
        op = self.get(name)
        schema = op.input_schema or {}
        problems: list[str] = []
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(arguments, dict):
            return ["arguments must be an object"]
        for r in required:
            if r not in arguments:
                problems.append(f"missing required field: {r}")
        for key in arguments:
            if key not in props:
                problems.append(f"unknown field: {key}")
                continue
            spec = props[key]
            problems.extend(_check_type(key, arguments[key], spec))
        return problems

    def to_canonical_json(self) -> str:
        """Pure JSON catalogue; no surrounding text or banners (AC-04)."""
        return json.dumps(
            {"schema_version": SCHEMA_VERSION, "operations": [op.to_dict() for op in self._ops.values()]},
            sort_keys=True,
        )

    def to_mcp_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": op.name, "description": op.description, "inputSchema": op.input_schema}
            for op in self._ops.values()
        ]

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": op.name,
                    "description": op.description,
                    "parameters": op.input_schema,
                },
            }
            for op in self._ops.values()
        ]

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": op.name, "description": op.description, "input_schema": op.input_schema}
            for op in self._ops.values()
        ]

    def inventory_matches_handlers(self, dispatcher: "AgentDispatcher") -> bool:
        """Schema and handler inventories must match exactly (spec §7.1)."""
        expected = {op.name for op in self._ops.values() if op.handler}
        return expected == set(dispatcher.handlers.keys())


def _check_type(field_name: str, value: Any, spec: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    expected = spec.get("type")
    if expected == "string" and not isinstance(value, str):
        problems.append(f"{field_name}: expected string")
    elif expected == "integer" and not (isinstance(value, int) and not isinstance(value, bool)):
        problems.append(f"{field_name}: expected integer")
    elif expected == "number" and not (isinstance(value, (int, float)) and not isinstance(value, bool)):
        problems.append(f"{field_name}: expected number")
    elif expected == "boolean" and not isinstance(value, bool):
        problems.append(f"{field_name}: expected boolean")
    elif expected == "array" and not isinstance(value, list):
        problems.append(f"{field_name}: expected array")
    elif expected == "object" and not isinstance(value, dict):
        problems.append(f"{field_name}: expected object")
    if "enum" in spec and value not in spec["enum"]:
        problems.append(f"{field_name}: invalid enum value {value!r}")
    return problems


def default_catalogue() -> AgentCatalogue:
    """Catalogue with the 15 spec §7.1 operations registered."""
    catalogue = AgentCatalogue()
    for op in _default_operations():
        catalogue.register(op)
    return catalogue


class AgentDispatcher:
    """Dispatch tasks to agents (spec §7.1: validate before dispatch)."""

    def __init__(self, catalogue: AgentCatalogue, handlers: dict[str, Callable] | None = None):
        self.catalogue = catalogue
        self.handlers: dict[str, Callable] = dict(handlers or {})

    def register_handler(self, name: str, fn: Callable) -> None:
        self.handlers[name] = fn

    def dispatch(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        envelope = _envelope(operation)
        if operation not in self.catalogue:
            envelope["ok"] = False
            envelope["error"] = _error("alecto.tool.unknown_operation", f"unknown operation: {operation}")
            return envelope
        problems = self.catalogue.validate_input(operation, arguments)
        if problems:
            envelope["ok"] = False
            envelope["error"] = _error("alecto.validation.failed", "; ".join(problems), details=problems)
            return envelope
        handler = self.handlers.get(operation)
        if handler is None:
            envelope["ok"] = False
            envelope["error"] = _error("alecto.tool.unknown_operation", f"no handler for operation: {operation}")
            return envelope
        try:
            envelope["data"] = handler(arguments)
        except Exception as exc:  # handler failure
            envelope["ok"] = False
            envelope["error"] = _error("alecto.tool.error", str(exc), retryable=False)
        return envelope

    def dispatch_all(self, operations: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        return [self.dispatch(name, args) for name, args in operations]


def _envelope(operation: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "operation": operation,
        "data": None,
        "error": None,
        "diagnostics": [],
    }


def _error(code: str, message: str, retryable: bool = False, details: Any = None) -> dict[str, Any]:
    err = {"code": code, "message": message, "retryable": retryable}
    if details is not None:
        err["details"] = details
    return err


class MCPStdioAdapter:
    """MCP stdio protocol adapter for agent communication.

    Stdio transport only; no listening port is created (spec §7.1).
    tools/list schemas match the catalogue (AC-03).
    """

    def __init__(self, catalogue: AgentCatalogue, dispatcher: AgentDispatcher | None = None):
        self.catalogue = catalogue
        self.dispatcher = dispatcher or AgentDispatcher(catalogue)

    def handle_request(self, line: str) -> str | None:
        line = line.strip()
        if not line:
            return None
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            return self._rpc_error(None, -32700, "parse error")
        if not isinstance(request, dict):
            return self._rpc_error(None, -32700, "parse error")
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if method == "initialize":
            return self._rpc_result(req_id, {"protocolVersion": MCP_PROTOCOL_VERSION, "capabilities": {"tools": {}}})
        if method == "tools/list":
            return self._rpc_result(req_id, {"tools": self.catalogue.to_mcp_tools()})
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            result = self.dispatcher.dispatch(name, arguments)
            return self._rpc_result(
                req_id,
                {"content": [{"type": "text", "text": json.dumps(result)}]},
            )
        if method == "notifications/initialized":
            return None
        return self._rpc_error(req_id, -32601, f"method not found: {method}")

    def run(self, transport_in, transport_out) -> None:
        """Loop over file-like objects until EOF or a shutdown request."""
        while True:
            line = transport_in.readline()
            if not line:
                break
            response = self.handle_request(line)
            if response is not None:
                transport_out.write(response + "\n")
                transport_out.flush()

    def encode_request(self, method: str, params: dict[str, Any] | None = None, req_id: int | None = None) -> str:
        request: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            request["params"] = params
        if req_id is not None:
            request["id"] = req_id
        return json.dumps(request)

    def _rpc_result(self, req_id: int | None, result: dict[str, Any]) -> str:
        return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})

    def _rpc_error(self, req_id: int | None, code: int, message: str) -> str:
        return json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


class ManifestFormat:
    """Manifest format for agent task definitions (spec §1.1.2).

    The machine manifest MUST be valid JSON with no surrounding text;
    documentation output is a separate option.
    """

    REQUIRED_KEYS = ("schema_version", "targets")

    def parse(self, text: str, flavor: str = "canonical") -> dict[str, Any]:
        if flavor == "doc":
            text = self._unwrap_doc(text)
        stripped = text.strip()
        try:
            manifest = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"manifest is not valid JSON: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ValidationError("manifest must be a JSON object")
        return manifest

    def render(self, manifest: dict[str, Any], flavor: str = "canonical") -> str:
        payload = json.dumps(manifest, sort_keys=True, indent=2)
        if flavor == "doc":
            return f"```json\n{payload}\n```"
        return payload

    def validate(self, manifest: dict[str, Any]) -> list[str]:
        problems: list[str] = []
        if not isinstance(manifest, dict):
            return ["manifest must be a JSON object"]
        for key in self.REQUIRED_KEYS:
            if key not in manifest:
                problems.append(f"missing required key: {key}")
        if "schema_version" in manifest and not isinstance(manifest["schema_version"], str):
            problems.append("schema_version must be a string")
        targets = manifest.get("targets")
        if targets is not None and not isinstance(targets, list):
            problems.append("targets must be a list")
        ops = manifest.get("operations", manifest.get("tasks"))
        if ops is not None and not isinstance(ops, list):
            problems.append("operations/tasks must be a list")
        return problems

    @staticmethod
    def _unwrap_doc(text: str) -> str:
        stripped = text.strip()
        if "```json" in stripped:
            start = stripped.index("```json") + len("```json")
            end = stripped.rfind("```")
            if end > start:
                return stripped[start:end].strip()
        if "```" in stripped:
            start = stripped.index("```") + 3
            end = stripped.rfind("```")
            if end > start:
                return stripped[start:end].strip()
        return stripped
