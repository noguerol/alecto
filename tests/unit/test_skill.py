"""Accuracy guard for the bundled agent skill (``alecto/SKILL.md``).

The skill is the contract an external harness reads, so it must describe the
implementation that actually exists. This suite fails whenever the skill and the
code disagree: a documented operation that is missing, an operation that is not
documented, an import that does not resolve, a Python example that does not
compile, a CLI command that is not registered, or an environment variable that
the code does not read.

It exists because the user guide once shipped examples for classes that did not
exist, and a skill is exactly the kind of artefact that silently rots.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

from alecto.cli import build_parser
from alecto.service import build_catalogue

PKG = Path(__file__).resolve().parents[2] / "alecto"
SKILL = PKG / "SKILL.md"
REPO = PKG.parent


@pytest.fixture(scope="module")
def skill_text() -> str:
    assert SKILL.is_file(), f"skill missing from the package: {SKILL}"
    return SKILL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict[str, str]:
    assert skill_text.startswith("---\n"), "skill must open with YAML frontmatter"
    end = skill_text.index("\n---", 4)
    block = skill_text[4:end]
    data: dict[str, str] = {}
    key = None
    for line in block.splitlines():
        if not line.strip():
            continue
        if line.startswith((" ", "\t")) and key:
            data[key] += " " + line.strip()
        elif ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            data[key] = value.strip()
    return data


# ---------------------------------------------------------------------------
# Frontmatter — what a harness reads to decide whether to load the skill
# ---------------------------------------------------------------------------


class TestFrontmatter:
    def test_has_name_and_description(self, frontmatter):
        assert frontmatter.get("name") == "alecto"
        assert frontmatter.get("description")

    def test_description_has_enough_trigger_context(self, frontmatter):
        description = frontmatter["description"].lower()
        # A harness matches on this text, so it must mention the real triggers.
        for trigger in ("benchmark", "endpoint", "quality", "ttft", "report"):
            assert trigger in description, f"description never mentions {trigger!r}"


# ---------------------------------------------------------------------------
# Operation coverage in both directions
# ---------------------------------------------------------------------------


class TestOperationCoverage:
    def test_every_catalogued_operation_is_documented(self, skill_text):
        """A harness must be able to reach the full functionality."""
        missing = [op.name for op in build_catalogue() if f"`{op.name}`" not in skill_text]
        assert not missing, f"operations absent from SKILL.md: {missing}"

    def test_no_operation_is_documented_that_does_not_exist(self, skill_text):
        known = set(build_catalogue().names())
        claimed = set(re.findall(r"`(alecto_[a-z_]+)`", skill_text))
        assert claimed <= known, f"documented but not catalogued: {sorted(claimed - known)}"

    def test_documented_argument_names_match_the_schemas(self, skill_text):
        """Each operation row must list the argument names the schema declares."""
        problems = []
        for op in build_catalogue():
            row = re.search(rf"^\|\s*`{re.escape(op.name)}`\s*\|([^|]*)\|", skill_text, re.M)
            if row is None:
                continue  # covered by the coverage test
            documented = set(re.findall(r"`([a-z_]+)\??(?::[a-z|]+)?`", row.group(1)))
            declared = set(op.input_schema.get("properties", {}))
            if documented != declared:
                problems.append(
                    f"{op.name}: documented {sorted(documented)} vs schema {sorted(declared)}"
                )
        assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# Executable content
# ---------------------------------------------------------------------------


class TestExamples:
    def test_all_documented_imports_resolve(self, skill_text):
        import importlib

        imports = set(re.findall(r"^\s*from (alecto[\w.]*) import ([\w, ]+)", skill_text, re.M))
        assert imports, "the skill documents no alecto imports"
        problems = []
        for module_name, names in sorted(imports):
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover
                problems.append(f"{module_name}: {type(exc).__name__}: {exc}")
                continue
            for name in (n.strip() for n in names.split(",") if n.strip()):
                if not hasattr(module, name):
                    problems.append(f"{module_name}.{name} does not exist")
        assert not problems, "documented API missing:\n" + "\n".join(problems)

    def test_python_blocks_are_syntactically_valid(self, skill_text):
        blocks = re.findall(r"```python\n(.*?)```", skill_text, re.S)
        assert blocks, "no python examples found"
        for index, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as exc:  # pragma: no cover
                raise AssertionError(f"python block {index} does not parse: {exc}") from exc

    def test_json_session_example_is_valid(self, skill_text):
        import json

        for payload in re.findall(r"^[→←] (\{.*\})$", skill_text, re.M):
            json.loads(payload)  # raises if the protocol example is not JSON

    def test_self_contained_python_blocks_execute(self, skill_text, monkeypatch):
        """Every example a harness might copy must actually run.

        Targets are redirected to the offline mock. The MCP loop example gets an
        in-memory transport so it is genuinely exercised rather than skipped, and
        the streaming example is skipped because the mock adapter refuses to
        stream by design — which is itself the documented behaviour.
        """
        import io
        import json as _json
        import sys as _sys

        blocks = re.findall(r"```python\n(.*?)```", skill_text, re.S)
        assert blocks, "no python examples found"

        request = _json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        monkeypatch.setattr(_sys, "stdin", io.StringIO(request + "\n"))
        monkeypatch.setattr(_sys, "stdout", io.StringIO())

        executed = 0
        namespace: dict = {"__name__": "__skill__"}
        for index, block in enumerate(blocks):
            source = (
                block.replace("TargetKind.OPENAI", "TargetKind.MOCK")
                .replace("http://localhost:8080/v1", "mock://skill")
                .replace("http://localhost:8000/v1", "mock://skill")
            )
            if "streaming=True" in source and "make_streaming_request_fn" in source:
                continue
            exec(compile(source, f"SKILL.md#{index}", "exec"), namespace)
            executed += 1
        assert executed >= 2, "too few examples actually executed"


# ---------------------------------------------------------------------------
# CLI and configuration surface claimed by the skill
# ---------------------------------------------------------------------------


class TestCliSurface:
    @pytest.fixture(scope="class")
    def subcommands(self):
        parser = build_parser()
        action = next(a for a in parser._subparsers._actions if getattr(a, "choices", None))
        return action.choices

    def test_every_advertised_subcommand_exists(self, skill_text, subcommands):
        advertised = set(re.findall(r"^alecto ([a-z-]+)", skill_text, re.M))
        assert advertised, "the skill advertises no CLI commands"
        missing = advertised - set(subcommands)
        assert not missing, f"documented CLI commands not registered: {sorted(missing)}"

    def test_catalogue_formats_exist(self, skill_text, subcommands):
        recorded = set(
            re.findall(r"alecto catalogue --format ([a-z]+)", skill_text)
        )
        available = set(subcommands["catalogue"]._actions[-1].choices)
        assert recorded <= available, f"unknown catalogue formats: {recorded - available}"

    def test_skill_command_supports_the_documented_flags(self, subcommands):
        dests = {a.dest for a in subcommands["skill"]._actions}
        assert {"path", "install_dir"} <= dests

    def test_mcp_command_accepts_data_dir(self, subcommands):
        assert "data_dir" in {a.dest for a in subcommands["mcp"]._actions}


class TestConfigurationSurface:
    def test_documented_env_vars_match_the_code(self, skill_text):
        source = (PKG / "config.py").read_text(encoding="utf-8")
        documented = set(re.findall(r"`(ALECTO_[A-Z_]+)`", skill_text))
        implemented = set(re.findall(r"['\"](ALECTO_[A-Z_]+)['\"]", source))
        assert documented, "the skill documents no environment variables"
        assert documented == implemented, (
            f"documented but not implemented: {sorted(documented - implemented)}; "
            f"implemented but not documented: {sorted(implemented - documented)}"
        )

    def test_metric_field_names_exist(self, skill_text):
        import dataclasses

        from alecto.performance import TimingMetrics

        fields = {f.name for f in dataclasses.fields(TimingMetrics)}
        for name in ("ttfb_ms", "ttft_ms", "e2e_ms", "tokens_per_s"):
            assert name in fields, f"{name} documented but not a TimingMetrics field"
            assert name in skill_text

    def test_quality_score_field_names_exist(self, skill_text):
        import asyncio

        from alecto.adapters import MockAdapter
        from alecto.domain import TargetSpec
        from alecto.enums import TargetKind
        from alecto.quality import run_quality_suite

        adapter = MockAdapter(TargetSpec(kind=TargetKind.MOCK, endpoint="mock://x"))
        artifact = asyncio.run(run_quality_suite("gsm8k", adapter, limit=1))
        for name in ("capability_score", "end_to_end_success"):
            assert name in artifact, f"{name} is not produced by run_quality_suite"
            assert name in skill_text

    def test_documented_suites_are_registered(self, skill_text):
        from alecto.quality import QUALITY_SUITES

        for suite in QUALITY_SUITES:
            assert suite.lower() in skill_text.lower(), f"{suite} not mentioned in the skill"


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


class TestPackaging:
    def test_skill_ships_inside_the_package(self):
        # Anything under alecto/ is included in the wheel by the build config.
        assert SKILL.parent == PKG
        with open(REPO / "pyproject.toml", "rb") as handle:
            config = tomllib.load(handle)
        assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["alecto"]

    def test_skill_is_not_ignored_by_git(self):
        import subprocess

        result = subprocess.run(
            ["git", "check-ignore", "--quiet", str(SKILL)],
            cwd=REPO, capture_output=True,
        )
        assert result.returncode != 0, "SKILL.md is git-ignored and would not ship"
