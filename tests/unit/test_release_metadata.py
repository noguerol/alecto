"""Release metadata consistency.

The version string appears in the package metadata, the importable module and
the documentation. A mismatch between them ships a package that misreports its
own version, which is easy to do and hard to notice.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import alecto
from alecto.cli import build_parser

REPO = Path(__file__).resolve().parents[2]


def _pyproject_version() -> str:
    with open(REPO / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_pyproject_and_module_version_agree():
    assert _pyproject_version() == alecto.__version__


def test_docs_reference_the_current_version():
    text = (REPO / "docs" / "alecto.md").read_text(encoding="utf-8")
    found = set(re.findall(r"(?i)version[\"*\s:]*([0-9]+\.[0-9]+\.[0-9]+)", text))
    assert found, "no version reference found in docs/alecto.md"
    assert found == {alecto.__version__}, f"stale version in docs: {found}"


def test_changelog_documents_the_current_version():
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{alecto.__version__}]" in text


def test_license_file_is_present_and_mit():
    text = (REPO / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License")
    assert "WITHOUT WARRANTY OF ANY KIND" in text


def test_readme_does_not_reference_the_placeholder_org():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert "your-org" not in text


class TestReadmeSurface:
    """The README is the front door; it must not advertise what does not exist.

    It previously showed `alecto configure`, a command that was never
    implemented, and a `run --target my-target` invocation that could not reach
    a real endpoint because the URL was hardcoded to the mock.
    """

    @pytest.fixture(scope="class")
    def parser(self):
        return build_parser()

    def _subcommands(self, parser):
        action = next(a for a in parser._subparsers._actions if getattr(a, "choices", None))
        return action.choices

    def _documented_commands(self):
        """Extract `alecto <command>` invocations from the README's bash blocks.

        Only whole command invocations count: prose and arguments that merely
        mention the name (`ruff check alecto tests`) are not commands.
        """
        text = (REPO / "README.md").read_text(encoding="utf-8")
        invocations = set()
        for block in re.findall(r"```bash\n(.*?)```", text, re.S):
            for line in block.splitlines():
                line = line.strip()
                # allow `report=$(alecto list ...)` style capture
                line = re.sub(r"^\w+=\$\(", "", line)
                match = re.match(r"alecto ([a-z][a-z-]+)\b", line)
                if match:
                    invocations.add(match.group(1))
        return invocations

    def test_every_documented_command_exists(self, parser):
        documented = self._documented_commands()
        assert documented, "README documents no CLI invocations"
        unknown = documented - set(self._subcommands(parser))
        assert not unknown, f"README documents commands that are not registered: {sorted(unknown)}"

    def test_readme_exercises_the_core_commands(self, parser):
        """A front page should show the real entry points, not just a corner."""
        documented = self._documented_commands()
        for expected in ("self-test", "run", "list", "report", "mcp", "skill", "catalogue"):
            assert expected in documented, f"README never shows `alecto {expected}`"

    def test_documented_run_flags_exist(self, parser):
        text = (REPO / "README.md").read_text(encoding="utf-8")
        run_flags = {a.option_strings[0] for a in self._subcommands(parser)["run"]._actions
                     if a.option_strings}
        documented = set(re.findall(r"--(target|endpoint|model|benchmark|data-dir)\b", text))
        missing = {f"--{d}" for d in documented} - run_flags
        assert not missing, f"README documents run flags that do not exist: {sorted(missing)}"

    def test_documented_benchmark_names_are_registered(self):
        from alecto.benchmarks import BENCHMARKS

        text = (REPO / "README.md").read_text(encoding="utf-8")
        for name in re.findall(r"`(\w+_basic)`", text):
            assert name in BENCHMARKS, f"README names an unknown benchmark: {name}"

    def test_banner_image_is_present_and_referenced(self):
        text = (REPO / "README.md").read_text(encoding="utf-8")
        banner = REPO / "readme.png"
        assert banner.is_file(), "readme.png is missing"
        assert "readme.png" in text, "the banner is not referenced by the README"
        assert text.strip().startswith("!"), "the banner must open the README"

    def test_readme_is_english_only(self):
        text = (REPO / "README.md").read_text(encoding="utf-8")
        spanish = ["Rendimiento", "Calidad", "Resumen", "Conclusión", "Limitaciones",
                   "Instalación", "Ejemplo", "Uso"]
        for word in spanish:
            assert word not in text, f"README contains Spanish text: {word}"

    def test_documented_env_vars_match_the_code(self):
        text = (REPO / "README.md").read_text(encoding="utf-8")
        source = (REPO / "alecto" / "config.py").read_text(encoding="utf-8")
        documented = set(re.findall(r"`(ALECTO_[A-Z_]+)`", text))
        implemented = set(re.findall(r"['\"](ALECTO_[A-Z_]+)['\"]", source))
        assert documented == implemented, (
            f"documented but not implemented: {sorted(documented - implemented)}; "
            f"implemented but not documented: {sorted(implemented - documented)}"
        )


class TestDocumentedApiExists:
    """Every ``from alecto… import …`` shown in the docs must resolve.

    The user guide previously demonstrated classes and methods that did not
    exist (``QualitySuite``, ``ContextSuite``, ``run_performance_suite``,
    ``AlectoConfig.from_file``), so following the documentation raised
    ImportError or AttributeError. This guards the whole family at once.
    """

    def _doc_imports(self):
        import re
        from pathlib import Path as _Path

        text = (_Path(REPO) / "docs" / "alecto.md").read_text(encoding="utf-8")
        return sorted(set(re.findall(r"^\s*from (alecto[\w.]*) import ([\w, ]+)", text, re.M)))

    def test_doc_has_importable_examples(self):
        import importlib

        found = self._doc_imports()
        assert found, "no alecto imports found in docs/alecto.md"
        problems = []
        for module_name, names in found:
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - failure path
                problems.append(f"{module_name}: {type(exc).__name__}: {exc}")
                continue
            for name in (n.strip() for n in names.split(",") if n.strip()):
                if not hasattr(module, name):
                    problems.append(f"{module_name}.{name} does not exist")
        assert not problems, "documented API missing:\n" + "\n".join(problems)

    def test_documented_timing_fields_exist(self):
        import dataclasses

        from alecto.performance import TimingMetrics

        documented = {"ttfb_ms", "ttft_ms", "e2e_ms", "tokens_per_s", "p50_ms",
                      "p90_ms", "p95_ms", "n", "ttft_status", "ttfb_status"}
        actual = {f.name for f in dataclasses.fields(TimingMetrics)}
        assert documented <= actual, f"missing: {sorted(documented - actual)}"

    def test_documented_quality_suites_exist(self):
        from alecto.quality import QUALITY_SUITES

        for suite in ("mmlu_pro", "gsm8k", "humaneval", "ifeval"):
            assert suite in QUALITY_SUITES, f"{suite} documented but not registered"
