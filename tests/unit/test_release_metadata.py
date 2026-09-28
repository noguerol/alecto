"""Release metadata consistency.

The version string appears in the package metadata, the importable module and
the documentation. A mismatch between them ships a package that misreports its
own version, which is easy to do and hard to notice.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import alecto

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
