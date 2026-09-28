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
