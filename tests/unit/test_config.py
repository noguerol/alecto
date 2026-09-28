"""Unit tests for configuration."""

import json
from pathlib import Path

from alecto.config import AlectoConfig, load_config


class TestAlectoConfig:
    def test_defaults(self):
        config = AlectoConfig()
        assert config.default_timeout_s == 30.0
        assert config.max_concurrent_tasks == 4
        assert config.mock_backend is False

    def test_ensure_dirs(self, tmp_path):
        config = AlectoConfig(data_dir=tmp_path)
        config.ensure_dirs()
        assert config.data_dir.exists()
        assert config.evidence_dir.exists()
        assert config.report_dir.exists()

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ALECTO_DATA", str(tmp_path))
        config = load_config()
        assert config.data_dir == tmp_path


class TestLoadConfig:
    def test_from_file(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"default_timeout_s": 60.0, "max_concurrent_tasks": 8}))
        config = load_config(config_file)
        assert config.default_timeout_s == 60.0
        assert config.max_concurrent_tasks == 8

    def test_missing_file(self, tmp_path):
        config = load_config(tmp_path / "nonexistent.json")
        assert config.default_timeout_s == 30.0  # default


class TestConfigAcceptsPlainStrings:
    """The declared Path fields must not be a runtime trap for str callers.

    The documented construction in README.md passes plain strings, which used
    to raise ``AttributeError: 'str' object has no attribute 'expanduser'``.
    """

    def test_str_data_dir_is_coerced(self):
        config = AlectoConfig(data_dir="./data")
        assert config.data_dir == Path("data")
        assert config.evidence_dir == Path("data/evidence")
        assert config.report_dir == Path("data/reports")

    def test_str_evidence_and_report_dirs_are_coerced(self):
        config = AlectoConfig(data_dir="/tmp/d", evidence_dir="/tmp/e", report_dir="/tmp/r")
        assert config.evidence_dir == Path("/tmp/e")
        assert config.report_dir == Path("/tmp/r")

    def test_path_inputs_still_work(self):
        config = AlectoConfig(data_dir=Path("/tmp/p"))
        assert config.data_dir == Path("/tmp/p")

    def test_tilde_is_expanded(self):
        config = AlectoConfig(data_dir="~/alecto-test")
        assert "~" not in str(config.data_dir)

    def test_readme_snippet_is_executable(self):
        from alecto import AlectoConfig as PublicConfig

        config = PublicConfig(data_dir="./data", default_timeout_s=60.0, max_concurrent_tasks=4)
        assert config.default_timeout_s == 60.0
        assert config.max_concurrent_tasks == 4


class TestPublicApi:
    def test_config_is_exported(self):
        import alecto

        assert "AlectoConfig" in alecto.__all__
        assert "default_config" in alecto.__all__

    def test_every_export_is_importable(self):
        import alecto

        for name in alecto.__all__:
            assert hasattr(alecto, name), f"{name} is exported but missing"


class TestDocumentedEnvVarsMatchCode:
    """README must not document env vars the code ignores (or omit real ones).

    A documented-but-unread variable is a silent no-op for the user.
    """

    def test_readme_and_code_agree(self):
        import re
        from pathlib import Path as _Path

        repo = _Path(__file__).resolve().parents[2]
        readme = (repo / "README.md").read_text(encoding="utf-8")
        source = (repo / "alecto" / "config.py").read_text(encoding="utf-8")

        documented = set(re.findall(r"`(ALECTO_[A-Z_]+)`", readme))
        implemented = set(re.findall(r"['\"](ALECTO_[A-Z_]+)['\"]", source))

        assert documented, "no env vars documented in README"
        assert documented == implemented, (
            f"documented but not implemented: {sorted(documented - implemented)}; "
            f"implemented but not documented: {sorted(implemented - documented)}"
        )
