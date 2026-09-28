"""Unit tests for configuration."""

import json


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
