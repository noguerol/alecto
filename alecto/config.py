"""Configuration loading for alecto (spec §4.2)."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AlectoConfig:
    """Runtime configuration for the alecto engine."""

    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("ALECTO_DATA", "~/.alecto")).expanduser())
    default_timeout_s: float = 30.0
    max_concurrent_tasks: int = 4
    heartbeat_interval_s: float = 5.0
    lease_timeout_s: float = 30.0
    max_retries: int = 2
    evidence_dir: Path | None = None
    report_dir: Path | None = None
    mock_backend: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be > 0")
        self.data_dir = self.data_dir.expanduser()
        if self.evidence_dir is None:
            self.evidence_dir = self.data_dir / "evidence"
        if self.report_dir is None:
            self.report_dir = self.data_dir / "reports"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> AlectoConfig:
    """Load configuration from a JSON file or environment variables."""
    import json

    config = AlectoConfig()

    if path is not None:
        p = Path(path)
        if p.exists():
            data: dict[str, Any]
            try:
                with open(p, "rb") as f:
                    data = tomllib.load(f)
            except tomllib.TOMLDecodeError:
                with open(p) as f:
                    data = json.load(f)
            unknown = [k for k in data if not hasattr(config, k)]
            if unknown:
                raise ValueError(f"Unknown config key(s): {', '.join(unknown)}")
            for key, value in data.items():
                setattr(config, key, value)

    # Environment overrides
    env_map = {
        "ALECTO_DATA": "data_dir",
        "ALECTO_TIMEOUT": "default_timeout_s",
        "ALECTO_MAX_CONCURRENT": "max_concurrent_tasks",
        "ALECTO_MOCK": "mock_backend",
    }
    for env_var, attr in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            if attr == "data_dir":
                config.data_dir = Path(val).expanduser()
            elif attr in ("default_timeout_s", "max_concurrent_tasks", "heartbeat_interval_s", "lease_timeout_s", "max_retries"):
                setattr(config, attr, float(val) if attr == "default_timeout_s" else int(val))
            elif attr == "mock_backend":
                config.mock_backend = val.lower() in ("1", "true", "yes")

    return config


def default_config() -> AlectoConfig:
    return load_config()
