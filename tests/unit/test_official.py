"""Tests for official benchmarks and the external-harness integration.

Official benchmarks are the only path to leaderboard-comparable numbers, so the
risk here is not a crash: it is quietly reporting a fixture score as an official
one, or silently running a different harness than the one the user pinned. These
tests drive a stub engine that emits harness-shaped output, so the conversion is
exercised end to end without installing lm-evaluation-harness.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from alecto.config import AlectoConfig
from alecto.official import (
    ENGINE_ENV_VAR,
    OFFICIAL_BENCHMARKS,
    ExternalEngineUnavailable,
    engine_status,
    find_engine,
    get_official,
    is_official,
    official_benchmarks,
    run_official_benchmark,
)
from alecto.service import AlectoService

# ---------------------------------------------------------------------------
# A stub engine: answers the capability probe and writes harness-shaped output.
# ---------------------------------------------------------------------------

STUB = '''\
#!/usr/bin/env python3
"""Stand-in for an lm_eval installation, sufficient for converter tests."""
import json
import sys
from pathlib import Path

args = sys.argv[1:]

if "-c" in args:                      # capability probe: `python -c "import lm_eval"`
    print("lm_eval")
    raise SystemExit(0)

out = None
task = "gpqa"
for i, a in enumerate(args):
    if a == "--output_path":
        out = Path(args[i + 1])
    if a == "--tasks":
        task = args[i + 1]

if out is None:
    print("no --output_path given", file=sys.stderr)
    raise SystemExit(2)

model_dir = out / "stub-model"
model_dir.mkdir(parents=True, exist_ok=True)

metric = "exact_match"
rows = [
    {"id": i, metric: 1.0 if i < 6 else 0.0, "doc": {"question": f"q{i}"}}
    for i in range(10)
]
with open(model_dir / f"samples_{task}_2026-01-01T00-00-00.jsonl", "w") as fh:
    for row in rows:
        fh.write(json.dumps(row) + "\\n")

results = {
    "results": {task: {f"{metric},none": 0.6, f"{metric}_stderr,none": 0.155}},
    "configs": {task: {"num_fewshot": 0}},
    "versions": {task: 1},
    "n-samples": {task: {"original": 448, "effective": 10}},
    "config": {"model": "local-chat-completions"},
}
with open(model_dir / "results_2026-01-01T00-00-00.json", "w") as fh:
    json.dump(results, fh)
'''


@pytest.fixture
def stub_engine(tmp_path) -> str:
    path = tmp_path / "stub_lm_eval"
    path.write_text(STUB, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


@pytest.fixture
def config(tmp_path, stub_engine) -> AlectoConfig:
    return AlectoConfig(data_dir=tmp_path / "data", eval_python=stub_engine)


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_names_and_tasks_are_unique(self):
        names = [b.name for b in official_benchmarks()]
        tasks = [b.task for b in official_benchmarks()]
        assert len(names) == len(set(names))
        assert len(tasks) == len(set(tasks))

    def test_every_entry_is_usable(self):
        for bench in official_benchmarks():
            assert bench.name and bench.task
            assert bench.description, f"{bench.name} has no description"
            assert bench.category, f"{bench.name} has no category"

    def test_the_four_fixture_suites_are_also_available_officially(self):
        """The suites Alecto ships fixtures for must be runnable for real."""
        from alecto.quality import QUALITY_SUITES

        for suite in QUALITY_SUITES:
            assert is_official(suite), f"{suite} has fixtures but no official path"

    def test_lookup_helpers(self):
        assert get_official("gsm8k") is not None
        assert get_official("not-a-benchmark") is None
        assert not is_official("not-a-benchmark")

    def test_metadata_marks_them_comparable(self):
        payload = get_official("gpqa").to_dict()
        assert payload["comparable"] is True
        assert payload["official"] is True
        assert payload["engine"] == "lm-evaluation-harness"


# ---------------------------------------------------------------------------
# Engine resolution
# ---------------------------------------------------------------------------


class TestEngineResolution:
    def test_stub_engine_is_detected(self, config, stub_engine):
        assert find_engine(config) == [stub_engine]

    def test_status_reports_the_engine(self, config):
        status = engine_status(config)
        assert status["available"] is True
        assert status["env_var"] == ENGINE_ENV_VAR

    def test_explicit_engine_is_authoritative(self, tmp_path):
        """A pinned interpreter must never be silently swapped for another.

        Falling back would run a different harness than the user pinned, which
        makes measured numbers non-reproducible.
        """
        config = AlectoConfig(data_dir=tmp_path / "d", eval_python="/nonexistent/python")
        assert find_engine(config) is None
        assert engine_status(config)["available"] is False

    def test_env_var_pins_the_engine(self, stub_engine, monkeypatch, tmp_path):
        monkeypatch.setenv(ENGINE_ENV_VAR, stub_engine)
        assert find_engine(AlectoConfig(data_dir=tmp_path / "d")) == [stub_engine]

    def test_env_var_pointing_nowhere_disables_official_runs(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENGINE_ENV_VAR, "/nonexistent/python")
        assert find_engine(AlectoConfig(data_dir=tmp_path / "d")) is None


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


class TestRunOfficialBenchmark:
    def test_produces_a_comparable_artifact(self, config):
        artifact = run_official_benchmark(
            "gpqa",
            base_url="http://localhost:8080/v1/chat/completions",
            model="test-model",
            config=config,
        )
        assert artifact["comparable"] is True
        assert artifact["benchmark"] == "gpqa"
        assert artifact["task"] == "gpqa"
        assert artifact["engine"] == "lm-evaluation-harness"
        suite = artifact["results"][0]
        assert suite["suite"] == "gpqa"
        assert suite["score"] == pytest.approx(0.6)
        assert suite["comparable"] is True
        assert suite["metric_id"] == "exact_match,none"

    def test_records_the_command_for_reproduction(self, config):
        artifact = run_official_benchmark(
            "gpqa", base_url="http://x/v1", model="m", config=config,
        )
        assert "lm_eval" in artifact["command"]
        assert "gpqa" in artifact["command"]

    def test_unknown_benchmark_is_rejected_before_running(self, config):
        with pytest.raises(ValueError, match="unknown official benchmark"):
            run_official_benchmark("nope", base_url="http://x/v1", model="m", config=config)

    def test_missing_engine_raises_a_typed_error(self, tmp_path):
        config = AlectoConfig(data_dir=tmp_path / "d", eval_python="/nonexistent/python")
        with pytest.raises(ExternalEngineUnavailable) as exc:
            run_official_benchmark("gsm8k", base_url="http://x/v1", model="m", config=config)
        assert exc.value.code == "alecto.tool.unsupported"
        # The message must say how to fix it, not just that it failed.
        assert ENGINE_ENV_VAR in str(exc.value)

    def test_engine_that_produces_nothing_is_reported(self, tmp_path):
        silent = tmp_path / "silent_engine"
        silent.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(0)\n", encoding="utf-8")
        silent.chmod(0o755)
        config = AlectoConfig(data_dir=tmp_path / "d", eval_python=str(silent))
        with pytest.raises(ExternalEngineUnavailable, match="no results"):
            run_official_benchmark("gsm8k", base_url="http://x/v1", model="m", config=config)


# ---------------------------------------------------------------------------
# Service integration
# ---------------------------------------------------------------------------


class TestServiceIntegration:
    def test_list_suites_separates_fixtures_from_official(self, config):
        service = AlectoService(config)
        payload = service.list_suites({})
        assert [s["name"] for s in payload["suites"]] == sorted(
            {"gsm8k", "humaneval", "ifeval", "mmlu_pro"}
        )
        assert len(payload["official_benchmarks"]) == len(OFFICIAL_BENCHMARKS)
        assert payload["official_engine"]["available"] is True
        # The distinction must be stated, not implied.
        assert "NOT leaderboard comparable" in payload["note"]

    def test_installed_only_hides_official_when_no_engine(self, tmp_path):
        service = AlectoService(
            AlectoConfig(data_dir=tmp_path / "d", eval_python="/nonexistent/python")
        )
        payload = service.list_suites({"installed_only": True})
        assert payload["official_benchmarks"] == []
        assert payload["official_engine"]["available"] is False

    def test_plan_marks_official_benchmarks(self, config):
        service = AlectoService(config)
        service.configure_target(
            {"target": "m", "mode": "openai", "endpoint": "http://localhost:1/v1", "model": "mm"}
        )
        plan = service.create_plan({"target_ids": ["m"], "profile": "smoke", "suite": "gpqa"})
        assert plan["coverage"]["official"] is True
        assert plan["coverage"]["source"] == "official"
        assert plan["coverage"]["comparable"] is True
        assert plan["coverage"]["task"] == "gpqa"
        assert plan["throughput_source"] == "unknown"

    def test_ambiguous_name_defaults_to_fixtures(self, config):
        """A name in both registries must not change meaning on its own.

        Defaulting to the offline fixtures keeps CI and quick checks working;
        the real dataset is an explicit request.
        """
        service = AlectoService(config)
        service.configure_target(
            {"target": "m", "mode": "mock", "endpoint": "mock://x", "model": "mm"}
        )
        plan = service.create_plan({"target_ids": ["m"], "profile": "smoke", "suite": "gsm8k"})
        assert plan["coverage"]["source"] == "fixtures"
        assert plan["coverage"]["comparable"] is False

    def test_official_can_be_requested_explicitly_for_a_fixture_name(self, config):
        service = AlectoService(config)
        service.configure_target(
            {"target": "m", "mode": "openai", "endpoint": "http://localhost:1/v1", "model": "mm"}
        )
        plan = service.create_plan(
            {"target_ids": ["m"], "profile": "smoke", "suite": "gsm8k", "source": "official"}
        )
        assert plan["coverage"]["source"] == "official"
        assert plan["coverage"]["comparable"] is True

    def test_asking_for_fixtures_that_do_not_exist_is_an_error(self, config):
        service = AlectoService(config)
        service.configure_target(
            {"target": "m", "mode": "mock", "endpoint": "mock://x", "model": "mm"}
        )
        with pytest.raises(ValueError, match="no bundled fixtures"):
            service.create_plan(
                {"target_ids": ["m"], "profile": "smoke", "suite": "gpqa", "source": "fixtures"}
            )

    def test_unknown_suite_lists_both_registries(self, config):
        service = AlectoService(config)
        service.configure_target(
            {"target": "m", "mode": "mock", "endpoint": "mock://x", "model": "mm"}
        )
        with pytest.raises(ValueError) as exc:
            service.create_plan({"target_ids": ["m"], "profile": "smoke", "suite": "nope"})
        assert "Bundled fixtures" in str(exc.value)
        assert "Official benchmarks" in str(exc.value)

    def test_official_only_benchmark_cannot_be_requested_as_fixtures(self, config):
        """Asking for fixtures that do not exist must fail, not run something else."""
        service = AlectoService(config)
        service.configure_target(
            {"target": "m", "mode": "mock", "endpoint": "mock://x", "model": "mm"}
        )
        with pytest.raises(ValueError, match="no bundled fixtures"):
            service.create_plan(
                {"target_ids": ["m"], "profile": "smoke", "suite": "gpqa", "source": "fixtures"}
            )

    def test_start_run_records_comparable_flag(self, config):
        """A run must say whether its number is comparable."""
        service = AlectoService(config)
        service.configure_target(
            {"target": "m", "mode": "openai", "endpoint": "http://localhost:1/v1", "model": "mm"}
        )
        plan = service.create_plan({"target_ids": ["m"], "profile": "smoke", "suite": "gpqa"})
        run = service.start_run({"plan_id": plan["plan_id"], "idempotency_key": "k"})
        assert run["comparable"] is True
        stored = service.runs.get(run["run_id"])
        assert stored["comparable"] is True
        payload = json.loads(Path(run["result_path"]).read_text())
        assert payload["results"][0]["comparable"] is True

    def test_fixture_run_is_not_marked_comparable(self, config):
        service = AlectoService(config)
        service.configure_target(
            {"target": "m", "mode": "mock", "endpoint": "mock://x", "model": "mm"}
        )
        plan = service.create_plan({"target_ids": ["m"], "profile": "smoke", "suite": "gsm8k"})
        run = service.start_run({"plan_id": plan["plan_id"], "idempotency_key": "k"})
        assert run["comparable"] is False


# ---------------------------------------------------------------------------
# Configuration surface
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_eval_python_defaults_to_none(self, tmp_path):
        assert AlectoConfig(data_dir=tmp_path / "d").eval_python is None

    def test_env_var_is_read_into_the_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENGINE_ENV_VAR, "/opt/harness/bin/python")
        from alecto.config import load_config

        assert load_config().eval_python == "/opt/harness/bin/python"

    def test_documented_env_var_matches_this_module(self):
        """The variable this module reads must be the one the docs describe."""
        assert ENGINE_ENV_VAR == "ALECTO_EVAL_PYTHON"
        source = (Path(__file__).resolve().parents[2] / "alecto" / "config.py").read_text()
        assert f'"{ENGINE_ENV_VAR}"' in source

    def test_no_env_var_leaks_between_tests(self):
        assert os.environ.get(ENGINE_ENV_VAR) is None


class TestScoreExtraction:
    """``start_run`` must not report ``null`` for a run that did score.

    Fixture runs and official runs return different artifact shapes, and the
    official one keeps its score inside ``results[0]``. Reporting ``null`` there
    is indistinguishable from a failed measurement.
    """

    def test_official_artifact_score_is_read(self):
        from alecto.service import _artifact_score

        assert _artifact_score({"schema": "alecto.quality_results.v1",
                                "results": [{"suite": "gpqa", "score": 0.6}]}) == pytest.approx(0.6)

    def test_fixture_artifact_score_is_read(self):
        from alecto.service import _artifact_score

        assert _artifact_score({"suite": "gsm8k", "score": 0.75}) == pytest.approx(0.75)

    def test_zero_is_reported_as_zero_not_missing(self):
        """A measured zero is a result; it must survive the shape handling."""
        from alecto.service import _artifact_score

        assert _artifact_score({"score": 0.0}) == 0.0
        assert _artifact_score({"results": [{"score": 0.0}]}) == 0.0

    def test_missing_score_is_none(self):
        from alecto.service import _artifact_score

        assert _artifact_score({"results": []}) is None
        assert _artifact_score({}) is None
        assert _artifact_score({"results": [{"score": None}]}) is None

    def test_empty_results_does_not_raise(self):
        from alecto.service import _artifact_score

        assert _artifact_score({"results": []}) is None
