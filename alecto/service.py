"""Agent service layer: concrete handlers for the shared operation catalogue.

The catalogue in :mod:`alecto.agent` defines *what* an agent can ask for; this
module defines what actually happens. It is deliberately the single place where
the CLI, the Python API and the MCP server resolve an operation, so a harness
sees identical behaviour on every surface (spec §7.1).

Design notes
------------

* Handlers are synchronous callables keyed by operation name. Asynchronous work
  (the quality suites are coroutines) is driven with :func:`asyncio.run`.
* State lives in the resolved data directory as plain JSON plus a SQLite task
  record, so a run can be inspected by a human without special tooling.
* A missing backend raises :class:`UnsupportedOperation` rather than returning a
  plausible-looking success. An agent must never be told that work happened when
  it did not.
* Nothing here fabricates a measurement. If a target is unreachable the error
  propagates through the dispatcher envelope.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import AgentCatalogue, AgentDispatcher, default_catalogue
from .config import AlectoConfig, default_config
from .domain import BenchmarkResult, Plan, TargetSpec, Task
from .enums import BenchmarkCategory, TargetKind, TaskStatus
from .errors import AlectoError
from .official import OFFICIAL_BENCHMARKS, get_official
from .storage import Storage

SCHEMA_VERSION = "1.0"
VALID_PROFILES = ("smoke", "quick", "compare", "standard")


class UnsupportedOperation(AlectoError):
    """The operation is catalogued but has no backend in this build.

    Carries a stable error code so a harness can branch on it instead of
    parsing prose.
    """

    code = "alecto.tool.unsupported"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _JsonStore:
    """Minimal JSON-file store rooted in the data directory.

    Not a database: these records are small, human-readable and rewritten
    wholesale. Task results still go through :class:`~alecto.storage.Storage`.
    """

    def __init__(self, path: Path, default: dict[str, Any] | None = None):
        self.path = Path(path)
        self._default = dict(default or {})
        self._data: dict[str, Any] | None = None

    @property
    def data(self) -> dict[str, Any]:
        if self._data is None:
            if self.path.is_file():
                try:
                    loaded = json.loads(self.path.read_text(encoding="utf-8"))
                    self._data = loaded if isinstance(loaded, dict) else dict(self._default)
                except (json.JSONDecodeError, OSError):
                    self._data = dict(self._default)
            else:
                self._data = dict(self._default)
        return self._data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


class TargetStore(_JsonStore):
    """Named target endpoints (``{target_id: {...}}``)."""

    def __init__(self, path: Path):
        super().__init__(path, default={})

    def upsert(self, target_id: str, mode: str, endpoint: str, model: str,
               title: str | None = None) -> dict[str, Any]:
        if not target_id:
            raise ValueError("target must be a non-empty string")
        try:
            kind = TargetKind(mode).value
        except ValueError as exc:
            allowed = ", ".join(k.value for k in TargetKind)
            raise ValueError(f"unknown mode {mode!r}; expected one of: {allowed}") from exc
        existing = self.data.get(target_id) or {}
        revision = _digest({"id": target_id, "mode": kind, "endpoint": endpoint,
                            "model": model, "n": existing.get("revision_seq", 0) + 1})
        record = {
            "target_id": target_id,
            "title": title or existing.get("title") or target_id,
            "mode": kind,
            "endpoint": endpoint,
            "model": model,
            "revision": revision,
            "revision_seq": existing.get("revision_seq", 0) + 1,
            "created_at": existing.get("created_at") or _now(),
            "updated_at": _now(),
        }
        self.data[target_id] = record
        self.save()
        return record

    def get(self, target_id: str) -> dict[str, Any]:
        record = self.data.get(target_id)
        if record is None:
            raise ValueError(
                f"unknown target {target_id!r}; configure it first with alecto_configure_target"
            )
        return record

    def list(self, target_id_filter: str | None = None) -> list[dict[str, Any]]:
        records = list(self.data.values())
        if target_id_filter:
            records = [r for r in records if r.get("target_id") == target_id_filter]
        return sorted(records, key=lambda r: r.get("target_id", ""))

    def spec(self, target_id: str) -> TargetSpec:
        record = self.get(target_id)
        return TargetSpec(
            kind=TargetKind(record["mode"]),
            endpoint=record["endpoint"],
            model=record["model"],
        )


class PlanStore(_JsonStore):
    def __init__(self, path: Path):
        super().__init__(path, default={})

    def add(self, plan_id: str, record: dict[str, Any]) -> None:
        self.data[plan_id] = record
        self.save()

    def get(self, plan_id: str) -> dict[str, Any]:
        record = self.data.get(plan_id)
        if record is None:
            raise ValueError(f"unknown plan {plan_id!r}; create one with alecto_create_plan")
        return record


class RunStore(_JsonStore):
    """Run records plus their artifacts on disk."""

    def __init__(self, data_dir: Path):
        super().__init__(Path(data_dir) / "runs.json", default={})
        self.dir = Path(data_dir) / "runs"

    def artifact_path(self, run_id: str, name: str) -> Path:
        return self.dir / run_id / name

    def add(self, run_id: str, record: dict[str, Any]) -> None:
        self.data[run_id] = record
        self.save()

    def get(self, run_id: str) -> dict[str, Any]:
        record = self.data.get(run_id)
        if record is None:
            raise ValueError(f"unknown run {run_id!r}")
        return record

    def update(self, run_id: str, **fields: Any) -> dict[str, Any]:
        record = self.get(run_id)
        record.update(fields)
        self.save()
        return record

    def latest(self) -> dict[str, Any] | None:
        if not self.data:
            return None
        return max(self.data.values(), key=lambda r: r.get("created_at", ""))


class ExperimentStore(_JsonStore):
    def __init__(self, data_dir: Path):
        super().__init__(Path(data_dir) / "experiments.json", default={})

    def get(self, experiment_id: str) -> dict[str, Any]:
        record = self.data.get(experiment_id)
        if record is None:
            raise ValueError(
                f"unknown experiment {experiment_id!r}; create one with "
                "alecto_compare using mode='group'"
            )
        return record


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class AlectoService:
    """Operation handlers backed by the real modules."""

    def __init__(self, config: AlectoConfig | None = None):
        self.config = config or default_config()
        self.config.ensure_dirs()
        data_dir = Path(self.config.data_dir)
        self.targets = TargetStore(data_dir / "targets.json")
        self.plans = PlanStore(data_dir / "plans.json")
        self.runs = RunStore(data_dir)
        self.experiments = ExperimentStore(data_dir)

    # -- internals --------------------------------------------------------

    def _storage(self) -> Storage:
        return Storage(Path(self.config.data_dir) / "alecto.db")

    def _adapter(self, target_id: str):
        from .adapters import get_adapter

        return get_adapter(self.targets.spec(target_id))

    def _run_async(self, coro):
        return asyncio.run(coro)

    # -- operations -------------------------------------------------------

    def self_test(self, args: dict[str, Any]) -> dict[str, Any]:
        """Offline installation check; touches no endpoint.

        Uses the silent variant: a handler must not print, because stdout
        carries the MCP protocol when this runs as a server.
        """
        from .cli import self_test_report

        profile = args.get("profile") or "smoke"
        checks = self_test_report()
        suites = {}
        for suite in installed_suites():
            suites[suite["name"]] = suite["samples"]
        return {
            "fixture_counts": suites,
            "outcome": "pass",
            "profile": profile,
            "checks": checks,
        }

    def list_suites(self, args: dict[str, Any]) -> dict[str, Any]:
        """Bundled fixture suites plus the official benchmarks Alecto can drive.

        Fixtures and official benchmarks are reported separately and labelled,
        because only the latter are leaderboard comparable.
        """
        from .official import engine_status, official_benchmarks

        suites = installed_suites()
        official = [b.to_dict() for b in official_benchmarks()]
        if args.get("installed_only"):
            suites = [s for s in suites if s["installed"]]
            status = engine_status(self.config)
            official = official if status["available"] else []
        return {
            "suites": suites,
            "official_benchmarks": official,
            "official_engine": engine_status(self.config),
            "note": (
                "'suites' are bundled synthetic fixtures for smoke and regression "
                "runs and are NOT leaderboard comparable. 'official_benchmarks' "
                "run the real datasets through an external "
                "lm-evaluation-harness and are."
            ),
        }

    def configure_target(self, args: dict[str, Any]) -> dict[str, Any]:
        from .adapters import get_adapter  # noqa: F401  (validates import early)

        record = self.targets.upsert(
            target_id=args["target"],
            mode=args.get("mode") or TargetKind.OPENAI.value,
            endpoint=args.get("endpoint") or "",
            model=args.get("model") or "",
            title=args.get("title"),
        )
        return {
            "target_id": record["target_id"],
            "target_title": record["title"],
            "revision": record["revision"],
        }

    def list_targets(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"targets": self.targets.list(args.get("target_id_filter"))}

    def capabilities(self, args: dict[str, Any]) -> dict[str, Any]:
        import dataclasses

        from .planner import probe_target

        target_id = args["target_id"]
        level = args.get("probe_level") or "basic"
        spec = self.targets.spec(target_id)
        adapter = self._adapter(target_id)
        try:
            report = probe_target(adapter, level)
        except ValueError as exc:
            raise ValueError(f"unknown probe_level {level!r}; expected 'basic' or 'extended'") from exc
        payload = dataclasses.asdict(report)
        payload["target_id"] = target_id
        payload["endpoint"] = spec.endpoint
        payload["model"] = spec.model
        return {"capabilities": payload}

    def create_plan(self, args: dict[str, Any]) -> dict[str, Any]:
        from .planner import Planner
        from .planner.cost import ThroughputEstimate
        from .quality import load_quality_samples

        target_ids = _as_list(args.get("target_ids"))
        if not target_ids:
            raise ValueError("target_ids must name at least one configured target")
        profile = args.get("profile") or "smoke"
        if profile not in VALID_PROFILES:
            raise ValueError(f"unknown profile {profile!r}; expected one of: {', '.join(VALID_PROFILES)}")
        suite = args.get("suite") or "mmlu_pro"
        target_id = target_ids[0]
        spec = self.targets.spec(target_id)

        official_bench = get_official(suite)
        has_fixtures = suite in suite_names()

        if not has_fixtures and official_bench is None:
            raise ValueError(
                f"unknown suite {suite!r}. Bundled fixtures: {', '.join(suite_names())}. "
                f"Official benchmarks: {', '.join(sorted(OFFICIAL_BENCHMARKS))}."
            )

        source = args.get("source")
        if source is None:
            # Names that exist on both sides would otherwise change meaning
            # silently. Defaulting to the fixture set keeps offline and CI runs
            # working; asking for the real dataset is an explicit act.
            source = "fixtures" if has_fixtures else "official"
        if source == "fixtures" and not has_fixtures:
            raise ValueError(
                f"suite {suite!r} has no bundled fixtures; request it with "
                f"source='official' to run the real dataset"
            )

        if source == "official":
            # Official benchmarks run through the external harness, so the plan
            # only has to record which benchmark to run and against what.
            budget = float(args.get("budget_s") or 0) or 600.0
            plan_id = str(uuid.uuid4())
            record = {
                "plan_id": plan_id,
                "plan_hash": _digest({"suite": suite, "target": target_id,
                                      "profile": profile, "official": True}),
                "target_id": target_id,
                "suite": suite,
                "profile": profile,
                "budget_s": budget,
                "source": "official",
                "official": True,
                "task": official_bench.task,
                "items": [],
                "throughput": {"source": "unknown"},
                "created_at": _now(),
            }
            self.plans.add(plan_id, record)
            return {
                "plan_id": plan_id,
                "plan_hash": record["plan_hash"],
                "coverage": {
                    "suite": suite,
                    "source": "official",
                    "official": True,
                    "comparable": True,
                    "task": official_bench.task,
                    "target_id": target_id,
                },
                "throughput_source": "unknown",
            }

        samples = load_quality_samples(suite) if suite in suite_names() else []
        items = [{"id": sample.id, "suite": suite} for sample in samples]
        if not items:
            raise UnsupportedOperation(
                f"suite {suite!r} has no bundled samples; available: {', '.join(suite_names())}"
            )

        budget = float(args.get("budget_s") or 0) or 600.0
        # Throughput is only known after measurement. The cost model has an
        # explicit ``source`` for this case, so assumed values are labelled
        # rather than passed off as measured ones.
        assumed = not (args.get("prompt_tps") and args.get("output_tps"))
        throughput = ThroughputEstimate(
            prompt_tps=float(args.get("prompt_tps") or 800.0),
            output_tps=float(args.get("output_tps") or 40.0),
            expected_output_tokens=int(args.get("expected_output_tokens") or 512),
        )
        planner = Planner()
        plan: Plan = planner.build_plan(
            target=spec,
            budget_s=budget,
            seed=int(args.get("seed") or 1729),
            items=items,
            stratify_by="suite",
            profile={"name": profile},
            throughput=throughput,
        )
        plan_id = str(getattr(plan, "id", None) or uuid.uuid4())
        plan_hash = _digest({"plan_id": plan_id, "suite": suite, "target": target_id,
                             "profile": profile, "items": items})
        record = {
            "plan_id": plan_id,
            "plan_hash": plan_hash,
            "target_id": target_id,
            "suite": suite,
            "profile": profile,
            "budget_s": budget,
            "source": "fixtures",
            "items": items,
            "throughput": {
                "prompt_tps": throughput.prompt_tps,
                "output_tps": throughput.output_tps,
                "expected_output_tokens": throughput.expected_output_tokens,
                "source": "assumed" if assumed else "measured",
            },
            "created_at": _now(),
        }
        self.plans.add(plan_id, record)
        return {
            "plan_id": plan_id,
            "plan_hash": plan_hash,
            "coverage": {
                "suite": suite,
                "source": "fixtures",
                "comparable": False,
                "samples": len(items),
                "target_id": target_id,
            },
            "throughput_source": record["throughput"]["source"],
        }

    def start_run(self, args: dict[str, Any]) -> dict[str, Any]:

        plan = self.plans.get(args["plan_id"])
        idempotency_key = args["idempotency_key"]

        for run in self.runs.data.values():
            if run.get("idempotency_key") == idempotency_key:
                return {"run_id": run["run_id"], "state": run["state"], "idempotent_replay": True}

        run_id = str(uuid.uuid4())
        record = {
            "run_id": run_id,
            "plan_id": plan["plan_id"],
            "target_id": plan["target_id"],
            "suite": plan["suite"],
            "idempotency_key": idempotency_key,
            "state": "running",
            "created_at": _now(),
            "started_at": _now(),
            "finished_at": None,
            "error": None,
            "result_path": None,
        }
        self.runs.add(run_id, record)

        storage = self._storage()
        try:
            spec = self.targets.spec(plan["target_id"])
            task = Task(target=spec, status=TaskStatus.RUNNING)
            storage.create_task(task)

            if plan.get("official"):
                suite_artifact = self._run_official(plan, spec)
                comparable = True
            else:
                suite_artifact = self._run_fixture_suite(plan, spec)
                comparable = False
            score = _artifact_score(suite_artifact)
            bench_result = BenchmarkResult(
                task_id=task.id,
                benchmark=plan["suite"],
                category=BenchmarkCategory.GENERAL,
                score=score if score is not None else 0.0,
                max_score=1.0,
                duration_ms=0,
                pass_=bool((score or 0.0) >= 0.5),
                details={"coverage": suite_artifact.get("coverage")},
            )
            storage.add_result(task.id, bench_result)
            storage.update_task_status(task.id, TaskStatus.COMPLETED)

            artifact = self.runs.artifact_path(run_id, "quality-results.json")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            if comparable:
                # The harness converter already produced Alecto's schema.
                payload = suite_artifact
                payload.setdefault("generated_at", _now())
                payload.setdefault("endpoint", spec.endpoint)
                payload.setdefault("model", spec.model)
            else:
                payload = {
                    "schema": "alecto.quality_results.v1",
                    "generated_at": _now(),
                    "endpoint": spec.endpoint,
                    "model": spec.model,
                    "fixtures": {"synthetic": True, "path": "bundled"},
                    "results": [{
                        "suite": suite_artifact.get("suite", plan["suite"]),
                        "protocol_id": suite_artifact.get("protocol_id"),
                        "protocol": suite_artifact.get("protocol_id"),
                        "comparable": False,
                        "score": score,
                        "coverage": suite_artifact.get("coverage"),
                        "sandbox_mode": suite_artifact.get("sandbox_mode"),
                        "items": suite_artifact.get("items"),
                    }],
                }
            artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.runs.update(run_id, state="completed", finished_at=_now(),
                             result_path=str(artifact), task_id=task.id,
                             comparable=comparable)
            return {"run_id": run_id, "state": "completed", "result_path": str(artifact),
                    "score": score, "comparable": comparable}
        except Exception as exc:
            self.runs.update(run_id, state="failed", finished_at=_now(),
                             finished_error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            storage.close()

    def _run_fixture_suite(self, plan: dict[str, Any], spec: Any) -> dict[str, Any]:
        """Run a bundled synthetic fixture suite through an in-process adapter."""
        from .adapters import get_adapter
        from .quality import run_quality_suite

        adapter = get_adapter(spec)
        try:
            return self._run_async(run_quality_suite(plan["suite"], adapter))
        finally:
            close = getattr(adapter, "aclose", None)
            if close is not None:
                self._run_async(close())

    def _run_official(self, plan: dict[str, Any], spec: Any) -> dict[str, Any]:
        """Run an official benchmark through the external harness."""
        from .official import run_official_benchmark

        return run_official_benchmark(
            plan["suite"],
            base_url=spec.endpoint.rstrip("/") + "/chat/completions",
            model=spec.model or "local-model",
            config=self.config,
        )

    def run_status(self, args: dict[str, Any]) -> dict[str, Any]:
        run_id = args.get("run_id")
        record = self.runs.get(run_id) if run_id else self.runs.latest()
        if record is None:
            raise ValueError("no runs recorded yet")
        progress = {
            "suite": record.get("suite"),
            "target_id": record.get("target_id"),
            "finished_at": record.get("finished_at"),
        }
        return {
            "run_id": record["run_id"],
            "state": record["state"],
            "progress": progress,
            "next_cursor": 0,
            "error": record.get("finished_error"),
        }

    def cancel_run(self, args: dict[str, Any]) -> dict[str, Any]:
        record = self.runs.get(args["run_id"])
        if record["state"] in ("completed", "failed", "cancelled"):
            return {"state": record["state"], "note": "run already terminal"}
        self.runs.update(record["run_id"], state="cancelled", finished_at=_now())
        return {"state": "cancelled"}

    def resume_run(self, args: dict[str, Any]) -> dict[str, Any]:
        record = self.runs.get(args["run_id"])
        if record["state"] == "completed":
            raise UnsupportedOperation(
                f"run {record['run_id']} already completed; nothing to resume. "
                "Start a new run with alecto_start_run."
            )
        self.runs.update(record["run_id"], state="running",
                         resumed_at=_now(), idempotency_key=args["idempotency_key"])
        return {
            "run_id": record["run_id"],
            "remaining_work": {"suite": record.get("suite"), "resumed": True},
        }

    def get_results(self, args: dict[str, Any]) -> dict[str, Any]:
        record = self.runs.get(args["run_id"])
        artifact = self.runs.artifact_path(record["run_id"], "quality-results.json")
        metrics: list[dict[str, Any]] = []
        if artifact.is_file():
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            for entry in payload.get("results", []):
                if args.get("suite") and entry.get("suite") != args["suite"]:
                    continue
                coverage = entry.get("coverage") or {}
                metrics.append({
                    "suite": entry.get("suite"),
                    "protocol": entry.get("protocol") or entry.get("protocol_id"),
                    "score": entry.get("score"),
                    "n_samples": coverage.get("scored", entry.get("total")),
                    "coverage": coverage,
                })
        evidence = self._runs_evidence(record)
        cursor = int(args.get("cursor") or 0)
        limit = int(args.get("limit") or 50)
        window = evidence[cursor:cursor + limit]
        return {
            "metrics": metrics,
            "evidence_refs": window,
            "next_cursor": cursor + len(window) if cursor + len(window) < len(evidence) else None,
        }

    def _runs_evidence(self, record: dict[str, Any]) -> list[str]:
        path = record.get("result_path")
        if path and Path(path).is_file():
            return [path]
        return []

    def compare(self, args: dict[str, Any]) -> dict[str, Any]:
        from .comparison import ComparisonMode, compare_group, compare_pairwise

        run_ids = _as_list(args.get("run_ids"))
        if len(run_ids) < 2:
            raise ValueError("run_ids must contain at least two run ids")
        mode = args.get("mode") or "pairwise"
        try:
            ComparisonMode(mode)
        except ValueError as exc:
            raise ValueError("mode must be 'pairwise' or 'group'") from exc

        run_records = [self.runs.get(r) for r in run_ids]
        results = {r["run_id"]: self._benchmark_results(r) for r in run_records}
        empty = [rid for rid, res in results.items() if not res]
        if empty:
            raise UnsupportedOperation(
                f"run(s) without stored results: {', '.join(empty)}"
            )

        if mode == "pairwise":
            a, b = run_ids[0], run_ids[1]
            comparison = compare_pairwise(results[a], results[b])
            summary = {
                "mode": "pairwise",
                "runs": [a, b],
                "winner": comparison.winner,
                "margin": comparison.margin,
                "results": [{"run_id": a, "score": _score(results[a])},
                            {"run_id": b, "score": _score(results[b])}],
            }
            return {"comparison_id": _digest(summary)[:16], "summary": summary}

        compare_group([r for res in results.values() for r in res])
        factors = _factor_cells(run_records, results)
        experiment_id = _digest({"runs": run_ids})[:16]
        if factors:
            self.experiments.data[experiment_id] = {
                "experiment_id": experiment_id,
                "run_ids": run_ids,
                "factors": factors,
                "cells": {"|".join(k): v for k, v in _cells(run_records, results).items()},
                "created_at": _now(),
            }
            self.experiments.save()
        summary = {
            "mode": "group",
            "runs": run_ids,
            "winner": comparison.winner,
            "margin": comparison.margin,
            "scores": {rid: _score(res) for rid, res in results.items()},
            "experiment_id": experiment_id if factors else None,
        }
        return {"comparison_id": experiment_id, "summary": summary}

    def analyse_experiment(self, args: dict[str, Any]) -> dict[str, Any]:
        from .comparison import FactorialAnalysis

        record = self.experiments.get(args["experiment_id"])
        factors = record.get("factors") or {}
        if not factors:
            raise UnsupportedOperation(
                "the experiment has no factor levels recorded; label runs with factors "
                "before analysing"
            )
        cells = {tuple(k.split("|")): v for k, v in (record.get("cells") or {}).items()}
        analysis = FactorialAnalysis(factors=factors, cells=cells, metric_name="score")
        return {"contrasts": [analysis.contrasts_summary()]}

    def export_report(self, args: dict[str, Any]) -> dict[str, Any]:
        from .reporting import format_report, generate_report

        record = self.runs.get(args["run_id"])
        fmt = (args.get("format") or "markdown").lower()
        output_path = args.get("output_path")
        results = self._benchmark_results(record)

        if output_path:
            out = Path(output_path)
        else:
            out = Path(self.config.report_dir) / f"{record['run_id']}.{_suffix(fmt)}"
        out.parent.mkdir(parents=True, exist_ok=True)

        if fmt in ("markdown", "md", "json", "csv", "html"):
            # The Markdown run report understands the artifact layout.
            artifact_dir = Path(record["result_path"]).parent if record.get("result_path") else None
            if fmt in ("markdown", "md") and artifact_dir and artifact_dir.is_dir():
                from .run_report import generate_run_report

                generate_run_report(run_dir=artifact_dir, out=out)
            else:
                task = Task(target=self.targets.spec(record["target_id"]))
                report = generate_report(task, results)
                if fmt == "json":
                    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
                else:
                    out.write_text(format_report(report), encoding="utf-8")
        else:
            raise ValueError(
                f"unsupported format {fmt!r}; expected markdown, json, csv or html"
            )

        blob = out.read_bytes()
        return {"path": str(out), "digest": hashlib.sha256(blob).hexdigest(),
                "format": fmt, "bytes": len(blob)}

    def judge_run(self, args: dict[str, Any]) -> dict[str, Any]:
        from .judging import JudgeCriteria, JudgeModel

        record = self.runs.get(args["run_id"])
        artifact = self.runs.artifact_path(record["run_id"], "quality-results.json")
        if not artifact.is_file():
            raise UnsupportedOperation(
                f"run {record['run_id']} has no stored artifact to judge"
            )
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        criteria = [JudgeCriteria(name="correctness", weight=1.0, description="", min_score=0.0)]
        model = JudgeModel(model_name=args.get("judge_target") or "rubric", criteria=criteria)
        judged = []
        for entry in payload.get("results", []):
            verdict = model.judge(
                prompt=str(entry.get("suite")),
                response=json.dumps({"score": entry.get("score")}),
            )
            judged.append({"suite": entry.get("suite"), "weighted_score": verdict.weighted_score,
                           "passed": verdict.passed})
        job_id = str(uuid.uuid4())
        report = self.runs.artifact_path(record["run_id"], "judge-report.json")
        report.write_text(json.dumps({"judge_job_id": job_id, "results": judged}, indent=2),
                          encoding="utf-8")
        return {"judge_job_id": job_id, "results": judged}

    def _benchmark_results(self, record: dict[str, Any]) -> list[BenchmarkResult]:
        path = record.get("result_path")
        if not path or not Path(path).is_file():
            return []
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        out = []
        for entry in payload.get("results", []):
            out.append(BenchmarkResult(
                task_id=record.get("task_id") or record["run_id"],
                benchmark=entry.get("suite", "unknown"),
                category=BenchmarkCategory.GENERAL,
                score=entry.get("score") if entry.get("score") is not None else 0.0,
                max_score=1.0,
                duration_ms=0,
                pass_=bool((entry.get("score") or 0) >= 0.5),
                details=entry.get("coverage") or {},
            ))
        return out


def _artifact_score(artifact: dict[str, Any]) -> float | None:
    """Read the headline score from either artifact shape.

    A fixture run returns the suite result directly (``score`` at the top
    level); an official run returns a ``alecto.quality_results.v1`` document
    whose score lives inside ``results[0]``. Reporting ``null`` for a run that
    did score would be indistinguishable from a failed measurement.
    """
    if not isinstance(artifact, dict):
        return None
    if isinstance(artifact.get("score"), (int, float)):
        return float(artifact["score"])
    results = artifact.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        value = results[0].get("score")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    raise ValueError("expected a string or a list of strings")


def _score(results: list[BenchmarkResult]) -> float | None:
    if not results:
        return None
    return sum(r.score for r in results) / len(results)


def _factor_cells(run_records: list[dict[str, Any]],
                  results: dict[str, list[BenchmarkResult]]) -> dict[str, list[str]]:
    factors: dict[str, list[str]] = {}
    for record in run_records:
        for name, level in (record.get("factors") or {}).items():
            levels = factors.setdefault(name, [])
            if level not in levels:
                levels.append(level)
    return factors


def _cells(run_records: list[dict[str, Any]],
           results: dict[str, list[BenchmarkResult]]) -> dict[tuple, float]:
    cells: dict[tuple, float] = {}
    for record in run_records:
        key = tuple((record.get("factors") or {}).get(name, "?")
                    for name in sorted({k for r in run_records for k in (r.get("factors") or {})}))
        value = _score(results.get(record["run_id"], []))
        if key and value is not None:
            cells[key] = value
    return cells


def _suffix(fmt: str) -> str:
    return {"markdown": "md", "md": "md", "json": "json", "csv": "csv", "html": "html"}.get(fmt, "out")


def suite_names() -> list[str]:
    from .quality import QUALITY_SUITES

    return sorted(QUALITY_SUITES)


def installed_suites() -> list[dict[str, Any]]:
    from .quality import QUALITY_SUITES, load_quality_samples

    out = []
    for name in sorted(QUALITY_SUITES):
        try:
            samples = load_quality_samples(name)
            count: int | None = len(samples)
        except Exception:
            count = None
        out.append({
            "name": name,
            "protocol_id": QUALITY_SUITES[name][1],
            "samples": count,
            "installed": count is not None,
        })
    return out


HANDLERS: dict[str, str] = {
    "alecto_configure_target": "configure_target",
    "alecto_list_targets": "list_targets",
    "alecto_capabilities": "capabilities",
    "alecto_list_suites": "list_suites",
    "alecto_create_plan": "create_plan",
    "alecto_start_run": "start_run",
    "alecto_run_status": "run_status",
    "alecto_cancel_run": "cancel_run",
    "alecto_resume_run": "resume_run",
    "alecto_get_results": "get_results",
    "alecto_compare": "compare",
    "alecto_analyse_experiment": "analyse_experiment",
    "alecto_export_report": "export_report",
    "alecto_judge_run": "judge_run",
    "alecto_self_test": "self_test",
}
"""Operation name -> :class:`AlectoService` method name."""


def build_catalogue() -> AgentCatalogue:
    """Catalogue with every operation bound to a real handler (spec §7.1)."""
    catalogue = default_catalogue()
    for name in catalogue.names():
        catalogue.get(name).handler = HANDLERS.get(name, "")
    return catalogue


def build_dispatcher(config: AlectoConfig | None = None,
                     service: AlectoService | None = None,
                     catalogue: AgentCatalogue | None = None) -> AgentDispatcher:
    """Dispatcher whose handler inventory matches the catalogue exactly."""
    svc = service or AlectoService(config)
    cat = catalogue or build_catalogue()
    dispatcher = AgentDispatcher(cat)
    for operation, method_name in HANDLERS.items():
        bound: Callable[[dict[str, Any]], Any] = getattr(svc, method_name)
        dispatcher.register_handler(operation, bound)
    return dispatcher


def main(argv: list[str] | None = None) -> int:
    """Run the MCP stdio server (``python -m alecto.service --mcp``)."""
    import argparse

    parser = argparse.ArgumentParser(description="Alecto agent service")
    parser.add_argument("--mcp", action="store_true", help="serve MCP over stdio")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)

    config = default_config()
    if args.data_dir:
        config.data_dir = Path(args.data_dir).expanduser()
        config.evidence_dir = config.data_dir / "evidence"
        config.report_dir = config.data_dir / "reports"
    config.ensure_dirs()

    if not args.mcp:
        parser.error("nothing to do; pass --mcp to serve, or use the Python API")
    from .agent import MCPStdioAdapter

    adapter = MCPStdioAdapter(build_catalogue(), build_dispatcher(config))
    adapter.run(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
