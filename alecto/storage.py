"""SQLite WAL storage for alecto (spec §4.3)."""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .domain import Task, TaskStatus, Verdict


def _json_default(obj):
    """Fallback for JSON serialization of non-serializable domain objects."""
    if hasattr(obj, "value"):
        return obj.value
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items()}
    return str(obj)


class Storage:
    """SQLite-based storage with WAL mode for task persistence."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                target_kind TEXT,
                target_endpoint TEXT,
                target_model TEXT,
                deadline_s REAL,
                retry_count INTEGER DEFAULT 0,
                error TEXT,
                verdict TEXT,
                plan_json TEXT,
                metrics_json TEXT
            );

            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(id),
                kind TEXT NOT NULL,
                path TEXT,
                text TEXT,
                metric_name TEXT,
                value REAL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS results (
                task_id TEXT NOT NULL REFERENCES tasks(id),
                benchmark TEXT NOT NULL,
                category TEXT,
                score REAL,
                max_score REAL,
                duration_ms INTEGER,
                pass INTEGER,
                details_json TEXT,
                PRIMARY KEY (task_id, benchmark)
            );

            CREATE TABLE IF NOT EXISTS reports (
                task_id TEXT NOT NULL REFERENCES tasks(id),
                title TEXT,
                verdict TEXT,
                sections_json TEXT,
                created_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_evidence_task ON evidence(task_id);
        """)
        conn.commit()

    def create_task(self, task: Task) -> None:
        with self._lock:
            conn = self._get_conn()
            plan_json = json.dumps(task.plan.__dict__, default=_json_default) if task.plan else None
            metrics_json = json.dumps(task.metrics, default=_json_default)
            conn.execute(
                """INSERT INTO tasks (id, status, created_at, started_at, finished_at,
                   target_kind, target_endpoint, target_model, deadline_s, retry_count, error, verdict, plan_json, metrics_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task.id,
                    task.status.value,
                    task.created_at.isoformat(),
                    task.started_at.isoformat() if task.started_at else None,
                    task.finished_at.isoformat() if task.finished_at else None,
                    task.target.kind.value if task.target else None,
                    task.target.endpoint if task.target else None,
                    task.target.model if task.target else None,
                    task.deadline_s,
                    task.retry_count,
                    task.error,
                    task.verdict.value if task.verdict else None,
                    plan_json,
                    metrics_json,
                ),
            )
            conn.commit()

    def update_task_status(self, task_id: str, status: TaskStatus, **kwargs) -> None:
        with self._lock:
            conn = self._get_conn()
            sets = ["status = ?"]
            params: list[Any] = [status.value]
            if "started_at" in kwargs:
                sets.append("started_at = ?")
                params.append(kwargs["started_at"].isoformat() if kwargs["started_at"] else None)
            if "finished_at" in kwargs:
                sets.append("finished_at = ?")
                params.append(kwargs["finished_at"].isoformat() if kwargs["finished_at"] else None)
            if "error" in kwargs:
                sets.append("error = ?")
                params.append(kwargs["error"])
            if "verdict" in kwargs:
                sets.append("verdict = ?")
                params.append(kwargs["verdict"].value if kwargs["verdict"] else None)
            if "retry_count" in kwargs:
                sets.append("retry_count = ?")
                params.append(kwargs["retry_count"])
            if "metrics" in kwargs:
                sets.append("metrics_json = ?")
                params.append(json.dumps(kwargs["metrics"]))
            params.append(task_id)
            conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", params)
            conn.commit()

    def get_task(self, task_id: str) -> Task | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            return None
        return self._row_to_task(row)

    def list_tasks(self, status: TaskStatus | None = None) -> list[Task]:
        conn = self._get_conn()
        if status:
            rows = conn.execute("SELECT * FROM tasks WHERE status = ?", (status.value,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
        return [self._row_to_task(r) for r in rows]

    def add_evidence(self, task_id: str, evidence) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO evidence (id, task_id, kind, path, text, metric_name, value, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    evidence.id,
                    task_id,
                    evidence.kind.value,
                    evidence.path,
                    evidence.text,
                    evidence.metric_name,
                    evidence.value,
                    evidence.created_at.isoformat(),
                ),
            )
            conn.commit()

    def add_result(self, task_id: str, result) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO results (task_id, benchmark, category, score, max_score, duration_ms, pass, details_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_id,
                    result.benchmark,
                    result.category.value,
                    result.score,
                    result.max_score,
                    result.duration_ms,
                    int(result.pass_),
                    json.dumps(result.details),
                ),
            )
            conn.commit()

    def get_results(self, task_id: str) -> list:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM results WHERE task_id = ?", (task_id,)).fetchall()
        results = []
        for row in rows:
            from .domain import BenchmarkResult
            results.append(
                BenchmarkResult(
                    task_id=row[0],
                    benchmark=row[1],
                    category=row[2],
                    score=row[3],
                    max_score=row[4],
                    duration_ms=row[5],
                    pass_=bool(row[6]),
                    details=json.loads(row[7]) if row[7] else {},
                )
            )
        return results

    def _row_to_task(self, row) -> Task:
        from .domain import TargetSpec
        from .enums import TargetKind

        task = Task(
            id=row[0],
            status=TaskStatus(row[1]),
            created_at=datetime.fromisoformat(row[2]),
            started_at=datetime.fromisoformat(row[3]) if row[3] else None,
            finished_at=datetime.fromisoformat(row[4]) if row[4] else None,
            target=None,
            deadline_s=row[8],
            retry_count=row[9],
            error=row[10],
            verdict=Verdict(row[11]) if row[11] else None,
        )
        if row[5]:
            task.target = TargetSpec(
                kind=TargetKind(row[5]),
                endpoint=row[6] or "",
                model=row[7],
            )
        if row[12]:
            plan_data = json.loads(row[12])
            from .domain import Plan
            task.plan = Plan(**{k: v for k, v in plan_data.items() if k in Plan.__dataclass_fields__})
        if row[13]:
            task.metrics = json.loads(row[13]) or {}
        return task

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
