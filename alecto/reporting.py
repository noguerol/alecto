"""Report generation (spec §4.11)."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .domain import BenchmarkResult, ComparisonResult, Report, Task


def generate_report(task: Task, results: list[BenchmarkResult], comparison: ComparisonResult | None = None) -> Report:
    """Generate a structured report for a completed task."""
    sections = []

    # Summary section
    sections.append({
        "title": "Summary",
        "content": {
            "task_id": task.id,
            "status": task.status.value,
            "verdict": task.verdict.value if task.verdict else None,
            "duration_ms": task.metrics.get("duration_ms"),
        },
    })

    # Results section
    if results:
        sections.append({
            "title": "Benchmark Results",
            "content": [
                {
                    "benchmark": r.benchmark,
                    "score": r.score,
                    "max_score": r.max_score,
                    "pass": r.pass_,
                    "duration_ms": r.duration_ms,
                }
                for r in results
            ],
        })

    # Comparison section
    if comparison:
        sections.append({
            "title": "Comparison",
            "content": {
                "mode": comparison.mode.value,
                "winner": comparison.winner,
                "margin": comparison.margin,
                "notes": comparison.notes,
            },
        })

    # Metrics section
    if task.metrics:
        sections.append({
            "title": "Metrics",
            "content": task.metrics,
        })

    # Evidence section
    if task.evidence:
        sections.append({
            "title": "Evidence",
            "content": [
                {
                    "id": e.id,
                    "kind": e.kind.value,
                    "text": e.text,
                    "path": e.path,
                }
                for e in task.evidence
            ],
        })

    return Report(
        task_id=task.id,
        title=f"Alecto Report: {task.id}",
        sections=sections,
        verdict=task.verdict,
        created_at=datetime.now(timezone.utc),
    )


def save_report(report: Report, output_dir: Path) -> Path:
    """Save a report to disk as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.task_id}.json"
    data = {
        "task_id": report.task_id,
        "title": report.title,
        "verdict": report.verdict.value if report.verdict else None,
        "sections": report.sections,
        "created_at": report.created_at.isoformat(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def format_report(report: Report) -> str:
    """Format a report as readable text."""
    lines = [f"# {report.title}", ""]
    for section in report.sections:
        lines.append(f"## {section['title']}")
        content = section["content"]
        if isinstance(content, dict):
            lines.append(json.dumps(content, indent=2))
        elif isinstance(content, list):
            for item in content:
                lines.append(f"- {json.dumps(item)}")
        else:
            lines.append(str(content))
        lines.append("")
    return "\n".join(lines)
