"""Generate a detailed English Markdown report from alecto run artifacts.

This module is intentionally separate from :mod:`alecto.reporting`.  The latter
renders the domain :class:`~alecto.domain.Report` object; this one consumes the
*JSON artifacts* written by the standalone runners:

* ``quality-results.json``     -- written by ``the quality-suite runner``
* ``refusal-results.json``     -- written by ``the refusal-suite runner``
* ``performance-results.json`` -- optional performance/streaming artifact

A partial run is fully supported: any artifact may be missing and the report
marks the corresponding section as ``not run`` / ``unsupported`` instead of
inventing data.

The function :func:`generate_run_report` is the public entry point and can be
used both as an execution parameter of the runners (``--report-md PATH``) and
standalone (``python3 -m alecto.run_report --run-dir . --out report.md``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Canonical artifact file names discovered inside a run directory.
ARTIFACT_FILES: dict[str, str] = {
    "quality": "quality-results.json",
    "refusal": "refusal-results.json",
    "performance": "performance-results.json",
}

#: Human-readable labels for the refusal metric ids (spec §13.3).
_REFUSAL_METRICS: tuple[tuple[str, str], ...] = (
    ("alecto.refusal.full_rate", "full"),
    ("alecto.refusal.any_rate", "any"),
    ("alecto.refusal.partial_rate", "partial"),
    ("alecto.refusal.benign_overrefusal_rate", "benign_overrefusal"),
    ("alecto.refusal.benign_success_rate", "benign_success"),
    ("alecto.refusal.unsafe_actionable_rate", "unsafe_actionable"),
    ("alecto.refusal.evasive_rate", "evasive"),
    ("alecto.refusal.unknown_rate", "unknown"),
)

_QUALITY_COVERAGE_COLUMNS: tuple[str, ...] = (
    "selected",
    "attempted",
    "scored",
    "correct",
    "failed_generation",
    "failed_scoring",
    "truncated",
    "unsupported",
)

_PERF_METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "ttft": ("ttft_ms", "ttft", "alecto.perf.ttft_ms"),
    "ttfb": ("ttfb_ms", "ttfb", "alecto.perf.ttfb_ms"),
    "e2e": ("e2e_ms", "e2e", "latency_ms", "alecto.perf.e2e_ms"),
    "p50": ("p50_ms", "p50", "e2e_ms", "latency_ms"),
    "p90": ("p90_ms", "p90", "e2e_ms", "latency_ms"),
    "p95": ("p95_ms", "p95", "e2e_ms", "latency_ms"),
    "tps": (
        "output_tps",
        "tps",
        "decode_tps",
        "tokens_per_second",
        "tokens_per_s",
        "alecto.perf.output_tps",
    ),
}

# Patterns scrubbed from the final document so that credentials never leak.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"),
    re.compile(
        r"(?i)\b(api[_-]?key|authorization|access[_-]?token|token|secret)"
        r"\s*[:=]\s*\S+"
    ),
    # URL userinfo credentials (spec §6.1: never display them).
    re.compile(r"(?<=://)[^/@\s]+(?=@)"),
)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def _scrub(text: str) -> str:
    """Remove anything that looks like a credential from the rendered text."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("<redacted>", text)
    return text


def _number(value: Any, digits: int = 1, unit: str = "") -> str:
    """Format a number or return ``unsupported`` when it is missing.

    Non-finite values (``nan``, ``inf``) are rejected: a report must never
    render ``nan`` as if it were a measurement.
    """
    if value is None or isinstance(value, bool):
        return "unsupported"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unsupported"
    if not math.isfinite(number):
        return "unsupported"
    return f"{number:.{digits}f} {unit}".strip()


def _as_float(value: Any) -> float | None:
    """Coerce a value to float; None for non-numeric or non-finite input."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _floats(values: Any) -> list[float]:
    """Coerce an iterable to a list of floats, dropping non-numeric values."""
    out: list[float] = []
    for v in values:
        f = _as_float(v)
        if f is not None:
            out.append(f)
    return out


def _rate_cell(metric: Any) -> str:
    """Render a refusal metric as ``value [lower, upper]`` or ``unsupported``."""
    if not isinstance(metric, dict):
        return "unsupported"
    value = metric.get("value")
    if metric.get("applicable") is False or value is None:
        return "unsupported"
    try:
        rendered = f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "unsupported"
    lower = metric.get("lower")
    upper = metric.get("upper")
    try:
        if lower is not None and upper is not None:
            rendered += f" [{float(lower):.3f}, {float(upper):.3f}]"
    except (TypeError, ValueError):
        pass
    return rendered


def _metric_n(metric: Any) -> str:
    if not isinstance(metric, dict) or metric.get("n") is None:
        return "n/a"
    return str(metric.get("n"))


def _coverage_get(coverage: Any, key: str) -> int | float | None:
    """Return a numeric coverage field, or None when absent/non-numeric."""
    if isinstance(coverage, dict):
        value = coverage.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value
    return None


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def _resolve_paths(
    run_dir: str | Path, artifacts: Any
) -> dict[str, Any]:
    """Resolve the path of every known artifact.

    ``artifacts`` may be ``None`` (discover inside ``run_dir``), a mapping
    ``{"quality": path, ...}`` or an iterable of paths whose file name matches
    one of :data:`ARTIFACT_FILES`. A mapping value may also be an inline
    payload (``{"data": {...}}``) or an explicit entry (``{"path": ...}``);
    those are collected under the private ``"__inline__"`` key for
    :func:`_load_artifacts`. Values that are neither paths nor payloads are
    ignored instead of raising.
    """
    base = Path(run_dir)
    resolved: dict[str, Any] = {}
    inline: dict[str, Any] = {}

    def _as_path(value: Any) -> Path | None:
        if isinstance(value, (str, Path)):
            path = Path(value)
        else:
            try:
                path = Path(os.fspath(value))
            except TypeError:
                return None
        return path if path.is_absolute() else base / path

    if artifacts:
        if isinstance(artifacts, dict):
            for name, value in artifacts.items():
                if not value:
                    continue
                if isinstance(value, dict) and ("data" in value or "path" in value):
                    entry = dict(value)
                    entry.setdefault("error", None)
                    path = entry.get("path")
                    entry["path"] = str(path) if path else f"<inline:{name}>"
                    inline[str(name)] = entry
                    continue
                path = _as_path(value)
                if path is not None:
                    resolved[str(name)] = path
        else:
            for value in artifacts:
                if not value:
                    continue
                path = _as_path(value)
                if path is None:
                    continue
                for name, file_name in ARTIFACT_FILES.items():
                    if path.name == file_name:
                        resolved[name] = path

    for name, file_name in ARTIFACT_FILES.items():
        resolved.setdefault(name, base / file_name)
    if inline:
        resolved["__inline__"] = inline
    return resolved


def _load_artifacts(run_dir: str | Path, artifacts: Any) -> dict[str, dict[str, Any]]:
    """Load every artifact, tolerating absent or unreadable files."""
    resolved = _resolve_paths(run_dir, artifacts)
    inline: dict[str, Any] = resolved.pop("__inline__", {})

    found: dict[str, dict[str, Any]] = {}
    for name, path in resolved.items():
        entry: dict[str, Any] = {
            "path": str(path),
            "data": None,
            "error": None,
        }
        if path.is_file():
            try:
                with open(path, encoding="utf-8") as handle:
                    entry["data"] = json.load(handle)
            except (OSError, ValueError) as exc:  # malformed JSON is tolerated
                entry["error"] = f"{type(exc).__name__}: {exc}"
        else:
            entry["error"] = "file not found"
        found[name] = entry

    for name, entry in inline.items():
        if name in ARTIFACT_FILES:
            found[name] = {
                "path": str(entry.get("path") or f"<inline:{name}>"),
                "data": entry.get("data"),
                "error": entry.get("error"),
            }
    return found


def _artifact_meta(entry: dict[str, Any]) -> dict[str, Any]:
    data = entry.get("data")
    if isinstance(data, dict):
        return data
    return {}


def _endpoint_from(quality: Any, refusal: Any, perf: Any) -> str | None:
    if isinstance(quality, dict) and quality.get("endpoint"):
        return str(quality["endpoint"])
    if isinstance(refusal, dict):
        target = refusal.get("evaluated_target")
        if isinstance(target, dict) and target.get("endpoint"):
            return str(target["endpoint"])
    if isinstance(perf, dict) and perf.get("endpoint"):
        return str(perf["endpoint"])
    return None


def _model_from(quality: Any, refusal: Any, perf: Any) -> str | None:
    if isinstance(quality, dict) and quality.get("model"):
        return str(quality["model"])
    if isinstance(refusal, dict):
        target = refusal.get("evaluated_target")
        if isinstance(target, dict) and target.get("model"):
            return str(target["model"])
    if isinstance(perf, dict) and perf.get("model"):
        return str(perf["model"])
    return None


def _artifact_generated_at(entry: dict[str, Any]) -> str:
    data = entry.get("data")
    if isinstance(data, dict) and data.get("generated_at"):
        return str(data["generated_at"])
    return "unknown"


def _artifact_schema(entry: dict[str, Any]) -> str:
    data = entry.get("data")
    if isinstance(data, dict) and data.get("schema"):
        return str(data["schema"])
    return "unknown"


# ---------------------------------------------------------------------------
# Performance normalisation
# ---------------------------------------------------------------------------


def _perf_cells(data: Any) -> list[dict[str, Any]]:
    """Extract a list of per-cell mappings from a performance artifact."""
    if isinstance(data, list):
        return [c for c in data if isinstance(c, dict)]
    if not isinstance(data, dict):
        return []

    for key in ("cells", "results", "performance"):
        value = data.get(key)
        if isinstance(value, list):
            return [c for c in value if isinstance(c, dict)]

    # ``speed-stream-results.json`` style: levels -> aggregate.
    levels = data.get("levels")
    if isinstance(levels, dict):
        cells: list[dict[str, Any]] = []
        for level, level_data in levels.items():
            aggregate = (
                level_data.get("aggregate", {})
                if isinstance(level_data, dict)
                else {}
            )
            cell = dict(aggregate) if isinstance(aggregate, dict) else {}
            cell.setdefault("cell_id", f"level-{level}")
            cell.setdefault("concurrency", cell.get("concurrency", level))
            cells.append(cell)
        return cells
    return []


def _perf_scalar(combined: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = combined.get(key)
        if isinstance(value, dict):
            value = value.get("value", value.get("mean"))
        if value is not None:
            return value
    return None


def _perf_percentile(
    combined: dict[str, Any], pkey: str, keys: tuple[str, ...]
) -> Any:
    for key in keys:
        value = combined.get(key)
        if isinstance(value, dict) and value.get(pkey) is not None:
            return value[pkey]
    for key in keys:
        if key == pkey:
            value = combined.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
    value = combined.get(f"{pkey}_ms")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _perf_row(cell: dict[str, Any]) -> dict[str, Any]:
    combined = dict(cell)
    metrics = combined.get("metrics")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            combined.setdefault(key, value)

    return {
        "cell_id": str(
            combined.get("cell_id")
            or combined.get("id")
            or combined.get("name")
            or "cell"
        ),
        "mode": str(combined.get("mode") or combined.get("loop_mode") or "-"),
        "workload": str(combined.get("workload") or "-"),
        "concurrency": combined.get("concurrency"),
        "ttft": _perf_scalar(combined, _PERF_METRIC_KEYS["ttft"]),
        "ttfb": _perf_scalar(combined, _PERF_METRIC_KEYS["ttfb"]),
        "e2e": _perf_scalar(combined, _PERF_METRIC_KEYS["e2e"]),
        "p50": _perf_percentile(combined, "p50", _PERF_METRIC_KEYS["p50"]),
        "p90": _perf_percentile(combined, "p90", _PERF_METRIC_KEYS["p90"]),
        "p95": _perf_percentile(combined, "p95", _PERF_METRIC_KEYS["p95"]),
        "tps": _perf_scalar(combined, _PERF_METRIC_KEYS["tps"]),
    }


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _section_metadata(
    endpoint: str | None,
    model: str | None,
    generated_at: str,
    mode: str,
    found: dict[str, dict[str, Any]],
) -> list[str]:
    lines = ["## Run Metadata", ""]
    lines.append(f"- **Endpoint:** `{endpoint or 'unknown'}`")
    lines.append(f"- **Model:** `{model or 'unknown'}`")
    lines.append(f"- **Generated at (UTC):** {generated_at}")
    lines.append(f"- **Run mode:** {mode}")
    present = [name for name, entry in found.items() if entry.get("data") is not None]
    lines.append(
        "- **Artifacts present:** "
        + (", ".join(present) if present else "none")
    )
    lines.append("")
    lines.append("### Artifact inventory")
    lines.append("")
    rows = []
    for name in ("quality", "refusal", "performance"):
        entry = found.get(name, {})
        status = "present" if entry.get("data") is not None else "missing"
        if entry.get("data") is None and entry.get("error") not in (None, "file not found"):
            status = f"unreadable ({entry['error']})"
        rows.append(
            [
                name,
                status,
                f"`{entry.get('path', '-')}`",
                _artifact_schema(entry),
                _artifact_generated_at(entry) if entry.get("data") is not None else "-",
            ]
        )
    lines.extend(
        _md_table(
            ["Artifact", "Status", "Path", "Schema", "Artifact generated_at"],
            rows,
        )
    )
    lines.append("")
    return lines


def _section_executive(
    quality: Any,
    refusal: Any,
    perf_cells: list[dict[str, Any]],
    endpoint: str | None,
) -> list[str]:
    lines = ["## Executive Summary", ""]
    lines.append(
        f"Execution report for the endpoint under test `{endpoint or 'unknown'}`. "
        "All values below are read verbatim from the run artifacts; sections whose "
        "artifact was not produced are explicitly marked as not run."
    )
    lines.append("")

    # Quality headline.
    if isinstance(quality, dict) and isinstance(quality.get("results"), list):
        results = quality["results"]
        scores = []
        for r in results:
            if isinstance(r, dict) and r.get("score") is not None:
                score = _as_float(r["score"])
                if score is not None:
                    scores.append(score)
        mean_score = sum(scores) / len(scores) if scores else None
        selected = sum(
            _coverage_get(r.get("coverage"), "selected") or 0
            for r in results
            if isinstance(r, dict)
        )
        scored = sum(
            _coverage_get(r.get("coverage"), "scored") or 0
            for r in results
            if isinstance(r, dict)
        )
        lines.append(
            f"- **Academic quality:** {len(results)} suites, mean score "
            f"{_number(mean_score, 3)} ({scored}/{selected} items scored)."
        )
    else:
        lines.append("- **Academic quality:** not run.")

    # Refusal headline.
    if isinstance(refusal, dict) and isinstance(refusal.get("overall"), dict):
        overall = refusal["overall"]
        coverage = overall.get("coverage", {})
        full = overall.get("alecto.refusal.full_rate", {})
        any_rate = overall.get("alecto.refusal.any_rate", {})
        unsafe = overall.get("alecto.refusal.unsafe_actionable_rate", {})
        lines.append(
            f"- **Refusal behaviour:** n={_coverage_get(coverage, 'classified') or 0}, "
            f"full_rate={_rate_cell(full)}, any_rate={_rate_cell(any_rate)}, "
            f"unsafe_actionable={_rate_cell(unsafe)}."
        )
    else:
        lines.append("- **Refusal behaviour:** not run.")

    # Performance headline.
    if perf_cells:
        e2e_values = _floats(c.get("e2e") for c in perf_cells)
        tps_values = _floats(c.get("tps") for c in perf_cells)
        mean_e2e = sum(e2e_values) / len(e2e_values) if e2e_values else None
        mean_tps = sum(tps_values) / len(tps_values) if tps_values else None
        lines.append(
            f"- **Performance:** {len(perf_cells)} cell(s); mean E2E "
            f"{_number(mean_e2e, 1, 'ms')}; mean output throughput "
            f"{_number(mean_tps, 2, 'tokens/s')}."
        )
    else:
        lines.append("- **Performance:** not run.")

    lines.append("")
    return lines


def _section_performance(perf_cells: list[dict[str, Any]]) -> list[str]:
    lines = ["## 1. Performance", ""]
    if not perf_cells:
        lines.append(
            "Not run: no `performance-results.json` artifact was found. "
            "Latency and throughput metrics are **unsupported** for this run."
        )
        lines.append("")
        return lines

    lines.append(
        "Per-cell timing and throughput. Null values are rendered as "
        "`unsupported`. TTFT/TTFB/E2E/p50/p90/p95 are in milliseconds (ms); "
        "throughput is in output tokens per second (tokens/s)."
    )
    lines.append("")
    headers = [
        "Cell",
        "Mode",
        "Workload",
        "Concurrency",
        "TTFT (ms)",
        "TTFB (ms)",
        "E2E (ms)",
        "p50 (ms)",
        "p90 (ms)",
        "p95 (ms)",
        "Output (tokens/s)",
    ]
    rows = []
    for cell in perf_cells:
        rows.append(
            [
                str(cell["cell_id"]),
                str(cell["mode"]),
                str(cell["workload"]),
                "unsupported" if cell["concurrency"] is None else str(cell["concurrency"]),
                _number(cell["ttft"], 1, "ms"),
                _number(cell["ttfb"], 1, "ms"),
                _number(cell["e2e"], 1, "ms"),
                _number(cell["p50"], 1, "ms"),
                _number(cell["p90"], 1, "ms"),
                _number(cell["p95"], 1, "ms"),
                _number(cell["tps"], 2, "tokens/s"),
            ]
        )
    lines.extend(_md_table(headers, rows))
    lines.append("")

    # TTFB vs TTFT note (spec: no streaming adapter -> TTFB == TTFT == E2E).
    equal_cells = [
        c
        for c in perf_cells
        if _as_float(c.get("ttfb")) is not None
        and _as_float(c.get("ttft")) is not None
        and abs(_as_float(c["ttfb"]) - _as_float(c["ttft"])) < 1e-6
    ]
    if equal_cells:
        lines.append(
            f"- Note: TTFB equals TTFT in {len(equal_cells)}/{len(perf_cells)} "
            "cell(s), which is consistent with an adapter that does not expose "
            "streaming first-byte timing."
        )
        lines.append("")
    return lines


def _section_quality(quality: Any) -> list[str]:
    lines = ["## 2. Academic Quality", ""]
    if not isinstance(quality, dict) or not isinstance(quality.get("results"), list):
        lines.append(
            "Not run: no `quality-results.json` artifact was found. Academic "
            "scores are **unsupported** for this run."
        )
        lines.append("")
        return lines

    fixtures = quality.get("fixtures") if isinstance(quality.get("fixtures"), dict) else {}
    synthetic = fixtures.get("synthetic")
    # A run is leaderboard comparable when the artifact says so at the top
    # level (e.g. produced by alecto.lmeval_adapter) or when every suite
    # declares itself comparable. Otherwise the synthetic caveat applies.
    suites_list = [r for r in quality["results"] if isinstance(r, dict)]
    per_suite_comparable = bool(suites_list) and all(
        r.get("comparable") is True for r in suites_list
    )
    comparable = quality.get("comparable") is True or per_suite_comparable
    if comparable:
        harness = quality.get("harness") if isinstance(quality.get("harness"), dict) else {}
        src = harness.get("name") or quality.get("source") or "official harness"
        lines.append(
            f"Source: `{src}` — official datasets, protocols and upstream "
            "scorers; results are **leaderboard comparable** (subject to the "
            "sample counts reported below)."
        )
    else:
        lines.append(
            f"Fixtures: `{fixtures.get('path', 'unknown')}` "
            f"(synthetic={synthetic if synthetic is not None else 'unknown'}). "
            "Synthetic fixtures are not official benchmark subsets and are not "
            "leaderboard comparable."
        )
    lines.append("")
    headers = [
        "Suite",
        "Protocol ID",
        "Comparable",
        "Score",
        "n (selected)",
        *[c.replace("_", " ") for c in _QUALITY_COVERAGE_COLUMNS],
    ]
    rows = []
    for result in quality["results"]:
        if not isinstance(result, dict):
            continue
        coverage = result.get("coverage", {})
        protocol = result.get("protocol_id") or result.get("metric_id") or result.get("protocol") or "-"
        suite_comparable = result.get("comparable")
        if suite_comparable is None:
            suite_comparable = not (synthetic if synthetic is not None else False)
        rows.append(
            [
                str(result.get("suite", "-")),
                str(protocol),
                "yes" if suite_comparable else "no",
                _number(result.get("score"), 3),
                str(_coverage_get(coverage, "selected")),
                *[str(_coverage_get(coverage, col)) for col in _QUALITY_COVERAGE_COLUMNS],
            ]
        )
    lines.extend(_md_table(headers, rows))
    lines.append("")
    return lines


def _refusal_population_block(name: str, metrics: dict[str, Any]) -> list[str]:
    lines = [f"### {name}", ""]
    coverage = metrics.get("coverage", {}) if isinstance(metrics, dict) else {}
    lines.append(
        "Coverage: "
        + ", ".join(
            f"{key}={_coverage_get(coverage, key)}"
            for key in (
                "selected",
                "attempted",
                "completed",
                "valid",
                "invalid",
                "classified",
                "unknown",
                "uncertain",
            )
        )
        + "."
    )
    lines.append("")
    rows = []
    for metric_id, label in _REFUSAL_METRICS:
        metric = metrics.get(metric_id, {}) if isinstance(metrics, dict) else {}
        rows.append(
            [
                label,
                f"`{metric_id}`",
                _rate_cell(metric),
                _metric_n(metric),
                "yes" if (isinstance(metric, dict) and metric.get("applicable") is False) else "no",
            ]
        )
    lines.extend(
        _md_table(
            ["Metric", "Canonical ID", "Value [Wilson 95% CI]", "n", "Unsupported"],
            rows,
        )
    )
    lines.append("")
    return lines


def _section_refusal(refusal: Any) -> list[str]:
    lines = ["## 3. Refusal Behaviour", ""]
    if not isinstance(refusal, dict) or not isinstance(refusal.get("populations"), dict):
        lines.append(
            "Not run: no `refusal-results.json` artifact was found. Refusal "
            "rates are **unsupported** for this run."
        )
        lines.append("")
        return lines

    judge = refusal.get("judge") if isinstance(refusal.get("judge"), dict) else {}
    lines.append(
        "Populations are never pooled into a single refusal number. Intervals "
        "are Wilson 95% confidence intervals over the classified items. "
        f"Judge target: `{judge.get('target_id', 'unknown')}` "
        f"(label_source={judge.get('label_source', 'unknown')}, "
        f"self_judge={judge.get('self_judge', 'unknown')})."
    )
    lines.append("")

    for name, metrics in refusal["populations"].items():
        if isinstance(metrics, dict):
            lines.extend(_refusal_population_block(name, metrics))

    overall = refusal.get("overall")
    if isinstance(overall, dict):
        headline = dict(overall)
        headline.setdefault("population", "overall")
        lines.extend(_refusal_population_block("overall", headline))
    return lines


def _section_coverage(quality: Any, refusal: Any, perf_cells: list[dict[str, Any]]) -> list[str]:
    lines = ["## 4. Coverage and Missingness", ""]

    # Quality coverage.
    lines.append("### Academic quality")
    lines.append("")
    if isinstance(quality, dict) and isinstance(quality.get("results"), list):
        headers = ["Suite", *[c.replace("_", " ") for c in _QUALITY_COVERAGE_COLUMNS]]
        rows = []
        for result in quality["results"]:
            if not isinstance(result, dict):
                continue
            coverage = result.get("coverage", {})
            rows.append(
                [str(result.get("suite", "-"))]
                + [str(_coverage_get(coverage, col)) for col in _QUALITY_COVERAGE_COLUMNS]
            )
        lines.extend(_md_table(headers, rows))
    else:
        lines.append("not run")
    lines.append("")

    # Refusal coverage.
    lines.append("### Refusal behaviour")
    lines.append("")
    if isinstance(refusal, dict) and isinstance(refusal.get("populations"), dict):
        headers = [
            "Population",
            "selected",
            "attempted",
            "completed",
            "valid",
            "invalid",
            "unknown",
        ]
        rows = []
        for name, metrics in refusal["populations"].items():
            coverage = metrics.get("coverage", {}) if isinstance(metrics, dict) else {}
            rows.append(
                [str(name)]
                + [str(_coverage_get(coverage, col)) for col in headers[1:]]
            )
        overall = refusal.get("overall")
        if isinstance(overall, dict):
            coverage = overall.get("coverage", {})
            rows.append(
                ["overall"]
                + [str(_coverage_get(coverage, col)) for col in headers[1:]]
            )
        lines.extend(_md_table(headers, rows))
    else:
        lines.append("not run")
    lines.append("")

    # Performance coverage.
    lines.append("### Performance")
    lines.append("")
    if perf_cells:
        total = len(perf_cells)
        supported = sum(
            1
            for cell in perf_cells
            if cell.get("e2e") is not None and cell.get("ttft") is not None
        )
        unsupported = total - supported
        lines.append(
            f"{total} cell(s); {supported} with supported E2E/TTFT metrics; "
            f"{unsupported} with unsupported (null) metrics."
        )
    else:
        lines.append("not run")
    lines.append("")
    return lines


def _section_limitations(
    quality: Any, refusal: Any, perf_cells: list[dict[str, Any]]
) -> list[str]:
    lines = ["## 5. Limitations", ""]
    bullets: list[str] = []

    if isinstance(quality, dict):
        fixtures = quality.get("fixtures") if isinstance(quality.get("fixtures"), dict) else {}
        if fixtures.get("synthetic"):
            bullets.append(
                "Academic quality uses **synthetic fixtures** authored by the "
                "Alecto project, not the official MMLU-Pro/GSM8K/HumanEval/IFEval "
                "subsets; the scores are smoke/regression signals only and are not "
                "leaderboard comparable."
            )
        truncated = 0
        truncated_suites: list[str] = []
        for result in quality.get("results", []) if isinstance(quality.get("results"), list) else []:
            cov = result.get("coverage", {}) if isinstance(result, dict) else {}
            count = _coverage_get(cov, "truncated") or 0
            truncated += int(count)
            if count:
                truncated_suites.append(f"{result.get('suite', '?')} ({count})")
        if truncated:
            bullets.append(
                f"Reasoning-model truncations: {truncated} quality item(s) hit the "
                "output-token budget (`finish_reason=length`) and were not scored on "
                "content" + (f": {', '.join(truncated_suites)}." if truncated_suites else ".")
            )
        sandbox = quality.get("humaneval_sandbox_mode")
        if sandbox and "unsupported" in str(sandbox):
            bullets.append(
                "HumanEval execution sandbox was **unsupported** in this run; coding "
                "items that require execution were not scored."
            )
    else:
        bullets.append("Academic quality was not run in this execution.")

    if isinstance(refusal, dict):
        judge = refusal.get("judge") if isinstance(refusal.get("judge"), dict) else {}
        if judge.get("self_judge"):
            bullets.append(
                "Refusal labels come from a **self_judge** configuration (the "
                "evaluated endpoint/model also acted as judge). Labels are not "
                "independent and may be biased."
            )
        overall_cov = (
            refusal.get("overall", {}).get("coverage", {})
            if isinstance(refusal.get("overall"), dict)
            else {}
        )
        invalid = _coverage_get(overall_cov, "invalid")
        if invalid:
            bullets.append(
                f"Unreachable/unclassifiable refusal items: {invalid} prompt(s) "
                "yielded no valid response (transport error, timeout or empty "
                "output) and are excluded from the classified denominators."
            )
        records = refusal.get("records")
        if isinstance(records, list):
            length_hits = sum(
                1
                for record in records
                if isinstance(record, dict) and record.get("finish_reason") == "length"
            )
            if length_hits:
                bullets.append(
                    f"Reasoning-model truncations: {length_hits} refusal response(s) "
                    "reached `finish_reason=length`."
                )
        catalog = refusal.get("catalog") if isinstance(refusal.get("catalog"), dict) else {}
        if catalog.get("note"):
            bullets.append(
                f"Refusal catalogue note: {catalog['note']}."
            )
    else:
        bullets.append("Refusal behaviour was not run in this execution.")

    if perf_cells:
        equal = sum(
            1
            for cell in perf_cells
            if _as_float(cell.get("ttfb")) is not None
            and _as_float(cell.get("ttft")) is not None
            and abs(_as_float(cell["ttfb"]) - _as_float(cell["ttft"])) < 1e-6
        )
        if equal:
            bullets.append(
                f"TTFB equals TTFT in {equal}/{len(perf_cells)} performance cell(s): "
                "the adapter does not expose a separate first-byte timestamp, so "
                "TTFB must not be interpreted as network streaming latency."
            )
        null_cells = sum(1 for cell in perf_cells if cell.get("e2e") is None)
        if null_cells:
            bullets.append(
                f"{null_cells} performance cell(s) have null E2E/TTFT values and are "
                "reported as unsupported."
            )
    else:
        bullets.append("Performance was not run in this execution.")

    for bullet in bullets:
        lines.append(f"- {bullet}")
    lines.append("")
    return lines


def _section_conclusion(
    quality: Any, refusal: Any, perf_cells: list[dict[str, Any]]
) -> list[str]:
    lines = ["## 6. Conclusion", ""]
    parts: list[str] = []
    if isinstance(quality, dict) and isinstance(quality.get("results"), list):
        scores = []
        for r in quality["results"]:
            if isinstance(r, dict) and r.get("score") is not None:
                score = _as_float(r["score"])
                if score is not None:
                    scores.append(score)
        mean_score = sum(scores) / len(scores) if scores else None
        parts.append(
            f"academic quality was evaluated over {len(quality['results'])} "
            f"synthetic suite(s) (mean score {_number(mean_score, 3)})"
        )
    else:
        parts.append("academic quality was not run")

    if isinstance(refusal, dict) and isinstance(refusal.get("overall"), dict):
        overall = refusal["overall"]
        parts.append(
            "refusal behaviour was measured per population with Wilson 95% "
            f"intervals (overall fully-refused rate {_rate_cell(overall.get('alecto.refusal.full_rate', {}))})"
        )
    else:
        parts.append("refusal behaviour was not run")

    if perf_cells:
        parts.append(f"performance covered {len(perf_cells)} cell(s)")
    else:
        parts.append("performance was not run")

    lines.append("This report summarises: " + "; ".join(parts) + ".")
    lines.append(
        "Every number is traceable to the artifact paths in Appendix A. Missing "
        "artifacts are reported as `not run` and must not be interpreted as "
        "passing results."
    )
    lines.append("")
    return lines


def _section_appendix(
    found: dict[str, dict[str, Any]],
    endpoint: str | None,
    model: str | None,
) -> list[str]:
    lines = ["## Appendix A. Artifact Paths", ""]
    rows = []
    for name in ("quality", "refusal", "performance"):
        entry = found.get(name, {})
        rows.append(
            [
                name,
                "present" if entry.get("data") is not None else "missing",
                f"`{entry.get('path', '-')}`",
                _artifact_schema(entry),
            ]
        )
    lines.extend(_md_table(["Artifact", "Status", "Absolute path", "Schema"], rows))
    lines.append("")
    lines.append("## Appendix B. Reproduction Commands")
    lines.append("")
    lines.append(
        "The commands below are reconstructed from the artifact metadata "
        "(endpoint and model). Actual runs may use additional flags."
    )
    lines.append("")
    lines.append("```bash")
    quality_entry = found.get("quality", {})
    if quality_entry.get("data") is not None:
        lines.append(
            f"python3 the quality-suite runner --endpoint {endpoint or '<ENDPOINT>'} "
            f"--model {model or '<MODEL>'} --out {quality_entry.get('path')}"
        )
    refusal_entry = found.get("refusal", {})
    if refusal_entry.get("data") is not None:
        lines.append(
            f"python3 the refusal-suite runner --endpoint {endpoint or '<ENDPOINT>'} "
            f"--model {model or '<MODEL>'} --out {refusal_entry.get('path')}"
        )
    perf_entry = found.get("performance", {})
    if perf_entry.get("data") is not None:
        lines.append(f"# performance artifact: {perf_entry.get('path')}")
    if not any(
        found.get(name, {}).get("data") is not None
        for name in ("quality", "refusal", "performance")
    ):
        lines.append("# no artifacts were available; nothing was run")
    lines.append(
        "python3 -m alecto.run_report --run-dir . --out run-report.md"
    )
    lines.append("```")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_run_report(
    *,
    run_dir: str | Path = ".",
    endpoint: str | None = None,
    model: str | None = None,
    out: str | Path | None = None,
    artifacts: Any = None,
    generated_at: str | None = None,
) -> str:
    """Build a fully detailed English Markdown report from run artifacts.

    Parameters
    ----------
    run_dir:
        Directory that contains the run artifacts.
    endpoint / model:
        Overrides for the values discovered in the artifacts.
    out:
        Optional path; when given the Markdown is written there.
    artifacts:
        Optional override: a mapping ``{"quality": path, ...}`` or an iterable
        of artifact paths.  Missing entries are simply treated as not run.
    generated_at:
        Optional ISO-8601 timestamp injected into the report; useful for
        deterministic output.  Defaults to the current UTC time.

    Returns
    -------
    str
        The rendered Markdown document.
    """
    found = _load_artifacts(run_dir, artifacts)
    quality_entry = found.get("quality", {})
    refusal_entry = found.get("refusal", {})
    performance_entry = found.get("performance", {})

    quality = quality_entry.get("data")
    refusal = refusal_entry.get("data")
    performance = performance_entry.get("data")

    resolved_endpoint = endpoint or _endpoint_from(quality, refusal, performance)
    resolved_model = model or _model_from(quality, refusal, performance)
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()

    present = {
        name
        for name, entry in found.items()
        if entry.get("data") is not None
    }
    if present == {"quality", "refusal", "performance"}:
        mode = "full"
    elif present:
        mode = "partial"
    else:
        mode = "partial (no artifacts)"

    perf_cells = [_perf_row(cell) for cell in _perf_cells(performance)]

    lines: list[str] = []
    lines.append(f"# Alecto Run Report — {resolved_endpoint or 'unknown endpoint'}")
    lines.append("")
    lines.extend(_section_metadata(resolved_endpoint, resolved_model, timestamp, mode, found))
    lines.extend(_section_executive(quality, refusal, perf_cells, resolved_endpoint))
    lines.extend(_section_performance(perf_cells))
    lines.extend(_section_quality(quality))
    lines.extend(_section_refusal(refusal))
    lines.extend(_section_coverage(quality, refusal, perf_cells))
    lines.extend(_section_limitations(quality, refusal, perf_cells))
    lines.extend(_section_conclusion(quality, refusal, perf_cells))
    lines.extend(_section_appendix(found, resolved_endpoint, resolved_model))

    document = "\n".join(lines).rstrip() + "\n"
    document = _scrub(document)

    if out is not None:
        out_path = Path(out)
        if out_path.parent and str(out_path.parent) != ".":
            out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(document)
    return document


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an English Markdown report from alecto run artifacts."
    )
    parser.add_argument("--run-dir", default=".",
                        help="directory containing the run artifacts")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default="run-report.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    text = generate_run_report(
        run_dir=args.run_dir,
        endpoint=args.endpoint,
        model=args.model,
        out=args.out,
    )
    print(f"Wrote {args.out} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
