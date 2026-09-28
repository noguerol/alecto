"""Convert ``lm-evaluation-harness`` artifacts into alecto quality artifacts.

This is the leaderboard-comparable path required by the system spec §10: the
official datasets, protocols and upstream scorers run inside ``lm-eval`` and
this module maps their output onto alecto's ``alecto.quality_results.v1``
schema so the run report consumes them unchanged.

The conversion is deliberately lossless about provenance: every suite records
the harness version, the task version, the exact filter and metric id, the
n-shot and the effective sample count, so a reader can tell a leaderboard
number from a synthetic one at a glance.

Typical use::

    python3 -m alecto.lmeval_adapter OUT_DIR --endpoint http://localhost:8081/v1 \
        --model local-model --out quality-lmeval.json

The input is a directory produced by::

    lm_eval --model local-chat-completions \
        --model_args base_url=...,model=... \
        --tasks gsm8k --log_samples --output_path OUT_DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone
from typing import Any

SCHEMA = "alecto.quality_results.v1"
SOURCE = "lm-evaluation-harness"

#: metric id -> (canonical alecto metric, leaderboard-comparable?)
#:
#: Only these are treated as headline scores; anything else in the harness
#: output is preserved under ``extra_metrics`` but never drives the score.
HEADLINE_METRICS: dict[str, str] = {
    "exact_match": "exact_match",
    "pass@1": "pass@1",
    "prompt_level_strict_acc": "prompt_level_strict_acc",
    "inst_level_strict_acc": "inst_level_strict_acc",
}


def _latest(paths: list[str]) -> list[str]:
    return sorted(paths)


def load_results(run_dir: str) -> dict[str, Any]:
    """Load the newest ``results_*.json`` under ``run_dir`` (recursively)."""
    paths = _latest(glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no lm-eval results_*.json under {run_dir!r}")
    with open(paths[-1], encoding="utf-8") as handle:
        return json.load(handle)


def load_samples(run_dir: str) -> dict[str, list[dict[str, Any]]]:
    """Load per-item samples grouped by task name."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in _latest(glob.glob(os.path.join(run_dir, "**", "samples_*.jsonl"), recursive=True)):
        name = os.path.basename(path).split("_", 2)
        task = name[1] if len(name) > 2 else os.path.basename(path)
        rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
        if rows:
            grouped[task] = rows
    return grouped


def _split_metric(key: str) -> tuple[str, str]:
    """``"exact_match,strict-match"`` -> ``("exact_match", "strict-match")``."""
    metric, _, filt = key.partition(",")
    return metric, filt


def _headline(task: str, metrics: dict[str, Any]) -> tuple[str, float, float, str, str] | None:
    """Pick the headline (metric, value, stderr, filter, metric_id).

    Preference order follows :data:`HEADLINE_METRICS`; among ties the
    ``flexible-extract`` filter wins over ``strict-match`` because it is the
    convention reported for GSM8K, then the alphabetically first filter so the
    choice is deterministic.
    """
    candidates: list[tuple[int, int, str, float, float, str]] = []
    for key, value in metrics.items():
        metric, filt = _split_metric(key)
        if metric not in HEADLINE_METRICS or not isinstance(value, (int, float)):
            continue
        if "stderr" in metric or metric.endswith("_err"):
            continue
        pref = list(HEADLINE_METRICS).index(metric)
        filt_pref = 0 if filt == "flexible-extract" else 1
        candidates.append((pref, filt_pref, filt, float(value), 0.0, metric))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    pref, _, filt, value, _, metric = candidates[0]
    stderr_key = f"{metric}_stderr,{filt}" if filt else f"{metric}_stderr"
    stderr = metrics.get(stderr_key)
    stderr = float(stderr) if isinstance(stderr, (int, float)) else None
    metric_id = f"{metric},{filt}" if filt else metric
    return metric, value, stderr, filt, metric_id


def convert(
    results: dict[str, Any],
    samples: dict[str, list[dict[str, Any]]],
    *,
    endpoint: str | None = None,
    model: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build an alecto quality artifact from lm-eval ``results`` + ``samples``."""
    harness_version = None
    config = results.get("config")
    if isinstance(config, dict):
        harness_version = config.get("model") or config.get("model_url")
    task_versions = results.get("versions") if isinstance(results.get("versions"), dict) else {}
    n_samples = results.get("n-samples") if isinstance(results.get("n-samples"), dict) else {}

    suites: list[dict[str, Any]] = []
    for task, metrics in (results.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        picked = _headline(task, metrics)
        rows = samples.get(task, [])
        if picked is None:
            suites.append(
                {
                    "suite": task,
                    "protocol": "lm-eval",
                    "comparable": True,
                    "score": None,
                    "correct": None,
                    "total": len(rows),
                    "error": "no headline metric in harness output",
                    "items": [],
                }
            )
            continue
        metric, value, stderr, filt, metric_id = picked
        # Re-running lm-eval into the same --output_path appends samples, so the
        # JSONL can contain the same item id more than once. Counting those twice
        # would inflate n and make the score look more reliable than it is, so
        # items are deduplicated by id (first occurrence wins) and the duplicate
        # count is reported instead of hidden.
        unique_rows: dict[Any, dict[str, Any]] = {}
        duplicates = 0
        for row in rows:
            key = row.get("id", row.get("doc_id", row.get("doc")))
            if key in unique_rows:
                duplicates += 1
                continue
            unique_rows[key] = row
        rows = list(unique_rows.values())
        correct = sum(1 for r in rows if r.get(metric) in (1, 1.0, True))
        total = len(rows)
        nsamp = n_samples.get(task) if isinstance(n_samples, dict) else None
        # alecto coverage columns; lm-eval scores every attempted item, so
        # selected == attempted == scored unless the harness skipped some.
        coverage = {
            "selected": total,
            "attempted": total,
            "scored": total,
            "completed": total,
            "valid": total,
            "invalid": 0,
            "classified": total,
            "correct": correct,
            "failed_generation": 0,
            "failed_scoring": 0,
            "truncated": 0,
            "unsupported": 0,
            "duplicate_rows": duplicates,
        }
        suites.append(
            {
                "suite": task,
                "protocol": "lm-eval",
                "comparable": True,
                "metric": metric,
                "metric_id": metric_id,
                "filter": filt,
                "score": value,
                "stderr": stderr,
                "correct": correct if rows else None,
                "total": total if rows else None,
                "n_shot": metrics.get("n-shot"),
                "task_version": task_versions.get(task),
                "n_samples_original": (nsamp or {}).get("original") if isinstance(nsamp, dict) else None,
                "n_samples_effective": (nsamp or {}).get("effective") if isinstance(nsamp, dict) else None,
                "coverage": coverage,
                "extra_metrics": {
                    k: v for k, v in metrics.items() if isinstance(v, (int, float)) and k != metric_id
                },
                "items": [
                    {
                        "id": r.get("doc_id"),
                        "ok": bool(r.get(metric)) if metric in r else None,
                        "target": r.get("target"),
                        "filtered": (r.get("filtered_resps") or [None])[0],
                    }
                    for r in rows
                ],
            }
        )

    scored = [s for s in suites if isinstance(s.get("score"), (int, float))]
    return {
        "schema": SCHEMA,
        "source": SOURCE,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "model": model,
        "harness": {"name": SOURCE, "results_version": results.get("results_version"), "model_path": harness_version},
        "comparable": True,
        "results": suites,
        "coverage": {
            "suites": len(suites),
            "scored": len(scored),
            "items": sum(int(s.get("total") or 0) for s in suites),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert lm-eval output to alecto quality artifacts")
    parser.add_argument("run_dir", help="directory produced by lm_eval --log_samples")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default="quality-lmeval.json")
    args = parser.parse_args(argv)

    results = load_results(args.run_dir)
    samples = load_samples(args.run_dir)
    artifact = convert(results, samples, endpoint=args.endpoint, model=args.model)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out} ({os.path.getsize(args.out)} bytes)")
    for suite in artifact["results"]:
        score = suite.get("score")
        print(f"  {suite['suite']:12s} {suite.get('metric_id', '-'):32s} "
              f"score={score if score is not None else 'unsupported'} n={suite.get('total')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
