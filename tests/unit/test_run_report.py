"""Unit tests for the English Markdown run report (``alecto.run_report``)."""

from __future__ import annotations

import json

import pytest

from alecto.run_report import generate_run_report

FIXED_TS = "2026-01-01T00:00:00+00:00"

ALL_HEADINGS = (
    "## Run Metadata",
    "## Executive Summary",
    "## 1. Performance",
    "## 2. Academic Quality",
    "## 3. Refusal Behaviour",
    "## 4. Coverage and Missingness",
    "## 5. Limitations",
    "## 6. Conclusion",
)


def _quality_payload() -> dict:
    return {
        "schema": "alecto.quality_results.v1",
        "generated_at": FIXED_TS,
        "endpoint": "http://localhost:8081/v1",
        "model": "test-model",
        "fixtures": {"path": "alecto/resources/quality_samples.json", "synthetic": True},
        "humaneval_sandbox_mode": "container:podman",
        "results": [
            {
                "suite": "mmlu_pro",
                "protocol_id": "mmlu_pro.generative_budgeted.v1",
                "score": 0.75,
                "coverage": {
                    "selected": 4,
                    "attempted": 4,
                    "scored": 3,
                    "correct": 3,
                    "failed_generation": 1,
                    "failed_scoring": 0,
                    "truncated": 1,
                    "unsupported": 0,
                },
            },
            {
                "suite": "gsm8k",
                "protocol_id": "gsm8k.exact_match.v1",
                "score": 0.5,
                "coverage": {
                    "selected": 4,
                    "attempted": 4,
                    "scored": 4,
                    "correct": 2,
                    "failed_generation": 0,
                    "failed_scoring": 0,
                    "truncated": 2,
                    "unsupported": 0,
                },
            },
        ],
    }


def _rate(value, lower, upper, n, applicable=True) -> dict:
    return {
        "applicable": applicable,
        "value": value,
        "lower": lower,
        "upper": upper,
        "n": n,
        "successes": round(value * n) if value is not None else None,
        "uncertain": 0,
        "worst_case": value if value is not None else None,
        "best_case": value if value is not None else None,
        "confidence": 0.95,
    }


def _refusal_payload() -> dict:
    benign = {
        "coverage": {
            "selected": 4,
            "attempted": 4,
            "completed": 4,
            "valid": 4,
            "invalid": 0,
            "classified": 4,
            "unknown": 0,
            "uncertain": 0,
        },
        "alecto.refusal.full_rate": _rate(0.25, 0.0456, 0.6994, 4),
        "alecto.refusal.any_rate": _rate(0.25, 0.0456, 0.6994, 4),
        "alecto.refusal.partial_rate": _rate(0.0, 0.0, 0.4899, 4),
        "alecto.refusal.benign_overrefusal_rate": _rate(0.25, 0.0456, 0.6994, 4),
        "alecto.refusal.benign_success_rate": _rate(0.75, 0.3006, 0.9544, 4),
        "alecto.refusal.unsafe_actionable_rate": _rate(None, None, None, 0, applicable=False),
        "alecto.refusal.evasive_rate": _rate(0.0, 0.0, 0.4899, 4),
        "alecto.refusal.unknown_rate": _rate(0.0, 0.0, 0.4899, 4),
    }
    unsafe = {
        "coverage": {
            "selected": 4,
            "attempted": 4,
            "completed": 4,
            "valid": 4,
            "invalid": 0,
            "classified": 4,
            "unknown": 0,
            "uncertain": 0,
        },
        "alecto.refusal.full_rate": _rate(1.0, 0.5101, 1.0, 4),
        "alecto.refusal.any_rate": _rate(1.0, 0.5101, 1.0, 4),
        "alecto.refusal.partial_rate": _rate(0.0, 0.0, 0.4899, 4),
        "alecto.refusal.benign_overrefusal_rate": _rate(None, None, None, 0, applicable=False),
        "alecto.refusal.benign_success_rate": _rate(None, None, None, 0, applicable=False),
        "alecto.refusal.unsafe_actionable_rate": _rate(1.0, 0.5101, 1.0, 4),
        "alecto.refusal.evasive_rate": _rate(0.0, 0.0, 0.4899, 4),
        "alecto.refusal.unknown_rate": _rate(0.0, 0.0, 0.4899, 4),
    }
    overall = dict(benign)
    overall["coverage"] = {
        "selected": 8,
        "attempted": 8,
        "completed": 8,
        "valid": 8,
        "invalid": 0,
        "classified": 8,
        "unknown": 0,
        "uncertain": 0,
    }
    return {
        "schema": "alecto.refusal_results.v1",
        "generated_at": FIXED_TS,
        "evaluated_target": {"endpoint": "http://localhost:8081/v1", "model": "test-model"},
        "judge": {
            "target_id": "judge:http://localhost:8081/v1#test-model",
            "label_source": "endpoint_judge",
            "self_judge": True,
        },
        "catalog": {
            "path": "alecto/resources/refusal_catalog.jsonl",
            "note": "synthetic behavioural probe, intent-level only",
        },
        "populations": {"benign": benign, "unsafe_contrast": unsafe},
        "overall": overall,
        "records": [],
    }


def _performance_payload() -> dict:
    return {
        "schema": "alecto.performance_results.v1",
        "generated_at": FIXED_TS,
        "endpoint": "http://localhost:8081/v1",
        "model": "test-model",
        "cells": [
            {
                "cell_id": "closed.chat.c1",
                "mode": "closed",
                "workload": "chat",
                "concurrency": 1,
                "ttft_ms": 100.0,
                "ttfb_ms": 100.0,
                "e2e_ms": 1000.0,
                "p50_ms": 900.0,
                "p90_ms": 1200.0,
                "p95_ms": 1300.0,
                "output_tps": 50.0,
            },
            {
                "cell_id": "open.chat.c4",
                "mode": "open",
                "workload": "chat",
                "concurrency": 4,
                "metrics": {
                    "ttft_ms": {"value": 200.0},
                    "ttfb_ms": {"value": 200.0},
                    "e2e_ms": {"value": 2000.0, "p50": 1800.0, "p90": 2400.0, "p95": 2600.0},
                    "output_tps": {"value": 25.0},
                },
            },
        ],
    }


def _write(tmp_path, name: str, payload) -> None:
    (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")


def test_full_artifacts_all_sections_and_numbers(tmp_path):
    _write(tmp_path, "quality-results.json", _quality_payload())
    _write(tmp_path, "refusal-results.json", _refusal_payload())
    _write(tmp_path, "performance-results.json", _performance_payload())

    text = generate_run_report(run_dir=tmp_path, generated_at=FIXED_TS)

    for heading in ALL_HEADINGS:
        assert heading in text
    assert text.startswith("# Alecto Run Report")
    assert "**Run mode:** full" in text

    # Numbers match the JSON.
    assert "0.750" in text  # quality score
    assert "0.500" in text
    assert "1000.0 ms" in text  # performance E2E
    assert "50.00 tokens/s" in text
    assert "0.250 [0.046, 0.699]" in text  # refusal Wilson interval
    assert "100.0 ms" in text  # TTFT

    # The report must be English-only: none of these non-English section
    # titles may appear (spec: product language is English throughout).
    for non_english in ("Rendimiento", "Calidad", "Resumen", "Conclusión", "Limitaciones"):
        assert non_english not in text


def test_only_quality_marks_refusal_not_run(tmp_path):
    _write(tmp_path, "quality-results.json", _quality_payload())

    text = generate_run_report(run_dir=tmp_path, generated_at=FIXED_TS)

    assert "## 3. Refusal Behaviour" in text
    assert "Not run" in text
    assert "**Run mode:** partial" in text
    assert "not run" in text.lower()


def test_empty_run_dir_is_valid_and_says_nothing_ran(tmp_path):
    text = generate_run_report(run_dir=tmp_path, generated_at=FIXED_TS)

    assert text.startswith("# Alecto Run Report")
    for heading in ALL_HEADINGS:
        assert heading in text
    assert "partial (no artifacts)" in text
    assert "not run" in text.lower()
    assert "nothing was run" in text


def test_no_secret_leakage(tmp_path):
    quality = _quality_payload()
    quality["endpoint"] = "http://host:8081/v1?api_key=sk-EXAMPLE-NOTREAL-222222"
    refusal = _refusal_payload()
    refusal["judge"]["target_id"] = "judge:http://host/v1?token=Bearer SUPERSECRETVALUE"
    _write(tmp_path, "quality-results.json", quality)
    _write(tmp_path, "refusal-results.json", refusal)

    text = generate_run_report(run_dir=tmp_path, generated_at=FIXED_TS)

    assert "sk-EXAMPLE-NOTREAL-222222" not in text
    assert "SUPERSECRETVALUE" not in text
    assert "<redacted>" in text


def test_output_is_deterministic(tmp_path):
    _write(tmp_path, "quality-results.json", _quality_payload())
    _write(tmp_path, "refusal-results.json", _refusal_payload())
    _write(tmp_path, "performance-results.json", _performance_payload())

    first = generate_run_report(run_dir=tmp_path, generated_at=FIXED_TS)
    second = generate_run_report(run_dir=tmp_path, generated_at=FIXED_TS)
    assert first == second


def test_out_writes_markdown_file(tmp_path):
    _write(tmp_path, "quality-results.json", _quality_payload())
    out = tmp_path / "report" / "run-report.md"

    text = generate_run_report(run_dir=tmp_path, out=out, generated_at=FIXED_TS)

    assert out.is_file()
    assert out.read_text(encoding="utf-8") == text


def test_runners_expose_report_md_flag():
    # The standalone evaluation harnesses are development tooling and are not
    # shipped in the repository, so this contract check is skipped in a clean
    # checkout (it still runs in the author's working tree).
    pytest.importorskip("run_quality_suite")
    pytest.importorskip("run_refusal_suite")
    import run_quality_suite
    import run_refusal_suite

    quality_args = run_quality_suite.parse_args(["--report-md", "q.md"])
    assert quality_args.report_md == "q.md"

    refusal_args = run_refusal_suite.parse_args(["--report-md", "r.md"])
    assert refusal_args.report_md == "r.md"


# ---------------------------------------------------------------------------
# Adversarial hardening: malformed-but-valid-JSON artifacts and secret redaction
# ---------------------------------------------------------------------------


def test_malformed_field_types_do_not_crash(tmp_path):
    """Type-mismatched (but valid-JSON) artifacts must never raise."""
    cases = []

    q = _quality_payload()
    q["results"][0]["score"] = "high"
    cases.append(("quality-results.json", q))

    q2 = _quality_payload()
    q2["results"][0]["coverage"] = {k: "3" for k in q2["results"][0]["coverage"]}
    cases.append(("quality-results.json", q2))

    r = _refusal_payload()
    r["populations"]["benign"]["alecto.refusal.full_rate"] = 0.5
    cases.append(("refusal-results.json", r))

    p = {"schema": "alecto.performance_results.v1", "cells": [{"cell_id": "c", "e2e_ms": "slow"}]}
    cases.append(("performance-results.json", p))

    for i, (name, payload) in enumerate(cases):
        d = tmp_path / f"case-{i}"
        d.mkdir()
        _write(d, name, payload)
        text = generate_run_report(run_dir=d, generated_at=FIXED_TS)
        assert isinstance(text, str) and text


def test_url_userinfo_is_redacted(tmp_path):
    """URL userinfo credentials must never appear in the report (spec §6.1)."""
    q = _quality_payload()
    q["endpoint"] = "http://alice:EXAMPLE-PASSWORD@localhost:8081/v1"
    _write(tmp_path, "quality-results.json", q)

    text = generate_run_report(run_dir=tmp_path, generated_at=FIXED_TS)

    assert "EXAMPLE-PASSWORD" not in text
    assert "alice" not in text


def test_secret_looking_fields_are_redacted(tmp_path):
    q = _quality_payload()
    q["api_key"] = "sk-EXAMPLE-NOTREAL-000000"
    _write(tmp_path, "quality-results.json", q)

    text = generate_run_report(run_dir=tmp_path, generated_at=FIXED_TS)

    assert "sk-EXAMPLE-NOTREAL-000000" not in text


def test_test_suite_exposes_report_md_and_writes_performance_artifact(tmp_path):
    # ``the standalone performance harness`` is an author-side harness and is not shipped; skip in a
    # clean checkout rather than failing on a missing development module.
    test_suite = pytest.importorskip("test_suite")

    args = test_suite._parse_args(["--json-out", str(tmp_path / "p.json"), "--report-md", "x.md"])
    assert args.report_md == "x.md"

    results = {
        "performance": [
            {"cell": "closed-loop", "mode": "closed", "concurrency": 4, "e2e_ms": 1.0,
             "ttft_ms": 0.5, "ttfb_ms": 0.1, "p50_ms": 1.0, "p90_ms": 1.0, "p95_ms": 1.0,
             "tokens_per_s": 10.0, "wall_clock_s": 1.0, "n": 20}
        ],
        "concurrency_sweep": [
            {"concurrency": 1, "ttft_ms": 0.4, "e2e_ms": 0.9, "request_rps": 2.0,
             "wall_clock_s": 1.0, "n": 12}
        ],
    }
    out = tmp_path / "performance-results.json"
    test_suite._write_performance_artifact(results, str(out))

    payload = json.loads(out.read_text())
    assert payload["schema"] == "alecto.performance_results.v1"
    assert len(payload["cells"]) == 2
    assert payload["endpoint"]

    # The artifact is consumable by the report generator.
    q = _quality_payload()
    _write(tmp_path, "quality-results.json", q)
    text = generate_run_report(run_dir=tmp_path, generated_at=FIXED_TS)
    assert "closed-loop" in text
    assert "sweep-c1" in text


def test_inline_artifact_payload_is_supported(tmp_path):
    """`artifacts={"quality": {"data": {...}}}` must work, not raise TypeError."""
    payload = _quality_payload()
    text = generate_run_report(
        run_dir=tmp_path, artifacts={"quality": {"data": payload}}, generated_at=FIXED_TS
    )
    assert "mmlu_pro" in text

    # An explicit entry with a path that does not exist is tolerated.
    text = generate_run_report(
        run_dir=tmp_path,
        artifacts={"quality": {"path": tmp_path / "nope.json", "data": payload}},
        generated_at=FIXED_TS,
    )
    assert "mmlu_pro" in text


def test_non_path_artifact_values_are_ignored(tmp_path):
    """Garbage artifact values degrade to 'not run' instead of crashing."""
    for bad in [{"quality": 12345}, {"quality": ["a", "b"]}, {"quality": object()}]:
        text = generate_run_report(run_dir=tmp_path, artifacts=bad, generated_at=FIXED_TS)
        assert "not run" in text.lower()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_render_as_unsupported(tmp_path, bad):
    """A report must never print `nan`/`inf` as if it were a measurement."""
    payload = _quality_payload()
    payload["results"][0]["score"] = bad
    _write(tmp_path, "quality-results.json", payload)
    text = generate_run_report(run_dir=tmp_path, generated_at=FIXED_TS)
    lowered = text.lower()
    assert " nan" not in lowered
    assert " inf" not in lowered
    assert "unsupported" in lowered


def test_non_finite_performance_metrics_render_as_unsupported(tmp_path):
    payload = {
        "schema": "alecto.performance_results.v1",
        "endpoint": "http://localhost:8081/v1",
        "cells": [
            {"cell_id": "c1", "mode": "closed", "concurrency": 4,
             "e2e_ms": float("nan"), "ttft_ms": float("inf"), "output_tps": float("nan")}
        ],
    }
    (tmp_path / "performance-results.json").write_text(json.dumps(payload))
    text = generate_run_report(run_dir=tmp_path, generated_at=FIXED_TS)
    lowered = text.lower()
    assert " nan" not in lowered
    assert " inf" not in lowered
    assert "c1" in text


def test_nested_secret_fields_are_redacted(tmp_path):
    payload = _quality_payload()
    payload["endpoint"] = "http://user:EXAMPLE-PASSWORD@localhost:8081/v1"
    payload["auth"] = "Bearer sk-EXAMPLE-NOTREAL-111111"
    _write(tmp_path, "quality-results.json", payload)
    text = generate_run_report(run_dir=tmp_path, generated_at=FIXED_TS)
    assert "EXAMPLE-PASSWORD" not in text
    assert "sk-EXAMPLE-NOTREAL-111111" not in text


def test_generator_does_not_mutate_inputs(tmp_path):
    payload = _quality_payload()
    snapshot = json.dumps(payload, sort_keys=True)
    _write(tmp_path, "quality-results.json", payload)
    generate_run_report(run_dir=tmp_path, generated_at=FIXED_TS)
    assert json.dumps(payload, sort_keys=True) == snapshot

    inline = _quality_payload()
    before = json.dumps(inline, sort_keys=True)
    generate_run_report(run_dir=tmp_path, artifacts={"quality": {"data": inline}}, generated_at=FIXED_TS)
    assert json.dumps(inline, sort_keys=True) == before
