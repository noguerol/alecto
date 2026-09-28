"""Unit tests for the lm-evaluation-harness -> alecto adapter."""

from __future__ import annotations

import json

import pytest

from alecto.lmeval_adapter import HEADLINE_METRICS, _headline, convert, load_samples

RESULTS = {
    "results": {
        "gsm8k": {
            "name": "gsm8k",
            "n-shot": 5,
            "exact_match,strict-match": 0.5,
            "exact_match_stderr,strict-match": 0.18,
            "exact_match,flexible-extract": 0.62,
            "exact_match_stderr,flexible-extract": 0.17,
        }
    },
    "versions": {"gsm8k": 3},
    "n-samples": {"gsm8k": {"original": 1319, "effective": 8}},
    "config": {"model": "test-model"},
    "results_version": 8,
}

SAMPLES = {
    "gsm8k": [
        {"doc_id": i, "target": "t", "filtered_resps": ["x"], "exact_match": 1.0 if i < 5 else 0.0}
        for i in range(8)
    ]
}


def test_headline_prefers_flexible_extract():
    metric, value, stderr, filt, metric_id = _headline("gsm8k", RESULTS["results"]["gsm8k"])
    assert metric == "exact_match"
    assert value == 0.62
    assert filt == "flexible-extract"
    assert metric_id == "exact_match,flexible-extract"
    assert stderr == 0.17


def test_headline_none_without_known_metric():
    assert _headline("x", {"some_other": 0.5}) is None


def test_convert_produces_comparable_artifact():
    artifact = convert(RESULTS, SAMPLES, endpoint="http://h/v1", model="m")
    assert artifact["schema"] == "alecto.quality_results.v1"
    assert artifact["comparable"] is True
    suite = artifact["results"][0]
    assert suite["suite"] == "gsm8k"
    assert suite["score"] == 0.62
    assert suite["correct"] == 5
    assert suite["total"] == 8
    assert suite["coverage"]["scored"] == 8
    assert suite["n_samples_original"] == 1319
    assert suite["task_version"] == 3
    assert len(suite["items"]) == 8


def test_convert_tolerates_missing_samples():
    artifact = convert(RESULTS, {}, endpoint=None, model=None)
    suite = artifact["results"][0]
    assert suite["score"] == 0.62
    assert suite["correct"] is None
    assert suite["items"] == []


def test_convert_marks_suite_without_metric():
    results = {"results": {"weird": {"acc_foo": 0.5}}, "versions": {}, "n-samples": {}, "config": {}}
    artifact = convert(results, {}, endpoint=None, model=None)
    assert artifact["results"][0]["score"] is None
    assert artifact["results"][0]["error"]


def test_convert_survives_garbage():
    for bad in [
        {"results": None},
        {"results": {"t": "not-a-dict"}},
        {"results": {"t": {"exact_match,x": "nan"}}},
        {},
    ]:
        artifact = convert(bad, {}, endpoint=None, model=None)
        assert isinstance(artifact["results"], list)


def test_load_samples_groups_by_task(tmp_path):
    line = json.dumps({"doc_id": 0, "exact_match": 1.0})
    (tmp_path / "samples_gsm8k_2026.jsonl").write_text(line + "\n" + line + "\n")
    grouped = load_samples(str(tmp_path))
    assert "gsm8k" in grouped
    assert len(grouped["gsm8k"]) == 2


def test_load_samples_skips_blank_and_corrupt(tmp_path):
    (tmp_path / "samples_gsm8k_x.jsonl").write_text("\n{not json}\n" + json.dumps({"doc_id": 1}) + "\n")
    grouped = load_samples(str(tmp_path))
    assert grouped.get("gsm8k") == [{"doc_id": 1}]


def test_headline_metric_ids_are_stable():
    assert set(HEADLINE_METRICS) >= {"exact_match", "pass@1", "prompt_level_strict_acc"}


def test_duplicate_sample_rows_are_deduplicated():
    """Re-running lm-eval into the same --output_path appends samples.

    Counting the same item twice would inflate n and make a slice look more
    reliable than it is, so ids are deduplicated and the duplicate count is
    reported rather than hidden.
    """
    results = {
        "results": {"gsm8k": {"exact_match,flexible-extract": 0.5, "stderr,flexible-extract": 0.19}},
        "n-samples": {"gsm8k": {"original": 1319, "effective": 8}},
        "versions": {"gsm8k": 3.0},
    }
    rows = [{"id": i % 8, "exact_match,flexible-extract": 1 if i % 8 < 4 else 0} for i in range(16)]
    suite = convert(results, {"gsm8k": rows}, endpoint="http://x/v1", model="m")["results"][0]
    assert suite["total"] == 8
    assert suite["coverage"]["duplicate_rows"] == 8
    assert suite["score"] == pytest.approx(0.5)


def test_no_duplicates_reports_zero():
    results = {
        "results": {"gsm8k": {"exact_match,flexible-extract": 0.5}},
        "n-samples": {"gsm8k": {"original": 1319, "effective": 4}},
    }
    rows = [{"id": i, "exact_match,flexible-extract": 1 if i < 2 else 0} for i in range(4)]
    suite = convert(results, {"gsm8k": rows}, endpoint="http://x/v1", model="m")["results"][0]
    assert suite["total"] == 4
    assert suite["coverage"]["duplicate_rows"] == 0
