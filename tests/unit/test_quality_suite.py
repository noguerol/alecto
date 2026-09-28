"""Tests for the bundled quality fixtures, registry, scorers and suite runner.

Covers the spec §10 additions: ``load_quality_samples``, ``QUALITY_SUITES``,
``run_quality_suite`` two-layer coverage, and the ``concurrency`` parameter of
``run_quality_benchmark``.
"""

from __future__ import annotations

import asyncio

import pytest

from alecto.quality import (
    QUALITY_CAPS,
    QUALITY_PROFILE_CAPS,
    QUALITY_SUITES,
    SANDBOX_HOST,
    GSM8KBenchmark,
    HumanEvalBenchmark,
    IFEvalBenchmark,
    MMLUProBenchmark,
    QualitySample,
    _extract_code,
    _extract_final_answer,
    _normalize_number,
    load_quality_manifest,
    load_quality_samples,
    run_quality_benchmark,
    run_quality_suite,
)

EXPECTED_PROTOCOLS = {
    "mmlu_pro": "mmlu_pro.generative_budgeted.v1",
    "gsm8k": "gsm8k.exact_match.v1",
    "humaneval": "humaneval.instruct_budgeted.v1",
    "ifeval": "ifeval.strict.v1",
}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class TrackingAdapter:
    """Adapter that records concurrency and returns canned responses."""

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        responder=None,
        fail: bool = False,
        fail_scoring: bool = False,
        delay: float = 0.03,
        finish_reason: str = "stop",
    ):
        self.responses = responses or []
        self.responder = responder
        self.fail = fail
        self.fail_scoring = fail_scoring
        self.delay = delay
        self.finish_reason = finish_reason
        self.calls = 0
        self.inflight = 0
        self.max_inflight = 0
        self.steps: list[dict] = []

    async def call(self, step: dict) -> dict:
        self.steps.append(step)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(self.delay)
            if self.fail:
                raise RuntimeError("simulated transport failure")
            if self.responder is not None:
                text = self.responder(step)
            elif self.responses:
                text = self.responses[self.calls % len(self.responses)]
            else:
                text = ""
            self.calls += 1
            return {"text": text, "finish_reason": self.finish_reason}
        finally:
            self.inflight -= 1


def mmlu_samples(n: int = 6) -> list[QualitySample]:
    return [
        QualitySample(
            id=f"m-{i}",
            prompt=f"Question {i}?\nA) a B) b C) c D) d",
            expected="B",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. Loader / registry
# ---------------------------------------------------------------------------


class TestQualityLoader:
    def test_registry_has_four_suites(self):
        assert set(QUALITY_SUITES) == set(EXPECTED_PROTOCOLS)
        for name, (_cls, protocol_id) in QUALITY_SUITES.items():
            assert protocol_id == EXPECTED_PROTOCOLS[name]
            assert name != protocol_id

    def test_registry_protocol_ids_match_classes(self):
        for name, (cls, protocol_id) in QUALITY_SUITES.items():
            assert cls.protocol_id == protocol_id, name

    def test_all_suites_load_and_are_well_formed(self):
        for name in QUALITY_SUITES:
            samples = load_quality_samples(name)
            assert len(samples) >= 12, name
            ids = [s.id for s in samples]
            assert len(ids) == len(set(ids)), f"duplicate ids in {name}"
            for sample in samples:
                assert sample.prompt.strip(), (name, sample.id)
                assert sample.expected not in ("", None, []), (name, sample.id)
                assert sample.metadata.get("synthetic") is True, (name, sample.id)

    def test_manifest_marks_synthetic_and_keeps_protocol_ids(self):
        manifest = load_quality_manifest()
        assert manifest["synthetic"] is True
        assert "not official" in manifest["provenance"].lower()
        for name, (cls, _protocol_id) in QUALITY_SUITES.items():
            entry = manifest["suites"][name]
            assert entry["protocol_id"] == cls.protocol_id
            assert len(entry["samples"]) >= 12

    def test_humaneval_expected_is_test_harness_source(self):
        for sample in load_quality_samples("humaneval"):
            assert isinstance(sample.expected, str)
            assert "from solution import" in sample.expected

    def test_ifeval_expected_is_constraint_list(self):
        for sample in load_quality_samples("ifeval"):
            assert isinstance(sample.expected, list)
            assert sample.expected
            assert all("type" in c for c in sample.expected)

    def test_unknown_suite_raises(self):
        with pytest.raises(KeyError):
            load_quality_samples("does_not_exist")

    def test_profile_caps_are_named_caps(self):
        assert QUALITY_PROFILE_CAPS["quick"] == {
            "mmlu_pro": 84,
            "gsm8k": 64,
            "humaneval": 64,
            "ifeval": 64,
        }
        assert QUALITY_PROFILE_CAPS["smoke"] == {name: 12 for name in QUALITY_SUITES}
        assert set(QUALITY_CAPS) == set(QUALITY_SUITES)


# ---------------------------------------------------------------------------
# 2. Scorers on hand-written golden strings
# ---------------------------------------------------------------------------


class TestGoldenScorers:
    def test_mmlu_golden(self):
        bench = MMLUProBenchmark()
        sample = QualitySample(id="g", prompt="q", expected="C")
        assert bench.scoring("The answer is C.", sample) == 1.0
        assert bench.scoring("I think\nC) because", sample) == 1.0
        assert bench.scoring("The answer is A.", sample) == 0.0
        assert bench.scoring("no option given", sample) == 0.0

    def test_gsm8k_golden_numeric_normalisation(self):
        bench = GSM8KBenchmark()
        sample = QualitySample(id="g", prompt="q", expected="1234")
        assert _normalize_number("1,234") == "1234"
        assert _extract_final_answer("we get 1234\n#### 1,234") == "1234"
        assert bench.scoring("reasoning ...\n#### 1,234", sample) == 1.0
        assert bench.scoring("reasoning ... the answer is 1,235", sample) == 0.0

    def test_gsm8k_golden_decimals_and_signs(self):
        bench = GSM8KBenchmark()
        assert bench.scoring("#### -3.5", QualitySample("s", "q", expected="-3.5")) == 1.0
        assert bench.scoring("#### 0.50", QualitySample("s", "q", expected="0.5")) == 0.0
        assert _normalize_number("42.") == "42"

    def test_humaneval_golden_code_extraction_and_scoring(self):
        assert _extract_code("```python\ndef f():\n    return 1\n```") == "def f():\n    return 1"
        sample = QualitySample(
            id="g",
            prompt="write f",
            expected="from solution import f\n\n\ndef check(candidate):\n    assert candidate() == 1\n\n\ncheck(f)\n",
        )
        bench = HumanEvalBenchmark(sandbox_mode=SANDBOX_HOST)
        assert bench.scoring("def f():\n    return 1", sample) == 1.0
        assert bench.scoring("def f():\n    return 2", sample) == 0.0

    def test_humaneval_unsupported_raises(self):
        from alecto.quality import SandboxUnavailableError

        sample = QualitySample("g", "write f", "from solution import f\ncheck(f)\n")
        bench = HumanEvalBenchmark(sandbox_mode="unsupported")
        with pytest.raises(SandboxUnavailableError):
            bench.scoring("def f():\n    return 1", sample)

    def test_humaneval_default_is_not_host_exec(self):
        """Spec §16.2: host subprocess execution must never be the default.

        A bare HumanEvalBenchmark() must refuse to score (container sandbox
        unavailable) instead of silently executing model code on the host.
        """
        from alecto.quality import SandboxUnavailableError

        sample = QualitySample("g", "write f", "from solution import f\ncheck(f)\n")
        bench = HumanEvalBenchmark()
        assert bench.sandbox_mode == "unsupported"
        with pytest.raises(SandboxUnavailableError):
            bench.scoring("def f():\n    return 1", sample)

    def test_ifeval_golden_strings(self):
        bench = IFEvalBenchmark()
        upper = QualitySample("u", "q", [{"type": "uppercase"}])
        assert bench.scoring("HELLO", upper) == 1.0
        assert bench.scoring("Hello", upper) == 0.0

        kw = QualitySample("k", "q", [{"type": "contains_keyword", "keyword": "ocean"}])
        assert bench.scoring("The OCEAN is deep.", kw) == 1.0
        assert bench.scoring("The sea is deep.", kw) == 0.0

        strict = QualitySample(
            "s", "q",
            [{"type": "word_count_exact", "count": 5},
             {"type": "not_contains_keyword", "keyword": "the"}],
        )
        assert bench.scoring("alpha beta gamma delta epsilon", strict) == 1.0
        assert bench.scoring("the alpha beta gamma delta", strict) == 0.0


# ---------------------------------------------------------------------------
# 3. run_quality_benchmark concurrency
# ---------------------------------------------------------------------------


class TestRunQualityBenchmarkConcurrency:
    async def test_baseline_concurrency_is_one(self):
        adapter = TrackingAdapter(["B"] * 6, delay=0.02)
        bench = MMLUProBenchmark(samples=mmlu_samples(6))
        result = await run_quality_benchmark(bench, adapter)
        assert result.score == 1.0
        assert len(result.raw_responses) == 6
        assert adapter.max_inflight == 1

    async def test_higher_concurrency_overlaps(self):
        adapter = TrackingAdapter(["B"] * 9, delay=0.04)
        bench = MMLUProBenchmark(samples=mmlu_samples(9))
        result = await run_quality_benchmark(bench, adapter, concurrency=3)
        assert result.score == 1.0
        assert len(result.raw_responses) == 9
        assert 2 <= adapter.max_inflight <= 3

    async def test_returns_quality_result_type(self):
        adapter = TrackingAdapter(["B", "A"], delay=0.0)
        bench = MMLUProBenchmark(samples=mmlu_samples(2))
        result = await run_quality_benchmark(bench, adapter, concurrency=2)
        assert result.benchmark_name == "mmlu_pro"
        assert result.details["protocol_id"] == "mmlu_pro.generative_budgeted.v1"
        assert result.details["n_samples"] == 2


# ---------------------------------------------------------------------------
# 4. run_quality_suite coverage (spec §10 two-layer outcome)
# ---------------------------------------------------------------------------


class TestRunQualitySuite:
    async def test_mmlu_suite_coverage_and_caps(self):
        answers = {s.prompt: s.expected for s in load_quality_samples("mmlu_pro")}
        adapter = TrackingAdapter(
            responder=lambda step: answers[step["messages"][0]["content"]],
            delay=0.0,
        )
        out = await run_quality_suite("mmlu_pro", adapter, limit=4, concurrency=1)
        assert out["suite"] == "mmlu_pro"
        assert out["protocol_id"] == "mmlu_pro.generative_budgeted.v1"
        assert out["score"] == 1.0
        assert out["end_to_end_success"] == 1.0
        cov = out["coverage"]
        assert cov["selected"] == 4
        assert cov["attempted"] == 4
        assert cov["scored"] == 4
        assert cov["correct"] == 4
        assert cov["failed_generation"] == 0
        assert cov["truncated"] == 0
        # temperature=0 and the §8.2 cap are forwarded; one adapter reused.
        assert all(s["temperature"] == 0.0 for s in adapter.steps)
        assert all(s["max_tokens"] == QUALITY_CAPS["mmlu_pro"] for s in adapter.steps)
        assert len(adapter.steps) == 4

    async def test_failed_generation_is_not_a_wrong_answer(self):
        adapter = TrackingAdapter(["B"] * 3, fail=True, delay=0.0)
        out = await run_quality_suite("mmlu_pro", adapter, limit=3)
        cov = out["coverage"]
        assert cov["attempted"] == 3
        assert cov["failed_generation"] == 3
        assert cov["scored"] == 0
        assert cov["correct"] == 0
        assert out["capability_score"] == 0.0
        assert out["end_to_end_success"] == 0.0
        assert out["items"][0]["status"] == "generation_failed"

    async def test_truncated_finish_reason_is_counted(self):
        adapter = TrackingAdapter(["#### 2"] * 2, finish_reason="length", delay=0.0)
        out = await run_quality_suite("gsm8k", adapter, limit=2)
        cov = out["coverage"]
        assert cov["truncated"] == 2
        assert cov["scored"] == 2

    async def test_humaneval_without_container_is_unsupported(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "alecto.quality.detect_container_sandbox", lambda: None
        )
        adapter = TrackingAdapter(["def add(a, b):\n    return a + b"] * 12)
        out = await run_quality_suite("humaneval", adapter, limit=3)
        cov = out["coverage"]
        assert out["sandbox_mode"] == "unsupported"
        assert cov["selected"] == 3
        assert cov["attempted"] == 0
        assert cov["unsupported"] == 3
        assert cov["scored"] == 0
        # No generated code is executed when the sandbox is unavailable.
        assert adapter.steps == []
        assert all(it["status"] == "unsupported" for it in out["items"])

    async def test_ifeval_suite_reports_strict_and_loose(self):
        adapter = TrackingAdapter(["ocean waves crash loudly tonight"] * 12, delay=0.0)
        out = await run_quality_suite("ifeval", adapter, limit=3, concurrency=2)
        assert "strict_accuracy" in out["result"]["details"]
        assert "loose_accuracy" in out["result"]["details"]
        assert out["coverage"]["scored"] == 3
