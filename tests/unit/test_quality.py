"""Unit tests for the quality suite (spec §10)."""

import json

import pytest

from alecto import quality
from alecto.enums import BenchmarkCategory
from alecto.quality import (
    SANDBOX_HOST,
    GSM8KBenchmark,
    HumanEvalBenchmark,
    IFEvalBenchmark,
    MMLUProBenchmark,
    QualityBenchmark,
    QualityResult,
    QualitySample,
    _check_constraint,
    _extract_final_answer,
    _normalize_number,
    _pass_at_k,
    run_quality_benchmark,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mmlu_samples():
    return [
        QualitySample(
            id="mmlu-001",
            prompt="What is the capital of France?\nA) London B) Paris C) Berlin D) Madrid",
            expected="B",
            metadata={"domain": "geography"},
        ),
        QualitySample(
            id="mmlu-002",
            prompt="Which planet is closest to the Sun?\nA) Venus B) Mercury C) Mars D) Earth",
            expected="B",
            metadata={"domain": "astronomy"},
        ),
    ]


@pytest.fixture
def gsm8k_samples():
    return [
        QualitySample(
            id="gsm-001",
            prompt="Q: A farmer has 12 sheep. 5 run away. How many remain?",
            expected="7",
        ),
        QualitySample(
            id="gsm-002",
            prompt="Q: What is 3 × 4?",
            expected="12",
        ),
    ]


@pytest.fixture
def humaneval_samples():
    return [
        QualitySample(
            id="he-001",
            prompt='Write a Python function "add(a, b)" that returns the sum.',
            expected='from solution import add\nassert add(1, 2) == 3\nassert add(-1, 1) == 0',
        ),
        QualitySample(
            id="he-002",
            prompt='Write a Python function "mul(a, b)" that returns the product.',
            expected='from solution import mul\nassert mul(3, 4) == 12',
        ),
    ]


@pytest.fixture
def ifeval_samples():
    return [
        QualitySample(
            id="ife-001",
            prompt="Write a sentence containing the word 'hello'.",
            expected=[{"type": "contains_keyword", "keyword": "hello"}],
        ),
        QualitySample(
            id="ife-002",
            prompt="Write exactly 5 words.",
            expected=[{"type": "word_count_exact", "count": 5}],
        ),
    ]


class _MockAdapter:
    """Minimal adapter that returns canned responses in order."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._idx = 0

    async def call(self, step: dict) -> dict:
        text = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return {"text": text, "tokens": 10, "raw": {}}


# ---------------------------------------------------------------------------
# 1. Benchmark creation
# ---------------------------------------------------------------------------


class TestBenchmarkCreation:
    def test_mmlu_pro_creation(self, mmlu_samples):
        bench = MMLUProBenchmark(samples=mmlu_samples)
        assert bench.name == "mmlu_pro"
        assert bench.category == BenchmarkCategory.LANGUAGE
        assert bench.protocol_id == "mmlu_pro.generative_budgeted.v1"
        assert len(bench.samples) == 2

    def test_gsm8k_creation(self, gsm8k_samples):
        bench = GSM8KBenchmark(samples=gsm8k_samples)
        assert bench.name == "gsm8k"
        assert bench.category == BenchmarkCategory.MATH
        assert bench.protocol_id == "gsm8k.exact_match.v1"
        assert len(bench.samples) == 2

    def test_humaneval_creation(self, humaneval_samples):
        bench = HumanEvalBenchmark(samples=humaneval_samples, sandbox_mode=SANDBOX_HOST)
        assert bench.name == "humaneval"
        assert bench.category == BenchmarkCategory.CODING
        assert bench.protocol_id == "humaneval.instruct_budgeted.v1"
        assert bench.k == 1
        assert len(bench.samples) == 2

    def test_ifeval_creation(self, ifeval_samples):
        bench = IFEvalBenchmark(samples=ifeval_samples)
        assert bench.name == "ifeval"
        assert bench.category == BenchmarkCategory.LANGUAGE
        assert bench.protocol_id == "ifeval.strict.v1"
        assert len(bench.samples) == 2

    def test_base_benchmark(self):
        bench = QualityBenchmark()
        assert bench.name == "base"
        assert bench.samples == []

    def test_sample_to_prompt(self, mmlu_samples):
        assert mmlu_samples[0].to_prompt() == mmlu_samples[0].prompt


# ---------------------------------------------------------------------------
# 2. Scoring logic
# ---------------------------------------------------------------------------


class TestMMLUScoring:
    def test_correct_answer(self, mmlu_samples):
        bench = MMLUProBenchmark(samples=mmlu_samples)
        assert bench.scoring("The answer is B", mmlu_samples[0]) == 1.0

    def test_wrong_answer(self, mmlu_samples):
        bench = MMLUProBenchmark(samples=mmlu_samples)
        assert bench.scoring("The answer is A", mmlu_samples[0]) == 0.0

    def test_no_letter(self, mmlu_samples):
        bench = MMLUProBenchmark(samples=mmlu_samples)
        assert bench.scoring("I don't know", mmlu_samples[0]) == 0.0

    def test_aggregate_accuracy(self, mmlu_samples):
        bench = MMLUProBenchmark(samples=mmlu_samples)
        responses = ["Answer: B", "Answer: A"]  # 1 correct, 1 wrong
        result = bench.score_all(responses)
        assert result["score"] == 0.5
        assert result["n_samples"] == 2


class TestGSM8KScoring:
    def test_exact_match(self, gsm8k_samples):
        bench = GSM8KBenchmark(samples=gsm8k_samples)
        assert bench.scoring("The answer is #### 7", gsm8k_samples[0]) == 1.0

    def test_wrong_answer(self, gsm8k_samples):
        bench = GSM8KBenchmark(samples=gsm8k_samples)
        assert bench.scoring("The answer is #### 8", gsm8k_samples[0]) == 0.0

    def test_fallback_last_number(self, gsm8k_samples):
        bench = GSM8KBenchmark(samples=gsm8k_samples)
        # No #### marker; last number is 12
        assert bench.scoring("12 sheep remain", gsm8k_samples[1]) == 1.0

    def test_no_number(self, gsm8k_samples):
        bench = GSM8KBenchmark(samples=gsm8k_samples)
        assert bench.scoring("I cannot solve this", gsm8k_samples[0]) == 0.0

    def test_normalize_number(self):
        assert _normalize_number("1,000") == "1000"
        assert _normalize_number("-3.14") == "-3.14"
        assert _normalize_number("42.") == "42"
        assert _normalize_number("abc") is None

    def test_extract_final_answer(self):
        assert _extract_final_answer("The answer is #### 42") == "42"
        assert _extract_final_answer("42") == "42"
        assert _extract_final_answer("no numbers here") is None


class TestHumanEvalScoring:
    def test_correct_function(self, humaneval_samples):
        bench = HumanEvalBenchmark(samples=humaneval_samples, sandbox_mode=SANDBOX_HOST)
        code = "def add(a, b):\n    return a + b"
        assert bench.scoring(code, humaneval_samples[0]) == 1.0

    def test_wrong_function(self, humaneval_samples):
        bench = HumanEvalBenchmark(samples=humaneval_samples, sandbox_mode=SANDBOX_HOST)
        code = "def add(a, b):\n    return a - b"
        assert bench.scoring(code, humaneval_samples[0]) == 0.0

    def test_markdown_fence(self, humaneval_samples):
        bench = HumanEvalBenchmark(samples=humaneval_samples, sandbox_mode=SANDBOX_HOST)
        code = "```python\ndef add(a, b):\n    return a + b\n```"
        assert bench.scoring(code, humaneval_samples[0]) == 1.0

    def test_pass_at_k(self):
        assert _pass_at_k([True, False], k=1) == 0.5
        assert _pass_at_k([True, True], k=1) == 1.0
        assert _pass_at_k([], k=1) == 0.0
        # pass@2 with 2 candidates, 1 pass
        assert _pass_at_k([True, False], k=2) == 1.0

    def test_score_all_humaneval(self, humaneval_samples):
        bench = HumanEvalBenchmark(samples=humaneval_samples, sandbox_mode=SANDBOX_HOST)
        responses = [
            "def add(a, b):\n    return a + b",
            "def mul(a, b):\n    return a * b",
        ]
        result = bench.score_all(responses)
        assert result["score"] == 1.0
        assert result["pass_at_k"] == 1.0
        assert result["k"] == 1


class TestIFEvalScoring:
    def test_all_constraints_met(self, ifeval_samples):
        bench = IFEvalBenchmark(samples=ifeval_samples)
        # "hello world" contains "hello"
        assert bench.scoring("hello world", ifeval_samples[0]) == 1.0

    def test_keyword_missing(self, ifeval_samples):
        bench = IFEvalBenchmark(samples=ifeval_samples)
        assert bench.scoring("goodbye world", ifeval_samples[0]) == 0.0

    def test_word_count_exact(self, ifeval_samples):
        bench = IFEvalBenchmark(samples=ifeval_samples)
        # Exactly 5 words
        assert bench.scoring("one two three four five", ifeval_samples[1]) == 1.0
        # 4 words
        assert bench.scoring("one two three four", ifeval_samples[1]) == 0.0

    def test_strict_vs_loose(self, ifeval_samples):
        bench = IFEvalBenchmark(samples=ifeval_samples)
        # Sample 0: keyword met; Sample 1: 4 words (fail)
        responses = ["hello there", "one two three four"]
        result = bench.score_all(responses)
        assert result["strict_accuracy"] == 0.5
        # loose: per-constraint: 1/2 (keyword ok, word count fail)
        assert result["loose_accuracy"] == 0.5

    def test_check_constraint_uppercase(self):
        assert _check_constraint("HELLO WORLD", {"type": "uppercase"}) is True
        assert _check_constraint("hello world", {"type": "uppercase"}) is False

    def test_check_constraint_lowercase(self):
        assert _check_constraint("hello world", {"type": "lowercase"}) is True
        assert _check_constraint("HELLO WORLD", {"type": "lowercase"}) is False


# ---------------------------------------------------------------------------
# 3. Benchmark execution with mock adapter
# ---------------------------------------------------------------------------


class TestRunQualityBenchmark:
    @pytest.mark.asyncio
    async def test_mmlu_run(self, mmlu_samples):
        bench = MMLUProBenchmark(samples=mmlu_samples)
        adapter = _MockAdapter(["Answer: B", "Answer: A"])
        result = await run_quality_benchmark(bench, adapter)
        assert isinstance(result, QualityResult)
        assert result.benchmark_name == "mmlu_pro"
        assert result.score == 0.5
        assert len(result.raw_responses) == 2
        assert result.details["protocol_id"] == "mmlu_pro.generative_budgeted.v1"

    @pytest.mark.asyncio
    async def test_gsm8k_run(self, gsm8k_samples):
        bench = GSM8KBenchmark(samples=gsm8k_samples)
        adapter = _MockAdapter(["#### 7", "#### 12"])
        result = await run_quality_benchmark(bench, adapter)
        assert result.score == 1.0
        assert result.details["n_samples"] == 2

    @pytest.mark.asyncio
    async def test_ifeval_run(self, ifeval_samples):
        bench = IFEvalBenchmark(samples=ifeval_samples)
        adapter = _MockAdapter(["hello world", "one two three four five"])
        result = await run_quality_benchmark(bench, adapter)
        assert result.score == 1.0
        assert "strict_accuracy" in result.details
        assert "loose_accuracy" in result.details

    @pytest.mark.asyncio
    async def test_humaneval_run(self, humaneval_samples):
        bench = HumanEvalBenchmark(samples=humaneval_samples, sandbox_mode=SANDBOX_HOST)
        adapter = _MockAdapter([
            "def add(a, b):\n    return a + b",
            "def mul(a, b):\n    return a * b",
        ])
        result = await run_quality_benchmark(bench, adapter)
        assert result.benchmark_name == "humaneval"
        assert result.score == 1.0
        assert "pass_at_k" in result.details

    @pytest.mark.asyncio
    async def test_empty_samples(self):
        bench = MMLUProBenchmark(samples=[])
        adapter = _MockAdapter([])
        result = await run_quality_benchmark(bench, adapter)
        assert result.score == 0.0
        assert result.raw_responses == []


# ---------------------------------------------------------------------------
# 4. Result serialization
# ---------------------------------------------------------------------------


class TestQualityResultSerialization:
    def test_to_dict(self):
        r = QualityResult(
            benchmark_name="mmlu_pro",
            score=0.75,
            details={"n_samples": 4},
            raw_responses=["A", "B", "C", "D"],
        )
        d = r.to_dict()
        assert d["benchmark_name"] == "mmlu_pro"
        assert d["score"] == 0.75
        assert d["details"] == {"n_samples": 4}
        assert d["raw_responses"] == ["A", "B", "C", "D"]

    def test_to_json(self):
        r = QualityResult(
            benchmark_name="gsm8k",
            score=1.0,
            details={"n_samples": 2},
            raw_responses=["#### 7", "#### 12"],
        )
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["benchmark_name"] == "gsm8k"
        assert parsed["score"] == 1.0
        assert parsed["raw_responses"] == ["#### 7", "#### 12"]

    def test_json_roundtrip(self):
        r = QualityResult(
            benchmark_name="humaneval",
            score=0.5,
            details={"pass_at_k": 0.5, "k": 1},
            raw_responses=["code1", "code2"],
        )
        parsed = json.loads(r.to_json())
        assert parsed["details"]["pass_at_k"] == 0.5
        assert parsed["details"]["k"] == 1

    def test_empty_result(self):
        r = QualityResult(benchmark_name="test", score=0.0)
        d = r.to_dict()
        assert d["details"] == {}
        assert d["raw_responses"] == []


class TestContainerSandboxReaping:
    """A timed-out sandbox must not leave a container running.

    ``subprocess.run(timeout=...)`` only kills the client (``podman run``); the
    container itself keeps running under ``conmon`` until something removes it.
    The first version of this code cleaned up only on the timeout branch, and an
    interrupt or an unexpected error leaked a container that ran until the host
    was rebooted — dozens of them accumulated before anyone noticed.

    These tests drive the real code path with a stubbed runner, so they need no
    container runtime and still assert the exact commands issued.
    """

    @staticmethod
    def _calls(monkeypatch, *, expire: bool):
        import subprocess as real_subprocess

        recorded = []

        def fake_run(cmd, **kwargs):
            recorded.append(list(cmd))
            if cmd[:2] == ["podman", "run"] and expire:
                raise real_subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1))
            return real_subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(quality.subprocess, "run", fake_run)
        monkeypatch.setattr(quality, "detect_container_sandbox", lambda: "podman")
        monkeypatch.setattr(quality, "_sandbox_image", lambda: "img")
        return recorded

    def test_timeout_removes_the_container(self, monkeypatch):
        recorded = self._calls(monkeypatch, expire=True)
        result = quality._run_sandbox_container("code", "test", timeout=1)
        assert result["stderr"] == "timeout"
        assert result["passed"] is False
        # The container was launched by name...
        run_cmd = next(c for c in recorded if c[:2] == ["podman", "run"])
        assert "--name" in run_cmd
        name = run_cmd[run_cmd.index("--name") + 1]
        # ...and removed despite the timeout.
        assert ["podman", "rm", "-f", "--time", "0", name] in recorded

    def test_success_also_removes_the_container(self, monkeypatch):
        """Cleanup happens on every path, not just timeout."""
        recorded = self._calls(monkeypatch, expire=False)
        quality._run_sandbox_container("code", "test", timeout=1)
        run_cmd = next(c for c in recorded if c[:2] == ["podman", "run"])
        name = run_cmd[run_cmd.index("--name") + 1]
        assert ["podman", "rm", "-f", "--time", "0", name] in recorded

    def test_cleanup_failure_does_not_mask_the_result(self, monkeypatch):
        """A failing ``podman rm`` must not turn a pass into an exception."""
        import subprocess as real_subprocess

        def fake_run(cmd, **kwargs):
            if cmd[:3] == ["podman", "rm", "-f"]:
                raise real_subprocess.TimeoutExpired(cmd, 1)
            return real_subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(quality.subprocess, "run", fake_run)
        monkeypatch.setattr(quality, "detect_container_sandbox", lambda: "podman")
        monkeypatch.setattr(quality, "_sandbox_image", lambda: "img")
        result = quality._run_sandbox_container("code", "test", timeout=1)
        assert result["passed"] is True

    def test_container_name_is_unique_per_call(self, monkeypatch):
        """A shared name would let one run's cleanup kill another's container."""
        recorded = self._calls(monkeypatch, expire=False)
        quality._run_sandbox_container("a", "t", timeout=1)
        quality._run_sandbox_container("b", "t", timeout=1)
        names = [c[c.index("--name") + 1] for c in recorded if c[:2] == ["podman", "run"]]
        assert len(names) == 2
        assert names[0] != names[1]
