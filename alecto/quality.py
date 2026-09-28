"""Academic quality suite: MMLU-Pro, GSM8K, HumanEval, IFEval (spec §10).

Each suite has a versioned task adapter, a pinned sample manifest, a prompt
protocol, a parser, and a scorer.  The benchmark classes here are the
deterministic scoring layer: they never issue inference themselves.  The
``run_quality_benchmark`` coroutine drives samples through an adapter and
aggregates a :class:`QualityResult`.

Protocol IDs follow spec §10, e.g. ``mmlu_pro.generative_budgeted.v1``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .enums import BenchmarkCategory

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


@dataclass
class QualitySample:
    """One item in a quality suite.

    ``prompt`` is the full user-facing prompt for the item.
    ``expected`` is the ground-truth value used by the scorer
    (letter for MMLU-Pro, numeric string for GSM8K, test harness for
    HumanEval, constraint list for IFEval).
    ``metadata`` carries item id, domain, etc.
    """

    id: str
    prompt: str
    expected: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """Return the user-facing prompt for this sample."""
        return self.prompt


# ---------------------------------------------------------------------------
# Base benchmark
# ---------------------------------------------------------------------------


class QualityBenchmark:
    """Base class for all quality benchmarks.

    Subclasses set ``name``, ``category``, ``protocol_id`` and provide
    ``samples`` and ``scoring``.
    """

    name: str = "base"
    category: BenchmarkCategory = BenchmarkCategory.GENERAL
    protocol_id: str = "base.v1"

    def __init__(self, samples: list[QualitySample] | None = None):
        self.samples: list[QualitySample] = samples if samples is not None else []

    # -- abstract-ish interface ------------------------------------------------
    def scoring(self, response: str, sample: QualitySample) -> float:
        """Score a single response against its sample.  Returns 0..1."""
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------------
    def score_all(
        self, responses: list[str], samples: list[QualitySample | None] = None
    ) -> dict[str, Any]:
        """Score a list of responses.  Returns per-sample scores and aggregate."""
        samples = samples if samples is not None else self.samples
        per_sample: list[dict[str, Any]] = []
        total = 0.0
        for i, resp in enumerate(responses):
            sample = samples[i] if i < len(samples) else None
            s = self.scoring(resp, sample) if sample is not None else 0.0
            per_sample.append({"sample_id": sample.id if sample else f"idx-{i}", "score": s})
            total += s
        n = len(responses)
        return {
            "per_sample": per_sample,
            "score": total / n if n else 0.0,
            "n_samples": n,
        }


# ---------------------------------------------------------------------------
# MMLU-Pro
# ---------------------------------------------------------------------------

_MMLU_LETTER_RE = re.compile(r"\b([A-D])\b")


class MMLUProBenchmark(QualityBenchmark):
    """MMLU-Pro: multiple-choice accuracy.

    Each sample carries ``expected`` as the correct letter (``'A'``–``'D'``)
    and the prompt includes the four options.  The parser extracts the first
    standalone letter A–D from the response.
    """

    name = "mmlu_pro"
    category = BenchmarkCategory.LANGUAGE
    protocol_id = "mmlu_pro.generative_budgeted.v1"

    def scoring(self, response: str, sample: QualitySample) -> float:
        expected = sample.expected
        match = _MMLU_LETTER_RE.search(response)
        if match is None:
            return 0.0
        predicted = match.group(1)
        return 1.0 if predicted == expected else 0.0


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------


def _normalize_number(text: str) -> str | None:
    """Normalise a numeric string: strip commas, whitespace; keep sign/decimals."""
    text = text.strip()
    # Remove thousands separators
    text = text.replace(",", "")
    # Strip trailing period (e.g. "42.")
    if text.endswith("."):
        text = text[:-1]
    # Validate it looks like a number (optional sign, digits, optional decimal)
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", text):
        return text
    return None


def _extract_final_answer(response: str) -> str | None:
    """Extract the final numeric answer from a GSM8K-style response.

    Looks for a ``#### <number>`` marker first; falls back to the last
    number-like token in the response.  Parse failures return ``None``.
    """
    # Primary: #### marker
    marker = re.search(r"####\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*$", response.strip())
    if marker:
        norm = _normalize_number(marker.group(1))
        if norm is not None:
            return norm
    # Fallback: last number in response
    numbers = re.findall(r"[+-]?\d[\d,]*(?:\.\d+)?", response)
    if numbers:
        norm = _normalize_number(numbers[-1])
        if norm is not None:
            return norm
    return None


class GSM8KBenchmark(QualityBenchmark):
    """GSM8K: math word problems, exact numeric match.

    ``expected`` is the canonical numeric answer string (e.g. ``"42"``).
    """

    name = "gsm8k"
    category = BenchmarkCategory.MATH
    protocol_id = "gsm8k.exact_match.v1"

    def scoring(self, response: str, sample: QualitySample) -> float:
        predicted = _extract_final_answer(response)
        if predicted is None:
            return 0.0
        expected = str(sample.expected).strip()
        # Normalise both sides for comparison
        exp_norm = _normalize_number(expected)
        if exp_norm is None:
            exp_norm = expected
        return 1.0 if predicted == exp_norm else 0.0


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------


def _extract_code(response: str) -> str:
    """Extract Python code from a response, stripping markdown fences."""
    code = response.strip()
    # Remove markdown code fences
    fence = re.search(r"```(?:python)?\s*(.*?)```", code, re.DOTALL)
    if fence:
        code = fence.group(1).strip()
    return code


def _run_sandbox(code: str, test_code: str, timeout: float = 10.0) -> dict[str, Any]:
    """Run candidate code + test harness in a subprocess sandbox.

    Writes the candidate code as ``solution.py`` and the test harness as
    ``run_test.py`` in a temporary directory, then executes the test harness
    in a separate Python process with a wall-clock timeout.
    """
    tmpdir = tempfile.mkdtemp(prefix="alecto_sandbox_")
    solution_path = os.path.join(tmpdir, "solution.py")
    test_path = os.path.join(tmpdir, "run_test.py")
    try:
        with open(solution_path, "w") as f:
            f.write(code)
        with open(test_path, "w") as f:
            f.write(test_code)
        proc = subprocess.run(
            [sys.executable, test_path],
            capture_output=True,
            timeout=timeout,
            cwd=tmpdir,
            env={"PYTHONPATH": tmpdir},
        )
        return {
            "passed": proc.returncode == 0,
            "stdout": proc.stdout.decode(errors="replace"),
            "stderr": proc.stderr.decode(errors="replace"),
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "stdout": "",
            "stderr": "timeout",
            "returncode": -1,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Container sandbox (spec §16.2)
# ---------------------------------------------------------------------------

SANDBOX_HOST = "host_subprocess"
SANDBOX_CONTAINER = "container"
SANDBOX_UNSUPPORTED = "unsupported"


class SandboxUnavailableError(RuntimeError):
    """Raised when a spec-compliant container sandbox cannot be used."""


def detect_container_sandbox() -> str | None:
    """Return the name of an available OCI runtime (``podman``/``docker``).

    Only the presence of the CLI is checked here; the image itself is a
    separately prepared, pinned artifact (``ALECTO_SANDBOX_IMAGE``).
    """
    for candidate in ("podman", "docker"):
        if shutil.which(candidate):
            return candidate
    return None


def _sandbox_image() -> str | None:
    """Return the configured pinned sandbox image reference, if any."""
    image = os.environ.get("ALECTO_SANDBOX_IMAGE", "").strip()
    return image or None


def _run_sandbox_container(
    code: str,
    test_code: str,
    timeout: float = 10.0,
    image: str | None = None,
    runtime: str | None = None,
) -> dict[str, Any]:
    """Run candidate code + tests in a rootless OCI container (spec §16.2).

    Isolation: no network, read-only root filesystem, read-only bind of the
    work directory, a bounded tmpfs scratch area, non-root UID, dropped
    capabilities, no-new-privileges, memory/CPU/PID limits and a whole-process
    wall-clock timeout.  ``--pull=never`` keeps the run offline: the image must
    be prepared in advance and is referenced by ``ALECTO_SANDBOX_IMAGE``.

    Raises :class:`SandboxUnavailableError` when no runtime or no image is
    configured.  A sandbox setup failure is reported as an error result rather
    than silently falling back to host execution.
    """
    runtime = runtime or detect_container_sandbox()
    if not runtime:
        raise SandboxUnavailableError("no podman/docker runtime available")
    image = image or _sandbox_image()
    if not image:
        raise SandboxUnavailableError(
            "ALECTO_SANDBOX_IMAGE is not set; no pinned sandbox image available"
        )

    tmpdir = tempfile.mkdtemp(prefix="alecto_sandbox_")
    name = f"alecto_sandbox_{os.getpid()}_{int(time.time() * 1000) % 100000}"
    try:
        solution_path = os.path.join(tmpdir, "solution.py")
        test_path = os.path.join(tmpdir, "run_test.py")
        with open(solution_path, "w") as f:
            f.write(code)
        with open(test_path, "w") as f:
            f.write(test_code)
        # The container runs as an unprivileged UID, so the bind-mounted
        # directory must be world-readable.
        os.chmod(tmpdir, 0o755)
        os.chmod(solution_path, 0o644)
        os.chmod(test_path, 0o644)

        cmd = [
            runtime,
            "run",
            "--rm",
            "--name",
            name,
            "--network=none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=16m",
            "--user",
            "65534:65534",
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--pids-limit",
            "256",
            "--pull=never",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "PYTHONPATH=/work",
            "-v",
            # ``:z`` sets a shared SELinux label; without it a host with
            # SELinux enforcing denies the container read access.
            f"{tmpdir}:/work:ro,z",
            "-w",
            "/work",
            image,
            "python3",
            "run_test.py",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=tmpdir)
        stderr = proc.stderr.decode(errors="replace")
        # A missing (non-prepared) image is an infrastructure gap, not a
        # candidate failure: surface it as unsupported rather than a 0 score.
        if proc.returncode != 0 and re.search(
            r"unable to find image|no such image|image not known|not found locally",
            stderr,
            re.IGNORECASE,
        ):
            raise SandboxUnavailableError(
                f"sandbox image {image!r} not available locally"
            )
        return {
            "passed": proc.returncode == 0,
            "stdout": proc.stdout.decode(errors="replace"),
            "stderr": stderr,
            "returncode": proc.returncode,
            "sandbox_mode": f"{SANDBOX_CONTAINER}:{runtime}",
        }
    except subprocess.TimeoutExpired:
        subprocess.run([runtime, "rm", "-f", name], capture_output=True)
        return {
            "passed": False,
            "stdout": "",
            "stderr": "timeout",
            "returncode": -1,
            "sandbox_mode": f"{SANDBOX_CONTAINER}:{runtime}",
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _pass_at_k(passed: list[bool], k: int = 1) -> float:
    """Compute pass@k from a list of per-candidate pass/fail booleans.

    For k=1 this is simply the fraction of passed candidates.
    For k>1 the standard unbiased estimator is used.
    """
    n = len(passed)
    if n == 0:
        return 0.0
    if k == 1:
        return sum(passed) / n
    # Unbiased pass@k estimator: 1 - C(fail, k) / C(n, k)
    from math import comb
    fail = n - sum(passed)
    if fail == 0:
        return 1.0
    if n < k:
        return 1.0 if fail == 0 else 0.0
    return 1.0 - comb(fail, k) / comb(n, k)


class HumanEvalBenchmark(QualityBenchmark):
    """HumanEval: code generation with pass@k scoring via sandbox execution.

    ``expected`` is the test harness code (string) that imports and tests the
    generated function.  ``metadata`` may carry ``k`` for pass@k.
    """

    name = "humaneval"
    category = BenchmarkCategory.CODING
    protocol_id = "humaneval.instruct_budgeted.v1"

    def __init__(
        self,
        samples: list[QualitySample] | None = None,
        k: int = 1,
        timeout: float = 10.0,
        sandbox_mode: str = SANDBOX_UNSUPPORTED,
        sandbox_image: str | None = None,
    ):
        super().__init__(samples)
        self.k = k
        self.timeout = timeout
        # Host ``subprocess`` execution is NOT spec-compliant (§16.2), so it is
        # never the default: callers must explicitly opt in with
        # ``sandbox_mode=SANDBOX_HOST`` (dev/test only).  Production runs use
        # ``container`` or mark the suite ``unsupported``.
        self.sandbox_mode = sandbox_mode
        self.sandbox_image = sandbox_image

    def scoring(self, response: str, sample: QualitySample) -> float:
        code = _extract_code(response)
        test_code = sample.expected if isinstance(sample.expected, str) else ""
        if self.sandbox_mode in (SANDBOX_UNSUPPORTED, None):
            raise SandboxUnavailableError(
                "HumanEval container sandbox unavailable (spec §16.2)"
            )
        if self.sandbox_mode == SANDBOX_CONTAINER:
            result = _run_sandbox_container(
                code,
                test_code,
                timeout=self.timeout,
                image=self.sandbox_image,
            )
        else:
            result = _run_sandbox(code, test_code, timeout=self.timeout)
        return 1.0 if result["passed"] else 0.0

    def score_all(
        self, responses: list[str], samples: list[QualitySample | None] = None
    ) -> dict[str, Any]:
        samples = samples if samples is not None else self.samples
        per_sample: list[dict[str, Any]] = []
        passed: list[bool] = []
        for i, resp in enumerate(responses):
            sample = samples[i] if i < len(samples) else None
            s = self.scoring(resp, sample) if sample is not None else 0.0
            per_sample.append({"sample_id": sample.id if sample else f"idx-{i}", "score": s})
            passed.append(s >= 1.0)
        pass_at_k = _pass_at_k(passed, k=self.k)
        return {
            "per_sample": per_sample,
            "score": pass_at_k,
            "n_samples": len(responses),
            "pass_at_k": pass_at_k,
            "k": self.k,
        }


# ---------------------------------------------------------------------------
# IFEval
# ---------------------------------------------------------------------------

# Constraint types supported by the checker
_CONSTRAINT_TYPES = {
    "contains_keyword",
    "not_contains_keyword",
    "word_count_at_least",
    "word_count_at_most",
    "word_count_exact",
    "uppercase",
    "lowercase",
    "sentence_count_at_least",
    "sentence_count_at_most",
}


def _check_constraint(response: str, constraint: dict[str, Any]) -> bool:
    """Check a single IFEval constraint against a response."""
    ctype = constraint.get("type")
    if ctype not in _CONSTRAINT_TYPES:
        return False

    text = response.strip()

    if ctype == "contains_keyword":
        return constraint.get("keyword", "").lower() in text.lower()

    if ctype == "not_contains_keyword":
        return constraint.get("keyword", "").lower() not in text.lower()

    if ctype == "word_count_at_least":
        return len(text.split()) >= constraint.get("count", 0)

    if ctype == "word_count_at_most":
        return len(text.split()) <= constraint.get("count", 0)

    if ctype == "word_count_exact":
        return len(text.split()) == constraint.get("count", 0)

    if ctype == "uppercase":
        return text == text.upper()

    if ctype == "lowercase":
        return text == text.lower()

    if ctype == "sentence_count_at_least":
        sentences = re.split(r"[.!?]+", text)
        return len([s for s in sentences if s.strip()]) >= constraint.get("count", 0)

    if ctype == "sentence_count_at_most":
        sentences = re.split(r"[.!?]+", text)
        return len([s for s in sentences if s.strip()]) <= constraint.get("count", 0)

    return False


class IFEvalBenchmark(QualityBenchmark):
    """IFEval: instruction-following constraint satisfaction.

    ``expected`` is a list of constraint dicts, each with a ``type`` and
    parameters.  A sample scores 1.0 only if *all* constraints are satisfied
    (strict).  The aggregate also reports the loose (per-constraint) rate.
    """

    name = "ifeval"
    category = BenchmarkCategory.LANGUAGE
    protocol_id = "ifeval.strict.v1"

    def scoring(self, response: str, sample: QualitySample) -> float:
        constraints = sample.expected if isinstance(sample.expected, list) else []
        if not constraints:
            return 1.0
        results = [_check_constraint(response, c) for c in constraints]
        return 1.0 if all(results) else 0.0

    def score_all(
        self, responses: list[str], samples: list[QualitySample | None] = None
    ) -> dict[str, Any]:
        samples = samples if samples is not None else self.samples
        per_sample: list[dict[str, Any]] = []
        strict_total = 0.0
        loose_total = 0.0
        loose_n = 0
        for i, resp in enumerate(responses):
            sample = samples[i] if i < len(samples) else None
            s = self.scoring(resp, sample) if sample is not None else 0.0
            per_sample.append({"sample_id": sample.id if sample else f"idx-{i}", "score": s})
            strict_total += s
            # loose: per-constraint satisfaction
            if sample is not None:
                constraints = sample.expected if isinstance(sample.expected, list) else []
                for c in constraints:
                    loose_n += 1
                    if _check_constraint(resp, c):
                        loose_total += 1
        n = len(responses)
        strict = strict_total / n if n else 0.0
        loose = loose_total / loose_n if loose_n else 0.0
        return {
            "per_sample": per_sample,
            "score": strict,
            "strict_accuracy": strict,
            "loose_accuracy": loose,
            "n_samples": n,
        }


# ---------------------------------------------------------------------------
# QualityResult
# ---------------------------------------------------------------------------


@dataclass
class QualityResult:
    """Aggregated result of running a quality benchmark."""

    benchmark_name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    raw_responses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "score": self.score,
            "details": self.details,
            "raw_responses": self.raw_responses,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_quality_benchmark(
    benchmark: QualityBenchmark,
    adapter: Any,
    concurrency: int = 1,
) -> QualityResult:
    """Run a quality benchmark through an adapter and return a QualityResult.

    ``adapter`` must have an ``async call(step: dict) -> dict`` method
    (see :class:`alecto.adapters.TargetAdapter`).  Each sample's prompt is
    sent as a user message; the response text is scored by the benchmark.

    Concurrency is 1 by default (spec §10 baseline quality concurrency).
    A higher value is a separately labelled throughput-oriented mode that
    must be paired with an invariance probe -- it is accepted here but the
    caller is responsible for labelling it.
    """
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def _one(sample: QualitySample) -> str:
        step = {
            "messages": [{"role": "user", "content": sample.prompt}],
            "temperature": 0.0,
        }
        async with semaphore:
            result = await adapter.call(step)
        return result.get("text", "")

    if benchmark.samples:
        raw_responses = list(await asyncio.gather(*[_one(s) for s in benchmark.samples]))
    else:
        raw_responses = []

    scores = benchmark.score_all(raw_responses)
    details = {
        "protocol_id": benchmark.protocol_id,
        "n_samples": scores["n_samples"],
        "per_sample": scores["per_sample"],
    }
    # Benchmark-specific extra details
    if "pass_at_k" in scores:
        details["pass_at_k"] = scores["pass_at_k"]
        details["k"] = scores["k"]
    if "strict_accuracy" in scores:
        details["strict_accuracy"] = scores["strict_accuracy"]
        details["loose_accuracy"] = scores["loose_accuracy"]

    return QualityResult(
        benchmark_name=benchmark.name,
        score=scores["score"],
        details=details,
        raw_responses=raw_responses,
    )


# ---------------------------------------------------------------------------
# Bundled fixtures, registry and suite runner
# ---------------------------------------------------------------------------

_RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
_QUALITY_SAMPLES_PATH = _RESOURCES_DIR / "quality_samples.json"

# suite name -> (benchmark class, Alecto protocol id)
# The protocol ids keep their Alecto names; the bundled fixtures are
# SYNTHETIC and are not official benchmark subsets.
QUALITY_SUITES: dict[str, tuple[type[QualityBenchmark], str]] = {
    "mmlu_pro": (MMLUProBenchmark, MMLUProBenchmark.protocol_id),
    "gsm8k": (GSM8KBenchmark, GSM8KBenchmark.protocol_id),
    "humaneval": (HumanEvalBenchmark, HumanEvalBenchmark.protocol_id),
    "ifeval": (IFEvalBenchmark, IFEvalBenchmark.protocol_id),
}

# Default academic output-token caps (spec §8.2).  These cap the generation
# budget per item; they are not completion guarantees.
QUALITY_CAPS: dict[str, int] = {
    "mmlu_pro": 768,
    "gsm8k": 768,
    "humaneval": 1024,
    "ifeval": 1024,
}

# Profile caps (spec §8.2): sample counts are caps, not targets.
QUALITY_PROFILE_CAPS: dict[str, dict[str, int]] = {
    "smoke": {name: 12 for name in QUALITY_SUITES},
    "quick": {"mmlu_pro": 84, "gsm8k": 64, "humaneval": 64, "ifeval": 64},
    "compare": {"mmlu_pro": 140, "gsm8k": 100, "humaneval": 164, "ifeval": 100},
}


def load_quality_manifest(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the bundled synthetic quality-samples manifest."""
    manifest_path = Path(path) if path is not None else _QUALITY_SAMPLES_PATH
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_quality_samples(
    suite_name: str, path: str | os.PathLike[str] | None = None
) -> list[QualitySample]:
    """Load the bundled synthetic samples for ``suite_name``.

    Raises ``KeyError`` for an unknown suite.  Every returned sample carries
    ``metadata['synthetic'] is True`` so fixtures are never mistaken for
    measured model results.
    """
    if suite_name not in QUALITY_SUITES:
        raise KeyError(
            f"unknown quality suite {suite_name!r}; known: {sorted(QUALITY_SUITES)}"
        )
    manifest = load_quality_manifest(path)
    suites = manifest.get("suites", {})
    if suite_name not in suites:
        raise KeyError(f"suite {suite_name!r} missing from bundled fixtures")
    samples: list[QualitySample] = []
    for raw in suites[suite_name].get("samples", []):
        metadata = dict(raw.get("metadata") or {})
        metadata.setdefault("synthetic", True)
        samples.append(
            QualitySample(
                id=raw["id"],
                prompt=raw["prompt"],
                expected=raw["expected"],
                metadata=metadata,
            )
        )
    return samples


async def run_quality_suite(
    suite_name: str,
    adapter: Any,
    limit: int | None = None,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Run one quality suite through a single adapter (spec §10).

    Returns a dict containing the :class:`QualityResult` (``result``) plus the
    two-layer outcome coverage (spec §10): ``capability_score`` over valid
    completed evaluations and ``end_to_end_success`` over attempted items,
    with ``selected``/``attempted``/``scored``/``failed_generation``/
    ``failed_scoring``/``truncated``/``unsupported``/``correct`` counts.

    ``temperature=0`` is enforced and the caller-supplied adapter is reused for
    every item.  HumanEval runs in a container sandbox when one is prepared; if
    not, the suite is marked ``unsupported`` rather than executing generated
    code on the host (spec §16.2).
    """
    if suite_name not in QUALITY_SUITES:
        raise KeyError(
            f"unknown quality suite {suite_name!r}; known: {sorted(QUALITY_SUITES)}"
        )
    benchmark_cls, protocol_id = QUALITY_SUITES[suite_name]
    all_samples = load_quality_samples(suite_name)
    if limit is not None:
        selected = all_samples[: max(0, int(limit))]
    else:
        selected = list(all_samples)

    benchmark = benchmark_cls(samples=selected)
    sandbox_mode: str | None = None
    if isinstance(benchmark, HumanEvalBenchmark):
        runtime = detect_container_sandbox()
        if runtime is None or not _sandbox_image():
            benchmark.sandbox_mode = SANDBOX_UNSUPPORTED
            benchmark.sandbox_image = None
            sandbox_mode = SANDBOX_UNSUPPORTED
        else:
            benchmark.sandbox_mode = SANDBOX_CONTAINER
            sandbox_mode = f"{SANDBOX_CONTAINER}:{runtime}"

    max_tokens = QUALITY_CAPS.get(suite_name)
    records: list[dict[str, Any]] = []
    if not selected:
        records = []
    elif sandbox_mode == SANDBOX_UNSUPPORTED:
        records = [
            {
                "sample": sample,
                "text": "",
                "finish_reason": None,
                "status": "unsupported",
                "error": "container sandbox unavailable",
            }
            for sample in selected
        ]
    else:
        semaphore = asyncio.Semaphore(max(1, int(concurrency)))

        async def _generate(sample: QualitySample) -> dict[str, Any]:
            step: dict[str, Any] = {
                "messages": [{"role": "user", "content": sample.prompt}],
                "temperature": 0.0,
            }
            if max_tokens is not None:
                step["max_tokens"] = max_tokens
            async with semaphore:
                try:
                    response = await adapter.call(step)
                except Exception as exc:  # noqa: BLE001 - retained as coverage
                    return {
                        "sample": sample,
                        "text": "",
                        "finish_reason": None,
                        "status": "generation_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            return {
                "sample": sample,
                "text": response.get("text", "") or "",
                "finish_reason": response.get("finish_reason"),
                "status": "ok",
                "error": None,
            }

        records = list(await asyncio.gather(*[_generate(s) for s in selected]))

    scored_texts: list[str] = []
    scored_samples: list[QualitySample] = []
    per_item: list[dict[str, Any]] = []
    correct = 0
    failed_scoring = 0
    truncated = 0

    for record in records:
        sample = record["sample"]
        item: dict[str, Any] = {
            "id": sample.id,
            "status": record["status"],
            "score": None,
            "error": record["error"],
            "finish_reason": record["finish_reason"],
            "response_len": len(record["text"]),
        }
        if record["finish_reason"] == "length":
            truncated += 1
            item["truncated"] = True
        if record["status"] in ("unsupported", "generation_failed"):
            per_item.append(item)
            continue
        try:
            score = benchmark.scoring(record["text"], sample)
        except SandboxUnavailableError as exc:
            item["status"] = "unsupported"
            item["error"] = str(exc)
            per_item.append(item)
            continue
        except Exception as exc:  # noqa: BLE001 - retained as coverage
            item["status"] = "scoring_failed"
            item["error"] = f"{type(exc).__name__}: {exc}"
            failed_scoring += 1
            per_item.append(item)
            continue
        item["status"] = "scored"
        item["score"] = score
        if score >= 1.0:
            correct += 1
        scored_texts.append(record["text"])
        scored_samples.append(sample)
        per_item.append(item)

    if scored_samples:
        scores = benchmark.score_all(scored_texts, scored_samples)
    else:
        scores = {"per_sample": [], "score": 0.0, "n_samples": 0}

    details: dict[str, Any] = {
        "protocol_id": protocol_id,
        "n_samples": scores["n_samples"],
        "per_sample": scores["per_sample"],
    }
    if "pass_at_k" in scores:
        details["pass_at_k"] = scores["pass_at_k"]
        details["k"] = scores["k"]
    if "strict_accuracy" in scores:
        details["strict_accuracy"] = scores["strict_accuracy"]
        details["loose_accuracy"] = scores["loose_accuracy"]

    result = QualityResult(
        benchmark_name=benchmark.name,
        score=scores["score"],
        details=details,
        raw_responses=[record["text"] for record in records],
    )

    unsupported = sum(1 for r in records if r["status"] == "unsupported")
    attempted = len(records) - unsupported
    failed_generation = sum(
        1 for r in records if r["status"] == "generation_failed"
    )
    scored = len(scored_samples)
    end_to_end = correct / attempted if attempted else 0.0
    coverage = {
        "selected": len(selected),
        "attempted": attempted,
        "scored": scored,
        "correct": correct,
        "failed_generation": failed_generation,
        "failed_scoring": failed_scoring,
        "truncated": truncated,
        "unsupported": unsupported,
    }

    return {
        "suite": suite_name,
        "protocol_id": protocol_id,
        "benchmark": benchmark.name,
        "score": result.score,
        "capability_score": result.score,
        "end_to_end_success": end_to_end,
        "coverage": coverage,
        "sandbox_mode": sandbox_mode,
        "items": per_item,
        "result": result.to_dict(),
    }
