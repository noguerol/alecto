"""Official benchmark suites, run through an external evaluation harness.

Alecto ships small synthetic fixtures so its quality suites work offline and
regressions are caught in CI. Those fixtures are **not** official benchmark
subsets and their scores are **not** leaderboard comparable. This module closes
that gap without pulling the evaluation stack into Alecto's own dependency tree.

Design
------

Alecto stays thin (``httpx`` and ``jsonschema``) and treats the evaluation
harness as an **optional external engine**: a separate interpreter that already
has ``lm-evaluation-harness`` installed. Alecto knows which benchmarks exist and
what they measure, drives that engine against a target, and converts its output
through :mod:`alecto.lmeval_adapter` into Alecto's own schema with
``comparable: true``.

The engine is never vendored and never a hard dependency. If it is absent, the
operation fails with a typed ``alecto.tool.unsupported`` error that says how to
provide one — it never silently falls back to the synthetic fixtures, because a
fixture score presented as an official one is exactly the failure this project
exists to prevent.

Why not bundle a second framework: an alternative harness (EvalScope) carries
122 transitive packages against Alecto's 13, and would introduce a *second*
implementation of benchmarks Alecto already reaches. Two scorers under the same
benchmark name means two different numbers for "GSM8K", which destroys the
comparability the integration exists to provide.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AlectoConfig
from .errors import AlectoError


class ExternalEngineUnavailable(AlectoError):
    """No usable external evaluation harness was found."""

    code = "alecto.tool.unsupported"


@dataclass(frozen=True)
class OfficialBenchmark:
    """A benchmark Alecto can request from the external harness."""

    name: str
    task: str
    category: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "category": self.category,
            "description": self.description,
            "official": True,
            "comparable": True,
            "engine": "lm-evaluation-harness",
        }


# Curated set: the four suites Alecto also ships fixtures for, plus the widely
# quoted classics. Every ``task`` id below was verified to exist in
# lm-evaluation-harness before being listed — a name that resolves to nothing is
# worse than no name at all.
OFFICIAL_BENCHMARKS: dict[str, OfficialBenchmark] = {
    # --- the four suites that also have bundled fixtures -------------------
    "mmlu_pro": OfficialBenchmark(
        "mmlu_pro", "mmlu_pro", "knowledge",
        "MMLU-Pro: harder, reasoning-heavy multiple-choice across many subjects.",
    ),
    "gsm8k": OfficialBenchmark(
        "gsm8k", "gsm8k", "math",
        "GSM8K: grade-school maths word problems scored on the final answer.",
    ),
    "humaneval": OfficialBenchmark(
        "humaneval", "humaneval", "coding",
        "HumanEval: Python function synthesis, scored by executing the tests.",
    ),
    "ifeval": OfficialBenchmark(
        "ifeval", "ifeval", "instruction",
        "IFEval: verifiable instruction following, strict and loose accuracy.",
    ),
    # --- the classics ------------------------------------------------------
    "mmlu": OfficialBenchmark(
        "mmlu", "mmlu", "knowledge",
        "MMLU: 57-subject multiple-choice knowledge benchmark.",
    ),
    "arc_challenge": OfficialBenchmark(
        "arc_challenge", "arc_challenge", "knowledge",
        "ARC-Challenge: grade-school science questions requiring reasoning.",
    ),
    "hellaswag": OfficialBenchmark(
        "hellaswag", "hellaswag", "commonsense",
        "HellaSwag: commonsense sentence completion.",
    ),
    "truthfulqa": OfficialBenchmark(
        "truthfulqa", "truthfulqa_mc2", "truthfulness",
        "TruthfulQA (MC2): resistance to imitative falsehoods.",
    ),
    "winogrande": OfficialBenchmark(
        "winogrande", "winogrande", "commonsense",
        "WinoGrande: pronoun resolution requiring commonsense.",
    ),
    "gpqa": OfficialBenchmark(
        "gpqa", "gpqa", "reasoning",
        "GPQA: graduate-level science questions written to resist search.",
    ),
    "bbh": OfficialBenchmark(
        "bbh", "bbh", "reasoning",
        "BIG-Bench Hard: tasks where models historically underperformed humans.",
    ),
    "drop": OfficialBenchmark(
        "drop", "drop", "reasoning",
        "DROP: discrete reasoning over paragraphs, needing arithmetic.",
    ),
    "piqa": OfficialBenchmark(
        "piqa", "piqa", "commonsense",
        "PIQA: physical commonsense reasoning.",
    ),
    "mathqa": OfficialBenchmark(
        "mathqa", "mathqa", "math",
        "MathQA: multiple-choice maths word problems.",
    ),
    "mbpp": OfficialBenchmark(
        "mbpp", "mbpp", "coding",
        "MBPP: mostly-basic Python problems, scored by executing the tests.",
    ),
    "lambada_openai": OfficialBenchmark(
        "lambada_openai", "lambada_openai", "language",
        "LAMBADA (OpenAI): last-word prediction requiring broad context.",
    ),
}


def official_benchmarks() -> list[OfficialBenchmark]:
    return [OFFICIAL_BENCHMARKS[name] for name in sorted(OFFICIAL_BENCHMARKS)]


def get_official(name: str) -> OfficialBenchmark | None:
    return OFFICIAL_BENCHMARKS.get(name)


def is_official(name: str) -> bool:
    return name in OFFICIAL_BENCHMARKS


# ---------------------------------------------------------------------------
# Engine discovery
# ---------------------------------------------------------------------------

#: Environment variable naming the interpreter that has the harness installed.
ENGINE_ENV_VAR = "ALECTO_EVAL_PYTHON"


def find_engine(config: AlectoConfig | None = None) -> list[str] | None:
    """Return the command prefix that can run ``lm_eval``, or ``None``.

    Resolution order:

    1. ``config.eval_python`` (set from ``ALECTO_EVAL_PYTHON``)
    2. a ``.venv-eval`` next to the package, for development checkouts
    3. the current interpreter, when the harness happens to be importable there

    An **explicitly configured** interpreter is authoritative: if it does not
    work, no fallback is attempted. Silently running a different harness than
    the one the user pinned would make benchmark results non-reproducible, which
    is the one thing this integration exists to guarantee.
    """
    configured = (getattr(config, "eval_python", None) if config else None) or os.environ.get(
        ENGINE_ENV_VAR
    )
    if configured:
        prefix = [str(configured)]
        return prefix if _engine_works(prefix) else None

    candidates: list[list[str]] = []

    repo_root = Path(__file__).resolve().parent.parent
    for relative in (".venv-eval/bin/python", ".venv-eval/Scripts/python.exe"):
        candidate = repo_root / relative
        if candidate.is_file():
            candidates.append([str(candidate)])

    candidates.append([sys.executable])

    for prefix in candidates:
        if _engine_works(prefix):
            return prefix
    return None


def _engine_works(prefix: list[str]) -> bool:
    try:
        proc = subprocess.run(
            [*prefix, "-c", "import lm_eval; print(lm_eval.__name__)"],
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _require_engine(config: AlectoConfig | None = None) -> list[str]:
    prefix = find_engine(config)
    if prefix is None:
        raise ExternalEngineUnavailable(
            "no evaluation harness found. Official benchmarks need "
            "lm-evaluation-harness in a separate environment; create one with "
            "`python -m venv .venv-eval && .venv-eval/bin/pip install lm-eval`, "
            f"then point {ENGINE_ENV_VAR} at its interpreter. Alecto does not "
            "vendor the harness, and it will not report fixture scores as "
            "official ones."
        )
    return prefix


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_official_benchmark(
    name: str,
    *,
    base_url: str,
    model: str,
    config: AlectoConfig | None = None,
    limit: int | None = None,
    num_fewshot: int | None = None,
    max_gen_toks: int = 2048,
    retries: int = 3,
    timeout: int = 300,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run one official benchmark and return an Alecto quality artifact.

    ``base_url`` is the completion route of an OpenAI-compatible endpoint. The
    result is converted by :mod:`alecto.lmeval_adapter`, so it carries the
    upstream metric id, its standard error and the effective sample count, and
    is marked ``comparable: true``.
    """
    bench = get_official(name)
    if bench is None:
        raise ValueError(
            f"unknown official benchmark {name!r}; known: {', '.join(sorted(OFFICIAL_BENCHMARKS))}"
        )
    prefix = _require_engine(config)

    with tempfile.TemporaryDirectory(prefix="alecto_official_") as out_dir:
        cmd = [
            *prefix, "-m", "lm_eval",
            "--model", "local-chat-completions",
            "--model_args",
            f"base_url={base_url},model={model},max_gen_toks={max_gen_toks},"
            f"num_retries={retries},timeout={timeout}",
            "--tasks", bench.task,
            "--batch_size", "1",
            "--apply_chat_template",
            "--log_samples",
            "--output_path", out_dir,
        ]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        if num_fewshot is not None:
            cmd += ["--num_fewshot", str(num_fewshot)]
        if extra_args:
            cmd += list(extra_args)

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as exc:  # pragma: no cover - engine vanished
            raise ExternalEngineUnavailable(f"could not start the harness: {exc}") from exc

        results = _load_harness_results(out_dir)
        if results is None:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise ExternalEngineUnavailable(
                "the harness produced no results "
                f"(exit {proc.returncode}): {detail[-1][:200] if detail else 'no output'}"
            )

        artifact = _convert(results, out_dir, base_url=base_url, model=model)

    artifact["benchmark"] = bench.name
    artifact["task"] = bench.task
    artifact["engine"] = "lm-evaluation-harness"
    artifact["command"] = " ".join(cmd)
    return artifact


def _load_harness_results(out_dir: str) -> dict[str, Any] | None:
    """Find the harness results JSON, which it nests under a model directory."""
    candidates = sorted(Path(out_dir).rglob("results_*.json"))
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _convert(results: dict[str, Any], out_dir: str, *, base_url: str, model: str) -> dict[str, Any]:
    """Reuse the tested lm-eval converter rather than reimplementing it."""
    from .lmeval_adapter import convert, load_samples

    samples = load_samples(out_dir)
    return convert(results, samples, endpoint=base_url, model=model)


def engine_status(config: AlectoConfig | None = None) -> dict[str, Any]:
    """Report whether official benchmarks are available, for ``list_suites``."""
    prefix = find_engine(config)
    return {
        "available": prefix is not None,
        "command": prefix[0] if prefix else None,
        "env_var": ENGINE_ENV_VAR,
        "benchmarks": len(OFFICIAL_BENCHMARKS),
    }
