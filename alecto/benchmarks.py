"""Benchmark execution and scoring (spec §4.8)."""

import time
from typing import Any

from .domain import BenchmarkResult, Evidence, TargetSpec, Task
from .enums import BenchmarkCategory, EvidenceKind


class Benchmark:
    """Base benchmark class."""

    name: str = "base"
    category: BenchmarkCategory = BenchmarkCategory.GENERAL
    max_score: float = 1.0

    async def execute(self, task: Task, target: TargetSpec) -> BenchmarkResult:
        start = time.monotonic()
        from .adapters import get_adapter
        adapter = get_adapter(target)
        step: dict[str, Any] = {"messages": [{"role": "user", "content": self.prompt}]}
        if "temperature" in target.extra:
            step["temperature"] = target.extra["temperature"]
        if "max_tokens" in target.extra:
            step["max_tokens"] = target.extra["max_tokens"]
        try:
            response = await adapter.call(step)
        finally:
            # Close the adapter after use
            if hasattr(adapter, "aclose"):
                await adapter.aclose()
            elif hasattr(adapter, "close"):
                adapter.close()
        duration_ms = int((time.monotonic() - start) * 1000)
        score = self.score(response)
        return BenchmarkResult(
            task_id=task.id,
            benchmark=self.name,
            category=self.category,
            score=score,
            max_score=self.max_score,
            duration_ms=duration_ms,
            pass_=score >= self.pass_threshold,
            details={"response": response.get("text", "")},
            evidence=[Evidence(kind=EvidenceKind.TEXT, text=response.get("text", ""))],
        )

    @property
    def prompt(self) -> str:
        return "Base prompt"

    @property
    def pass_threshold(self) -> float:
        return self.max_score * 0.5

    def score(self, response: dict[str, Any]) -> float:
        return self.max_score


_NUMBER_WORDS = {
    "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
    "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0,
    "ten": 10.0,
}


def _parse_numeric(text: str) -> float | None:
    """Parse a numeric response: try float() first, then common number words."""
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    return _NUMBER_WORDS.get(text.lower())


class MathBenchmark(Benchmark):
    name = "math_basic"
    category = BenchmarkCategory.MATH
    max_score = 10.0

    @property
    def prompt(self) -> str:
        return "What is 2 + 2? Answer with just the number."

    def score(self, response: dict[str, Any]) -> float:
        text = response.get("text", "")
        value = _parse_numeric(text)
        if value is not None and value == 4:
            return self.max_score
        return 0.0


class CodingBenchmark(Benchmark):
    name = "coding_basic"
    category = BenchmarkCategory.CODING
    max_score = 10.0

    @property
    def prompt(self) -> str:
        return 'Write a Python function "add(a, b)" that returns the sum. Only the function.'

    def score(self, response: dict[str, Any]) -> float:
        text = response.get("text", "")
        if "def add" in text and "return" in text:
            return self.max_score
        return 0.0


class ReasoningBenchmark(Benchmark):
    name = "reasoning_basic"
    category = BenchmarkCategory.REASONING
    max_score = 10.0

    @property
    def prompt(self) -> str:
        return "If all bloops are rals, and all rals are things, are all bloops things? Yes or no."

    def score(self, response: dict[str, Any]) -> float:
        text = response.get("text", "").strip().lower()
        if "yes" in text:
            return self.max_score
        return 0.0


class LanguageBenchmark(Benchmark):
    name = "language_basic"
    category = BenchmarkCategory.LANGUAGE
    max_score = 10.0

    @property
    def prompt(self) -> str:
        return "Translate 'hello world' to Spanish."

    def score(self, response: dict[str, Any]) -> float:
        text = response.get("text", "").lower()
        if "hola" in text:
            return self.max_score
        return 0.0


BENCHMARKS: dict[str, type[Benchmark]] = {
    "math_basic": MathBenchmark,
    "coding_basic": CodingBenchmark,
    "reasoning_basic": ReasoningBenchmark,
    "language_basic": LanguageBenchmark,
}


async def run_benchmark(name: str, task: Task, target: TargetSpec) -> BenchmarkResult:
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}")
    bench = BENCHMARKS[name]()
    return await bench.execute(task, target)
