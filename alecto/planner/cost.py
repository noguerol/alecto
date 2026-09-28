"""Cost estimation: wall-clock budget as input constraint (spec §8.1).

Estimates MUST include ranges (low/high) and their source.
Never promise fixed samples in fixed time.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThroughputEstimate:
    prompt_tps: float
    output_tps: float
    expected_output_tokens: int
    concurrent_factor: float = 1.0

    def __post_init__(self):
        if self.prompt_tps <= 0 or self.output_tps <= 0:
            raise ValueError("throughputs must be positive")
        if self.expected_output_tokens < 0:
            raise ValueError("expected_output_tokens must be non-negative")
        if self.concurrent_factor <= 0:
            raise ValueError("concurrent_factor must be positive")


@dataclass
class CostEstimate:
    low_s: float
    high_s: float
    reserve_s: float
    source: str  # 'observed_calibration' or 'assumed'
    budget_ok: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "low_s": self.low_s,
            "high_s": self.high_s,
            "reserve_s": self.reserve_s,
            "source": self.source,
            "budget_ok": self.budget_ok,
            "notes": list(self.notes),
        }


def estimate_cost(
    sample_count: int,
    prompt_tokens: int,
    throughput: ThroughputEstimate,
    wall_clock_budget_s: float,
    reserve_ratio: float = 0.2,
    source: str = "assumed",
) -> CostEstimate:
    """Estimate wall-clock cost for sample_count samples.

    Uses observed prompt/output throughput when calibrated, expected
    output lengths, concurrent throughput, and a conservative reserve.
    Returns a range (low/high) with source. Never promises fixed
    samples in fixed time.
    """
    if source not in ("observed_calibration", "assumed"):
        raise ValueError("source must be 'observed_calibration' or 'assumed'")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if wall_clock_budget_s <= 0:
        raise ValueError("wall_clock_budget_s must be positive")

    # Per-sample time: prompt prefill + output generation.
    prompt_time = prompt_tokens / throughput.prompt_tps
    output_time = throughput.expected_output_tokens / throughput.output_tps
    per_sample = prompt_time + output_time

    # Concurrent throughput: divide by concurrent factor.
    effective = per_sample / throughput.concurrent_factor

    # Range: low = ideal concurrent, high = single-stream (conservative).
    low_s = sample_count * effective
    high_s = sample_count * per_sample

    # Conservative reserve.
    reserve_s = high_s * reserve_ratio
    total_high = high_s + reserve_s

    budget_ok = total_high <= wall_clock_budget_s

    notes = [
        f"range: {low_s:.1f}s–{high_s:.1f}s (+{reserve_s:.1f}s reserve)",
        f"source: {source}",
        "estimates are ranges, not fixed-sample promises",
    ]

    return CostEstimate(
        low_s=low_s,
        high_s=high_s,
        reserve_s=reserve_s,
        source=source,
        budget_ok=budget_ok,
        notes=notes,
    )
