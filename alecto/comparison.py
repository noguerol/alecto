"""Comparison engine for benchmark results (spec §4.9, §12, §15.2).

Provides:
- PairedDelta: paired differences between two model outputs with seeded
  bootstrap confidence intervals (spec §15.2).
- ConfidenceInterval: Wilson interval for binary rates and paired
  bootstrap intervals (spec §15.2).
- FactorialAnalysis: factorial design for multi-factor comparisons
  (abliteration × quantisation, spec §13.4, AC-17).
- PPLLogitAdapter: perplexity/logit adapter for language model
  comparisons (spec §12.1, AC-15).

This module is stdlib-only.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from .domain import BenchmarkResult, ComparisonResult
from .enums import ComparisonMode

# ---------------------------------------------------------------------------
# PairedDelta — paired differences between two model outputs
# ---------------------------------------------------------------------------

@dataclass
class PairedDelta:
    """Paired difference between two model outputs on matched items.

    Each pair is (item_id, value_a, value_b). The delta is value_a - value_b
    per item. Bootstrap CI on the mean delta uses a seeded RNG with the
    specified number of resamples (default 2000, spec §15.2).
    """

    item_ids: list[str]
    values_a: list[float]
    values_b: list[float]
    seed: int = 1729
    n_resamples: int = 2000
    confidence: float = 0.95

    def __post_init__(self):
        if len(self.item_ids) != len(self.values_a) or len(self.item_ids) != len(self.values_b):
            raise ValueError("item_ids, values_a and values_b must have the same length")
        if len(self.item_ids) == 0:
            raise ValueError("at least one paired observation is required")

    @property
    def deltas(self) -> list[float]:
        """Per-item differences: value_a - value_b."""
        return [a - b for a, b in zip(self.values_a, self.values_b)]

    @property
    def mean_delta(self) -> float:
        """Mean of the paired differences."""
        d = self.deltas
        return sum(d) / len(d)

    @property
    def median_delta(self) -> float:
        """Median of the paired differences."""
        d = sorted(self.deltas)
        n = len(d)
        if n % 2 == 1:
            return d[n // 2]
        return (d[n // 2 - 1] + d[n // 2]) / 2.0

    def bootstrap_ci(self) -> tuple[float, float]:
        """Seeded paired bootstrap confidence interval on the mean delta.

        Resamples item IDs with replacement, computes the mean delta for each
        resample, and returns the (lower, upper) percentile bounds.
        """
        rng = random.Random(self.seed)
        n = len(self.deltas)
        deltas = self.deltas
        means: list[float] = []
        for _ in range(self.n_resamples):
            sample = [deltas[rng.randrange(n)] for _ in range(n)]
            means.append(sum(sample) / n)
        means.sort()
        alpha = 1.0 - self.confidence
        lo_idx = int(math.floor(alpha / 2.0 * self.n_resamples))
        hi_idx = int(math.ceil((1.0 - alpha / 2.0) * self.n_resamples))
        lo_idx = min(lo_idx, self.n_resamples - 1)
        hi_idx = min(hi_idx, self.n_resamples - 1)
        return (means[lo_idx], means[hi_idx])

    def summary(self) -> dict[str, Any]:
        """Structured summary for reporting."""
        ci = self.bootstrap_ci()
        return {
            "n_pairs": len(self.item_ids),
            "mean_delta": self.mean_delta,
            "median_delta": self.median_delta,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "confidence": self.confidence,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# ConfidenceInterval — Wilson interval and paired bootstrap CI
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceInterval:
    """Confidence interval for a binary rate (Wilson) or a paired
    bootstrap mean (spec §15.2).

    Wilson 95% interval: uses the standard Wilson score interval formula.
    Paired bootstrap: delegates to PairedDelta.
    """

    method: str = "wilson"  # "wilson" or "bootstrap"
    n: int = 0
    successes: int = 0
    confidence: float = 0.95
    seed: int = 1729
    n_resamples: int = 2000

    def __post_init__(self):
        if self.n < 0:
            raise ValueError("n must be non-negative")
        if self.successes < 0 or self.successes > self.n:
            raise ValueError("successes must be between 0 and n")
        if self.method not in ("wilson", "bootstrap"):
            raise ValueError(f"unknown method: {self.method}")

    def wilson_bounds(self) -> tuple[float, float]:
        """Wilson score interval for a binary proportion.

        Returns (lower, upper) bounds for the proportion p = successes/n.
        """
        if self.n == 0:
            return (0.0, 0.0)
        z = self._z_score()
        p_hat = self.successes / self.n
        denom = 1.0 + z * z / self.n
        center = (p_hat + z * z / (2.0 * self.n)) / denom
        spread = (z / denom) * math.sqrt(
            p_hat * (1.0 - p_hat) / self.n + z * z / (4.0 * self.n * self.n)
        )
        lo = max(0.0, center - spread)
        hi = min(1.0, center + spread)
        return (lo, hi)

    def paired_bootstrap_ci(self, deltas: list[float]) -> tuple[float, float]:
        """Paired bootstrap CI on the mean of the given deltas."""
        rng = random.Random(self.seed)
        n = len(deltas)
        means: list[float] = []
        for _ in range(self.n_resamples):
            sample = [deltas[rng.randrange(n)] for _ in range(n)]
            means.append(sum(sample) / n)
        means.sort()
        alpha = 1.0 - self.confidence
        lo_idx = int(math.floor(alpha / 2.0 * self.n_resamples))
        hi_idx = int(math.ceil((1.0 - alpha / 2.0) * self.n_resamples))
        lo_idx = min(lo_idx, self.n_resamples - 1)
        hi_idx = min(hi_idx, self.n_resamples - 1)
        return (means[lo_idx], means[hi_idx])

    def _z_score(self) -> float:
        """Approximate the z-score for the given confidence level.

        Uses the Acklam approximation for the inverse normal CDF.
        """
        # Standard normal quantile for common confidence levels
        # 0.95 → 1.959964, 0.90 → 1.644854, 0.99 → 2.575829
        conf = self.confidence
        if abs(conf - 0.95) < 1e-9:
            return 1.959964
        if abs(conf - 0.90) < 1e-9:
            return 1.644854
        if abs(conf - 0.99) < 1e-9:
            return 2.575829
        # General Acklam approximation
        a = -2.795507533082860
        b = 16.15858368388498
        c = -12.12324909777090
        d = -0.313082909153487
        p = (conf + 1.0) / 2.0
        q = 1.0 - p
        # For p in [0.5, 1.0]
        if p >= 0.5:
            r = math.sqrt(-math.log(q))
            return r + (a + b * r + c * r * r + d * r * r * r) / (1.0 + 0.0368347 * r)
        # For p < 0.5 (symmetric)
        r = math.sqrt(-math.log(p))
        return -(r + (a + b * r + c * r * r + d * r * r * r) / (1.0 + 0.0368347 * r))

    def bounds(self) -> tuple[float, float]:
        """Return the interval bounds based on the method."""
        if self.method == "wilson":
            return self.wilson_bounds()
        raise ValueError("paired bootstrap CI requires a delta list; use paired_bootstrap_ci()")

    def summary(self) -> dict[str, Any]:
        """Structured summary for reporting."""
        if self.method == "wilson":
            lo, hi = self.wilson_bounds()
            p_hat = self.successes / self.n if self.n > 0 else 0.0
            return {
                "method": "wilson",
                "n": self.n,
                "successes": self.successes,
                "point_estimate": p_hat,
                "lower": lo,
                "upper": hi,
                "confidence": self.confidence,
            }
        return {
            "method": "bootstrap",
            "confidence": self.confidence,
            "seed": self.seed,
            "n_resamples": self.n_resamples,
        }


# ---------------------------------------------------------------------------
# FactorialAnalysis — factorial design for multi-factor comparisons
# ---------------------------------------------------------------------------

@dataclass
class FactorialAnalysis:
    """Factorial design analysis for multi-factor comparisons.

    Supports the 2×2 abliteration × quantisation design (spec §13.4):
    - O,H: original / high precision
    - A,H: abliterated / high precision
    - O,Q: original / quantised
    - A,Q: abliterated / quantised

    Computes main contrasts and interaction per metric (AC-17).
    """

    # Factor levels: factor name -> list of level labels
    factors: dict[str, list[str]] = field(default_factory=dict)
    # Cell values: (factor_value_tuple) -> metric value
    # e.g. {("original", "high"): 0.85, ("abliterated", "high"): 0.80, ...}
    cells: dict[tuple, float] = field(default_factory=dict)
    # Direction: "higher_is_better" or "lower_is_better"
    direction: str = "higher_is_better"
    # Metric name for reporting
    metric_name: str = ""

    def __post_init__(self):
        if not self.factors:
            raise ValueError("at least one factor is required")
        expected = 1
        for levels in self.factors.values():
            expected *= len(levels)
        if len(self.cells) != expected:
            raise ValueError(
                f"expected {expected} cells for {len(self.factors)} factors, got {len(self.cells)}"
            )
        if self.direction not in ("higher_is_better", "lower_is_better"):
            raise ValueError(f"unknown direction: {self.direction}")

    def cell_keys(self) -> list[tuple]:
        """All expected cell keys."""
        factor_names = list(self.factors.keys())
        levels = [self.factors[name] for name in factor_names]
        keys: list[tuple] = []
        # Generate all combinations
        def _recurse(depth: int, current: tuple):
            if depth == len(factor_names):
                keys.append(current)
                return
            for level in levels[depth]:
                _recurse(depth + 1, current + (level,))
        _recurse(0, ())
        return keys

    def _cell(self, **kwargs: str) -> float:
        """Look up a cell value by factor levels."""
        key = tuple(kwargs[name] for name in self.factors.keys())
        if key not in self.cells:
            raise KeyError(f"missing cell: {key}")
        return self.cells[key]

    def main_contrasts(self) -> dict[str, float]:
        """Compute main effect contrasts for each factor.

        For a 2×2 design with factors F1 and F2:
        - Main effect of F1 = mean(F1=level2) - mean(F1=level1)
        - Main effect of F2 = mean(F2=level2) - mean(F2=level1)

        For the abliteration × quantisation design:
        - abliteration_effect_at_H = m(A,H) - m(O,H)
        - quantisation_effect_on_O = m(O,Q) - m(O,H)
        """
        factor_names = list(self.factors.keys())
        contrasts: dict[str, float] = {}

        if len(factor_names) == 2:
            f1, f2 = factor_names
            levels1 = self.factors[f1]
            levels2 = self.factors[f2]

            # Main effect of f1: average over f2 levels
            val_f1_lo = (
                self._cell(**{f1: levels1[0], f2: levels2[0]})
                + self._cell(**{f1: levels1[0], f2: levels2[1]})
            ) / 2.0
            val_f1_hi = (
                self._cell(**{f1: levels1[1], f2: levels2[0]})
                + self._cell(**{f1: levels1[1], f2: levels2[1]})
            ) / 2.0
            contrasts[f1] = val_f1_hi - val_f1_lo

            # Main effect of f2: average over f1 levels
            val_f2_lo = (
                self._cell(**{f1: levels1[0], f2: levels2[0]})
                + self._cell(**{f1: levels1[1], f2: levels2[0]})
            ) / 2.0
            val_f2_hi = (
                self._cell(**{f1: levels1[0], f2: levels2[1]})
                + self._cell(**{f1: levels1[1], f2: levels2[1]})
            ) / 2.0
            contrasts[f2] = val_f2_hi - val_f2_lo
        else:
            # For 1 factor: simple contrast
            f = factor_names[0]
            levels = self.factors[f]
            contrasts[f] = self._cell(**{f: levels[-1]}) - self._cell(**{f: levels[0]})

        return contrasts

    def interaction(self) -> float:
        """Compute the interaction contrast.

        For a 2×2 design:
        interaction = [m(A,Q) - m(A,H)] - [m(O,Q) - m(O,H)]

        This is the difference in the quantisation effect between the
        abliterated and original models.
        """
        factor_names = list(self.factors.keys())
        if len(factor_names) != 2:
            raise ValueError("interaction requires exactly 2 factors")

        f1, f2 = factor_names
        levels1 = self.factors[f1]
        levels2 = self.factors[f2]

        # Effect of f2 at f1=level1: m(f1=lo, f2=hi) - m(f1=lo, f2=lo)
        effect_f2_at_f1_lo = (
            self._cell(**{f1: levels1[0], f2: levels2[1]})
            - self._cell(**{f1: levels1[0], f2: levels2[0]})
        )
        # Effect of f2 at f1=level2: m(f1=hi, f2=hi) - m(f1=hi, f2=lo)
        effect_f2_at_f1_hi = (
            self._cell(**{f1: levels1[1], f2: levels2[1]})
            - self._cell(**{f1: levels1[1], f2: levels2[0]})
        )

        return effect_f2_at_f1_hi - effect_f2_at_f1_lo

    def contrasts_summary(self) -> dict[str, Any]:
        """Full summary of contrasts for reporting."""
        main = self.main_contrasts()
        interaction = self.interaction() if len(self.factors) == 2 else None
        return {
            "metric": self.metric_name,
            "direction": self.direction,
            "factors": {name: levels for name, levels in self.factors.items()},
            "cells": {str(k): v for k, v in self.cells.items()},
            "main_contrasts": main,
            "interaction": interaction,
        }


# ---------------------------------------------------------------------------
# PPLLogitAdapter — perplexity/logit adapter for language model comparisons
# ---------------------------------------------------------------------------

@dataclass
class PPLLogitAdapter:
    """Perplexity/logit adapter for language model comparisons (spec §12.1).

    NLL = -sum(log p(x_i | x_<i)) / N_scored
    PPL = exp(NLL)

    Uses natural logarithms. Stores token count and NLL even when
    exponentiation would overflow; PPL then has a numeric-limit diagnostic.
    """

    # Per-token log probabilities (natural log)
    log_probs: list[float] = field(default_factory=list)
    # Number of scored tokens
    n_scored: int = 0
    # Numeric limit for PPL (exp overflow threshold)
    ppl_overflow_threshold: float = 709.0  # ln(1e308) ≈ 709

    def __post_init__(self):
        if self.n_scored < 0:
            raise ValueError("n_scored must be non-negative")
        if self.n_scored != len(self.log_probs):
            raise ValueError(
                f"n_scored ({self.n_scored}) must match len(log_probs) ({len(self.log_probs)})"
            )

    def nll(self) -> float:
        """Negative log-likelihood: -sum(log p) / N_scored."""
        if self.n_scored == 0:
            return 0.0
        return -sum(self.log_probs) / self.n_scored

    def ppl(self) -> float | None:
        """Perplexity: exp(NLL). Returns None if overflow would occur."""
        nll = self.nll()
        if nll > self.ppl_overflow_threshold:
            return None
        return math.exp(nll)

    def ppl_with_diagnostic(self) -> tuple[float | None, list[str]]:
        """PPL with numeric-limit diagnostics."""
        diagnostics: list[str] = []
        ppl = self.ppl()
        if ppl is None:
            diagnostics.append(
                "ppl_overflow: NLL exceeds numeric limit; PPL not representable"
            )
        return ppl, diagnostics

    def mean_log_prob(self) -> float:
        """Mean of the per-token log probabilities."""
        if not self.log_probs:
            return 0.0
        return sum(self.log_probs) / len(self.log_probs)

    def top_k_overlap(self, ref_probs: list[float], candidate_probs: list[float], k: int = 5) -> float:
        """Top-k overlap between reference and candidate token distributions.

        Returns the fraction of top-k tokens in the reference that appear
        in the top-k of the candidate.
        """
        if len(ref_probs) != len(candidate_probs):
            raise ValueError("ref_probs and candidate_probs must have the same length")
        if k <= 0:
            return 0.0
        k = min(k, len(ref_probs))

        ref_top = sorted(range(len(ref_probs)), key=lambda i: -ref_probs[i])[:k]
        cand_top = sorted(range(len(candidate_probs)), key=lambda i: -candidate_probs[i])[:k]
        overlap = len(set(ref_top) & set(cand_top))
        return overlap / k

    def kl_divergence(self, ref_probs: list[float], candidate_probs: list[float]) -> float:
        """KL(P_reference || P_candidate).

        KL(P||Q) = sum(p_i * log(p_i / q_i))

        Returns 0.0 if distributions are identical.
        """
        if len(ref_probs) != len(candidate_probs):
            raise ValueError("ref_probs and candidate_probs must have the same length")
        kl = 0.0
        for p, q in zip(ref_probs, candidate_probs):
            if p > 0 and q > 0:
                kl += p * math.log(p / q)
            elif p > 0 and q == 0:
                # p * log(inf) = inf
                return float("inf")
        return kl

    def js_divergence(self, ref_probs: list[float], candidate_probs: list[float]) -> float:
        """Jensen-Shannon divergence between two distributions."""
        m = [(p + q) / 2.0 for p, q in zip(ref_probs, candidate_probs)]
        kl1 = 0.0
        kl2 = 0.0
        for p, q, m_i in zip(ref_probs, candidate_probs, m):
            if p > 0 and m_i > 0:
                kl1 += p * math.log(p / m_i)
            if q > 0 and m_i > 0:
                kl2 += q * math.log(q / m_i)
        return (kl1 + kl2) / 2.0

    def summary(self) -> dict[str, Any]:
        """Structured summary for reporting."""
        ppl, diagnostics = self.ppl_with_diagnostic()
        return {
            "n_scored": self.n_scored,
            "nll": self.nll(),
            "ppl": ppl,
            "mean_log_prob": self.mean_log_prob(),
            "diagnostics": diagnostics,
        }


# ---------------------------------------------------------------------------
# Legacy comparison functions (kept for backward compatibility)
# ---------------------------------------------------------------------------

def compare_pairwise(results_a: list[BenchmarkResult], results_b: list[BenchmarkResult]) -> ComparisonResult:
    """Compare two sets of benchmark results pairwise."""
    winner = None
    margin = None
    notes = []

    total_a = sum(r.score for r in results_a)
    total_b = sum(r.score for r in results_b)

    if total_a > total_b:
        winner = "A"
        margin = total_a - total_b
    elif total_b > total_a:
        winner = "B"
        margin = total_b - total_a
    else:
        winner = None
        margin = 0.0

    notes.append(f"Total A: {total_a:.2f}, Total B: {total_b:.2f}")

    return ComparisonResult(
        mode=ComparisonMode.PAIRWISE,
        results=results_a + results_b,
        winner=winner,
        margin=margin,
        notes=notes,
    )


def compare_group(results: list[BenchmarkResult]) -> ComparisonResult:
    """Compare a group of benchmark results."""
    if not results:
        return ComparisonResult(mode=ComparisonMode.GROUP, results=[])

    max_score = max(r.score for r in results)
    min_score = min(r.score for r in results)
    avg_score = sum(r.score for r in results) / len(results)

    winner = next((r.benchmark for r in results if r.score == max_score), None)
    margin = max_score - min_score

    notes = [
        f"Max: {max_score:.2f}, Min: {min_score:.2f}, Avg: {avg_score:.2f}",
        f"Winner: {winner}",
    ]

    return ComparisonResult(
        mode=ComparisonMode.GROUP,
        results=results,
        winner=winner,
        margin=margin,
        notes=notes,
    )
