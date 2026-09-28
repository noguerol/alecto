"""Unit tests for alecto.comparison module (spec §12, §15.2, §13.4, AC-15, AC-17)."""

import math

import pytest

from alecto.comparison import (
    ConfidenceInterval,
    FactorialAnalysis,
    PairedDelta,
    PPLLogitAdapter,
    compare_group,
    compare_pairwise,
)
from alecto.domain import BenchmarkResult
from alecto.enums import BenchmarkCategory, ComparisonMode

# ---------------------------------------------------------------------------
# PairedDelta tests
# ---------------------------------------------------------------------------

class TestPairedDelta:
    def test_basic_deltas(self):
        pd = PairedDelta(
            item_ids=["a", "b", "c"],
            values_a=[10.0, 20.0, 30.0],
            values_b=[5.0, 15.0, 25.0],
        )
        assert pd.deltas == [5.0, 5.0, 5.0]
        assert pd.mean_delta == 5.0
        assert pd.median_delta == 5.0

    def test_mean_and_median(self):
        pd = PairedDelta(
            item_ids=["a", "b", "c", "d"],
            values_a=[10.0, 20.0, 30.0, 40.0],
            values_b=[0.0, 0.0, 0.0, 0.0],
        )
        assert pd.mean_delta == 25.0
        assert pd.median_delta == 25.0

    def test_bootstrap_ci_deterministic(self):
        pd1 = PairedDelta(
            item_ids=["a", "b", "c", "d", "e"],
            values_a=[1.0, 2.0, 3.0, 4.0, 5.0],
            values_b=[0.5, 1.5, 2.5, 3.5, 4.5],
            seed=42,
            n_resamples=1000,
        )
        pd2 = PairedDelta(
            item_ids=["a", "b", "c", "d", "e"],
            values_a=[1.0, 2.0, 3.0, 4.0, 5.0],
            values_b=[0.5, 1.5, 2.5, 3.5, 4.5],
            seed=42,
            n_resamples=1000,
        )
        ci1 = pd1.bootstrap_ci()
        ci2 = pd2.bootstrap_ci()
        assert ci1 == ci2  # deterministic with same seed

    def test_bootstrap_ci_contains_mean(self):
        pd = PairedDelta(
            item_ids=[f"item_{i}" for i in range(50)],
            values_a=[float(i) for i in range(50)],
            values_b=[float(i) - 2.0 for i in range(50)],
            seed=123,
            n_resamples=2000,
        )
        lo, hi = pd.bootstrap_ci()
        assert lo <= pd.mean_delta <= hi

    def test_bootstrap_ci_negative_deltas(self):
        pd = PairedDelta(
            item_ids=["a", "b", "c"],
            values_a=[1.0, 2.0, 3.0],
            values_b=[2.0, 3.0, 4.0],
            seed=99,
            n_resamples=500,
        )
        assert pd.mean_delta == -1.0
        lo, hi = pd.bootstrap_ci()
        assert lo < 0 < hi or lo <= -1.0

    def test_summary(self):
        pd = PairedDelta(
            item_ids=["a", "b"],
            values_a=[10.0, 20.0],
            values_b=[5.0, 15.0],
            seed=1,
            n_resamples=100,
        )
        s = pd.summary()
        assert s["n_pairs"] == 2
        assert s["mean_delta"] == 5.0
        assert s["confidence"] == 0.95
        assert s["seed"] == 1
        assert s["n_resamples"] == 100
        assert s["ci_lower"] <= s["ci_upper"]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            PairedDelta(
                item_ids=["a", "b"],
                values_a=[1.0],
                values_b=[1.0, 2.0],
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            PairedDelta(item_ids=[], values_a=[], values_b=[])


# ---------------------------------------------------------------------------
# ConfidenceInterval tests
# ---------------------------------------------------------------------------

class TestConfidenceInterval:
    def test_wilson_basic(self):
        ci = ConfidenceInterval(method="wilson", n=100, successes=50)
        lo, hi = ci.wilson_bounds()
        assert 0.0 <= lo <= 0.5 <= hi <= 1.0
        # Wilson interval for 50/100 should be roughly [0.40, 0.60]
        assert lo < 0.45
        assert hi > 0.55

    def test_wilson_all_successes(self):
        ci = ConfidenceInterval(method="wilson", n=10, successes=10)
        lo, hi = ci.wilson_bounds()
        assert lo > 0.0
        assert hi > 0.99

    def test_wilson_no_successes(self):
        ci = ConfidenceInterval(method="wilson", n=10, successes=0)
        lo, hi = ci.wilson_bounds()
        assert lo == 0.0
        assert hi < 1.0

    def test_wilson_zero_n(self):
        ci = ConfidenceInterval(method="wilson", n=0, successes=0)
        lo, hi = ci.wilson_bounds()
        assert lo == 0.0
        assert hi == 0.0

    def test_wilson_small_sample(self):
        ci = ConfidenceInterval(method="wilson", n=5, successes=1)
        lo, hi = ci.wilson_bounds()
        assert 0.0 <= lo <= hi <= 1.0

    def test_bootstrap_ci(self):
        ci = ConfidenceInterval(
            method="bootstrap",
            n=50,
            successes=25,
            seed=42,
            n_resamples=1000,
        )
        deltas = [float(i) - 10.0 for i in range(50)]
        lo, hi = ci.paired_bootstrap_ci(deltas)
        mean_d = sum(deltas) / len(deltas)
        assert lo <= mean_d <= hi

    def test_bounds_method_dispatch(self):
        ci = ConfidenceInterval(method="wilson", n=100, successes=50)
        lo, hi = ci.bounds()
        assert 0.0 <= lo <= hi <= 1.0

        ci2 = ConfidenceInterval(method="bootstrap", n=10, successes=5)
        with pytest.raises(ValueError):
            ci2.bounds()  # bootstrap requires delta list

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            ConfidenceInterval(method="unknown", n=10, successes=5)

    def test_invalid_successes(self):
        with pytest.raises(ValueError):
            ConfidenceInterval(method="wilson", n=10, successes=15)

    def test_summary_wilson(self):
        ci = ConfidenceInterval(method="wilson", n=100, successes=50)
        s = ci.summary()
        assert s["method"] == "wilson"
        assert s["n"] == 100
        assert s["successes"] == 50
        assert s["point_estimate"] == 0.5
        assert s["lower"] <= s["upper"]

    def test_z_score_95(self):
        ci = ConfidenceInterval(method="wilson", n=100, successes=50, confidence=0.95)
        assert abs(ci._z_score() - 1.959964) < 1e-6

    def test_z_score_99(self):
        ci = ConfidenceInterval(method="wilson", n=100, successes=50, confidence=0.99)
        assert abs(ci._z_score() - 2.575829) < 1e-6


# ---------------------------------------------------------------------------
# FactorialAnalysis tests (AC-17)
# ---------------------------------------------------------------------------

class TestFactorialAnalysis:
    def test_2x2_main_contrasts(self):
        """Test the abliteration × quantisation design (spec §13.4)."""
        fa = FactorialAnalysis(
            factors={
                "transformation": ["original", "abliterated"],
                "precision": ["high", "q4"],
            },
            cells={
                ("original", "high"): 0.90,
                ("abliterated", "high"): 0.85,
                ("original", "q4"): 0.80,
                ("abliterated", "q4"): 0.70,
            },
            metric_name="accuracy",
        )
        contrasts = fa.main_contrasts()
        # Main effect of transformation: mean(abliterated) - mean(original)
        # mean(original) = (0.90 + 0.80) / 2 = 0.85
        # mean(abliterated) = (0.85 + 0.70) / 2 = 0.775
        assert abs(contrasts["transformation"] - (0.775 - 0.85)) < 1e-10
        # Main effect of precision: mean(q4) - mean(high)
        # mean(high) = (0.90 + 0.85) / 2 = 0.875
        # mean(q4) = (0.80 + 0.70) / 2 = 0.75
        assert abs(contrasts["precision"] - (0.75 - 0.875)) < 1e-10

    def test_interaction(self):
        """Test the interaction contrast (AC-17)."""
        fa = FactorialAnalysis(
            factors={
                "transformation": ["original", "abliterated"],
                "precision": ["high", "q4"],
            },
            cells={
                ("original", "high"): 0.90,
                ("abliterated", "high"): 0.85,
                ("original", "q4"): 0.80,
                ("abliterated", "q4"): 0.70,
            },
            metric_name="accuracy",
        )
        # interaction = [m(A,Q) - m(A,H)] - [m(O,Q) - m(O,H)]
        # = (0.70 - 0.85) - (0.80 - 0.90)
        # = -0.15 - (-0.10) = -0.05
        assert abs(fa.interaction() - (-0.05)) < 1e-10

    def test_interaction_known_value(self):
        """Synthetic data with a known interaction (AC-17)."""
        fa = FactorialAnalysis(
            factors={
                "factor_a": ["low", "high"],
                "factor_b": ["low", "high"],
            },
            cells={
                ("low", "low"): 10.0,
                ("high", "low"): 12.0,
                ("low", "high"): 13.0,
                ("high", "high"): 18.0,
            },
            metric_name="score",
        )
        # interaction = [m(hi,hi) - m(hi,lo)] - [m(lo,hi) - m(lo,lo)]
        # = (18 - 12) - (13 - 10) = 6 - 3 = 3
        assert abs(fa.interaction() - 3.0) < 1e-10

    def test_contrasts_summary(self):
        fa = FactorialAnalysis(
            factors={
                "transformation": ["original", "abliterated"],
                "precision": ["high", "q4"],
            },
            cells={
                ("original", "high"): 0.90,
                ("abliterated", "high"): 0.85,
                ("original", "q4"): 0.80,
                ("abliterated", "q4"): 0.70,
            },
            metric_name="accuracy",
            direction="higher_is_better",
        )
        s = fa.contrasts_summary()
        assert s["metric"] == "accuracy"
        assert s["direction"] == "higher_is_better"
        assert s["interaction"] is not None
        assert "main_contrasts" in s
        assert len(s["cells"]) == 4

    def test_single_factor(self):
        fa = FactorialAnalysis(
            factors={"precision": ["high", "q4"]},
            cells={("high",): 0.90, ("q4",): 0.80},
            metric_name="accuracy",
        )
        contrasts = fa.main_contrasts()
        assert abs(contrasts["precision"] - (-0.10)) < 1e-10
        # No interaction for single factor
        with pytest.raises(ValueError):
            fa.interaction()

    def test_missing_cell_raises(self):
        with pytest.raises(ValueError):
            FactorialAnalysis(
                factors={
                    "transformation": ["original", "abliterated"],
                    "precision": ["high", "q4"],
                },
                cells={
                    ("original", "high"): 0.90,
                    ("abliterated", "high"): 0.85,
                    ("original", "q4"): 0.80,
                    # missing ("abliterated", "q4")
                },
            )

    def test_no_factors_raises(self):
        with pytest.raises(ValueError):
            FactorialAnalysis(factors={}, cells={})

    def test_invalid_direction(self):
        with pytest.raises(ValueError):
            FactorialAnalysis(
                factors={"f": ["a", "b"]},
                cells={("a",): 1.0, ("b",): 2.0},
                direction="unknown",
            )


# ---------------------------------------------------------------------------
# PPLLogitAdapter tests (AC-15)
# ---------------------------------------------------------------------------

class TestPPLLogitAdapter:
    def test_nll_basic(self):
        """NLL = -sum(log p) / N."""
        adapter = PPLLogitAdapter(
            log_probs=[math.log(0.5), math.log(0.5), math.log(0.5)],
            n_scored=3,
        )
        # NLL = -(-0.6931 - 0.6931 - 0.6931) / 3 = 0.6931
        assert abs(adapter.nll() - math.log(2)) < 1e-10

    def test_ppl_basic(self):
        adapter = PPLLogitAdapter(
            log_probs=[math.log(0.5)] * 4,
            n_scored=4,
        )
        # PPL = exp(log(2)) = 2
        assert abs(adapter.ppl() - 2.0) < 1e-10

    def test_ppl_overflow(self):
        """PPL overflow produces None + diagnostic (AC-15)."""
        # NLL = 800 > 709 threshold
        adapter = PPLLogitAdapter(
            log_probs=[-800.0],
            n_scored=1,
        )
        ppl = adapter.ppl()
        assert ppl is None
        ppl, diagnostics = adapter.ppl_with_diagnostic()
        assert ppl is None
        assert any("overflow" in d for d in diagnostics)

    def test_ppl_no_overflow(self):
        adapter = PPLLogitAdapter(
            log_probs=[-1.0, -1.0, -1.0],
            n_scored=3,
        )
        ppl = adapter.ppl()
        assert ppl is not None
        assert abs(ppl - math.e) < 1e-10

    def test_mean_log_prob(self):
        adapter = PPLLogitAdapter(
            log_probs=[-1.0, -2.0, -3.0],
            n_scored=3,
        )
        assert abs(adapter.mean_log_prob() - (-2.0)) < 1e-10

    def test_top_k_overlap(self):
        ref = [0.1, 0.3, 0.5, 0.1]
        cand = [0.4, 0.2, 0.3, 0.1]
        adapter = PPLLogitAdapter(log_probs=[-1.0], n_scored=1)
        # Top-2 of ref: indices 2 (0.5), 1 (0.3)
        # Top-2 of cand: indices 0 (0.4), 2 (0.3)
        overlap = adapter.top_k_overlap(ref, cand, k=2)
        assert overlap == 0.5  # 1 of 2 overlap (index 2)

    def test_top_k_overlap_identical(self):
        ref = [0.1, 0.3, 0.5, 0.1]
        adapter = PPLLogitAdapter(log_probs=[-1.0], n_scored=1)
        overlap = adapter.top_k_overlap(ref, ref, k=3)
        assert overlap == 1.0

    def test_kl_divergence_identical(self):
        probs = [0.25, 0.25, 0.25, 0.25]
        adapter = PPLLogitAdapter(log_probs=[-1.0], n_scored=1)
        assert abs(adapter.kl_divergence(probs, probs)) < 1e-10

    def test_kl_divergence_different(self):
        p = [0.5, 0.5, 0.0, 0.0]
        q = [0.25, 0.25, 0.25, 0.25]
        adapter = PPLLogitAdapter(log_probs=[-1.0], n_scored=1)
        kl = adapter.kl_divergence(p, q)
        # KL(P||Q) = 0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.25) = 0.5*ln(2) + 0.5*ln(2) = ln(2)
        assert abs(kl - math.log(2)) < 1e-10

    def test_kl_divergence_zero_q(self):
        p = [0.5, 0.5]
        q = [1.0, 0.0]
        adapter = PPLLogitAdapter(log_probs=[-1.0], n_scored=1)
        kl = adapter.kl_divergence(p, q)
        assert kl == float("inf")

    def test_js_divergence(self):
        p = [1.0, 0.0]
        q = [0.0, 1.0]
        adapter = PPLLogitAdapter(log_probs=[-1.0], n_scored=1)
        js = adapter.js_divergence(p, q)
        # M = [0.5, 0.5]
        # KL(P||M) = 1*ln(1/0.5) = ln(2)
        # KL(Q||M) = 1*ln(1/0.5) = ln(2)
        # JS = (ln(2) + ln(2)) / 2 = ln(2)
        assert abs(js - math.log(2)) < 1e-10

    def test_summary(self):
        adapter = PPLLogitAdapter(
            log_probs=[-1.0, -1.0, -1.0],
            n_scored=3,
        )
        s = adapter.summary()
        assert s["n_scored"] == 3
        assert abs(s["nll"] - 1.0) < 1e-10
        assert s["ppl"] is not None
        assert abs(s["ppl"] - math.e) < 1e-10
        assert s["diagnostics"] == []

    def test_empty_log_probs(self):
        adapter = PPLLogitAdapter(log_probs=[], n_scored=0)
        assert adapter.nll() == 0.0
        assert adapter.ppl() == 1.0  # exp(0) = 1
        assert adapter.mean_log_prob() == 0.0

    def test_n_scored_mismatch(self):
        with pytest.raises(ValueError):
            PPLLogitAdapter(log_probs=[-1.0, -2.0], n_scored=0)


# ---------------------------------------------------------------------------
# Legacy comparison functions (backward compatibility)
# ---------------------------------------------------------------------------

class TestComparePairwise:
    def test_a_wins(self):
        results_a = [make_result("math", 8.0)]
        results_b = [make_result("math", 5.0)]
        result = compare_pairwise(results_a, results_b)
        assert result.winner == "A"
        assert result.margin == 3.0
        assert result.mode == ComparisonMode.PAIRWISE

    def test_b_wins(self):
        results_a = [make_result("math", 5.0)]
        results_b = [make_result("math", 8.0)]
        result = compare_pairwise(results_a, results_b)
        assert result.winner == "B"
        assert result.margin == 3.0

    def test_tie(self):
        results_a = [make_result("math", 5.0)]
        results_b = [make_result("math", 5.0)]
        result = compare_pairwise(results_a, results_b)
        assert result.winner is None
        assert result.margin == 0.0


class TestCompareGroup:
    def test_basic(self):
        results = [
            make_result("math", 8.0),
            make_result("coding", 5.0),
            make_result("reasoning", 10.0),
        ]
        result = compare_group(results)
        assert result.winner == "reasoning"
        assert result.mode == ComparisonMode.GROUP
        assert result.margin == 5.0

    def test_empty(self):
        result = compare_group([])
        assert result.winner is None
        assert result.results == []


def make_result(benchmark: str, score: float) -> BenchmarkResult:
    return BenchmarkResult(
        task_id="task-1",
        benchmark=benchmark,
        category=BenchmarkCategory.MATH,
        score=score,
        max_score=10.0,
        duration_ms=100,
        pass_=score >= 5.0,
    )
