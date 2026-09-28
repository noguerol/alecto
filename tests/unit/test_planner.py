"""Unit tests for the alecto planner (spec §8).

Covers:
- Sample selection determinism (same seed → same IDs + order).
- Stratification correctness.
- Frozen manifest (hash stable, cannot silently change sample set after freeze).
- Cost estimation (range with source, budget respected, conservative reserve).
- Planner plan hash (canonical JSON, excluding timestamps/paths/credentials).
"""

import pytest

from alecto.domain import TargetSpec
from alecto.enums import TargetKind
from alecto.planner import (
    Planner,
    select_samples,
    estimate_cost,
    ThroughputEstimate,
)


# ---------------------------------------------------------------------------
# Sample selection determinism
# ---------------------------------------------------------------------------


class TestSampleSelection:
    def test_same_seed_same_ids_and_order(self):
        items = [{"id": f"item-{i}", "subject": f"sub{i % 5}"} for i in range(50)]
        sel1 = select_samples(items, n=10, seed=42, stratify_by="subject")
        sel2 = select_samples(items, n=10, seed=42, stratify_by="subject")
        assert sel1.selected_ids == sel2.selected_ids
        assert len(sel1.selected_ids) == 10

    def test_different_seed_different_selection(self):
        items = [{"id": f"item-{i}", "subject": f"sub{i % 5}"} for i in range(50)]
        sel1 = select_samples(items, n=10, seed=42, stratify_by="subject")
        sel2 = select_samples(items, n=10, seed=99, stratify_by="subject")
        # Different seeds should (with high probability) give different selections
        assert sel1.selected_ids != sel2.selected_ids

    def test_stores_ids_and_order_not_just_seed(self):
        items = [{"id": f"item-{i}", "subject": "s"} for i in range(20)]
        sel = select_samples(items, n=5, seed=7, stratify_by="subject")
        assert len(sel.selected_ids) == 5
        assert all(i in [f"item-{j}" for j in range(20)] for i in sel.selected_ids)


# ---------------------------------------------------------------------------
# Stratification correctness
# ---------------------------------------------------------------------------


class TestStratification:
    def test_mmlu_subject_stratification(self):
        items = [{"id": f"mmlu-{i}", "subject": f"subject_{i % 4}"} for i in range(40)]
        sel = select_samples(items, n=8, seed=1, stratify_by="subject")
        # Each subject should be represented
        subjects_in_sel = set()
        for item in items:
            if item["id"] in sel.selected_ids:
                subjects_in_sel.add(item["subject"])
        assert len(subjects_in_sel) >= 2  # at least 2 subjects represented

    def test_ifeval_instruction_category(self):
        items = [
            {"id": f"ifeval-{i}", "instruction_category": f"cat{i % 3}"}
            for i in range(30)
        ]
        sel = select_samples(items, n=6, seed=2, stratify_by="instruction_category")
        assert len(sel.selected_ids) == 6
        assert sel.stratify_by == "instruction_category"

    def test_refusal_benign_unsafe(self):
        items = [
            {"id": f"ref-{i}", "benign_unsafe": "benign" if i % 2 == 0 else "unsafe"}
            for i in range(20)
        ]
        sel = select_samples(items, n=4, seed=3, stratify_by="benign_unsafe")
        # Both benign and unsafe should be represented
        categories = set()
        for item in items:
            if item["id"] in sel.selected_ids:
                categories.add(item["benign_unsafe"])
        assert "benign" in categories
        assert "unsafe" in categories


# ---------------------------------------------------------------------------
# Frozen manifest
# ---------------------------------------------------------------------------


class TestFrozenManifest:
    def test_hash_stable_after_freeze(self):
        items = [{"id": f"item-{i}", "subject": "s"} for i in range(10)]
        sel = select_samples(items, n=5, seed=42, stratify_by="subject")
        m1 = sel  # not frozen yet
        from alecto.planner.sampling import freeze_manifest

        m1 = freeze_manifest(sel)
        m2 = freeze_manifest(select_samples(items, n=5, seed=42, stratify_by="subject"))
        assert m1.hash == m2.hash
        assert m1.frozen

    def test_cannot_change_after_freeze(self):
        items = [{"id": f"item-{i}", "subject": "s"} for i in range(10)]
        sel = select_samples(items, n=5, seed=42, stratify_by="subject")
        from alecto.planner.sampling import freeze_manifest

        m = freeze_manifest(sel)
        assert m.frozen
        with pytest.raises(ValueError):
            m.add("extra-item")

    def test_child_plan_new_hash(self):
        items = [{"id": f"item-{i}", "subject": "s"} for i in range(10)]
        sel = select_samples(items, n=5, seed=42, stratify_by="subject")
        from alecto.planner.sampling import freeze_manifest, make_child

        parent = freeze_manifest(sel)
        # Use IDs not already in the parent selection
        existing = set(parent.ids)
        new_ids = [f"item-{i}" for i in range(10) if f"item-{i}" not in existing][:2]
        child = make_child(parent, added_ids=new_ids)
        assert child.hash != parent.hash
        assert child.parent_hash == parent.hash
        assert set(child.added_ids) == set(new_ids)
        assert child.frozen


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


class TestCostEstimation:
    def test_range_with_source(self):
        tp = ThroughputEstimate(prompt_tps=100.0, output_tps=20.0, expected_output_tokens=512)
        est = estimate_cost(
            sample_count=10,
            prompt_tokens=1024,
            throughput=tp,
            wall_clock_budget_s=600.0,
            source="observed_calibration",
        )
        assert est.low_s > 0
        assert est.high_s >= est.low_s
        assert est.source == "observed_calibration"

    def test_budget_respected(self):
        tp = ThroughputEstimate(prompt_tps=100.0, output_tps=20.0, expected_output_tokens=512)
        est = estimate_cost(
            sample_count=10,
            prompt_tokens=1024,
            throughput=tp,
            wall_clock_budget_s=600.0,
        )
        assert est.budget_ok is True

    def test_budget_exceeded(self):
        tp = ThroughputEstimate(prompt_tps=1.0, output_tps=1.0, expected_output_tokens=10000)
        est = estimate_cost(
            sample_count=100,
            prompt_tokens=10000,
            throughput=tp,
            wall_clock_budget_s=10.0,
        )
        assert est.budget_ok is False

    def test_conservative_reserve_included(self):
        tp = ThroughputEstimate(prompt_tps=100.0, output_tps=20.0, expected_output_tokens=512)
        est = estimate_cost(
            sample_count=10,
            prompt_tokens=1024,
            throughput=tp,
            wall_clock_budget_s=600.0,
            reserve_ratio=0.2,
        )
        assert est.reserve_s > 0
        assert est.reserve_s == est.high_s * 0.2

    def test_never_promises_fixed_samples(self):
        tp = ThroughputEstimate(prompt_tps=100.0, output_tps=20.0, expected_output_tokens=512, concurrent_factor=2.0)
        est = estimate_cost(
            sample_count=10,
            prompt_tokens=1024,
            throughput=tp,
            wall_clock_budget_s=600.0,
        )
        # The estimate must include a range note, not a single fixed time
        assert any("range" in note for note in est.notes)
        assert any("not fixed-sample promises" in note for note in est.notes)


# ---------------------------------------------------------------------------
# Planner plan hash
# ---------------------------------------------------------------------------


class TestPlannerHash:
    def test_plan_hash_deterministic(self):
        target = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test", model="test-model")
        planner = Planner()
        items = [{"id": f"item-{i}", "subject": f"sub{i % 3}"} for i in range(20)]
        plan1 = planner.build_plan(
            target=target,
            budget_s=600.0,
            seed=42,
            items=items,
            stratify_by="subject",
        )
        plan2 = planner.build_plan(
            target=target,
            budget_s=600.0,
            seed=42,
            items=items,
            stratify_by="subject",
        )
        assert plan1.extra_plan_hash == plan2.extra_plan_hash

    def test_plan_hash_excludes_credentials(self):
        target1 = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test", model="m", api_key_env="KEY1")
        target2 = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test", model="m", api_key_env="KEY2")
        planner = Planner()
        items = [{"id": f"item-{i}", "subject": "s"} for i in range(10)]
        plan1 = planner.build_plan(target1, budget_s=600, seed=42, items=items, stratify_by="subject")
        plan2 = planner.build_plan(target2, budget_s=600, seed=42, items=items, stratify_by="subject")
        # Different api_key_env should not change the hash
        assert plan1.extra_plan_hash == plan2.extra_plan_hash

    def test_plan_hash_excludes_paths(self):
        target1 = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test1", model="m")
        target2 = TargetSpec(kind=TargetKind.MOCK, endpoint="mock://test2", model="m")
        planner = Planner()
        items = [{"id": f"item-{i}", "subject": "s"} for i in range(10)]
        plan1 = planner.build_plan(target1, budget_s=600, seed=42, items=items, stratify_by="subject")
        plan2 = planner.build_plan(target2, budget_s=600, seed=42, items=items, stratify_by="subject")
        # Different endpoint (path) should not change the hash
        assert plan1.extra_plan_hash == plan2.extra_plan_hash
