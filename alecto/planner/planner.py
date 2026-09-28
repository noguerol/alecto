"""Planner: builds a plan from target + profile + budget, freezes sample
selection, computes cost estimate, and produces a plan hash (canonical
JSON excluding timestamps/paths/credentials). Spec §8.
"""

import hashlib
import json
from typing import Any

from ..domain import Plan, TargetSpec
from .capabilities import CapabilityReport, probe_target
from .cost import CostEstimate, ThroughputEstimate, estimate_cost
from .sampling import SelectionResult, freeze_manifest, select_samples


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _plan_hash_dict(plan: Plan, selection: SelectionResult, cost: CostEstimate) -> dict:
    """Build a canonical dict excluding timestamps, paths, credentials."""
    d: dict[str, Any] = {
        "steps": plan.steps,
        "loop_mode": plan.loop_mode.value,
        "loop_strategy": plan.loop_strategy.value,
        "retry_policy": plan.retry_policy.value,
        "notes": plan.notes,
        "selection": selection.as_dict(),
        "cost": cost.as_dict(),
    }
    if plan.target is not None:
        # Exclude credentials (api_key_env) and paths (endpoint).
        d["target"] = {
            "kind": plan.target.kind.value,
            "model": plan.target.model,
        }
    return d


class Planner:
    """Builds execution plans with frozen sample manifests and cost estimates."""

    def build_plan(
        self,
        target: TargetSpec,
        budget_s: float,
        seed: int,
        items: list[dict],
        stratify_by: str,
        profile: dict | None = None,
        throughput: ThroughputEstimate | None = None,
        capability: CapabilityReport | None = None,
    ) -> Plan:
        profile = profile or {}
        sample_count = int(profile.get("sample_count", len(items)))
        prompt_tokens = int(profile.get("prompt_tokens", 1024))

        # Sample selection (seeded, stratified, stable IDs + order).
        selection = select_samples(items, sample_count, seed, stratify_by)
        manifest = freeze_manifest(selection)

        # Cost estimate (range + source, never fixed-sample promise).
        tp = throughput or ThroughputEstimate(
            prompt_tps=100.0,
            output_tps=20.0,
            expected_output_tokens=512,
        )
        cost = estimate_cost(
            sample_count=sample_count,
            prompt_tokens=prompt_tokens,
            throughput=tp,
            wall_clock_budget_s=budget_s,
        )

        # Build the plan.
        plan = Plan(
            target=target,
            deadline_s=budget_s,
            notes=[f"seed={seed}", f"stratify_by={stratify_by}"],
        )
        plan.steps = [
            {"type": "noop", "name": "plan_build", "selection": selection.as_dict()},
            {"type": "noop", "name": "cost_estimate", "cost": cost.as_dict()},
        ]

        # Attach selection, manifest, cost, capability, and hash.
        plan.extra_selection = selection  # type: ignore[attr-defined]
        plan.extra_manifest = manifest  # type: ignore[attr-defined]
        plan.extra_cost = cost  # type: ignore[attr-defined]
        plan.extra_capability = capability or probe_target(target)  # type: ignore[attr-defined]

        # Plan hash: canonical JSON excluding timestamps/paths/credentials.
        h = hashlib.sha256(
            _canonical_json(_plan_hash_dict(plan, selection, cost)).encode("utf-8")
        ).hexdigest()
        plan.extra_plan_hash = h  # type: ignore[attr-defined]

        return plan
