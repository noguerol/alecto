"""Seeded, stratified sample selection and frozen manifests (spec §8.3)."""

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any

# Stratification keys per spec §8.3
STRATIFY_SUBJECT = "subject"                 # MMLU-Pro
STRATIFY_INSTRUCTION = "instruction_category"  # IFEval
STRATIFY_BENIGN_UNSAFE = "benign_unsafe"       # refusal suites


@dataclass
class SelectionResult:
    """Stores the selected IDs and their order, not just the seed."""

    selected_ids: list[str]
    seed: int
    stratify_by: str
    strata: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_ids": list(self.selected_ids),
            "seed": self.seed,
            "stratify_by": self.stratify_by,
            "strata": {k: list(v) for k, v in self.strata.items()},
        }


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass
class FrozenManifest:
    """Immutable sample manifest after freeze.

    The hash is the sha256 of the canonical JSON of (ids, order).
    """

    ids: list[str]
    order: list[str]
    hash: str
    frozen: bool = False
    parent_hash: str | None = None
    added_ids: list[str] = field(default_factory=list)

    def _rehash(self) -> None:
        self.hash = hashlib.sha256(
            _canonical_json({"ids": self.ids, "order": self.order}).encode("utf-8")
        ).hexdigest()

    def freeze(self) -> "FrozenManifest":
        if self.frozen:
            return self
        self.frozen = True
        self._rehash()
        return self

    def add(self, item_id: str) -> None:
        """Mutating the manifest after freeze is an error."""
        if self.frozen:
            raise ValueError("FrozenManifest is immutable after freeze; create a child plan instead")
        self.ids.append(item_id)
        self.order.append(item_id)


def select_samples(
    items: list[dict],
    n: int,
    seed: int,
    stratify_by: str,
) -> SelectionResult:
    """Select n items using a seeded, stratified algorithm.

    All randomness is derived deterministically from ``seed``, so the
    same inputs and seed always yield the same selection. Stratifies
    by the given key when the items carry it.
    Returns stable IDs + order, not just the seed.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not items:
        raise ValueError("items must be non-empty")

    # Group items into strata by the stratification key.
    strata: dict[str, list[dict]] = {}
    for item in items:
        key = str(item.get(stratify_by, "unknown"))
        strata.setdefault(key, []).append(item)

    # Allocate quota per stratum proportionally, then top up to n.
    total = len(items)
    allocation: dict[str, int] = {}
    for key, group in strata.items():
        allocation[key] = max(1, round(n * len(group) / total)) if n > 0 else 0

    selected: list[str] = []
    strata_ids: dict[str, list[str]] = {}

    # Deterministic per-stratum seeded selection.
    for key in sorted(strata.keys()):
        group = strata[key]
        quota = allocation.get(key, 0)
        if quota <= 0:
            continue
        # Shuffle a copy with a derived seed for determinism.
        sub_rng = random.Random(f"{seed}:{key}")
        shuffled = list(group)
        sub_rng.shuffle(shuffled)
        chosen = [str(it["id"]) for it in shuffled[: min(quota, len(shuffled))]]
        strata_ids[key] = chosen
        selected.extend(chosen)

    # Top up if proportional allocation undershoots n.
    if len(selected) < n:
        remaining = [str(it["id"]) for it in items if str(it["id"]) not in set(selected)]
        top_rng = random.Random(f"{seed}:topup")
        top_rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    # Final deterministic ordering: stable by stratum then selection order.
    order = list(selected)

    return SelectionResult(
        selected_ids=order,
        seed=seed,
        stratify_by=stratify_by,
        strata=strata_ids,
    )


def freeze_manifest(selection: SelectionResult) -> FrozenManifest:
    """Create and freeze a manifest from a selection result."""
    manifest = FrozenManifest(
        ids=list(selection.selected_ids),
        order=list(selection.selected_ids),
        hash="",
    )
    manifest.freeze()
    return manifest


def make_child(parent: FrozenManifest, added_ids: list[str]) -> FrozenManifest:
    """Create a child plan with a new hash and explicit added IDs.

    Increasing coverage creates a child plan with a new hash and the
    explicit added IDs (spec §8.3).
    """
    existing = set(parent.ids)
    new_ids = [i for i in added_ids if i not in existing]
    child = FrozenManifest(
        ids=list(parent.ids) + new_ids,
        order=list(parent.ids) + new_ids,
        hash="",
        parent_hash=parent.hash,
        added_ids=new_ids,
    )
    child.freeze()
    return child
