from __future__ import annotations

from copy import deepcopy
from typing import Any


def normalize_experiment_action(
    decision: dict[str, Any], experiment_policy: str, remaining_patch_count: int
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply an explicit ablation policy without changing the default production loop."""
    if experiment_policy != "no_zoom":
        return decision, None
    action = decision.get("next_action") or {}
    if action.get("type") != "zoom":
        return decision, None
    original = deepcopy(action)
    targets = action.get("target_patches") if isinstance(action.get("target_patches"), list) else []
    replacement = "inspect" if targets else "retrieve" if remaining_patch_count > 0 else "abstain"
    action.update(
        {
            "type": replacement,
            "target_patches": targets if replacement == "inspect" else [],
            "magnification": None,
        }
    )
    if replacement == "retrieve" and not action.get("query"):
        missing = decision.get(
            "executor_missing_evidence", decision.get("missing_evidence")
        )
        if isinstance(missing, list):
            missing = "; ".join(str(value) for value in missing if value)
        action["query"] = missing or "additional relevant morphology"
    decision["next_action"] = action
    repair = {
        "reason": "no_zoom_experiment_policy",
        "original_action": original,
        "normalized_action": deepcopy(action),
        "remaining_patch_count": remaining_patch_count,
    }
    decision["experiment_policy_repair"] = repair
    return decision, repair


def initial_retrieval_count(
    available_count: int, initial_sample_ratio: float, initial_top_k: int | None
) -> int:
    if available_count < 1:
        return 0
    if initial_top_k is not None:
        if initial_top_k < 1:
            raise ValueError("--initial_top_k must be at least 1 when provided")
        return min(available_count, initial_top_k)
    return max(1, int(available_count * initial_sample_ratio))


def replenishment_count(
    available_count: int, replenish_ratio: float, replenish_top_k: int | None
) -> int:
    if available_count < 1:
        return 0
    if replenish_top_k is not None:
        if replenish_top_k < 1:
            raise ValueError("--replenish_top_k must be at least 1 when provided")
        return min(available_count, replenish_top_k)
    return max(1, int(available_count * replenish_ratio))
