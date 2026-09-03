from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .common import ContractViolation, sha256_json
from .s7 import bootstrap_mean_ci, cluster_bootstrap_mean_ci, preference_score


BOOTSTRAP_SEED = 20260827
BOOTSTRAP_REPLICATES = 10000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolation(message)


def adapter_enabled(*, task: str, magnification: str) -> bool:
    """Frozen v1 router predicate; S8 audits it but does not deploy it."""
    return task == "schema_morphology_v1" and magnification == "5x"


def validate_s8_frozen_contract(contract: Mapping[str, Any]) -> None:
    require(contract.get("schema_version") == "patho_lora_s8_frozen_input_contract_v1", "S8 contract schema changed")
    require(contract.get("status") == "frozen_waiting_explicit_s8_execution_authorization", "S8 contract status changed")
    require(contract.get("selected_rank") == "r16" and contract.get("selected_epoch") == 1, "S8 selected checkpoint changed")
    require(contract.get("selected_adapter_relpath") == "s7/formal_v2_runtime_resume/r16/epoch1/adapter", "S8 adapter path changed")
    require(contract.get("s8_authorized") is False and contract.get("s10_authorized") is False, "Frozen S8/S10 boundary changed")
    require(contract["locked_dataset"] == {
        "content_or_target_reads_during_freeze": 0,
        "dataset_controls": 5,
        "evaluation_control_total": 10,
        "real_records": 95,
        "record_count": 100,
        "sha256_metadata_only": "a281aae7243888ddbd2868039446d8d028c92448662e0dad4d43cc879331bbbd",
        "supplemental_controls": 5,
    }, "S8 Locked composition changed")
    require(contract["local_evaluation"] == {
        "arms": ["B1", "B2", "T1", "T2"],
        "blank_exact_empty_required": 10,
        "deployed_schema_required": 100,
        "max_new_tokens": 192,
        "native_and_deployed_metrics_separate": True,
        "native_schema_minimum": 99,
        "same_images_prompt_max_tokens_greedy_parser": True,
    }, "S8 local evaluation contract changed")
    require(contract["paired_calls"] == {
        "estimated_max_cost_per_call_usd": 0.005,
        "maximum_billable_completions": 380,
        "maximum_stage_cost_usd": 1.9,
        "project_cap_usd": 9.5,
        "project_spend_before_s8_usd": 4.3366178500000085,
        "projected_project_spend_ceiling_usd": 6.236617850000009,
    }, "S8 paired-call or budget contract changed")
    require(contract["paired_evaluation"]["max_tokens"] == 1536, "S8 judge max_tokens changed")
    require(contract["paired_evaluation"]["orders_per_judge"] == ["base_then_lora", "lora_then_base"], "S8 order contract changed")
    require(contract["human_review"] == {
        "anonymous_repeats": 2,
        "automatic_scores_hidden": True,
        "model_identity_hidden": True,
        "required_before_final_locked_gate": True,
        "unique_locked_images": 20,
    }, "S8 human-review contract changed")
    require(contract["one_shot_rules"] == {
        "failed_adapter_deploy_allowed": False,
        "failure_preserved_verbatim": True,
        "locked_evaluations_allowed": 1,
        "modify_then_retry_on_same_locked_allowed": False,
        "test10_after_locked_failure_allowed": False,
    }, "S8 one-shot rules changed")
    require(contract["privacy"] == {
        "coordinate_patch_only": True,
        "full_wsi_reads_or_uploads": 0,
        "pii_uploads": 0,
        "request_or_base64_persistence": False,
    }, "S8 privacy contract changed")
    require(contract["router_and_rollback"] == {
        "adapter_merge_allowed": False,
        "adapter_on_only_for_magnification": "5x",
        "adapter_on_only_for_task": "schema_morphology_v1",
        "magnification_10x_default": "adapter_off",
        "magnification_20x_default": "adapter_off",
        "native_and_deployed_metrics_separate": True,
    }, "S8 router/rollback contract changed")


def summarize_locked_pairing(
    *,
    rows: Sequence[Mapping[str, str]],
    clusters: Mapping[str, str],
    outcomes: Mapping[str, Mapping[str, Any]],
    local_base_summary: Mapping[str, Any],
    local_treatment_summary: Mapping[str, Any],
    rollback_audit: Mapping[str, Any],
) -> dict[str, Any]:
    require(len(rows) == 95 and len({row["sample_id"] for row in rows}) == 95, "S8 requires 95 unique real Locked rows")
    require(set(clusters) == {row["sample_id"] for row in rows}, "S8 Locked cluster mapping is incomplete")
    patch_scores: list[float] = []
    cluster_scores: dict[str, list[float]] = defaultdict(list)
    claim_counts = {"base": Counter(), "treatment": Counter()}
    forbidden_counts = {"base": 0, "treatment": 0}
    calls_by_model: Counter[str] = Counter()
    flagged_samples: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        scores: list[float] = []
        for judge_name in ("gemini", "luna"):
            for order in ("base_then_lora", "lora_then_base"):
                task_id = f"s8_locked_{sample_id}_{judge_name}_{order}"
                record = outcomes[task_id]
                parsed = record["parsed_pair_judge"]
                treatment_position = "B" if order == "base_then_lora" else "A"
                scores.append(preference_score(parsed["preference"], treatment_position=treatment_position))
                roles = {"candidate_a": "base", "candidate_b": "treatment"} if order == "base_then_lora" else {"candidate_a": "treatment", "candidate_b": "base"}
                for candidate_key, role in roles.items():
                    candidate = parsed[candidate_key]
                    claim_counts[role].update(claim["status"] for claim in candidate["claims"])
                    forbidden = candidate["forbidden_inference"] is True
                    forbidden_counts[role] += int(forbidden)
                    if forbidden and role == "treatment":
                        flagged_samples.add(sample_id)
                calls_by_model[str(record["model"])] += 1
        require(len(scores) == 4, "S8 patch lacks four paired-judge observations")
        score = sum(scores) / 4.0
        patch_scores.append(score)
        cluster_scores[clusters[sample_id]].append(score)

    def precision(role: str) -> float | None:
        denominator = sum(claim_counts[role].values())
        return claim_counts[role]["supported"] / denominator if denominator else None

    base_precision = precision("base")
    treatment_precision = precision("treatment")
    patch_ci = bootstrap_mean_ci(patch_scores, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES)
    cluster_ci = cluster_bootstrap_mean_ci(cluster_scores, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES)
    gates = {
        "native_schema_at_least_99_of_100": int(local_treatment_summary["native_schema_valid_dataset"]) >= 99,
        "deployed_schema_100_of_100": int(local_treatment_summary["deployed_schema_valid_dataset"]) == 100,
        "native_blank_exact_empty_10_of_10": int(local_treatment_summary["native_control_exact_empty"]) == 10,
        "deployed_blank_exact_empty_10_of_10": int(local_treatment_summary["deployed_control_exact_empty"]) == 10,
        "treatment_claim_precision_defined": treatment_precision is not None,
        "treatment_claim_precision_at_least_95_percent": treatment_precision is not None and treatment_precision >= 0.95,
        "claim_precision_drop_vs_base_at_most_2pp": treatment_precision is not None and base_precision is not None and treatment_precision >= base_precision - 0.02,
        "preference_mean_strictly_above_half": float(patch_ci["mean"]) > 0.5,
        "patch_bootstrap_ci95_lower_strictly_above_half": float(patch_ci["ci95_lower"]) > 0.5,
        "adapter_off_byte_exact_after_unload": rollback_audit.get("adapter_off_byte_exact_after_unload") is True,
        "base_checkpoint_unchanged": rollback_audit.get("base_checkpoint_unchanged") is True,
        "adapter_unmerged_and_separate": rollback_audit.get("adapter_unmerged_and_separate") is True,
        "router_5x_task_only": rollback_audit.get("router_5x_task_only") is True,
        "router_10x_adapter_off": rollback_audit.get("router_10x_adapter_off") is True,
        "router_20x_adapter_off": rollback_audit.get("router_20x_adapter_off") is True,
    }
    screening = {
        "local_base_rule_flag_outputs": int(local_base_summary["native_forbidden_output_count"]) + int(local_base_summary["deployed_forbidden_output_count"]),
        "local_treatment_rule_flag_outputs": int(local_treatment_summary["native_forbidden_output_count"]) + int(local_treatment_summary["deployed_forbidden_output_count"]),
        "paired_base_flag_occurrences": forbidden_counts["base"],
        "paired_treatment_flag_occurrences": forbidden_counts["treatment"],
        "paired_treatment_flagged_unique_samples": sorted(flagged_samples),
        "requires_anonymous_human_confirmation": bool(
            int(local_treatment_summary["native_forbidden_output_count"])
            + int(local_treatment_summary["deployed_forbidden_output_count"])
            + forbidden_counts["treatment"]
        ),
    }
    return {
        "schema_version": "patho_lora_s8_locked_automatic_summary_v1",
        "patch_count": len(rows),
        "paired_judge_call_count": sum(calls_by_model.values()),
        "calls_by_model": dict(calls_by_model),
        "preference": {"patch_bootstrap": patch_ci, "wsi_cluster_bootstrap_sensitivity": cluster_ci},
        "claim_counts": {role: dict(counts) for role, counts in claim_counts.items()},
        "claim_precision": {"base": base_precision, "treatment": treatment_precision},
        "forbidden_screening": screening,
        "local_summaries": {"base": dict(local_base_summary), "treatment": dict(local_treatment_summary)},
        "rollback_audit": dict(rollback_audit),
        "non_human_hard_gates": gates,
        "all_non_human_hard_gates_passed": all(gates.values()),
        "final_locked_gate_pending_human_review": True,
    }


def select_human_review_sample_ids(
    sample_ids: Sequence[str], *, flagged_sample_ids: Sequence[str], count: int = 20
) -> list[str]:
    unique = sorted(set(sample_ids))
    flagged = sorted(set(flagged_sample_ids))
    require(len(unique) == len(sample_ids), "S8 human-review source IDs are duplicated")
    require(set(flagged) <= set(unique), "S8 flagged sample is absent from Locked real rows")
    require(len(flagged) <= count <= len(unique), "S8 human-review count cannot cover all automatic flags")
    remaining = [sample_id for sample_id in unique if sample_id not in set(flagged)]
    remaining.sort(key=lambda sample_id: sha256_json({"seed": BOOTSTRAP_SEED, "sample_id": sample_id}))
    selected = flagged + remaining[: count - len(flagged)]
    require(len(selected) == count and set(flagged) <= set(selected), "S8 human-review selection is incomplete")
    return selected
