from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .common import ContractViolation, sha256_json


S10_SEED = 20260827
EXPECTED_WSI = 10
EXPECTED_SOURCE_PATCHES = 1408
PATCHES_PER_WSI = 10
HUMAN_PATCHES_PER_WSI = 2
EXPECTED_EVALUATION_PATCHES = 100
EXPECTED_HUMAN_UNIQUE = 20
EXPECTED_HUMAN_REPEATS = 2


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolation(message)


def _score(*parts: str, seed: int = S10_SEED) -> str:
    return sha256_json({"seed": seed, "parts": list(parts)})


def select_test10_metadata(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = S10_SEED,
    per_wsi: int = PATCHES_PER_WSI,
) -> list[dict[str, str]]:
    """Select metadata only; callers must not attach or read image bytes here."""

    require(len(rows) == EXPECTED_SOURCE_PATCHES, "Test10 source patch count changed")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        require(set(row) == {"wsi_id", "patch_id"}, "Test10 selection received non-metadata fields")
        wsi_id, patch_id = str(row["wsi_id"]), str(row["patch_id"])
        require(bool(wsi_id and patch_id) and patch_id not in seen_ids, "Test10 metadata ID is missing or duplicated")
        seen_ids.add(patch_id)
        groups[wsi_id].append(row)
    require(len(groups) == EXPECTED_WSI, "Test10 WSI count changed")
    selected: list[dict[str, str]] = []
    for wsi_id in sorted(groups):
        ranked = sorted(groups[wsi_id], key=lambda row: _score("test10", wsi_id, str(row["patch_id"]), seed=seed))
        require(len(ranked) >= per_wsi, "Test10 WSI lacks the frozen per-WSI quota")
        selected.extend({"wsi_id": wsi_id, "patch_id": str(row["patch_id"])} for row in ranked[:per_wsi])
    require(len(selected) == EXPECTED_EVALUATION_PATCHES, "Test10 selected patch count changed")
    return selected


def select_human_review_presentations(
    selected: Sequence[Mapping[str, str]], *, seed: int = S10_SEED
) -> list[dict[str, Any]]:
    require(len(selected) == EXPECTED_EVALUATION_PATCHES, "Human selection requires the frozen 100-patch list")
    groups: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in selected:
        groups[str(row["wsi_id"])].append(row)
    require(len(groups) == EXPECTED_WSI, "Human selection WSI count changed")
    unique: list[dict[str, Any]] = []
    for wsi_id in sorted(groups):
        ranked = sorted(groups[wsi_id], key=lambda row: _score("human", wsi_id, str(row["patch_id"]), seed=seed))
        require(len(ranked) == PATCHES_PER_WSI, "Human selection per-WSI source count changed")
        for row in ranked[:HUMAN_PATCHES_PER_WSI]:
            unique.append({"wsi_id": wsi_id, "patch_id": str(row["patch_id"]), "repeat": False})
    require(len(unique) == EXPECTED_HUMAN_UNIQUE, "Human unique selection count changed")
    repeat_sources = sorted(unique, key=lambda row: _score("repeat", str(row["patch_id"]), seed=seed))[:EXPECTED_HUMAN_REPEATS]
    presentations = list(unique) + [{**row, "repeat": True} for row in repeat_sources]
    presentations.sort(key=lambda row: _score("presentation", str(row["patch_id"]), str(row["repeat"]), seed=seed))
    require(len(presentations) == 22, "Human presentation count changed")
    return presentations


def morphology_claim_precision(labels: Iterable[str]) -> dict[str, Any]:
    counts = Counter(str(label) for label in labels)
    require(set(counts) <= {"supported", "unsupported", "not_assessable"}, "Unknown morphology claim label")
    denominator = sum(counts.values())
    return {
        "supported": counts["supported"],
        "unsupported": counts["unsupported"],
        "not_assessable": counts["not_assessable"],
        "denominator": denominator,
        "precision": counts["supported"] / denominator if denominator else None,
    }


def parse_pair_claim_audit(content: str) -> dict[str, Any]:
    """Strict S10 audit parser; preference fields and extra keys are rejected."""

    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ContractViolation("S10 claim audit is not JSON") from exc
    require(isinstance(value, dict) and set(value) == {"candidate_a", "candidate_b"}, "S10 audit top-level keys changed")
    parsed: dict[str, Any] = {}
    for key in ("candidate_a", "candidate_b"):
        candidate = value[key]
        require(isinstance(candidate, dict) and set(candidate) == {"claims", "forbidden_inference"}, "S10 candidate keys changed")
        require(isinstance(candidate["forbidden_inference"], bool), "S10 forbidden flag is not boolean")
        require(isinstance(candidate["claims"], list), "S10 claims is not an array")
        require(len(candidate["claims"]) <= 5, "S10 candidate contains more than five claims")
        claims = []
        for claim in candidate["claims"]:
            require(isinstance(claim, dict) and set(claim) == {"claim", "status"}, "S10 claim keys changed")
            require(isinstance(claim["claim"], str) and claim["claim"].strip(), "S10 claim text is empty")
            require(claim["status"] in {"supported", "unsupported", "not_assessable"}, "S10 claim status changed")
            claims.append({"claim": claim["claim"], "status": claim["status"]})
        parsed[key] = {"claims": claims, "forbidden_inference": candidate["forbidden_inference"]}
    return parsed


def audit_pair_claims(
    *,
    client: Any,
    contract: Any,
    image: Any,
    item_id: str,
    candidate_a_json: str,
    candidate_b_json: str,
    prompt: str,
    response_format: Mapping[str, Any],
    stage: str,
    response_path: Any,
    balance: Any,
    max_tokens: int = 1536,
) -> dict[str, Any]:
    """Issue one frozen no-preference S10 A/B claim audit through the validated client."""

    for candidate in (candidate_a_json, candidate_b_json):
        require(isinstance(candidate, str) and 0 < len(candidate) <= 4096, "S10 candidate JSON size changed")
        decoded = json.loads(candidate)
        require(isinstance(decoded, dict) and set(decoded) == {"findings"}, "S10 candidate is not constrained findings JSON")
        require(
            isinstance(decoded["findings"], list)
            and len(decoded["findings"]) <= 5
            and all(isinstance(value, str) and value.strip() for value in decoded["findings"]),
            "S10 candidate findings changed",
        )
    pair = json.dumps(
        {"candidate_a": candidate_a_json, "candidate_b": candidate_b_json},
        ensure_ascii=False,
        separators=(",", ":"),
    )

    def validate(content: str) -> tuple[bool, str | None, dict[str, Any]]:
        try:
            parsed = parse_pair_claim_audit(content)
        except ContractViolation as exc:
            return False, str(exc), {}
        return True, None, {"parsed_pair_claim_audit": parsed}

    return client._complete_strict_json(
        contract=contract,
        image=image,
        item_id=item_id,
        system_prompt=prompt,
        user_prompt=f"Anonymous candidate pair:\n{pair}",
        response_format=dict(response_format),
        stage=stage,
        response_path=response_path,
        balance=balance,
        estimated_max_cost_usd=0.005,
        max_tokens=max_tokens,
        response_kind="s10_pair_claim_audit_no_preference",
        validator=validate,
        strict_schema_required=True,
    )


def s10_preregistered_contract(*, source_path: str, source_sha256: str, spend_before_usd: float) -> dict[str, Any]:
    require(len(source_sha256) == 64, "Authoritative source SHA-256 is missing")
    worst = spend_before_usd + 2.0
    require(worst <= 9.0, "S10 budget no longer preserves the $0.50 project reserve")
    return {
        "schema_version": "patho_lora_s10_preregistered_evaluation_contract_v1",
        "status": "frozen_waiting_separate_explicit_s10_authorization",
        "authorization": {"s10_authorized": False, "test10_reads_during_freeze": 0},
        "source": {
            "path": source_path,
            "sha256_inherited_without_rehash_or_decode": source_sha256,
            "expected_wsi": EXPECTED_WSI,
            "expected_patches": EXPECTED_SOURCE_PATCHES,
        },
        "local_generation": {
            "all_1408_patches": True,
            "candidates": ["base", "r16"],
            "same_prompt_schema_max_tokens": True,
            "output": "canonical_constrained_json_object_only",
        },
        "sampling": {
            "seed": S10_SEED,
            "patches_per_wsi": PATCHES_PER_WSI,
            "expected_selected": EXPECTED_EVALUATION_PATCHES,
            "metadata_before_image_read": True,
        },
        "vlm_audit": {
            "models": [
                {"model": "google/gemini-3.7-flash", "provider": "Google", "zdr": True},
                {"model": "openai/gpt-5.6-luna", "provider": "OpenAI", "zdr": False},
            ],
            "orders": ["base_then_r16", "r16_then_base"],
            "logical_calls": 400,
            "preference_collected_or_reported": False,
        },
        "only_capability_metric": {
            "name_zh": "图像支持的形态描述准确率",
            "formula": "supported/(supported+unsupported+not_assessable)",
            "aggregation": "micro_average_separate_for_base_and_r16",
            "zero_denominator": "undefined",
            "r16_vlm_minimum": 0.95,
            "base_is_comparator_and_may_be_undefined": True,
        },
        "human_review": {
            "unique_patches": EXPECTED_HUMAN_UNIQUE,
            "per_wsi": HUMAN_PATCHES_PER_WSI,
            "anonymous_repeats": EXPECTED_HUMAN_REPEATS,
            "presentations": 22,
            "r16_precision_minimum": 0.95,
            "repeat_consistency_minimum": 0.90,
            "human_confirmed_forbidden_inference_required": 0,
            "ui": "single_scrolling_page_autosave_one_click_json_no_csv",
        },
        "safety": {"human_confirmed_forbidden_inference_required": 0},
        "budget": {
            "maximum_cost_per_call_usd": 0.005,
            "maximum_stage_cost_usd": 2.0,
            "project_spend_before_s10_usd": spend_before_usd,
            "worst_case_cumulative_usd": worst,
            "project_cap_usd": 9.5,
            "minimum_reserve_usd": 0.5,
        },
        "one_shot": {
            "same_test10_version_modify_then_retry_allowed": False,
            "results_may_not_change_weights_prompt_schema_decoder_thresholds_or_stopping": True,
        },
    }
