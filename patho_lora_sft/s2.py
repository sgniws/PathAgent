from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable

from .common import ContractViolation
from .contracts import SPLIT_SEED
from .privacy import assert_generated_text_private
from .schema import parse_bare_findings, scan_forbidden


TEACHER_SOURCES = ("gemini", "luna")
REVIEWER_ROLE = {
    "reviewer_role": "expert_trained_morphology_reviewer",
    "medical_review_status": "expert_trained_reviewer_reviewed",
    "review_scope": "direct_visual_morphology_claim_support",
    "pathologist_verified": False,
    "diagnostic_validation": False,
}


def _score(*parts: Any) -> str:
    return hashlib.sha256(
        "\x1f".join([str(SPLIT_SEED), *(str(part) for part in parts)]).encode("utf-8")
    ).hexdigest()


def calibration_sample_id(candidate_id: str) -> str:
    if not candidate_id.startswith("cand_"):
        raise ContractViolation("Calibration source must be a frozen anonymous candidate")
    return "cal_" + _score("s2_calibration", candidate_id)[:24]


def select_calibration_candidates(
    pilot_rows: Iterable[dict[str, Any]], *, count: int = 50
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in pilot_rows]
    if count != 50:
        raise ContractViolation("S2 calibration size must remain exactly 50 real patches")
    selected = rows[:count]
    if len(selected) != count:
        raise ContractViolation("Frozen Train pilot has fewer than 50 real candidates")
    candidate_ids = [str(row.get("candidate_id", "")) for row in selected]
    coordinate_keys = [
        (str(row.get("slide_id", "")), str(row.get("patch_id", "")), row.get("x_level0"), row.get("y_level0"))
        for row in selected
    ]
    if len(candidate_ids) != len(set(candidate_ids)) or len(coordinate_keys) != len(set(coordinate_keys)):
        raise ContractViolation("S2 calibration selection contains a duplicate")
    if any(str(row.get("assigned_split")) != "train" for row in selected):
        raise ContractViolation("S2 calibration selection escaped the Train split")
    return selected


def candidate_observation(response: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    raw = response.get("content")
    if not isinstance(raw, str):
        raise ContractViolation("Teacher response content is unavailable")
    assert_generated_text_private(raw)
    parsed = parse_bare_findings(raw)
    forbidden = scan_forbidden(parsed.findings if parsed.valid else [raw])
    parsed_judge = audit.get("parsed_judge")
    if not audit.get("strict_schema_valid") or not isinstance(parsed_judge, dict):
        raise ContractViolation("Cross-audit response is not a strict parsed judge result")
    statuses = parsed_judge.get("statuses")
    if not isinstance(statuses, list):
        raise ContractViolation("Cross-audit statuses are unavailable")
    return {
        "raw": raw,
        "findings": list(parsed.findings) if parsed.valid else [],
        "strict_schema_valid": parsed.valid,
        "strict_schema_error": parsed.error,
        "raw_forbidden_hits": forbidden,
        "audit": {
            "statuses": statuses,
            "forbidden_inference": parsed_judge.get("forbidden_inference"),
            "strict_schema_valid": True,
        },
    }


def control_observation(response: dict[str, Any]) -> dict[str, Any]:
    raw = response.get("content")
    if not isinstance(raw, str):
        raise ContractViolation("Control response content is unavailable")
    assert_generated_text_private(raw)
    parsed = parse_bare_findings(raw)
    return {
        "raw": raw,
        "findings": list(parsed.findings) if parsed.valid else [],
        "strict_schema_valid": parsed.valid,
        "strict_schema_error": parsed.error,
        "exact_empty": parsed.valid and not parsed.findings,
        "raw_forbidden_hits": scan_forbidden(parsed.findings if parsed.valid else [raw]),
    }


def automated_calibration_summary(
    real_records: Iterable[dict[str, Any]], control_records: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    real = list(real_records)
    controls = list(control_records)
    if len(real) != 50 or len(controls) != 10:
        raise ContractViolation("S2 automated summary requires 50 real patches and 10 controls")
    result: dict[str, Any] = {
        "real_patch_count": len(real),
        "control_count": len(controls),
        "teachers": {},
        "human_gate_pending": True,
    }
    for source in TEACHER_SOURCES:
        candidates = [row["candidates"][source] for row in real]
        control_candidates = [row["candidates"][source] for row in controls]
        statuses = [status for row in candidates for status in row["audit"]["statuses"]]
        supported = sum(status == "supported" for status in statuses)
        result["teachers"][source] = {
            "first_strict_schema_pass": sum(row["strict_schema_valid"] for row in candidates),
            "first_strict_schema_total": len(candidates),
            "raw_forbidden_output_count": sum(bool(row["raw_forbidden_hits"]) for row in candidates),
            "control_exact_empty_pass": sum(row["exact_empty"] for row in control_candidates),
            "control_exact_empty_total": len(control_candidates),
            "cross_audit_supported_claims": supported,
            "cross_audit_total_claims": len(statuses),
            "cross_audit_claim_precision": supported / len(statuses) if statuses else None,
            "cross_audit_forbidden_count": sum(
                bool(row["audit"]["forbidden_inference"]) for row in candidates
            ),
            "automatic_hard_gate_pass": (
                sum(row["strict_schema_valid"] for row in candidates) >= 49
                and not any(row["raw_forbidden_hits"] for row in candidates)
                and sum(row["exact_empty"] for row in control_candidates) == 10
            ),
        }
    return result


def select_human_review_records(
    real_records: Iterable[dict[str, Any]],
    private_by_sample: dict[str, dict[str, Any]],
    *,
    count: int = 20,
) -> list[dict[str, Any]]:
    real = [dict(row) for row in real_records]
    if len(real) != 50 or count != 20:
        raise ContractViolation("S2 human review must select 20 of 50 calibration patches")

    def risk(row: dict[str, Any]) -> tuple[int, str]:
        candidates = row["candidates"]
        score = 3 if row.get("tier") == "risk" else 0
        for source in TEACHER_SOURCES:
            candidate = candidates[source]
            score += 10 if not candidate["strict_schema_valid"] else 0
            score += 8 if candidate["raw_forbidden_hits"] else 0
            score += 5 * sum(status != "supported" for status in candidate["audit"]["statuses"])
            score += 6 if candidate["audit"]["forbidden_inference"] else 0
        score += min(3, abs(len(candidates["gemini"]["findings"]) - len(candidates["luna"]["findings"])))
        return score, _score("human_review", row["sample_id"])

    ranked = sorted(real, key=lambda row: (-risk(row)[0], risk(row)[1]))
    chosen: list[dict[str, Any]] = []
    seen_slides: set[str] = set()
    for row in ranked:
        private = private_by_sample.get(str(row["sample_id"]))
        if private is None:
            raise ContractViolation("Human review selection lacks private WSI mapping")
        slide = str(private["slide_id"])
        if slide not in seen_slides:
            chosen.append(row)
            seen_slides.add(slide)
            if len(chosen) == count:
                break
    if len(chosen) < count:
        selected_ids = {str(row["sample_id"]) for row in chosen}
        chosen.extend(row for row in ranked if str(row["sample_id"]) not in selected_ids)
        chosen = chosen[:count]
    if len(chosen) != count or len({row["sample_id"] for row in chosen}) != count:
        raise ContractViolation("Unable to build the frozen 20-image S2 human sample")
    return chosen


def paired_review_source_records(selected: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in selected:
        def review_findings(source: str) -> list[str]:
            candidate = row["candidates"][source]
            if candidate["strict_schema_valid"]:
                return candidate["findings"]
            return [
                f"Recognizable raw claim {index + 1} (inspect the raw output above)."
                for index in range(len(candidate["audit"]["statuses"]))
            ]

        records.append(
            {
                "review_id": "s2review_" + _score("review", row["sample_id"])[:20],
                "image_relpath": row["human_review_image_relpath"],
                "candidate_a_source": "gemini",
                "candidate_a_raw": row["candidates"]["gemini"]["raw"],
                "candidate_a_findings": review_findings("gemini"),
                "candidate_b_source": "luna",
                "candidate_b_raw": row["candidates"]["luna"]["raw"],
                "candidate_b_findings": review_findings("luna"),
            }
        )
    return records


def human_teacher_metrics(
    normalized_rows: Iterable[dict[str, Any]], private_mapping: Iterable[dict[str, str]]
) -> dict[str, Any]:
    mapping = {str(row["presentation_id"]): dict(row) for row in private_mapping}
    metrics = {
        source: {
            "supported_claims": 0,
            "total_claims": 0,
            "forbidden_presentations": 0,
            "pair_wins": 0,
        }
        for source in TEACHER_SOURCES
    }
    non_tie_preferences = 0
    reviewed_sources: set[str] = set()
    for row in normalized_rows:
        private = mapping[str(row["presentation_id"])]
        if private["is_repeat"] == "true":
            continue
        reviewed_sources.add(private["source_review_id"])
        for side in ("a", "b"):
            source = private[f"candidate_{side}_source"]
            labels = row[f"candidate_{side}_claim_labels"]
            metrics[source]["supported_claims"] += sum(label == "supported" for label in labels)
            metrics[source]["total_claims"] += len(labels)
            metrics[source]["forbidden_presentations"] += int(
                row[f"candidate_{side}_forbidden_inference"]
            )
        preference = row["pair_preference"]
        if preference != "tie":
            non_tie_preferences += 1
            source = private[f"candidate_{preference.casefold()}_source"]
            metrics[source]["pair_wins"] += 1
    if len(reviewed_sources) != 20:
        raise ContractViolation("Human S2 metrics require 20 unique non-repeat review images")
    for source in TEACHER_SOURCES:
        total = metrics[source]["total_claims"]
        metrics[source]["claim_precision"] = (
            metrics[source]["supported_claims"] / total if total else None
        )
        metrics[source]["non_tie_preference_rate"] = (
            metrics[source]["pair_wins"] / non_tie_preferences if non_tie_preferences else None
        )
    return {
        "unique_review_images": len(reviewed_sources),
        "non_tie_preferences": non_tie_preferences,
        "teachers": metrics,
        "reviewer_role": REVIEWER_ROLE,
    }


def choose_primary_teacher(
    automated: dict[str, Any], human: dict[str, Any]
) -> dict[str, Any]:
    eligible: dict[str, bool] = {}
    reasons: dict[str, list[str]] = {}
    for source in TEACHER_SOURCES:
        auto = automated["teachers"][source]
        human_row = human["teachers"][source]
        failures = []
        if auto["first_strict_schema_pass"] < 49:
            failures.append("first_strict_schema_below_49_of_50")
        if auto["raw_forbidden_output_count"] != 0:
            failures.append("raw_forbidden_content_nonzero")
        if auto["control_exact_empty_pass"] != 10:
            failures.append("control_exact_empty_below_10_of_10")
        if human_row["claim_precision"] is None or human_row["claim_precision"] < 0.95:
            failures.append("human_claim_precision_below_95_percent")
        if human_row["forbidden_presentations"] != 0:
            failures.append("human_forbidden_inference_nonzero")
        reasons[source] = failures
        eligible[source] = not failures
    passing = [source for source in TEACHER_SOURCES if eligible[source]]
    if not passing:
        return {
            "status": "stopped_no_eligible_primary",
            "primary": None,
            "auditor": None,
            "eligible": eligible,
            "failures": reasons,
            "reason": "both_teachers_failed_hard_gate",
        }
    if len(passing) == 1:
        primary = passing[0]
        return {
            "status": "selected",
            "primary": primary,
            "auditor": "luna" if primary == "gemini" else "gemini",
            "eligible": eligible,
            "failures": reasons,
            "reason": "only_one_teacher_passed_all_hard_gates",
        }
    gemini_precision = human["teachers"]["gemini"]["claim_precision"]
    luna_precision = human["teachers"]["luna"]["claim_precision"]
    assert gemini_precision is not None and luna_precision is not None
    if abs(gemini_precision - luna_precision) >= 0.05:
        primary = "gemini" if gemini_precision > luna_precision else "luna"
        reason = "human_claim_precision_difference_at_least_five_points"
    else:
        gemini_preference = human["teachers"]["gemini"]["non_tie_preference_rate"]
        luna_preference = human["teachers"]["luna"]["non_tie_preference_rate"]
        if gemini_preference is not None and gemini_preference >= 0.60:
            primary, reason = "gemini", "gemini_reached_sixty_percent_non_tie_preference"
        elif luna_preference is not None and luna_preference >= 0.60:
            primary, reason = "luna", "luna_reached_sixty_percent_non_tie_preference"
        else:
            primary, reason = "luna", "tie_default_to_luna_for_cost_control"
    return {
        "status": "selected",
        "primary": primary,
        "auditor": "luna" if primary == "gemini" else "gemini",
        "eligible": eligible,
        "failures": reasons,
        "reason": reason,
    }
