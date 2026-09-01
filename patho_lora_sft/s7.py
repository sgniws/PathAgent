from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import random
from typing import Any, Sequence

from .audit import CanonicalDecision
from .common import ContractViolation
from .dataset import build_training_record
from .schema import parse_wrapped_findings


FULL_SPLIT_COUNTS = {"train": 2000, "validation": 100, "locked": 100}
FULL_KIND_COUNTS = {
    "train": Counter(real=1900, control=100),
    "validation": Counter(real=95, control=5),
    "locked": Counter(real=95, control=5),
}


@dataclass(frozen=True)
class PairCandidateAudit:
    claims: tuple[tuple[str, str], ...]
    forbidden_inference: bool


@dataclass(frozen=True)
class PairJudgeResult:
    valid: bool
    candidate_a: PairCandidateAudit | None
    candidate_b: PairCandidateAudit | None
    preference: str | None
    error: str | None


def parse_pair_judge_result(raw: str) -> PairJudgeResult:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return PairJudgeResult(False, None, None, None, "invalid_json")
    if not isinstance(value, dict) or set(value) != {"candidate_a", "candidate_b", "preference"}:
        return PairJudgeResult(False, None, None, None, "top_level_shape")
    if value.get("preference") not in {"A", "B", "tie"}:
        return PairJudgeResult(False, None, None, None, "preference_invalid")

    def candidate(name: str) -> PairCandidateAudit | None:
        item = value.get(name)
        if not isinstance(item, dict) or set(item) != {"claims", "forbidden_inference"}:
            return None
        if not isinstance(item.get("forbidden_inference"), bool):
            return None
        claims = item.get("claims")
        if not isinstance(claims, list) or len(claims) > 5:
            return None
        parsed: list[tuple[str, str]] = []
        for claim in claims:
            if not isinstance(claim, dict) or set(claim) != {"claim", "status"}:
                return None
            text = claim.get("claim")
            status = claim.get("status")
            if not isinstance(text, str) or not text.strip() or status not in {"supported", "unsupported", "not_assessable"}:
                return None
            parsed.append((text, status))
        return PairCandidateAudit(tuple(parsed), bool(item["forbidden_inference"]))

    candidate_a = candidate("candidate_a")
    candidate_b = candidate("candidate_b")
    if candidate_a is None or candidate_b is None:
        return PairJudgeResult(False, None, None, None, "candidate_shape")
    return PairJudgeResult(True, candidate_a, candidate_b, str(value["preference"]), None)


def preference_score(preference: str, *, treatment_position: str) -> float:
    if preference not in {"A", "B", "tie"} or treatment_position not in {"A", "B"}:
        raise ContractViolation("Pair preference or treatment position is invalid")
    if preference == "tie":
        return 0.5
    return 1.0 if preference == treatment_position else 0.0


def bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, replicates: int = 10000
) -> dict[str, float | int]:
    if not values or replicates < 1000 or any(not math.isfinite(float(value)) for value in values):
        raise ContractViolation("Bootstrap inputs are invalid")
    rng = random.Random(seed)
    materialized = [float(value) for value in values]
    draws = []
    for _ in range(replicates):
        draws.append(sum(rng.choice(materialized) for _ in materialized) / len(materialized))
    draws.sort()
    lo = draws[int(0.025 * replicates)]
    hi = draws[min(replicates - 1, int(0.975 * replicates))]
    return {
        "mean": sum(materialized) / len(materialized),
        "ci95_lower": lo,
        "ci95_upper": hi,
        "replicates": replicates,
        "seed": seed,
    }


def cluster_bootstrap_mean_ci(
    values_by_cluster: dict[str, Sequence[float]], *, seed: int, replicates: int = 10000
) -> dict[str, float | int]:
    if not values_by_cluster or any(not values for values in values_by_cluster.values()):
        raise ContractViolation("Cluster bootstrap inputs are empty")
    clusters = sorted(values_by_cluster)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = [rng.choice(clusters) for _ in clusters]
        values = [float(value) for cluster in sampled for value in values_by_cluster[cluster]]
        draws.append(sum(values) / len(values))
    draws.sort()
    observed = [float(value) for cluster in clusters for value in values_by_cluster[cluster]]
    return {
        "mean": sum(observed) / len(observed),
        "ci95_lower": draws[int(0.025 * replicates)],
        "ci95_upper": draws[min(replicates - 1, int(0.975 * replicates))],
        "cluster_count": len(clusters),
        "replicates": replicates,
        "seed": seed,
    }


def validate_s6_holistic_attestation(attestation: dict[str, Any]) -> dict[str, bool]:
    conclusions = attestation.get("attested_conclusions", {})
    gates = {
        "all_325_unique_images_reviewed": conclusions.get("all_325_unique_images_reviewed") is True,
        "all_reviewed_content_approved": conclusions.get("all_reviewed_content_approved") is True,
        "approve_s6_gate_closure": conclusions.get("approve_s6_gate_closure") is True,
    }
    if not all(gates.values()):
        raise ContractViolation("S6 holistic attestation does not approve every declared review gate")
    if attestation.get("numeric_metrics_fabricated") is not False:
        raise ContractViolation("S6 attestation must explicitly forbid fabricated numeric metrics")
    return gates


def _decision_for_s6_row(row: dict[str, Any], *, human_gate_passed: bool) -> CanonicalDecision:
    assessment = row.get("assessment") if isinstance(row.get("assessment"), dict) else {}
    status = str(assessment.get("status", ""))
    if status == "waiting_human_review" and not human_gate_passed:
        raise ContractViolation("S6 Medium-risk row lacks positive human review")
    if status not in {"accepted_automatic", "waiting_human_review"}:
        raise ContractViolation("S6 row is not eligible for the frozen dataset")
    target = str(assessment.get("target", ""))
    parsed = parse_wrapped_findings(target)
    if not parsed.valid:
        raise ContractViolation("S6 target is not canonical wrapped findings JSON")
    return CanonicalDecision(
        status="accepted",
        risk_tier="medium" if status == "waiting_human_review" else "low",
        source="luna",
        target=target,
        findings=parsed.findings,
        reason="s6_explicit_human_gate" if status == "waiting_human_review" else "s6_automatic_gate",
        requires_human_review=False,
    )


def build_full_training_splits(
    *,
    pilot_train: Sequence[dict[str, Any]],
    s6_real: Sequence[dict[str, Any]],
    s6_controls: Sequence[dict[str, Any]],
    user_prompt: str,
    human_gate_passed: bool,
) -> dict[str, list[dict[str, Any]]]:
    if len(pilot_train) != 200:
        raise ContractViolation("S7 requires the frozen 200-record Train prefix")
    if Counter(str(row.get("record_kind", "")) for row in pilot_train) != Counter(real=190, control=10):
        raise ContractViolation("S7 Train prefix real/control composition changed")
    if len(s6_real) != 1900 or len(s6_controls) != 100:
        raise ContractViolation("S7 requires exactly 1,900 S6 real rows and 100 controls")

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [dict(row) for row in pilot_train],
        "validation": [],
        "locked": [],
    }
    seen_ids = {str(row.get("sample_id", "")) for row in pilot_train}
    if "" in seen_ids or len(seen_ids) != len(pilot_train):
        raise ContractViolation("S7 Train prefix IDs are missing or duplicated")

    for row in s6_real:
        split = str(row.get("assigned_split", ""))
        if split not in splits:
            raise ContractViolation("S6 real row has an unexpected split")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in seen_ids:
            raise ContractViolation("Full dataset sample IDs are missing or duplicated")
        seen_ids.add(sample_id)
        record = build_training_record(
            sample_id=sample_id,
            image_relpath=str(row.get("image_relpath_from_run", "")),
            user_prompt=user_prompt,
            decision=_decision_for_s6_row(row, human_gate_passed=human_gate_passed),
        )
        record["record_kind"] = "real"
        splits[split].append(record)

    for row in s6_controls:
        split = str(row.get("assigned_split", ""))
        if split not in splits:
            raise ContractViolation("S6 control has an unexpected split")
        if row.get("exact_empty") is not True or row.get("findings") != []:
            raise ContractViolation("S6 control is not frozen exact-empty")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in seen_ids:
            raise ContractViolation("Full dataset control IDs are missing or duplicated")
        seen_ids.add(sample_id)
        decision = CanonicalDecision(
            status="accepted",
            risk_tier="control",
            source="frozen_empty",
            target=str(row.get("target", "")),
            findings=(),
            reason="blank_control",
            requires_human_review=False,
        )
        record = build_training_record(
            sample_id=sample_id,
            image_relpath=str(row.get("image_relpath_from_run", "")),
            user_prompt=user_prompt,
            decision=decision,
        )
        record["record_kind"] = "control"
        splits[split].append(record)

    for split, rows in splits.items():
        rows.sort(key=lambda row: str(row["sample_id"]))
        if len(rows) != FULL_SPLIT_COUNTS[split]:
            raise ContractViolation(f"Full {split} record count changed")
        if Counter(str(row["record_kind"]) for row in rows) != FULL_KIND_COUNTS[split]:
            raise ContractViolation(f"Full {split} real/control composition changed")
    if len(seen_ids) != 2200:
        raise ContractViolation("Full dataset is not 2,200 unique sample IDs")
    return splits
