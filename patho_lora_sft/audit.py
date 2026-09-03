from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from .common import ContractViolation
from .schema import FindingsResult, canonical_target, scan_forbidden, validate_teacher_candidate


ClaimStatus = Literal["supported", "unsupported", "not_assessable"]


@dataclass(frozen=True)
class JudgeResult:
    valid: bool
    statuses: tuple[ClaimStatus, ...]
    forbidden_inference: bool
    error: str | None = None

    @property
    def all_supported(self) -> bool:
        return self.valid and not self.forbidden_inference and all(status == "supported" for status in self.statuses)


@dataclass(frozen=True)
class CanonicalDecision:
    status: str
    risk_tier: str
    source: str | None
    target: str | None
    findings: tuple[str, ...]
    reason: str
    requires_human_review: bool


def parse_judge_result(raw: str, expected_claims: int | None) -> JudgeResult:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return JudgeResult(False, (), False, "judge_json_parse_failed")
    if not isinstance(value, dict) or set(value) != {"claims", "forbidden_inference"}:
        return JudgeResult(False, (), False, "judge_unexpected_keys")
    claims = value["claims"]
    if not isinstance(claims, list):
        return JudgeResult(False, (), False, "judge_claims_not_array")
    if expected_claims is not None and len(claims) != expected_claims:
        return JudgeResult(False, (), False, "judge_claim_count_mismatch")
    if expected_claims is None and len(claims) > 5:
        return JudgeResult(False, (), False, "judge_too_many_claims")
    statuses: list[ClaimStatus] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict) or set(claim) != {"finding_index", "status"}:
            return JudgeResult(False, (), False, "judge_claim_shape_invalid")
        if claim["finding_index"] != index:
            return JudgeResult(False, (), False, "judge_claim_order_invalid")
        if claim["status"] not in {"supported", "unsupported", "not_assessable"}:
            return JudgeResult(False, (), False, "judge_status_invalid")
        statuses.append(claim["status"])
    if not isinstance(value["forbidden_inference"], bool):
        return JudgeResult(False, (), False, "judge_forbidden_flag_invalid")
    return JudgeResult(True, tuple(statuses), value["forbidden_inference"])


def _candidate_state(raw: str, audit_raw: str) -> tuple[FindingsResult, list[dict[str, str]], JudgeResult]:
    candidate, forbidden = validate_teacher_candidate(raw)
    expected = len(candidate.findings) if candidate.valid else 0
    judge = parse_judge_result(audit_raw, expected)
    return candidate, forbidden, judge


def choose_canonical_candidate(
    *,
    primary_raw: str,
    secondary_audit_raw: str,
    fallback_raw: str | None = None,
    primary_audit_raw: str | None = None,
    natural_qc_risk: bool = False,
    human_review_passed: bool | None = None,
) -> CanonicalDecision:
    primary, primary_forbidden, primary_judge = _candidate_state(primary_raw, secondary_audit_raw)

    def decide(candidate: FindingsResult, forbidden: list[dict[str, str]], judge: JudgeResult, source: str, fallback: bool) -> CanonicalDecision | None:
        if not candidate.valid or forbidden or not judge.valid or judge.forbidden_inference:
            return None
        if any(status == "unsupported" for status in judge.statuses):
            return None
        medium = natural_qc_risk or fallback or any(status == "not_assessable" for status in judge.statuses)
        if medium and human_review_passed is not True:
            return CanonicalDecision(
                status="waiting_human_review" if human_review_passed is None else "rejected",
                risk_tier="medium",
                source=source,
                target=None,
                findings=candidate.findings,
                reason="medium_risk_requires_positive_human_review",
                requires_human_review=True,
            )
        return CanonicalDecision(
            status="accepted",
            risk_tier="medium" if medium else "low",
            source=source,
            target=canonical_target(candidate.findings),
            findings=candidate.findings,
            reason="complete_candidate_passed_schema_safety_and_cross_audit",
            requires_human_review=medium,
        )

    primary_decision = decide(primary, primary_forbidden, primary_judge, "primary", False)
    if primary_decision is not None:
        return primary_decision
    if fallback_raw is None or primary_audit_raw is None:
        return CanonicalDecision("fallback_required", "high", None, None, (), "primary_failed", False)
    fallback_candidate, fallback_forbidden, fallback_judge = _candidate_state(fallback_raw, primary_audit_raw)
    fallback_decision = decide(fallback_candidate, fallback_forbidden, fallback_judge, "secondary_fallback", True)
    if fallback_decision is not None:
        return fallback_decision
    return CanonicalDecision("rejected", "high", None, None, (), "both_complete_candidates_failed", False)


def claim_precision(judgments: list[JudgeResult]) -> float | None:
    statuses = [status for judgment in judgments if judgment.valid for status in judgment.statuses]
    if not statuses:
        return None
    return sum(status == "supported" for status in statuses) / len(statuses)


def validate_accepted_decision(decision: CanonicalDecision) -> None:
    if decision.status != "accepted" or decision.target is None:
        raise ContractViolation("Only an accepted canonical decision can enter SFT data")
    if scan_forbidden(decision.findings):
        raise ContractViolation("Accepted decision contains forbidden inference")
