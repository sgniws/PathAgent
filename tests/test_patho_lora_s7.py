from __future__ import annotations

from collections import Counter

import pytest

from patho_lora_sft.common import ContractViolation
from patho_lora_sft.s7 import (
    bootstrap_mean_ci,
    build_full_training_splits,
    cluster_bootstrap_mean_ci,
    parse_pair_judge_result,
    preference_score,
    validate_s6_holistic_attestation,
)


TARGET = '<answer>{"findings":["Visible compact tissue"]}</answer>'
EMPTY = '<answer>{"findings":[]}</answer>'


def _pilot() -> list[dict[str, object]]:
    rows = []
    for index in range(200):
        rows.append({
            "sample_id": f"pilot_{index}",
            "record_kind": "real" if index < 190 else "control",
            "messages": [{"role": "assistant", "content": [{"type": "text", "text": TARGET}]}],
        })
    return rows


def _s6_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    real = []
    controls = []
    index = 0
    for split, clean, risk in (("train", 1620, 90), ("validation", 90, 5), ("locked", 90, 5)):
        for tier, count in (("clean", clean), ("risk", risk)):
            for _ in range(count):
                sample_id = f"s6_{index}"
                real.append({
                    "sample_id": sample_id,
                    "assigned_split": split,
                    "tier": tier,
                    "image_relpath_from_run": f"images/{split}/{sample_id}.png",
                    "assessment": {
                        "status": "waiting_human_review" if tier == "risk" else "accepted_automatic",
                        "target": TARGET,
                    },
                })
                index += 1
    for split, count in (("train", 90), ("validation", 5), ("locked", 5)):
        for _ in range(count):
            sample_id = f"control_{index}"
            controls.append({
                "sample_id": sample_id,
                "assigned_split": split,
                "image_relpath_from_run": f"images/{split}/{sample_id}.png",
                "exact_empty": True,
                "findings": [],
                "target": EMPTY,
            })
            index += 1
    return real, controls


def test_s6_holistic_attestation_requires_all_declared_conclusions() -> None:
    attestation = {
        "attested_conclusions": {
            "all_325_unique_images_reviewed": True,
            "all_reviewed_content_approved": True,
            "approve_s6_gate_closure": True,
        },
        "numeric_metrics_fabricated": False,
    }
    assert all(validate_s6_holistic_attestation(attestation).values())
    attestation["attested_conclusions"]["approve_s6_gate_closure"] = False
    with pytest.raises(ContractViolation):
        validate_s6_holistic_attestation(attestation)


def test_full_training_splits_are_exact_and_keep_locked_separate() -> None:
    real, controls = _s6_rows()
    splits = build_full_training_splits(
        pilot_train=_pilot(),
        s6_real=real,
        s6_controls=controls,
        user_prompt="frozen prompt",
        human_gate_passed=True,
    )
    assert {key: len(value) for key, value in splits.items()} == {"train": 2000, "validation": 100, "locked": 100}
    assert Counter(row["record_kind"] for row in splits["train"]) == Counter(real=1900, control=100)
    assert Counter(row["record_kind"] for row in splits["validation"]) == Counter(real=95, control=5)
    assert Counter(row["record_kind"] for row in splits["locked"]) == Counter(real=95, control=5)
    ids = [row["sample_id"] for rows in splits.values() for row in rows]
    assert len(ids) == len(set(ids)) == 2200


def test_medium_risk_rows_cannot_enter_without_positive_human_gate() -> None:
    real, controls = _s6_rows()
    with pytest.raises(ContractViolation, match="lacks positive human review"):
        build_full_training_splits(
            pilot_train=_pilot(),
            s6_real=real,
            s6_controls=controls,
            user_prompt="frozen prompt",
            human_gate_passed=False,
        )


def test_pair_judge_parser_and_order_normalization() -> None:
    raw = '{"candidate_a":{"claims":[{"claim":"visible tissue","status":"supported"}],"forbidden_inference":false},"candidate_b":{"claims":[],"forbidden_inference":false},"preference":"A"}'
    parsed = parse_pair_judge_result(raw)
    assert parsed.valid and parsed.candidate_a is not None
    assert parsed.candidate_a.claims == (("visible tissue", "supported"),)
    assert preference_score("A", treatment_position="A") == 1.0
    assert preference_score("A", treatment_position="B") == 0.0
    assert preference_score("tie", treatment_position="B") == 0.5
    assert not parse_pair_judge_result('{"preference":"A"}').valid


def test_bootstrap_summaries_are_deterministic() -> None:
    first = bootstrap_mean_ci([0.0, 0.5, 1.0], seed=7, replicates=1000)
    second = bootstrap_mean_ci([0.0, 0.5, 1.0], seed=7, replicates=1000)
    assert first == second and first["mean"] == 0.5
    clustered = cluster_bootstrap_mean_ci({"w1": [1.0, 1.0], "w2": [0.0]}, seed=7, replicates=1000)
    assert clustered["cluster_count"] == 2
