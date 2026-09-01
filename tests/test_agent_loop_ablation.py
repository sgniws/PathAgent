from copy import deepcopy

import pytest

from models.experiment_policy import (
    initial_retrieval_count,
    normalize_experiment_action,
    replenishment_count,
)
def test_fixed_initial_top_k_overrides_ratio_and_clamps_to_available():
    assert initial_retrieval_count(200, 0.10, 5) == 5
    assert initial_retrieval_count(3, 0.10, 5) == 3
    assert initial_retrieval_count(200, 0.10, None) == 20


def test_fixed_initial_top_k_rejects_non_positive_values():
    with pytest.raises(ValueError, match="initial_top_k"):
        initial_retrieval_count(10, 0.10, 0)


def test_fixed_replenish_top_k_overrides_ratio():
    assert replenishment_count(200, 0.05, 5) == 5
    assert replenishment_count(3, 0.05, 5) == 3
    assert replenishment_count(200, 0.05, None) == 10


def test_no_zoom_policy_turns_targeted_zoom_into_inspect():
    decision = {
        "missing_evidence": "higher-power nuclei",
        "next_action": {
            "type": "zoom",
            "query": "inspect atypical nuclei",
            "target_patches": ["x1_y2"],
            "magnification": 20,
        },
    }
    normalized, repair = normalize_experiment_action(deepcopy(decision), "no_zoom", 12)
    assert normalized["next_action"] == {
        "type": "inspect",
        "query": "inspect atypical nuclei",
        "target_patches": ["x1_y2"],
        "magnification": None,
    }
    assert repair["reason"] == "no_zoom_experiment_policy"


def test_no_zoom_policy_turns_untargeted_zoom_into_retrieve():
    decision = {
        "missing_evidence": "additional architecture",
        "next_action": {
            "type": "zoom",
            "query": "",
            "target_patches": [],
            "magnification": 20,
        },
    }
    normalized, _ = normalize_experiment_action(deepcopy(decision), "no_zoom", 12)
    assert normalized["next_action"]["type"] == "retrieve"
    assert normalized["next_action"]["query"] == "additional architecture"


def test_full_loop_policy_does_not_modify_zoom_action():
    decision = {
        "next_action": {
            "type": "zoom",
            "query": "inspect atypical nuclei",
            "target_patches": ["x1_y2"],
            "magnification": 20,
        }
    }
    normalized, repair = normalize_experiment_action(deepcopy(decision), "full_loop", 12)
    assert normalized == decision
    assert repair is None


def test_model_self_assessment_cannot_override_deterministic_review_layers():
    from pathagent_v2 import _pathologist_assist_fields

    model_output = {
        "candidate_evidence_found": True,
        "ready_for_pathologist_review": True,
        "strict_evidence_sufficient": True,
    }
    verification = {
        "candidate_evidence_found": False,
        "ready_for_pathologist_review": False,
        "strict_evidence_sufficient": False,
        "cited_patches": [],
        "supporting_evidence": [],
        "opposing_or_conflicting_evidence": [],
        "missing_evidence": ["valid_visible_citation_required"],
        "review_required": True,
        "review_reason": "research_output_requires_pathologist_review",
        "contract_version": "vqa_evidence_contracts_dev50_v1",
        "output_schema_version": "pathagent_pathologist_assist_output_v1",
    }

    model_output.update(_pathologist_assist_fields(verification, "A", []))

    assert model_output["candidate_evidence_found"] is False
    assert model_output["ready_for_pathologist_review"] is False
    assert model_output["strict_evidence_sufficient"] is False
    assert model_output["review_required"] is True


def test_assistive_projection_keeps_all_deterministic_citations_aligned():
    from pathagent_v2 import _pathologist_assist_fields

    cited_patches = [{"patch_id": f"p{index}"} for index in range(8)]
    projected = _pathologist_assist_fields(
        {
            "candidate_evidence_found": True,
            "ready_for_pathologist_review": True,
            "strict_evidence_sufficient": True,
            "cited_patches": cited_patches,
            "supporting_evidence": [],
            "opposing_or_conflicting_evidence": [],
            "missing_evidence": [],
            "review_required": True,
            "review_reason": "research_output_requires_pathologist_review",
            "contract_version": "contract-v1",
            "output_schema_version": "pathagent_pathologist_assist_output_v1",
        },
        "A",
        [],
    )

    assert projected["evidence_refs"] == [f"p{index}" for index in range(8)]
    assert {
        row["patch_id"] for row in projected["cited_patches"]
    } == set(projected["evidence_refs"])
