import json
from pathlib import Path

from models.evidence_contract import (
    candidate_record,
    choose_better_candidate,
    evaluate_contract_state,
    positive_term_match,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = json.loads(
    (ROOT / "configs/vqa_evidence_contracts_v0.1.json").read_text(
        encoding="utf-8"
    )
)
ONTOLOGY = json.loads(
    (ROOT / "configs/vqa_ontology_v0.1.json").read_text(encoding="utf-8")
)


def _metadata():
    return {
        "p1": {
            "x_level0": 0,
            "y_level0": 0,
            "width_level0": 100,
            "height_level0": 100,
            "magnification": 5,
        },
        "p2": {
            "x_level0": 200,
            "y_level0": 0,
            "width_level0": 100,
            "height_level0": 100,
            "magnification": 5,
        },
        "z1": {
            "x_level0": 0,
            "y_level0": 0,
            "width_level0": 25,
            "height_level0": 25,
            "magnification": 20,
            "parent_patch_id": "p1",
        },
    }


def _descriptions():
    return {
        "p1": "VISIBLE: Infiltrative glandular architecture is present.\nABSENT_OR_NOT_SEEN: none",
        "p2": "VISIBLE: Irregular infiltrative glands are present.\nABSENT_OR_NOT_SEEN: none",
        "z1": "VISIBLE: Nuclear atypia is present.\nABSENT_OR_NOT_SEEN: none",
    }


def _evaluate(question_type, choices, candidate, refs, ranked_differential=None):
    return evaluate_contract_state(
        contracts=CONTRACTS,
        ontology=ONTOLOGY,
        question_type=question_type,
        choices=choices,
        candidate_answer=candidate,
        evidence_tier="strict",
        visible_patch_ids=["p1", "p2", "z1"],
        reported_evidence_refs=refs,
        descriptions=_descriptions(),
        metadata=_metadata(),
        clean_patch_ids={"p1", "p2"},
        ranked_differential=ranked_differential,
    )


def test_positive_term_match_rejects_negated_occurrence():
    assert positive_term_match("Infiltrative glands are present.", "infiltrat")
    assert not positive_term_match("No infiltrative glands are identified.", "infiltrat")


def test_unique_structure_contract_with_valid_citations_is_sufficient():
    choices = [
        "浸润性不规则腺体生长",
        "实性伴假乳头状结构",
        "筛孔状导管内生长",
        "腺泡样生长",
    ]
    result = _evaluate("structure", choices, choices[0], ["p1", "p2"])
    assert result["visible_passing_answers"] == [choices[0]]
    assert result["cited_passing_answers"] == [choices[0]]
    assert result["evidence_found"] is True
    assert result["citation_valid"] is True
    assert result["citation_supports_answer"] is True
    assert result["evidence_sufficient"] is True
    assert result["candidate_evidence_found"] is True
    assert result["ready_for_pathologist_review"] is True
    assert result["strict_evidence_sufficient"] is True
    assert result["evidence_sufficient"] == result["strict_evidence_sufficient"]
    assert result["review_required"] is True
    assert result["output_schema_version"] == "pathagent_pathologist_assist_output_v1"


def test_one_clean_cited_patch_can_be_review_ready_but_not_strict():
    choices = [
        "浸润性不规则腺体生长",
        "实性伴假乳头状结构",
        "筛孔状导管内生长",
        "腺泡样生长",
    ]
    result = _evaluate("structure", choices, choices[0], ["p1"])

    assert result["candidate_evidence_found"] is True
    assert result["ready_for_pathologist_review"] is True
    assert result["strict_evidence_sufficient"] is False
    assert result["evidence_sufficient"] is False
    assert [row["patch_id"] for row in result["supporting_evidence"]] == ["p1"]
    assert isinstance(result["opposing_or_conflicting_evidence"], list)
    assert "strict_contract_not_satisfied" in result["missing_evidence"]


def test_no_valid_citation_cannot_promote_any_review_layer():
    choices = [
        "浸润性不规则腺体生长",
        "实性伴假乳头状结构",
        "筛孔状导管内生长",
        "腺泡样生长",
    ]
    result = _evaluate("structure", choices, choices[0], ["invisible"])

    assert result["citation_valid"] is False
    assert result["candidate_evidence_found"] is False
    assert result["ready_for_pathologist_review"] is False
    assert result["strict_evidence_sufficient"] is False
    assert result["supporting_evidence"] == []
    assert "valid_visible_citation_required" in result["missing_evidence"]


def test_visible_evidence_does_not_replace_invalid_or_insufficient_citations():
    choices = [
        "浸润性不规则腺体生长",
        "实性伴假乳头状结构",
        "筛孔状导管内生长",
        "腺泡样生长",
    ]
    result = _evaluate("structure", choices, choices[0], ["p1", "invisible"])
    assert result["evidence_found"] is True
    assert result["citation_valid"] is False
    assert result["citation_supports_answer"] is False
    assert result["evidence_sufficient"] is False
    assert result["invalid_evidence_refs"] == ["invisible"]


def test_best_candidate_prefers_contract_coverage_then_stability_then_earlier():
    low = candidate_record(
        {"benchmark_answer": "A", "evidence_refs": []},
        {
            "candidate_support_score": 0.5,
            "candidate_citation_support_score": 0.0,
            "valid_evidence_refs": [],
        },
        attempt=1,
        stability_count=1,
    )
    stable = candidate_record(
        {"benchmark_answer": "B", "evidence_refs": []},
        {
            "candidate_support_score": 0.5,
            "candidate_citation_support_score": 0.0,
            "valid_evidence_refs": [],
        },
        attempt=2,
        stability_count=2,
    )
    higher = candidate_record(
        {"benchmark_answer": "C", "evidence_refs": []},
        {
            "candidate_support_score": 0.75,
            "candidate_citation_support_score": 0.0,
            "valid_evidence_refs": [],
        },
        attempt=3,
        stability_count=1,
    )
    best = choose_better_candidate(None, low)
    best = choose_better_candidate(best, stable)
    assert best["answer"] == "B"
    best = choose_better_candidate(best, higher)
    assert best["answer"] == "C"
