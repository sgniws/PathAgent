import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if "torch" not in sys.modules:
    sys.modules["torch"] = types.SimpleNamespace()
if "qwen_vl_utils" not in sys.modules:
    sys.modules["qwen_vl_utils"] = types.SimpleNamespace(process_vision_info=lambda messages: ([], []))
if "datasets" not in sys.modules:
    sys.modules["datasets"] = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: None)

from models.inference import (
    ALLOWED_AGENT_ACTIONS,
    evaluate_general_vqa_action,
    evaluate_pancreatic_vqa_action,
    normalize_patho_morphology_output,
    normalize_patho_r1_output,
    prepare_patho_r1_canvas,
    sanitize_morphology_evidence_text,
    slide_llm_answer,
)
from models.trace_recorder import TraceRecorder, assert_blind_raw_trace
from models.llm_backend import ExecutorContextBudgetExceeded
from data_processing.utils import extract_coords_from_name, split_patch_for_zoom
from pathagent_v2 import _build_accumulated_executor_evidence, _filter_visible_evidence_refs


def test_patho_r1_preprocessor_always_returns_frozen_784_canvas():
    original = Image.new("RGB", (504, 1008), "white")
    prepared = prepare_patho_r1_canvas(original)
    assert prepared.size == (784, 784)
    already_frozen = Image.new("RGB", (784, 784), "white")
    assert prepare_patho_r1_canvas(already_frozen) is already_frozen


def test_morphology_normalizer_drops_diagnostic_sentence_but_keeps_visible_features():
    raw = (
        "<answer>**Benign lesion**: Fibroadenoma of breast tissue. "
        "The area is well-circumscribed with uniform glandular structures, "
        "no nuclear atypia, and a pushing border. "
        "No features suggestive of malignancy are present.</answer>"
    )
    normalized = normalize_patho_morphology_output(raw)
    assert "fibroadenoma" not in normalized.lower()
    assert "breast" not in normalized.lower()
    assert "benign" not in normalized.lower()
    assert "malignan" not in normalized.lower()
    assert "uniform glandular structures" in normalized


def test_trace_recorder_writes_paired_events_and_blind_raw_trace(tmp_path):
    recorder = TraceRecorder(
        tmp_path,
        run_id="run_1",
        trace_id="trace_1",
        rollout_id=0,
        task_input={
            "case_id": "demo_case_001",
            "slide_id": "demo_slide_001",
            "question_id": "q1",
            "question": "Is invasion visible?",
            "choices": ["Positive", "Negative", "Insufficient evidence"],
        },
    )
    call_id = recorder.before_call("plip", "retrieve", {"query": "invasion"}, step_id=1, attempt=1)
    recorder.after_call(
        call_id,
        "plip",
        "retrieve",
        {"selected": [{"patch_id": "1_2.jpg", "score": 0.8}]},
        step_id=1,
        attempt=1,
    )
    trace = recorder.finalize(
        {
            "answer": "Insufficient evidence",
            "evidence_refs": ["1_2.jpg"],
            "explanation": "No definite invasion focus.",
        }
    )

    assert [event["phase"] for event in trace["events"]] == ["before", "after"]
    assert trace["events"][0]["call_id"] == trace["events"][1]["call_id"]
    assert (tmp_path / "logs/trace_1.events.jsonl").exists()
    assert (tmp_path / "raw/trace_1.json").exists()
    assert len((tmp_path / "raw_trace.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert_blind_raw_trace(json.loads((tmp_path / "raw/trace_1.json").read_text(encoding="utf-8")))


def test_trace_recorder_never_rewrites_a_completed_historical_trace(tmp_path):
    recorder = TraceRecorder(
        tmp_path,
        run_id="run_1",
        trace_id="trace_1",
        rollout_id=0,
        task_input={"question": "q"},
    )
    recorder.finalize({"answer": None, "evidence_refs": []})
    original = (tmp_path / "raw/trace_1.json").read_bytes()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        TraceRecorder(
            tmp_path,
            run_id="run_2",
            trace_id="trace_1",
            rollout_id=0,
            task_input={"question": "different"},
        )

    assert (tmp_path / "raw/trace_1.json").read_bytes() == original


@pytest.mark.parametrize("forbidden", ["ground_truth", "gold_answer", "source_report_evidence", "结构化金标"])
def test_trace_recorder_rejects_gold_fields(tmp_path, forbidden):
    with pytest.raises(ValueError, match="forbidden gold field"):
        TraceRecorder(
            tmp_path,
            run_id="run_1",
            trace_id="trace_1",
            rollout_id=0,
            task_input={"question": "q", forbidden: "secret"},
        )


def test_pancreatic_executor_normalizes_to_five_action_protocol():
    class FakeBackend:
        def generate_messages(self, messages, max_new_tokens, temperature=None, do_sample=None, **kwargs):
            assert "pancreatic histopathology WSI agent" in messages[0]["content"]
            assert "thinking block" in messages[0]["content"]
            return json.dumps(
                {
                    "candidate_answer": "Insufficient evidence",
                    "evidence_refs": ["1_2.jpg"],
                    "evidence_summary": "No nerve is visible.",
                    "sufficient": False,
                    "missing_evidence": "nerve-tumor interface",
                    "next_action": {
                        "type": "retrieve",
                        "query": "nerve-tumor interface",
                        "target_patches": [],
                        "magnification": None,
                    },
                    "action_reason": "More evidence is required.",
                }
            )

    result = evaluate_pancreatic_vqa_action(
        FakeBackend(),
        None,
        "Patch 1_2.jpg: fibrotic stroma.",
        "Is perineural invasion present?",
        ["Positive", "Negative", "Insufficient evidence"],
        visible_patch_ids=["1_2.jpg"],
        remaining_patch_count=5,
    )

    assert result["next_action"]["type"] in ALLOWED_AGENT_ACTIONS
    assert result["next_action"]["type"] == "retrieve"
    assert "thinking_steps" not in result


def test_general_v2_separates_benchmark_answer_from_visual_sufficiency():
    class FakeBackend:
        def generate_messages(self, messages, max_new_tokens, temperature=None, do_sample=None, **kwargs):
            system = messages[0]["content"]
            assert "general histopathology whole-slide-image VQA agent" in system
            assert "pancreatic VQA cohort" not in system
            assert "best-effort benchmark answer" in system
            return json.dumps(
                {
                    "benchmark_answer": "positive",
                    "evidence_sufficient": False,
                    "abstain_recommended": True,
                    "unsupported_answer_reason": "Receptor status is not established by H&E morphology.",
                    "evidence_refs": [],
                    "evidence_summary": "No direct receptor evidence is available.",
                    "missing_evidence": "non-visual report-only biomarker result",
                    "next_action": {
                        "type": "abstain",
                        "query": "",
                        "target_patches": [],
                        "magnification": None,
                    },
                    "action_reason": "The requested fact is not visually establishable.",
                }
            )

    result = evaluate_general_vqa_action(
        FakeBackend(),
        None,
        "Patch 1_2.jpg: invasive glands in fibrotic stroma.",
        "What is the estrogen receptor result?",
        ["positive", "negative", "equivocal", "not reported"],
        visible_patch_ids=["1_2.jpg"],
        remaining_patch_count=5,
    )

    assert result["benchmark_answer"] == "positive"
    assert result["candidate_answer"] == "positive"
    assert result["evidence_sufficient"] is False
    assert result["abstain_recommended"] is True
    assert result["next_action"]["type"] == "abstain"
    assert result["parse_status"] == "valid_action_json"


def test_contract_v1_prompt_has_no_literal_false_sufficiency_anchor():
    class FakeBackend:
        def generate_messages(self, messages, max_new_tokens, **kwargs):
            system = messages[0]["content"]
            assert '"evidence_sufficient": false' not in system
            assert '"evidence_sufficient": "true or false (advisory)"' in system
            assert "separate deterministic public-contract verifier" in system
            return json.dumps(
                {
                    "benchmark_answer": "A",
                    "evidence_sufficient": True,
                    "abstain_recommended": False,
                    "unsupported_answer_reason": "",
                    "evidence_refs": ["p1"],
                    "evidence_summary": "Visible morphology supports A.",
                    "missing_evidence": "",
                    "next_action": {
                        "type": "answer",
                        "query": "",
                        "target_patches": [],
                        "magnification": None,
                    },
                    "action_reason": "Answer from current morphology.",
                }
            )

    result = evaluate_general_vqa_action(
        FakeBackend(),
        None,
        "Patch p1: direct morphology.",
        "Question",
        ["A", "B", "C", "D"],
        visible_patch_ids=["p1"],
        evidence_policy="contract_v1",
    )

    assert result["evidence_sufficient"] is True
    assert result["next_action"]["type"] == "answer"


def test_general_v2_normalizes_assistive_recommendation_and_differential():
    class FakeBackend:
        def generate_messages(self, messages, max_new_tokens, **kwargs):
            system = messages[0]["content"]
            assert '"provisional_recommendation"' in system
            assert '"ranked_differential"' in system
            assert "model suggestions only" in system
            return json.dumps(
                {
                    "provisional_recommendation": "B",
                    "ranked_differential": ["A", "B", "invalid", "A", "C"],
                    "advisory_evidence_state": {
                        "candidate_evidence_found": True,
                        "ready_for_pathologist_review": True,
                    },
                    "evidence_sufficient": True,
                    "abstain_recommended": False,
                    "unsupported_answer_reason": "",
                    "evidence_refs": ["p1"],
                    "evidence_summary": "Patch p1 contains directly visible morphology.",
                    "missing_evidence": "",
                    "next_action": {
                        "type": "answer",
                        "query": "",
                        "target_patches": [],
                        "magnification": None,
                    },
                    "action_reason": "Current evidence permits a provisional recommendation.",
                }
            )

    result = evaluate_general_vqa_action(
        FakeBackend(),
        None,
        "Patch p1: visible morphology.",
        "Question",
        ["A", "B", "C", "D"],
        visible_patch_ids=["p1"],
        evidence_policy="contract_v1",
    )

    assert result["provisional_recommendation"] == "B"
    assert result["benchmark_answer"] == "B"
    assert result["ranked_differential"] == ["A", "C"]
    assert result["advisory_evidence_state"] == {
        "candidate_evidence_found": True,
        "ready_for_pathologist_review": True,
        "strict_evidence_sufficient": True,
    }
    assert result["parse_status"] == "normalized_action_json"


def test_general_v2_rejects_non_official_benchmark_answer():
    class FakeBackend:
        def generate_messages(self, messages, max_new_tokens, **kwargs):
            return json.dumps(
                {
                    "benchmark_answer": "Insufficient evidence",
                    "evidence_sufficient": False,
                    "abstain_recommended": True,
                    "unsupported_answer_reason": "No direct evidence.",
                    "evidence_refs": [],
                    "evidence_summary": "",
                    "missing_evidence": "direct morphology",
                    "next_action": {"type": "abstain", "query": "", "target_patches": [], "magnification": None},
                    "action_reason": "Stop.",
                }
            )

    result = evaluate_general_vqa_action(
        FakeBackend(), None, "", "Question", ["A", "B", "C", "D"]
    )

    assert result["benchmark_answer"] == "A"
    assert result["parse_status"] == "invalid_action_json"


def test_general_v2_stops_futile_search_for_report_only_target():
    class FakeBackend:
        def generate_messages(self, messages, max_new_tokens, **kwargs):
            return json.dumps(
                {
                    "benchmark_answer": "2.8 cm",
                    "evidence_sufficient": False,
                    "abstain_recommended": False,
                    "unsupported_answer_reason": "No direct size measurement is visible.",
                    "evidence_refs": [],
                    "evidence_summary": "No scale is present.",
                    "missing_evidence": "measurement scale",
                    "next_action": {
                        "type": "retrieve",
                        "query": "tumor with ruler",
                        "target_patches": [],
                        "magnification": None,
                    },
                    "action_reason": "Search for a ruler.",
                }
            )

    result = evaluate_general_vqa_action(
        FakeBackend(),
        None,
        "",
        "What was the size of the primary invasive carcinoma?",
        ["2.8 cm", "4 mm", "0.4 cm", "3 mm"],
    )

    assert result["benchmark_answer"] == "2.8 cm"
    assert result["next_action"]["type"] == "abstain"
    assert result["abstain_recommended"] is True
    assert result["parse_status"] == "normalized_action_json"
    assert "report_only_search_changed_to_abstain" in result["semantic_repairs"]


def test_recursive_zoom_preserves_global_coordinates_and_relative_magnification(tmp_path):
    first_zoom = tmp_path / "78400_29984_m20_76352_28960.jpg"
    Image.new("RGB", (1024, 1024), "white").save(first_zoom)

    assert extract_coords_from_name(first_zoom.name) == (78400, 29984)
    subpatches = split_patch_for_zoom(first_zoom, 40, source_magnification=20)

    assert len(subpatches) == 4
    assert [coords for _, coords in subpatches] == [
        (78400, 29984),
        (78912, 29984),
        (78400, 30496),
        (78912, 30496),
    ]
    assert all(image.size == (512, 512) for image, _ in subpatches)


@pytest.mark.parametrize(
    "question",
    [
        "What is the closest margin of resection?",
        "What stage is the tumor classified as?",
        "What is the patient's age?",
        "Who signed the slide?",
        "What were the results of the immunohistochemical staining?",
    ],
)
def test_general_v2_stops_search_for_nonvisual_report_metadata(question):
    class FakeBackend:
        def generate_messages(self, messages, max_new_tokens, **kwargs):
            return json.dumps(
                {
                    "benchmark_answer": "A",
                    "evidence_sufficient": False,
                    "abstain_recommended": False,
                    "unsupported_answer_reason": "Not established by H&E patches.",
                    "evidence_refs": [],
                    "evidence_summary": "",
                    "missing_evidence": "report metadata",
                    "next_action": {"type": "retrieve", "query": "more evidence", "target_patches": [], "magnification": None},
                    "action_reason": "Search.",
                }
            )

    result = evaluate_general_vqa_action(FakeBackend(), None, "", question, ["A", "B", "C", "D"])
    assert result["next_action"]["type"] == "abstain"
    assert result["parse_status"] == "normalized_action_json"


def test_agent_decision_drops_evidence_refs_outside_visible_state():
    decision = {
        "evidence_refs": ["28672_16384.jpg", "16384_3684.jpg", "28672_16384.jpg"],
        "raw_texts": {"action_raw": "original model output"},
    }

    result = _filter_visible_evidence_refs(
        decision,
        ["28672_16384.jpg", "16384_36864.jpg"],
    )

    assert result["evidence_refs"] == ["28672_16384.jpg"]
    assert result["evidence_ref_validation"] == {
        "status": "filtered",
        "dropped_invisible_refs": ["16384_3684.jpg"],
    }
    assert result["raw_texts"]["action_raw"] == "original model output"


def test_executor_evidence_is_cumulative_and_visible_ids_match_descriptions():
    text, visible = _build_accumulated_executor_evidence(
        {"a.jpg": "first finding", "b.jpg": "second finding"},
        ["a.jpg", "missing.jpg", "b.jpg", "a.jpg"],
        magnification=5,
        char_limit=10_000,
    )

    assert visible == ["a.jpg", "b.jpg"]
    assert "first finding" in text
    assert "second finding" in text
    assert "missing.jpg" not in text


def test_executor_evidence_hard_limit_never_silently_truncates():
    with pytest.raises(ExecutorContextBudgetExceeded, match="hard limit"):
        _build_accumulated_executor_evidence(
            {"a.jpg": "x" * 500},
            ["a.jpg"],
            magnification=5,
            char_limit=100,
        )



def test_patho_letter_answer_is_mapped_to_full_choice_after_thinking_is_removed():
    raw = "<think>private rationale</think>\n<answer>B</answer>"
    assert normalize_patho_r1_output(raw, ["PDAC", "IPMN", "Insufficient evidence"]) == "<answer>IPMN</answer>"


def test_patho_morphology_keeps_visible_fields_without_thinking_or_diagnosis_label():
    raw = (
        "<think>private diagnostic rationale</think>\n"
        "<answer>VISIBLE: Irregular infiltrative glands in fibrotic stroma.\n"
        "ABSENT_OR_NOT_SEEN: Pseudopapillary structures are not seen.\n"
        "UNCERTAIN: Tumor extent cannot be assessed.\n"
        "QUALITY: adequate.</answer>"
    )
    result = normalize_patho_morphology_output(
        raw,
        ["Conventional pancreatic ductal adenocarcinoma", "Intraductal epithelial neoplasm"],
    )
    assert result.startswith("[PATHO_MORPHOLOGY | pancreatic_morphology_v4]")
    assert "Irregular infiltrative glands" in result
    assert "private diagnostic rationale" not in result


def test_patho_morphology_rejects_bare_choice_letter():
    result = normalize_patho_morphology_output(
        "<think>claims a tumor</think><answer>D</answer>",
        ["PDAC", "IPMN", "Insufficient evidence"],
    )
    assert "answer label rather than visual evidence" in result
    assert "<answer>D</answer>" not in result


def test_patho_morphology_redacts_exact_candidate_label():
    result = normalize_patho_morphology_output(
        "<answer>VISIBLE: Intraductal epithelial neoplasm with papillary structures.</answer>",
        ["Intraductal epithelial neoplasm", "Insufficient evidence"],
    )
    assert "Intraductal epithelial neoplasm" not in result
    assert "papillary structures" in result


def test_patho_morphology_coerces_diagnostic_prose_to_four_evidence_fields():
    raw = (
        "<think>private diagnostic discussion and KRAS testing</think>"
        "<answer>The H&E morphology is most consistent with pancreatic ductal adenocarcinoma. "
        "Key features include infiltrative glands with nuclear pleomorphism, desmoplastic stroma, and luminal necrosis. "
        "CK7 and KRAS testing could confirm the diagnosis.</answer>"
    )
    result = normalize_patho_morphology_output(
        raw,
        ["Conventional pancreatic ductal adenocarcinoma", "Intraductal epithelial neoplasm"],
    )
    assert "VISIBLE: infiltrative glands with nuclear pleomorphism" in result
    assert "ABSENT_OR_NOT_SEEN:" in result
    assert "UNCERTAIN:" in result
    assert "QUALITY:" in result
    assert "adenocarcinoma" not in result.casefold()
    assert "KRAS" not in result


def test_morphology_sanitizer_removes_organ_and_diagnosis_but_keeps_architecture():
    result = sanitize_morphology_evidence_text(
        "The breast tissue shows pancreatic ductal adenocarcinoma with irregular infiltrative glands and desmoplasia. "
        "CK7 and KRAS testing could confirm the diagnosis.",
        ["Conventional pancreatic ductal adenocarcinoma", "Insufficient evidence"],
    )
    assert "breast" not in result.casefold()
    assert "adenocarcinoma" not in result.casefold()
    assert "KRAS" not in result
    assert "irregular infiltrative glands" in result
    assert "desmoplasia" in result



def test_final_answer_recovers_only_an_exact_choice_from_truncated_json():
    class FakeBackend:
        def generate_messages(self, messages, max_new_tokens, **kwargs):
            return '{"answer":"Not demonstrated","evidence_refs":["1_2.jpg"],"confidence":0.'

    result = slide_llm_answer(
        FakeBackend(),
        None,
        "Patch 1_2.jpg: no definite nerve invasion.",
        "Is perineural invasion present?",
        ["Present", "Not demonstrated", "Insufficient evidence"],
        question_type="judgment",
    )

    assert result["answer"] == "Not demonstrated"
    assert result["parse_status"] == "recovered_exact_choice"


def test_final_answer_uses_safe_choice_when_malformed_answer_is_not_a_choice():
    class FakeBackend:
        def generate_messages(self, messages, max_new_tokens, **kwargs):
            return "{"

    result = slide_llm_answer(
        FakeBackend(),
        None,
        "Patch 1_2.jpg: limited tissue.",
        "What is the diagnosis?",
        ["PDAC", "IPMN", "Insufficient evidence"],
        question_type="single_choice",
    )

    assert result["answer"] == "Insufficient evidence"
    assert result["parse_status"] == "safe_fallback"
