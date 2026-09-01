from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.trace_recorder import assert_blind_raw_trace


ALLOWED_ACTIONS = {"retrieve", "inspect", "zoom", "answer", "abstain"}
ALLOWED_STOP_REASONS = {
    "answered",
    "answered_after_zoom",
    "abstained",
    "budget_exhausted",
    "no_additional_evidence",
    "no_evidence",
    "runtime_failure",
    "executor_api_failure",
    "context_budget_exceeded",
    "one_shot_complete",
}
ASSISTIVE_OUTPUT_SCHEMA = "pathagent_pathologist_assist_output_v1"


def _audit_assistive_fields(container, choices, context, verification=None):
    errors = []
    required_lists = (
        "ranked_differential",
        "cited_patches",
        "supporting_evidence",
        "opposing_or_conflicting_evidence",
        "missing_evidence",
    )
    required_bools = (
        "candidate_evidence_found",
        "ready_for_pathologist_review",
        "strict_evidence_sufficient",
        "review_required",
    )
    for field in required_lists:
        if not isinstance(container.get(field), list):
            errors.append(f"{context} lacks explicit list {field}")
    for field in required_bools:
        if not isinstance(container.get(field), bool):
            errors.append(f"{context} lacks boolean {field}")
    provisional = container.get("provisional_recommendation")
    if provisional not in choices:
        errors.append(f"{context} provisional_recommendation is not an official choice")
    differential = container.get("ranked_differential") or []
    if any(value not in choices or value == provisional for value in differential):
        errors.append(f"{context} ranked_differential contains an invalid choice")
    if len(differential) != len(set(differential)):
        errors.append(f"{context} ranked_differential contains duplicates")
    candidate = container.get("candidate_evidence_found") is True
    ready = container.get("ready_for_pathologist_review") is True
    strict = container.get("strict_evidence_sufficient") is True
    if strict and not ready:
        errors.append(f"{context} strict evidence does not imply review readiness")
    if ready and not candidate:
        errors.append(f"{context} review readiness does not imply candidate evidence")
    if container.get("evidence_sufficient") is not strict:
        errors.append(f"{context} legacy evidence_sufficient is not the strict alias")
    if container.get("review_required") is not True:
        errors.append(f"{context} review_required must remain true")
    if not str(container.get("review_reason") or "").strip():
        errors.append(f"{context} review_reason is empty")
    if not str(container.get("contract_version") or "").strip():
        errors.append(f"{context} contract_version is empty")
    if container.get("output_schema_version") != ASSISTIVE_OUTPUT_SCHEMA:
        errors.append(f"{context} has the wrong output_schema_version")
    cited_ids = {
        row.get("patch_id")
        for row in container.get("cited_patches") or []
        if isinstance(row, dict)
    }
    evidence_refs = set(container.get("evidence_refs") or [])
    if cited_ids != evidence_refs:
        errors.append(f"{context} cited_patches do not match evidence_refs")
    for field in ("supporting_evidence", "opposing_or_conflicting_evidence"):
        rows = container.get(field) or []
        if any(
            not isinstance(row, dict) or row.get("patch_id") not in cited_ids
            for row in rows
        ):
            errors.append(f"{context} {field} contains uncited evidence")
    if isinstance(verification, dict):
        for field in (
            "candidate_evidence_found",
            "ready_for_pathologist_review",
            "strict_evidence_sufficient",
            "cited_patches",
            "supporting_evidence",
            "opposing_or_conflicting_evidence",
            "missing_evidence",
            "contract_version",
            "output_schema_version",
        ):
            if container.get(field) != verification.get(field):
                errors.append(
                    f"{context} {field} differs from deterministic verification"
                )
    return errors


def parse_args():
    parser = argparse.ArgumentParser(description="Audit a completed PathAgent trace v2/v3 run.")
    parser.add_argument("--trace_dir", required=True)
    parser.add_argument("--expected_traces", type=int, default=None)
    parser.add_argument("--expected_rollouts_per_question", type=int, default=None)
    return parser.parse_args()


def _audit_retriever_v3(trace):
    runtime_retriever = trace.get("runtime", {}).get("retriever")
    if not isinstance(runtime_retriever, dict):
        return []
    errors = []
    expected_backend = runtime_retriever.get("backend")
    expected_model = runtime_retriever.get("model_id")
    if not expected_backend or not expected_model:
        errors.append("Trace v3 runtime retriever identity is incomplete")
        return errors
    events = trace.get("events", [])
    if any(event.get("component") == "plip" for event in events):
        errors.append("Trace v3 contains a legacy plip component")
    retriever_events = [
        event for event in events if event.get("component") == "retriever"
    ]
    before_by_call = {
        event.get("call_id"): event
        for event in retriever_events
        if event.get("phase") == "before"
    }
    after_by_call = {
        event.get("call_id"): event
        for event in retriever_events
        if event.get("phase") == "after"
    }
    for call_id, before_event in before_by_call.items():
        after_event = after_by_call.get(call_id)
        request = before_event.get("request", {})
        response = after_event.get("response", {}) if after_event else {}
        for container, phase in ((request, "before"), (response, "after")):
            if container.get("backend") != expected_backend:
                errors.append(
                    f"retriever {phase} backend drift for call_id={call_id}"
                )
            if container.get("model_id") != expected_model:
                errors.append(
                    f"retriever {phase} model drift for call_id={call_id}"
                )

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"plip_score", "parent_plip_score"}:
                    errors.append(f"Trace v3 contains legacy score field {key}")
                if key == "retriever_backend" and item != expected_backend:
                    errors.append("Trace v3 ranking backend differs from runtime")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(events)
    return errors


def _audit_evidence_handle_state(payload):
    errors = []
    visible_patches = list(payload.get("state_visible_patches") or [])
    visible_handles = list(payload.get("state_visible_evidence_handles") or [])
    handle_map = payload.get("state_evidence_handle_map")
    decision = payload.get("decision") or {}
    validation = decision.get("evidence_handle_validation")
    if not isinstance(handle_map, dict) or not isinstance(validation, dict):
        return ["short-handle agent action lacks handle mapping or validation"]
    if list(handle_map) != visible_handles or list(handle_map.values()) != visible_patches:
        errors.append("short-handle visible mapping differs from visible patch state")
    if validation.get("protocol") != "short_handle_v1":
        errors.append("short-handle validation has the wrong protocol")

    requested_refs = validation.get("requested_evidence_handles") or []
    expected_ref_records = []
    expected_refs = []
    for handle in requested_refs:
        if isinstance(handle, str) and handle in handle_map:
            patch_id = handle_map[handle]
            if patch_id not in expected_refs:
                expected_refs.append(patch_id)
                expected_ref_records.append({"handle": handle, "patch_id": patch_id})
    if validation.get("resolved_evidence_refs") != expected_ref_records:
        errors.append("short-handle evidence resolution differs from frozen mapping")
    if decision.get("evidence_refs") != expected_refs[:8]:
        errors.append("short-handle resolved evidence refs differ from decision")

    requested_targets = validation.get("requested_target_handles") or []
    expected_target_records = []
    expected_targets = []
    for handle in requested_targets:
        if isinstance(handle, str) and handle in handle_map:
            patch_id = handle_map[handle]
            if patch_id not in expected_targets:
                expected_targets.append(patch_id)
                expected_target_records.append({"handle": handle, "patch_id": patch_id})
    if validation.get("resolved_target_patches") != expected_target_records:
        errors.append("short-handle target resolution differs from frozen mapping")
    if validation.get("dropped_invalid_target_handles"):
        errors.append("Executor targeted evidence handles outside the visible state")
    return errors


def audit_trace(trace):
    errors = []
    try:
        assert_blind_raw_trace(trace)
    except ValueError as exc:
        errors.append(str(exc))
    before = Counter(event.get("call_id") for event in trace.get("events", []) if event.get("phase") == "before")
    after = Counter(event.get("call_id") for event in trace.get("events", []) if event.get("phase") == "after")
    if before != after:
        errors.append("unpaired before/after calls")
    if trace.get("execution", {}).get("status") != "completed":
        errors.append(f"execution status={trace.get('execution', {}).get('status')}")
    if trace.get("schema_version") == "pathagent_trace_v3":
        errors.extend(_audit_retriever_v3(trace))
    task_input = trace.get("task_input", {})
    evidence_policy = trace.get("runtime", {}).get("evidence_policy", "model_v1")
    evidence_reference_protocol = trace.get("runtime", {}).get(
        "evidence_reference_protocol"
    )
    assistive_v1 = (
        trace.get("runtime", {}).get("output_schema_version")
        == ASSISTIVE_OUTPUT_SCHEMA
    )
    choices = task_input.get("choices") or []
    answer = trace.get("final_output", {}).get("answer")
    question_type = task_input.get("question_type")
    if question_type == "multiple_choice":
        if not isinstance(answer, list) or any(item not in choices for item in answer):
            errors.append("final multiple-choice answer is not a list of provided choices")
    elif choices and answer not in choices:
        errors.append("final answer is not one of the provided choices")
    evidence_index_events = [
        event
        for event in trace.get("events", [])
        if event.get("phase") == "state" and event.get("operation") == "evidence_index_loaded"
    ]
    known_patches = {
        patch.get("patch_id")
        for event in evidence_index_events
        for patch in event.get("payload", {}).get("patches", [])
    }
    registered_evidence = [
        event.get("payload", {}).get("patch", {})
        for event in trace.get("events", [])
        if event.get("phase") == "state" and event.get("operation") == "evidence_registered"
    ]
    known_patches.update(patch.get("patch_id") for patch in registered_evidence)
    for patch in registered_evidence:
        evidence_path = patch.get("path")
        if not patch.get("patch_id") or not evidence_path or not Path(evidence_path).is_file():
            errors.append("registered evidence is missing its persisted image")
    for event in trace.get("events", []):
        if event.get("phase") != "state" or event.get("operation") != "agent_action":
            continue
        payload = event.get("payload", {})
        decision = payload.get("decision", {})
        if evidence_reference_protocol == "short_handle_v1":
            errors.extend(_audit_evidence_handle_state(payload))
        if decision.get("parse_status") not in {"valid_action_json", "normalized_action_json"}:
            errors.append(f"invalid Executor parse status={decision.get('parse_status')}")
        if decision.get("evidence_ref_validation", {}).get("dropped_invisible_refs"):
            if evidence_policy == "contract_v1":
                if decision.get("citation_valid") is not False:
                    errors.append("contract_v1 failed to mark invisible citations invalid")
                errors.append("Executor cited patch identifiers outside the visible state")
            else:
                errors.append("Executor cited patch identifiers outside the visible state")
        action = decision.get("next_action", {}).get("type")
        if action not in ALLOWED_ACTIONS:
            errors.append(f"invalid agent action={action}")
        if evidence_policy == "contract_v1":
            verification = decision.get("evidence_contract_verification")
            if not isinstance(verification, dict) or verification.get("policy_version") != "contract_v1":
                errors.append("contract_v1 agent action lacks deterministic verification")
            for field in (
                "evidence_found",
                "citation_valid",
                "citation_supports_answer",
                "evidence_sufficient",
            ):
                if not isinstance(decision.get(field), bool):
                    errors.append(f"contract_v1 agent action lacks boolean {field}")
            if decision.get("evidence_sufficient") is True and action != "answer":
                errors.append("contract_v1 sufficient action did not stop with answer")
            if assistive_v1:
                errors.extend(
                    _audit_assistive_fields(
                        decision,
                        choices,
                        "contract_v1 agent action",
                        verification=verification,
                    )
                )
                if decision.get("provisional_recommendation") != decision.get(
                    "benchmark_answer"
                ):
                    errors.append(
                        "contract_v1 provisional recommendation differs from benchmark answer"
                    )
        visible = set(payload.get("state_visible_patches", []))
        refs = set(decision.get("evidence_refs", []))
        if not refs.issubset(visible):
            errors.append("agent action cites evidence outside the visible state")
    final_refs = set(trace.get("final_output", {}).get("evidence_refs", []))
    if known_patches and not final_refs.issubset(known_patches):
        errors.append("final output cites unknown patch identifiers")
    if trace.get("final_output", {}).get("parse_status") in {"safe_fallback", "recovered_exact_choice"}:
        errors.append(f"degraded final parse status={trace['final_output']['parse_status']}")
    morphology_v2 = trace.get("runtime", {}).get("patho_cache_schema") == "patho_morphology_cache_v2"
    general_v2 = trace.get("runtime", {}).get("executor_protocol") == "general_v2"
    if general_v2:
        final_output = trace.get("final_output", {})
        if not isinstance(final_output.get("evidence_sufficient"), bool):
            errors.append("general_v2 final output is missing boolean evidence_sufficient")
        if not isinstance(final_output.get("abstain_recommended"), bool):
            errors.append("general_v2 final output is missing boolean abstain_recommended")
        if choices and final_output.get("answer") not in choices:
            errors.append("general_v2 benchmark_answer is not an exact official choice")
        if evidence_policy == "contract_v1":
            for field in (
                "evidence_found",
                "citation_valid",
                "citation_supports_answer",
            ):
                if not isinstance(final_output.get(field), bool):
                    errors.append(f"contract_v1 final output is missing boolean {field}")
            verification = final_output.get("evidence_contract_verification")
            if not isinstance(verification, dict):
                errors.append("contract_v1 final output lacks evidence contract verification")
            elif verification and verification.get("policy_version") != "contract_v1":
                errors.append("contract_v1 verification has the wrong policy version")
            if final_output.get("evidence_sufficient") is True:
                if not all(
                    final_output.get(field) is True
                    for field in (
                        "evidence_found",
                        "citation_valid",
                        "citation_supports_answer",
                    )
                ):
                    errors.append("contract_v1 sufficient answer failed a component gate")
                if final_output.get("abstain_recommended") is not False:
                    errors.append("contract_v1 sufficient answer recommends abstention")
            elif final_output.get("abstain_recommended") is not True:
                errors.append("contract_v1 insufficient answer does not recommend abstention")
            if assistive_v1:
                errors.extend(
                    _audit_assistive_fields(
                        final_output,
                        choices,
                        "contract_v1 final output",
                        verification=verification,
                    )
                )
                if final_output.get("provisional_recommendation") != final_output.get(
                    "answer"
                ):
                    errors.append(
                        "final provisional recommendation differs from benchmark answer"
                    )
    if assistive_v1:
        runtime = trace.get("runtime", {})
        if runtime.get("evidence_sufficient_compatibility_mapping") != "strict_evidence_sufficient":
            errors.append("assistive runtime lacks the strict legacy-field mapping")
        prompt_hash = str(runtime.get("executor_prompt_sha256") or "")
        if len(prompt_hash) != 64 or any(char not in "0123456789abcdef" for char in prompt_hash):
            errors.append("assistive runtime lacks a valid Executor prompt hash")
        if runtime.get("executor_provider") == "deepseek":
            if runtime.get("executor_endpoint_category") != "external_https_text_api":
                errors.append("DeepSeek runtime has an invalid endpoint category")
            returned_models = {
                event.get("response", {}).get("returned_model")
                for event in trace.get("events", [])
                if event.get("phase") == "after"
                and event.get("component") == "deepseek_executor"
                and event.get("status") == "ok"
            }
            returned_models.discard(None)
            if len(returned_models) != 1:
                errors.append(
                    "DeepSeek trace does not contain one stable returned model identity"
                )
    stop_reason = trace.get("final_output", {}).get("stop_reason")
    if morphology_v2 and stop_reason not in ALLOWED_STOP_REASONS:
        errors.append(f"invalid stop_reason={stop_reason}")
    for event in trace.get("events", []):
        request = event.get("request", {})
        response = event.get("response", {})
        if event.get("component") in {"qwen_executor", "deepseek_executor"} and "<think>" in str(response.get("raw_output", "")):
            errors.append("Executor output contains <think>")
        if event.get("component") == "patho_r1" and "<think>" in str(response.get("returned_output", "")):
            errors.append("Patho-R1 returned_output contains <think>")
        if general_v2 and event.get("component") == "patho_r1" and event.get("phase") == "before":
            if request.get("morphology_only") is not True:
                errors.append("general_v2 called Patho-R1 outside morphology-only mode")
            if request.get("question") is not None or request.get("choices") is not None:
                errors.append("general_v2 Patho-R1 request trace contains benchmark question or choices")
            prompt = str(request.get("prompt", ""))
            question = str(task_input.get("question") or "").strip()
            if question and question in prompt:
                errors.append("general_v2 Patho-R1 prompt contains the benchmark question")
        if morphology_v2 and event.get("component") in {"patho_r1", "patho_r1_cache"} and event.get("phase") == "after":
            returned = str(response.get("returned_output", ""))
            prompt_version = trace.get("runtime", {}).get("patho_prompt_version")
            if not prompt_version or f"[PATHO_MORPHOLOGY | {prompt_version}]" not in returned:
                errors.append("Patho-R1 output does not follow the recorded morphology protocol")
            returned_lines = {line.strip().casefold() for line in returned.splitlines() if line.strip()}
            choices_casefold = [str(choice).strip().casefold() for choice in choices]
            if any(choice and choice in returned_lines for choice in choices_casefold):
                errors.append("Patho-R1 morphology output contains a candidate answer label")
    budget_indexes = [
        index for index, event in enumerate(trace.get("events", [])) if event.get("operation") == "step_budget_exhausted"
    ]
    if budget_indexes and budget_indexes[-1] != len(trace.get("events", [])) - 1:
        errors.append("events occurred after step_budget_exhausted")
    return errors


def main():
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    traces = []
    for path in sorted((trace_dir / "raw").glob("*.json")):
        traces.append(json.loads(path.read_text(encoding="utf-8")))
    errors = {}
    trace_ids = [trace.get("trace_id") for trace in traces]
    if len(trace_ids) != len(set(trace_ids)):
        errors["run"] = ["duplicate trace_id"]
    if args.expected_traces is not None and len(traces) != args.expected_traces:
        errors.setdefault("run", []).append(f"expected {args.expected_traces} traces, found {len(traces)}")
    by_question = Counter(trace.get("task_input", {}).get("question_id") for trace in traces)
    if args.expected_rollouts_per_question is not None:
        invalid = {key: value for key, value in by_question.items() if value != args.expected_rollouts_per_question}
        if invalid:
            errors.setdefault("run", []).append(f"invalid rollout counts: {invalid}")
    for trace in traces:
        trace_errors = audit_trace(trace)
        if trace_errors:
            errors[trace.get("trace_id", "unknown")] = trace_errors
    summary = {
        "trace_count": len(traces),
        "question_count": len(by_question),
        "rollout_counts": dict(sorted(by_question.items())),
        "error_count": sum(len(value) for value in errors.values()),
        "errors": errors,
    }
    output = trace_dir / "trace_audit_report.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
