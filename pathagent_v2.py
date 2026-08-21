from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import torch

from data_processing.utils import (
    build_descriptions_with_meta,
    extract_coords_from_name,
    get_patch_fullpath,
    load_all_vqa_pairs,
    load_image,
    make_unique_id,
    split_patch_for_zoom,
)
from data_processing.wsi_pyramid import (
    FOCUS_AMBIGUOUS_PENALTY_SCALE,
    WSIPyramidReader,
    WSIRegion,
    classify_focus_candidate,
    image_white_fraction,
    load_binary_mask,
    load_patch_manifest,
    load_plip_h5,
    load_wsi_manifest,
    mask_fraction_for_region,
    partition_region,
    rank_focus_candidates_c,
)
from models.inference import (
    build_general_executor_system_prompt,
    evaluate_general_vqa_action,
    evaluate_pancreatic_vqa_action,
    patho_r1_describe,
    sanitize_morphology_evidence_text,
)
from models.retrieval_policy import initial_retrieval_count, replenishment_count
from models.evidence_contract import (
    OUTPUT_SCHEMA_VERSION,
    candidate_record,
    choose_better_candidate,
    description_is_clean,
    evaluate_contract_state,
    load_clean_description_manifest,
    load_contract_policy,
)
from models.llm_backend import (
    ExecutorAPIError,
    ExecutorContextBudgetExceeded,
    ExecutorCostBudgetExceeded,
    detect_checkpoint_family,
    load_llm_backend,
)
from models.trace_recorder import TraceRecorder


GENERIC_MORPHOLOGY_FOCUS = (
    "Record glandular, ductal, papillary, pseudopapillary, solid, nested or trabecular architecture; "
    "cellular atypia; stroma; mucin; necrosis; inflammation; vessels; nerves; and artifacts when visible."
)

FROZEN_PATHO_R1_CANVAS_SIZE = 784

PATHOLOGIST_ASSIST_OUTPUT_FIELDS = (
    "provisional_recommendation",
    "ranked_differential",
    "cited_patches",
    "supporting_evidence",
    "opposing_or_conflicting_evidence",
    "missing_evidence",
    "candidate_evidence_found",
    "ready_for_pathologist_review",
    "strict_evidence_sufficient",
    "review_required",
    "review_reason",
    "contract_version",
    "output_schema_version",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_frozen_patho_canvas(args) -> None:
    actual = int(args.patho_observation_size)
    if actual != FROZEN_PATHO_R1_CANVAS_SIZE:
        raise ValueError(
            "Patho-R1 canvas is frozen at 784x784; "
            f"received {actual}x{actual}. Alternative canvas experiments are disabled."
        )


def _now_run_id() -> str:
    return datetime.now().strftime("pathagent_qwen35_%Y%m%d_%H%M%S")


def _resolve_slide_key(long_id: str, keys: list[str]) -> str:
    matches = [key for key in keys if key.startswith(long_id)]
    if not matches:
        return long_id
    return sorted(matches)[0]


def _insufficient_choice(choices: list[str] | None, question_type: str | None = None):
    for choice in choices or []:
        if "insufficient" in str(choice).lower() or "证据不足" in str(choice):
            return [choice] if question_type == "multiple_choice" else choice
    fallback = "Insufficient evidence"
    return [fallback] if question_type == "multiple_choice" else fallback


def _benchmark_fallback(choices: list[str] | None):
    return str((choices or [""])[0])


def _missing_evidence_text(decision: dict[str, Any]) -> str:
    value = decision.get("executor_missing_evidence")
    if value is None or value == "":
        value = decision.get("missing_evidence")
    if isinstance(value, list):
        return "; ".join(
            str(item.get("text") if isinstance(item, dict) else item)
            for item in value
            if item
        )
    return str(value or "").strip()


def _pathologist_assist_fields(
    verification: dict[str, Any] | None,
    provisional_recommendation: str,
    ranked_differential: list[str] | None = None,
) -> dict[str, Any]:
    """Project deterministic verification into the public assistive schema."""
    verification = verification or {}
    defaults: dict[str, Any] = {
        "provisional_recommendation": provisional_recommendation,
        "ranked_differential": list(ranked_differential or []),
        "cited_patches": [],
        "supporting_evidence": [],
        "opposing_or_conflicting_evidence": [],
        "missing_evidence": ["deterministic_evidence_evaluation_unavailable"],
        "candidate_evidence_found": False,
        "ready_for_pathologist_review": False,
        "strict_evidence_sufficient": False,
        "review_required": True,
        "review_reason": "research_output_requires_pathologist_review",
        "contract_version": verification.get("contract_version"),
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
    }
    for field in PATHOLOGIST_ASSIST_OUTPUT_FIELDS:
        if field in verification:
            defaults[field] = deepcopy(verification[field])
    # Keep the legacy citation list exactly aligned with the deterministic
    # assistive schema.  Action decisions may report more than five valid
    # patches; truncating only ``evidence_refs`` makes an otherwise valid
    # contract output internally inconsistent and unauditable.
    defaults["evidence_refs"] = [
        row["patch_id"]
        for row in defaults["cited_patches"]
        if isinstance(row, dict) and row.get("patch_id")
    ]
    defaults["provisional_recommendation"] = provisional_recommendation
    if ranked_differential is not None and "ranked_differential" not in verification:
        defaults["ranked_differential"] = list(ranked_differential)
    if defaults["strict_evidence_sufficient"]:
        defaults["candidate_evidence_found"] = True
        defaults["ready_for_pathologist_review"] = True
    return defaults


def _contract_final_from_candidate(
    best_candidate: dict[str, Any] | None,
    choices: list[str],
    *,
    explanation: str,
    stop_reason_detail: str,
) -> dict[str, Any]:
    best_candidate = best_candidate or {
        "answer": _benchmark_fallback(choices),
        "attempt": None,
        "evidence_refs": [],
        "evidence_summary": "",
        "unsupported_answer_reason": stop_reason_detail,
        "verification": {},
    }
    verification = best_candidate.get("verification") or {}
    assist_fields = _pathologist_assist_fields(
        verification,
        str(best_candidate["answer"]),
        list(best_candidate.get("ranked_differential") or []),
    )
    return {
        "answer": best_candidate["answer"],
        "explanation": best_candidate.get("evidence_summary") or explanation,
        "evidence_refs": list(best_candidate.get("evidence_refs") or []),
        "confidence": None,
        "confidence_status": "uncalibrated",
        "raw_output": None,
        "parse_status": "contract_v1_best_candidate",
        "evidence_policy": "contract_v1",
        "evidence_sufficient": bool(assist_fields["strict_evidence_sufficient"]),
        "evidence_found": bool(verification.get("evidence_found", False)),
        "citation_valid": bool(verification.get("citation_valid", False)),
        "citation_supports_answer": bool(
            verification.get("citation_supports_answer", False)
        ),
        "abstain_recommended": True,
        "unsupported_answer_reason": (
            best_candidate.get("unsupported_answer_reason") or stop_reason_detail
        ),
        "selected_candidate_attempt": best_candidate.get("attempt"),
        "evidence_contract_verification": verification,
        **assist_fields,
    }


def _sampling(args, rollout_id: int) -> dict[str, Any]:
    if rollout_id == 0:
        return {"temperature": 0.0, "top_p": 1.0, "seed": args.rollout_seed}
    return {
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "seed": args.rollout_seed + rollout_id,
    }


def _plip_text(plip, text: str, recorder: TraceRecorder, operation: str, step_id: int, attempt: int):
    call_id = recorder.before_call(
        "plip", operation, {"text": text, "batch_size": 1}, step_id=step_id, attempt=attempt
    )
    started = time.time()
    try:
        embedding = plip.encode_text([text], batch_size=1)
        recorder.after_call(
            call_id,
            "plip",
            operation,
            {"shape": list(np.asarray(embedding).shape), "latency_ms": round((time.time() - started) * 1000)},
            step_id=step_id,
            attempt=attempt,
        )
        return embedding
    except Exception as exc:
        recorder.after_call(
            call_id,
            "plip",
            operation,
            {"latency_ms": round((time.time() - started) * 1000)},
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            step_id=step_id,
            attempt=attempt,
        )
        raise


def _plip_images(plip, images, image_ids, recorder, operation, step_id, attempt):
    call_id = recorder.before_call(
        "plip",
        operation,
        {"image_ids": image_ids, "batch_size": 4},
        step_id=step_id,
        attempt=attempt,
    )
    started = time.time()
    try:
        embeddings = plip.encode_images(images, batch_size=4)
        recorder.after_call(
            call_id,
            "plip",
            operation,
            {"shape": list(np.asarray(embeddings).shape), "latency_ms": round((time.time() - started) * 1000)},
            step_id=step_id,
            attempt=attempt,
        )
        return embeddings
    except Exception as exc:
        recorder.after_call(
            call_id,
            "plip",
            operation,
            {"latency_ms": round((time.time() - started) * 1000)},
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            step_id=step_id,
            attempt=attempt,
        )
        raise


def _rank(names: list[str], embeddings: np.ndarray, query_embedding: np.ndarray):
    similarities = np.atleast_1d((embeddings @ query_embedding.T).squeeze())
    order = np.argsort(similarities)[::-1]
    return [(names[int(index)], float(similarities[int(index)])) for index in order]


def _resolve_focus_tissue_mask_path(args, slide_id: str) -> Path:
    configured_dirs = getattr(args, "focus_tissue_mask_dirs", None) or []
    if isinstance(configured_dirs, (str, Path)):
        configured_dirs = [configured_dirs]
    search_dirs = [Path(value) for value in configured_dirs]
    if not search_dirs:
        raise ValueError(
            "--focus_tissue_mask_dirs is required with --zoom_backend wsi"
        )
    seen = set()
    for directory in search_dirs:
        resolved = directory.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidate = resolved / f"{slide_id}.grandqc.png"
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in search_dirs)
    raise FileNotFoundError(
        f"GrandQC focus mask for {slide_id} was not found; searched: {searched}"
    )


def _initial_retrieval_query(question: str) -> str:
    return (
        f"{question} "
        "Retrieve lesional H&E regions with direct abnormal morphology: architectural distortion, infiltrative atypical "
        "glands, desmoplastic stroma, papillary or pseudopapillary structures, solid tumor cells, mucin, necrosis, "
        "or a tumor-stroma interface."
    )


def _safe_inspection_focus(text: str | None, choices: list[str]) -> str:
    focus = str(text or "").strip()
    lowered = focus.casefold()
    diagnostic_terms = (
        "tumor category",
        "tumour category",
        "adenocarcinoma",
        "neoplasm",
        "neuroendocrine",
        "pseudopapillary",
        "diagnosis",
    )
    if not focus or any(term in lowered for term in diagnostic_terms):
        return GENERIC_MORPHOLOGY_FOCUS
    for choice in choices:
        focus = focus.replace(str(choice), "visible target morphology")
    return focus


def _filter_visible_evidence_refs(decision: dict[str, Any], visible_patch_ids: list[str]) -> dict[str, Any]:
    refs = decision.get("evidence_refs")
    refs = refs if isinstance(refs, list) else []
    visible = set(visible_patch_ids)
    valid_refs = []
    invalid_refs = []
    for ref in refs:
        if isinstance(ref, str) and ref in visible:
            if ref not in valid_refs:
                valid_refs.append(ref)
        else:
            invalid_refs.append(ref)
    decision["evidence_refs"] = valid_refs[:8]
    if invalid_refs:
        decision["evidence_ref_validation"] = {
            "status": "filtered",
            "dropped_invisible_refs": invalid_refs,
        }
    return decision


def _normalize_repeated_focus_action(
    decision: dict[str, Any], focus_actions_used: int, remaining_patch_count: int
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Enforce one focus action without silently terminating the agent loop."""
    action = decision.get("next_action") or {}
    if action.get("type") != "zoom" or focus_actions_used < 1:
        return decision, None
    original = deepcopy(action)
    replacement = "retrieve" if remaining_patch_count > 0 else "abstain"
    action.update(
        {
            "type": replacement,
            "target_patches": [],
            "magnification": None,
        }
    )
    if replacement == "retrieve" and not action.get("query"):
        action["query"] = _missing_evidence_text(decision) or "additional relevant morphology"
    decision["next_action"] = action
    repair = {
        "reason": "one_focus_action_limit",
        "original_action": original,
        "normalized_action": deepcopy(action),
        "remaining_patch_count": remaining_patch_count,
    }
    decision["action_repair"] = repair
    return decision, repair


def _build_accumulated_executor_evidence(
    descriptions: dict[str, str],
    accumulated_patch_ids: list[str],
    magnification: int,
    char_limit: int,
    evidence_metadata: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, list[str]]:
    """Build one Executor state whose visible IDs exactly match its described evidence."""
    visible_patch_ids = []
    evidence_items = []
    for patch_id in accumulated_patch_ids:
        if patch_id in descriptions and str(descriptions[patch_id]).strip() and patch_id not in visible_patch_ids:
            visible_patch_ids.append(patch_id)
            evidence_items.append((patch_id, descriptions[patch_id]))
    if evidence_metadata:
        parts = ["[Per-patch WSI evidence metadata]"]
        for patch_id, description in evidence_items:
            meta = evidence_metadata.get(patch_id, {})
            x = meta.get("x_level0")
            y = meta.get("y_level0")
            mag = meta.get("magnification", magnification)
            width_um = meta.get("physical_width_um")
            height_um = meta.get("physical_height_um")
            fov = (
                f"{float(width_um):.1f}x{float(height_um):.1f}um"
                if width_um is not None and height_um is not None
                else "unknown"
            )
            parts.append(
                f"[{patch_id} | Level0=({x},{y}) | Magnification={mag}x | FOV={fov}] {description}"
            )
        evaluation_text = "\n\n".join(parts)
    else:
        evaluation_text = build_descriptions_with_meta(
            evidence_items,
            mag_level=magnification,
            include_header=True,
            include_coords=True,
        )
    if len(evaluation_text) > char_limit:
        raise ExecutorContextBudgetExceeded(
            f"Accumulated Executor evidence has {len(evaluation_text)} characters; hard limit is {char_limit}."
        )
    return evaluation_text, visible_patch_ids


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _describe_patch(
    patch_name,
    descriptions,
    args,
    slide_id,
    question,
    choices,
    patho_model,
    patho_processor,
    recorder,
    step_id,
    attempt,
    magnification,
    operation="question_focused_morphology",
    inspection_focus=None,
    description_cache=None,
    cache_path=None,
    image_provider=None,
    evidence_metadata=None,
):
    patch_geometry = (evidence_metadata or {}).get(patch_name, {})
    cache_payload = {
        "cache_schema": "patho_morphology_cache_v2",
        "prompt_version": args.patho_prompt_version,
        "model": args.patho_r1_ckpt,
        "slide_id": slide_id,
        "patch_id": patch_name,
        "inspection_focus": inspection_focus or GENERIC_MORPHOLOGY_FOCUS,
        "magnification": magnification,
        "operation": operation,
        "evidence_geometry": {
            key: patch_geometry.get(key)
            for key in (
                "x_level0",
                "y_level0",
                "width_level0",
                "height_level0",
                "mpp_x",
                "mpp_y",
                "magnification",
                "output_size",
            )
            if patch_geometry.get(key) is not None
        },
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if description_cache is not None and cache_key in description_cache:
        if image_provider is not None:
            _, registration = image_provider(patch_name, magnification)
            recorder.record_state(
                "evidence_registered",
                {"patch": {**registration, "description_source": "patho_r1_cache"}},
                step_id=step_id,
                attempt=attempt,
            )
        call_id = recorder.before_call(
            "patho_r1_cache",
            operation,
            {**cache_payload, "cache_key": cache_key},
            step_id=step_id,
            attempt=attempt,
        )
        description = description_cache[cache_key]["returned_output"]
        recorder.after_call(
            call_id,
            "patho_r1_cache",
            operation,
            {"cache_hit": True, "cache_key": cache_key, "returned_output": description},
            step_id=step_id,
            attempt=attempt,
        )
        original = descriptions.get(patch_name, "")
        descriptions[patch_name] = (original + " " + description).strip()
        return description
    if image_provider is not None:
        patch_img, registration = image_provider(patch_name, magnification)
        recorder.record_state(
            "evidence_registered",
            {"patch": registration},
            step_id=step_id,
            attempt=attempt,
        )
        x = registration.get("x_level0")
        y = registration.get("y_level0")
    else:
        patch_path = get_patch_fullpath(args.patch_root, slide_id, patch_name)
        patch_img = load_image(patch_path)
        x, y = extract_coords_from_name(patch_name)
    description = patho_r1_describe(
        patch_img,
        question=None if args.executor_protocol == "general_v2" else question,
        patho_r1_processor=patho_processor,
        patho_r1_model=patho_model,
        coords=(x, y),
        magnification=magnification,
        choices=choices,
        morphology_only=True,
        inspection_focus=inspection_focus or GENERIC_MORPHOLOGY_FOCUS,
        prompt_version=args.patho_prompt_version,
        trace_recorder=recorder,
        trace_context={"step_id": step_id, "attempt": attempt},
        patch_id=patch_name,
        operation=operation,
        max_new_tokens=args.patho_max_new_tokens,
    )
    original = descriptions.get(patch_name, "")
    descriptions[patch_name] = (original + " " + description).strip()
    if description_cache is not None:
        description_cache[cache_key] = {**cache_payload, "returned_output": description}
        if cache_path is not None:
            _write_json_atomic(Path(cache_path), description_cache)
    return description


def run_pancreatic_v2(args, plip_class, patho_model_class, patho_processor_class):
    _validate_frozen_patho_canvas(args)
    general_v2 = args.executor_protocol == "general_v2"
    protocol_name = "general_v2" if general_v2 else "pancreatic_v2"
    evidence_policy = getattr(args, "evidence_policy", "model_v1")
    if evidence_policy == "contract_v1" and not general_v2:
        raise ValueError("--evidence_policy contract_v1 requires --executor_protocol general_v2")
    if getattr(args, "initial_top_k", None) is not None and args.initial_top_k < 1:
        raise ValueError("--initial_top_k must be at least 1 when provided")
    if getattr(args, "replenish_top_k", None) is not None and args.replenish_top_k < 1:
        raise ValueError("--replenish_top_k must be at least 1 when provided")
    zoom_backend = args.zoom_backend
    if evidence_policy == "contract_v1" and zoom_backend != "wsi":
        raise ValueError("--evidence_policy contract_v1 requires --zoom_backend wsi")
    focus_tissue_mask_dirs = getattr(args, "focus_tissue_mask_dirs", None) or []
    if isinstance(focus_tissue_mask_dirs, (str, Path)):
        focus_tissue_mask_dirs = [focus_tissue_mask_dirs]
    if args.rollouts_per_question < 1:
        raise ValueError("--rollouts_per_question must be at least 1")
    if args.focus_parent_top_k < 1:
        raise ValueError("--focus_parent_top_k must be at least 1")
    if zoom_backend == "wsi":
        required_wsi_args = {
            "--wsi_manifest": args.wsi_manifest,
            "--patch_manifest_dir": args.patch_manifest_dir,
            "--feature_h5_dir": args.feature_h5_dir,
        }
        missing = [name for name, value in required_wsi_args.items() if not value]
        if missing:
            raise ValueError(f"WSI backend requires: {', '.join(missing)}")
        wsi_rows = load_wsi_manifest(args.wsi_manifest)
    else:
        missing = [
            name
            for name, value in {
                "--descriptions_file": args.descriptions_file,
                "--feature_dir": args.feature_dir,
                "--patch_root": args.patch_root,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"legacy_jpeg backend requires: {', '.join(missing)}")
        wsi_rows = {}
    if args.disable_patho_r1_runtime:
        print(f"Warning: {protocol_name} is running without online Patho-R1 by explicit request.")

    evidence_contracts = None
    option_ontology = None
    clean_description_rows: dict[tuple[str, str], dict[str, Any]] = {}
    evidence_contracts_path = None
    option_ontology_path = None
    description_manifest_path = None
    if evidence_policy == "contract_v1":
        required_contract_args = {
            "--evidence_contracts_path": getattr(args, "evidence_contracts_path", None),
            "--option_ontology_path": getattr(args, "option_ontology_path", None),
            "--descriptions_file": args.descriptions_file,
        }
        missing = [name for name, value in required_contract_args.items() if not value]
        if missing:
            raise ValueError(f"contract_v1 requires: {', '.join(missing)}")
        evidence_contracts_path = Path(args.evidence_contracts_path).resolve()
        option_ontology_path = Path(args.option_ontology_path).resolve()
        evidence_contracts, option_ontology = load_contract_policy(
            evidence_contracts_path, option_ontology_path
        )
        description_manifest_path = (
            Path(args.description_manifest).resolve()
            if getattr(args, "description_manifest", None)
            else Path(args.descriptions_file).resolve().with_name("description_manifest.jsonl")
        )
        if not description_manifest_path.is_file():
            raise FileNotFoundError(
                "contract_v1 requires an audited description manifest: "
                f"{description_manifest_path}"
            )
        clean_description_rows = load_clean_description_manifest(
            description_manifest_path
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trace_root = Path(args.trace_dir or save_dir / "traces")
    run_id = args.run_id or _now_run_id()

    pairs = load_all_vqa_pairs(
        args.questions_file,
        dataset_name=args.dataset_name,
        image_dir=args.patch_root if zoom_backend == "legacy_jpeg" else None,
    )
    print(f"Loaded {len(pairs)} {protocol_name} VQA pairs; rollouts per question={args.rollouts_per_question}.")
    model, tokenizer = load_llm_backend(
        args.qwen_ckpt,
        backend=args.qwen_backend,
        api_base_url=args.qwen_api_base_url if args.executor_provider == "qwen" else None,
        api_model=args.qwen_api_model if args.executor_provider == "qwen" else None,
        api_key=args.qwen_api_key if args.executor_provider == "qwen" else None,
        provider=args.executor_provider,
        env_file=args.executor_env_file,
        timeout=args.executor_api_timeout,
        api_max_attempts=args.executor_api_max_attempts,
        request_char_limit=args.executor_request_char_limit,
        budget_rmb=args.executor_budget_rmb,
    )
    plip = plip_class(args.plip_ckpt)
    patho_model = None
    patho_processor = None
    if not args.disable_patho_r1_runtime:
        patho_model = patho_model_class.from_pretrained(
            args.patho_r1_ckpt, torch_dtype="auto", device_map="auto"
        )
        patho_processor = patho_processor_class.from_pretrained(args.patho_r1_ckpt)

    all_descriptions = (
        json.loads(Path(args.descriptions_file).read_text(encoding="utf-8"))
        if args.descriptions_file
        else {}
    )
    all_keys = sorted(set(all_descriptions) | set(wsi_rows))
    resolved_executor_model = getattr(model, "model_name", None) or args.qwen_ckpt
    executor_base_url = str(getattr(model, "base_url", "") or "")
    executor_endpoint = urlsplit(executor_base_url)
    executor_prompt_sha256 = (
        hashlib.sha256(
            build_general_executor_system_prompt(evidence_policy).encode("utf-8")
        ).hexdigest()
        if general_v2
        else None
    )
    runtime = {
        "executor_provider": args.executor_provider,
        "executor_model": resolved_executor_model,
        "executor_family": (
            "deepseek_v4_flash" if args.executor_provider == "deepseek" else detect_checkpoint_family(args.qwen_ckpt)
        ),
        "executor_backend": args.qwen_backend,
        "executor_endpoint_category": (
            "external_https_text_api"
            if args.executor_provider == "deepseek"
            and executor_endpoint.scheme == "https"
            and executor_endpoint.hostname
            else "local_or_other"
        ),
        "executor_endpoint_hostname": executor_endpoint.hostname,
        "executor_prompt_sha256": executor_prompt_sha256,
        "executor_api_max_attempts": args.executor_api_max_attempts,
        "executor_request_char_limit": args.executor_request_char_limit,
        "executor_budget_rmb": args.executor_budget_rmb,
        "thinking_enabled": False,
        "patho_r1_model": args.patho_r1_ckpt,
        "patho_r1_runtime": not args.disable_patho_r1_runtime,
        "plip_model": args.plip_ckpt,
        "executor_protocol": protocol_name,
        "agent_policy_version": (
            "general_v2_wsi_focus_c_v2" if general_v2 and zoom_backend == "wsi"
            else "pancreatic_v4_wsi_focus_c_v2" if zoom_backend == "wsi"
            else "general_v2_dual_output_v1" if general_v2
            else "pancreatic_v3_single_answer_source"
        ),
        "patho_prompt_version": args.patho_prompt_version,
        "patho_cache_schema": "patho_morphology_cache_v2",
        "zoom_backend": zoom_backend,
        "patho_observation_size": args.patho_observation_size,
        "patho_observation_canvas": [
            FROZEN_PATHO_R1_CANVAS_SIZE,
            FROZEN_PATHO_R1_CANVAS_SIZE,
        ],
        "patho_observation_config_status": "frozen",
        "focus_parent_top_k": args.focus_parent_top_k,
        "focus_child_strategy": "C_hard_blank_gate_ambiguous_soft_penalty_top2" if zoom_backend == "wsi" else None,
        "focus_child_top_k": 2 if zoom_backend == "wsi" else 1,
        "focus_ambiguous_penalty_scale": FOCUS_AMBIGUOUS_PENALTY_SCALE if zoom_backend == "wsi" else None,
        "focus_target_morphology_recall_status": "not_yet_validated" if zoom_backend == "wsi" else None,
        "max_focus_actions": 1,
        "confidence_status": "uncalibrated",
        "evidence_policy": evidence_policy,
        "output_schema_version": (
            OUTPUT_SCHEMA_VERSION if evidence_policy == "contract_v1" else None
        ),
        "evidence_sufficient_compatibility_mapping": (
            "strict_evidence_sufficient"
            if evidence_policy == "contract_v1"
            else None
        ),
        "review_required_default": (
            True if evidence_policy == "contract_v1" else None
        ),
        "evidence_contract_version": (
            evidence_contracts.get("contract_version") if evidence_contracts else None
        ),
        "evidence_term_matcher_version": (
            evidence_contracts.get("term_matcher_version") if evidence_contracts else None
        ),
        "initial_top_k": getattr(args, "initial_top_k", None),
        "replenish_top_k": getattr(args, "replenish_top_k", None),
        "prefer_precomputed_descriptions": bool(
            getattr(args, "prefer_precomputed_descriptions", False)
        ),
    }
    trace_root.mkdir(parents=True, exist_ok=True)
    cache_path = trace_root / "patho_morphology_cache_v2.json"
    if cache_path.exists():
        patho_description_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        patho_description_cache = {}
    run_manifest = {
        "run_id": run_id,
        "schema_version": "pathagent_trace_v2",
        "question_count": len(pairs),
        "rollouts_per_question": args.rollouts_per_question,
        "runtime": runtime,
        "questions_file": str(Path(args.questions_file).resolve()),
        "descriptions_file": str(Path(args.descriptions_file).resolve()) if args.descriptions_file else None,
        "wsi_manifest": str(Path(args.wsi_manifest).resolve()) if args.wsi_manifest else None,
        "patch_manifest_dir": str(Path(args.patch_manifest_dir).resolve()) if args.patch_manifest_dir else None,
        "feature_h5_dir": str(Path(args.feature_h5_dir).resolve()) if args.feature_h5_dir else None,
        "focus_tissue_mask_dirs": [
            str(Path(value).resolve())
            for value in focus_tissue_mask_dirs
        ],
        "focus_tissue_mask_resolution": "explicit_dirs_only" if zoom_backend == "wsi" else None,
        "patho_prompt_version": args.patho_prompt_version,
        "patho_cache_file": str(cache_path.resolve()),
        "evidence_contracts_path": (
            str(evidence_contracts_path) if evidence_contracts_path else None
        ),
        "evidence_contracts_sha256": (
            _sha256_file(evidence_contracts_path) if evidence_contracts_path else None
        ),
        "option_ontology_path": (
            str(option_ontology_path) if option_ontology_path else None
        ),
        "option_ontology_sha256": (
            _sha256_file(option_ontology_path) if option_ontology_path else None
        ),
        "description_manifest": (
            str(description_manifest_path) if description_manifest_path else None
        ),
        "description_manifest_sha256": (
            _sha256_file(description_manifest_path) if description_manifest_path else None
        ),
    }
    run_manifest_path = trace_root / "run_manifest.json"
    if run_manifest_path.exists():
        existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != run_manifest:
            raise RuntimeError("Refusing unsafe resume: run_manifest.json does not match the requested run.")
    else:
        _write_json_atomic(run_manifest_path, run_manifest)

    all_results = []
    focus_mask_cache: dict[str, tuple[Path, np.ndarray]] = {}
    expected_rollouts = len(pairs) * args.rollouts_per_question
    executor_api_failures = 0
    consecutive_executor_api_failures = 0

    existing_raw_traces = {}
    for raw_path in sorted((trace_root / "raw").glob("*.json")):
        raw_trace = json.loads(raw_path.read_text(encoding="utf-8"))
        if raw_trace.get("execution", {}).get("status") != "completed":
            raise RuntimeError(f"Refusing unsafe resume: failed trace exists at {raw_path}.")
        existing_raw_traces[str(raw_trace.get("trace_id"))] = raw_trace
    existing_cost_rmb = sum(
        float(event.get("response", {}).get("estimated_cost_rmb") or 0.0)
        for trace in existing_raw_traces.values()
        for event in trace.get("events", [])
        if event.get("phase") == "after" and event.get("component") in {"qwen_executor", "deepseek_executor"}
    )
    if hasattr(model, "cumulative_cost_rmb"):
        model.cumulative_cost_rmb = existing_cost_rmb

    def write_executor_summary(status: str) -> None:
        summary = {
            "run_id": run_id,
            "status": status,
            "provider": args.executor_provider,
            "requested_model": resolved_executor_model,
            "returned_model": getattr(model, "returned_model_name", None),
            "endpoint_category": runtime["executor_endpoint_category"],
            "endpoint_hostname": runtime["executor_endpoint_hostname"],
            "prompt_sha256": runtime["executor_prompt_sha256"],
            "output_schema_version": runtime["output_schema_version"],
            "expected_rollouts": expected_rollouts,
            "completed_or_recorded_rollouts": len(all_results),
            "api_failures": executor_api_failures,
            "estimated_cost_rmb": round(float(getattr(model, "cumulative_cost_rmb", 0.0)), 8),
            "budget_rmb": args.executor_budget_rmb,
        }
        _write_json_atomic(trace_root / "executor_usage_summary.json", summary)

    for pair_index, pair in enumerate(pairs):
        requested_slide_id = str(pair["long_id"])
        slide_id = _resolve_slide_key(requested_slide_id, all_keys)
        question = pair["question"]
        choices = pair.get("choices") or []
        supplied_descriptions = deepcopy(all_descriptions.get(slide_id, {}))
        supplied_descriptions = {
            patch_name: sanitize_morphology_evidence_text(description, forbidden_labels=choices)
            for patch_name, description in supplied_descriptions.items()
        }
        question_id = pair.get("question_id") or make_unique_id(slide_id, question)
        unique_id = make_unique_id(slide_id, question)

        if zoom_backend == "wsi":
            if slide_id not in wsi_rows:
                print(f"Skipping {question_id}: slide is absent from WSI manifest.")
                continue
            region_path = Path(args.patch_manifest_dir) / f"{slide_id}.jsonl"
            feature_path = Path(args.feature_h5_dir) / f"{slide_id}.plip.v1.h5"
            base_regions = load_patch_manifest(region_path, selected_only=True)
            available, feature_matrix = load_plip_h5(feature_path)
            if set(available) != set(base_regions):
                raise RuntimeError(
                    f"Patch manifest/PLIP HDF5 mismatch for {slide_id}: "
                    f"manifest={len(base_regions)}, features={len(available)}"
                )
            base_feature_cache = {
                patch_id: feature_matrix[index]
                for index, patch_id in enumerate(available)
            }
            base_descriptions = {
                patch_id: supplied_descriptions.get(patch_id, "")
                for patch_id in available
            }
            base_evidence_paths = {}
            base_evidence_metadata = {
                patch_id: base_regions[patch_id].to_dict() for patch_id in available
            }
            if slide_id not in focus_mask_cache:
                focus_mask_path = _resolve_focus_tissue_mask_path(args, slide_id)
                focus_mask_cache[slide_id] = (
                    focus_mask_path,
                    load_binary_mask(focus_mask_path),
                )
            focus_mask_path, focus_tissue_mask = focus_mask_cache[slide_id]
        else:
            if not supplied_descriptions:
                print(f"Skipping {requested_slide_id}: no descriptions for resolved slide {slide_id}.")
                continue
            base_descriptions = supplied_descriptions
            base_feature_cache = {}
            for patch_name in base_descriptions:
                feature_path = Path(args.feature_dir) / slide_id / f"{Path(patch_name).stem}.npy"
                if feature_path.exists():
                    base_feature_cache[patch_name] = np.load(feature_path)
            available = [name for name in base_descriptions if name in base_feature_cache]
            feature_matrix = np.stack([base_feature_cache[name] for name in available]) if available else np.empty((0, 0))
            base_regions = {}
            base_evidence_paths = {
                name: Path(get_patch_fullpath(args.patch_root, slide_id, name)) for name in available
            }
            base_evidence_metadata = {}
        base_clean_patch_ids = {
            patch_id
            for patch_id in available
            if description_is_clean(clean_description_rows.get((slide_id, patch_id)))
        }
        if evidence_policy == "contract_v1" and not base_clean_patch_ids:
            raise RuntimeError(
                f"contract_v1 found no audited clean descriptions for slide {slide_id}"
            )
        if not available:
            print(f"Skipping {question_id}: no PLIP features.")
            continue
        if args.disable_patho_r1_runtime and not any(base_descriptions.values()):
            raise RuntimeError(
                f"{slide_id} has no descriptions and online Patho-R1 is disabled"
            )

        for rollout_id in range(args.rollouts_per_question):
            suffix = "" if args.rollouts_per_question == 1 else f"_rollout_{rollout_id:02d}"
            result_path = save_dir / f"{unique_id}{suffix}.json"
            if result_path.exists():
                existing_result = json.loads(result_path.read_text(encoding="utf-8"))
                expected_trace_id = f"{run_id}_{question_id}_r{rollout_id:02d}"
                if (
                    existing_result.get("question_id") != question_id
                    or existing_result.get("rollout_id") != rollout_id
                    or existing_result.get("trace_id") != expected_trace_id
                    or expected_trace_id not in existing_raw_traces
                ):
                    raise RuntimeError(f"Refusing unsafe resume: result/trace mismatch for {result_path}.")
                all_results.append(existing_result)
                print(f"Validated completed rollout for resume: {result_path.name}")
                continue
            sampling = _sampling(args, rollout_id)
            trace_id = f"{run_id}_{question_id}_r{rollout_id:02d}"
            feature_cache = dict(base_feature_cache)
            evidence_paths = dict(base_evidence_paths)
            evidence_magnifications = {name: 5 for name in available}
            evidence_regions = dict(base_regions)
            evidence_metadata = deepcopy(base_evidence_metadata)
            recorder = TraceRecorder(
                trace_root,
                run_id,
                trace_id,
                rollout_id,
                task_input={
                    "case_id": pair.get("case_id") or requested_slide_id.split("_")[0],
                    "slide_id": slide_id,
                    "question_id": question_id,
                    "question": question,
                    "question_type": pair.get("question_type"),
                    "difficulty": pair.get("difficulty"),
                    "evidence_tier": pair.get("evidence_tier", "strict"),
                    "choices": choices,
                    "patch_count": len(base_descriptions),
                },
                runtime={**runtime, "sampling": sampling},
            )
            descriptions = deepcopy(base_descriptions)
            evidence_index = []
            for patch_name in descriptions:
                item = {
                    "patch_id": patch_name,
                    "base_description": descriptions[patch_name],
                    "description_status": "available" if descriptions[patch_name] else "empty",
                    "contract_description_clean": patch_name in base_clean_patch_ids,
                    "feature_available": patch_name in feature_cache,
                }
                if zoom_backend == "wsi":
                    item.update(evidence_metadata[patch_name])
                    item["evidence_source"] = "original_wsi_pyramid"
                else:
                    x, y = extract_coords_from_name(patch_name)
                    item.update({"x": x, "y": y, "evidence_source": "legacy_jpeg"})
                evidence_index.append(item)
            recorder.record_state("evidence_index_loaded", {"patches": evidence_index})

            def image_provider(patch_name: str, _magnification: int):
                region = evidence_regions[patch_name]
                with WSIPyramidReader(wsi_rows[slide_id]) as reader:
                    observation = reader.read(region, args.patho_observation_size)
                observation_dir = trace_root / "observations" / trace_id
                observation_dir.mkdir(parents=True, exist_ok=True)
                observation_path = observation_dir / f"{patch_name}.png"
                observation.image.save(observation_path, format="PNG")
                registration = {
                    **observation.metadata,
                    "patch_id": patch_name,
                    "path": str(observation_path.resolve()),
                    "evidence_type": "wsi_region",
                    "evidence_source": "original_wsi_pyramid",
                }
                evidence_paths[patch_name] = observation_path
                evidence_metadata[patch_name] = registration
                return observation.image, registration

            rollout_image_provider = image_provider if zoom_backend == "wsi" else None
            process = []
            final_answer = None
            accumulated = []
            remaining = list(available)
            patches_this_round = []
            zoom_level = 5
            focus_actions_used = 0
            described_online = set()
            termination_reason = None
            best_candidate = None
            previous_candidate_answer = None
            candidate_stability_count = 0
            try:
                initial_query = _initial_retrieval_query(question)
                query_embedding = _plip_text(plip, initial_query, recorder, "initial_morphology_embedding", 1, 0)
                query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=-1, keepdims=True)
                ranked = _rank(available, feature_matrix, query_embedding)
                initial_count = initial_retrieval_count(
                    len(available), args.initial_sample_ratio, getattr(args, "initial_top_k", None)
                )
                initial_ranked = ranked[:initial_count]
                patches_this_round = [name for name, _ in initial_ranked]
                accumulated.extend(patches_this_round)
                remaining = [name for name in remaining if name not in accumulated]
                recorder.record_state(
                    "retrieve",
                    {
                        "query": initial_query,
                        "ranking_method": "plip_cosine_similarity",
                        "selected": [{"patch_id": name, "score": score} for name, score in initial_ranked],
                        "candidate_count": len(available),
                    },
                    step_id=1,
                    attempt=0,
                )

                for attempt in range(1, args.max_attempts + 1):
                    step_id = len(process) + 1
                    if not patches_this_round:
                        if evidence_policy == "contract_v1":
                            final_answer = _contract_final_from_candidate(
                                best_candidate,
                                choices,
                                explanation="No additional patch evidence is available.",
                                stop_reason_detail="No additional evidence was available for contract_v1.",
                            )
                        else:
                            final_answer = {
                                "answer": _benchmark_fallback(choices) if general_v2 else _insufficient_choice(choices, pair.get("question_type")),
                                "explanation": "No additional patch evidence is available.",
                                "evidence_refs": [],
                                "confidence": None,
                                "confidence_status": "uncalibrated",
                                "raw_output": None,
                                "evidence_sufficient": False,
                                "abstain_recommended": True,
                                "unsupported_answer_reason": "No additional patch evidence is available.",
                            }
                        termination_reason = "no_additional_evidence"
                        break
                    if patho_model is not None:
                        for patch_name in patches_this_round[: args.question_specific_desc_top_k]:
                            if patch_name in described_online:
                                continue
                            if (
                                getattr(args, "prefer_precomputed_descriptions", False)
                                and str(descriptions.get(patch_name) or "").strip()
                            ):
                                described_online.add(patch_name)
                                continue
                            _describe_patch(
                                patch_name,
                                descriptions,
                                args,
                                slide_id,
                                question,
                                choices,
                                patho_model,
                                patho_processor,
                                recorder,
                                step_id,
                                attempt,
                                evidence_magnifications.get(patch_name, 5),
                                description_cache=patho_description_cache,
                                cache_path=cache_path,
                                image_provider=rollout_image_provider,
                                evidence_metadata=evidence_metadata,
                            )
                            described_online.add(patch_name)

                    try:
                        evaluation_text, executor_visible_patch_ids = _build_accumulated_executor_evidence(
                            descriptions,
                            accumulated,
                            zoom_level,
                            args.executor_request_char_limit,
                            evidence_metadata=evidence_metadata if zoom_backend == "wsi" else None,
                        )
                        action_evaluator = evaluate_general_vqa_action if general_v2 else evaluate_pancreatic_vqa_action
                        evaluator_kwargs = {
                            "visible_patch_ids": executor_visible_patch_ids,
                            "remaining_patch_count": len(remaining),
                            "can_zoom": patho_model is not None and focus_actions_used < 1,
                            "question_type": pair.get("question_type"),
                            "temperature": sampling["temperature"],
                            "top_p": sampling["top_p"],
                            "seed": sampling["seed"] + attempt,
                            "trace_recorder": recorder,
                            "trace_context": {"step_id": step_id, "attempt": attempt},
                        }
                        if general_v2:
                            evaluator_kwargs["evidence_policy"] = evidence_policy
                        decision = action_evaluator(
                            model,
                            tokenizer,
                            evaluation_text,
                            question,
                            choices,
                            **evaluator_kwargs,
                        )
                    except ExecutorContextBudgetExceeded as exc:
                        recorder.record_state(
                            "context_budget_exceeded",
                            {
                                "hard_limit_chars": args.executor_request_char_limit,
                                "visible_patch_count": len(accumulated),
                                "error": str(exc),
                            },
                            step_id=step_id,
                            attempt=attempt,
                        )
                        if evidence_policy == "contract_v1":
                            final_answer = _contract_final_from_candidate(
                                best_candidate,
                                choices,
                                explanation="The accumulated evidence exceeded the fixed Executor context budget.",
                                stop_reason_detail="Executor context budget exceeded before contract_v1 sufficiency.",
                            )
                            final_answer["parse_status"] = "context_budget_exceeded"
                        else:
                            final_answer = {
                                "answer": _benchmark_fallback(choices) if general_v2 else _insufficient_choice(choices, pair.get("question_type")),
                                "explanation": "The accumulated evidence exceeded the fixed Executor context budget.",
                                "evidence_refs": [],
                                "confidence": None,
                                "confidence_status": "uncalibrated",
                                "raw_output": None,
                                "parse_status": "context_budget_exceeded",
                                "evidence_sufficient": False,
                                "abstain_recommended": True,
                                "unsupported_answer_reason": "Executor context budget exceeded.",
                            }
                        termination_reason = "context_budget_exceeded"
                        break
                    reported_evidence_refs = deepcopy(decision.get("evidence_refs"))
                    decision = _filter_visible_evidence_refs(decision, executor_visible_patch_ids)
                    if evidence_policy == "contract_v1":
                        executor_missing_evidence = deepcopy(
                            decision.get("missing_evidence")
                        )
                        model_evidence_sufficient = bool(
                            decision.get("evidence_sufficient", False)
                        )
                        verification = evaluate_contract_state(
                            contracts=evidence_contracts,
                            ontology=option_ontology,
                            question_type=str(pair.get("question_type") or ""),
                            choices=[str(choice) for choice in choices],
                            candidate_answer=str(decision.get("benchmark_answer") or ""),
                            evidence_tier=str(pair.get("evidence_tier") or "strict"),
                            visible_patch_ids=executor_visible_patch_ids,
                            reported_evidence_refs=reported_evidence_refs,
                            descriptions=descriptions,
                            metadata=evidence_metadata,
                            clean_patch_ids=base_clean_patch_ids,
                            ranked_differential=decision.get(
                                "ranked_differential", []
                            ),
                        )
                        candidate_answer = str(decision.get("benchmark_answer") or "")
                        if candidate_answer == previous_candidate_answer:
                            candidate_stability_count += 1
                        else:
                            previous_candidate_answer = candidate_answer
                            candidate_stability_count = 1
                        challenger = candidate_record(
                            decision,
                            verification,
                            attempt,
                            candidate_stability_count,
                        )
                        best_candidate = choose_better_candidate(
                            best_candidate, challenger
                        )
                        decision.update(
                            {
                                "model_evidence_sufficient": model_evidence_sufficient,
                                "model_evidence_assessment": deepcopy(
                                    decision.get("advisory_evidence_state") or {}
                                ),
                                "executor_missing_evidence": executor_missing_evidence,
                                "evidence_policy": "contract_v1",
                                "evidence_contract_verification": verification,
                                "evidence_found": verification["evidence_found"],
                                "citation_valid": verification["citation_valid"],
                                "citation_supports_answer": verification[
                                    "citation_supports_answer"
                                ],
                                "evidence_sufficient": verification[
                                    "evidence_sufficient"
                                ],
                                "sufficient": verification["evidence_sufficient"],
                                "candidate_stability_count": candidate_stability_count,
                                "best_candidate_so_far": {
                                    key: value
                                    for key, value in best_candidate.items()
                                    if key != "verification"
                                },
                                **_pathologist_assist_fields(
                                    verification,
                                    candidate_answer,
                                    decision.get("ranked_differential", []),
                                ),
                            }
                        )
                        recorder.record_state(
                            "evidence_contract_evaluated",
                            verification,
                            step_id=step_id,
                            attempt=attempt,
                        )
                        action_before_contract = deepcopy(decision.get("next_action") or {})
                        if verification["evidence_sufficient"]:
                            decision["benchmark_answer"] = verification[
                                "unique_visible_answer"
                            ]
                            decision["candidate_answer"] = decision["benchmark_answer"]
                            decision["abstain_recommended"] = False
                            decision["next_action"] = {
                                "type": "answer",
                                "query": "",
                                "target_patches": [],
                                "magnification": None,
                            }
                        elif action_before_contract.get("type") == "answer":
                            replacement = "retrieve" if remaining else "abstain"
                            decision["next_action"] = {
                                "type": replacement,
                                "query": _missing_evidence_text(decision)
                                or "additional contract-matching morphology",
                                "target_patches": [],
                                "magnification": None,
                            }
                            decision["abstain_recommended"] = replacement == "abstain"
                        if decision.get("next_action") != action_before_contract:
                            decision["contract_policy_repair"] = {
                                "original_action": action_before_contract,
                                "normalized_action": deepcopy(
                                    decision.get("next_action")
                                ),
                                "reason": (
                                    "deterministic_contract_passed"
                                    if verification["evidence_sufficient"]
                                    else "model_answer_lacked_contract_support"
                                ),
                            }
                    decision, focus_repair = _normalize_repeated_focus_action(
                        decision, focus_actions_used, len(remaining)
                    )
                    if focus_repair is not None:
                        recorder.record_state(
                            "repeated_focus_normalized",
                            focus_repair,
                            step_id=step_id,
                            attempt=attempt,
                        )
                    action = decision["next_action"]
                    process_item = {
                        "attempt": attempt,
                        "mode": action["type"],
                        "evaluated_patches_this_round": list(patches_this_round),
                        "total_accumulated_patches": len(accumulated),
                        "evaluation_result": decision,
                    }
                    process.append(process_item)
                    recorder.record_state(
                        "agent_action",
                        {"state_visible_patches": list(executor_visible_patch_ids), "decision": decision},
                        step_id=step_id,
                        attempt=attempt,
                    )

                    if attempt == args.max_attempts and action["type"] not in {"answer", "abstain"}:
                        recorder.record_state(
                            "step_budget_exhausted",
                            {"requested_action": action, "max_attempts": args.max_attempts},
                            step_id=step_id,
                            attempt=attempt,
                        )
                        process_item["mode"] = "max_steps_abstain"
                        if evidence_policy == "contract_v1":
                            final_answer = _contract_final_from_candidate(
                                best_candidate,
                                choices,
                                explanation="The action budget was exhausted before unique cited contract support was obtained.",
                                stop_reason_detail="Action budget exhausted without contract_v1 sufficiency.",
                            )
                        else:
                            final_answer = {
                                "answer": decision.get("benchmark_answer", _benchmark_fallback(choices)) if general_v2 else _insufficient_choice(choices, pair.get("question_type")),
                                "explanation": "The action budget was exhausted before sufficient visible evidence was obtained.",
                                "evidence_refs": decision.get("evidence_refs", []),
                                "confidence": None,
                                "confidence_status": "uncalibrated",
                                "raw_output": decision.get("raw_texts", {}).get("action_raw"),
                                "parse_status": decision.get("parse_status", "invalid_action_json"),
                                "evidence_sufficient": False,
                                "abstain_recommended": True,
                                "unsupported_answer_reason": decision.get("unsupported_answer_reason") or "Action budget exhausted.",
                            }
                        process_item.update(
                            {"answer": final_answer["answer"], "explanation": final_answer["explanation"]}
                        )
                        termination_reason = "budget_exhausted"
                        break

                    if action["type"] == "answer":
                        candidate = decision.get("candidate_answer")
                        if candidate not in choices:
                            candidate = _insufficient_choice(choices, pair.get("question_type"))
                        visible_refs = set(accumulated)
                        evidence_refs = [
                            ref for ref in decision.get("evidence_refs", []) if ref in visible_refs
                        ][:5]
                        final_answer = {
                            "answer": candidate,
                            "evidence_refs": evidence_refs,
                            "explanation": decision.get("evidence_summary") or decision.get("action_reason"),
                            "confidence": None,
                            "confidence_status": "uncalibrated",
                            "raw_output": decision.get("raw_texts", {}).get("action_raw"),
                            "parse_status": decision.get("parse_status", "invalid_action_json"),
                            "evidence_policy": evidence_policy,
                            "evidence_sufficient": bool(decision.get("evidence_sufficient", decision.get("sufficient"))),
                            "evidence_found": bool(decision.get("evidence_found", False)),
                            "citation_valid": bool(decision.get("citation_valid", False)),
                            "citation_supports_answer": bool(decision.get("citation_supports_answer", False)),
                            "abstain_recommended": bool(decision.get("abstain_recommended", False)),
                            "unsupported_answer_reason": decision.get("unsupported_answer_reason", ""),
                            "selected_candidate_attempt": attempt,
                            "ranked_differential": decision.get(
                                "ranked_differential", []
                            ),
                            "evidence_contract_verification": decision.get("evidence_contract_verification", {}),
                        }
                        process_item.update({"answer": final_answer["answer"], "explanation": final_answer["explanation"]})
                        termination_reason = "answered"
                        break

                    if action["type"] == "abstain":
                        if evidence_policy == "contract_v1":
                            final_answer = _contract_final_from_candidate(
                                best_candidate,
                                choices,
                                explanation=decision.get("action_reason")
                                or _missing_evidence_text(decision)
                                or "The Executor abstained before a unique contract pass.",
                                stop_reason_detail=decision.get("unsupported_answer_reason")
                                or _missing_evidence_text(decision)
                                or "Executor abstained without contract_v1 sufficiency.",
                            )
                        else:
                            final_answer = {
                                "answer": decision.get("benchmark_answer", _benchmark_fallback(choices)) if general_v2 else _insufficient_choice(choices, pair.get("question_type")),
                                "explanation": decision.get("action_reason")
                                or _missing_evidence_text(decision),
                                "evidence_refs": decision.get("evidence_refs", []),
                                "confidence": None,
                                "confidence_status": "uncalibrated",
                                "raw_output": decision.get("raw_texts", {}).get("action_raw"),
                                "parse_status": decision.get("parse_status", "invalid_action_json"),
                                "evidence_sufficient": False,
                                "abstain_recommended": True,
                                "unsupported_answer_reason": decision.get("unsupported_answer_reason")
                                or _missing_evidence_text(decision),
                            }
                        process_item.update({"answer": final_answer["answer"], "explanation": final_answer["explanation"]})
                        termination_reason = "abstained"
                        break

                    if action["type"] == "inspect":
                        targets = [name for name in action["target_patches"] if name in descriptions]
                        targets = targets or patches_this_round[:1]
                        if patho_model is None:
                            action["type"] = "abstain"
                            continue
                        for patch_name in targets:
                            _describe_patch(
                                patch_name,
                                descriptions,
                                args,
                                slide_id,
                                question,
                                choices,
                                patho_model,
                                patho_processor,
                                recorder,
                                step_id,
                                attempt,
                                evidence_magnifications.get(patch_name, 5),
                                operation="executor_requested_inspect",
                                inspection_focus=_safe_inspection_focus(
                                    action.get("query")
                                    or _missing_evidence_text(decision)
                                    or question,
                                    choices,
                                ),
                                description_cache=patho_description_cache,
                                cache_path=cache_path,
                                image_provider=rollout_image_provider,
                                evidence_metadata=evidence_metadata,
                            )
                            described_online.add(patch_name)
                        patches_this_round = targets
                        continue

                    if action["type"] == "retrieve":
                        if not remaining:
                            patches_this_round = []
                            continue
                        query = (
                            action.get("query")
                            or _missing_evidence_text(decision)
                            or question
                        )
                        query_emb = _plip_text(plip, query, recorder, "replenish_query_embedding", step_id, attempt)
                        query_emb = query_emb / np.linalg.norm(query_emb, axis=-1, keepdims=True)
                        remaining_matrix = np.stack([feature_cache[name] for name in remaining])
                        replenished = _rank(remaining, remaining_matrix, query_emb)
                        add_count = replenishment_count(
                            len(available), args.replenish_ratio, getattr(args, "replenish_top_k", None)
                        )
                        selected = replenished[:add_count]
                        patches_this_round = [name for name, _ in selected]
                        accumulated.extend(patches_this_round)
                        remaining = [name for name in remaining if name not in patches_this_round]
                        recorder.record_state(
                            "retrieve",
                            {"query": query, "selected": [{"patch_id": name, "score": score} for name, score in selected]},
                            step_id=step_id,
                            attempt=attempt,
                        )
                        continue

                    if action["type"] == "zoom":
                        if patho_model is None:
                            patches_this_round = []
                            continue
                        requested_mag = action.get("magnification")
                        try:
                            requested_mag = int(requested_mag or 20)
                        except (TypeError, ValueError):
                            requested_mag = 20
                        zoom_level = requested_mag if requested_mag in {10, 20} else 20
                        if zoom_level != requested_mag:
                            recorder.record_state(
                                "focus_magnification_normalized",
                                {"requested": requested_mag, "normalized": zoom_level},
                                step_id=step_id,
                                attempt=attempt,
                            )
                        targets = [
                            name
                            for name in action["target_patches"]
                            if name in accumulated
                            and evidence_magnifications.get(name, 5) == 5
                        ]
                        targets = targets or [
                            name
                            for name in patches_this_round
                            if evidence_magnifications.get(name, 5) == 5
                        ]
                        if not targets:
                            patches_this_round = []
                            continue
                        target_matrix = np.stack([feature_cache[name] for name in targets])
                        zoom_query = (
                            action.get("query")
                            or _missing_evidence_text(decision)
                            or question
                        )
                        zoom_query_emb = _plip_text(plip, zoom_query, recorder, "zoom_query_embedding", step_id, attempt)
                        zoom_query_emb = zoom_query_emb / np.linalg.norm(zoom_query_emb, axis=-1, keepdims=True)
                        parent_ranked = _rank(targets, target_matrix, zoom_query_emb)[: args.focus_parent_top_k]
                        candidate_images = []
                        candidate_ids = []
                        candidate_meta = []
                        if zoom_backend == "wsi":
                            with WSIPyramidReader(wsi_rows[slide_id]) as reader:
                                for parent_name, parent_score in parent_ranked:
                                    for child in partition_region(evidence_regions[parent_name], zoom_level):
                                        observation = reader.read(child, 224)
                                        grandqc_tissue_fraction = mask_fraction_for_region(
                                            focus_tissue_mask,
                                            int(wsi_rows[slide_id]["width"]),
                                            int(wsi_rows[slide_id]["height"]),
                                            child,
                                        )
                                        white_fraction = image_white_fraction(observation.image)
                                        candidate_class = classify_focus_candidate(
                                            grandqc_tissue_fraction, white_fraction
                                        )
                                        candidate_images.append(observation.image)
                                        candidate_ids.append(child.patch_id)
                                        candidate_meta.append(
                                            {
                                                **observation.metadata,
                                                "parent_plip_score": parent_score,
                                                "evidence_source": "original_wsi_pyramid",
                                                "grandqc_mask_path": str(focus_mask_path.resolve()),
                                                "grandqc_tissue_fraction": grandqc_tissue_fraction,
                                                "white_fraction": white_fraction,
                                                "candidate_class": candidate_class,
                                            }
                                        )
                        else:
                            for parent_name, parent_score in parent_ranked:
                                patch_path = evidence_paths[parent_name]
                                for sub_image, (x, y) in split_patch_for_zoom(
                                    patch_path,
                                    zoom_level,
                                    source_magnification=5,
                                ):
                                    candidate_images.append(sub_image)
                                    candidate_id = f"{x}_{y}_m{zoom_level}_{Path(parent_name).stem}.jpg"
                                    candidate_ids.append(candidate_id)
                                    candidate_meta.append(
                                        {
                                            "x": x,
                                            "y": y,
                                            "parent_patch_id": parent_name,
                                            "magnification": zoom_level,
                                            "parent_plip_score": parent_score,
                                            "evidence_source": "legacy_jpeg_crop",
                                        }
                                    )
                        if not candidate_images:
                            patches_this_round = []
                            continue
                        sub_embeddings = _plip_images(
                            plip, candidate_images, candidate_ids, recorder, "zoom_subpatch_embeddings", step_id, attempt
                        )
                        sub_embeddings = sub_embeddings / np.linalg.norm(sub_embeddings, axis=-1, keepdims=True)
                        if zoom_backend == "wsi":
                            raw_scores = np.atleast_1d(
                                (sub_embeddings @ zoom_query_emb.T).squeeze()
                            )
                            # Strategy C is enabled from the blank-safety A/B/C experiment. Its
                            # target-morphology recall is intentionally marked unvalidated in traces/docs.
                            selected_ranked, focus_ranking_rows = rank_focus_candidates_c(
                                candidate_ids,
                                raw_scores,
                                candidate_meta,
                                top_k=2,
                            )
                            ranking_by_id = {
                                row["patch_id"]: row for row in focus_ranking_rows
                            }
                            selected_ids = [patch_id for patch_id, _score in selected_ranked]
                            recorder.record_state(
                                "focus_candidate_ranking",
                                {
                                    "strategy": "C_hard_blank_gate_ambiguous_soft_penalty_top2",
                                    "requested_magnification": requested_mag,
                                    "effective_magnification": zoom_level,
                                    "query": zoom_query,
                                    "parents": [
                                        {"patch_id": patch_id, "score": score}
                                        for patch_id, score in parent_ranked
                                    ],
                                    "grandqc_mask_path": str(focus_mask_path.resolve()),
                                    "target_morphology_recall_status": "not_yet_validated",
                                    "candidates": [
                                        {
                                            **meta,
                                            **ranking_by_id[candidate_id],
                                        }
                                        for candidate_id, meta in zip(candidate_ids, candidate_meta)
                                    ],
                                    "selected_patch_ids": selected_ids,
                                    "selected_patch_id": selected_ids[0] if selected_ids else None,
                                    "requested_top_k": 2,
                                    "returned_top_k": len(selected_ids),
                                },
                                step_id=step_id,
                                attempt=attempt,
                            )
                            if not selected_ids:
                                recorder.record_state(
                                    "focus_no_eligible_candidate",
                                    {
                                        "reason": "all_focus_candidates_invalid",
                                        "strategy": "C_hard_blank_gate_ambiguous_soft_penalty_top2",
                                    },
                                    step_id=step_id,
                                    attempt=attempt,
                                )
                                process_item.update(
                                    {
                                        "selected_zoom_patches": [],
                                        "focus_status": "all_focus_candidates_invalid",
                                    }
                                )
                                focus_actions_used += 1
                                patches_this_round = []
                                continue

                            zoom_dir = trace_root / "focus_observations" / trace_id
                            zoom_dir.mkdir(parents=True, exist_ok=True)
                            zoom_descriptions = {}
                            for best_id in selected_ids:
                                best_index = candidate_ids.index(best_id)
                                candidate_gate_meta = candidate_meta[best_index]
                                best_region = WSIRegion(
                                    slide_id=slide_id,
                                    patch_id=best_id,
                                    x_level0=int(candidate_gate_meta["x_level0"]),
                                    y_level0=int(candidate_gate_meta["y_level0"]),
                                    width_level0=int(candidate_gate_meta["width_level0"]),
                                    height_level0=int(candidate_gate_meta["height_level0"]),
                                    mpp_x=float(candidate_gate_meta["mpp_x"]),
                                    mpp_y=float(candidate_gate_meta["mpp_y"]),
                                    magnification=int(candidate_gate_meta["magnification"]),
                                    parent_patch_id=candidate_gate_meta["parent_patch_id"],
                                )
                                with WSIPyramidReader(wsi_rows[slide_id]) as reader:
                                    patho_observation = reader.read(
                                        best_region, args.patho_observation_size
                                    )
                                patho_image = patho_observation.image
                                registered_patch = {
                                    **candidate_gate_meta,
                                    **patho_observation.metadata,
                                    **ranking_by_id[best_id],
                                    "patch_id": best_id,
                                    "evidence_source": "original_wsi_pyramid",
                                    "evidence_type": "wsi_focus_region",
                                }
                                zoom_path = zoom_dir / f"{best_id}.png"
                                patho_image.save(zoom_path, format="PNG")
                                registered_patch["path"] = str(zoom_path.resolve())
                                evidence_regions[best_id] = best_region
                                recorder.record_state(
                                    "evidence_registered",
                                    {"patch": registered_patch},
                                    step_id=step_id,
                                    attempt=attempt,
                                )
                                x = registered_patch["x_level0"]
                                y = registered_patch["y_level0"]
                                zoom_description = patho_r1_describe(
                                    patho_image,
                                    question=None if general_v2 else question,
                                    patho_r1_processor=patho_processor,
                                    patho_r1_model=patho_model,
                                    coords=(x, y),
                                    magnification=int(registered_patch["magnification"]),
                                    choices=choices,
                                    morphology_only=True,
                                    inspection_focus=_safe_inspection_focus(zoom_query, choices),
                                    prompt_version=args.patho_prompt_version,
                                    trace_recorder=recorder,
                                    trace_context={"step_id": step_id, "attempt": attempt},
                                    patch_id=best_id,
                                    operation="zoom_description",
                                    max_new_tokens=args.patho_max_new_tokens,
                                )
                                zoom_descriptions[best_id] = zoom_description
                                descriptions[best_id] = zoom_description
                                if best_id not in accumulated:
                                    accumulated.append(best_id)
                                feature_cache[best_id] = sub_embeddings[best_index]
                                evidence_paths[best_id] = zoom_path
                                evidence_magnifications[best_id] = int(
                                    registered_patch["magnification"]
                                )
                                evidence_metadata[best_id] = registered_patch
                                described_online.add(best_id)

                            first_id = selected_ids[0]
                            process_item.update(
                                {
                                    "selected_zoom_patch": first_id,
                                    "selected_zoom_patches": selected_ids,
                                    "zoom_patch_desc": zoom_descriptions[first_id],
                                    "zoom_patch_descs": zoom_descriptions,
                                    "focus_status": "selected",
                                }
                            )
                            focus_actions_used += 1
                            patches_this_round = selected_ids
                            continue

                        sub_ranked = _rank(candidate_ids, sub_embeddings, zoom_query_emb)
                        best_id = sub_ranked[0][0]
                        best_index = candidate_ids.index(best_id)
                        best_meta = candidate_meta[best_index]
                        x = best_meta.get("x_level0", best_meta.get("x"))
                        y = best_meta.get("y_level0", best_meta.get("y"))
                        zoom_level = int(best_meta["magnification"])
                        zoom_dir = trace_root / "focus_observations" / trace_id
                        zoom_dir.mkdir(parents=True, exist_ok=True)
                        patho_image = candidate_images[best_index]
                        zoom_path = zoom_dir / best_id
                        patho_image.save(zoom_path, format="JPEG", quality=95)
                        candidate_scores = {patch_id: score for patch_id, score in sub_ranked}
                        recorder.record_state(
                            "focus_candidate_ranking",
                            {
                                "requested_magnification": requested_mag,
                                "effective_magnification": zoom_level,
                                "query": zoom_query,
                                "parents": [
                                    {"patch_id": patch_id, "score": score}
                                    for patch_id, score in parent_ranked
                                ],
                                "candidates": [
                                    {
                                        **meta,
                                        "patch_id": candidate_id,
                                        "plip_score": candidate_scores[candidate_id],
                                    }
                                    for candidate_id, meta in zip(candidate_ids, candidate_meta)
                                ],
                                "selected_patch_id": best_id,
                            },
                            step_id=step_id,
                            attempt=attempt,
                        )
                        registered_patch = {
                            **best_meta,
                            "patch_id": best_id,
                            "path": str(zoom_path.resolve()),
                            "evidence_type": "zoom_subpatch",
                        }
                        recorder.record_state(
                            "evidence_registered",
                            {"patch": registered_patch},
                            step_id=step_id,
                            attempt=attempt,
                        )
                        zoom_description = patho_r1_describe(
                            patho_image,
                            question=None if general_v2 else question,
                            patho_r1_processor=patho_processor,
                            patho_r1_model=patho_model,
                            coords=(x, y),
                            magnification=zoom_level,
                            choices=choices,
                            morphology_only=True,
                            inspection_focus=_safe_inspection_focus(zoom_query, choices),
                            prompt_version=args.patho_prompt_version,
                            trace_recorder=recorder,
                            trace_context={"step_id": step_id, "attempt": attempt},
                            patch_id=best_id,
                            operation="zoom_description",
                            max_new_tokens=args.patho_max_new_tokens,
                        )
                        process_item.update({"selected_zoom_patch": best_id, "zoom_patch_desc": zoom_description})
                        descriptions[best_id] = zoom_description
                        accumulated.append(best_id)
                        feature_cache[best_id] = sub_embeddings[best_index]
                        evidence_paths[best_id] = zoom_path
                        evidence_magnifications[best_id] = zoom_level
                        evidence_metadata[best_id] = registered_patch
                        described_online.add(best_id)
                        focus_actions_used += 1
                        patches_this_round = [best_id]
                        continue

                if final_answer is None:
                    if evidence_policy == "contract_v1":
                        final_answer = _contract_final_from_candidate(
                            best_candidate,
                            choices,
                            explanation="Maximum action steps reached without unique cited contract support.",
                            stop_reason_detail="Maximum action steps reached without contract_v1 sufficiency.",
                        )
                    else:
                        final_answer = {
                            "answer": _benchmark_fallback(choices) if general_v2 else _insufficient_choice(choices, pair.get("question_type")),
                            "explanation": "Maximum action steps reached without sufficient visible evidence.",
                            "evidence_refs": [],
                            "confidence": None,
                            "confidence_status": "uncalibrated",
                            "raw_output": None,
                            "evidence_sufficient": False,
                            "abstain_recommended": True,
                            "unsupported_answer_reason": "Maximum action steps reached.",
                        }
                    termination_reason = termination_reason or "budget_exhausted"
                if evidence_policy == "contract_v1":
                    assist_fields = _pathologist_assist_fields(
                        final_answer.get("evidence_contract_verification"),
                        str(final_answer["answer"]),
                        final_answer.get("ranked_differential", []),
                    )
                    final_answer.update(assist_fields)
                    # The historical field remains an explicit compatibility
                    # alias of the unchanged strict contract layer.
                    final_answer["evidence_sufficient"] = bool(
                        assist_fields["strict_evidence_sufficient"]
                    )
                raw_trace = recorder.finalize(
                    {
                        "answer": final_answer["answer"],
                        "evidence_refs": final_answer.get("evidence_refs", []),
                        "explanation": final_answer["explanation"],
                        "confidence": final_answer.get("confidence"),
                        "confidence_status": final_answer.get("confidence_status", "uncalibrated"),
                        "evidence_policy": final_answer.get("evidence_policy", evidence_policy),
                        "evidence_sufficient": final_answer.get("evidence_sufficient"),
                        "evidence_found": final_answer.get("evidence_found", False),
                        "citation_valid": final_answer.get("citation_valid", False),
                        "citation_supports_answer": final_answer.get("citation_supports_answer", False),
                        "abstain_recommended": final_answer.get("abstain_recommended"),
                        "unsupported_answer_reason": final_answer.get("unsupported_answer_reason", ""),
                        "selected_candidate_attempt": final_answer.get("selected_candidate_attempt"),
                        "evidence_contract_verification": final_answer.get("evidence_contract_verification", {}),
                        **{
                            field: deepcopy(final_answer.get(field))
                            for field in PATHOLOGIST_ASSIST_OUTPUT_FIELDS
                            if field in final_answer
                        },
                        "parse_status": final_answer.get("parse_status", "not_applicable"),
                        "stop_reason": termination_reason or "unknown_termination",
                        "total_steps": len(process),
                        "total_visible_patches": len(accumulated),
                    }
                )
            except ExecutorAPIError as exc:
                recorder.finalize(
                    {"answer": None, "evidence_refs": [], "explanation": None, "stop_reason": "executor_api_failure"},
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                executor_api_failures += 1
                consecutive_executor_api_failures += 1
                failed_result = {
                    "long_id": slide_id,
                    "case_id": pair.get("case_id"),
                    "question_id": question_id,
                    "question": question,
                    "question_type": pair.get("question_type"),
                    "difficulty": pair.get("difficulty"),
                    "choices": choices,
                    "ground_truth": pair.get("answer", ""),
                    "pred_answer": None,
                    "explanation": None,
                    "parse_status": "executor_api_failure",
                    "trace_id": trace_id,
                    "rollout_id": rollout_id,
                    "process": process,
                }
                _write_json_atomic(result_path, failed_result)
                all_results.append(failed_result)
                write_executor_summary("running_with_api_failures")
                failure_rate = executor_api_failures / expected_rollouts if expected_rollouts else 1.0
                print(
                    f"[{pair_index + 1}/{len(pairs)} rollout={rollout_id}] {question_id}: "
                    f"Executor API failure ({executor_api_failures}/{expected_rollouts}): {exc}"
                )
                if (
                    consecutive_executor_api_failures >= args.executor_max_consecutive_failures
                    or failure_rate > args.executor_max_failure_rate
                ):
                    write_executor_summary("stopped_by_api_failure_gate")
                    raise ExecutorAPIError(
                        "Executor batch stopped by failure gate: "
                        f"consecutive={consecutive_executor_api_failures}, rate={failure_rate:.2%}."
                    ) from exc
                continue
            except Exception as exc:
                recorder.finalize(
                    {"answer": None, "evidence_refs": [], "explanation": None, "stop_reason": "runtime_failure"},
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                write_executor_summary("failed")
                raise

            legacy_result = {
                "long_id": slide_id,
                "case_id": pair.get("case_id"),
                "question_id": question_id,
                "question": question,
                "question_type": pair.get("question_type"),
                "difficulty": pair.get("difficulty"),
                "evidence_tier": pair.get("evidence_tier", "strict"),
                "choices": choices,
                "ground_truth": pair.get("answer", ""),
                "pred_answer": final_answer["answer"],
                "benchmark_answer": final_answer["answer"],
                "evidence_policy": final_answer.get("evidence_policy", evidence_policy),
                "evidence_sufficient": final_answer.get("evidence_sufficient"),
                "evidence_found": final_answer.get("evidence_found", False),
                "citation_valid": final_answer.get("citation_valid", False),
                "citation_supports_answer": final_answer.get("citation_supports_answer", False),
                "abstain_recommended": final_answer.get("abstain_recommended"),
                "unsupported_answer_reason": final_answer.get("unsupported_answer_reason", ""),
                "selected_candidate_attempt": final_answer.get("selected_candidate_attempt"),
                "evidence_contract_verification": final_answer.get("evidence_contract_verification", {}),
                **{
                    field: deepcopy(final_answer.get(field))
                    for field in PATHOLOGIST_ASSIST_OUTPUT_FIELDS
                    if field in final_answer
                },
                "evidence_refs": final_answer.get("evidence_refs", []),
                "explanation": final_answer["explanation"],
                "confidence": final_answer.get("confidence"),
                "confidence_status": final_answer.get("confidence_status", "uncalibrated"),
                "parse_status": final_answer.get("parse_status", "not_applicable"),
                "trace_id": trace_id,
                "rollout_id": rollout_id,
                "process": process,
            }
            _write_json_atomic(result_path, legacy_result)
            all_results.append(legacy_result)
            consecutive_executor_api_failures = 0
            write_executor_summary("running")
            print(
                f"[{pair_index + 1}/{len(pairs)} rollout={rollout_id}] "
                f"{question_id}: {final_answer['answer']} (trace events={len(raw_trace['events'])})"
            )
            gc.collect()
            torch.cuda.empty_cache()

    write_executor_summary("completed")
    print(f"Completed {len(all_results)} new rollouts. Traces: {trace_root}")
    return all_results
