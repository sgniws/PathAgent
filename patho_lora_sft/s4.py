from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from .common import ContractViolation, PrivacyViolation, sha256_file
from .dataset import build_training_record
from .audit import CanonicalDecision
from .schema import parse_wrapped_findings


TARGET_MODULE_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)
EXPECTED_LANGUAGE_LAYER_COUNT = 28
EXPECTED_TARGETS_PER_LAYER = 7
EXPECTED_TARGET_MODULE_COUNT = EXPECTED_LANGUAGE_LAYER_COUNT * EXPECTED_TARGETS_PER_LAYER
EXPECTED_PILOT_COUNTS = {"real": 190, "control": 10, "total": 200}
EMPTY_TARGET = '<answer>{"findings":[]}</answer>'


def validate_s3_attested_human_gate(attestation: dict[str, Any]) -> dict[str, bool]:
    conclusions = attestation.get("attested_conclusions", {})
    gates = {
        "unsupported_plus_not_assessable_at_most_5_percent":
            conclusions.get("unsupported_plus_not_assessable_at_most_5_percent") is True,
        "forbidden_inference_zero": conclusions.get("forbidden_inference_zero") is True,
        "anonymous_repeat_consistency_at_least_90_percent":
            conclusions.get("anonymous_repeat_consistency_at_least_90_percent") is True,
        "approve_s3_gate_closure": conclusions.get("approve_s3_gate_closure") is True,
        "approve_s4_execution": conclusions.get("approve_s4_execution") is True,
    }
    if not all(gates.values()):
        raise ContractViolation("S3 direct-review attestation does not cover every frozen human hard gate")
    return gates


def extract_user_prompt(frozen_prompt: str) -> str:
    marker = "\nUSER\n"
    if marker not in frozen_prompt:
        raise ContractViolation("Frozen morphology prompt lacks the USER section")
    user_prompt = frozen_prompt.split(marker, 1)[1].strip()
    if not user_prompt or "<answer>" not in user_prompt:
        raise ContractViolation("Frozen morphology user prompt is incomplete")
    return user_prompt


def build_pilot_training_records(
    real_records: Sequence[dict[str, Any]],
    control_records: Sequence[dict[str, Any]],
    private_rows: Sequence[dict[str, Any]],
    *,
    user_prompt: str,
) -> list[dict[str, Any]]:
    if len(real_records) != EXPECTED_PILOT_COUNTS["real"]:
        raise ContractViolation("S4 requires exactly 190 accepted real pilot records")
    if len(control_records) != EXPECTED_PILOT_COUNTS["control"]:
        raise ContractViolation("S4 requires exactly 10 blank control records")
    private_by_id = {str(row["sample_id"]): row for row in private_rows}
    if len(private_by_id) != len(private_rows) or set(private_by_id) != {str(row["sample_id"]) for row in real_records}:
        raise ContractViolation("S4 real public/private provenance join changed")

    records: list[dict[str, Any]] = []
    for row in real_records:
        sample_id = str(row["sample_id"])
        if row.get("assigned_split") != "train" or row.get("assessment", {}).get("status") not in {
            "accepted_automatic", "waiting_human_review"
        }:
            raise ContractViolation("S4 received a non-Train or non-accepted real record")
        target = str(row.get("assessment", {}).get("target", ""))
        parsed = parse_wrapped_findings(target)
        if not parsed.valid:
            raise ContractViolation("S4 real target is not canonical wrapped findings JSON")
        decision = CanonicalDecision("accepted", str(row["tier"]), "luna", target, parsed.findings, "s3_accepted", False)
        image_relpath = str(private_by_id[sample_id]["image_relpath_from_run"])
        record = build_training_record(
            sample_id=sample_id,
            image_relpath=image_relpath,
            user_prompt=user_prompt,
            decision=decision,
        )
        record["record_kind"] = "real"
        records.append(record)

    for row in control_records:
        control_id = str(row["control_id"])
        if not row.get("strict_schema_valid") or not row.get("exact_empty") or row.get("findings") != []:
            raise ContractViolation("S4 control record is not exact-empty")
        decision = CanonicalDecision("accepted", "control", "frozen_empty", EMPTY_TARGET, (), "blank_control", False)
        record = build_training_record(
            sample_id=control_id,
            image_relpath=f"images/calibration50/{control_id}.png",
            user_prompt=user_prompt,
            decision=decision,
        )
        record["record_kind"] = "control"
        records.append(record)

    records.sort(key=lambda row: str(row["sample_id"]))
    ids = [str(row["sample_id"]) for row in records]
    if len(records) != EXPECTED_PILOT_COUNTS["total"] or len(ids) != len(set(ids)):
        raise ContractViolation("S4 pilot training IDs are not exactly 200 unique values")
    if Counter(str(row["record_kind"]) for row in records) != Counter(real=190, control=10):
        raise ContractViolation("S4 pilot real/control composition changed")
    return records


def validate_training_images(run_root: Path, records: Sequence[dict[str, Any]], expected_hashes: dict[str, str]) -> dict[str, Any]:
    from PIL import Image

    observed_hashes: set[str] = set()
    total_bytes = 0
    for row in records:
        image_relpath = str(row["messages"][1]["content"][0]["image"])
        if image_relpath.startswith("/") or ".." in Path(image_relpath).parts:
            raise PrivacyViolation("S4 trainer image path escaped the run root")
        image_path = (run_root / image_relpath).resolve()
        try:
            image_path.relative_to(run_root.resolve())
        except ValueError as exc:
            raise PrivacyViolation("S4 resolved image escaped the run root") from exc
        if not image_path.is_file():
            raise ContractViolation("S4 training image is missing")
        digest = sha256_file(image_path)
        expected = expected_hashes.get(str(row["sample_id"]))
        if expected is not None and digest != expected:
            raise ContractViolation("S4 training image hash changed")
        with Image.open(image_path) as image:
            if image.size != (784, 784) or image.mode != "RGB" or image.format != "PNG":
                raise ContractViolation("S4 training image is not a 784x784 RGB PNG")
        observed_hashes.add(digest)
        total_bytes += image_path.stat().st_size
    if len(observed_hashes) != len(records):
        raise ContractViolation("S4 training images are not unique")
    return {
        "record_count": len(records),
        "unique_image_hash_count": len(observed_hashes),
        "total_image_bytes": total_bytes,
        "all_images_784_rgb_png": True,
        "full_wsi_reads": 0,
    }


def select_target_modules(module_names: Iterable[str], target_pattern: str) -> list[str]:
    if target_pattern != TARGET_MODULE_RE.pattern:
        raise ContractViolation("S4 target-module regex differs from the frozen full-path whitelist")
    def canonical(runtime_name: str) -> str:
        # Transformers 5.x wraps the language backbone as
        # ``model.language_model`` although the checkpoint keys and frozen
        # contract use ``model``.  Normalize only that exact framework-owned
        # prefix; suffix matching remains forbidden and visual names cannot
        # enter the whitelist.
        prefix = "model.language_model.layers."
        if runtime_name.startswith(prefix):
            return "model.layers." + runtime_name[len(prefix):]
        return runtime_name

    selected = sorted(
        str(name) for name in module_names if TARGET_MODULE_RE.fullmatch(canonical(str(name)))
    )
    if len(selected) != EXPECTED_TARGET_MODULE_COUNT:
        raise ContractViolation(f"Expected {EXPECTED_TARGET_MODULE_COUNT} language LoRA targets, found {len(selected)}")
    canonical_selected = [canonical(name) for name in selected]
    per_layer = Counter(int(name.split(".")[2]) for name in canonical_selected)
    if set(per_layer) != set(range(EXPECTED_LANGUAGE_LAYER_COUNT)) or set(per_layer.values()) != {EXPECTED_TARGETS_PER_LAYER}:
        raise ContractViolation("S4 LoRA target coverage is not seven modules in every language layer")
    if any("visual." in name for name in selected):
        raise ContractViolation("A visual module entered the S4 LoRA whitelist")
    return selected


def audit_trainable_parameter_names(names: Iterable[str], selected_modules: Sequence[str]) -> dict[str, Any]:
    materialized = sorted(str(name) for name in names)
    if not materialized:
        raise ContractViolation("S4 model has no trainable LoRA parameters")
    forbidden_fragments = ("visual.", "model.embed_tokens", "lm_head")
    if any(any(fragment in name for fragment in forbidden_fragments) for name in materialized):
        raise ContractViolation("S4 trainable parameters entered visual/embed/lm_head modules")
    uncovered = [name for name in materialized if not any(module in name for module in selected_modules)]
    if uncovered:
        raise ContractViolation("S4 trainable parameter lies outside the exact language-module whitelist")
    if any("lora_" not in name for name in materialized):
        raise ContractViolation("S4 found a non-LoRA trainable parameter")
    return {
        "trainable_tensor_count": len(materialized),
        "all_lora_only": True,
        "visual_trainable_count": 0,
        "embed_tokens_trainable_count": 0,
        "lm_head_trainable_count": 0,
    }


def expanded_assistant_boundary(
    expanded_ids: Sequence[int],
    unexpanded_full_ids: Sequence[int],
    unexpanded_prompt_ids: Sequence[int],
) -> int:
    prompt = list(unexpanded_prompt_ids)
    full = list(unexpanded_full_ids)
    expanded = list(expanded_ids)
    if full[: len(prompt)] != prompt:
        raise ContractViolation("S4 chat-template prompt is not a token prefix of the training sequence")
    expansion = len(expanded) - len(full)
    boundary = len(prompt) + expansion
    if expansion < 0 or boundary <= 0 or boundary >= len(expanded):
        raise ContractViolation("S4 assistant token boundary is invalid")
    if expanded[boundary:] != full[len(prompt):]:
        raise ContractViolation("S4 multimodal token expansion crossed the assistant boundary")
    return boundary


def summarize_mask(labels: Sequence[int], *, assistant_start: int, image_token_id: int, input_ids: Sequence[int]) -> dict[str, Any]:
    if len(labels) != len(input_ids) or not 0 < assistant_start < len(labels):
        raise ContractViolation("S4 loss-mask dimensions are invalid")
    prefix_active = sum(int(value) != -100 for value in labels[:assistant_start])
    assistant_active = sum(int(value) != -100 for value in labels[assistant_start:])
    image_active = sum(
        int(label) != -100 for label, token in zip(labels, input_ids) if int(token) == int(image_token_id)
    )
    if prefix_active or image_active or assistant_active != len(labels) - assistant_start:
        raise ContractViolation("S4 loss mask is not assistant-only")
    return {
        "sequence_tokens": len(labels),
        "assistant_start": assistant_start,
        "assistant_tokens": assistant_active,
        "prefix_active": prefix_active,
        "image_active": image_active,
        "passed": True,
    }


def validate_canary_arm_summary(summary: dict[str, Any], *, expected_rank: int) -> None:
    required_true = (
        "dataset_train_only",
        "loss_mask_passed",
        "trainable_parameter_boundary_passed",
        "finite_loss",
        "gradient_nonzero_and_finite",
        "checkpoint_loadable",
        "adapter_unmerged",
        "base_checkpoint_unchanged",
    )
    if summary.get("rank") != expected_rank or not all(summary.get(key) is True for key in required_true):
        raise ContractViolation(f"S4 r{expected_rank} canary did not pass every engineering hard gate")
    if int(summary.get("optimizer_steps", 0)) <= 0 or int(summary.get("examples_seen", 0)) != 200:
        raise ContractViolation(f"S4 r{expected_rank} canary training coverage is incomplete")
    losses = summary.get("losses", [])
    if len(losses) != 200:
        raise ContractViolation(f"S4 r{expected_rank} did not record one finite loss per example")


def forecast_remaining_scale_budget(
    *,
    luna_p95_usd: float,
    gemini_p95_usd: float,
    remaining_records: int,
    maximum_replacements: int,
    uncertainty_multiplier: float,
    cumulative_project_spend_usd: float,
    prior_unconfirmed_reserve_usd: float,
    project_cap_usd: float,
    key_remaining_usd: float,
    minimum_account_reserve_usd: float,
) -> dict[str, Any]:
    if remaining_records != 2000 or maximum_replacements != 200 or uncertainty_multiplier != 1.2:
        raise ContractViolation("S5 remaining-call upper-bound contract changed")
    calls_per_role = remaining_records + maximum_replacements
    billable_completion_upper_bound = calls_per_role * 2
    forecast = (float(luna_p95_usd) + float(gemini_p95_usd)) * calls_per_role * uncertainty_multiplier
    projected_project_spend = cumulative_project_spend_usd + prior_unconfirmed_reserve_usd + forecast
    projected_key_remaining = key_remaining_usd - prior_unconfirmed_reserve_usd - forecast
    project_gate = projected_project_spend <= project_cap_usd
    account_gate = projected_key_remaining >= minimum_account_reserve_usd
    if not project_gate or not account_gate:
        raise ContractViolation("S5 P95+20% remaining scale forecast does not preserve the frozen budget reserve")
    return {
        "remaining_records": remaining_records,
        "maximum_replacements": maximum_replacements,
        "calls_per_role_upper_bound": calls_per_role,
        "billable_completion_upper_bound": billable_completion_upper_bound,
        "luna_p95_usd": float(luna_p95_usd),
        "gemini_p95_usd": float(gemini_p95_usd),
        "uncertainty_multiplier": uncertainty_multiplier,
        "forecast_scale_cost_usd": forecast,
        "prior_unconfirmed_reserve_usd": prior_unconfirmed_reserve_usd,
        "cumulative_project_spend_usd": cumulative_project_spend_usd,
        "projected_project_spend_usd": projected_project_spend,
        "project_cap_usd": project_cap_usd,
        "projected_key_remaining_usd": projected_key_remaining,
        "minimum_account_reserve_usd": minimum_account_reserve_usd,
        "project_budget_gate_passed": project_gate,
        "account_reserve_gate_passed": account_gate,
        "fresh_balance_and_endpoint_preflight_required_before_calls": True,
    }
