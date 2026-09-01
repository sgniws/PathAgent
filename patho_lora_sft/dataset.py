from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .audit import CanonicalDecision, validate_accepted_decision
from .common import ContractViolation, PrivacyViolation, atomic_write_jsonl
from .contracts import TEST_SPLIT_NAMES


SYSTEM_TEXT = "You are a pathology image morphology recorder. Report only directly visible H&E morphology and follow the requested output contract."
FORBIDDEN_TRAINER_KEYS = frozenset(
    {
        "patient_group_id",
        "slide_id",
        "patch_id",
        "x_level0",
        "y_level0",
        "report",
        "diagnosis",
        "judge",
        "teacher",
        "cost",
    }
)


def build_training_record(
    *, sample_id: str, image_relpath: str, user_prompt: str, decision: CanonicalDecision
) -> dict[str, Any]:
    validate_accepted_decision(decision)
    if image_relpath.startswith("/") or ".." in Path(image_relpath).parts:
        raise PrivacyViolation("Trainer image path must be a contained relative path")
    record = {
        "sample_id": sample_id,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_TEXT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_relpath},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": decision.target}]},
        ],
    }
    assert_trainer_record_private(record)
    return record


def assert_trainer_record_private(record: dict[str, Any]) -> None:
    def walk(value: Any, key: str | None = None) -> None:
        if key is not None and key.casefold() in FORBIDDEN_TRAINER_KEYS:
            raise PrivacyViolation(f"Private provenance leaked into trainer record: {key}")
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                walk(child, key)

    walk(record)


def validate_dataset_isolation(public_rows: Iterable[dict[str, Any]], private_rows: Iterable[dict[str, Any]]) -> None:
    public = list(public_rows)
    private = list(private_rows)
    public_ids = [str(row["sample_id"]) for row in public]
    private_ids = [str(row["sample_id"]) for row in private]
    if len(public_ids) != len(set(public_ids)) or set(public_ids) != set(private_ids):
        raise ContractViolation("Public/private sample IDs are not a unique one-to-one join")
    patient_splits: dict[str, set[str]] = defaultdict(set)
    slide_splits: dict[str, set[str]] = defaultdict(set)
    coordinate_keys: set[tuple[str, str, int, int]] = set()
    image_hashes: dict[str, str] = {}
    for row in private:
        split = str(row["split"]).casefold()
        if split in TEST_SPLIT_NAMES:
            raise PrivacyViolation("Test10 provenance entered an SFT sidecar")
        patient_splits[str(row["patient_group_id"])].add(split)
        slide_splits[str(row["slide_id"])].add(split)
        coordinate = (str(row["slide_id"]), str(row["patch_id"]), int(row["x_level0"]), int(row["y_level0"]))
        if coordinate in coordinate_keys:
            raise ContractViolation("Duplicate source coordinate key")
        coordinate_keys.add(coordinate)
        image_sha = str(row["image_sha256"])
        if image_sha in image_hashes and image_hashes[image_sha] != split:
            raise PrivacyViolation("Identical image hash crossed a split")
        image_hashes[image_sha] = split
    if any(len(value) != 1 for value in patient_splits.values()) or any(len(value) != 1 for value in slide_splits.values()):
        raise PrivacyViolation("Patient or WSI crossed an SFT split")


def assistant_only_labels(input_ids: Sequence[int], assistant_token_spans: Sequence[tuple[int, int]]) -> list[int]:
    labels = [-100] * len(input_ids)
    covered: set[int] = set()
    for start, end in assistant_token_spans:
        if not (0 <= start < end <= len(input_ids)):
            raise ContractViolation("Assistant token span is out of bounds or empty")
        for index in range(start, end):
            if index in covered:
                raise ContractViolation("Assistant token spans overlap")
            labels[index] = int(input_ids[index])
            covered.add(index)
    if not covered:
        raise ContractViolation("Assistant target mask is empty")
    return labels


def audit_loss_mask(
    labels: Sequence[int], *, system_span: tuple[int, int], user_span: tuple[int, int], image_token_indices: Iterable[int], assistant_span: tuple[int, int]
) -> dict[str, Any]:
    def active(span: tuple[int, int]) -> int:
        return sum(labels[index] != -100 for index in range(*span))

    image_active = sum(labels[index] != -100 for index in image_token_indices)
    assistant_active = active(assistant_span)
    passed = active(system_span) == 0 and active(user_span) == 0 and image_active == 0 and assistant_active == assistant_span[1] - assistant_span[0]
    if not passed:
        raise ContractViolation("Loss mask includes non-assistant tokens or excludes assistant target tokens")
    return {
        "system_active": 0,
        "user_active": 0,
        "image_active": 0,
        "assistant_active": assistant_active,
        "passed": True,
    }


def export_training_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    for row in materialized:
        assert_trainer_record_private(row)
    atomic_write_jsonl(path, materialized, mode=0o600)
