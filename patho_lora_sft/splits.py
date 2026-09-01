from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .common import ContractViolation, PrivacyViolation, atomic_write_bytes, atomic_write_jsonl, canonical_json
from .contracts import SPLIT_SEED, TEST_SPLIT_NAMES


SPLIT_MARKER_RE = re.compile(r'"split"\s*:\s*"(?P<split>[^"]+)"', re.I)
ALLOWED_PATCH_FIELDS = frozenset(
    {
        "slide_id",
        "patch_id",
        "x_level0",
        "y_level0",
        "width_level0",
        "height_level0",
        "mpp_x",
        "mpp_y",
        "physical_fov_um",
        "split",
        "selection_auto_status",
        "qc_risk",
    }
)


@dataclass
class GuardStats:
    source_lines: int = 0
    non_test_payload_rows_decoded: int = 0
    test_payload_rows_decoded: int = 0
    test_split_markers_rejected_before_decode: int = 0
    test_image_reads: int = 0


@dataclass(frozen=True)
class SplitContract:
    clean: int
    risk: int
    max_per_slide: int


SPLIT_CONTRACTS = {
    "train": SplitContract(clean=1800, risk=100, max_per_slide=16),
    "validation": SplitContract(clean=90, risk=5, max_per_slide=10),
    "locked": SplitContract(clean=90, risk=5, max_per_slide=10),
}


def guarded_non_test_jsonl(path: Path, stats: GuardStats) -> Iterator[dict[str, Any]]:
    """Decode no Test10 payload: inspect only its split marker and reject the line."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stats.source_lines += 1
            marker = SPLIT_MARKER_RE.search(raw)
            if not marker:
                raise ContractViolation(f"Missing split marker in {path}:{line_number}")
            split = marker.group("split").casefold()
            if split in TEST_SPLIT_NAMES:
                stats.test_split_markers_rejected_before_decode += 1
                continue
            row = json.loads(raw)
            stats.non_test_payload_rows_decoded += 1
            if str(row.get("split", "")).casefold() in TEST_SPLIT_NAMES:
                stats.test_payload_rows_decoded += 1
                raise PrivacyViolation("Test10 payload crossed the pre-decode split guard")
            yield row


def load_or_create_hmac_salt(path: Path) -> bytes:
    if path.exists():
        if path.stat().st_mode & 0o077:
            raise PrivacyViolation("Patient HMAC salt permissions must be 0600 or stricter")
        salt = path.read_bytes()
    else:
        salt = os.urandom(32)
        atomic_write_bytes(path, salt, mode=0o600)
    if len(salt) < 32:
        raise PrivacyViolation("Patient HMAC salt is too short")
    return salt


def patient_group_id(raw_patient_key: str, salt: bytes) -> str:
    if not raw_patient_key.strip():
        raise ContractViolation("Missing private patient key")
    return "pg_" + hmac.new(salt, raw_patient_key.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def _stable_score(*parts: Any, seed: int = SPLIT_SEED) -> str:
    value = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_non_test_pool(
    *,
    core_manifest: Path,
    wsi_metadata_manifests: Iterable[Path],
    patient_hmac_salt: bytes,
    expected_non_test_wsi: int = 163,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], GuardStats]:
    stats = GuardStats()
    patches: list[dict[str, Any]] = []
    seen_patch_keys: set[tuple[str, str]] = set()
    for raw in guarded_non_test_jsonl(core_manifest, stats):
        row = {key: raw.get(key) for key in ALLOWED_PATCH_FIELDS}
        key = (str(row["slide_id"]), str(row["patch_id"]))
        if key in seen_patch_keys:
            raise ContractViolation("Duplicate (slide_id, patch_id) in source manifest")
        seen_patch_keys.add(key)
        if row["x_level0"] is None or row["y_level0"] is None:
            raise ContractViolation("Patch is missing Level-0 coordinates")
        patches.append(row)

    metadata_by_slide: dict[str, dict[str, str]] = {}
    for manifest in wsi_metadata_manifests:
        for raw in guarded_non_test_jsonl(manifest, stats):
            slide_id = str(raw.get("slide_id", ""))
            raw_patient_key = str(raw.get("case_id") or raw.get("病例ID") or "")
            if not slide_id or not raw_patient_key:
                raise ContractViolation("Non-test WSI metadata lacks slide_id or patient key")
            private = {
                "slide_id": slide_id,
                "patient_group_id": patient_group_id(raw_patient_key, patient_hmac_salt),
                "source_split": str(raw.get("split", "")).casefold(),
            }
            existing = metadata_by_slide.get(slide_id)
            if existing is not None and existing != private:
                raise ContractViolation("Conflicting WSI metadata across source manifests")
            metadata_by_slide[slide_id] = private

    patch_slides = {str(row["slide_id"]) for row in patches}
    missing_metadata = patch_slides - set(metadata_by_slide)
    if missing_metadata:
        raise ContractViolation(f"Missing private patient metadata for {len(missing_metadata)} non-test WSI")
    if len(patch_slides) != expected_non_test_wsi:
        raise ContractViolation(f"Expected {expected_non_test_wsi} non-test WSI, found {len(patch_slides)}")
    assignments = [metadata_by_slide[slide_id] for slide_id in sorted(patch_slides)]
    groups = defaultdict(set)
    for row in assignments:
        groups[row["patient_group_id"]].add(row["source_split"])
    for row in patches:
        metadata = metadata_by_slide[str(row["slide_id"])]
        row["patient_group_id"] = metadata["patient_group_id"]
        if str(row["split"]).casefold() != metadata["source_split"]:
            raise ContractViolation("Patch and WSI source split disagree")
    if stats.test_payload_rows_decoded != 0 or stats.test_image_reads != 0:
        raise PrivacyViolation("Test10 access counter is nonzero")
    return patches, assignments, stats


def _patch_tier(row: dict[str, Any]) -> str | None:
    if bool(row.get("qc_risk")):
        return "risk"
    if str(row.get("selection_auto_status", "")).casefold() == "pass":
        return "clean"
    return None


def _group_capacity(
    group_ids: set[str],
    assignments: list[dict[str, Any]],
    patches: list[dict[str, Any]],
    max_per_slide: int,
) -> dict[str, int]:
    slides = {row["slide_id"] for row in assignments if row["patient_group_id"] in group_ids}
    per_slide_tier: dict[str, Counter[str]] = defaultdict(Counter)
    for patch in patches:
        if patch["slide_id"] in slides:
            tier = _patch_tier(patch)
            if tier:
                per_slide_tier[patch["slide_id"]][tier] += 1
    clean = sum(min(max_per_slide, counts["clean"]) for counts in per_slide_tier.values())
    risk = sum(min(max_per_slide, counts["risk"]) for counts in per_slide_tier.values())
    total = sum(min(max_per_slide, counts["clean"] + counts["risk"]) for counts in per_slide_tier.values())
    return {"slides": len(slides), "clean": clean, "risk": risk, "total": total}


def allocate_patient_groups(
    patches: list[dict[str, Any]], assignments: list[dict[str, Any]], *, minimum_eval_wsi: int = 12
) -> list[dict[str, Any]]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        group_splits[row["patient_group_id"]].add(row["source_split"])
    forced_train = {group for group, splits in group_splits.items() if "dev" in splits}
    eligible_eval = set(group_splits) - forced_train

    def choose_eval(name: str, available: set[str]) -> set[str]:
        chosen: set[str] = set()
        ranked = sorted(available, key=lambda group: _stable_score("allocate", name, group))
        for group in ranked:
            chosen.add(group)
            capacity = _group_capacity(chosen, assignments, patches, SPLIT_CONTRACTS[name].max_per_slide)
            contract = SPLIT_CONTRACTS[name]
            if (
                capacity["slides"] >= minimum_eval_wsi
                and capacity["clean"] >= contract.clean
                and capacity["risk"] >= contract.risk
                and capacity["total"] >= contract.clean + contract.risk
            ):
                return chosen
        raise ContractViolation(f"Unable to allocate a feasible patient-isolated {name} split")

    locked_groups = choose_eval("locked", eligible_eval)
    validation_groups = choose_eval("validation", eligible_eval - locked_groups)
    train_groups = set(group_splits) - locked_groups - validation_groups
    if not forced_train <= train_groups:
        raise ContractViolation("A previously used dev patient escaped the Train split")
    train_capacity = _group_capacity(train_groups, assignments, patches, SPLIT_CONTRACTS["train"].max_per_slide)
    train_contract = SPLIT_CONTRACTS["train"]
    if train_capacity["clean"] < train_contract.clean or train_capacity["risk"] < train_contract.risk or train_capacity["total"] < train_contract.clean + train_contract.risk:
        raise ContractViolation("Patient-isolated Train split lacks frozen quota capacity")

    split_by_group = {group: "train" for group in train_groups}
    split_by_group.update({group: "validation" for group in validation_groups})
    split_by_group.update({group: "locked" for group in locked_groups})
    result = []
    for row in assignments:
        result.append({**row, "assigned_split": split_by_group[row["patient_group_id"]]})
    assert_patient_isolation(result)
    return sorted(result, key=lambda row: (row["assigned_split"], row["patient_group_id"], row["slide_id"]))


def assert_patient_isolation(assignments: Iterable[dict[str, Any]]) -> None:
    patient_splits: dict[str, set[str]] = defaultdict(set)
    slide_splits: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        patient_splits[str(row["patient_group_id"])].add(str(row["assigned_split"]))
        slide_splits[str(row["slide_id"])].add(str(row["assigned_split"]))
    if any(len(splits) != 1 for splits in patient_splits.values()):
        raise PrivacyViolation("Patient group occurs in multiple splits")
    if any(len(splits) != 1 for splits in slide_splits.values()):
        raise PrivacyViolation("WSI occurs in multiple splits")


def _select_tier_round_robin(
    *,
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    selected_keys: set[tuple[str, str]],
    per_slide: Counter[str],
    count: int,
    max_per_slide: int,
    split: str,
    tier: str,
) -> None:
    by_slide: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if _patch_tier(row) == tier and (str(row["slide_id"]), str(row["patch_id"])) not in selected_keys:
            by_slide[str(row["slide_id"])].append(row)
    for slide, rows in by_slide.items():
        rows.sort(key=lambda row: _stable_score("patch", split, tier, slide, row["patch_id"]))
    slides = sorted(by_slide, key=lambda slide: _stable_score("slide", split, tier, slide))
    needed = count
    while needed:
        progressed = False
        for slide in slides:
            if needed == 0:
                break
            if per_slide[slide] >= max_per_slide or not by_slide[slide]:
                continue
            row = by_slide[slide].pop(0)
            key = (str(row["slide_id"]), str(row["patch_id"]))
            if key in selected_keys:
                continue
            selected_keys.add(key)
            per_slide[slide] += 1
            selected.append({**row, "assigned_split": split, "tier": tier})
            needed -= 1
            progressed = True
        if not progressed:
            raise ContractViolation(f"Unable to fill {split}/{tier} quota within per-WSI cap")


def select_split_candidates(
    patches: list[dict[str, Any]], assignments: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    split_by_slide = {str(row["slide_id"]): str(row["assigned_split"]) for row in assignments}
    result: dict[str, list[dict[str, Any]]] = {}
    for split, contract in SPLIT_CONTRACTS.items():
        pool = [row for row in patches if split_by_slide[str(row["slide_id"])] == split and _patch_tier(row)]
        selected: list[dict[str, Any]] = []
        selected_keys: set[tuple[str, str]] = set()
        per_slide: Counter[str] = Counter()
        split_slides = sorted({str(row["slide_id"]) for row in pool}, key=lambda slide: _stable_score("coverage", split, slide))
        # First guarantee union coverage of every assigned WSI without changing tier quotas.
        mandatory_clean = 0
        mandatory_risk = 0
        for slide in split_slides:
            clean = [row for row in pool if str(row["slide_id"]) == slide and _patch_tier(row) == "clean"]
            risk = [row for row in pool if str(row["slide_id"]) == slide and _patch_tier(row) == "risk"]
            if clean and mandatory_clean < contract.clean:
                row = min(clean, key=lambda item: _stable_score("mandatory", split, slide, item["patch_id"]))
                tier = "clean"
                mandatory_clean += 1
            elif risk and mandatory_risk < contract.risk:
                row = min(risk, key=lambda item: _stable_score("mandatory", split, slide, item["patch_id"]))
                tier = "risk"
                mandatory_risk += 1
            else:
                raise ContractViolation(f"Cannot cover assigned WSI {slide!r} within {split} tier quotas")
            selected_keys.add((str(row["slide_id"]), str(row["patch_id"])))
            per_slide[slide] += 1
            selected.append({**row, "assigned_split": split, "tier": tier})
        _select_tier_round_robin(
            candidates=pool,
            selected=selected,
            selected_keys=selected_keys,
            per_slide=per_slide,
            count=contract.risk - mandatory_risk,
            max_per_slide=contract.max_per_slide,
            split=split,
            tier="risk",
        )
        _select_tier_round_robin(
            candidates=pool,
            selected=selected,
            selected_keys=selected_keys,
            per_slide=per_slide,
            count=contract.clean - mandatory_clean,
            max_per_slide=contract.max_per_slide,
            split=split,
            tier="clean",
        )
        counts = Counter(row["tier"] for row in selected)
        if counts != Counter(clean=contract.clean, risk=contract.risk):
            raise ContractViolation(f"Frozen {split} tier counts were not met")
        if max(per_slide.values()) > contract.max_per_slide:
            raise ContractViolation(f"Frozen {split} per-WSI cap was exceeded")
        for row in selected:
            row["candidate_id"] = "cand_" + _stable_score(
                "candidate", split, row["patient_group_id"], row["slide_id"], row["patch_id"]
            )[:24]
        result[split] = sorted(selected, key=lambda row: row["candidate_id"])
    assert_candidate_isolation(result)
    return result


def assert_candidate_isolation(candidates: dict[str, list[dict[str, Any]]]) -> None:
    patient_splits: dict[str, set[str]] = defaultdict(set)
    slide_splits: dict[str, set[str]] = defaultdict(set)
    keys: set[tuple[str, str]] = set()
    candidate_ids: set[str] = set()
    for split, rows in candidates.items():
        for row in rows:
            if str(row.get("split", "")).casefold() in TEST_SPLIT_NAMES:
                raise PrivacyViolation("Test10 candidate entered a frozen split")
            patient_splits[str(row["patient_group_id"])].add(split)
            slide_splits[str(row["slide_id"])].add(split)
            key = (str(row["slide_id"]), str(row["patch_id"]))
            if key in keys or str(row["candidate_id"]) in candidate_ids:
                raise ContractViolation("Duplicate source or candidate ID")
            keys.add(key)
            candidate_ids.add(str(row["candidate_id"]))
    if any(len(value) != 1 for value in patient_splits.values()) or any(len(value) != 1 for value in slide_splits.values()):
        raise PrivacyViolation("Patient or WSI occurs across candidate splits")


def select_pilot_prefix(train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_rows = [row for row in train_rows if row["tier"] == "clean"]
    risk_rows = [row for row in train_rows if row["tier"] == "risk"]
    by_slide: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        by_slide[str(row["slide_id"])].append(row)
    slides = sorted(by_slide, key=lambda slide: _stable_score("pilot_coverage", slide))
    if len(slides) < 50:
        raise ContractViolation("Pilot cannot cover 50 Train WSI")
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_slide: Counter[str] = Counter()
    for slide in slides[:50]:
        row = min(by_slide[slide], key=lambda item: _stable_score("pilot_mandatory", item["candidate_id"]))
        selected.append(row)
        selected_ids.add(str(row["candidate_id"]))
        per_slide[slide] += 1

    def fill(pool: list[dict[str, Any]], total_for_tier: int, tier: str) -> None:
        already = sum(row["tier"] == tier for row in selected)
        ranked = sorted(pool, key=lambda row: _stable_score("pilot", tier, row["candidate_id"]))
        while already < total_for_tier:
            progressed = False
            for row in ranked:
                slide = str(row["slide_id"])
                if row["candidate_id"] in selected_ids or per_slide[slide] >= 4:
                    continue
                selected.append(row)
                selected_ids.add(str(row["candidate_id"]))
                per_slide[slide] += 1
                already += 1
                progressed = True
                if already == total_for_tier:
                    break
            if not progressed:
                raise ContractViolation(f"Pilot cannot fill {tier} quota within four-patch WSI cap")

    fill(risk_rows, 10, "risk")
    fill(clean_rows, 180, "clean")
    if len(selected) != 190 or len({row["slide_id"] for row in selected}) < 50 or max(per_slide.values()) > 4:
        raise ContractViolation("Pilot prefix contract failed")
    return sorted(selected, key=lambda row: _stable_score("pilot_order", row["candidate_id"]))


def control_slots(split: str, count: int) -> list[dict[str, Any]]:
    return [
        {
            "control_id": "control_" + _stable_score("control", split, index)[:24],
            "assigned_split": split,
            "tier": "deterministic_unassessable_control",
            "target": "<answer>{\"findings\":[]}</answer>",
            "generator_seed": SPLIT_SEED,
            "generator_index": index,
        }
        for index in range(count)
    ]


def write_private_split_artifacts(
    root: Path,
    *,
    assignments: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    pilot: list[dict[str, Any]],
) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    def immutable_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        expected = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
        if path.exists():
            if path.read_bytes() != expected:
                raise ContractViolation(f"Refusing to overwrite changed private split artifact: {path.name}")
            os.chmod(path, 0o600)
            return
        atomic_write_bytes(path, expected, mode=0o600)

    immutable_jsonl(root / "patient_group_assignment.jsonl", assignments)
    for split, rows in candidates.items():
        immutable_jsonl(root / f"{split}_candidates.jsonl", rows)
    immutable_jsonl(root / "pilot200_real_candidates.jsonl", pilot)
    immutable_jsonl(root / "pilot200_control_slots.jsonl", control_slots("train", 10))
    immutable_jsonl(root / "validation_control_slots.jsonl", control_slots("validation", 5))
    immutable_jsonl(root / "locked_control_slots.jsonl", control_slots("locked", 5))
