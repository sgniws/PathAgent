"""Deterministic evidence-contract evaluation for general WSI VQA.

The contract policy deliberately separates four questions:

* was contract-matching evidence visible;
* were the reported citations valid;
* did the cited subset support the candidate answer; and
* was exactly one official answer supported by both visible and cited evidence.

The module never reads report-derived labels or answer keys.  It maps public
answer choices to the public option ontology and evaluates only visible patch
descriptions plus registered WSI geometry.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


POLICY_VERSION = "contract_v1"
TERM_MATCHER_VERSION = "positive_term_negation_scope_v1"
OUTPUT_SCHEMA_VERSION = "pathagent_pathologist_assist_output_v1"

PRE_TERM_NEGATION = re.compile(
    r"(?:\bno(?:\s+(?:evidence|signs?)\s+of)?\b|\bnot\b|\bwithout\b|"
    r"\babsence\s+of\b|\black(?:ing)?\s+of\b|\bfree\s+of\b|"
    r"\bnegative\s+for\b|\bneither\b|\bnon(?:[-‐‑‒–—]\s*|\s+)|"
    r"(?:未见|未发现|没有|缺乏|不存在|不支持))",
    re.I,
)
ATTACHED_NEGATION = re.compile(
    r"(?:\bnon(?:[-‐‑‒–—]\s*)?|(?:非|无|未见|未发现|没有|缺乏|不存在|不支持))$",
    re.I,
)
POST_TERM_NEGATION = re.compile(
    r"^\s*(?:[:=\-]\s*)?"
    r"(?:(?:growth|pattern|activity|formation|invasion|expression|features?|finding)\b\s*){0,2}"
    r"(?:[:=\-]\s*)?"
    r"(?:(?:is|are|was|were|appears?|remains?)\s+)?"
    r"(?:absent|negative|lacking|not\s+(?:seen|observed|identified|present|evident|detected))\b|"
    r"^\s*性?\s*(?:(?:生长|结构|形成|活动|表达|征象|证据)\s*){0,2}"
    r"(?:为|呈)?\s*(?:未见|未发现|不存在|缺如|阴性)",
    re.I,
)
CLAUSE_BOUNDARY = re.compile(
    r"(?:[.!?;\n\r。！？；]|"
    r",\s*(?=(?:but|however|yet|whereas|although|while|with|showing|featuring)\b)|"
    r"，\s*(?=(?:但|但是|然而|不过|却|同时可见|并可见)))",
    re.I,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_contract_policy(
    contracts_path: str | Path, ontology_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    contracts = _read_json(contracts_path)
    ontology = _read_json(ontology_path)
    validate_contract_policy(contracts, ontology)
    return contracts, ontology


def validate_contract_policy(
    contracts: dict[str, Any], ontology: dict[str, Any]
) -> None:
    if contracts.get("schema_version") != "pathagent_vqa_evidence_contracts_v1":
        raise ValueError("Unexpected VQA evidence-contract schema")
    if contracts.get("term_matcher_version") != TERM_MATCHER_VERSION:
        raise ValueError("Evidence contract term matcher does not match contract_v1")
    if ontology.get("schema_version") != "pathagent_vqa_ontology_v1":
        raise ValueError("Unexpected VQA option-ontology schema")
    for question_type in ("structure", "diagnosis", "differentiation"):
        type_rule = (ontology.get("question_types") or {}).get(question_type)
        if not isinstance(type_rule, dict) or not type_rule.get("option_concepts"):
            raise ValueError(f"Ontology lacks option concepts for {question_type}")
    visual_contracts = contracts.get("visual_concept_contracts") or {}
    for concept_id, rule in visual_contracts.items():
        groups = rule.get("feature_groups")
        if not isinstance(groups, list) or not groups or any(not group for group in groups):
            raise ValueError(f"Evidence contract feature groups missing: {concept_id}")
        minimum = int(rule.get("minimum_feature_groups_per_patch", 0))
        if minimum < 1 or minimum > len(groups):
            raise ValueError(f"Invalid feature-group minimum: {concept_id}")
        if int(rule.get("minimum_clean_evidence_patches", 0)) < 1:
            raise ValueError(f"Invalid clean-patch minimum: {concept_id}")


def load_clean_description_manifest(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row.get("slide_id") or ""), str(row.get("patch_id") or ""))
        if not all(key):
            raise ValueError("Description manifest row lacks slide_id or patch_id")
        if key in rows:
            raise ValueError(f"Duplicate description manifest key: {key}")
        rows[key] = row
    return rows


def description_is_clean(row: dict[str, Any] | None) -> bool:
    if not row or row.get("status") != "success":
        return False
    audit = row.get("audit") or {}
    return bool(
        audit.get("contract_pass")
        and audit.get("safety_pass")
        and not audit.get("fallback_or_uninformative")
    )


def visible_text(description: str) -> str:
    match = re.search(
        r"(?ims)^\s*VISIBLE\s*:\s*(.*?)(?=^\s*ABSENT_OR_NOT_SEEN\s*:|\Z)",
        description,
    )
    return (match.group(1) if match else description).strip()


def _ascii_word_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0 and text[start - 1].isascii() and (
        text[start - 1].isalnum() or text[start - 1] == "_"
    ):
        start -= 1
    while end < len(text) and text[end].isascii() and (
        text[end].isalnum() or text[end] == "_"
    ):
        end += 1
    return start, end


def _local_clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    clause_start = 0
    for boundary in CLAUSE_BOUNDARY.finditer(text, 0, start):
        clause_start = boundary.end()
    next_boundary = CLAUSE_BOUNDARY.search(text, end)
    return clause_start, next_boundary.start() if next_boundary else len(text)


def _occurrence_is_negated(text: str, start: int, end: int) -> bool:
    word_start, word_end = _ascii_word_span(text, start, end)
    if ATTACHED_NEGATION.search(text[max(0, start - 16) : start]):
        return True
    clause_start, clause_end = _local_clause_bounds(text, word_start, word_end)
    prefix = text[max(clause_start, word_start - 96) : word_start]
    prefix = re.sub(r"\bnot\s+only\b", "", prefix, flags=re.I)
    if PRE_TERM_NEGATION.search(prefix):
        return True
    suffix = text[word_end : min(clause_end, word_end + 96)]
    return bool(POST_TERM_NEGATION.search(suffix))


def positive_term_match(text: str, term: str) -> bool:
    lowered = text.casefold()
    needle = term.casefold()
    if not needle:
        return False
    start = 0
    while True:
        index = lowered.find(needle, start)
        if index < 0:
            return False
        end = index + len(needle)
        if not _occurrence_is_negated(lowered, index, end):
            return True
        start = end


def matching_feature_groups(text: str, groups: list[list[str]]) -> list[int]:
    return [
        index
        for index, alternatives in enumerate(groups)
        if any(positive_term_match(text, term) for term in alternatives)
    ]


def _patch_rule_match(
    *,
    patch_id: str,
    rule: dict[str, Any],
    descriptions: dict[str, str],
    metadata: dict[str, dict[str, Any]],
    clean_patch_ids: set[str],
) -> dict[str, Any] | None:
    """Return deterministic per-patch feature matches for the review layers.

    Unlike the strict contract, the candidate layer needs to know whether even
    one allowed feature group is visible.  A patch still has to be cited,
    quality-audited, geometrically localizable, and at an allowed
    magnification.  PLIP similarity and model self-assessment never enter this
    function.
    """
    if patch_id not in clean_patch_ids:
        return None
    patch_meta = metadata.get(patch_id) or {}
    required_geometry = ("x_level0", "y_level0", "width_level0", "height_level0")
    if any(patch_meta.get(key) is None for key in required_geometry):
        return None
    allowed_magnifications = {
        int(value) for value in rule.get("allowed_magnifications", [5])
    }
    magnification = int(patch_meta.get("magnification", 5))
    if magnification not in allowed_magnifications:
        return None
    groups = list(rule.get("feature_groups") or [])
    matched_groups = matching_feature_groups(
        visible_text(str(descriptions.get(patch_id) or "")), groups
    )
    if not matched_groups:
        return None
    minimum_groups = int(rule.get("minimum_feature_groups_per_patch", len(groups)))
    return {
        "patch_id": patch_id,
        "x_level0": int(patch_meta["x_level0"]),
        "y_level0": int(patch_meta["y_level0"]),
        "width_level0": int(patch_meta["width_level0"]),
        "height_level0": int(patch_meta["height_level0"]),
        "magnification": magnification,
        "parent_patch_id": patch_meta.get("parent_patch_id"),
        "matched_feature_groups": matched_groups,
        "direct_support": len(matched_groups) >= minimum_groups,
    }


def _rectangles_overlap(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_right = int(first["x_level0"]) + int(first["width_level0"])
    first_bottom = int(first["y_level0"]) + int(first["height_level0"])
    second_right = int(second["x_level0"]) + int(second["width_level0"])
    second_bottom = int(second["y_level0"]) + int(second["height_level0"])
    return not (
        first_right <= int(second["x_level0"])
        or second_right <= int(first["x_level0"])
        or first_bottom <= int(second["y_level0"])
        or second_bottom <= int(first["y_level0"])
    )


def _minimum_patches(
    rule: dict[str, Any], defaults: dict[str, Any], evidence_tier: str
) -> int:
    if evidence_tier == "limited":
        return int(defaults.get("limited_minimum_clean_evidence_patches", 1))
    return max(
        int(rule.get("minimum_clean_evidence_patches", 1)),
        int(defaults.get("strict_minimum_clean_evidence_patches", 1)),
    )


def evaluate_visual_contract(
    concept_id: str,
    rule: dict[str, Any],
    patch_ids: Iterable[str],
    descriptions: dict[str, str],
    metadata: dict[str, dict[str, Any]],
    clean_patch_ids: set[str],
    defaults: dict[str, Any],
    evidence_tier: str,
    requires_zoom_confirmation: bool = False,
) -> dict[str, Any]:
    groups = list(rule.get("feature_groups") or [])
    minimum_groups = int(rule.get("minimum_feature_groups_per_patch", len(groups)))
    allowed_magnifications = {int(value) for value in rule.get("allowed_magnifications", [5])}
    ordered_patch_ids = list(dict.fromkeys(str(value) for value in patch_ids))
    candidates: list[dict[str, Any]] = []
    rejected_unclean: list[str] = []
    rejected_geometry: list[str] = []
    for patch_id in ordered_patch_ids:
        if patch_id not in clean_patch_ids:
            rejected_unclean.append(patch_id)
            continue
        patch_meta = metadata.get(patch_id) or {}
        if int(patch_meta.get("magnification", 5)) not in allowed_magnifications:
            continue
        required_geometry = ("x_level0", "y_level0", "width_level0", "height_level0")
        if any(patch_meta.get(key) is None for key in required_geometry):
            rejected_geometry.append(patch_id)
            continue
        text = visible_text(str(descriptions.get(patch_id) or ""))
        matched_groups = matching_feature_groups(text, groups)
        if len(matched_groups) < minimum_groups:
            continue
        candidates.append(
            {
                "patch_id": patch_id,
                "matched_groups": matched_groups,
                "match_score": len(matched_groups),
                **{key: int(patch_meta[key]) for key in required_geometry},
            }
        )
    candidates.sort(key=lambda item: (-item["match_score"], item["patch_id"]))
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if rule.get("require_nonoverlap", True) and any(
            _rectangles_overlap(candidate, existing) for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= int(defaults.get("maximum_evidence_patches", 4)):
            break
    represented = sorted(
        {group for candidate in selected for group in candidate["matched_groups"]}
    )
    required_groups = [int(value) for value in rule.get("required_anywhere_groups", [])]
    missing_groups = [group for group in required_groups if group not in represented]
    minimum_patches = _minimum_patches(rule, defaults, evidence_tier)
    zoom_patch_ids = [
        patch_id
        for patch_id in ordered_patch_ids
        if int((metadata.get(patch_id) or {}).get("magnification", 5)) > 5
    ]
    reasons = []
    if len(selected) < minimum_patches:
        reasons.append(f"clean_nonoverlapping_patch_count_lt_{minimum_patches}")
    if missing_groups:
        reasons.append("required_visual_feature_group_missing")
    if requires_zoom_confirmation and not zoom_patch_ids:
        reasons.append("zoom_confirmation_missing")
    patch_fraction = min(len(selected) / max(1, minimum_patches), 1.0)
    group_fraction = (
        sum(group in represented for group in required_groups) / len(required_groups)
        if required_groups
        else 1.0
    )
    zoom_fraction = 1.0 if not requires_zoom_confirmation or zoom_patch_ids else 0.0
    support_score = round((patch_fraction + group_fraction + zoom_fraction) / 3.0, 6)
    return {
        "visual_support_concept": concept_id,
        "evidence_contract_id": rule.get("contract_id"),
        "eligible": not reasons,
        "support_score": support_score,
        "minimum_clean_evidence_patches": minimum_patches,
        "selected_patch_ids": [item["patch_id"] for item in selected],
        "selected_patch_matches": [
            {"patch_id": item["patch_id"], "matched_groups": item["matched_groups"]}
            for item in selected
        ],
        "represented_feature_groups": represented,
        "missing_required_groups": missing_groups,
        "requires_zoom_confirmation": requires_zoom_confirmation,
        "zoom_confirmation_patch_ids": zoom_patch_ids,
        "rejected_unclean_patch_ids": rejected_unclean,
        "rejected_missing_geometry_patch_ids": rejected_geometry,
        "rejection_reasons": reasons,
    }


def _concept_display(concept: dict[str, Any]) -> str:
    return str(concept.get("display_zh") or concept.get("label_zh") or concept.get("label") or "")


def option_concepts_for_choices(
    ontology: dict[str, Any], question_type: str, choices: list[str]
) -> dict[str, str]:
    allowed = set(
        ((ontology.get("question_types") or {}).get(question_type) or {}).get(
            "option_concepts", []
        )
    )
    displays: dict[str, str] = {}
    for concept_id in allowed:
        display = _concept_display((ontology.get("concepts") or {}).get(concept_id) or {})
        if display:
            if display in displays:
                raise ValueError(f"Duplicate ontology display label: {display}")
            displays[display] = concept_id
    missing = [choice for choice in choices if str(choice) not in displays]
    if missing:
        raise ValueError(f"Official choices are absent from the public ontology: {missing}")
    return {str(choice): displays[str(choice)] for choice in choices}


def _support_relation(
    contracts: dict[str, Any], question_type: str, option_concept: str
) -> tuple[str | None, bool]:
    if question_type == "structure":
        return option_concept, False
    relation = (contracts.get("target_support_contracts") or {}).get(option_concept) or {}
    return relation.get("visual_support_concept"), bool(
        relation.get("requires_zoom_confirmation", False)
    )


def _normalize_ranked_differential(
    values: Iterable[Any] | None, choices: list[str], candidate_answer: str
) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        choice = str(value)
        if choice not in choices or choice == candidate_answer or choice in normalized:
            continue
        normalized.append(choice)
    return normalized[: max(0, len(choices) - 1)]


def _review_layer_evidence(
    *,
    contracts: dict[str, Any],
    question_type: str,
    choices: list[str],
    option_concepts: dict[str, str],
    candidate_answer: str,
    ranked_differential: list[str],
    valid_refs: list[str],
    citation_valid: bool,
    descriptions: dict[str, str],
    metadata: dict[str, dict[str, Any]],
    clean_patch_ids: set[str],
    option_evaluations: dict[str, Any],
    strict_evidence_sufficient: bool,
) -> dict[str, Any]:
    visual_contracts = contracts.get("visual_concept_contracts") or {}
    cited_patches = []
    matches_by_patch: dict[str, dict[str, Any]] = {}
    for patch_id in valid_refs:
        patch_meta = metadata.get(patch_id) or {}
        required_geometry = ("x_level0", "y_level0", "width_level0", "height_level0")
        localizable = all(patch_meta.get(key) is not None for key in required_geometry)
        cited_patches.append(
            {
                "patch_id": patch_id,
                "citation_valid": True,
                "quality_qualified": patch_id in clean_patch_ids,
                "localizable": localizable,
                "x_level0": int(patch_meta["x_level0"]) if localizable else None,
                "y_level0": int(patch_meta["y_level0"]) if localizable else None,
                "width_level0": int(patch_meta["width_level0"])
                if localizable
                else None,
                "height_level0": int(patch_meta["height_level0"])
                if localizable
                else None,
                "magnification": int(patch_meta.get("magnification", 5)),
                "parent_patch_id": patch_meta.get("parent_patch_id"),
            }
        )
        for choice in choices:
            option_concept = option_concepts[choice]
            support_id, requires_zoom = _support_relation(
                contracts, question_type, option_concept
            )
            if support_id is None:
                continue
            match = _patch_rule_match(
                patch_id=patch_id,
                rule=visual_contracts[support_id],
                descriptions=descriptions,
                metadata=metadata,
                clean_patch_ids=clean_patch_ids,
            )
            if match is None:
                continue
            patch_record = matches_by_patch.setdefault(
                patch_id,
                {
                    key: match[key]
                    for key in (
                        "patch_id",
                        "x_level0",
                        "y_level0",
                        "width_level0",
                        "height_level0",
                        "magnification",
                        "parent_patch_id",
                    )
                }
                | {"matched_supports": []},
            )
            patch_record["matched_supports"].append(
                {
                    "choice": choice,
                    "option_concept": option_concept,
                    "visual_support_concept": support_id,
                    "evidence_contract_id": visual_contracts[support_id].get(
                        "contract_id"
                    ),
                    "matched_feature_groups": match["matched_feature_groups"],
                    "direct_support": bool(match["direct_support"]),
                    "requires_zoom_confirmation": requires_zoom,
                }
            )

    candidate_records = list(matches_by_patch.values()) if citation_valid else []
    support_targets = {candidate_answer, *ranked_differential}

    def supporting_matches(record: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            item
            for item in record["matched_supports"]
            if item["direct_support"] and item["choice"] in support_targets
        ]

    def conflicting_matches(record: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            item
            for item in record["matched_supports"]
            if item["direct_support"] and item["choice"] != candidate_answer
        ]

    supporting_evidence = [
        {**record, "matched_supports": supporting_matches(record)}
        for record in candidate_records
        if supporting_matches(record)
    ]
    opposing_or_conflicting_evidence = [
        {**record, "matched_supports": conflicting_matches(record)}
        for record in candidate_records
        if conflicting_matches(record)
    ]
    candidate_evidence_found = bool(candidate_records)
    ready_for_pathologist_review = bool(
        candidate_evidence_found and supporting_evidence
    )
    if strict_evidence_sufficient:
        # The strict layer is a logical strengthening of the two review layers.
        candidate_evidence_found = True
        ready_for_pathologist_review = True

    missing_evidence: list[str] = []
    if not citation_valid:
        missing_evidence.append("valid_visible_citation_required")
    if not candidate_evidence_found:
        missing_evidence.append(
            "clean_localizable_cited_patch_matching_an_allowed_feature_group_required"
        )
    if not ready_for_pathologist_review:
        missing_evidence.append(
            "direct_support_for_provisional_or_ranked_differential_required"
        )
    candidate_eval = option_evaluations.get(candidate_answer) or {}
    for scope in ("visible", "cited"):
        for reason in (candidate_eval.get(scope) or {}).get("rejection_reasons", []):
            code = f"{scope}:{reason}"
            if code not in missing_evidence:
                missing_evidence.append(code)
    if not strict_evidence_sufficient:
        missing_evidence.append("strict_contract_not_satisfied")

    return {
        "provisional_recommendation": candidate_answer,
        "ranked_differential": ranked_differential,
        "cited_patches": cited_patches,
        "supporting_evidence": supporting_evidence,
        "opposing_or_conflicting_evidence": opposing_or_conflicting_evidence,
        "missing_evidence": missing_evidence,
        "candidate_evidence_found": candidate_evidence_found,
        "ready_for_pathologist_review": ready_for_pathologist_review,
        "strict_evidence_sufficient": strict_evidence_sufficient,
        "review_required": True,
        "review_reason": "research_output_requires_pathologist_review",
        "contract_version": contracts.get("contract_version"),
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
    }


def evaluate_contract_state(
    *,
    contracts: dict[str, Any],
    ontology: dict[str, Any],
    question_type: str,
    choices: list[str],
    candidate_answer: str,
    evidence_tier: str,
    visible_patch_ids: list[str],
    reported_evidence_refs: list[Any],
    descriptions: dict[str, str],
    metadata: dict[str, dict[str, Any]],
    clean_patch_ids: set[str],
    ranked_differential: list[str] | None = None,
) -> dict[str, Any]:
    option_concepts = option_concepts_for_choices(ontology, question_type, choices)
    visible_patch_ids = list(dict.fromkeys(str(value) for value in visible_patch_ids))
    visible = set(visible_patch_ids)
    raw_refs = reported_evidence_refs if isinstance(reported_evidence_refs, list) else []
    valid_refs: list[str] = []
    invalid_refs: list[Any] = []
    for ref in raw_refs:
        if isinstance(ref, str) and ref in visible:
            if ref not in valid_refs:
                valid_refs.append(ref)
        else:
            invalid_refs.append(ref)
    citation_valid = bool(valid_refs) and not invalid_refs
    defaults = contracts.get("defaults") or {}
    visual_contracts = contracts.get("visual_concept_contracts") or {}
    evaluations: dict[str, Any] = {}
    for choice in choices:
        option_concept = option_concepts[str(choice)]
        support_id, requires_zoom = _support_relation(
            contracts, question_type, option_concept
        )
        if support_id is None:
            visible_eval = cited_eval = {
                "visual_support_concept": None,
                "evidence_contract_id": None,
                "eligible": False,
                "support_score": 0.0,
                "selected_patch_ids": [],
                "rejection_reasons": ["no_frozen_visual_support_mapping"],
            }
        else:
            rule = visual_contracts[support_id]
            visible_eval = evaluate_visual_contract(
                support_id,
                rule,
                visible_patch_ids,
                descriptions,
                metadata,
                clean_patch_ids,
                defaults,
                evidence_tier,
                requires_zoom_confirmation=requires_zoom,
            )
            cited_eval = evaluate_visual_contract(
                support_id,
                rule,
                valid_refs,
                descriptions,
                metadata,
                clean_patch_ids,
                defaults,
                evidence_tier,
                requires_zoom_confirmation=requires_zoom,
            )
        evaluations[str(choice)] = {
            "option_concept": option_concept,
            "visible": visible_eval,
            "cited": cited_eval,
        }
    visible_passing = [
        choice for choice in choices if evaluations[str(choice)]["visible"]["eligible"]
    ]
    cited_passing = [
        choice for choice in choices if evaluations[str(choice)]["cited"]["eligible"]
    ]
    unique_visible = visible_passing[0] if len(visible_passing) == 1 else None
    unique_cited = cited_passing[0] if len(cited_passing) == 1 else None
    candidate_eval = evaluations.get(candidate_answer) or {
        "visible": {"support_score": 0.0, "eligible": False},
        "cited": {"support_score": 0.0, "eligible": False},
    }
    citation_supports_answer = bool(candidate_eval["cited"].get("eligible"))
    evidence_sufficient = bool(
        citation_valid
        and unique_visible == candidate_answer
        and unique_cited == candidate_answer
    )
    normalized_differential = _normalize_ranked_differential(
        ranked_differential, choices, candidate_answer
    )
    review_layers = _review_layer_evidence(
        contracts=contracts,
        question_type=question_type,
        choices=choices,
        option_concepts=option_concepts,
        candidate_answer=candidate_answer,
        ranked_differential=normalized_differential,
        valid_refs=valid_refs,
        citation_valid=citation_valid,
        descriptions=descriptions,
        metadata=metadata,
        clean_patch_ids=clean_patch_ids,
        option_evaluations=evaluations,
        strict_evidence_sufficient=evidence_sufficient,
    )
    return {
        "policy_version": POLICY_VERSION,
        "contract_version": contracts.get("contract_version"),
        "term_matcher_version": contracts.get("term_matcher_version"),
        "question_type": question_type,
        "evidence_tier": evidence_tier,
        "candidate_answer": candidate_answer,
        "reported_evidence_refs": raw_refs,
        "valid_evidence_refs": valid_refs,
        "invalid_evidence_refs": invalid_refs,
        "citation_valid": citation_valid,
        "evidence_found": bool(visible_passing),
        "citation_supports_answer": citation_supports_answer,
        "evidence_sufficient": evidence_sufficient,
        **review_layers,
        "visible_passing_answers": visible_passing,
        "cited_passing_answers": cited_passing,
        "unique_visible_answer": unique_visible,
        "unique_cited_answer": unique_cited,
        "candidate_support_score": float(candidate_eval["visible"].get("support_score", 0.0)),
        "candidate_citation_support_score": float(candidate_eval["cited"].get("support_score", 0.0)),
        "option_evaluations": evaluations,
    }


def candidate_record(
    decision: dict[str, Any], verification: dict[str, Any], attempt: int, stability_count: int
) -> dict[str, Any]:
    return {
        "answer": str(decision.get("benchmark_answer") or ""),
        "attempt": int(attempt),
        "stability_count": int(stability_count),
        "support_score": float(verification.get("candidate_support_score", 0.0)),
        "citation_support_score": float(
            verification.get("candidate_citation_support_score", 0.0)
        ),
        "evidence_refs": list(verification.get("valid_evidence_refs") or []),
        "evidence_summary": str(decision.get("evidence_summary") or ""),
        "ranked_differential": list(verification.get("ranked_differential") or []),
        "executor_missing_evidence": decision.get("missing_evidence"),
        "unsupported_answer_reason": str(
            decision.get("unsupported_answer_reason") or ""
        ),
        "verification": verification,
    }


def choose_better_candidate(
    current_best: dict[str, Any] | None, challenger: dict[str, Any]
) -> dict[str, Any]:
    if current_best is None:
        return challenger
    current_key = (
        float(current_best.get("support_score", 0.0)),
        int(current_best.get("stability_count", 0)),
        -int(current_best.get("attempt", 0)),
    )
    challenger_key = (
        float(challenger.get("support_score", 0.0)),
        int(challenger.get("stability_count", 0)),
        -int(challenger.get("attempt", 0)),
    )
    return challenger if challenger_key > current_key else current_best
