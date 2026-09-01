from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .common import ContractViolation, canonical_json


ANSWER_RE = re.compile(r"\A<answer>(?P<body>.*)</answer>\Z", re.DOTALL)
LEADING_THINK_RE = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("organ_or_site", re.compile(r"\b(?:pancrea(?:s|tic)|liver|lung|breast|colon|gastric|stomach|kidney|prostate|ovary|uter(?:us|ine)|organ|anatomic(?:al)?\s+site)\b", re.I)),
    ("diagnosis", re.compile(r"\b(?:diagnos(?:is|tic)|carcinoma|adenocarcinoma|sarcoma|lymphoma|melanoma|neoplasm|tumou?r|malignan(?:t|cy)|benign)\b", re.I)),
    ("grade_or_lineage", re.compile(r"\b(?:grade[sd]?|grading|lineage|well[- ]differentiated|poorly[- ]differentiated)\b", re.I)),
    ("ihc_or_molecular", re.compile(r"\b(?:immunohistochem(?:istry|ical)?|IHC|molecular|mutation|KRAS|TP53|SMAD4|CDX2|CK7|CK20)\b", re.I)),
    ("treatment_or_prognosis", re.compile(r"\b(?:treat(?:ment|ed)|therap(?:y|eutic)|chemotherapy|radiotherapy|prognos(?:is|tic)|survival)\b", re.I)),
)


@dataclass(frozen=True)
class FindingsResult:
    valid: bool
    findings: tuple[str, ...]
    error: str | None
    parsed: Any = None


def validate_findings_object(value: Any) -> FindingsResult:
    if not isinstance(value, dict):
        return FindingsResult(False, (), "result_not_object", value)
    if set(value) != {"findings"}:
        return FindingsResult(False, (), "unexpected_keys", value)
    findings = value["findings"]
    if not isinstance(findings, list):
        return FindingsResult(False, (), "findings_not_array", value)
    if len(findings) > 5:
        return FindingsResult(False, (), "too_many_findings", value)
    if any(not isinstance(item, str) for item in findings):
        return FindingsResult(False, (), "finding_not_string", value)
    if any(not item.strip() for item in findings):
        return FindingsResult(False, (), "finding_empty", value)
    return FindingsResult(True, tuple(findings), None, value)


def parse_bare_findings(raw: str) -> FindingsResult:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return FindingsResult(False, (), "json_parse_failed")
    return validate_findings_object(value)


def strip_one_leading_think(raw: str) -> str:
    return LEADING_THINK_RE.sub("", raw, count=1)


def parse_wrapped_findings(raw: str) -> FindingsResult:
    visible = strip_one_leading_think(raw)
    match = ANSWER_RE.fullmatch(visible)
    if not match:
        return FindingsResult(False, (), "answer_boundary_not_exact")
    return parse_bare_findings(match.group("body"))


def canonical_target(findings: Iterable[str]) -> str:
    result = validate_findings_object({"findings": list(findings)})
    if not result.valid:
        raise ContractViolation(f"Invalid findings target: {result.error}")
    return f"<answer>{canonical_json({'findings': list(result.findings)})}</answer>"


def scan_forbidden(texts: Iterable[str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for index, text in enumerate(texts):
        for category, pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                hits.append({"category": category, "finding_index": str(index)})
    return hits


def validate_teacher_candidate(raw: str) -> tuple[FindingsResult, list[dict[str, str]]]:
    result = parse_bare_findings(raw)
    hits = scan_forbidden(result.findings if result.valid else [raw])
    return result, hits
