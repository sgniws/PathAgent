from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common import (
    ContractViolation,
    atomic_write_bytes,
    atomic_write_text,
    canonical_json,
    sha256_file,
)
from .contracts import SPLIT_SEED
from .schema import canonical_target


S3_REVIEW_FIELDS = [
    "presentation_id",
    "image_relpath",
    "candidate_raw",
    "candidate_findings_json",
    "claim_labels_json",
    "forbidden_inference",
    "reviewer_notes",
]

S3_COMPLETED_FIELDS = [
    "presentation_id",
    "claim_labels_json",
    "forbidden_inference",
    "reviewer_notes",
]


def _score(*parts: Any) -> str:
    return hashlib.sha256(
        "\x1f".join([str(SPLIT_SEED), *(str(part) for part in parts)]).encode("utf-8")
    ).hexdigest()


def pilot_sample_id(candidate_id: str) -> str:
    if not candidate_id.startswith("cand_"):
        raise ContractViolation("S3 source must be a frozen anonymous candidate")
    return "pilot_" + _score("s3_pilot", candidate_id)[:24]


def assess_candidate(
    observation: dict[str, Any],
    *,
    natural_qc_risk: bool,
    inherited_source_review_override: bool = False,
) -> dict[str, Any]:
    """Apply the frozen Luna-generation/Gemini-audit acceptance boundary."""
    schema_valid = observation.get("strict_schema_valid") is True
    forbidden_hits = list(observation.get("raw_forbidden_hits") or [])
    audit = observation.get("audit") if isinstance(observation.get("audit"), dict) else {}
    audit_valid = audit.get("strict_schema_valid") is True
    statuses = list(audit.get("statuses") or [])
    audit_forbidden = audit.get("forbidden_inference") is True

    if not schema_valid:
        return {"status": "rejected", "risk_tier": "high", "reason": "luna_schema_failure"}
    if forbidden_hits:
        return {"status": "rejected", "risk_tier": "high", "reason": "luna_forbidden_content"}
    if not audit_valid:
        return {"status": "rejected", "risk_tier": "high", "reason": "gemini_audit_schema_failure"}
    if len(statuses) != len(observation.get("findings") or []):
        return {"status": "rejected", "risk_tier": "high", "reason": "gemini_claim_count_mismatch"}

    disagreement = audit_forbidden or any(status != "supported" for status in statuses)
    if disagreement and not inherited_source_review_override:
        return {
            "status": "rejected",
            "risk_tier": "high",
            "reason": "gemini_rejected_luna_candidate",
        }

    requires_human = natural_qc_risk
    if requires_human:
        status = "waiting_human_review"
        risk_tier = "medium"
        reason = "natural_qc_risk_requires_positive_human_review"
    else:
        status = "accepted_automatic"
        risk_tier = "low"
        reason = "luna_candidate_passed_schema_safety_and_gemini_audit"
    if inherited_source_review_override and disagreement:
        reason = "inherited_after_explicit_s2_source_review_pending_s3_sample_gate"
    return {
        "status": status,
        "risk_tier": risk_tier,
        "reason": reason,
        "requires_human_review": requires_human,
        "audit_disagreement": disagreement,
        "target": canonical_target(observation.get("findings") or []),
    }


def summarize_automated_pilot(
    real_records: Iterable[dict[str, Any]],
    control_records: Iterable[dict[str, Any]],
    *,
    first_attempt_schema_results: Iterable[bool],
    replacement_count: int,
) -> dict[str, Any]:
    real = [dict(row) for row in real_records]
    controls = [dict(row) for row in control_records]
    first_schema = list(first_attempt_schema_results)
    if len(real) != 190 or len(controls) != 10 or len(first_schema) != 200:
        raise ContractViolation("S3 automated pilot summary requires 190 real, 10 controls, and 200 first attempts")
    if not 0 <= replacement_count <= 20:
        raise ContractViolation("S3 replacement count exceeded the frozen ceiling")

    sample_ids = [str(row.get("sample_id", "")) for row in real]
    image_hashes = [str(row.get("image_sha256", "")) for row in real]
    statuses = [
        status
        for row in real
        for status in row.get("candidate", {}).get("audit", {}).get("statuses", [])
    ]
    supported = sum(status == "supported" for status in statuses)
    precision = supported / len(statuses) if statuses else None
    raw_forbidden = sum(bool(row.get("candidate", {}).get("raw_forbidden_hits")) for row in real)
    control_exact = sum(row.get("exact_empty") is True for row in controls)
    first_schema_pass = sum(bool(value) for value in first_schema)
    data_integrity = (
        len(sample_ids) == len(set(sample_ids))
        and all(sample_ids)
        and len(image_hashes) == len(set(image_hashes))
        and all(image_hashes)
        and all(row.get("assigned_split") == "train" for row in real)
    )
    automatic_gate = {
        "data_integrity": data_integrity,
        "first_strict_schema": first_schema_pass >= 198,
        "final_schema": all(row.get("candidate", {}).get("strict_schema_valid") is True for row in real),
        "accepted_forbidden_content": raw_forbidden == 0,
        "cross_model_claim_precision": precision is not None and precision >= 0.95,
        "blank_exact_empty": control_exact == 10,
        "replacement_rate": replacement_count <= 20,
    }
    return {
        "schema_version": "patho_lora_s3_automated_summary_v1",
        "status": "passed_waiting_human_review" if all(automatic_gate.values()) else "failed_automatic_gate",
        "pilot_records": 200,
        "real_records": len(real),
        "control_records": len(controls),
        "first_strict_schema_pass": first_schema_pass,
        "first_strict_schema_total": len(first_schema),
        "final_strict_schema_pass": sum(row.get("candidate", {}).get("strict_schema_valid") is True for row in real) + control_exact,
        "accepted_raw_forbidden_count": raw_forbidden,
        "cross_audit_supported_claims": supported,
        "cross_audit_total_claims": len(statuses),
        "cross_model_claim_precision": precision,
        "cross_audit_forbidden_flags": sum(
            row.get("candidate", {}).get("audit", {}).get("forbidden_inference") is True
            for row in real
        ),
        "blank_exact_empty_pass": control_exact,
        "blank_exact_empty_total": 10,
        "replacement_count": replacement_count,
        "replacement_rate": replacement_count / 200,
        "automatic_gates": automatic_gate,
        "human_gate_pending": True,
    }


def select_human_review_records(
    records: Iterable[dict[str, Any]], *, count: int = 20
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in records]
    if len(rows) != 190 or count != 20:
        raise ContractViolation("S3 human review selection requires 20 of 190 real records")
    risk_rows = [row for row in rows if row.get("tier") == "risk"]
    if len(risk_rows) != 10:
        raise ContractViolation("S3 human review must contain the frozen ten natural-risk records")

    def disagreement(row: dict[str, Any]) -> bool:
        candidate = row.get("candidate", {})
        audit = candidate.get("audit", {}) if isinstance(candidate.get("audit"), dict) else {}
        return bool(audit.get("forbidden_inference")) or any(
            status != "supported" for status in audit.get("statuses", [])
        )

    def forbidden_flag(row: dict[str, Any]) -> bool:
        return row.get("candidate", {}).get("audit", {}).get("forbidden_inference") is True

    def diversity_order(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            buckets[len(row.get("candidate", {}).get("findings") or [])].append(row)
        for bucket in buckets.values():
            bucket.sort(key=lambda row: _score("s3_human_review", row.get("sample_id")))
        ordered: list[dict[str, Any]] = []
        while any(buckets.values()):
            for finding_count in sorted(buckets):
                if buckets[finding_count]:
                    ordered.append(buckets[finding_count].pop(0))
        return ordered

    selected = diversity_order(risk_rows)
    selected_ids = {str(row["sample_id"]) for row in selected}

    def add(candidates: Iterable[dict[str, Any]], limit: int) -> None:
        if limit <= 0:
            return
        added = 0
        for row in diversity_order(candidates):
            sample_id = str(row["sample_id"])
            if sample_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(sample_id)
            added += 1
            if added == limit or len(selected) == count:
                return

    clean = [row for row in rows if row.get("tier") == "clean"]
    flagged = [row for row in clean if forbidden_flag(row)]
    if len(flagged) > 4:
        raise ContractViolation("S3 human queue cannot cover all automated forbidden flags")
    add(flagged, len(flagged))
    add([row for row in clean if disagreement(row)], max(0, 4 - len(flagged)))
    add([row for row in clean if bool(row.get("replacement_index"))], 3)
    add(clean, count - len(selected))
    if len({str(row.get("sample_id")) for row in selected}) != count:
        raise ContractViolation("S3 human review selection is not unique")
    if sum(row.get("tier") == "risk" for row in selected) != 10:
        raise ContractViolation("S3 human review omitted a natural-risk record")
    return selected


def _presentation_id(source_review_id: str, repeat_index: int) -> str:
    return "s3review_" + _score("s3_review_presentation", source_review_id, repeat_index)[:20]


def build_single_blind_review(
    records: Iterable[dict[str, Any]], *, repeat_fraction: float = 0.10, served_namespace: str = "pilot200"
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    source = [dict(row) for row in records]
    if len(source) != 20 or repeat_fraction != 0.10:
        raise ContractViolation("S3 review requires 20 unique images and exactly 10% repeats")
    if served_namespace not in {"pilot200", "pilot200_v2"}:
        raise ContractViolation("S3 served review namespace is not approved")
    review_ids = [str(row.get("review_id", "")) for row in source]
    if len(review_ids) != len(set(review_ids)) or any(not value for value in review_ids):
        raise ContractViolation("S3 source review IDs are missing or duplicated")
    rng = random.Random(SPLIT_SEED)

    def one(row: dict[str, Any], repeat_index: int) -> tuple[dict[str, str], dict[str, str]]:
        review_id = str(row["review_id"])
        served = f"images/{served_namespace}/{review_id}.png"
        presentation_id = _presentation_id(review_id, repeat_index)
        public = {
            "presentation_id": presentation_id,
            "image_relpath": served,
            "candidate_raw": str(row["candidate_raw"]),
            "candidate_findings_json": canonical_json(list(row["candidate_findings"])),
            "claim_labels_json": "",
            "forbidden_inference": "",
            "reviewer_notes": "",
        }
        private = {
            "presentation_id": presentation_id,
            "source_review_id": review_id,
            "is_repeat": "true" if repeat_index else "false",
            "source_image_relpath": str(row["source_image_relpath"]),
            "served_image_relpath": served,
        }
        return public, private

    pairs = [one(row, 0) for row in source]
    for row in rng.sample(source, 2):
        pairs.append(one(row, 1))
    rng.shuffle(pairs)
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def export_single_review_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=S3_REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue(), mode=0o600)


def export_single_review_ui(path: Path, rows: Iterable[dict[str, str]]) -> None:
    queue = [dict(row) for row in rows]
    if len(queue) != 22:
        raise ContractViolation("S3 review UI requires exactly 22 anonymous presentations")
    embedded = json.dumps(queue, ensure_ascii=False, separators=(",", ":"))
    embedded = embedded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    # Keep JavaScript escape sequences such as ``\n`` literal in the generated
    # document.  A normal Python string would turn them into physical newlines
    # and make the regular-expression and CSV string literals invalid.
    document = r"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><base href="../">
<title>PATHO-LORA-SFT-01 S3 blind review</title>
<style>
body{font:15px system-ui;max-width:1180px;margin:1.5rem auto;padding:0 1rem;line-height:1.45;background:#fafafa;color:#182026}
.warning{color:#8b1a1a}.grid{display:grid;grid-template-columns:minmax(360px,1fr) minmax(320px,1fr);gap:1rem}
img{width:100%;max-height:760px;object-fit:contain;background:#eee}.card{background:white;border:1px solid #ccd3d8;border-radius:8px;padding:1rem}
pre{white-space:pre-wrap;overflow-wrap:anywhere}.claim{display:grid;grid-template-columns:1fr 190px;gap:.5rem;margin:.6rem 0}
button,select,textarea{font:inherit;padding:.45rem}textarea{width:100%;box-sizing:border-box}.toolbar{display:flex;gap:.6rem;align-items:center;margin:1rem 0;flex-wrap:wrap}
@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style>
<h1>S3 blind morphology review</h1>
<p class="warning">Judge direct visible morphology only. Do not infer organ, diagnosis, grade, lineage, IHC, molecular result, treatment, prognosis, or clinical meaning.</p>
<p>Label every claim as supported, unsupported, or not_assessable and mark forbidden inference separately. Model identity, automatic scores, risk tier, replacement status, and repeat status are hidden.</p>
<div class="toolbar"><button id="prev">Previous</button><strong id="progress"></strong><button id="next">Save & next</button><button id="download">Download completed CSV</button></div>
<div class="grid"><div class="card"><img id="image" alt="anonymous H&amp;E patch"></div><div class="card" id="form"></div></div>
<script>
const queue=__QUEUE__,storageKey='patho-lora-sft-01-s3-review-v1';
const saved=JSON.parse(localStorage.getItem(storageKey)||'{}');let index=0;
const statuses=['','supported','unsupported','not_assessable'];
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function render(){const row=queue[index],state=saved[row.presentation_id]||{},findings=JSON.parse(row.candidate_findings_json);
 document.getElementById('progress').textContent=`${index+1} / ${queue.length} — ${Object.keys(saved).length} saved`;
 document.getElementById('image').src=row.image_relpath;
 const claims=findings.map((finding,i)=>`<div class="claim"><span>${i+1}. ${esc(finding)}</span><select data-label="${i}">${statuses.map(x=>`<option value="${x}">${x||'select status'}</option>`).join('')}</select></div>`).join('');
 document.getElementById('form').innerHTML=`<h2>Anonymous candidate</h2><pre>${esc(row.candidate_raw)}</pre>${claims||'<p>No parsed claims.</p>'}<p><label><input type="checkbox" id="forbidden"> contains forbidden inference</label></p><p><label>Notes (optional)<textarea id="notes" rows="4"></textarea></label></p>`;
 (state.claim_labels||[]).forEach((value,i)=>{const el=document.querySelector(`[data-label="${i}"]`);if(el)el.value=value;});
 document.getElementById('forbidden').checked=state.forbidden_inference===true;document.getElementById('notes').value=state.reviewer_notes||'';}
function save(){const row=queue[index],values=[...document.querySelectorAll('[data-label]')].map(el=>el.value);if(values.some(x=>!x)){alert('Complete every claim label.');return false;}
 saved[row.presentation_id]={presentation_id:row.presentation_id,claim_labels:values,forbidden_inference:document.getElementById('forbidden').checked,reviewer_notes:document.getElementById('notes').value};localStorage.setItem(storageKey,JSON.stringify(saved));return true;}
function csvCell(value){const s=String(value);return /[",\n]/.test(s)?`"${s.replace(/"/g,'""')}"`:s;}
function download(){if(!save())return;const missing=queue.filter(row=>!saved[row.presentation_id]);if(missing.length){alert(`${missing.length} presentations remain.`);return;}
 const fields=['presentation_id','claim_labels_json','forbidden_inference','reviewer_notes'],lines=[fields.join(',')];for(const row of queue){const s=saved[row.presentation_id],out={presentation_id:row.presentation_id,claim_labels_json:JSON.stringify(s.claim_labels),forbidden_inference:String(s.forbidden_inference),reviewer_notes:s.reviewer_notes};lines.push(fields.map(f=>csvCell(out[f])).join(','));}
 const blob=new Blob([lines.join('\n')+'\n'],{type:'text/csv'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='s3_completed_blind_review.csv';a.click();URL.revokeObjectURL(a.href);}
document.getElementById('prev').onclick=()=>{if(save()){index=(index+queue.length-1)%queue.length;render();}};document.getElementById('next').onclick=()=>{if(save()){index=(index+1)%queue.length;render();}};document.getElementById('download').onclick=download;render();
</script>
""".replace("__QUEUE__", embedded)
    atomic_write_text(path, document, mode=0o600)


def materialize_single_review_images(
    review_root: Path,
    run_root: Path,
    private_mapping: Iterable[dict[str, str]],
) -> dict[str, Any]:
    served_root = review_root.resolve()
    source_root = run_root.resolve()
    by_served: dict[str, str] = {}
    for row in private_mapping:
        served = str(row["served_image_relpath"])
        source = str(row["source_image_relpath"])
        prior = by_served.setdefault(served, source)
        if prior != source:
            raise ContractViolation("S3 repeated review image mapping changed")
    if len(by_served) != 20:
        raise ContractViolation("S3 review must materialize exactly 20 unique images")
    assets: list[dict[str, str]] = []
    for served, source in sorted(by_served.items()):
        served_rel = Path(served)
        source_rel = Path(source)
        if served_rel.is_absolute() or source_rel.is_absolute() or ".." in served_rel.parts or ".." in source_rel.parts:
            raise ContractViolation("S3 review image path escaped an allowed root")
        if served_rel.parts[:2] not in {("images", "pilot200"), ("images", "pilot200_v2")} or served_rel.suffix.casefold() != ".png":
            raise ContractViolation("S3 served review image namespace changed")
        src = (source_root / source_rel).resolve()
        dst = (served_root / served_rel).resolve()
        if source_root not in src.parents or served_root not in dst.parents or not src.is_file():
            raise ContractViolation("S3 review source image is missing or escaped the run")
        dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(dst.parent, 0o700)
        atomic_write_bytes(dst, src.read_bytes(), mode=0o600)
        if sha256_file(src) != sha256_file(dst):
            raise ContractViolation("S3 served review image hash mismatch")
        assets.append({"image_relpath": served, "sha256": sha256_file(dst)})
    return {"served_image_count": 20, "served_root": "human_review", "assets": assets}


def validate_completed_single_review(
    public_rows: Iterable[dict[str, str]], completed_rows: Iterable[dict[str, str]]
) -> list[dict[str, Any]]:
    public_by_id = {str(row["presentation_id"]): dict(row) for row in public_rows}
    completed = [dict(row) for row in completed_rows]
    ids = [str(row.get("presentation_id", "")) for row in completed]
    if len(ids) != len(set(ids)) or set(ids) != set(public_by_id):
        raise ContractViolation("Completed S3 review IDs do not exactly match the blind queue")
    normalized: list[dict[str, Any]] = []
    for row in completed:
        presentation_id = str(row["presentation_id"])
        try:
            labels = json.loads(str(row["claim_labels_json"]))
            findings = json.loads(public_by_id[presentation_id]["candidate_findings_json"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ContractViolation("S3 review claim labels are not valid JSON") from exc
        if not isinstance(labels, list) or len(labels) != len(findings):
            raise ContractViolation("S3 review claim-label count mismatch")
        if any(label not in {"supported", "unsupported", "not_assessable"} for label in labels):
            raise ContractViolation("S3 review contains an invalid claim label")
        forbidden = str(row.get("forbidden_inference", "")).casefold()
        if forbidden not in {"true", "false"}:
            raise ContractViolation("S3 review forbidden-inference flag is invalid")
        normalized.append({
            "presentation_id": presentation_id,
            "claim_labels": labels,
            "forbidden_inference": forbidden == "true",
            "reviewer_notes": str(row.get("reviewer_notes", "")),
        })
    return normalized


def audit_single_intra_rater_consistency(
    normalized_rows: Iterable[dict[str, Any]], private_mapping: Iterable[dict[str, str]]
) -> dict[str, Any]:
    mapping = {str(row["presentation_id"]): dict(row) for row in private_mapping}
    grouped: dict[str, list[str]] = defaultdict(list)
    normalized = [dict(row) for row in normalized_rows]
    if len(normalized) != len(mapping):
        raise ContractViolation("S3 completed review and private mapping row counts differ")
    for row in normalized:
        private = mapping.get(str(row["presentation_id"]))
        if private is None:
            raise ContractViolation("S3 review presentation is absent from private mapping")
        grouped[str(private["source_review_id"])].append(canonical_json({
            "labels": row["claim_labels"],
            "forbidden": row["forbidden_inference"],
        }))
    repeated = [values for values in grouped.values() if len(values) == 2]
    if len(repeated) != 2 or any(len(values) not in {1, 2} for values in grouped.values()):
        raise ContractViolation("S3 anonymous repeat structure is invalid")
    consistent = sum(len(set(values)) == 1 for values in repeated)
    rate = consistent / 2
    return {
        "repeat_source_count": 2,
        "consistent_source_count": consistent,
        "consistency": rate,
        "passed": rate >= 0.90,
        "existing_human_conclusions_valid": rate >= 0.90,
    }


def human_review_metrics(
    normalized_rows: Iterable[dict[str, Any]], private_mapping: Iterable[dict[str, str]]
) -> dict[str, Any]:
    normalized = [dict(row) for row in normalized_rows]
    mapping = {str(row["presentation_id"]): dict(row) for row in private_mapping}
    by_source: dict[str, dict[str, Any]] = {}
    for row in normalized:
        private = mapping[str(row["presentation_id"])]
        by_source.setdefault(str(private["source_review_id"]), row)
    labels = [label for row in by_source.values() for label in row["claim_labels"]]
    unsupported = sum(label in {"unsupported", "not_assessable"} for label in labels)
    consistency = audit_single_intra_rater_consistency(normalized, mapping.values())
    return {
        "unique_image_count": len(by_source),
        "presentation_count": len(normalized),
        "claim_count": len(labels),
        "unsupported_plus_not_assessable_count": unsupported,
        "unsupported_plus_not_assessable_rate": unsupported / len(labels) if labels else None,
        "forbidden_inference_count": sum(bool(row["forbidden_inference"]) for row in by_source.values()),
        "consistency": consistency,
        "passed": (
            bool(labels)
            and unsupported / len(labels) <= 0.05
            and not any(row["forbidden_inference"] for row in by_source.values())
            and consistency["passed"]
        ),
    }
