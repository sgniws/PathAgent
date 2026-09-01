from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import random
from collections import defaultdict
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


REVIEW_FIELDS = [
    "presentation_id",
    "source_review_id",
    "is_repeat",
    "image_relpath",
    "candidate_a",
    "candidate_b",
    "claim_labels_json",
    "pair_preference",
    "forbidden_inference",
    "reviewer_notes",
]

PAIRED_BLIND_FIELDS = [
    "presentation_id",
    "image_relpath",
    "candidate_a_raw",
    "candidate_a_findings_json",
    "candidate_b_raw",
    "candidate_b_findings_json",
    "candidate_a_claim_labels_json",
    "candidate_b_claim_labels_json",
    "candidate_a_forbidden_inference",
    "candidate_b_forbidden_inference",
    "pair_preference",
    "reviewer_notes",
]

PAIRED_COMPLETED_FIELDS = [
    "presentation_id",
    "candidate_a_claim_labels_json",
    "candidate_b_claim_labels_json",
    "candidate_a_forbidden_inference",
    "candidate_b_forbidden_inference",
    "pair_preference",
    "reviewer_notes",
]


def _presentation_id(source_id: str, repeat_index: int) -> str:
    digest = hashlib.sha256(f"{SPLIT_SEED}:{source_id}:{repeat_index}".encode()).hexdigest()[:20]
    return f"review_{digest}"


def build_review_presentations(
    records: Iterable[dict[str, Any]], *, repeat_fraction: float = 0.10, paired: bool
) -> list[dict[str, Any]]:
    source = [dict(row) for row in records]
    if not source:
        raise ContractViolation("Review export cannot be empty")
    source_ids = [str(row["review_id"]) for row in source]
    if len(source_ids) != len(set(source_ids)):
        raise ContractViolation("Review IDs are not unique")
    rng = random.Random(SPLIT_SEED)
    rows: list[dict[str, Any]] = []
    for row in source:
        candidates = [str(row["candidate_a"]), str(row.get("candidate_b", ""))]
        if paired and rng.randrange(2):
            candidates.reverse()
        rows.append(
            {
                "presentation_id": _presentation_id(str(row["review_id"]), 0),
                "source_review_id": str(row["review_id"]),
                "is_repeat": "false",
                "image_relpath": str(row["image_relpath"]),
                "candidate_a": candidates[0],
                "candidate_b": candidates[1] if paired else "",
                "claim_labels_json": "",
                "pair_preference": "",
                "forbidden_inference": "",
                "reviewer_notes": "",
            }
        )
    repeat_count = max(1, round(len(rows) * repeat_fraction))
    chosen = rng.sample(rows, repeat_count)
    for row in chosen:
        repeated = dict(row)
        repeated["presentation_id"] = _presentation_id(row["source_review_id"], 1)
        repeated["is_repeat"] = "true"
        rows.append(repeated)
    rng.shuffle(rows)
    return rows


def export_review_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    output = []
    output.append(",".join(REVIEW_FIELDS))
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue(), mode=0o600)


def export_review_ui(path: Path, csv_relpath: str) -> None:
    document = f"""<!doctype html>
<meta charset="utf-8"><title>PATHO-LORA-SFT-01 blind review</title>
<style>body{{font:16px system-ui;max-width:900px;margin:2rem auto;line-height:1.5}}code{{background:#eee;padding:.2rem}}.warning{{color:#8b1a1a}}</style>
<h1>Blind morphology review</h1>
<p class="warning">Do not infer organ, diagnosis, grade, IHC, molecular result, treatment, or prognosis.</p>
<p>Review the anonymous image first. Label every candidate claim as <code>supported</code>, <code>unsupported</code>, or <code>not_assessable</code>. Mark forbidden inference separately.</p>
<p>Queue: <a href="{html.escape(csv_relpath)}">{html.escape(csv_relpath)}</a>. Repeated items are anonymous in presentation order; do not inspect source IDs while reviewing.</p>
"""
    atomic_write_text(path, document, mode=0o600)


def audit_intra_rater_consistency(completed_rows: Iterable[dict[str, str]]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for row in completed_rows:
        labels = str(row.get("claim_labels_json", "")).strip()
        preference = str(row.get("pair_preference", "")).strip()
        forbidden = str(row.get("forbidden_inference", "")).strip().casefold()
        if not labels or not forbidden:
            raise ContractViolation("Completed review row is missing required labels")
        grouped[str(row["source_review_id"])].append((canonical_json(labels), preference, forbidden))
    repeated = [values for values in grouped.values() if len(values) > 1]
    if not repeated:
        raise ContractViolation("No anonymous repeat presentations were completed")
    consistent = sum(len(set(values)) == 1 for values in repeated)
    rate = consistent / len(repeated)
    return {
        "repeat_source_count": len(repeated),
        "consistent_source_count": consistent,
        "consistency": rate,
        "passed": rate >= 0.90,
        "existing_human_conclusions_valid": rate >= 0.90,
    }


def build_paired_blind_review(
    records: Iterable[dict[str, Any]], *, repeat_fraction: float = 0.10
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build a reviewer-facing queue and a separate private unblinding map."""
    source = [dict(row) for row in records]
    if not source:
        raise ContractViolation("Paired blind review cannot be empty")
    source_ids = [str(row["review_id"]) for row in source]
    if len(source_ids) != len(set(source_ids)):
        raise ContractViolation("Paired blind review IDs are not unique")
    if not 0 < repeat_fraction <= 0.5:
        raise ContractViolation("Blind repeat fraction must be in (0, 0.5]")
    rng = random.Random(SPLIT_SEED)

    def presentation(row: dict[str, Any], repeat_index: int) -> tuple[dict[str, str], dict[str, str]]:
        candidates = [
            {
                "source": str(row["candidate_a_source"]),
                "raw": str(row["candidate_a_raw"]),
                "findings": list(row["candidate_a_findings"]),
            },
            {
                "source": str(row["candidate_b_source"]),
                "raw": str(row["candidate_b_raw"]),
                "findings": list(row["candidate_b_findings"]),
            },
        ]
        if not candidates[0]["source"] or candidates[0]["source"] == candidates[1]["source"]:
            raise ContractViolation("Paired candidates require two distinct private sources")
        if rng.randrange(2):
            candidates.reverse()
        image_relpath = str(row["image_relpath"])
        if image_relpath.startswith("/") or ".." in Path(image_relpath).parts:
            raise ContractViolation("Blind review image path must be contained and relative")
        presentation_id = _presentation_id(str(row["review_id"]), repeat_index)
        public = {
            "presentation_id": presentation_id,
            "image_relpath": image_relpath,
            "candidate_a_raw": candidates[0]["raw"],
            "candidate_a_findings_json": canonical_json(candidates[0]["findings"]),
            "candidate_b_raw": candidates[1]["raw"],
            "candidate_b_findings_json": canonical_json(candidates[1]["findings"]),
            "candidate_a_claim_labels_json": "",
            "candidate_b_claim_labels_json": "",
            "candidate_a_forbidden_inference": "",
            "candidate_b_forbidden_inference": "",
            "pair_preference": "",
            "reviewer_notes": "",
        }
        private = {
            "presentation_id": presentation_id,
            "source_review_id": str(row["review_id"]),
            "is_repeat": "true" if repeat_index else "false",
            "candidate_a_source": candidates[0]["source"],
            "candidate_b_source": candidates[1]["source"],
        }
        return public, private

    pairs = [presentation(row, 0) for row in source]
    repeat_count = max(1, round(len(source) * repeat_fraction))
    for row in rng.sample(source, repeat_count):
        pairs.append(presentation(row, 1))
    rng.shuffle(pairs)
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def export_paired_review_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=PAIRED_BLIND_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue(), mode=0o600)


def export_paired_review_ui(path: Path, rows: Iterable[dict[str, str]]) -> None:
    queue = [dict(row) for row in rows]
    embedded = json.dumps(queue, ensure_ascii=False, separators=(",", ":"))
    embedded = embedded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    document = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><base href="../../">
<title>PATHO-LORA-SFT-01 S2 blind review</title>
<style>
body{font:15px system-ui;max-width:1180px;margin:1.5rem auto;padding:0 1rem;line-height:1.45;background:#fafafa;color:#182026}
.warning{color:#8b1a1a}.grid{display:grid;grid-template-columns:minmax(360px,1fr) minmax(320px,1fr);gap:1rem}
img{width:100%;max-height:760px;object-fit:contain;background:#eee}.candidate{background:white;border:1px solid #ccd3d8;border-radius:8px;padding:1rem;margin-bottom:1rem}
pre{white-space:pre-wrap;overflow-wrap:anywhere}.claim{display:grid;grid-template-columns:1fr 180px;gap:.5rem;margin:.5rem 0}
button,select,textarea{font:inherit;padding:.45rem}textarea{width:100%;box-sizing:border-box}.toolbar{display:flex;gap:.6rem;align-items:center;margin:1rem 0}
@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style>
<h1>S2 paired blind morphology review</h1>
<p class="warning">Judge direct visible morphology only. Do not infer organ, diagnosis, grade, lineage, IHC, molecular result, treatment, prognosis, or clinical meaning.</p>
<p>Label every claim for A and B as supported, unsupported, or not_assessable; mark forbidden inference separately; then choose A, B, or tie. Model identities, automatic scores, and repeat status are intentionally hidden.</p>
<div class="toolbar"><button id="prev">Previous</button><strong id="progress"></strong><button id="next">Save & next</button><button id="download">Download completed CSV</button></div>
<div class="grid"><div><img id="image" alt="anonymous H&E patch"></div><div id="form"></div></div>
<script>
const queue=__QUEUE__;
const storageKey='patho-lora-sft-01-s2-review-v1';
const saved=JSON.parse(localStorage.getItem(storageKey)||'{}'); let index=0;
const statuses=['','supported','unsupported','not_assessable'];
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function candidate(letter,row){
 const findings=JSON.parse(row[`candidate_${letter}_findings_json`]);
 const controls=findings.map((finding,i)=>`<div class="claim"><span>${i+1}. ${esc(finding)}</span><select data-kind="label" data-side="${letter}" data-index="${i}">${statuses.map(x=>`<option value="${x}">${x||'select status'}</option>`).join('')}</select></div>`).join('');
 return `<section class="candidate"><h2>Candidate ${letter.toUpperCase()}</h2><pre>${esc(row[`candidate_${letter}_raw`])}</pre>${controls||'<p>No parsed claims.</p>'}<label><input type="checkbox" data-kind="forbidden" data-side="${letter}"> contains forbidden inference</label></section>`;
}
function render(){
 const row=queue[index], state=saved[row.presentation_id]||{};
 document.getElementById('progress').textContent=`${index+1} / ${queue.length} — ${Object.keys(saved).length} saved`;
 document.getElementById('image').src=row.image_relpath;
 document.getElementById('form').innerHTML=candidate('a',row)+candidate('b',row)+`<section class="candidate"><label>Pair preference <select id="preference"><option value="">select</option><option>A</option><option>B</option><option>tie</option></select></label><p><label>Notes (optional)<textarea id="notes" rows="3"></textarea></label></p></section>`;
 for(const side of ['a','b']){
  const labels=state[`candidate_${side}_claim_labels`]||[];
  document.querySelectorAll(`[data-kind=label][data-side=${side}]`).forEach((el,i)=>el.value=labels[i]||'');
  document.querySelector(`[data-kind=forbidden][data-side=${side}]`).checked=state[`candidate_${side}_forbidden_inference`]===true;
 }
 document.getElementById('preference').value=state.pair_preference||''; document.getElementById('notes').value=state.reviewer_notes||'';
}
function save(){
 const row=queue[index], state={presentation_id:row.presentation_id};
 for(const side of ['a','b']){
  const values=[...document.querySelectorAll(`[data-kind=label][data-side=${side}]`)].map(el=>el.value);
  if(values.some(x=>!x)){alert(`Complete every Candidate ${side.toUpperCase()} claim label.`);return false;}
  state[`candidate_${side}_claim_labels`]=values;
  state[`candidate_${side}_forbidden_inference`]=document.querySelector(`[data-kind=forbidden][data-side=${side}]`).checked;
 }
 state.pair_preference=document.getElementById('preference').value; if(!state.pair_preference){alert('Choose A, B, or tie.');return false;}
 state.reviewer_notes=document.getElementById('notes').value; saved[row.presentation_id]=state; localStorage.setItem(storageKey,JSON.stringify(saved)); return true;
}
function csvCell(value){const s=String(value);return /[",\\n]/.test(s)?`"${s.replace(/"/g,'""')}"`:s;}
function download(){
 if(!save())return; const missing=queue.filter(row=>!saved[row.presentation_id]); if(missing.length){alert(`${missing.length} presentations remain.`);return;}
 const fields=['presentation_id','candidate_a_claim_labels_json','candidate_b_claim_labels_json','candidate_a_forbidden_inference','candidate_b_forbidden_inference','pair_preference','reviewer_notes'];
 const lines=[fields.join(',')]; for(const row of queue){const s=saved[row.presentation_id];const out={presentation_id:row.presentation_id,candidate_a_claim_labels_json:JSON.stringify(s.candidate_a_claim_labels),candidate_b_claim_labels_json:JSON.stringify(s.candidate_b_claim_labels),candidate_a_forbidden_inference:String(s.candidate_a_forbidden_inference),candidate_b_forbidden_inference:String(s.candidate_b_forbidden_inference),pair_preference:s.pair_preference,reviewer_notes:s.reviewer_notes};lines.push(fields.map(f=>csvCell(out[f])).join(','));}
 const blob=new Blob([lines.join('\\n')+'\\n'],{type:'text/csv'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='s2_completed_blind_review.csv';a.click();URL.revokeObjectURL(a.href);
}
document.getElementById('prev').onclick=()=>{if(save()){index=(index+queue.length-1)%queue.length;render();}};
document.getElementById('next').onclick=()=>{if(save()){index=(index+1)%queue.length;render();}};
document.getElementById('download').onclick=download;render();
</script>
""".replace("__QUEUE__", embedded)
    atomic_write_text(path, document, mode=0o600)


def materialize_paired_review_images(
    review_server_root: Path,
    run_root: Path,
    rows: Iterable[dict[str, str]],
) -> dict[str, Any]:
    """Copy only reviewer-selected, deidentified patches into the served tree."""
    served_root = review_server_root.resolve()
    source_root = run_root.resolve()
    relative_paths = sorted({str(row["image_relpath"]) for row in rows})
    if not relative_paths:
        raise ContractViolation("Paired review image export cannot be empty")

    assets: list[dict[str, str]] = []
    for relative in relative_paths:
        relpath = Path(relative)
        if relpath.is_absolute() or ".." in relpath.parts:
            raise ContractViolation("Blind review image path must be contained and relative")
        if len(relpath.parts) < 3 or relpath.parts[:2] != ("images", "calibration50"):
            raise ContractViolation("Blind review image path escaped the calibration image namespace")
        if relpath.suffix.casefold() != ".png":
            raise ContractViolation("Blind review assets must be re-encoded PNG files")

        source = (source_root / relpath).resolve()
        destination = (served_root / relpath).resolve()
        if source_root not in source.parents or served_root not in destination.parents:
            raise ContractViolation("Blind review image path escaped an allowed root")
        if not source.is_file():
            raise ContractViolation(f"Blind review image is missing: {relative}")

        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(destination.parent, 0o700)
        atomic_write_bytes(destination, source.read_bytes(), mode=0o600)
        source_sha = sha256_file(source)
        if sha256_file(destination) != source_sha:
            raise ContractViolation("Served blind review image hash mismatch")
        assets.append({"image_relpath": relative, "sha256": source_sha})

    return {
        "served_image_count": len(assets),
        "served_root": "human_review",
        "assets": assets,
    }


def validate_completed_paired_review(
    public_rows: Iterable[dict[str, str]], completed_rows: Iterable[dict[str, str]]
) -> list[dict[str, Any]]:
    public_by_id = {str(row["presentation_id"]): dict(row) for row in public_rows}
    completed = [dict(row) for row in completed_rows]
    completed_ids = [str(row.get("presentation_id", "")) for row in completed]
    if len(completed_ids) != len(set(completed_ids)) or set(completed_ids) != set(public_by_id):
        raise ContractViolation("Completed paired review IDs do not exactly match the blind queue")
    normalized: list[dict[str, Any]] = []
    for row in completed:
        presentation_id = str(row["presentation_id"])
        public = public_by_id[presentation_id]
        output: dict[str, Any] = {"presentation_id": presentation_id}
        for side in ("a", "b"):
            try:
                labels = json.loads(str(row[f"candidate_{side}_claim_labels_json"]))
                findings = json.loads(public[f"candidate_{side}_findings_json"])
            except (json.JSONDecodeError, TypeError) as exc:
                raise ContractViolation("Paired review claim labels are not valid JSON") from exc
            if not isinstance(labels, list) or len(labels) != len(findings):
                raise ContractViolation("Paired review claim-label count mismatch")
            if any(label not in {"supported", "unsupported", "not_assessable"} for label in labels):
                raise ContractViolation("Paired review contains an invalid claim label")
            forbidden = str(row[f"candidate_{side}_forbidden_inference"]).casefold()
            if forbidden not in {"true", "false"}:
                raise ContractViolation("Paired review forbidden-inference flag is invalid")
            output[f"candidate_{side}_claim_labels"] = labels
            output[f"candidate_{side}_forbidden_inference"] = forbidden == "true"
        preference = str(row.get("pair_preference", ""))
        if preference not in {"A", "B", "tie"}:
            raise ContractViolation("Paired review preference must be A, B, or tie")
        output["pair_preference"] = preference
        output["reviewer_notes"] = str(row.get("reviewer_notes", ""))
        normalized.append(output)
    return normalized


def audit_paired_intra_rater_consistency(
    normalized_rows: Iterable[dict[str, Any]], private_mapping: Iterable[dict[str, str]]
) -> dict[str, Any]:
    mapping = {str(row["presentation_id"]): dict(row) for row in private_mapping}
    grouped: dict[str, list[str]] = defaultdict(list)
    row_count = 0
    for row in normalized_rows:
        row_count += 1
        presentation_id = str(row["presentation_id"])
        if presentation_id not in mapping:
            raise ContractViolation("Completed review presentation is absent from private mapping")
        private = mapping[presentation_id]
        by_source = {}
        for side in ("a", "b"):
            source = private[f"candidate_{side}_source"]
            by_source[source] = {
                "labels": row[f"candidate_{side}_claim_labels"],
                "forbidden": row[f"candidate_{side}_forbidden_inference"],
            }
        preference = row["pair_preference"]
        preferred_source = "tie" if preference == "tie" else private[f"candidate_{preference.casefold()}_source"]
        grouped[private["source_review_id"]].append(
            canonical_json({"by_source": by_source, "preferred_source": preferred_source})
        )
    if row_count != len(mapping):
        raise ContractViolation("Completed review and private mapping row counts differ")
    repeated = [values for values in grouped.values() if len(values) == 2]
    if any(len(values) not in {1, 2} for values in grouped.values()) or not repeated:
        raise ContractViolation("Paired review repeat structure is invalid")
    consistent = sum(len(set(values)) == 1 for values in repeated)
    rate = consistent / len(repeated)
    return {
        "repeat_source_count": len(repeated),
        "consistent_source_count": consistent,
        "consistency": rate,
        "passed": rate >= 0.90,
        "existing_human_conclusions_valid": rate >= 0.90,
    }
