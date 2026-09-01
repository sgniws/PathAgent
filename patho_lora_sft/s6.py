from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from pathlib import Path

from .common import (
    ContractViolation,
    PrivacyViolation,
    atomic_write_bytes,
    atomic_write_text,
    canonical_json,
    sha256_file,
)
from .contracts import SPLIT_SEED, TEST_SPLIT_NAMES
from .schema import canonical_target


S6_REAL_TARGETS = {
    "train": {"clean": 1800, "risk": 100, "max_per_slide": 16},
    "validation": {"clean": 90, "risk": 5, "max_per_slide": 10},
    "locked": {"clean": 90, "risk": 5, "max_per_slide": 10},
}


def _score(*parts: Any) -> str:
    value = "\x1f".join([str(SPLIT_SEED), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def candidate_id_for(row: dict[str, Any], assigned_split: str) -> str:
    return "cand_" + _score(
        "candidate",
        assigned_split,
        row["patient_group_id"],
        row["slide_id"],
        row["patch_id"],
    )[:24]


def s6_sample_id(candidate_id: str) -> str:
    if not str(candidate_id).startswith("cand_"):
        raise ContractViolation("S6 sample source is not an anonymous candidate")
    return "sft_" + _score("s6_sample", candidate_id)[:24]


def _tier(row: dict[str, Any]) -> str | None:
    if row.get("tier") in {"clean", "risk"}:
        return str(row["tier"])
    if bool(row.get("qc_risk")):
        return "risk"
    if str(row.get("selection_auto_status", "")).casefold() == "pass":
        return "clean"
    return None


def build_s6_real_candidate_plan(
    *,
    frozen_candidates: dict[str, list[dict[str, Any]]],
    full_non_test_pool: list[dict[str, Any]],
    assignment_by_slide: dict[str, dict[str, str]],
    existing_train_final: list[dict[str, Any]],
    previously_rejected_candidate_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Fill the remaining 1,900 real slots without reusing an attempted rejection."""
    if set(frozen_candidates) != {"train", "validation", "locked"}:
        raise ContractViolation("S6 requires all three frozen candidate splits")
    if len(existing_train_final) != 190:
        raise ContractViolation("S6 requires the frozen 190-real Train prefix")

    existing_ids = {str(row.get("candidate_id", "")) for row in existing_train_final}
    if len(existing_ids) != 190 or any(not value.startswith("cand_") for value in existing_ids):
        raise ContractViolation("S3 final candidate IDs are missing or duplicated")
    existing_sources = {
        (str(row.get("slide_id", "")), str(row.get("patch_id", "")))
        for row in existing_train_final
    }
    if len(existing_sources) != 190:
        raise ContractViolation("S3 final source keys are missing or duplicated")
    existing_tiers = Counter(_tier(row) for row in existing_train_final)
    if existing_tiers != Counter(clean=180, risk=10):
        raise ContractViolation("S3 Train-prefix tier counts changed")

    used_ids = set(existing_ids)
    used_sources = set(existing_sources)
    per_split_slide: dict[str, Counter[str]] = {
        "train": Counter(str(row["slide_id"]) for row in existing_train_final),
        "validation": Counter(),
        "locked": Counter(),
    }
    plan: list[dict[str, Any]] = []
    carryover_replacements = Counter()

    pool_by_split_tier: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in full_non_test_pool:
        slide_id = str(row.get("slide_id", ""))
        assignment = assignment_by_slide.get(slide_id)
        if assignment is None:
            raise ContractViolation("A non-test pool row lacks a frozen split assignment")
        split = str(assignment["assigned_split"])
        if split not in S6_REAL_TARGETS:
            raise ContractViolation("Unexpected assigned split in S6 pool")
        if str(row.get("split", "")).casefold() in TEST_SPLIT_NAMES:
            raise PrivacyViolation("Test10 row entered the S6 candidate planner")
        tier = _tier(row)
        if tier is None:
            continue
        candidate = {
            **row,
            "patient_group_id": assignment["patient_group_id"],
            "assigned_split": split,
            "tier": tier,
        }
        candidate["candidate_id"] = candidate_id_for(candidate, split)
        pool_by_split_tier.setdefault((split, tier), []).append(candidate)
    for key, rows in pool_by_split_tier.items():
        rows.sort(key=lambda row: _score("s6_reserve", key[0], key[1], row["candidate_id"]))

    for split in ("train", "validation", "locked"):
        frozen = [dict(row) for row in frozen_candidates[split]]
        frozen_by_tier = Counter(_tier(row) for row in frozen)
        target = S6_REAL_TARGETS[split]
        expected_frozen = Counter(clean=target["clean"], risk=target["risk"])
        if frozen_by_tier != expected_frozen:
            raise ContractViolation(f"Frozen {split} candidate tier counts changed")
        existing_for_split = existing_tiers if split == "train" else Counter()
        required = {
            tier: target[tier] - int(existing_for_split[tier])
            for tier in ("clean", "risk")
        }
        for tier in ("clean", "risk"):
            eligible = sorted(
                (
                    row
                    for row in frozen
                    if _tier(row) == tier
                    and str(row["candidate_id"]) not in used_ids
                    and str(row["candidate_id"]) not in previously_rejected_candidate_ids
                ),
                key=lambda row: str(row["candidate_id"]),
            )
            chosen: list[dict[str, Any]] = []
            for row in eligible:
                source = (str(row["slide_id"]), str(row["patch_id"]))
                if source in used_sources:
                    continue
                if per_split_slide[split][str(row["slide_id"])] >= target["max_per_slide"]:
                    continue
                chosen.append({**row, "plan_source": "frozen_candidate"})
                used_ids.add(str(row["candidate_id"]))
                used_sources.add(source)
                per_split_slide[split][str(row["slide_id"])] += 1
                if len(chosen) == required[tier]:
                    break

            deficit = required[tier] - len(chosen)
            if deficit:
                for row in pool_by_split_tier.get((split, tier), []):
                    candidate_id = str(row["candidate_id"])
                    source = (str(row["slide_id"]), str(row["patch_id"]))
                    if (
                        candidate_id in used_ids
                        or candidate_id in previously_rejected_candidate_ids
                        or source in used_sources
                        or per_split_slide[split][str(row["slide_id"])] >= target["max_per_slide"]
                    ):
                        continue
                    chosen.append({**row, "plan_source": "carryover_replacement_for_prior_rejection"})
                    used_ids.add(candidate_id)
                    used_sources.add(source)
                    per_split_slide[split][str(row["slide_id"])] += 1
                    carryover_replacements[(split, tier)] += 1
                    if len(chosen) == required[tier]:
                        break
            if len(chosen) != required[tier]:
                raise ContractViolation(f"S6 cannot fill the frozen {split}/{tier} real quota")
            plan.extend(chosen)

    expected_new_real = sum(
        int(target["clean"]) + int(target["risk"])
        for target in S6_REAL_TARGETS.values()
    ) - len(existing_train_final)
    if len(plan) != expected_new_real:
        raise ContractViolation(
            f"S6 real candidate plan must contain exactly {expected_new_real:,} rows"
        )
    plan_ids = [str(row["candidate_id"]) for row in plan]
    plan_sources = [(str(row["slide_id"]), str(row["patch_id"])) for row in plan]
    if len(set(plan_ids)) != len(plan_ids) or len(set(plan_sources)) != len(plan_sources):
        raise ContractViolation("S6 candidate plan is not unique")

    reserve: list[dict[str, Any]] = []
    for split in ("train", "validation", "locked"):
        for tier in ("clean", "risk"):
            for row in pool_by_split_tier.get((split, tier), []):
                candidate_id = str(row["candidate_id"])
                source = (str(row["slide_id"]), str(row["patch_id"]))
                if candidate_id in used_ids or candidate_id in previously_rejected_candidate_ids or source in used_sources:
                    continue
                reserve.append(row)
    reserve.sort(key=lambda row: _score("s6_runtime_reserve", row["assigned_split"], row["tier"], row["candidate_id"]))
    if len({str(row["candidate_id"]) for row in reserve}) != len(reserve):
        raise ContractViolation("S6 runtime reserve candidate IDs are duplicated")

    summary = {
        "new_real_records": len(plan),
        "new_real_by_split_tier": {
            split: dict(Counter(row["tier"] for row in plan if row["assigned_split"] == split))
            for split in ("train", "validation", "locked")
        },
        "carryover_replacements_for_s3_rejections": {
            f"{split}_{tier}": carryover_replacements[(split, tier)]
            for split in ("train", "validation", "locked")
            for tier in ("clean", "risk")
            if carryover_replacements[(split, tier)]
        },
        "previously_rejected_candidate_ids_excluded": len(previously_rejected_candidate_ids),
        "runtime_reserve_rows": len(reserve),
        "maximum_real_per_slide": {
            split: max(per_split_slide[split].values(), default=0)
            for split in ("train", "validation", "locked")
        },
    }
    return sorted(plan, key=lambda row: _score("s6_plan_order", row["assigned_split"], row["candidate_id"])), reserve, summary


def assess_s6_candidate(
    observation: dict[str, Any], *, natural_qc_risk: bool
) -> dict[str, Any]:
    """Apply the frozen Luna-generation/Gemini-audit production boundary."""
    if observation.get("strict_schema_valid") is not True:
        return {"status": "rejected", "risk_tier": "high", "reason": "luna_schema_failure"}
    if observation.get("raw_forbidden_hits"):
        return {"status": "rejected", "risk_tier": "high", "reason": "luna_forbidden_content"}
    findings = list(observation.get("findings") or [])
    audit = observation.get("audit") if isinstance(observation.get("audit"), dict) else {}
    if audit.get("strict_schema_valid") is not True:
        return {"status": "rejected", "risk_tier": "high", "reason": "gemini_audit_schema_failure"}
    statuses = list(audit.get("statuses") or [])
    if len(statuses) != len(findings):
        return {"status": "rejected", "risk_tier": "high", "reason": "gemini_claim_count_mismatch"}
    if audit.get("forbidden_inference") is True:
        return {"status": "rejected", "risk_tier": "high", "reason": "gemini_forbidden_inference"}
    if any(status != "supported" for status in statuses):
        return {"status": "rejected", "risk_tier": "high", "reason": "gemini_rejected_luna_candidate"}
    if natural_qc_risk:
        return {
            "status": "waiting_human_review",
            "risk_tier": "medium",
            "reason": "natural_qc_risk_requires_positive_human_review",
            "requires_human_review": True,
            "target": canonical_target(findings),
        }
    return {
        "status": "accepted_automatic",
        "risk_tier": "low",
        "reason": "luna_candidate_passed_schema_safety_and_gemini_audit",
        "requires_human_review": False,
        "target": canonical_target(findings),
    }


def select_s6_human_review_records(
    real_records: list[dict[str, Any]],
    control_records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if len(real_records) != 1900:
        raise ContractViolation("S6 review selection requires 1,900 new real records")
    if len(control_records) != 100:
        raise ContractViolation("S6 review selection requires 100 new controls")

    def ranked(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda row: _score("s6_human_review", label, row["sample_id"]))

    train_risk = ranked(
        [row for row in real_records if row["assigned_split"] == "train" and row["tier"] == "risk"],
        "train_risk",
    )
    validation_risk = ranked(
        [row for row in real_records if row["assigned_split"] == "validation" and row["tier"] == "risk"],
        "validation_risk",
    )
    locked_risk = ranked(
        [row for row in real_records if row["assigned_split"] == "locked" and row["tier"] == "risk"],
        "locked_risk",
    )
    locked_clean = ranked(
        [row for row in real_records if row["assigned_split"] == "locked" and row["tier"] == "clean"],
        "locked_clean",
    )
    locked_controls = ranked(
        [row for row in control_records if row["assigned_split"] == "locked"],
        "locked_control",
    )
    if not (
        len(train_risk) == 90
        and len(validation_risk) == 5
        and len(locked_risk) == 5
        and len(locked_clean) == 90
        and len(locked_controls) == 5
    ):
        raise ContractViolation("S6 review source split/tier counts changed")
    selected = {
        "train90": train_risk,
        "validation5": validation_risk,
        "locked20": locked_risk + locked_clean[:14] + locked_controls[:1],
    }
    if len({str(row["sample_id"]) for rows in selected.values() for row in rows}) != 115:
        raise ContractViolation("S6 human review selections are not globally unique")
    return selected


def select_s6_expanded_train_review_records(
    real_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select the authorized S6 Train review expansion: 90 risk + 190 clean."""
    if len(real_records) != 1900:
        raise ContractViolation("S6 expanded Train review requires 1,900 new real records")

    def ranked(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda row: _score("s6_human_review", label, row["sample_id"]))

    train_risk = ranked(
        [
            row
            for row in real_records
            if row["assigned_split"] == "train" and row["tier"] == "risk"
        ],
        "train_risk",
    )
    train_clean = ranked(
        [
            row
            for row in real_records
            if row["assigned_split"] == "train" and row["tier"] == "clean"
        ],
        "train_clean_expanded_v2",
    )
    if len(train_risk) != 90 or len(train_clean) != 1620:
        raise ContractViolation("S6 expanded Train review source counts changed")
    selected = train_risk + train_clean[:190]
    if len(selected) != 280 or len({str(row["sample_id"]) for row in selected}) != 280:
        raise ContractViolation("S6 expanded Train review selection is not 280 unique images")
    return selected


def build_s6_single_blind_review(
    records: list[dict[str, Any]], *, namespace: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    expected = {"train90": 90, "train280": 280, "validation5": 5, "locked20": 20}
    if namespace not in expected or len(records) != expected[namespace]:
        raise ContractViolation("S6 blind review namespace or unique-image count changed")
    source_ids = [str(row.get("review_id", "")) for row in records]
    if any(not value for value in source_ids) or len(set(source_ids)) != len(source_ids):
        raise ContractViolation("S6 review IDs are missing or duplicated")

    def one(row: dict[str, Any], repeat_index: int) -> tuple[dict[str, str], dict[str, str]]:
        review_id = str(row["review_id"])
        presentation_id = "s6review_" + _score(
            "s6_review_presentation", namespace, review_id, repeat_index
        )[:20]
        served = f"images/{namespace}/{review_id}.png"
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
            "sample_id": str(row["sample_id"]),
            "assigned_split": str(row["assigned_split"]),
            "tier": str(row["tier"]),
        }
        return public, private

    pairs = [one(row, 0) for row in records]
    repeat_count = max(1, round(len(records) * 0.10))
    repeated = sorted(records, key=lambda row: _score("s6_review_repeat", namespace, row["review_id"]))[:repeat_count]
    pairs.extend(one(row, 1) for row in repeated)
    pairs.sort(key=lambda pair: _score("s6_review_order", namespace, pair[0]["presentation_id"]))
    public, private = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    if any(key in row for row in public for key in ("sample_id", "assigned_split", "tier", "is_repeat", "source_review_id")):
        raise PrivacyViolation("S6 public review queue leaked private sampling state")
    return public, private


def materialize_s6_review_images(
    *,
    review_root: Path,
    run_root: Path,
    private_mapping: list[dict[str, str]],
    expected_unique: int,
) -> dict[str, Any]:
    served_root = review_root.resolve()
    source_root = run_root.resolve()
    by_served: dict[str, str] = {}
    for row in private_mapping:
        served = str(row["served_image_relpath"])
        source = str(row["source_image_relpath"])
        prior = by_served.setdefault(served, source)
        if prior != source:
            raise ContractViolation("S6 repeated review image mapping changed")
    if len(by_served) != expected_unique:
        raise ContractViolation("S6 review unique-image count changed")
    assets = []
    for served, source in sorted(by_served.items()):
        served_rel, source_rel = Path(served), Path(source)
        if (
            served_rel.is_absolute()
            or source_rel.is_absolute()
            or ".." in served_rel.parts
            or ".." in source_rel.parts
        ):
            raise ContractViolation("S6 review image path escaped an allowed root")
        if served_rel.parts[:2] not in {
            ("images", "train90"),
            ("images", "train280"),
            ("images", "validation5"),
            ("images", "locked20"),
        } or served_rel.suffix.casefold() != ".png":
            raise ContractViolation("S6 served review image namespace changed")
        src = (source_root / source_rel).resolve()
        dst = (served_root / served_rel).resolve()
        if source_root not in src.parents or served_root not in dst.parents or not src.is_file():
            raise ContractViolation("S6 review source image is missing or escaped the run")
        dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(dst.parent, 0o700)
        if dst.exists():
            if sha256_file(src) != sha256_file(dst):
                raise ContractViolation("Existing S6 served review image hash changed")
        else:
            atomic_write_bytes(dst, src.read_bytes(), mode=0o600)
        assets.append({"image_relpath": served, "sha256": sha256_file(dst)})
    return {"served_image_count": len(assets), "assets": assets}


def export_s6_review_ui(
    path: Path,
    rows: list[dict[str, str]],
    *,
    namespace: str,
) -> None:
    expected_presentations = {
        "train90": 99,
        "train280": 308,
        "validation5": 6,
        "locked20": 22,
    }
    if namespace not in expected_presentations or len(rows) != expected_presentations[namespace]:
        raise ContractViolation("S6 review UI presentation count changed")
    embedded = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    embedded = embedded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    document = r"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><base href="../">
<title>PATHO-LORA-SFT-01 S6 blind review</title>
<style>
body{font:15px system-ui;margin:0;background:#f5f7f8;color:#17212b}.layout{display:grid;grid-template-columns:190px minmax(0,1fr);min-height:100vh}.side{background:#fff;border-right:1px solid #d7dde2;padding:1rem;position:sticky;top:0;height:100vh;overflow:auto;box-sizing:border-box}.main{padding:1rem;max-width:1250px;width:100%;box-sizing:border-box;margin:auto}.grid{display:grid;grid-template-columns:minmax(360px,1.15fr) minmax(330px,.85fr);gap:1rem}.card{background:white;border:1px solid #ccd3d8;border-radius:9px;padding:1rem}.warning{color:#8b1a1a}.toolbar{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;margin:.8rem 0}button,select,textarea{font:inherit;padding:.5rem}button{cursor:pointer}.jump{display:grid;grid-template-columns:repeat(5,1fr);gap:.25rem}.jump button{padding:.3rem;border:1px solid #ccd3d8;background:#fff}.jump button.done{background:#dff4e4}.jump button.current{outline:2px solid #2463a8}img{width:100%;height:min(70vh,760px);object-fit:contain;background:#eceff1}pre{white-space:pre-wrap;overflow-wrap:anywhere}.claim{display:grid;grid-template-columns:1fr 185px;gap:.5rem;margin:.6rem 0}textarea{width:100%;box-sizing:border-box}.error{color:#b00020}@media(max-width:850px){.layout{display:block}.side{position:static;height:auto;border-right:0}.grid{grid-template-columns:1fr}.jump{grid-template-columns:repeat(10,1fr)}}
</style>
<div class="layout"><aside class="side"><strong id="scope"></strong><p id="savedCount"></p><div id="jump" class="jump"></div></aside><main class="main">
<h1>S6 blind morphology review</h1><p class="warning">Judge only directly visible morphology. Do not infer organ, diagnosis, grade, lineage, IHC, molecular result, treatment, prognosis, or clinical meaning.</p>
<p>Model identity, automatic scores, risk tier, replacement status, split details, and repeat status are hidden. Changes are saved automatically in this browser.</p>
<div class="toolbar"><button id="prev">← Previous</button><strong id="progress"></strong><button id="next">Next →</button><button id="partial">Download backup</button><button id="complete">Download completed CSV</button></div>
<div class="grid"><section class="card"><img id="image" alt="anonymous H&amp;E patch"><p id="imageError" class="error"></p></section><section class="card" id="form"></section></div></main></div>
<script>
const queue=__QUEUE__,namespace='__NAMESPACE__',storageKey=`patho-lora-sft-01-s6-${namespace}-v1`;
const saved=JSON.parse(localStorage.getItem(storageKey)||'{}');let index=0;const statuses=['','supported','unsupported','not_assessable'];
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function persist(){localStorage.setItem(storageKey,JSON.stringify(saved));updateNav();}
function capture(requireAll=false){const row=queue[index],labels=[...document.querySelectorAll('[data-label]')].map(el=>el.value),reviewed=document.getElementById('reviewed').checked;if(requireAll&&(labels.some(x=>!x)||!reviewed)){alert('Complete every claim label and explicitly mark the image reviewed.');return false;}saved[row.presentation_id]={presentation_id:row.presentation_id,claim_labels:labels,forbidden_inference:document.getElementById('forbidden').checked,reviewed:reviewed,reviewer_notes:document.getElementById('notes').value};persist();return true;}
function completeState(id){const s=saved[id];return !!s&&s.reviewed===true&&Array.isArray(s.claim_labels)&&s.claim_labels.every(Boolean)&&typeof s.forbidden_inference==='boolean';}
function updateNav(){document.getElementById('savedCount').textContent=`${queue.filter(r=>completeState(r.presentation_id)).length} / ${queue.length} complete`;document.querySelectorAll('#jump button').forEach((b,i)=>{b.classList.toggle('done',completeState(queue[i].presentation_id));b.classList.toggle('current',i===index);});}
function render(){const row=queue[index],state=saved[row.presentation_id]||{},findings=JSON.parse(row.candidate_findings_json);document.getElementById('scope').textContent=namespace;document.getElementById('progress').textContent=`${index+1} / ${queue.length}`;const img=document.getElementById('image');document.getElementById('imageError').textContent='';img.onerror=()=>document.getElementById('imageError').textContent='Image failed to load. Stop review and report this item.';img.src=row.image_relpath;const claims=findings.map((f,i)=>`<div class="claim"><span>${i+1}. ${esc(f)}</span><select data-label="${i}">${statuses.map(x=>`<option value="${x}">${x||'select status'}</option>`).join('')}</select></div>`).join('');document.getElementById('form').innerHTML=`<h2>Anonymous candidate</h2><pre>${esc(row.candidate_raw)}</pre>${claims||'<p>Candidate contains no claims. Confirm the image is genuinely not assessable and mark forbidden inference if needed.</p>'}<p><label><input type="checkbox" id="forbidden"> contains forbidden inference</label></p><p><label><input type="checkbox" id="reviewed"> I inspected this image and candidate</label></p><p><label>Notes (optional)<textarea id="notes" rows="4"></textarea></label></p>`;(state.claim_labels||[]).forEach((v,i)=>{const el=document.querySelector(`[data-label="${i}"]`);if(el)el.value=v;});document.getElementById('forbidden').checked=state.forbidden_inference===true;document.getElementById('reviewed').checked=state.reviewed===true;document.getElementById('notes').value=state.reviewer_notes||'';document.querySelectorAll('select,input,textarea').forEach(el=>el.addEventListener('change',()=>capture(false)));updateNav();}
function move(delta){capture(false);index=(index+queue.length+delta)%queue.length;render();}
function cell(v){const s=String(v??'');return /[",\n]/.test(s)?`"${s.replace(/"/g,'""')}"`:s;}
function download(complete){capture(false);if(complete){const missing=queue.filter(r=>!completeState(r.presentation_id));if(missing.length){alert(`${missing.length} presentations remain.`);return;}}const fields=['presentation_id','claim_labels_json','forbidden_inference','reviewer_notes'],lines=[fields.join(',')];for(const row of queue){const s=saved[row.presentation_id]||{},out={presentation_id:row.presentation_id,claim_labels_json:JSON.stringify(s.claim_labels||[]),forbidden_inference:typeof s.forbidden_inference==='boolean'?String(s.forbidden_inference):'',reviewer_notes:s.reviewer_notes||''};lines.push(fields.map(f=>cell(out[f])).join(','));}const blob=new Blob([lines.join('\n')+'\n'],{type:'text/csv'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`s6_${namespace}_${complete?'completed':'backup'}_review.csv`;a.click();URL.revokeObjectURL(a.href);}
const jump=document.getElementById('jump');queue.forEach((_,i)=>{const b=document.createElement('button');b.textContent=i+1;b.onclick=()=>{capture(false);index=i;render();};jump.appendChild(b);});document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);document.getElementById('partial').onclick=()=>download(false);document.getElementById('complete').onclick=()=>download(true);document.addEventListener('keydown',e=>{if(e.target.matches('textarea,select,input'))return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);});render();
</script>
""".replace("__QUEUE__", embedded).replace("__NAMESPACE__", namespace)
    atomic_write_text(path, document, mode=0o600)


@dataclass(frozen=True)
class S6HumanReviewCapacity:
    full_train_natural_risk: int
    s3_reviewed_natural_risk: int
    s3_reviewed_clean: int
    full_train_unique_review_total: int
    full_validation_natural_risk: int = 5
    full_validation_unique_review_total: int = 0

    @property
    def remaining_natural_risk(self) -> int:
        return self.full_train_natural_risk - self.s3_reviewed_natural_risk

    @property
    def additional_review_capacity(self) -> int:
        return self.full_train_unique_review_total - (
            self.s3_reviewed_natural_risk + self.s3_reviewed_clean
        )

    @property
    def review_shortfall(self) -> int:
        return max(0, self.remaining_natural_risk - self.additional_review_capacity)

    @property
    def minimum_safe_train_unique_review_total(self) -> int:
        return self.full_train_natural_risk + self.s3_reviewed_clean

    @property
    def validation_review_shortfall(self) -> int:
        return max(
            0,
            self.full_validation_natural_risk
            - self.full_validation_unique_review_total,
        )

    @property
    def total_review_shortfall(self) -> int:
        return self.review_shortfall + self.validation_review_shortfall

    @property
    def minimum_safe_validation_unique_review_total(self) -> int:
        return self.full_validation_natural_risk

    def audit(self) -> dict[str, Any]:
        values = asdict(self)
        if any(not isinstance(value, int) or value < 0 for value in values.values()):
            raise ContractViolation("S6 human-review counts must be non-negative integers")
        if self.s3_reviewed_natural_risk > self.full_train_natural_risk:
            raise ContractViolation("S3 reviewed-risk count exceeds the full Train risk quota")
        if (
            self.s3_reviewed_natural_risk + self.s3_reviewed_clean
            > self.full_train_unique_review_total
        ):
            raise ContractViolation("S3 review count already exceeds the full Train review total")
        return {
            "full_train_natural_risk": self.full_train_natural_risk,
            "s3_reviewed_natural_risk": self.s3_reviewed_natural_risk,
            "s3_reviewed_clean": self.s3_reviewed_clean,
            "s3_reviewed_unique_total": self.s3_reviewed_natural_risk
            + self.s3_reviewed_clean,
            "remaining_natural_risk_requiring_positive_human_review": self.remaining_natural_risk,
            "additional_review_capacity_under_frozen_total": self.additional_review_capacity,
            "review_shortfall": self.review_shortfall,
            "minimum_safe_train_unique_review_total": self.minimum_safe_train_unique_review_total,
            "full_validation_natural_risk": self.full_validation_natural_risk,
            "full_validation_unique_review_total": self.full_validation_unique_review_total,
            "validation_review_shortfall": self.validation_review_shortfall,
            "minimum_safe_validation_unique_review_total": self.minimum_safe_validation_unique_review_total,
            "total_review_shortfall": self.total_review_shortfall,
            "minimum_safe_project_unique_review_total": (
                self.minimum_safe_train_unique_review_total
                + self.minimum_safe_validation_unique_review_total
                + 20
            ),
            "contract_conflict": self.total_review_shortfall > 0,
        }


def frozen_s6_human_review_capacity() -> dict[str, Any]:
    """Audit the literal S3/S6 counts without changing either frozen rule."""
    return S6HumanReviewCapacity(
        full_train_natural_risk=100,
        s3_reviewed_natural_risk=10,
        s3_reviewed_clean=10,
        full_train_unique_review_total=100,
        full_validation_natural_risk=5,
        full_validation_unique_review_total=0,
    ).audit()
