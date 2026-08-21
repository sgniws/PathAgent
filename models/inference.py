import os
import re
import json
import time
import torch
from PIL import Image

from copy import deepcopy
from qwen_vl_utils import process_vision_info
from data_processing.utils import extract_coords_from_name, build_descriptions_with_meta
from models.llm_backend import generate_chat_text, strip_thinking_block

PANCREATIC_EXECUTOR_SYSTEM_PROMPT = (
    "You are the Executor of a pancreatic histopathology WSI agent. "
    "The task metadata establishes that the specimen belongs to the pancreatic VQA cohort; do not demand patch-level "
    "proof of organ identity. Ignore any alternative organ guess in a patch description. "
    "You may use only the H&E patch evidence explicitly provided in the current state. "
    "Never assume that a pathology-report label is visually present, and never invent an organ, diagnosis, "
    "invasion focus, margin status, lymph-node status, immunohistochemistry result, or molecular result. "
    "For focal findings such as perineural or lymphovascular invasion, require direct supporting morphology. "
    "A diagnosis name or candidate-answer label inside a patch description is not visual evidence; use only concrete "
    "architecture, cytology, stroma, and spatial relationships stated for a cited patch. "
    "If a description contradicts itself, treat that patch as low-confidence evidence and inspect or retrieve another patch. "
    "If evidence is incomplete, retrieve more patches, inspect a named patch, zoom a named patch, or abstain. "
    "Use concise auditable evidence summaries and patch identifiers; do not output hidden reasoning or a thinking block."
)

GENERAL_EXECUTOR_SYSTEM_PROMPT = (
    "You are the Executor of a general histopathology whole-slide-image VQA agent. "
    "The specimen organ and diagnosis are not known unless they are directly supported by the supplied H&E patch evidence. "
    "You may use only the question, official answer choices, patch identifiers, coordinates, magnification metadata, "
    "and H&E morphology explicitly present in the current state. Never use a pathology report, answer key, patient outcome, "
    "immunohistochemistry, molecular result, treatment history, or dataset prior as visual evidence. "
    "A diagnosis name or candidate-answer label inside a patch description is not visual evidence; rely only on concrete "
    "architecture, cytology, stroma, and spatial relationships tied to cited patches. "
    "Distinguish not seen in one patch from absent on the whole slide, and distinguish not assessable from negative. "
    "For focal findings, require direct morphology and adequate search coverage. "
    "For report-only facts such as survival, receptor scores, exact gross size, pathologic stage, margin status/distance, "
    "patient metadata, specimen laterality, or report authorship, mark the H&E evidence "
    "insufficient even though the benchmark contract still requires a best-effort official choice. "
    "If useful visual evidence may still be found, retrieve, inspect, or zoom. If the requested fact is not visually "
    "establishable or the search budget is exhausted, abstain while preserving a separate best-effort benchmark answer. "
    "Use concise auditable evidence summaries and patch identifiers; do not output hidden reasoning or a thinking block."
)

ALLOWED_AGENT_ACTIONS = {"retrieve", "inspect", "zoom", "answer", "abstain"}
PATHO_MORPHOLOGY_PROMPT_VERSION = "pancreatic_morphology_v4"
FROZEN_PATHO_R1_CANVAS = (784, 784)


def prepare_patho_r1_canvas(image):
    """Return the frozen 784x784 Patho-R1 observation canvas."""
    if image.size == FROZEN_PATHO_R1_CANVAS:
        return image
    return image.resize(FROZEN_PATHO_R1_CANVAS, Image.Resampling.BICUBIC)

_REPORT_ONLY_QUESTION = re.compile(
    r"\b(?:"
    r"size|diameter|extent|stage|margin|"
    r"(?:tumou?r|mass|lesion|carcinoma).{0,30}(?:size|diameter|measurement)|"
    r"(?:size|diameter|measurement).{0,30}(?:tumou?r|mass|lesion|carcinoma)|"
    r"pathologic(?:al)?\s+stage|\btnm\b|\bpt[0-4]\b|\bpn[0-3]\b|"
    r"estrogen|progesterone|hormone\s+receptor|\bher-?2\b|\bki-?67\b|"
    r"immunohistochem\w*|immunostain\w*|nuclear\s+staining|"
    r"survival|follow[- ]?up|recurrence|treatment|therapy|neo[- ]?adjuvant|"
    r"patient(?:'s|’s)?\s+age|who\s+signed|pre[- ]?operative\s+diagnosis|"
    r"sections?\s+(?:are\s+)?included|which\s+breast"
    r")\b",
    flags=re.IGNORECASE,
)

_MORPHOLOGY_FIELDS = ("VISIBLE", "ABSENT_OR_NOT_SEEN", "UNCERTAIN", "QUALITY")
_MORPHOLOGY_TERMS = re.compile(
    r"\b(?:gland|duct|papill|solid|nest|trabec|cell|nucle|cytoplas|stroma|desmoplas|fibro|mucin|"
    r"necros|vessel|vascular|nerve|neural|inflamm|architecture|atyp|mito|artifact|hemorrhage)\w*\b",
    flags=re.IGNORECASE,
)
_ANCILLARY_TERMS = re.compile(
    r"\b(?:immunohistochem|molecular|mutation|clinical correlation|prognostic|CK7|CK20|CDX2|KRAS)\b",
    flags=re.IGNORECASE,
)
_DIAGNOSTIC_TERMS = re.compile(
    r"\b(?:pancreatic ductal adenocarcinoma|ductal adenocarcinoma|adenocarcinoma|carcinoma|"
    r"fibroadenoma|adenoma|cystadenoma|sarcoma|lymphoma|melanoma|"
    r"intraductal epithelial neoplasm|neuroendocrine neoplasm|solid pseudopapillary neoplasm|"
    r"neoplasm|tumou?r category|diagnosis|benign|malignant|malignancy)\b",
    flags=re.IGNORECASE,
)
_ORGAN_TERMS = re.compile(
    r"\b(?:pancreas|pancreatic|breast|mammary|salivary gland|gastric|stomach|colorectal|colon|"
    r"duodenal|duodenum|liver|hepatic|lung|pulmonary|prostate|ovarian|ovary)\b",
    flags=re.IGNORECASE,
)


def normalize_patho_r1_output(output_text, choices=None):
    visible_output = strip_thinking_block(output_text)
    match = re.fullmatch(r"\s*<answer>\s*([A-Z])\s*</answer>\s*", visible_output, flags=re.IGNORECASE)
    if match and choices:
        index = ord(match.group(1).upper()) - ord("A")
        if 0 <= index < len(choices):
            return f"<answer>{choices[index]}</answer>"
    return visible_output


def normalize_patho_morphology_output(
    output_text,
    forbidden_labels=None,
    prompt_version=PATHO_MORPHOLOGY_PROMPT_VERSION,
):
    """Keep Patho-R1's user-visible morphology while rejecting answer labels.

    Patho-R1 commonly places a long rationale inside ``<think>`` and a short
    user-visible result inside ``<answer>``.  The morphology protocol asks the
    model to put the observable findings in that final block.  A bare option
    letter or an exact VQA choice is treated as unusable visual evidence.
    """
    visible_output = strip_thinking_block(output_text).strip()
    answer_match = re.fullmatch(
        r"\s*<answer>\s*(.*?)\s*</answer>\s*",
        visible_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    morphology = answer_match.group(1).strip() if answer_match else visible_output
    labels = [str(label).strip() for label in forbidden_labels or [] if str(label).strip()]
    invalid_label = len(morphology) == 1 and morphology.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    invalid_label = invalid_label or any(morphology.casefold() == label.casefold() for label in labels)
    if invalid_label or not morphology:
        morphology = (
            "VISIBLE: No reliable morphology statement was returned.\n"
            "ABSENT_OR_NOT_SEEN: Not assessed.\n"
            "UNCERTAIN: The patch-level output was an answer label rather than visual evidence.\n"
            "QUALITY: Unusable for visual classification."
        )
    if not invalid_label and morphology:
        for label in sorted(labels, key=len, reverse=True):
            morphology = re.sub(re.escape(label), "[diagnostic label omitted]", morphology, flags=re.IGNORECASE)
        morphology = _DIAGNOSTIC_TERMS.sub("[diagnostic label omitted]", morphology)
        morphology = _ORGAN_TERMS.sub("[organ label omitted]", morphology)

        has_contract = all(re.search(rf"(?im)^\s*{field}\s*:", morphology) for field in _MORPHOLOGY_FIELDS)
        if not has_contract:
            visible_sentences = []
            for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", morphology):
                sentence = sentence.strip(" -*\t")
                if not sentence or _ANCILLARY_TERMS.search(sentence):
                    continue
                # Remove redacted label fragments before morphology extraction.
                # This preserves a safe suffix such as "with papillary
                # structures", while a label-only sentence becomes devoid of
                # morphology terms and is discarded below.
                sentence = sentence.replace("[diagnostic label omitted]", "")
                sentence = sentence.replace("[organ label omitted]", "")
                sentence = re.sub(
                    r"(?i)^\s*VISIBLE\s*:\s*(?:with|showing|characterized by|composed of)\s+",
                    "VISIBLE: ",
                    sentence,
                )
                if re.search(r"\b(?:consistent with|classification|rules? out|supports this)\b", sentence, re.I):
                    continue
                if _MORPHOLOGY_TERMS.search(sentence):
                    sentence = re.sub(r"^(?:key )?features include\s*", "", sentence, flags=re.I)
                    visible_sentences.append(sentence)
            visible = " ".join(visible_sentences) or "No reliable direct morphology statement was returned."
            morphology = (
                f"VISIBLE: {visible}\n"
                "ABSENT_OR_NOT_SEEN: Not reliably stated.\n"
                "UNCERTAIN: Patch-level assessment only; diagnosis and organ identity were not assessed.\n"
                "QUALITY: Not explicitly stated by the patch observer."
            )
        else:
            morphology = "\n".join(
                line for line in morphology.splitlines() if not _ANCILLARY_TERMS.search(line)
            )
    return f"[PATHO_MORPHOLOGY | {prompt_version}]\n{morphology}"


def sanitize_morphology_evidence_text(text, forbidden_labels=None):
    """Remove diagnostic, organ and ancillary claims from offline patch prose."""
    sanitized = str(text or "")
    for label in sorted(
        [str(label).strip() for label in forbidden_labels or [] if str(label).strip()],
        key=len,
        reverse=True,
    ):
        sanitized = re.sub(re.escape(label), "[diagnostic label omitted]", sanitized, flags=re.IGNORECASE)
    sanitized = _DIAGNOSTIC_TERMS.sub("[diagnostic label omitted]", sanitized)
    sanitized = _ORGAN_TERMS.sub("[organ label omitted]", sanitized)
    sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", sanitized):
        sentence = sentence.strip()
        if sentence and not _ANCILLARY_TERMS.search(sentence):
            sentences.append(sentence)
    return "[OFFLINE_MORPHOLOGY_SANITIZED] " + " ".join(sentences)


def _call_llm_return_json_simple(
    model,
    tokenizer,
    messages,
    max_new_tokens=256,
    retries=1,
    temperature=None,
    do_sample=None,
    top_p=None,
    seed=None,
    trace_recorder=None,
    operation="qwen_json",
    trace_context=None,
):
    """
    Call the qwen LLM and parse JSON.
    """
    def extract_json_block(s: str):
        """Extract the first complete { ... } block from the generated text (based on brace counting)."""
        start = s.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        return None

    effective_retries = 0 if getattr(model, "provider", None) == "deepseek" else retries
    for attempt in range(effective_retries + 1):
        current_messages = messages
        full_output = generate_chat_text(
            model,
            tokenizer,
            current_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            seed=seed,
            trace_recorder=trace_recorder,
            operation=f"{operation}_attempt_{attempt + 1}",
            trace_context=trace_context,
        )

        # Attempt to parse directly
        try:
            parsed = json.loads(full_output)
            return full_output, parsed
        except Exception:
            # Attempt to extract the first JSON block
            json_block = extract_json_block(full_output)
            if json_block:
                try:
                    parsed = json.loads(json_block)
                    return full_output, parsed
                except Exception:
                    pass

        # If failed, add a reminder prompt and try again
        if attempt < effective_retries:
            messages_retry = deepcopy(messages)
            messages_retry.append({
                "role": "user",
                "content": "Reminder: Respond with only a single valid JSON object containing the requested keys."
            })
            messages = messages_retry
            continue

        return full_output, None

    return None, None


def evaluate_pancreatic_vqa_action(
    model,
    tokenizer,
    description,
    question,
    choices=None,
    visible_patch_ids=None,
    remaining_patch_count=0,
    can_zoom=True,
    question_type=None,
    max_new_tokens=768,
    retries=1,
    temperature=0.0,
    top_p=1.0,
    seed=None,
    trace_recorder=None,
    trace_context=None,
):
    """Return one constrained PathAgent action without requesting a thinking chain."""
    choices_text = json.dumps(choices or [], ensure_ascii=False)
    state = {
        "question": question,
        "question_type": question_type,
        "choices": choices or [],
        "visible_patch_ids": visible_patch_ids or [],
        "remaining_patch_count": remaining_patch_count,
        "zoom_available": bool(can_zoom),
        "patch_evidence": description,
    }
    answer_contract = (
        "For a multiple_choice question, candidate_answer must be a JSON list containing exact choices. "
        if question_type == "multiple_choice"
        else "candidate_answer must be one exact choice. "
    )
    system_prompt = (
        PANCREATIC_EXECUTOR_SYSTEM_PROMPT
        + "\n"
        + answer_contract
        + "\nReturn exactly one JSON object with this schema:\n"
        + "{\n"
        + '  "candidate_answer": "one choice or Insufficient evidence",\n'
        + '  "evidence_refs": ["at most 8 patch identifiers"],\n'
        + '  "evidence_summary": "at most 3 brief sentences about visible evidence",\n'
        + '  "sufficient": true,\n'
        + '  "missing_evidence": "brief description or empty string",\n'
        + '  "next_action": {"type": "retrieve|inspect|zoom|answer|abstain", "query": "", '
          '"target_patches": [], "magnification": null},\n'
        + '  "action_reason": "brief auditable reason"\n'
        + "}\n"
        + "Never list more than 8 evidence_refs. Keep every text field concise so the JSON is complete. "
        + "Use answer only when current evidence directly supports one choice. Use abstain when the requested fact "
          "cannot be established from the available H&E evidence."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Choose the next action for this pancreatic WSI VQA state.\n"
                f"Valid choices: {choices_text}\n"
                f"State:\n{json.dumps(state, ensure_ascii=False)}"
            ),
        },
    ]
    raw, parsed = _call_llm_return_json_simple(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens,
        retries=retries,
        temperature=temperature,
        do_sample=temperature is not None and temperature > 0,
        top_p=top_p,
        seed=seed,
        trace_recorder=trace_recorder,
        operation="pancreatic_action_decision",
        trace_context=trace_context,
    )
    valid_action_json = isinstance(parsed, dict)
    if not valid_action_json:
        parsed = {
            "candidate_answer": "Insufficient evidence",
            "evidence_refs": [],
            "evidence_summary": "The Executor did not return valid structured output.",
            "sufficient": False,
            "missing_evidence": "valid Executor decision",
            "next_action": {"type": "abstain", "query": "", "target_patches": [], "magnification": None},
            "action_reason": "Invalid model output; stop conservatively.",
        }
    action = parsed.get("next_action")
    if isinstance(action, str):
        action = {"type": action}
    if not isinstance(action, dict):
        action = {}
    action_type = str(action.get("type", "")).strip().lower()
    if action_type not in ALLOWED_AGENT_ACTIONS:
        valid_action_json = False
        action_type = "answer" if parsed.get("sufficient") is True else "abstain"
    action.update(
        {
            "type": action_type,
            "query": str(action.get("query") or parsed.get("missing_evidence") or "").strip(),
            "target_patches": action.get("target_patches") if isinstance(action.get("target_patches"), list) else [],
            "magnification": action.get("magnification"),
        }
    )
    parsed["next_action"] = action
    parsed["evidence_refs"] = parsed.get("evidence_refs") if isinstance(parsed.get("evidence_refs"), list) else []
    parsed["parse_status"] = "valid_action_json" if valid_action_json else "invalid_action_json"
    parsed["raw_texts"] = {"action_raw": raw}
    return parsed


def build_general_executor_system_prompt(evidence_policy="model_v1"):
    """Build the exact versioned Executor template used for run hashing."""
    sufficiency_schema = (
        '  "evidence_sufficient": "true or false (advisory)",\n'
        if evidence_policy == "contract_v1"
        else '  "evidence_sufficient": false,\n'
    )
    sufficiency_instruction = (
        "Your evidence_sufficient field is an advisory morphology judgment; a separate deterministic public-contract "
        "verifier makes the final sufficiency and stopping decision. Evaluate both true and false explicitly and do not "
        "copy a default value from the schema. "
        if evidence_policy == "contract_v1"
        else ""
    )
    return (
        GENERAL_EXECUTOR_SYSTEM_PROMPT
        + "\nReturn exactly one JSON object with this schema:\n"
        + "{\n"
        + '  "provisional_recommendation": "one exact official choice",\n'
        + '  "ranked_differential": ["zero to three other exact official choices, most plausible first"],\n'
        + '  "advisory_evidence_state": {"candidate_evidence_found": false, '
          '"ready_for_pathologist_review": false},\n'
        + sufficiency_schema
        + '  "abstain_recommended": false,\n'
        + '  "unsupported_answer_reason": "brief reason or empty string",\n'
        + '  "evidence_refs": ["at most 8 currently visible patch identifiers"],\n'
        + '  "evidence_summary": "at most 3 brief sentences about direct visible evidence",\n'
        + '  "missing_evidence": "brief morphology target or empty string",\n'
        + '  "next_action": {"type": "retrieve|inspect|zoom|answer|abstain", "query": "", '
          '"target_patches": [], "magnification": null},\n'
        + '  "action_reason": "brief auditable reason"\n'
        + "}\n"
        + sufficiency_instruction
        + "provisional_recommendation must always be exactly one supplied official choice, including when evidence_sufficient is false. "
        + "ranked_differential may contain only other supplied choices and must not repeat provisional_recommendation. "
        + "The advisory evidence fields are model suggestions only and never override the local deterministic evaluator. "
        + "Never add an Insufficient evidence choice. evidence_sufficient is true only when cited H&E morphology directly "
          "supports provisional_recommendation. abstain_recommended must be true for a terminal abstain and false for a supported answer. "
        + "Use answer only with sufficient direct evidence. Use abstain when further visual search cannot establish the requested fact. "
        + "A retrieve, inspect, or zoom query must request visible morphology, not a diagnosis label or answer choice."
    )


def evaluate_general_vqa_action(
    model,
    tokenizer,
    description,
    question,
    choices=None,
    visible_patch_ids=None,
    remaining_patch_count=0,
    can_zoom=True,
    question_type=None,
    evidence_policy="model_v1",
    max_new_tokens=768,
    retries=1,
    temperature=0.0,
    top_p=1.0,
    seed=None,
    trace_recorder=None,
    trace_context=None,
):
    """Return one general-v2 action plus separate benchmark and evidence judgments."""
    valid_choices = [str(choice) for choice in choices or []]
    choices_text = json.dumps(valid_choices, ensure_ascii=False)
    state = {
        "question": question,
        "question_type": question_type or "single_choice",
        "choices": valid_choices,
        "visible_patch_ids": visible_patch_ids or [],
        "remaining_patch_count": remaining_patch_count,
        "zoom_available": bool(can_zoom),
        "patch_evidence": description,
    }
    system_prompt = build_general_executor_system_prompt(evidence_policy)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Choose the next action for this general WSI VQA state.\n"
                f"Official choices: {choices_text}\n"
                f"State:\n{json.dumps(state, ensure_ascii=False)}"
            ),
        },
    ]
    raw, parsed = _call_llm_return_json_simple(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens,
        retries=retries,
        temperature=temperature,
        do_sample=temperature is not None and temperature > 0,
        top_p=top_p,
        seed=seed,
        trace_recorder=trace_recorder,
        operation="general_v2_action_decision",
        trace_context=trace_context,
    )
    valid_action_json = isinstance(parsed, dict)
    semantic_repairs = []
    fallback_answer = valid_choices[0] if valid_choices else ""
    if not valid_action_json:
        parsed = {
            "provisional_recommendation": fallback_answer,
            "ranked_differential": [],
            "advisory_evidence_state": {
                "candidate_evidence_found": False,
                "ready_for_pathologist_review": False,
            },
            "evidence_sufficient": False,
            "abstain_recommended": True,
            "unsupported_answer_reason": "The Executor did not return valid structured output.",
            "evidence_refs": [],
            "evidence_summary": "No valid Executor evidence judgment was returned.",
            "missing_evidence": "valid Executor decision",
            "next_action": {"type": "abstain", "query": "", "target_patches": [], "magnification": None},
            "action_reason": "Invalid model output; stop conservatively.",
        }

    benchmark_answer = parsed.get("provisional_recommendation")
    if benchmark_answer is None:
        # Historical/local test backends may still emit the pre-v1 field.
        benchmark_answer = parsed.get("benchmark_answer")
    if benchmark_answer not in valid_choices:
        benchmark_answer = fallback_answer
        valid_action_json = False
    parsed["provisional_recommendation"] = benchmark_answer
    parsed["benchmark_answer"] = benchmark_answer
    parsed["candidate_answer"] = benchmark_answer
    raw_differential = parsed.get("ranked_differential")
    if not isinstance(raw_differential, list):
        if raw_differential is not None:
            semantic_repairs.append("invalid_ranked_differential_changed_to_empty")
        raw_differential = []
    ranked_differential = []
    invalid_differential = False
    for value in raw_differential:
        if (
            not isinstance(value, str)
            or value not in valid_choices
            or value == benchmark_answer
            or value in ranked_differential
        ):
            invalid_differential = True
            continue
        ranked_differential.append(value)
    if invalid_differential:
        semantic_repairs.append("invalid_ranked_differential_entries_removed")
    parsed["ranked_differential"] = ranked_differential[:3]
    advisory = parsed.get("advisory_evidence_state")
    if not isinstance(advisory, dict):
        advisory = {}
    parsed["advisory_evidence_state"] = {
        "candidate_evidence_found": advisory.get("candidate_evidence_found") is True,
        "ready_for_pathologist_review": advisory.get(
            "ready_for_pathologist_review"
        )
        is True,
        "strict_evidence_sufficient": parsed.get("evidence_sufficient") is True,
    }
    parsed["evidence_sufficient"] = parsed.get("evidence_sufficient") is True
    parsed["sufficient"] = parsed["evidence_sufficient"]
    parsed["abstain_recommended"] = parsed.get("abstain_recommended") is True
    parsed["unsupported_answer_reason"] = str(parsed.get("unsupported_answer_reason") or "").strip()

    action = parsed.get("next_action")
    if isinstance(action, str):
        action = {"type": action}
    if not isinstance(action, dict):
        action = {}
    action_type = str(action.get("type", "")).strip().lower()
    if action_type not in ALLOWED_AGENT_ACTIONS:
        valid_action_json = False
        action_type = "answer" if parsed["evidence_sufficient"] else "abstain"
    if action_type == "answer" and not parsed["evidence_sufficient"]:
        semantic_repairs.append("unsupported_answer_changed_to_abstain")
        action_type = "abstain"
        parsed["abstain_recommended"] = True
    if (
        action_type in {"retrieve", "inspect", "zoom"}
        and not parsed["evidence_sufficient"]
        and _REPORT_ONLY_QUESTION.search(str(question or ""))
    ):
        # These targets require report/gross/ancillary data rather than a more
        # exhaustive H&E patch search. Enforce the protocol's own boundary so
        # the Executor cannot spend rounds looking for nonexistent rulers,
        # TNM labels, receptor scores, outcomes, or treatment metadata.
        semantic_repairs.append("report_only_search_changed_to_abstain")
        action_type = "abstain"
        parsed["abstain_recommended"] = True
        parsed["unsupported_answer_reason"] = (
            parsed["unsupported_answer_reason"]
            or "The requested fact requires report, gross, ancillary, or clinical evidence not available from H&E patches."
        )
    if action_type == "abstain":
        parsed["evidence_sufficient"] = False
        parsed["sufficient"] = False
        parsed["abstain_recommended"] = True
    action.update(
        {
            "type": action_type,
            "query": str(action.get("query") or parsed.get("missing_evidence") or "").strip(),
            "target_patches": action.get("target_patches") if isinstance(action.get("target_patches"), list) else [],
            "magnification": action.get("magnification"),
        }
    )
    parsed["next_action"] = action
    parsed["evidence_refs"] = parsed.get("evidence_refs") if isinstance(parsed.get("evidence_refs"), list) else []
    parsed["semantic_repairs"] = semantic_repairs
    parsed["parse_status"] = (
        "normalized_action_json" if valid_action_json and semantic_repairs
        else "valid_action_json" if valid_action_json
        else "invalid_action_json"
    )
    parsed["raw_texts"] = {"action_raw": raw}
    return parsed

def evaluate_with_llm_chain(model, tokenizer, description, question, choices=None,
                               max_new_tokens_a=512, max_new_tokens_b=256, max_new_tokens_c=256,
                               retries=1):
    """
    Three-step logic chain (simplified):
      Step A: Generate answer + thinking_steps
      Step B: Judge if sufficient (Yes / No)
      Step C: If insufficient, analyze missing info and zoom strategy
    """
    # --- Step A ---
    system_a = (
        "You are an expert AI pathology assistant.\n"
        "Task: Based on the patch descriptions, try to answer the question step-by-step.\n"
        "Output ONLY a JSON object:\n"
        "{\n"
        '  "answer": "the final predicted answer (string)",\n'
        '  "thinking_steps": "your detailed reasoning, step-by-step (string)"\n'
        "}"
    )

    choices_text = f"\nChoices: {choices}" if choices else ""
    user_a = f"--- Patch Descriptions ---\n{description}\n--- End ---\nQuestion: {question}{choices_text}\nNow output the JSON with 'answer' and 'thinking_steps'."

    messages_a = [
        {"role": "system", "content": system_a},
        {"role": "user", "content": user_a},
    ]
    full_a, parsed_a = _call_llm_return_json_simple(model, tokenizer, messages_a,
                                                   max_new_tokens=max_new_tokens_a, retries=retries)

    if parsed_a is None:
        parsed_a = {
            "answer": "Uncertain",
            "thinking_steps": full_a or ""
        }

    parsed_a.setdefault("answer", "Uncertain")
    parsed_a.setdefault("thinking_steps", "")

    # --- Step B ---
    system_b = (
        "You are an expert AI pathology assistant.\n"
        "Task: Judge whether the current patch descriptions are sufficient to confidently support the answer.\n"
        "Output ONLY a JSON object like:\n"
        '{"sufficient": "Yes" or "No" }'
    )
    user_b = (
        f"Descriptions:\n{description}\n\n"
        f"Question: {question}{choices_text}\n\n"
        f"Previous answer and reasoning:\n{json.dumps(parsed_a, ensure_ascii=False)}\n\n"
        "Return JSON only."
    )
    messages_b = [{"role": "system", "content": system_b}, {"role": "user", "content": user_b}]
    full_b, parsed_b = _call_llm_return_json_simple(model, tokenizer, messages_b,
                                                   max_new_tokens=max_new_tokens_b, retries=retries)

    if parsed_b is None:
        parsed_b = {"sufficient": "Uncertain"}

    suff = parsed_b.get("sufficient", "").strip().lower()

    # --- If sufficient == "yes", return immediately ---
    if suff == "yes":
        return {
            "answer": parsed_a.get("answer"),
            "thinking_steps": parsed_a.get("thinking_steps"),
            "sufficient": parsed_b.get("sufficient", ""),
            "raw_texts": {
                "step_a_raw": full_a,
                "step_b_raw": full_b,
                "step_c_raw": None
            }
        }

    # --- Step C ---
    system_c = (
        "You are an expert AI pathology assistant.\n"
        "Task: If current data is insufficient, specify what visual evidence is missing, "
        "and whether zooming in could help obtain that evidence.\n"
        "Output ONLY a JSON object like:\n"
        "{\n"
        '  "missing_info": "short noun phrase",\n'
        '  "zoom_recommendation": "Yes" or "No",\n'
        '  "recommended_zoom_level": "None" or an integer like 10 or 20 or 40,\n'
        '  "zoom_reason": "brief reason why zooming helps"\n'
        "}"
    )

    user_c = (
        f"Descriptions:\n{description}\n\n"
        f"Question: {question}{choices_text}\n\n"
        f"Previous answer: {json.dumps(parsed_a, ensure_ascii=False)}\n"
        f"Sufficiency judgement: {json.dumps(parsed_b, ensure_ascii=False)}\n\n"
        "Now provide the JSON for missing info and zoom recommendation."
    )
    messages_c = [{"role": "system", "content": system_c}, {"role": "user", "content": user_c}]
    full_c, parsed_c = _call_llm_return_json_simple(model, tokenizer, messages_c,
                                                   max_new_tokens=max_new_tokens_c, retries=retries)

    if parsed_c is None:
        parsed_c = {
            "missing_info": "Uncertain",
            "zoom_recommendation": "Uncertain",
            "recommended_zoom_level": "Uncertain",
            "zoom_reason": full_c or ""
        }

    return {
        "answer": parsed_a.get("answer"),
        "thinking_steps": parsed_a.get("thinking_steps"),
        "sufficient": parsed_b.get("sufficient"),
        "missing_info": parsed_c.get("missing_info"),
        "zoom_recommendation": parsed_c.get("zoom_recommendation"),
        "zoom_level": parsed_c.get("recommended_zoom_level", 5),
        "zoom_reason": parsed_c.get("zoom_reason"),
        "raw_texts": {
            "step_a_raw": full_a,
            "step_b_raw": full_b,
            "step_c_raw": full_c
        }
    }

def slide_llm_answer(
    model,
    tokenizer,
    descriptions_text,
    question,
    choices=None,
    magnification=None,
    case_name=None,
    question_type=None,
    temperature=0.0,
    top_p=1.0,
    seed=None,
    trace_recorder=None,
    trace_context=None,
):
    """
    Slide-level LLM: Generates the final answer based on multiple patch descriptions.
    Optimized the prompt to focus the model on specific slide-level results rather than conceptual explanations.
    Automatically repairs JSON format errors in the model output.
    """
    answer_contract = (
        "For multiple_choice, answer must be a JSON list containing only exact provided choices. "
        if question_type == "multiple_choice"
        else "The answer must exactly match one provided choice. "
    )
    system_prompt = (
        PANCREATIC_EXECUTOR_SYSTEM_PROMPT
        + "\nProduce the final pancreatic WSI VQA response. "
        + answer_contract
        + "Cite only patch identifiers present in the evidence. If the visible evidence is inadequate, choose the "
          "provided Insufficient evidence option. Respond strictly as JSON with keys answer, evidence_refs, "
          "explanation, and confidence. Include at most five evidence_refs and keep the explanation under 80 words. "
          "Do not output a thinking block."
    )

    if magnification is not None:
        system_prompt = f"[Slide-level Context | Magnification={magnification}x]\n" + system_prompt

    choices_text = f"\nChoices: {choices}" if choices else ""

    user_prompt = (
        f"Question: {question}{choices_text}\n\n"
        "Now, based on the following patch-level descriptions of the slide, "
        "determine the **slide-level result** that directly answers the question.\n\n"
        f"--- Patch Descriptions ---\n{descriptions_text}\n--- End of Descriptions ---\n\n"
        "Answer in JSON format:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_answer = generate_chat_text(
        model,
        tokenizer,
        messages,
        max_new_tokens=512,
        temperature=temperature,
        do_sample=temperature is not None and temperature > 0,
        top_p=top_p,
        seed=seed,
        trace_recorder=trace_recorder,
        operation="pancreatic_final_answer",
        trace_context=trace_context,
    )

    # ------------------- JSON Extraction and Repair Logic -------------------
    parsed = None
    json_candidates = re.findall(r"\{.*?\}", raw_answer, flags=re.DOTALL)

    for candidate in reversed(json_candidates):  # Parse the last one first
        try:
            parsed = json.loads(candidate)
            break
        except Exception:
            continue

    if parsed is None:
        print(f"JSON parsing failed ({case_name if case_name else 'unknown_case'}), raw output:\n{raw_answer}\n")
        recovered = None
        answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', raw_answer)
        if answer_match and choices:
            candidate = answer_match.group(1).strip()
            recovered = next((choice for choice in choices if str(choice).strip() == candidate), None)
        insufficient = next(
            (
                choice
                for choice in choices or []
                if "insufficient" in str(choice).lower() or "证据不足" in str(choice)
            ),
            None,
        )
        answer = {
            "answer": recovered or insufficient or (choices[-1] if choices else "Unknown"),
            "explanation": (
                "The exact choice was recovered from a truncated JSON response; detailed evidence fields were discarded."
                if recovered
                else "The model did not return valid JSON, so the safe fallback choice was used."
            ),
            "evidence_refs": [],
            "confidence": 0.0,
            "raw_output": raw_answer,
            "parse_status": "recovered_exact_choice" if recovered else "safe_fallback",
        }
    else:
        answer_value = parsed.get("answer", "")
        if question_type == "multiple_choice":
            candidates = answer_value if isinstance(answer_value, list) else [answer_value] if answer_value else []
            answer_text = [choice for choice in candidates if choice in (choices or [])]
        else:
            answer_text = str(answer_value).strip()
            if choices and answer_text not in choices:
                answer_text = next(
                    (
                        choice
                        for choice in choices
                        if "insufficient" in str(choice).lower() or "证据不足" in str(choice)
                    ),
                    choices[-1],
                )
        if not answer_text:
            if question_type == "multiple_choice":
                answer_text = []
            else:
                answer_text = next(
                    (
                        choice
                        for choice in choices or []
                        if "insufficient" in str(choice).lower() or "证据不足" in str(choice)
                    ),
                    choices[-1] if choices else "Unknown",
                )

        explanation_text = parsed.get("explanation", "").strip()
        if not explanation_text:
            explanation_text = "No explanation provided."

        answer = {
            "answer": answer_text,
            "explanation": explanation_text,
            "evidence_refs": parsed.get("evidence_refs", []) if isinstance(parsed.get("evidence_refs"), list) else [],
            "confidence": parsed.get("confidence"),
            "raw_output": raw_answer,
            "parse_status": "valid_json",
        }

    return answer

def patho_r1_describe(image, question=None,
                    patho_r1_processor=None, patho_r1_model=None,
                    coords=None, magnification=None, max_new_tokens=1024, choices=None, missing_info=None,
                    trace_recorder=None, trace_context=None, patch_id=None, operation="patho_r1_describe",
                    morphology_only=False, inspection_focus=None,
                    prompt_version=PATHO_MORPHOLOGY_PROMPT_VERSION):

    image = prepare_patho_r1_canvas(image)

    meta_lines = []
    if coords is not None:
        meta_lines.append(f"Patch coordinates: ({coords[0]},{coords[1]})")
    if magnification is not None:
        meta_lines.append(f"Magnification: {magnification}x")
    meta_text = ""
    if meta_lines:
        meta_text = "[IMAGE META] " + " | ".join(meta_lines) + "\n\n"


    if morphology_only:
        focus = inspection_focus or question or "Identify the most informative visible histologic architecture."
        prompt_body = (
            "This is a visual transcription task, not a diagnostic task. Describe only morphology directly visible in this H&E patch. "
            "Do not choose an answer, name a disease, name a tumor category, infer an organ, or repeat diagnostic labels. "
            "Do not discuss immunohistochemistry, molecular testing, treatment, prognosis, or clinical correlation. "
            "Separate observed findings from absent and uncertain findings. "
            "Pay attention to gland formation, infiltration, papillary or pseudopapillary architecture, solid growth, "
            "stroma, mucin, necrosis, vessels, nerves, cytologic atypia, mitoses, inflammation, artifacts, and image quality "
            "when they are actually assessable.\n"
            f"Inspection focus: {focus}\n"
            "Return the final user-visible result inside <answer>...</answer> using exactly these four fields:\n"
            "VISIBLE: concise directly observed morphology\n"
            "ABSENT_OR_NOT_SEEN: relevant features not identified in this patch\n"
            "UNCERTAIN: limitations or features that cannot be assessed\n"
            "QUALITY: adequate, limited, or unusable with a brief reason"
        )
    elif question is None:
        prompt_body = (
            "Describe only visible histopathology morphology in this H&E pathology patch. "
            "Do not name or guess any organ, anatomic site, disease, diagnosis, tumor type, "
            "cell lineage, or specialized cell type. "
            "Do not infer biological function or clinical meaning. "
            "Do not copy category names from the prompt; mention a feature only if it is clearly visible. "
            "If a feature is absent or uncertain, do not list it as present. "
            "Write one or two concise sentences describing only observed architecture, cell density, stroma, "
            "fat spaces, vessels, staining, and artifacts when visible. "
            "Return exactly three short lines: visible_features; uncertainty_or_limitations; excluded_inferences. "
            "For excluded_inferences, state that organ, site, diagnosis, tumor type, lineage, and specialized cell type are not assessed from this patch."
        )
    else:
        prompt_body = (
            f"Question: {question}\n\n"
            "Answer the question and list the pathological features visible in the image that support your answer."
        )

    if choices is not None and not morphology_only:
        enumerated_choices = "\n".join(
            f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(choices)
        )
        prompt_body += (
            "\nChoices:\n"
            + enumerated_choices
            + "\nReturn the exact full choice text in the answer, not only its letter."
        )
    if missing_info is not None and not morphology_only:
        prompt_body += f"\nMissing information: {missing_info}"

    full_text = meta_text + prompt_body

    # === Construct system prompt ===
    system_prompt = (
        "A conversation between a curious user and an AI medical assistant specialized in pathology image analysis. "
        "The assistant can interpret pathology images, describe observed features, and provide possible explanations based on medical knowledge, "
        "but will never give a definitive diagnosis or prescribe treatment. "
        "The assistant must always maintain a polite, clear, and professional tone. "
        "All answers should be supported by established, reliable medical sources. "
        "The assistant should carefully consider visual details in pathology images, such as cell morphology, staining patterns, and tissue architecture. "
        "If choices are given, the answer must be given from the choices."
        "If Missing information is given, you need to focus on this part of the image."
    )
    if morphology_only:
        system_prompt = (
            "You are a pathology image morphology recorder. Report only directly visible H&E morphology and image quality. "
            "Do not provide a diagnosis, organ prediction, tumor category, candidate answer, clinical interpretation, "
            "immunohistochemistry, or molecular claim. Follow the requested four-field output contract exactly."
        )

    # === Construct message input ===
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_text},
            ],
        },
    ]

    context = trace_context or {}
    call_id = None
    started_at = time.time()
    if trace_recorder is not None:
        call_id = trace_recorder.before_call(
            "patho_r1",
            operation,
            {
                "patch_id": patch_id,
                "coords": coords,
                "magnification": magnification,
                # In morphology-only mode the downstream model never receives
                # the benchmark question or choices. Keep the trace aligned
                # with that actual model boundary as well.
                "question": None if morphology_only else question,
                "choices": None if morphology_only else choices,
                "missing_info": missing_info,
                "morphology_only": morphology_only,
                "inspection_focus": inspection_focus,
                "prompt_version": prompt_version,
                "max_new_tokens": max_new_tokens,
                "observation_canvas": list(FROZEN_PATHO_R1_CANVAS),
                "prompt": full_text,
            },
            step_id=context.get("step_id"),
            attempt=context.get("attempt"),
        )
    try:
        text = patho_r1_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = patho_r1_processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(patho_r1_model.device)
        with torch.no_grad():
            generated_ids = patho_r1_model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = patho_r1_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        visible_output = (
            normalize_patho_morphology_output(
                output_text,
                forbidden_labels=choices,
                prompt_version=prompt_version,
            )
            if morphology_only
            else normalize_patho_r1_output(output_text, choices=choices)
        )
        del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
        torch.cuda.empty_cache()
        if trace_recorder is not None:
            trace_recorder.after_call(
                call_id,
                "patho_r1",
                operation,
                {
                    "raw_output": output_text,
                    "returned_output": visible_output,
                    "latency_ms": round((time.time() - started_at) * 1000),
                },
                step_id=context.get("step_id"),
                attempt=context.get("attempt"),
            )
        return visible_output
    except Exception as exc:
        torch.cuda.empty_cache()
        if trace_recorder is not None:
            trace_recorder.after_call(
                call_id,
                "patho_r1",
                operation,
                {"latency_ms": round((time.time() - started_at) * 1000)},
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                step_id=context.get("step_id"),
                attempt=context.get("attempt"),
            )
        raise

def summarize_patches_in_chunks(
    model, tokenizer, descriptions_dict, patch_names,
    question_text=None, chunk_size=5, threshold=50, magnification=None,
    temperature=0.0, top_p=1.0, seed=None, trace_recorder=None, trace_context=None,
):
    """
    If the number of patches exceeds the threshold, summarize descriptions in chunks of `chunk_size`.
    The summary for each chunk will explicitly list the included patch coordinates and conduct a guided summary based on the question.
    """
    if len(patch_names) <= threshold:
        # Does not exceed threshold, concatenate original descriptions directly
        items = [(name, descriptions_dict[name]) for name in patch_names]
        return build_descriptions_with_meta(
            items, mag_level=magnification, include_header=True, include_coords=True
        )

    print(f"Patch count {len(patch_names)} exceeds {threshold}, performing chunked summarization...")
    summaries = []
    for i in range(0, len(patch_names), chunk_size):
        chunk_names = patch_names[i:i+chunk_size]
        items = [(name, descriptions_dict[name]) for name in chunk_names]

        # Extract coordinate list
        coords_list = []
        for name, _ in items:
            x, y = extract_coords_from_name(name)
            coords_list.append(f"({x},{y})" if x is not None and y is not None else "(unknown)")
        coords_str = ", ".join(coords_list)

        # Concatenate patch descriptions (with coordinates)
        chunk_text = build_descriptions_with_meta(
            items, mag_level=magnification, include_header=False, include_coords=True
        )

        # === Construct Prompt ===
        system_prompt = (
            "You are an expert pathology assistant. "
            "You will be given multiple patch-level descriptions of histopathology images. "
            "Your task is to summarize the key pathological features across these patches. "
            "The summary must be concise yet informative, highlighting significant morphological patterns. "
            "Additionally, focus on information that could help answer the following question "
            "about the slide, emphasizing details relevant to the diagnostic or interpretive context."
        )

        if magnification is not None:
            system_prompt = f"[Patch-level Summarization | Magnification={magnification}x]\n" + system_prompt

        user_prompt = (
            f"--- Patch Descriptions (with coordinates) ---\n{chunk_text}\n"
            "--- End of Descriptions ---\n\n"
        )

        if question_text:
            user_prompt += f"Related Question: {question_text}\n\n"

        user_prompt += (
            "Summarize the main pathological findings across these patches. "
            "Your summary should:\n"
            "- Capture key morphological features and cellular details.\n"
            "- If a question is provided, emphasize features that are relevant to answering it.\n"
            "- Do not provide a final answer or diagnosis.\n"
            "Output only the summary text, no JSON or extra formatting."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        summary_text = generate_chat_text(
            model,
            tokenizer,
            messages,
            max_new_tokens=512,
            temperature=temperature,
            do_sample=temperature is not None and temperature > 0,
            top_p=top_p,
            seed=seed,
            trace_recorder=trace_recorder,
            operation=f"pancreatic_patch_summary_chunk_{i//chunk_size+1}",
            trace_context=trace_context,
        )

        summaries.append(
            f"[Chunk {i//chunk_size+1} | Patches={coords_str}]\n{summary_text}"
        )

        print(f"Completed summary for chunk {i//chunk_size+1} (patch count: {len(chunk_names)})")

    # Concatenate summaries of all chunks
    combined_summary = (
        f"[Current Magnification: {magnification}x]\n\n" +
        "\n\n".join(summaries)
    )
    return combined_summary
