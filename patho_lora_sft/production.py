from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from .common import ContractViolation, canonical_json
from .constrained import constrained_generation_kwargs
from .schema import parse_wrapped_findings


PROJECT_TASK = "schema_morphology_v1"
OUTPUT_SCHEMA_VERSION = "schema_morphology_v1"
MAX_NEW_TOKENS = 192
DEVICE = "cuda:0"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolation(message)


@dataclass(frozen=True)
class ProductionResponse:
    """The only public response representation for the S9 production candidate."""

    findings: tuple[str, ...]

    def as_object(self) -> dict[str, list[str]]:
        return {"findings": list(self.findings)}

    def as_bytes(self) -> bytes:
        return canonical_json(self.as_object()).encode("utf-8")


@dataclass(frozen=True)
class InternalConstrainedGeneration:
    """Private execution evidence; raw wrapped text is never a production response."""

    raw_wrapped_output: str
    response: ProductionResponse


def validate_project_request(*, task: str) -> None:
    require(task == PROJECT_TASK, f"S9 production candidate only serves {PROJECT_TASK}")


def load_r16_production_candidate(*, base_path: Path, adapter_path: Path) -> tuple[Any, Any]:
    """Load the frozen read-only base with the separate r16 adapter enabled.

    There is intentionally no task or magnification router and no merged-model path.
    """

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    require(torch.cuda.is_available(), "CUDA is required for the S9 production candidate")
    require(torch.cuda.is_bf16_supported(), "BF16 is required for the S9 production candidate")
    require(base_path.is_dir() and adapter_path.is_dir(), "Frozen base or r16 adapter is missing")
    processor = AutoProcessor.from_pretrained(base_path, local_files_only=True, use_fast=False)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    base.to(DEVICE)
    base.eval()
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False, local_files_only=True)
    model.eval()
    require(getattr(model, "active_adapter", None) == "default", "r16 adapter is not active after load")
    return model, processor


def generate_constrained(
    *,
    model: Any,
    processor: Any,
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> InternalConstrainedGeneration:
    """Generate through the frozen grammar and expose only a canonical JSON object."""

    import torch

    require(image.mode == "RGB" and image.size == (784, 784), "Production input must be 784x784 RGB")
    require(max_new_tokens == MAX_NEW_TOKENS, "S9 production max_new_tokens changed")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "anonymous_project_patch"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=False)
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        moved[key] = value.to(DEVICE, dtype=torch.bfloat16) if key == "pixel_values" else value.to(DEVICE)
    prompt_length = int(moved["input_ids"].shape[1])
    kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        **constrained_generation_kwargs(processor.tokenizer, [prompt_length]),
    }
    with torch.inference_mode():
        output = model.generate(**moved, **kwargs)
    raw = processor.batch_decode(
        output[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    parsed = parse_wrapped_findings(raw)
    require(parsed.valid, f"Constrained production output failed Schema: {parsed.error}")
    response = ProductionResponse(tuple(parsed.findings))
    require(json.loads(response.as_bytes()) == response.as_object(), "Canonical production JSON is not decodable")
    return InternalConstrainedGeneration(raw_wrapped_output=raw, response=response)


def production_contract() -> Mapping[str, Any]:
    return {
        "schema_version": "patho_lora_s9_production_contract_v1",
        "project_task": PROJECT_TASK,
        "adapter_policy": "r16_always_enabled_for_every_project_request",
        "conditional_task_or_magnification_router": False,
        "routing_experiments": False,
        "production_output": "canonical_constrained_json_object_only",
        "native_production_output_path": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy_prefix_constrained",
        "adapter_merged": False,
        "base_read_only": True,
    }
