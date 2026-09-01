from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image

from .common import ContractViolation
from .constrained import constrained_generation_kwargs
from .schema import parse_wrapped_findings


@dataclass(frozen=True)
class SchemaGeneration:
    raw_output: str
    decoding_path: str
    metric_name: str
    schema_valid: bool
    schema_error: str | None
    findings: tuple[str, ...]


class PathoSchemaGenerator:
    """Native and grammar-constrained local Patho-R1 generation paths."""

    def __init__(self, *, model: Any, processor: Any, vision_info_fn: Callable[[list[dict[str, Any]]], tuple[Any, Any]] | None = None) -> None:
        self.model = model
        self.processor = processor
        if vision_info_fn is None:
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError as exc:
                raise ContractViolation("qwen_vl_utils is required for local Patho-R1 image generation") from exc
            vision_info_fn = process_vision_info
        self.vision_info_fn = vision_info_fn

    def generate(
        self,
        *,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        constrained: bool,
        max_new_tokens: int = 192,
    ) -> SchemaGeneration:
        if image.mode != "RGB" or image.size != (784, 784):
            raise ContractViolation("Patho schema generation requires frozen 784x784 RGB input")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.vision_info_fn(messages)
        inputs = self.processor(
            text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        if not hasattr(inputs, "input_ids"):
            raise ContractViolation("Processor output does not expose input_ids")
        prompt_length = int(inputs.input_ids.shape[1])
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)
        generation = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "use_cache": True,
        }
        path = "constrained" if constrained else "native"
        if constrained:
            generation.update(constrained_generation_kwargs(self.processor.tokenizer, [prompt_length]))
        output_ids = self.model.generate(**inputs, **generation)
        generated_ids = output_ids[:, prompt_length:]
        raw = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        parsed = parse_wrapped_findings(raw)
        return SchemaGeneration(
            raw_output=raw,
            decoding_path=path,
            metric_name="deployed_schema_valid" if constrained else "native_schema_valid",
            schema_valid=parsed.valid,
            schema_error=parsed.error,
            findings=parsed.findings,
        )
