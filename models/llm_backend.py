from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


class UnsupportedModelError(RuntimeError):
    """Raised when the selected checkpoint cannot be loaded by the chosen backend."""


class ExecutorAPIError(RuntimeError):
    """Raised when a remote Executor call cannot produce a usable response."""


class ExecutorContextBudgetExceeded(RuntimeError):
    """Raised when the serialized Executor request exceeds the configured hard limit."""


class ExecutorCostBudgetExceeded(RuntimeError):
    """Raised before a request that would continue after the configured cost budget was exhausted."""


DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
DEEPSEEK_PRICES_RMB_PER_MILLION = {
    "prompt_cache_hit": 0.02,
    "prompt_cache_miss": 1.0,
    "completion": 2.0,
}


def read_env_file(path: str | os.PathLike[str]) -> Dict[str, str]:
    """Read a small KEY=VALUE file without mutating the process environment."""
    values: Dict[str, str] = {}
    env_path = os.fspath(path)
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Executor environment file not found: {env_path}")
    with open(env_path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                raise ValueError(f"Invalid api.env entry on line {line_number}: expected KEY=VALUE")
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value[:1] in {"\"", "'"} and value[-1:] == value[:1]:
                value = value[1:-1]
            values[key] = value
    return values


def _deepseek_cost_rmb(usage: Dict[str, Any]) -> float:
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    hit_tokens = int(usage.get("prompt_cache_hit_tokens") or 0)
    miss_value = usage.get("prompt_cache_miss_tokens")
    miss_tokens = int(miss_value) if miss_value is not None else max(0, prompt_tokens - hit_tokens)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return (
        hit_tokens * DEEPSEEK_PRICES_RMB_PER_MILLION["prompt_cache_hit"]
        + miss_tokens * DEEPSEEK_PRICES_RMB_PER_MILLION["prompt_cache_miss"]
        + completion_tokens * DEEPSEEK_PRICES_RMB_PER_MILLION["completion"]
    ) / 1_000_000


def _read_config(checkpoint_path: str) -> Dict[str, Any]:
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_checkpoint_family(checkpoint_path: str) -> str:
    config = _read_config(checkpoint_path)
    model_type = str(config.get("model_type", "")).lower()
    architectures = [str(x).lower() for x in config.get("architectures", [])]

    if model_type == "qwen3_5" or any("qwen3_5" in x for x in architectures):
        return "qwen3_5"
    if model_type:
        return model_type
    return "unknown"


def strip_thinking_block(text: str) -> str:
    return re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def generate_chat_text(
    model,
    tokenizer,
    messages,
    max_new_tokens,
    temperature=None,
    do_sample=None,
    top_p=None,
    seed=None,
    trace_recorder=None,
    operation="qwen_generate",
    trace_context=None,
):
    context = trace_context or {}
    request = {
        "messages": messages,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "top_p": top_p,
            "seed": seed,
            "enable_thinking": False,
        },
    }
    call_id = None
    started_at = time.time()
    trace_component = getattr(model, "trace_component", "qwen_executor")
    if trace_recorder is not None:
        call_id = trace_recorder.before_call(
            trace_component,
            operation,
            request,
            step_id=context.get("step_id"),
            attempt=context.get("attempt"),
        )

    try:
        if hasattr(model, "generate_messages"):
            backend_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
            }
            if top_p is not None:
                backend_kwargs["top_p"] = top_p
            if seed is not None:
                backend_kwargs["seed"] = seed
            output_text = model.generate_messages(messages, **backend_kwargs)
            if trace_recorder is not None:
                response_metadata = getattr(model, "last_response_metadata", {}) or {}
                trace_recorder.after_call(
                    call_id,
                    trace_component,
                    operation,
                    {
                        "raw_output": output_text,
                        "latency_ms": round((time.time() - started_at) * 1000),
                        **response_metadata,
                    },
                    step_id=context.get("step_id"),
                    attempt=context.get("attempt"),
                )
            return output_text

        import torch

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            try:
                model_inputs = tokenizer(text=[text], return_tensors="pt").to(model.device)
            except TypeError:
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generate_kwargs = {"max_new_tokens": max_new_tokens}
            if temperature is not None:
                generate_kwargs["temperature"] = temperature
            if do_sample is not None:
                generate_kwargs["do_sample"] = do_sample
            if top_p is not None:
                generate_kwargs["top_p"] = top_p
            generated_ids = model.generate(**model_inputs, **generate_kwargs)

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        if hasattr(output_ids, "tolist"):
            output_ids = output_ids.tolist()
        if hasattr(tokenizer, "batch_decode"):
            output_text = tokenizer.batch_decode(
                [output_ids],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        else:
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        del model_inputs, generated_ids, output_ids
        torch.cuda.empty_cache()

        output_text = strip_thinking_block(output_text)
        if trace_recorder is not None:
            trace_recorder.after_call(
                call_id,
                "qwen_executor",
                operation,
                {"raw_output": output_text, "latency_ms": round((time.time() - started_at) * 1000)},
                step_id=context.get("step_id"),
                attempt=context.get("attempt"),
            )
        return output_text
    except Exception as exc:
        if trace_recorder is not None:
            response_metadata = getattr(model, "last_response_metadata", {}) or {}
            trace_recorder.after_call(
                call_id,
                trace_component,
                operation,
                {"latency_ms": round((time.time() - started_at) * 1000), **response_metadata},
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                step_id=context.get("step_id"),
                attempt=context.get("attempt"),
            )
        raise


@dataclass
class OpenAICompatibleLLM:
    base_url: str
    model_name: str
    api_key: str = "EMPTY"
    timeout: int = 600
    provider: str = "qwen"
    max_attempts: int = 1
    retry_base_seconds: float = 1.0
    request_char_limit: int = 120_000
    budget_rmb: Optional[float] = None

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.provider = self.provider.lower()
        if self.provider not in {"qwen", "deepseek", "openai_compatible"}:
            raise ValueError(f"Unsupported Executor provider: {self.provider}")
        if self.max_attempts < 1:
            raise ValueError("Executor API max_attempts must be at least 1")
        if self.request_char_limit < 1:
            raise ValueError("Executor request_char_limit must be positive")
        self.trace_component = "deepseek_executor" if self.provider == "deepseek" else "qwen_executor"
        self.last_response_metadata: Dict[str, Any] = {}
        self.returned_model_name: Optional[str] = None
        self.cumulative_cost_rmb = 0.0

    def generate_messages(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        self.last_response_metadata = {}
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
        }

        if self.provider == "deepseek":
            payload["thinking"] = {"type": "disabled"}
            payload["response_format"] = {"type": "json_object"}
        else:
            payload["top_k"] = 20
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        if temperature is not None:
            payload["temperature"] = temperature
        elif do_sample is False:
            payload["temperature"] = 0.0

        if do_sample is False:
            payload["top_p"] = 1.0
        elif top_p is not None:
            payload["top_p"] = top_p
        if seed is not None and self.provider != "deepseek":
            payload["seed"] = seed

        serialized_chars = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        if serialized_chars > self.request_char_limit:
            raise ExecutorContextBudgetExceeded(
                f"Executor request has {serialized_chars} characters; hard limit is {self.request_char_limit}."
            )
        if self.budget_rmb is not None and self.cumulative_cost_rmb >= self.budget_rmb:
            raise ExecutorCostBudgetExceeded(
                f"Executor cost budget exhausted: {self.cumulative_cost_rmb:.6f} / {self.budget_rmb:.6f} RMB."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        started_at = time.time()
        last_error: Optional[BaseException] = None
        attempts_used = 0
        for api_attempt in range(1, self.max_attempts + 1):
            attempts_used = api_attempt
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                status_code = int(getattr(response, "status_code", 200))
                if status_code == 429 or status_code >= 500:
                    raise ExecutorAPIError(f"retryable HTTP {status_code}")
                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    raise ExecutorAPIError(f"non-retryable HTTP {status_code}") from exc
                try:
                    data = response.json()
                    choice = data["choices"][0]
                    content = choice["message"]["content"]
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise ExecutorAPIError("invalid chat completions response") from exc
                finish_reason = choice.get("finish_reason")
                if finish_reason == "length":
                    raise ExecutorAPIError("finish_reason=length (truncated response)")
                if content is None or not str(content).strip():
                    raise ExecutorAPIError("empty response content")

                returned_model = data.get("model")
                if self.provider == "deepseek":
                    if not isinstance(returned_model, str) or not returned_model.strip():
                        raise ExecutorAPIError(
                            "non-retryable response omitted the returned model identity"
                        )
                    if (
                        self.returned_model_name is not None
                        and returned_model != self.returned_model_name
                    ):
                        self.last_response_metadata = {
                            "provider": self.provider,
                            "requested_model": self.model_name,
                            "returned_model": returned_model,
                            "previous_returned_model": self.returned_model_name,
                            "api_attempts": api_attempt,
                            "request_chars": serialized_chars,
                            "provider_latency_ms": round(
                                (time.time() - started_at) * 1000
                            ),
                            "model_identity_drift": True,
                        }
                        raise ExecutorAPIError(
                            "non-retryable returned model identity changed within the run"
                        )
                    self.returned_model_name = returned_model

                usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                request_cost = _deepseek_cost_rmb(usage) if self.provider == "deepseek" else 0.0
                self.cumulative_cost_rmb += request_cost
                self.last_response_metadata = {
                    "provider": self.provider,
                    "requested_model": self.model_name,
                    "returned_model": returned_model,
                    "model_identity_drift": False,
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "estimated_cost_rmb": round(request_cost, 8),
                    "cumulative_cost_rmb": round(self.cumulative_cost_rmb, 8),
                    "api_attempts": api_attempt,
                    "request_chars": serialized_chars,
                    "provider_latency_ms": round((time.time() - started_at) * 1000),
                }
                return strip_thinking_block(str(content))
            except (requests.RequestException, ExecutorAPIError) as exc:
                last_error = exc
                retryable = isinstance(exc, requests.RequestException) or str(exc).startswith(
                    ("retryable HTTP", "empty response content")
                )
                if not retryable or api_attempt >= self.max_attempts:
                    break
                time.sleep(self.retry_base_seconds * (2 ** (api_attempt - 1)))

        if not self.last_response_metadata.get("model_identity_drift"):
            self.last_response_metadata = {
                "provider": self.provider,
                "requested_model": self.model_name,
                "returned_model": self.returned_model_name,
                "api_attempts": attempts_used,
                "request_chars": serialized_chars,
                "provider_latency_ms": round((time.time() - started_at) * 1000),
            }
        raise ExecutorAPIError(f"Chat completions request failed: {last_error}") from last_error


def _load_transformers_backend(checkpoint_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype="auto",
        device_map="auto",
    )
    return model, tokenizer


def _load_qwen35_transformers_backend(checkpoint_path: str):
    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise UnsupportedModelError(
            "qwen3_5 local transformers loading requires a newer transformers build with "
            "AutoModelForMultimodalLM and AutoProcessor support."
        ) from exc

    processor = AutoProcessor.from_pretrained(checkpoint_path)
    model = AutoModelForMultimodalLM.from_pretrained(
        checkpoint_path,
        torch_dtype="auto",
        device_map="auto",
    )
    return model, processor


def load_llm_backend(
    checkpoint_path: Optional[str],
    backend: str = "auto",
    api_base_url: Optional[str] = None,
    api_model: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: str = "qwen",
    env_file: Optional[str] = None,
    timeout: int = 600,
    api_max_attempts: int = 1,
    request_char_limit: int = 120_000,
    budget_rmb: Optional[float] = None,
) -> Tuple[Any, Any]:
    provider = provider.lower()
    if provider == "deepseek":
        env_values = read_env_file(env_file) if env_file else {}
        explicit_key = api_key if api_key not in {None, "", "EMPTY"} else None
        resolved_key = explicit_key or env_values.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if not resolved_key:
            location = env_file or "DEEPSEEK_API_KEY"
            raise UnsupportedModelError(f"DeepSeek API key is missing; fill DEEPSEEK_API_KEY in {location}.")
        return OpenAICompatibleLLM(
            base_url=api_base_url or env_values.get("DEEPSEEK_BASE_URL") or DEEPSEEK_DEFAULT_BASE_URL,
            model_name=api_model or env_values.get("DEEPSEEK_MODEL") or DEEPSEEK_DEFAULT_MODEL,
            api_key=resolved_key,
            timeout=timeout,
            provider="deepseek",
            max_attempts=api_max_attempts,
            request_char_limit=request_char_limit,
            budget_rmb=budget_rmb,
        ), None

    if not checkpoint_path:
        raise UnsupportedModelError("--qwen_ckpt is required when --executor_provider=qwen.")
    family = detect_checkpoint_family(checkpoint_path)
    backend = backend.lower()

    if backend not in {"auto", "transformers", "openai_compatible"}:
        raise ValueError(f"Unsupported qwen backend: {backend}")

    if backend == "openai_compatible" or (backend == "auto" and family == "qwen3_5" and api_base_url):
        if not api_base_url:
            raise UnsupportedModelError("--qwen_api_base_url is required for --qwen_backend openai_compatible.")
        model_name = api_model or os.path.basename(os.path.normpath(checkpoint_path))
        return OpenAICompatibleLLM(
            base_url=api_base_url,
            model_name=model_name,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            timeout=timeout,
            provider="qwen",
            max_attempts=api_max_attempts,
            request_char_limit=request_char_limit,
            budget_rmb=budget_rmb,
        ), None

    if family == "qwen3_5" and backend == "auto":
        raise UnsupportedModelError(
            "Checkpoint family qwen3_5 cannot be loaded by this PathAgent transformers environment. "
            "Use --qwen_backend openai_compatible with a Qwen3.5 OpenAI-compatible server, or create a fresh "
            "environment with a transformers build that recognizes qwen3_5."
        )

    if family == "qwen3_5" and backend == "transformers":
        return _load_qwen35_transformers_backend(checkpoint_path)

    return _load_transformers_backend(checkpoint_path)
