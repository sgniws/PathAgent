import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.llm_backend import (
    ExecutorAPIError,
    ExecutorContextBudgetExceeded,
    OpenAICompatibleLLM,
    UnsupportedModelError,
    detect_checkpoint_family,
    generate_chat_text,
    load_llm_backend,
    read_env_file,
)


def test_detects_qwen35_from_config(tmp_path):
    ckpt = tmp_path / "Qwen3.5-9B"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}),
        encoding="utf-8",
    )

    assert detect_checkpoint_family(str(ckpt)) == "qwen3_5"


def test_qwen35_auto_without_api_base_url_fails_with_actionable_message(tmp_path):
    ckpt = tmp_path / "Qwen3.5-9B"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")

    with pytest.raises(UnsupportedModelError) as exc_info:
        load_llm_backend(str(ckpt), backend="auto", api_base_url=None, api_model=None)

    message = str(exc_info.value)
    assert "qwen3_5" in message
    assert "--qwen_backend openai_compatible" in message
    assert "transformers" in message


def test_openai_compatible_backend_posts_non_thinking_request(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "<think>\nprivate reasoning\n</think>\n\n{\"answer\": \"A\"}"
                        }
                    }
                ]
            }

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("models.llm_backend.requests.post", fake_post)

    backend = OpenAICompatibleLLM(
        base_url="http://localhost:8000/v1",
        model_name="Qwen/Qwen3.5-9B",
        api_key="EMPTY",
    )

    text = backend.generate_messages(
        [{"role": "user", "content": "Return JSON"}],
        max_new_tokens=128,
        temperature=0.0,
        do_sample=False,
    )

    assert text == '{"answer": "A"}'
    assert calls[0]["url"] == "http://localhost:8000/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer EMPTY"
    assert calls[0]["json"]["model"] == "Qwen/Qwen3.5-9B"
    assert calls[0]["json"]["messages"] == [{"role": "user", "content": "Return JSON"}]
    assert calls[0]["json"]["max_tokens"] == 128
    assert calls[0]["json"]["temperature"] == 0.0
    assert "extra_body" not in calls[0]["json"]
    assert calls[0]["json"]["top_k"] == 20
    assert calls[0]["json"]["chat_template_kwargs"]["enable_thinking"] is False


def test_deepseek_backend_uses_provider_contract_and_records_rmb_cost(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "DeepSeek-V4-Flash-0731",
                "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":"A"}'}}],
                "usage": {
                    "prompt_tokens": 1000,
                    "prompt_cache_hit_tokens": 400,
                    "prompt_cache_miss_tokens": 600,
                    "completion_tokens": 100,
                },
            }

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("models.llm_backend.requests.post", fake_post)
    backend = OpenAICompatibleLLM(
        base_url="https://api.deepseek.com",
        model_name="deepseek-v4-flash",
        api_key="test-only",
        provider="deepseek",
        max_attempts=3,
    )

    result = backend.generate_messages(
        [{"role": "user", "content": "Return a JSON object."}],
        max_new_tokens=128,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        seed=129,
    )

    assert result == '{"answer":"A"}'
    payload = calls[0]["json"]
    assert calls[0]["url"] == "https://api.deepseek.com/chat/completions"
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert "seed" not in payload
    assert "top_k" not in payload
    assert "chat_template_kwargs" not in payload
    assert backend.last_response_metadata["returned_model"] == "DeepSeek-V4-Flash-0731"
    assert backend.last_response_metadata["finish_reason"] == "stop"
    assert backend.last_response_metadata["estimated_cost_rmb"] == pytest.approx(0.000808)


def test_deepseek_retries_500_and_empty_content_with_exponential_backoff(monkeypatch):
    responses = [
        types.SimpleNamespace(status_code=500),
        types.SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
        ),
        types.SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {
                "model": "DeepSeek-V4-Flash-0731",
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            },
        ),
    ]
    sleeps = []
    monkeypatch.setattr("models.llm_backend.requests.post", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr("models.llm_backend.time.sleep", sleeps.append)
    backend = OpenAICompatibleLLM(
        "https://api.deepseek.com", "deepseek-v4-flash", "test-only", provider="deepseek", max_attempts=3
    )

    assert backend.generate_messages([{"role": "user", "content": "JSON"}], 32) == "{}"
    assert sleeps == [1.0, 2.0]
    assert backend.last_response_metadata["api_attempts"] == 3


def test_deepseek_stops_when_returned_model_identity_changes(monkeypatch):
    returned_models = iter(["DeepSeek-V4-Flash-0731", "DeepSeek-V4-Flash-0801"])

    def fake_post(*args, **kwargs):
        model = next(returned_models)
        return types.SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {
                "model": model,
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "{}"}}
                ],
            },
        )

    monkeypatch.setattr("models.llm_backend.requests.post", fake_post)
    backend = OpenAICompatibleLLM(
        "https://api.deepseek.com",
        "deepseek-v4-flash",
        "test-only",
        provider="deepseek",
    )

    assert backend.generate_messages([{"role": "user", "content": "JSON"}], 32) == "{}"
    with pytest.raises(ExecutorAPIError, match="returned model identity changed"):
        backend.generate_messages([{"role": "user", "content": "JSON"}], 32)
    assert backend.last_response_metadata["model_identity_drift"] is True


def test_deepseek_does_not_retry_non_retryable_4xx(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 401

        def raise_for_status(self):
            raise __import__("requests").HTTPError("unauthorized")

    monkeypatch.setattr("models.llm_backend.requests.post", lambda *args, **kwargs: calls.append(1) or FakeResponse())
    backend = OpenAICompatibleLLM(
        "https://api.deepseek.com", "deepseek-v4-flash", "bad", provider="deepseek", max_attempts=3
    )

    with pytest.raises(ExecutorAPIError, match="non-retryable HTTP 401"):
        backend.generate_messages([{"role": "user", "content": "JSON"}], 32)
    assert len(calls) == 1


def test_deepseek_treats_length_finish_as_structured_failure_without_retry(monkeypatch):
    calls = []
    response = types.SimpleNamespace(
        status_code=200,
        raise_for_status=lambda: None,
        json=lambda: {"choices": [{"finish_reason": "length", "message": {"content": "{"}}]},
    )
    monkeypatch.setattr("models.llm_backend.requests.post", lambda *args, **kwargs: calls.append(1) or response)
    backend = OpenAICompatibleLLM(
        "https://api.deepseek.com", "deepseek-v4-flash", "test-only", provider="deepseek", max_attempts=3
    )

    with pytest.raises(ExecutorAPIError, match="finish_reason=length"):
        backend.generate_messages([{"role": "user", "content": "JSON"}], 32)
    assert len(calls) == 1


def test_executor_request_hard_limit_is_checked_before_network(monkeypatch):
    monkeypatch.setattr(
        "models.llm_backend.requests.post",
        lambda *args, **kwargs: pytest.fail("network must not be called after the context gate"),
    )
    backend = OpenAICompatibleLLM(
        "https://api.deepseek.com",
        "deepseek-v4-flash",
        "test-only",
        provider="deepseek",
        request_char_limit=100,
    )
    with pytest.raises(ExecutorContextBudgetExceeded):
        backend.generate_messages([{"role": "user", "content": "x" * 200}], 32)


def test_deepseek_loads_secret_from_local_env_file(tmp_path):
    env_path = tmp_path / "api.env"
    env_path.write_text(
        "DEEPSEEK_API_KEY='test-key'\nDEEPSEEK_BASE_URL=https://example.invalid\n"
        "DEEPSEEK_MODEL=deepseek-v4-flash\n",
        encoding="utf-8",
    )

    assert read_env_file(env_path)["DEEPSEEK_API_KEY"] == "test-key"
    backend, tokenizer = load_llm_backend(
        None,
        provider="deepseek",
        env_file=str(env_path),
        api_max_attempts=3,
    )
    assert tokenizer is None
    assert backend.base_url == "https://example.invalid"
    assert backend.model_name == "deepseek-v4-flash"
    assert backend.api_key == "test-key"


def test_qwen35_transformers_backend_uses_multimodal_loader(monkeypatch, tmp_path):
    ckpt = tmp_path / "Qwen3.5-9B"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")

    calls = []
    fake_processor = object()
    fake_model = object()

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(path):
            calls.append(("processor", path))
            return fake_processor

    class FakeAutoModelForMultimodalLM:
        @staticmethod
        def from_pretrained(path, torch_dtype, device_map):
            calls.append(("model", path, torch_dtype, device_map))
            return fake_model

    fake_transformers = types.SimpleNamespace(
        AutoProcessor=FakeAutoProcessor,
        AutoModelForMultimodalLM=FakeAutoModelForMultimodalLM,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model, processor = load_llm_backend(str(ckpt), backend="transformers")

    assert model is fake_model
    assert processor is fake_processor
    assert calls == [
        ("processor", str(ckpt)),
        ("model", str(ckpt), "auto", "auto"),
    ]


def test_generate_chat_text_uses_backend_object():
    class FakeBackend:
        def __init__(self):
            self.calls = []

        def generate_messages(self, messages, max_new_tokens, temperature=None, do_sample=None):
            self.calls.append((messages, max_new_tokens, temperature, do_sample))
            return '{"sufficient": "Yes"}'

    backend = FakeBackend()
    messages = [{"role": "user", "content": "Can you answer?"}]

    assert generate_chat_text(
        backend,
        None,
        messages,
        max_new_tokens=64,
        temperature=0.2,
        do_sample=False,
    ) == '{"sufficient": "Yes"}'
    assert backend.calls == [(messages, 64, 0.2, False)]


def test_generate_chat_text_supports_processor_style_inputs(monkeypatch):
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def __len__(self):
            return len(self.values)

        def __getitem__(self, item):
            return self.values[item]

    class FakeInputs(dict):
        input_ids = [FakeTensor([1, 2, 3])]

        def to(self, device):
            return self

    class FakeProcessor:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
            self.calls.append((messages, tokenize, add_generation_prompt, enable_thinking))
            return "templated"

        def __call__(self, *args, **kwargs):
            if args:
                raise TypeError("processor requires text keyword")
            assert kwargs["text"] == ["templated"]
            return FakeInputs()

        def batch_decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
            assert ids == [[4, 5]]
            return ["{\"answer\": \"A\"}"]

    class FakeModel:
        device = "cuda:0"

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 32
            return [FakeTensor([1, 2, 3, 4, 5])]

    class FakeTorch:
        class no_grad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return None

        class cuda:
            @staticmethod
            def empty_cache():
                return None

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    processor = FakeProcessor()

    assert generate_chat_text(
        FakeModel(),
        processor,
        [{"role": "user", "content": "Return JSON"}],
        max_new_tokens=32,
        do_sample=False,
    ) == '{"answer": "A"}'
    assert processor.calls[0][3] is False
