from __future__ import annotations

import json
from pathlib import Path

import pytest

from patho_lora_sft.audit import choose_canonical_candidate
from patho_lora_sft.budget import AccountBalance, BudgetController
from patho_lora_sft.common import BudgetViolation, ContractViolation, PrivacyViolation, ProviderViolation
from patho_lora_sft.contracts import GEMINI_CONTRACT, LUNA_CONTRACT
from patho_lora_sft.dataset import validate_dataset_isolation
from patho_lora_sft.openrouter import OpenRouterClient
from patho_lora_sft.privacy import assert_external_payload_private, nonmedical_smoke_image, sanitize_image
from patho_lora_sft.recovery import AtomicRunJournal
from patho_lora_sft.review import audit_intra_rater_consistency
from patho_lora_sft.splits import assert_candidate_isolation


def no_text(_: bytes):
    return []


def no_codes(_):
    return []


def balance(remaining: float = 10.0) -> AccountBalance:
    return AccountBalance("now", 10.0, 10.0 - remaining, remaining, 30.0, 20.0, 10.0)


class FakeTransport:
    def __init__(
        self,
        *,
        model: str = GEMINI_CONTRACT.model,
        provider: str = "Google",
        cost=0.001,
        content: str = '{"findings":["A red square is left of a blue circle."]}',
    ):
        self.model = model
        self.provider = provider
        self.cost = cost
        self.content = content
        self.last_body = None

    def request(self, method, url, *, headers, body=None):
        self.last_body = body
        return 200, {
            "id": "gen_fake",
            "model": self.model,
            "provider": self.provider,
            "choices": [{"message": {"content": self.content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": self.cost},
        }, {}


def sanitized_image():
    return sanitize_image(nonmedical_smoke_image(), source_kind="nonmedical_synthetic_smoke", text_scanner=no_text, code_scanner=no_codes)


def response_format():
    return {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {"type": "object"}}}


def make_client(tmp_path: Path, transport: FakeTransport):
    budget = BudgetController(tmp_path / "ledger.json", initial_project_spend_usd=0)
    return OpenRouterClient(api_key="secret-never-written", budget=budget, transport=transport)


def invoke(client: OpenRouterClient, tmp_path: Path):
    return client.generate_findings(
        contract=GEMINI_CONTRACT,
        image=sanitized_image(),
        item_id="item_0123456789abcdef0123456789abcdef",
        system_prompt="synthetic geometry",
        user_prompt="describe red and blue shapes",
        response_format=response_format(),
        stage="test",
        response_path=tmp_path / "response.json",
        balance=balance(),
    )


@pytest.mark.parametrize(
    ("model", "provider", "error"),
    [
        ("google/gemini-3.7-flash:latest", "Google", ProviderViolation),
        (GEMINI_CONTRACT.model, "Google AI Studio", ProviderViolation),
    ],
)
def test_model_or_provider_mismatch_stops(tmp_path: Path, model: str, provider: str, error) -> None:
    client = make_client(tmp_path, FakeTransport(model=model, provider=provider))
    with pytest.raises(error):
        invoke(client, tmp_path)
    assert (tmp_path / "response.json").exists()


def test_missing_cost_stops_and_does_not_persist_response(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeTransport(cost=None))
    with pytest.raises(BudgetViolation):
        invoke(client, tmp_path)
    assert (tmp_path / "response.json").exists()


def test_success_record_excludes_request_and_base64(tmp_path: Path) -> None:
    transport = FakeTransport()
    client = make_client(tmp_path, transport)
    result = invoke(client, tmp_path)
    raw = (tmp_path / "response.json").read_text()
    assert result["request_body_saved"] is False
    assert "data:image" not in raw and "authorization" not in raw and "secret-never-written" not in raw
    assert transport.last_body["provider"]["allow_fallbacks"] is False
    assert "temperature" not in transport.last_body
    assert transport.last_body["reasoning"] == {"effort": "low", "exclude": True}


def test_cross_audit_is_anonymous_strict_and_persisted_without_request(tmp_path: Path) -> None:
    transport = FakeTransport(
        model=LUNA_CONTRACT.model,
        provider="OpenAI",
        content='{"claims":[{"finding_index":0,"status":"supported"}],"forbidden_inference":false}',
    )
    client = make_client(tmp_path, transport)
    result = client.audit_findings(
        contract=LUNA_CONTRACT,
        image=sanitized_image(),
        item_id="item_0123456789abcdef0123456789abcdef",
        candidate_findings=["A red square is left of a blue circle."],
        system_prompt="anonymous visual support auditor",
        user_prompt="return one support status per finding",
        response_format=response_format(),
        stage="test_cross_audit",
        response_path=tmp_path / "judge.json",
        balance=balance(),
    )
    payload_text = json.dumps(transport.last_body)
    persisted = (tmp_path / "judge.json").read_text()
    assert result["response_kind"] == "judge"
    assert result["parsed_judge"] == {"statuses": ["supported"], "forbidden_inference": False}
    assert GEMINI_CONTRACT.model not in payload_text and "primary" not in payload_text
    assert "findings" in payload_text and "data:image/png;base64," in payload_text
    assert "data:image" not in persisted and "candidate_findings" not in persisted


@pytest.mark.parametrize(
    "payload",
    [
        {"slide_id": "s", "image": {"url": "data:image/png;base64,AA=="}},
        {"messages": [{"text": "patient_id: P123"}], "image": {"url": "data:image/png;base64,AA=="}},
        {"messages": [{"text": "/data/private/file"}], "image": {"url": "data:image/png;base64,AA=="}},
        {"messages": [{"text": "file.svs"}], "image": {"url": "data:image/png;base64,AA=="}},
    ],
)
def test_pii_or_source_path_in_payload_stops(payload) -> None:
    with pytest.raises(PrivacyViolation):
        assert_external_payload_private(payload, allowed_item_id="item_0123456789abcdef")


def test_privacy_text_or_code_detection_stops() -> None:
    with pytest.raises(PrivacyViolation):
        sanitize_image(
            nonmedical_smoke_image(),
            source_kind="nonmedical_synthetic_smoke",
            text_scanner=lambda _: [{"kind": "ocr_text"}],
            code_scanner=no_codes,
        )
    with pytest.raises(PrivacyViolation):
        sanitize_image(
            nonmedical_smoke_image(),
            source_kind="nonmedical_synthetic_smoke",
            text_scanner=no_text,
            code_scanner=lambda _: [{"kind": "qr_code"}],
        )


def test_budget_cap_and_reserve_stop_before_request(tmp_path: Path) -> None:
    controller = BudgetController(tmp_path / "ledger.json", initial_project_spend_usd=9.49)
    with pytest.raises(BudgetViolation):
        controller.authorize(stage="x", model=GEMINI_CONTRACT.model, estimated_max_cost_usd=0.02, balance=balance())
    controller = BudgetController(tmp_path / "ledger2.json", initial_project_spend_usd=0)
    with pytest.raises(BudgetViolation):
        controller.authorize(stage="x", model=GEMINI_CONTRACT.model, estimated_max_cost_usd=0.02, balance=balance(0.51))


def test_schema_failure_fallback_and_double_failure() -> None:
    supported = '{"claims":[{"finding_index":0,"status":"supported"}],"forbidden_inference":false}'
    fallback = choose_canonical_candidate(
        primary_raw="bad",
        secondary_audit_raw='{"claims":[],"forbidden_inference":false}',
        fallback_raw='{"findings":["Pink bands"]}',
        primary_audit_raw=supported,
        human_review_passed=True,
    )
    assert fallback.status == "accepted" and fallback.source == "secondary_fallback" and fallback.risk_tier == "medium"
    rejected = choose_canonical_candidate(
        primary_raw="bad",
        secondary_audit_raw='{"claims":[],"forbidden_inference":false}',
        fallback_raw="also bad",
        primary_audit_raw='{"claims":[],"forbidden_inference":false}',
    )
    assert rejected.status == "rejected" and rejected.risk_tier == "high"


def test_unsupported_and_not_assessable_do_not_auto_accept() -> None:
    unsupported = choose_canonical_candidate(
        primary_raw='{"findings":["Pink bands"]}',
        secondary_audit_raw='{"claims":[{"finding_index":0,"status":"unsupported"}],"forbidden_inference":false}',
    )
    assert unsupported.status == "fallback_required"
    medium = choose_canonical_candidate(
        primary_raw='{"findings":["Pink bands"]}',
        secondary_audit_raw='{"claims":[{"finding_index":0,"status":"not_assessable"}],"forbidden_inference":false}',
    )
    assert medium.status == "waiting_human_review" and medium.target is None


def test_cross_split_duplicate_and_patient_isolation_stop() -> None:
    base = {
        "candidate_id": "a",
        "split": "dev",
        "patient_group_id": "pg1",
        "slide_id": "s1",
        "patch_id": "p1",
    }
    other = {**base, "candidate_id": "b", "patch_id": "p2"}
    with pytest.raises(PrivacyViolation):
        assert_candidate_isolation({"train": [base], "validation": [other], "locked": []})
    public = [{"sample_id": "x"}, {"sample_id": "y"}]
    private = [
        {"sample_id": "x", "split": "train", "patient_group_id": "pg1", "slide_id": "s1", "patch_id": "p1", "x_level0": 0, "y_level0": 0, "image_sha256": "h"},
        {"sample_id": "y", "split": "validation", "patient_group_id": "pg2", "slide_id": "s2", "patch_id": "p2", "x_level0": 1, "y_level0": 1, "image_sha256": "h"},
    ]
    with pytest.raises(PrivacyViolation):
        validate_dataset_isolation(public, private)


def test_test10_sidecar_stops() -> None:
    with pytest.raises(PrivacyViolation):
        validate_dataset_isolation(
            [{"sample_id": "x"}],
            [{"sample_id": "x", "split": "test", "patient_group_id": "pg", "slide_id": "s", "patch_id": "p", "x_level0": 0, "y_level0": 0, "image_sha256": "h"}],
        )


def test_human_consistency_below_ninety_percent_invalidates_conclusions() -> None:
    rows = []
    for index in range(10):
        base = {"source_review_id": str(index), "claim_labels_json": "supported", "pair_preference": "tie", "forbidden_inference": "false"}
        rows.extend([dict(base), dict(base)])
    rows[-1]["claim_labels_json"] = "unsupported"
    rows[-3]["claim_labels_json"] = "unsupported"
    result = audit_intra_rater_consistency(rows)
    assert result["consistency"] == 0.8
    assert result["passed"] is False and result["existing_human_conclusions_valid"] is False


def test_recovery_rejects_request_material(tmp_path: Path) -> None:
    journal = AtomicRunJournal(tmp_path / "journal")
    with pytest.raises(PrivacyViolation):
        journal.record_complete(item_id="item", task_fingerprint="task", record={"request_body": {"secret": 1}})
