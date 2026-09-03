from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from patho_lora_sft.audit import choose_canonical_candidate, parse_judge_result
from patho_lora_sft.budget import (
    AccountBalance,
    BudgetController,
    reconcile_response_costs,
    wait_for_cost_reconciliation,
)
from patho_lora_sft.common import ContractViolation, sha256_file
from patho_lora_sft.constrained import PREFIX, SUFFIX, FindingsPrefixConstraint, evaluate_native_and_deployed, grammar_state
from patho_lora_sft.contracts import GEMINI_CONTRACT, LUNA_CONTRACT, load_and_verify_contract
from patho_lora_sft.dataset import assistant_only_labels, audit_loss_mask, build_training_record
from patho_lora_sft.openrouter import OpenRouterClient
from patho_lora_sft.patho_generation import PathoSchemaGenerator
from patho_lora_sft.privacy import (
    deterministic_unassessable_control,
    nonmedical_smoke_image,
    opencv_burned_text_scanner,
    sanitize_image,
)
from patho_lora_sft.recovery import AtomicRunJournal
from patho_lora_sft.review import audit_intra_rater_consistency, build_review_presentations, export_review_csv
from patho_lora_sft.schema import canonical_target, parse_bare_findings, parse_wrapped_findings, scan_forbidden
from patho_lora_sft.splits import (
    GuardStats,
    allocate_patient_groups,
    guarded_non_test_jsonl,
    load_non_test_pool,
    select_pilot_prefix,
    select_split_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def no_text(_: bytes) -> list[dict]:
    return []


def no_codes(_):
    return []


def balance(remaining: float = 10.0) -> AccountBalance:
    return AccountBalance(
        observed_at_utc="2026-08-27T00:00:00+00:00",
        key_limit_usd=10.0,
        key_usage_usd=10.0 - remaining,
        key_remaining_usd=remaining,
        account_total_credits_usd=30.0,
        account_total_usage_usd=20.0,
        account_remaining_usd=10.0,
    )


def accepted_decision():
    return choose_canonical_candidate(
        primary_raw='{"findings":["Branching pink structures"]}',
        secondary_audit_raw='{"claims":[{"finding_index":0,"status":"supported"}],"forbidden_inference":false}',
    )


def test_provider_policies_fail_closed() -> None:
    assert GEMINI_CONTRACT.routing_object() == {
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
        "only": ["Google"],
        "zdr": True,
    }
    assert LUNA_CONTRACT.routing_object() == {
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
        "only": ["OpenAI"],
    }


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ('{"findings":[]}', None),
        ('{"findings":["x"]}', None),
        ('{"findings":"x"}', "findings_not_array"),
        ('{"findings":[""]}', "finding_empty"),
        ('{"findings":[1]}', "finding_not_string"),
        ('{"findings":[],"extra":1}', "unexpected_keys"),
        ('{"findings":["1","2","3","4","5","6"]}', "too_many_findings"),
        ('not json', "json_parse_failed"),
    ],
)
def test_strict_bare_schema(raw: str, error: str | None) -> None:
    result = parse_bare_findings(raw)
    assert result.error == error
    assert result.valid is (error is None)


def test_wrapped_schema_and_deterministic_target() -> None:
    assert canonical_target(["Pink stroma"]) == '<answer>{"findings":["Pink stroma"]}</answer>'
    assert parse_wrapped_findings('<answer>{"findings":[]}</answer>').valid
    assert not parse_wrapped_findings('prefix<answer>{"findings":[]}</answer>').valid
    assert evaluate_native_and_deployed("bad", '<answer>{"findings":[]}</answer>') == {
        "native_schema_valid": False,
        "native_schema_error": "answer_boundary_not_exact",
        "deployed_schema_valid": True,
        "deployed_schema_error": None,
        "metrics_separated": True,
    }


def test_forbidden_scan_and_cross_audit_acceptance() -> None:
    assert scan_forbidden(["Pancreatic adenocarcinoma"])
    decision = accepted_decision()
    assert decision.status == "accepted"
    assert decision.risk_tier == "low"
    assert decision.target == '<answer>{"findings":["Branching pink structures"]}</answer>'


def test_privacy_reencode_removes_png_text_and_controls_are_unique() -> None:
    image = nonmedical_smoke_image()
    image.info["comment"] = "case=secret"
    sanitized = sanitize_image(image, source_kind="nonmedical_synthetic_smoke", text_scanner=no_text, code_scanner=no_codes)
    assert sanitized.approved and "comment" in sanitized.metadata_removed
    with Image.open(io.BytesIO(sanitized.png_bytes)) as check:
        assert check.info == {}
        assert check.mode == "RGB" and check.size == (784, 784)
    hashes = {
        sanitize_image(
            deterministic_unassessable_control(index),
            source_kind="deterministic_unassessable_control",
            text_scanner=no_text,
            code_scanner=no_codes,
        ).sha256
        for index in range(10)
    }
    assert len(hashes) == 10


def test_opencv_burned_text_geometry_detects_identifier_like_overlay() -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    image = np.full((784, 784, 3), 255, dtype=np.uint8)
    cv2.putText(image, "CASE123", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    assert opencv_burned_text_scanner(buffer.getvalue())


def test_budget_ledger_atomic_and_bounded(tmp_path: Path) -> None:
    controller = BudgetController(tmp_path / "ledger.json", initial_project_spend_usd=0.001)
    authorization = controller.authorize(stage="S0", model=GEMINI_CONTRACT.model, estimated_max_cost_usd=0.05, balance=balance())
    controller.record_actual(authorization, actual_cost_usd=0.002, request_id="gen_test")
    assert controller.public_summary()["cumulative_project_spend_usd"] == pytest.approx(0.003)
    reloaded = BudgetController(tmp_path / "ledger.json", initial_project_spend_usd=999)
    assert reloaded.cumulative_spend_usd == pytest.approx(0.003)


def test_cost_reconciliation_accepts_documented_key_usage_lag() -> None:
    before = balance()
    after = AccountBalance(
        observed_at_utc="later",
        key_limit_usd=10.0,
        key_usage_usd=before.key_usage_usd,
        key_remaining_usd=before.key_remaining_usd,
        account_total_credits_usd=30.0,
        account_total_usage_usd=before.account_total_usage_usd + 0.001,
        account_remaining_usd=before.account_remaining_usd - 0.001,
    )
    result = reconcile_response_costs(before=before, after=after, response_cost_usd=0.001)
    assert result["account_total_usage_match"] is True
    assert result["key_usage_lagging"] is True


def test_cost_reconciliation_uses_bounded_read_only_polling() -> None:
    before = balance()
    stale = before
    current = AccountBalance(
        "later", 10.0, before.key_usage_usd, before.key_remaining_usd,
        30.0, before.account_total_usage_usd + 0.001, before.account_remaining_usd - 0.001,
    )
    snapshots = iter([stale, stale, current])
    sleeps = []
    after, result = wait_for_cost_reconciliation(
        before=before,
        response_cost_usd=0.001,
        balance_reader=lambda: next(snapshots),
        max_attempts=3,
        interval_seconds=0.25,
        sleep_fn=sleeps.append,
    )
    assert after is current and result["poll_attempt"] == 3
    assert sleeps == [0.25, 0.25]


def test_review_export_has_ten_percent_anonymous_repeats(tmp_path: Path) -> None:
    records = [
        {"review_id": f"r{index}", "image_relpath": f"images/{index}.png", "candidate_a": "A", "candidate_b": "B"}
        for index in range(20)
    ]
    rows = build_review_presentations(records, paired=True)
    assert len(rows) == 22
    assert sum(row["is_repeat"] == "true" for row in rows) == 2
    output = tmp_path / "review.csv"
    export_review_csv(output, rows)
    parsed = list(csv.DictReader(output.open()))
    assert len(parsed) == 22


def test_intra_rater_consistency_passes_at_ninety_percent() -> None:
    rows = []
    for index in range(10):
        base = {
            "source_review_id": f"r{index}",
            "claim_labels_json": '["supported"]',
            "pair_preference": "tie",
            "forbidden_inference": "false",
        }
        rows.extend([dict(base), dict(base)])
    rows[-1]["pair_preference"] = "A"
    result = audit_intra_rater_consistency(rows)
    assert result["consistency"] == 0.9 and result["passed"]


def test_training_record_and_assistant_only_mask() -> None:
    record = build_training_record(
        sample_id="sft_abc",
        image_relpath="images/sft_abc.png",
        user_prompt="Return the frozen schema.",
        decision=accepted_decision(),
    )
    assert record["messages"][-1]["role"] == "assistant"
    input_ids = list(range(20))
    labels = assistant_only_labels(input_ids, [(15, 20)])
    audit = audit_loss_mask(labels, system_span=(0, 3), user_span=(3, 10), image_token_indices=range(10, 15), assistant_span=(15, 20))
    assert audit["passed"] and labels[:15] == [-100] * 15 and labels[15:] == input_ids[15:]


class FakeTokenizer:
    def __init__(self) -> None:
        self.tokens = [PREFIX, "]", '"', "A", "\",", SUFFIX, "BAD<", "<eos>"]
        self.eos_token_id = 7
        self.all_special_ids = [7]

    def __len__(self):
        return len(self.tokens)

    def decode(self, ids, **_kwargs):
        return "".join(self.tokens[index] for index in ids if index != self.eos_token_id)


def test_constrained_prefix_grammar_and_transformers_callback() -> None:
    assert grammar_state(PREFIX + "]" + SUFFIX).complete
    assert grammar_state(PREFIX + '"A"]' + SUFFIX).complete
    with pytest.raises(ContractViolation):
        grammar_state(PREFIX + '""]' + SUFFIX)
    tokenizer = FakeTokenizer()
    constraint = FindingsPrefixConstraint(tokenizer, [1])
    assert 0 in constraint(0, [7])
    assert set(constraint(0, [7, 0])) >= {1, 2}
    finished = [7, 0, 2, 3, 2, 1, 5]
    assert constraint(0, finished) == [7]


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values[0]))

    def __getitem__(self, item):
        rows, columns = item
        assert rows == slice(None)
        return FakeTensor([self.values[0][columns]])


class FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=FakeTensor([[99]]))
        self.input_ids = self["input_ids"]

    def to(self, _device):
        return self


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, *_args, **_kwargs):
        return "prompt"

    def __call__(self, **_kwargs):
        return FakeInputs()

    def batch_decode(self, _ids, **_kwargs):
        return ['<answer>{"findings":[]}</answer>']


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return FakeTensor([[99, 0, 1, 5]])


def test_local_patho_native_and_constrained_routes_use_separate_metrics() -> None:
    model = FakeModel()
    generator = PathoSchemaGenerator(
        model=model,
        processor=FakeProcessor(),
        vision_info_fn=lambda _messages: ([nonmedical_smoke_image()], None),
    )
    native = generator.generate(
        image=nonmedical_smoke_image(), system_prompt="system", user_prompt="user", constrained=False
    )
    assert native.metric_name == "native_schema_valid" and native.schema_valid
    assert "prefix_allowed_tokens_fn" not in model.kwargs
    deployed = generator.generate(
        image=nonmedical_smoke_image(), system_prompt="system", user_prompt="user", constrained=True
    )
    assert deployed.metric_name == "deployed_schema_valid" and deployed.schema_valid
    assert callable(model.kwargs["prefix_allowed_tokens_fn"])


def test_atomic_recovery_is_idempotent_and_nonoverwriting(tmp_path: Path) -> None:
    journal = AtomicRunJournal(tmp_path / "journal")
    record = {"item_id": "item_abc", "content": '{"findings":[]}'}
    path = journal.record_complete(item_id="item_abc", task_fingerprint="task1", record=record)
    assert journal.completed("item_abc", "task1")
    assert journal.record_complete(item_id="item_abc", task_fingerprint="task1", record=record) == path
    with pytest.raises(ContractViolation):
        journal.record_complete(item_id="item_abc", task_fingerprint="task2", record=record)


def test_guard_rejects_test_line_before_json_decode(tmp_path: Path) -> None:
    manifest = tmp_path / "source.jsonl"
    manifest.write_text('{"split":"test", this is intentionally invalid}\n{"split":"dev","slide_id":"s"}\n')
    stats = GuardStats()
    rows = list(guarded_non_test_jsonl(manifest, stats))
    assert rows == [{"split": "dev", "slide_id": "s"}]
    assert stats.test_split_markers_rejected_before_decode == 1
    assert stats.test_payload_rows_decoded == 0
