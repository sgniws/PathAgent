from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .common import BudgetViolation, atomic_write_json, load_json
from .contracts import MINIMUM_ACCOUNT_RESERVE_USD, PROJECT_COST_CAP_USD


@dataclass(frozen=True)
class AccountBalance:
    observed_at_utc: str
    key_limit_usd: float
    key_usage_usd: float
    key_remaining_usd: float
    account_total_credits_usd: float
    account_total_usage_usd: float
    account_remaining_usd: float

    @property
    def effective_remaining_usd(self) -> float:
        return min(self.key_remaining_usd, self.account_remaining_usd)


class BudgetController:
    """Atomic, fail-closed project ledger with an account reserve."""

    def __init__(
        self,
        ledger_path: Path,
        *,
        initial_project_spend_usd: float,
        project_cap_usd: float = PROJECT_COST_CAP_USD,
        minimum_account_reserve_usd: float = MINIMUM_ACCOUNT_RESERVE_USD,
    ) -> None:
        self._lock = threading.RLock()
        self.ledger_path = ledger_path
        self.project_cap_usd = float(project_cap_usd)
        self.minimum_account_reserve_usd = float(minimum_account_reserve_usd)
        if self.project_cap_usd != PROJECT_COST_CAP_USD:
            raise BudgetViolation("Project hard cap must remain exactly 9.50 USD")
        if self.minimum_account_reserve_usd < MINIMUM_ACCOUNT_RESERVE_USD:
            raise BudgetViolation("Account reserve cannot be below 0.50 USD")
        if ledger_path.exists():
            self._ledger = load_json(ledger_path)
        else:
            self._ledger = {
                "schema_version": "patho_lora_budget_ledger_v1",
                "project_cap_usd": self.project_cap_usd,
                "minimum_account_reserve_usd": self.minimum_account_reserve_usd,
                "initial_project_spend_usd": float(initial_project_spend_usd),
                "entries": [],
            }
            self._persist()
        self._validate_ledger()

    def _validate_ledger(self) -> None:
        if float(self._ledger.get("project_cap_usd", -1)) != self.project_cap_usd:
            raise BudgetViolation("Existing ledger cap mismatch")
        if float(self._ledger.get("minimum_account_reserve_usd", -1)) < self.minimum_account_reserve_usd:
            raise BudgetViolation("Existing ledger reserve mismatch")
        values = [float(self._ledger.get("initial_project_spend_usd", 0.0))]
        values.extend(float(entry["actual_cost_usd"]) for entry in self._ledger.get("entries", []))
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise BudgetViolation("Ledger contains an invalid cost")
        if sum(values) > self.project_cap_usd + 1e-12:
            raise BudgetViolation("Existing ledger already exceeds the project cap")

    @property
    def cumulative_spend_usd(self) -> float:
        with self._lock:
            return float(self._ledger["initial_project_spend_usd"]) + sum(
                float(entry["actual_cost_usd"]) for entry in self._ledger["entries"]
            )

    def authorize(self, *, stage: str, model: str, estimated_max_cost_usd: float, balance: AccountBalance) -> dict[str, Any]:
        with self._lock:
            estimate = float(estimated_max_cost_usd)
            if not math.isfinite(estimate) or estimate <= 0:
                raise BudgetViolation("Estimated request cost must be finite and positive")
            projected = self.cumulative_spend_usd + estimate
            if projected > self.project_cap_usd + 1e-12:
                raise BudgetViolation("Request could exceed the 9.50 USD project cap")
            if balance.effective_remaining_usd - estimate < self.minimum_account_reserve_usd - 1e-12:
                raise BudgetViolation("Request could consume the required 0.50 USD account reserve")
            return {
                "authorization_id": f"costauth_{uuid.uuid4().hex}",
                "stage": stage,
                "model": model,
                "estimated_max_cost_usd": estimate,
                "projected_project_spend_usd": projected,
                "balance_observed_at_utc": balance.observed_at_utc,
            }

    def authorize_stage(
        self, *, stage: str, estimated_max_stage_cost_usd: float, balance: AccountBalance
    ) -> dict[str, Any]:
        """Reserve-check a complete bounded stage without mutating the ledger."""
        with self._lock:
            estimate = float(estimated_max_stage_cost_usd)
            if not math.isfinite(estimate) or estimate <= 0:
                raise BudgetViolation("Estimated stage cost must be finite and positive")
            projected = self.cumulative_spend_usd + estimate
            if projected > self.project_cap_usd + 1e-12:
                raise BudgetViolation("Stage could exceed the 9.50 USD project cap")
            if balance.effective_remaining_usd - estimate < self.minimum_account_reserve_usd - 1e-12:
                raise BudgetViolation("Stage could consume the required 0.50 USD account reserve")
            return {
                "stage_authorization_id": f"stageauth_{uuid.uuid4().hex}",
                "stage": stage,
                "estimated_max_stage_cost_usd": estimate,
                "projected_project_spend_usd": projected,
                "projected_effective_account_remaining_usd": balance.effective_remaining_usd - estimate,
                "balance_observed_at_utc": balance.observed_at_utc,
                "ledger_mutated": False,
            }

    def record_actual(self, authorization: dict[str, Any], *, actual_cost_usd: float, request_id: str) -> None:
        with self._lock:
            actual = float(actual_cost_usd)
            if not math.isfinite(actual) or actual < 0:
                raise BudgetViolation("API response cost is missing or invalid")
            entry = {
                **authorization,
                "actual_cost_usd": actual,
                "request_id": request_id,
                "recorded_unix": time.time(),
            }
            self._ledger["entries"].append(entry)
            self._persist()
            if actual > float(authorization["estimated_max_cost_usd"]) + 1e-12:
                raise BudgetViolation("Actual request cost exceeded its preauthorization bound")
            if self.cumulative_spend_usd > self.project_cap_usd + 1e-12:
                raise BudgetViolation("Actual project spend exceeded the hard cap")

    def record_reconciled_failure(
        self,
        *,
        reconciliation_id: str,
        stage: str,
        model: str,
        actual_cost_usd: float,
        evidence: dict[str, Any],
    ) -> None:
        with self._lock:
            if any(entry.get("reconciliation_id") == reconciliation_id for entry in self._ledger["entries"]):
                return
            actual = float(actual_cost_usd)
            if not math.isfinite(actual) or actual <= 0:
                raise BudgetViolation("Reconciled failure cost must be finite and positive")
            entry = {
                "authorization_id": "reconciled_after_billed_failure",
                "reconciliation_id": reconciliation_id,
                "stage": stage,
                "model": model,
                "estimated_max_cost_usd": actual,
                "actual_cost_usd": actual,
                "request_id": "unavailable_due_preledger_schema_failure",
                "evidence": evidence,
                "recorded_unix": time.time(),
            }
            self._ledger["entries"].append(entry)
            self._persist()
            if self.cumulative_spend_usd > self.project_cap_usd + 1e-12:
                raise BudgetViolation("Reconciled project spend exceeded the hard cap")

    def public_summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "project_cap_usd": self.project_cap_usd,
                "minimum_account_reserve_usd": self.minimum_account_reserve_usd,
                "initial_project_spend_usd": float(self._ledger["initial_project_spend_usd"]),
                "new_request_count": len(self._ledger["entries"]),
                "new_spend_usd": sum(float(item["actual_cost_usd"]) for item in self._ledger["entries"]),
                "cumulative_project_spend_usd": self.cumulative_spend_usd,
                "remaining_project_authorization_usd": self.project_cap_usd - self.cumulative_spend_usd,
            }

    def has_recorded_request(self, *, request_id: str, actual_cost_usd: float) -> bool:
        with self._lock:
            expected = float(actual_cost_usd)
            return any(
                entry.get("request_id") == request_id
                and abs(float(entry.get("actual_cost_usd", -1.0)) - expected) <= 1e-12
                for entry in self._ledger["entries"]
            )

    def _persist(self) -> None:
        atomic_write_json(self.ledger_path, self._ledger, mode=0o600)


def reconcile_response_costs(
    *, before: AccountBalance, after: AccountBalance, response_cost_usd: float
) -> dict[str, Any]:
    expected = float(response_cost_usd)
    if not math.isfinite(expected) or expected < 0:
        raise BudgetViolation("Response cost sum is invalid")
    key_delta = after.key_usage_usd - before.key_usage_usd
    account_delta = after.account_total_usage_usd - before.account_total_usage_usd
    tolerance = max(0.00001, expected * 0.05)
    key_match = key_delta >= -1e-12 and abs(key_delta - expected) <= tolerance
    account_match = account_delta >= -1e-12 and abs(account_delta - expected) <= tolerance
    if not key_match and not account_match:
        raise BudgetViolation("Neither key usage nor account total usage reconciles with response usage.cost")
    return {
        "response_cost_usd": expected,
        "key_usage_delta_usd": key_delta,
        "account_total_usage_delta_usd": account_delta,
        "key_usage_match": key_match,
        "account_total_usage_match": account_match,
        "key_usage_lagging": account_match and not key_match,
        "cost_reconciled": True,
    }


def wait_for_cost_reconciliation(
    *,
    before: AccountBalance,
    response_cost_usd: float,
    balance_reader: Callable[[], AccountBalance],
    max_attempts: int = 6,
    interval_seconds: float = 2.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[AccountBalance, dict[str, Any]]:
    if max_attempts < 1 or interval_seconds < 0:
        raise ValueError("Invalid reconciliation polling contract")
    last_error: BudgetViolation | None = None
    for attempt in range(1, max_attempts + 1):
        after = balance_reader()
        try:
            result = reconcile_response_costs(
                before=before, after=after, response_cost_usd=response_cost_usd
            )
            result["poll_attempt"] = attempt
            result["max_poll_attempts"] = max_attempts
            result["bounded_polling"] = True
            return after, result
        except BudgetViolation as exc:
            last_error = exc
            if attempt < max_attempts:
                sleep_fn(interval_seconds)
    raise BudgetViolation(
        "OpenRouter billing did not reconcile within the bounded read-only polling window"
    ) from last_error
