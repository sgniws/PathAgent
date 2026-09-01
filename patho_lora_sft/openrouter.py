from __future__ import annotations

import base64
import http.client
import json
import math
import ssl
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from .audit import parse_judge_result
from .budget import AccountBalance, BudgetController
from .common import BudgetViolation, ContractViolation, NetworkBillingUnconfirmed, PrivacyViolation, ProviderViolation, TransientOpenRouterError, atomic_write_json, sha256_bytes
from .contracts import ProviderContract
from .privacy import SanitizedImage, assert_external_payload_private
from .schema import parse_bare_findings


OPENROUTER_BASE = "https://openrouter.ai"
TRANSIENT_HTTP_STATUSES = {408, 429, 500, 502, 503, 504, 524, 529}
SAFE_ERROR_HEADERS = {
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "x-generation-id",
}


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
        return seconds if seconds >= 0 else None
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _safe_http_error_metadata(exc: urllib.error.HTTPError) -> dict[str, Any]:
    headers = {
        key.lower(): value
        for key, value in exc.headers.items()
        if key.lower() in SAFE_ERROR_HEADERS
    }
    error_type = None
    provider_code = None
    try:
        parsed = json.loads(exc.read(1024 * 1024).decode("utf-8"))
        error = parsed.get("error", {}) if isinstance(parsed, dict) else {}
        metadata = error.get("metadata", {}) if isinstance(error, dict) else {}
        if isinstance(metadata, dict):
            if isinstance(metadata.get("error_type"), str):
                error_type = metadata["error_type"]
            if isinstance(metadata.get("provider_code"), (str, int)):
                provider_code = str(metadata["provider_code"])
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        pass
    return {
        "error_type": error_type,
        "provider_code": provider_code,
        "retry_after_seconds": _retry_after_seconds(headers.get("retry-after")),
        "rate_limit_headers": headers,
    }


class JSONTransport(Protocol):
    def request(self, method: str, url: str, *, headers: dict[str, str], body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any], dict[str, str]]: ...


class UrllibJSONTransport:
    def request(self, method: str, url: str, *, headers: dict[str, str], body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any], dict[str, str]]:
        encoded = None if body is None else json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(url, method=method, headers=headers, data=encoded)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                parsed = json.loads(response.read().decode("utf-8"))
                return response.status, parsed, {key.lower(): value for key, value in response.headers.items()}
        except urllib.error.HTTPError as exc:
            # Never surface a provider error body: it can echo request content.
            if exc.code in TRANSIENT_HTTP_STATUSES:
                safe = _safe_http_error_metadata(exc)
                raise TransientOpenRouterError(status_code=exc.code, **safe) from exc
            raise ContractViolation(f"OpenRouter HTTP failure: {exc.code}; response_body_redacted=true") from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.RemoteDisconnected,
            ssl.SSLError,
        ) as exc:
            raise NetworkBillingUnconfirmed("OpenRouter network failure; billing status unconfirmed") from exc


def load_api_key(env_path: Path) -> str:
    if env_path.stat().st_mode & 0o077:
        raise PrivacyViolation("OpenRouter env file permissions must be 0600 or stricter")
    found: list[str] = []
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        if raw.strip().startswith("OPENROUTER_API_KEY="):
            found.append(raw.split("=", 1)[1].strip().strip('"').strip("'"))
    if len(found) != 1 or not found[0]:
        raise PrivacyViolation("Exactly one non-empty OPENROUTER_API_KEY is required")
    return found[0]


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}


def read_account_balance(api_key: str, transport: JSONTransport | None = None) -> AccountBalance:
    transport = transport or UrllibJSONTransport()
    status_key, key_body, _ = transport.request("GET", f"{OPENROUTER_BASE}/api/v1/key", headers=_auth_headers(api_key))
    status_credits, credits_body, _ = transport.request("GET", f"{OPENROUTER_BASE}/api/v1/credits", headers=_auth_headers(api_key))
    if status_key != 200 or status_credits != 200:
        raise BudgetViolation("Unable to read both key and account balances")
    key_data = key_body.get("data", {})
    credit_data = credits_body.get("data", {})
    required_key = ("limit", "usage", "limit_remaining")
    required_credit = ("total_credits", "total_usage")
    if any(key_data.get(name) is None for name in required_key) or any(credit_data.get(name) is None for name in required_credit):
        raise BudgetViolation("Balance response is missing required numeric fields")
    values = [float(key_data[name]) for name in required_key] + [float(credit_data[name]) for name in required_credit]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise BudgetViolation("Balance response contains an invalid number")
    return AccountBalance(
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        key_limit_usd=float(key_data["limit"]),
        key_usage_usd=float(key_data["usage"]),
        key_remaining_usd=float(key_data["limit_remaining"]),
        account_total_credits_usd=float(credit_data["total_credits"]),
        account_total_usage_usd=float(credit_data["total_usage"]),
        account_remaining_usd=float(credit_data["total_credits"]) - float(credit_data["total_usage"]),
    )


class OpenRouterClient:
    def __init__(self, *, api_key: str, budget: BudgetController, transport: JSONTransport | None = None) -> None:
        self._api_key = api_key
        self._budget = budget
        self._transport = transport or UrllibJSONTransport()

    def generate_findings(
        self,
        *,
        contract: ProviderContract,
        image: SanitizedImage,
        item_id: str,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any],
        stage: str,
        response_path: Path,
        balance: AccountBalance,
        estimated_max_cost_usd: float = 0.05,
        max_tokens: int = 192,
        strict_schema_required: bool = True,
    ) -> dict[str, Any]:
        def validate(content: str) -> tuple[bool, str | None, dict[str, Any]]:
            parsed = parse_bare_findings(content)
            return parsed.valid, parsed.error, {"parsed_findings": list(parsed.findings) if parsed.valid else []}

        return self._complete_strict_json(
            contract=contract,
            image=image,
            item_id=item_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=response_format,
            stage=stage,
            response_path=response_path,
            balance=balance,
            estimated_max_cost_usd=estimated_max_cost_usd,
            max_tokens=max_tokens,
            response_kind="findings",
            validator=validate,
            strict_schema_required=strict_schema_required,
        )

    def audit_findings(
        self,
        *,
        contract: ProviderContract,
        image: SanitizedImage,
        item_id: str,
        candidate_findings: list[str],
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any],
        stage: str,
        response_path: Path,
        balance: AccountBalance,
        estimated_max_cost_usd: float = 0.05,
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        if not 0 <= len(candidate_findings) <= 5 or any(
            not isinstance(finding, str) or not finding.strip() for finding in candidate_findings
        ):
            raise ContractViolation("Cross-audit candidates must contain zero to five non-empty findings")
        anonymous_candidate = json.dumps(
            {"findings": candidate_findings}, ensure_ascii=False, separators=(",", ":")
        )

        def validate(content: str) -> tuple[bool, str | None, dict[str, Any]]:
            parsed = parse_judge_result(content, len(candidate_findings))
            return parsed.valid, parsed.error, {
                "parsed_judge": {
                    "statuses": list(parsed.statuses) if parsed.valid else [],
                    "forbidden_inference": parsed.forbidden_inference if parsed.valid else None,
                }
            }

        return self._complete_strict_json(
            contract=contract,
            image=image,
            item_id=item_id,
            system_prompt=system_prompt,
            user_prompt=f"{user_prompt}\n\nAnonymous candidate JSON:\n{anonymous_candidate}",
            response_format=response_format,
            stage=stage,
            response_path=response_path,
            balance=balance,
            estimated_max_cost_usd=estimated_max_cost_usd,
            max_tokens=max_tokens,
            response_kind="judge",
            validator=validate,
            strict_schema_required=True,
        )

    def audit_raw_candidate(
        self,
        *,
        contract: ProviderContract,
        image: SanitizedImage,
        item_id: str,
        candidate_raw: str,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any],
        stage: str,
        response_path: Path,
        balance: AccountBalance,
        estimated_max_cost_usd: float = 0.05,
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        if not isinstance(candidate_raw, str) or not candidate_raw.strip():
            raise ContractViolation("Raw cross-audit candidate must be a non-empty string")
        if len(candidate_raw) > 4096:
            raise ContractViolation("Raw cross-audit candidate exceeds the frozen audit limit")
        anonymous_candidate = json.dumps(
            {"raw_candidate": candidate_raw}, ensure_ascii=False, separators=(",", ":")
        )

        def validate(content: str) -> tuple[bool, str | None, dict[str, Any]]:
            parsed = parse_judge_result(content, None)
            return parsed.valid, parsed.error, {
                "parsed_judge": {
                    "statuses": list(parsed.statuses) if parsed.valid else [],
                    "forbidden_inference": parsed.forbidden_inference if parsed.valid else None,
                }
            }

        return self._complete_strict_json(
            contract=contract,
            image=image,
            item_id=item_id,
            system_prompt=system_prompt,
            user_prompt=f"{user_prompt}\n\nAnonymous raw candidate follows:\n{anonymous_candidate}",
            response_format=response_format,
            stage=stage,
            response_path=response_path,
            balance=balance,
            estimated_max_cost_usd=estimated_max_cost_usd,
            max_tokens=max_tokens,
            response_kind="raw_judge",
            validator=validate,
            strict_schema_required=True,
        )

    def audit_pair(
        self,
        *,
        contract: ProviderContract,
        image: SanitizedImage,
        item_id: str,
        candidate_a_raw: str,
        candidate_b_raw: str,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any],
        stage: str,
        response_path: Path,
        balance: AccountBalance,
        estimated_max_cost_usd: float = 0.005,
        max_tokens: int = 1536,
    ) -> dict[str, Any]:
        from .s7 import parse_pair_judge_result

        for value in (candidate_a_raw, candidate_b_raw):
            if not isinstance(value, str) or not value.strip() or len(value) > 4096:
                raise ContractViolation("Paired audit candidates must be non-empty strings of at most 4096 characters")
        anonymous_pair = json.dumps(
            {"candidate_a_raw": candidate_a_raw, "candidate_b_raw": candidate_b_raw},
            ensure_ascii=False,
            separators=(",", ":"),
        )

        def validate(content: str) -> tuple[bool, str | None, dict[str, Any]]:
            parsed = parse_pair_judge_result(content)
            return parsed.valid, parsed.error, {
                "parsed_pair_judge": {
                    "candidate_a": {
                        "claims": [
                            {"claim": claim, "status": status}
                            for claim, status in (parsed.candidate_a.claims if parsed.candidate_a else ())
                        ],
                        "forbidden_inference": parsed.candidate_a.forbidden_inference if parsed.candidate_a else None,
                    },
                    "candidate_b": {
                        "claims": [
                            {"claim": claim, "status": status}
                            for claim, status in (parsed.candidate_b.claims if parsed.candidate_b else ())
                        ],
                        "forbidden_inference": parsed.candidate_b.forbidden_inference if parsed.candidate_b else None,
                    },
                    "preference": parsed.preference,
                }
            }

        return self._complete_strict_json(
            contract=contract,
            image=image,
            item_id=item_id,
            system_prompt=system_prompt,
            user_prompt=f"{user_prompt}\n\nAnonymous candidate pair:\n{anonymous_pair}",
            response_format=response_format,
            stage=stage,
            response_path=response_path,
            balance=balance,
            estimated_max_cost_usd=estimated_max_cost_usd,
            max_tokens=max_tokens,
            response_kind="paired_judge",
            validator=validate,
            strict_schema_required=True,
        )

    def _complete_strict_json(
        self,
        *,
        contract: ProviderContract,
        image: SanitizedImage,
        item_id: str,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any],
        stage: str,
        response_path: Path,
        balance: AccountBalance,
        estimated_max_cost_usd: float,
        max_tokens: int,
        response_kind: str,
        validator: Callable[[str], tuple[bool, str | None, dict[str, Any]]],
        strict_schema_required: bool,
    ) -> dict[str, Any]:
        if not item_id.startswith("item_") or len(item_id) < 20:
            raise PrivacyViolation("External item_id must be random and anonymous")
        if not image.approved:
            raise PrivacyViolation("Only a privacy-approved re-encoded image can be sent")
        authorization = self._budget.authorize(
            stage=stage,
            model=contract.model,
            estimated_max_cost_usd=estimated_max_cost_usd,
            balance=balance,
        )
        payload = {
            "model": contract.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Anonymous item ID: {item_id}\n\n{user_prompt}"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image.png_bytes).decode("ascii")}},
                    ],
                },
            ],
            "provider": contract.routing_object(),
            "response_format": response_format,
            "reasoning": {"effort": "low", "exclude": True},
            "max_tokens": max_tokens,
            "stream": False,
        }
        assert_external_payload_private(payload, allowed_item_id=item_id)
        headers = {
            **_auth_headers(self._api_key),
            "Content-Type": "application/json",
            "X-OpenRouter-Metadata": "enabled",
            "HTTP-Referer": "https://localhost/PATHO-LORA-SFT-01",
            "X-Title": "PATHO-LORA-SFT-01",
        }
        status, body, response_headers = self._transport.request(
            "POST", f"{OPENROUTER_BASE}/api/v1/chat/completions", headers=headers, body=payload
        )
        if status != 200:
            raise ContractViolation(f"OpenRouter returned unexpected status {status}")
        response_model = body.get("model")
        response_provider = body.get("provider")
        choices = body.get("choices")
        content = choices[0].get("message", {}).get("content") if isinstance(choices, list) and len(choices) == 1 else None
        if isinstance(content, str):
            strict_valid, strict_error, parsed_fields = validator(content)
        else:
            strict_valid, strict_error, parsed_fields = False, "completion_content_not_string", {}
        usage = body.get("usage")
        cost_value = usage.get("cost") if isinstance(usage, dict) else None
        request_id = str(body.get("id") or response_headers.get("x-generation-id") or "")
        safe_record = {
            "schema_version": "patho_lora_openrouter_response_v1",
            "response_kind": response_kind,
            "item_id": item_id,
            "image_sha256": image.sha256,
            "model": response_model,
            "provider": response_provider,
            "provider_policy": contract.routing_object(),
            "request_id": request_id,
            "content": content if isinstance(content, str) else None,
            **parsed_fields,
            "strict_schema_valid": strict_valid,
            "strict_schema_error": strict_error,
            "strict_schema_required": strict_schema_required,
            "requested_max_tokens": max_tokens,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
                "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
                "total_tokens": usage.get("total_tokens") if isinstance(usage, dict) else None,
                "cost": cost_value,
            },
            "request_body_saved": False,
            "image_base64_saved": False,
        }
        atomic_write_json(response_path, safe_record, mode=0o600)
        if cost_value is None:
            raise BudgetViolation("OpenRouter response did not contain usage.cost")
        cost = float(cost_value)
        ledger_request_id = request_id or "missing_" + sha256_bytes(
            json.dumps(safe_record, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )[:24]
        self._budget.record_actual(authorization, actual_cost_usd=cost, request_id=ledger_request_id)
        if not request_id:
            raise ContractViolation("OpenRouter response did not contain a request/generation ID")
        if response_model != contract.model:
            raise ProviderViolation(f"Response model mismatch: expected {contract.model!r}, got {response_model!r}")
        if response_provider != contract.response_provider:
            raise ProviderViolation(
                f"Response provider mismatch: expected {contract.response_provider!r}, got {response_provider!r}"
            )
        if not isinstance(choices, list) or len(choices) != 1:
            raise ContractViolation("Expected exactly one completion choice")
        if not isinstance(content, str):
            raise ContractViolation("Completion content is not a string")
        if strict_schema_required and not strict_valid:
            raise ContractViolation(f"Strict response schema failed: {safe_record['strict_schema_error']}")
        return safe_record


def public_balance_record(balance: AccountBalance) -> dict[str, Any]:
    record = asdict(balance)
    record["account_settings"] = {
        "private_input_output_logging": False,
        "openrouter_use_of_inputs_outputs": False,
        "evidence_kind": "explicit_user_attestation",
    }
    return record
