from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import ContractViolation, load_json, sha256_file


EXPERIMENT_ID = "PATHO-LORA-SFT-01"
TASK_NAME = "schema_morphology_v1"
CANVAS_SIZE = (784, 784)
SPLIT_SEED = 20260827
PROJECT_COST_CAP_USD = 9.50
MINIMUM_ACCOUNT_RESERVE_USD = 0.50
TEST_SPLIT_NAMES = frozenset({"test", "test10"})
MODEL_SLUGS = frozenset({"google/gemini-3.7-flash", "openai/gpt-5.6-luna"})


@dataclass(frozen=True)
class ProviderContract:
    model: str
    routing_provider: str
    response_provider: str
    require_zdr: bool

    def __post_init__(self) -> None:
        if self.model not in MODEL_SLUGS:
            raise ContractViolation(f"Unapproved model slug: {self.model}")
        expected = {
            "google/gemini-3.7-flash": ("Google", "Google", True),
            "openai/gpt-5.6-luna": ("OpenAI", "OpenAI", False),
        }[self.model]
        if (self.routing_provider, self.response_provider, self.require_zdr) != expected:
            raise ContractViolation(f"Provider contract differs from frozen route for {self.model}")

    def routing_object(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "only": [self.routing_provider],
        }
        if self.require_zdr:
            result["zdr"] = True
        return result


GEMINI_CONTRACT = ProviderContract("google/gemini-3.7-flash", "Google", "Google", True)
LUNA_CONTRACT = ProviderContract("openai/gpt-5.6-luna", "OpenAI", "OpenAI", False)


def verify_frozen_hash(path: Path, expected_sha256: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ContractViolation(f"Frozen input hash mismatch: {path}")
    return actual


def load_and_verify_contract(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ContractViolation("Configuration experiment_id mismatch")
    if value.get("test10", {}).get("authorized") is not False:
        raise ContractViolation("Test10 must remain explicitly unauthorized")
    if float(value.get("budget", {}).get("project_cost_cap_usd", -1)) != PROJECT_COST_CAP_USD:
        raise ContractViolation("Project cost cap changed")
    return value
