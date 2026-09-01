from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Any

from .common import ContractViolation, PrivacyViolation, atomic_write_json, load_json, sha256_json


class AtomicRunJournal:
    """Immutable completion records and an atomic index for safe resume."""

    def __init__(self, root: Path) -> None:
        self._lock = threading.RLock()
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.records = root / "records"
        self.index_path = root / "journal.json"
        self.records.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.index_path.exists():
            self.index = load_json(self.index_path)
        else:
            self.index = {"schema_version": "patho_lora_atomic_journal_v1", "items": {}}
            atomic_write_json(self.index_path, self.index, mode=0o600)

    def completed(self, item_id: str, task_fingerprint: str) -> bool:
        with self._lock:
            entry = self.index["items"].get(item_id)
            return bool(entry and entry.get("status") == "complete" and entry.get("task_fingerprint") == task_fingerprint)

    def entry(self, item_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self.index["items"].get(item_id)
            return dict(value) if value is not None else None

    def entries_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a detached journal-index snapshot for concurrent audit code."""
        with self._lock:
            return {
                str(item_id): dict(entry)
                for item_id, entry in self.index["items"].items()
            }

    def begin_call(self, *, item_id: str, task_fingerprint: str, intent: dict[str, Any]) -> None:
        with self._lock:
            if any(key in intent for key in ("request_body", "image_base64", "authorization")):
                raise PrivacyViolation("Journal intent contains forbidden request material")
            existing = self.index["items"].get(item_id)
            if existing is not None:
                raise ContractViolation("Journal task already exists; inspect it before any retry")
            self.index["items"][item_id] = {
                "status": "pending_external_call",
                "task_fingerprint": task_fingerprint,
                "intent": intent,
                "recorded_unix": time.time(),
            }
            atomic_write_json(self.index_path, self.index, mode=0o600)

    def load_complete(self, *, item_id: str, task_fingerprint: str) -> dict[str, Any]:
        with self._lock:
            entry = self.index["items"].get(item_id)
            if not entry or entry.get("status") != "complete" or entry.get("task_fingerprint") != task_fingerprint:
                raise ContractViolation("Journal task is not complete for the requested fingerprint")
            return load_json(self.root / entry["record_relpath"])

    def record_failed_response(
        self,
        *,
        item_id: str,
        task_fingerprint: str,
        record: dict[str, Any],
        failure_reason: str,
    ) -> Path:
        with self._lock:
            if any(key in record for key in ("request_body", "image_base64", "authorization")):
                raise PrivacyViolation("Failed journal response contains forbidden request material")
            existing = self.index["items"].get(item_id)
            if not existing or existing.get("status") != "pending_external_call":
                raise ContractViolation("Only a pending external call can become a failed response")
            if existing.get("task_fingerprint") != task_fingerprint:
                raise ContractViolation("Failed response task fingerprint mismatch")
            record_path = self.records / f"{item_id}.failed.json"
            if record_path.exists():
                raise ContractViolation("Failed response record already exists")
            atomic_write_json(record_path, record, mode=0o600)
            self.index["items"][item_id] = {
                "status": "failed_external_response",
                "task_fingerprint": task_fingerprint,
                "record_sha256": sha256_json(record),
                "record_relpath": str(record_path.relative_to(self.root)),
                "failure_reason": failure_reason,
                "recorded_unix": time.time(),
            }
            atomic_write_json(self.index_path, self.index, mode=0o600)
            return record_path

    def record_confirmed_unbilled_failure(
        self,
        *,
        item_id: str,
        task_fingerprint: str,
        reconciliation: dict[str, Any],
    ) -> None:
        with self._lock:
            existing = self.index["items"].get(item_id)
            if not existing or existing.get("status") != "pending_external_call":
                raise ContractViolation("Only a pending call can be confirmed unbilled")
            if existing.get("task_fingerprint") != task_fingerprint:
                raise ContractViolation("Confirmed-unbilled task fingerprint mismatch")
            if reconciliation.get("cost_reconciled") is not True:
                raise ContractViolation("Unbilled confirmation lacks passing cost reconciliation")
            self.index["items"][item_id] = {
                **existing,
                "status": "confirmed_unbilled_network_failure",
                "billing_confirmation": reconciliation,
                "recorded_unix": time.time(),
            }
            atomic_write_json(self.index_path, self.index, mode=0o600)

    def record_complete(self, *, item_id: str, task_fingerprint: str, record: dict[str, Any]) -> Path:
        with self._lock:
            if any(key in record for key in ("request_body", "image_base64", "authorization")):
                raise PrivacyViolation("Journal record contains forbidden request material")
            existing = self.index["items"].get(item_id)
            record_sha = sha256_json(record)
            if existing:
                if existing.get("status") == "complete" and existing.get("task_fingerprint") == task_fingerprint and existing.get("record_sha256") == record_sha:
                    return self.root / existing["record_relpath"]
                if existing.get("status") != "pending_external_call" or existing.get("task_fingerprint") != task_fingerprint:
                    raise ContractViolation("Refusing to overwrite an existing journal item")
            record_path = self.records / f"{item_id}.json"
            if record_path.exists():
                raise ContractViolation("Orphan completion record already exists")
            atomic_write_json(record_path, record, mode=0o600)
            self.index["items"][item_id] = {
                "status": "complete",
                "task_fingerprint": task_fingerprint,
                "record_sha256": record_sha,
                "record_relpath": str(record_path.relative_to(self.root)),
                "recorded_unix": time.time(),
            }
            atomic_write_json(self.index_path, self.index, mode=0o600)
            return record_path
