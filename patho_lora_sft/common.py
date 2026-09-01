from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


class ContractViolation(RuntimeError):
    """A frozen experiment contract was violated."""


class PrivacyViolation(ContractViolation):
    """An artifact or request failed the privacy contract."""


class BudgetViolation(ContractViolation):
    """A request cannot be made within the frozen budget."""


class ProviderViolation(ContractViolation):
    """The requested or returned model/provider identity changed."""


class NetworkBillingUnconfirmed(ContractViolation):
    """A network failure occurred after request authorization and billing is unknown."""


class TransientOpenRouterError(NetworkBillingUnconfirmed):
    """A retryable OpenRouter HTTP response with privacy-safe metadata only."""

    def __init__(
        self,
        *,
        status_code: int,
        error_type: str | None,
        provider_code: str | None,
        retry_after_seconds: float | None,
        rate_limit_headers: dict[str, str],
    ) -> None:
        self.status_code = status_code
        self.error_type = error_type
        self.provider_code = provider_code
        self.retry_after_seconds = retry_after_seconds
        self.rate_limit_headers = dict(rate_limit_headers)
        super().__init__(
            f"OpenRouter transient HTTP failure: {status_code}; "
            f"error_type={error_type or 'unknown'}; response_body_redacted=true"
        )

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "error_type": self.error_type,
            "provider_code": self.provider_code,
            "retry_after_seconds": self.retry_after_seconds,
            "rate_limit_headers": dict(self.rate_limit_headers),
            "response_body_redacted": True,
        }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str, mode: int = 0o600) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def atomic_write_json(path: Path, value: Any, mode: int = 0o600) -> None:
    atomic_write_text(path, canonical_json(value) + "\n", mode=mode)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: int = 0o600) -> None:
    atomic_write_text(path, "".join(canonical_json(row) + "\n" for row in rows), mode=mode)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_new_directory(path: Path, mode: int = 0o700) -> None:
    try:
        path.mkdir(parents=True, mode=mode, exist_ok=False)
    except FileExistsError as exc:
        raise ContractViolation(f"Refusing to overwrite existing directory: {path}") from exc
    os.chmod(path, mode)


def redact_path(path: Path) -> dict[str, str]:
    """Represent a private path in public provenance without its components."""
    return {"basename_sha256": sha256_bytes(path.name.encode("utf-8"))}


def checkpoint_fingerprint(model_path: Path) -> dict[str, Any]:
    if not model_path.is_dir():
        raise ContractViolation(f"Checkpoint directory does not exist: {model_path}")
    entries: list[dict[str, Any]] = []
    for path in sorted(model_path.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        entry: dict[str, Any] = {
            "relative_path": str(path.relative_to(model_path)),
            "size_bytes": stat.st_size,
        }
        if stat.st_size <= 16 * 1024 * 1024:
            entry["sha256"] = sha256_file(path)
        else:
            entry["mtime_ns"] = stat.st_mtime_ns
        entries.append(entry)
    return {
        "path": str(model_path.resolve()),
        "fingerprint_kind": "all_files_small_content_large_stat_v1",
        "sha256": sha256_json(entries),
        "file_count": len(entries),
    }


def directory_fingerprint(root: Path) -> dict[str, Any]:
    """Hash every adapter file without exposing its absolute local path."""
    if not root.is_dir():
        raise ContractViolation(f"Adapter directory does not exist: {root}")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            entries.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {"file_count": len(entries), "sha256": sha256_json(entries), "files": entries}
