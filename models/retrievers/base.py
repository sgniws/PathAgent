from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RetrieverSpec:
    backend: str
    model_id: str
    embedding_dim: int
    input_size: int
    checkpoint_sha256: str
    source_revision: str
    source_dirty: bool
    model_config_sha256: str
    preprocess_sha256: str
    output_dtype: str = "float32"
    normalized: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_and_validate(
    embeddings: Any,
    *,
    expected_count: int,
    embedding_dim: int,
    operation: str,
) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape != (expected_count, embedding_dim):
        raise ValueError(
            f"{operation} returned shape {array.shape}; expected "
            f"({expected_count}, {embedding_dim})"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{operation} returned non-finite embeddings")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{operation} returned a zero-norm embedding")
    normalized = array / norms
    if not np.isfinite(normalized).all():
        raise ValueError(f"{operation} normalization produced non-finite values")
    return np.ascontiguousarray(normalized, dtype=np.float32)


class PathologyRetriever(ABC):
    """Common normalized contrastive-embedding contract used by PathAgent."""

    spec: RetrieverSpec

    @abstractmethod
    def encode_text(self, texts: Sequence[str], batch_size: int = 1) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_images(
        self, images: Sequence[Image.Image], batch_size: int = 4
    ) -> np.ndarray:
        raise NotImplementedError

    def _validate_texts(self, texts: Sequence[str]) -> list[str]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("Retriever text input must be a sequence of strings")
        values = [str(value).strip() for value in texts]
        if not values or any(not value for value in values):
            raise ValueError("Retriever text input must contain non-empty strings")
        return values

    def _validate_images(self, images: Sequence[Image.Image]) -> list[Image.Image]:
        if isinstance(images, Image.Image):
            raise TypeError("Retriever image input must be a sequence of PIL images")
        values = list(images)
        if not values:
            raise ValueError("Retriever image input must not be empty")
        if any(not isinstance(image, Image.Image) for image in values):
            raise TypeError("Retriever image input must contain PIL images")
        return [image.convert("RGB") for image in values]

    @staticmethod
    def _validate_batch_size(batch_size: int) -> int:
        value = int(batch_size)
        if value < 1:
            raise ValueError("Retriever batch_size must be positive")
        return value
