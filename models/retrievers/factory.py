from __future__ import annotations

from pathlib import Path

from .base import PathologyRetriever


SUPPORTED_RETRIEVERS = ("plip", "conch_v1")


def create_retriever(
    backend: str,
    library_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
) -> PathologyRetriever:
    backend = str(backend).strip().lower()
    if backend == "plip":
        from .plip import PLIPRetriever

        return PLIPRetriever(library_path, checkpoint_path)
    if backend == "conch_v1":
        from .conch_v1 import CONCHv1Retriever

        return CONCHv1Retriever(
            library_path, checkpoint_path, device=device
        )
    raise ValueError(
        f"Unsupported retriever backend {backend!r}; expected one of {SUPPORTED_RETRIEVERS}"
    )
