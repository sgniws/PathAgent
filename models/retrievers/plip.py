from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .base import (
    PathologyRetriever,
    RetrieverSpec,
    canonical_sha256,
    normalize_and_validate,
    sha256_file,
)


def _git_state(path: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unversioned", False


class PLIPRetriever(PathologyRetriever):
    def __init__(self, library_path: str | Path, checkpoint_path: str | Path):
        library_path = Path(library_path).resolve()
        checkpoint_path = Path(checkpoint_path).resolve()
        if not library_path.is_dir():
            raise FileNotFoundError(f"PLIP library directory not found: {library_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"PLIP checkpoint not found: {checkpoint_path}")
        sys.path.insert(0, str(library_path))
        from plip import PLIP

        self.model = PLIP(str(checkpoint_path))
        weight_path = (
            checkpoint_path / "pytorch_model.bin"
            if checkpoint_path.is_dir()
            else checkpoint_path
        )
        revision, dirty = _git_state(library_path)
        preprocess = {
            "implementation": "transformers.CLIPProcessor",
            "input_size": 224,
            "rgb": True,
            "normalized_output": True,
        }
        self.spec = RetrieverSpec(
            backend="plip",
            model_id="PLIP",
            embedding_dim=512,
            input_size=224,
            checkpoint_sha256=sha256_file(weight_path),
            source_revision=revision,
            source_dirty=dirty,
            model_config_sha256=(
                sha256_file(checkpoint_path / "config.json")
                if checkpoint_path.is_dir()
                and (checkpoint_path / "config.json").is_file()
                else canonical_sha256({"model": "PLIP"})
            ),
            preprocess_sha256=canonical_sha256(preprocess),
        )

    def encode_text(self, texts: Sequence[str], batch_size: int = 1) -> np.ndarray:
        values = self._validate_texts(texts)
        batch_size = self._validate_batch_size(batch_size)
        output = self.model.encode_text(values, batch_size=batch_size)
        return normalize_and_validate(
            output,
            expected_count=len(values),
            embedding_dim=self.spec.embedding_dim,
            operation="PLIP text encoding",
        )

    def encode_images(
        self, images: Sequence[Image.Image], batch_size: int = 4
    ) -> np.ndarray:
        values = self._validate_images(images)
        batch_size = self._validate_batch_size(batch_size)
        output = self.model.encode_images(values, batch_size=batch_size)
        return normalize_and_validate(
            output,
            expected_count=len(values),
            embedding_dim=self.spec.embedding_dim,
            operation="PLIP image encoding",
        )
