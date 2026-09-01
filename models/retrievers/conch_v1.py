from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .base import (
    PathologyRetriever,
    RetrieverSpec,
    canonical_sha256,
    normalize_and_validate,
    sha256_file,
)


EXPECTED_MISSING_KEY_COUNT = 315
EXPECTED_MISSING_KEYS_SHA256 = (
    "bc21feb1ab170230079c4b5c70cac39ae6fd3591f90a454ce3ac09388b3a373b"
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
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to resolve CONCH source revision: {path}") from exc


class CONCHv1Retriever(PathologyRetriever):
    def __init__(
        self,
        library_path: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
    ):
        library_path = Path(library_path).resolve()
        checkpoint_path = Path(checkpoint_path).resolve()
        if not library_path.is_dir():
            raise FileNotFoundError(f"CONCH library directory not found: {library_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"CONCH checkpoint not found: {checkpoint_path}")
        config_path = (
            library_path
            / "conch"
            / "open_clip_custom"
            / "model_configs"
            / "conch_ViT-B-16.json"
        )
        if not config_path.is_file():
            raise FileNotFoundError(f"CONCH model config not found: {config_path}")
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(model_config.get("embed_dim", -1)) != 512:
            raise ValueError("CONCH v1 contrastive embedding dimension must be 512")
        if int((model_config.get("vision_cfg") or {}).get("image_size", -1)) != 448:
            raise ValueError("CONCH v1 native image size must be 448")

        sys.path.insert(0, str(library_path))
        from conch.open_clip_custom import (
            create_model_from_pretrained,
            get_tokenizer,
            tokenize,
        )
        from conch.open_clip_custom.factory import read_state_dict

        import conch.open_clip_custom as conch_open_clip_custom

        imported_source = Path(conch_open_clip_custom.__file__).resolve()
        if library_path not in imported_source.parents:
            raise RuntimeError(
                f"Imported CONCH source {imported_source} is outside {library_path}"
            )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.preprocess = create_model_from_pretrained(
            model_cfg="conch_ViT-B-16",
            checkpoint_path=str(checkpoint_path),
            device=self.device,
        )
        self.model.eval()
        self.tokenizer = get_tokenizer()
        self._tokenize = tokenize

        state_dict = read_state_dict(str(checkpoint_path), map_location="cpu")
        model_keys = set(self.model.state_dict())
        checkpoint_keys = set(state_dict)
        self.missing_keys = sorted(model_keys - checkpoint_keys)
        self.unexpected_keys = sorted(checkpoint_keys - model_keys)
        missing_keys_sha256 = canonical_sha256(self.missing_keys)
        if (
            len(self.missing_keys) != EXPECTED_MISSING_KEY_COUNT
            or missing_keys_sha256 != EXPECTED_MISSING_KEYS_SHA256
            or self.unexpected_keys
        ):
            raise RuntimeError(
                "CONCH checkpoint key audit failed: "
                f"missing_count={len(self.missing_keys)}, "
                f"missing_sha256={missing_keys_sha256}, "
                f"unexpected={self.unexpected_keys[:10]}"
            )
        self.missing_keys_sha256 = missing_keys_sha256

        revision, dirty = _git_state(library_path)
        preprocess_contract = {
            "model_id": "conch_ViT-B-16",
            "input_size": 448,
            "image_mode": "RGB",
            "resize": "bicubic",
            "center_crop": 448,
            "image_mean": list(getattr(self.model.visual, "image_mean")),
            "image_std": list(getattr(self.model.visual, "image_std")),
            "projection": "contrastive",
            "l2_normalized": True,
            "source_revision": revision,
        }
        self.spec = RetrieverSpec(
            backend="conch_v1",
            model_id="conch_ViT-B-16",
            embedding_dim=512,
            input_size=448,
            checkpoint_sha256=sha256_file(checkpoint_path),
            source_revision=revision,
            source_dirty=dirty,
            model_config_sha256=sha256_file(config_path),
            preprocess_sha256=canonical_sha256(preprocess_contract),
        )

    def encode_text(self, texts: Sequence[str], batch_size: int = 1) -> np.ndarray:
        values = self._validate_texts(texts)
        batch_size = self._validate_batch_size(batch_size)
        outputs = []
        with torch.inference_mode():
            for offset in range(0, len(values), batch_size):
                batch = values[offset : offset + batch_size]
                tokens = self._tokenize(tokenizer=self.tokenizer, texts=batch).to(self.device)
                output = self.model.encode_text(tokens, normalize=True)
                outputs.append(output.detach().float().cpu().numpy())
        return normalize_and_validate(
            np.concatenate(outputs, axis=0),
            expected_count=len(values),
            embedding_dim=self.spec.embedding_dim,
            operation="CONCH v1 text encoding",
        )

    def encode_images(
        self, images: Sequence[Image.Image], batch_size: int = 4
    ) -> np.ndarray:
        values = self._validate_images(images)
        batch_size = self._validate_batch_size(batch_size)
        outputs = []
        with torch.inference_mode():
            for offset in range(0, len(values), batch_size):
                batch = values[offset : offset + batch_size]
                tensor = torch.stack([self.preprocess(image) for image in batch]).to(
                    self.device, dtype=torch.float32
                )
                output = self.model.encode_image(
                    tensor, proj_contrast=True, normalize=True
                )
                outputs.append(output.detach().float().cpu().numpy())
        return normalize_and_validate(
            np.concatenate(outputs, axis=0),
            expected_count=len(values),
            embedding_dim=self.spec.embedding_dim,
            operation="CONCH v1 image encoding",
        )
