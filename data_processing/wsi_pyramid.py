"""Auditable WSI-native observations for PathAgent 5x/10x/20x evidence."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

try:
    import openslide
except ImportError:  # Keep manifest/geometry helpers usable in lightweight test environments.
    openslide = None


SUPPORTED_MAGNIFICATIONS = (5, 10, 20)
FOCUS_STRICT_BLANK_TISSUE_LT = 0.01
FOCUS_STRICT_BLANK_WHITE_GTE = 0.98
FOCUS_DEFINITE_NONBLANK_TISSUE_GTE = 0.05
FOCUS_DEFINITE_NONBLANK_WHITE_LT = 0.95
FOCUS_AMBIGUOUS_PENALTY_SCALE = 0.05


@dataclass(frozen=True)
class WSIRegion:
    slide_id: str
    patch_id: str
    x_level0: int
    y_level0: int
    width_level0: int
    height_level0: int
    mpp_x: float
    mpp_y: float
    magnification: int = 5
    parent_patch_id: str | None = None

    @property
    def physical_width_um(self) -> float:
        return self.width_level0 * self.mpp_x

    @property
    def physical_height_um(self) -> float:
        return self.height_level0 * self.mpp_y

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "physical_width_um": round(self.physical_width_um, 6),
            "physical_height_um": round(self.physical_height_um, 6),
            "coordinate_system": "level0",
        }


@dataclass(frozen=True)
class WSIObservation:
    image: Image.Image
    metadata: dict[str, Any]


def load_wsi_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {}
    for row in rows:
        slide_id = str(row["slide_id"])
        if slide_id in result:
            raise ValueError(f"Duplicate slide_id in WSI manifest: {slide_id}")
        result[slide_id] = row
    return result


def load_patch_manifest(path: str | Path, selected_only: bool = True) -> dict[str, WSIRegion]:
    regions = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if selected_only and not row.get("selected"):
            continue
        patch_id = str(row["patch_id"])
        if patch_id in regions:
            raise ValueError(f"Duplicate patch_id in patch manifest: {patch_id}")
        regions[patch_id] = WSIRegion(
            slide_id=str(row["slide_id"]),
            patch_id=patch_id,
            x_level0=int(row["x_level0"]),
            y_level0=int(row["y_level0"]),
            width_level0=int(row["width_level0"]),
            height_level0=int(row["height_level0"]),
            mpp_x=float(row["mpp_x"]),
            mpp_y=float(row["mpp_y"]),
            magnification=5,
        )
    return regions


def load_plip_h5(path: str | Path) -> tuple[list[str], np.ndarray]:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("status") != "complete":
            raise RuntimeError(f"PLIP feature file is not complete: {path}")
        patch_ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["patch_id"][:]
        ]
        features = handle["features"][:].astype(np.float32)
    if features.ndim != 2 or features.shape[0] != len(patch_ids):
        raise ValueError(f"Invalid PLIP feature shape in {path}: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite PLIP feature in {path}")
    return patch_ids, features


def load_binary_mask(path: str | Path) -> np.ndarray:
    mask_path = Path(path)
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    if mask.ndim != 2 or not mask.size:
        raise ValueError(f"Invalid binary mask: {mask_path}")
    return mask


def mask_fraction_for_region(
    mask: np.ndarray,
    slide_width: int,
    slide_height: int,
    region: WSIRegion,
) -> float:
    """Map a Level-0 region to a low-resolution whole-slide mask."""
    if slide_width <= 0 or slide_height <= 0:
        raise ValueError("Slide dimensions must be positive")
    mask_height, mask_width = mask.shape
    left_raw = int(math.floor(region.x_level0 / slide_width * mask_width))
    top_raw = int(math.floor(region.y_level0 / slide_height * mask_height))
    right_raw = int(
        math.ceil((region.x_level0 + region.width_level0) / slide_width * mask_width)
    )
    bottom_raw = int(
        math.ceil((region.y_level0 + region.height_level0) / slide_height * mask_height)
    )
    left = max(0, min(mask_width, left_raw))
    top = max(0, min(mask_height, top_raw))
    right = max(left, min(mask_width, right_raw))
    bottom = max(top, min(mask_height, bottom_raw))
    expected_area = max(1, right_raw - left_raw) * max(1, bottom_raw - top_raw)
    return float(mask[top:bottom, left:right].sum() / expected_area)


def image_white_fraction(image: Image.Image) -> float:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    brightness = rgb.mean(axis=2)
    white = (brightness >= 240) & (hsv[:, :, 1] <= 15)
    return float(white.mean())


def classify_focus_candidate(tissue_fraction: float, white_fraction: float) -> str:
    if (
        tissue_fraction < FOCUS_STRICT_BLANK_TISSUE_LT
        and white_fraction >= FOCUS_STRICT_BLANK_WHITE_GTE
    ):
        return "strict_blank"
    if (
        tissue_fraction >= FOCUS_DEFINITE_NONBLANK_TISSUE_GTE
        and white_fraction < FOCUS_DEFINITE_NONBLANK_WHITE_LT
    ):
        return "definite_nonblank"
    return "ambiguous"


def focus_ambiguous_risk(
    tissue_fraction: float, white_fraction: float, candidate_class: str
) -> float:
    if candidate_class != "ambiguous":
        return 0.0
    tissue_risk = float(np.clip((0.05 - tissue_fraction) / 0.04, 0.0, 1.0))
    white_risk = float(np.clip((white_fraction - 0.95) / 0.03, 0.0, 1.0))
    return max(tissue_risk, white_risk)


def rank_focus_candidates_c(
    candidate_ids: list[str],
    raw_scores: np.ndarray,
    candidate_meta: list[dict[str, Any]],
    top_k: int = 2,
) -> tuple[list[tuple[str, float]], list[dict[str, Any]]]:
    """Apply strategy C and return selected candidates plus fully auditable ranking rows."""
    scores = np.atleast_1d(np.asarray(raw_scores, dtype=np.float64)).reshape(-1)
    if len(candidate_ids) != len(candidate_meta) or len(candidate_ids) != len(scores):
        raise ValueError("Focus candidate IDs, scores, and metadata must have equal lengths")
    if top_k < 1:
        raise ValueError("Focus top_k must be positive")

    ranking_rows = []
    for index, (candidate_id, score, meta) in enumerate(
        zip(candidate_ids, scores, candidate_meta)
    ):
        candidate_class = str(meta["candidate_class"])
        padding = meta.get("padding") or {}
        has_padding = any(int(value) > 0 for value in padding.values())
        hard_rejection_reasons = []
        if candidate_class == "strict_blank":
            hard_rejection_reasons.append("strict_blank")
        if has_padding:
            hard_rejection_reasons.append("padding")
        risk = focus_ambiguous_risk(
            float(meta["grandqc_tissue_fraction"]),
            float(meta["white_fraction"]),
            candidate_class,
        )
        adjusted_score = float(score - FOCUS_AMBIGUOUS_PENALTY_SCALE * risk)
        ranking_rows.append(
            {
                "patch_id": candidate_id,
                "candidate_index": index,
                "candidate_class": candidate_class,
                "plip_score": float(score),
                "ambiguous_risk": risk,
                "adjusted_score": adjusted_score,
                "eligible": not hard_rejection_reasons,
                "hard_rejection_reasons": hard_rejection_reasons,
            }
        )

    eligible_rows = [row for row in ranking_rows if row["eligible"]]
    eligible_rows.sort(
        key=lambda row: (
            row["adjusted_score"],
            row["plip_score"],
            -row["candidate_index"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(eligible_rows, 1):
        row["eligible_rank"] = rank
    selected = [
        (row["patch_id"], row["adjusted_score"])
        for row in eligible_rows[:top_k]
    ]
    return selected, ranking_rows


def partition_region(parent: WSIRegion, target_magnification: int) -> list[WSIRegion]:
    if parent.magnification != 5:
        raise ValueError("WSI focus must start from a 5x parent region")
    if target_magnification not in (10, 20):
        raise ValueError(f"Unsupported focus magnification: {target_magnification}")
    factor = target_magnification // 5
    x_edges = [parent.x_level0 + index * parent.width_level0 // factor for index in range(factor + 1)]
    y_edges = [parent.y_level0 + index * parent.height_level0 // factor for index in range(factor + 1)]
    children = []
    for row in range(factor):
        for column in range(factor):
            x0, x1 = x_edges[column], x_edges[column + 1]
            y0, y1 = y_edges[row], y_edges[row + 1]
            patch_id = (
                f"x{x0}_y{y0}_w{x1 - x0}_h{y1 - y0}_m{target_magnification}"
            )
            children.append(
                WSIRegion(
                    slide_id=parent.slide_id,
                    patch_id=patch_id,
                    x_level0=x0,
                    y_level0=y0,
                    width_level0=x1 - x0,
                    height_level0=y1 - y0,
                    mpp_x=parent.mpp_x,
                    mpp_y=parent.mpp_y,
                    magnification=target_magnification,
                    parent_patch_id=parent.patch_id,
                )
            )
    return children


def choose_read_level(
    level_downsamples: tuple[float, ...] | list[float],
    width_level0: int,
    height_level0: int,
    output_size: int,
) -> int:
    """Choose the coarsest native level that still supplies >= output pixels."""
    desired = max(
        1.0,
        min(float(width_level0) / output_size, float(height_level0) / output_size),
    )
    eligible = [
        index
        for index, downsample in enumerate(level_downsamples)
        if float(downsample) <= desired
    ]
    return eligible[-1] if eligible else 0


def _rgba_on_white(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")
    background = Image.new("RGBA", image.size, "white")
    return Image.alpha_composite(background, image).convert("RGB")


class WSIPyramidReader:
    def __init__(self, wsi_row: dict[str, Any]):
        if openslide is None:
            raise RuntimeError("WSI reading requires the openslide-python package")
        self.slide_id = str(wsi_row["slide_id"])
        self.slide_path = Path(wsi_row["slide_path"])
        if not self.slide_path.is_file():
            raise FileNotFoundError(self.slide_path)
        self.slide = openslide.OpenSlide(str(self.slide_path))

    def close(self) -> None:
        self.slide.close()

    def __enter__(self) -> "WSIPyramidReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def read(self, region: WSIRegion, output_size: int) -> WSIObservation:
        if region.slide_id != self.slide_id:
            raise ValueError(
                f"Region belongs to {region.slide_id}, reader belongs to {self.slide_id}"
            )
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        level = choose_read_level(
            self.slide.level_downsamples,
            region.width_level0,
            region.height_level0,
            output_size,
        )
        downsample = float(self.slide.level_downsamples[level])
        read_width = max(1, int(math.ceil(region.width_level0 / downsample)))
        read_height = max(1, int(math.ceil(region.height_level0 / downsample)))
        image = _rgba_on_white(
            self.slide.read_region(
                (region.x_level0, region.y_level0),
                level,
                (read_width, read_height),
            )
        )
        exact_width = max(1.0, region.width_level0 / downsample)
        exact_height = max(1.0, region.height_level0 / downsample)
        image = image.crop((0.0, 0.0, exact_width, exact_height))
        image = image.resize((output_size, output_size), Image.Resampling.BICUBIC)

        slide_width, slide_height = self.slide.dimensions
        padding = {
            "left_level0": max(0, -region.x_level0),
            "top_level0": max(0, -region.y_level0),
            "right_level0": max(
                0, region.x_level0 + region.width_level0 - slide_width
            ),
            "bottom_level0": max(
                0, region.y_level0 + region.height_level0 - slide_height
            ),
        }
        metadata = {
            **region.to_dict(),
            "slide_path": str(self.slide_path.resolve()),
            "read_level": level,
            "read_downsample": downsample,
            "read_size": [read_width, read_height],
            "output_size": [output_size, output_size],
            "padding": padding,
            "implicit_window_shift": False,
        }
        return WSIObservation(image=image, metadata=metadata)
