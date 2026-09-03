from __future__ import annotations

import io
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageOps, features

from .common import ContractViolation, PrivacyViolation, sha256_bytes
from .contracts import CANVAS_SIZE, TEST_SPLIT_NAMES


FORBIDDEN_EXTERNAL_KEYS = frozenset(
    {
        "filename",
        "file_name",
        "slide_id",
        "slide_path",
        "wsi_path",
        "patient_id",
        "case_id",
        "pathology_no",
        "report",
        "diagnosis",
        "organ",
        "x_level0",
        "y_level0",
        "coordinates",
        "local_path",
    }
)
PATH_RE = re.compile(r"(?:/data/|/home/|[A-Za-z]:\\)")
IDENTIFIER_RE = re.compile(
    r"\b(?:(?:patient|case|slide|pathology)(?:[_ -]?(?:id|no))?|病理号|患者号|病例号)\s*[:#=]\s*[A-Za-z0-9_-]+",
    re.I,
)
WSI_SUFFIXES = frozenset({".svs", ".ndpi", ".mrxs", ".tif", ".tiff"})


class TextScanner(Protocol):
    def __call__(self, png_bytes: bytes) -> list[dict[str, Any]]: ...


class CodeScanner(Protocol):
    def __call__(self, rgb: np.ndarray) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class SanitizedImage:
    png_bytes: bytes
    sha256: str
    width: int
    height: int
    mode: str
    encoder: str
    source_kind: str
    approved: bool
    metadata_removed: tuple[str, ...]
    text_hits: tuple[dict[str, Any], ...]
    code_hits: tuple[dict[str, Any], ...]


def tesseract_text_scanner(png_bytes: bytes, confidence_threshold: float = 70.0) -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            ["tesseract", "stdin", "stdout", "--psm", "11", "-l", "eng", "tsv"],
            input=png_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise PrivacyViolation("Required in-memory OCR scanner is unavailable") from exc
    hits: list[dict[str, Any]] = []
    lines = completed.stdout.decode("utf-8", errors="replace").splitlines()
    if not lines:
        return hits
    header = lines[0].split("\t")
    for raw in lines[1:]:
        values = raw.split("\t")
        if len(values) != len(header):
            continue
        row = dict(zip(header, values))
        text = row.get("text", "").strip()
        try:
            confidence = float(row.get("conf", "-1"))
        except ValueError:
            continue
        if confidence >= confidence_threshold and len(re.sub(r"\W", "", text)) >= 3:
            hits.append({"kind": "ocr_text", "confidence": confidence, "text_sha256": sha256_bytes(text.encode("utf-8"))})
    return hits


def opencv_code_scanner(rgb: np.ndarray) -> list[dict[str, Any]]:
    try:
        import cv2
    except ImportError as exc:
        raise PrivacyViolation("Required QR/barcode scanner is unavailable") from exc
    detector = cv2.QRCodeDetector()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hits: list[dict[str, Any]] = []
    try:
        ok, decoded, points, _ = detector.detectAndDecodeMulti(bgr)
    except cv2.error:
        ok, decoded, points = False, (), None
    if ok or points is not None:
        decoded_values = list(decoded or ())
        hits.append(
            {
                "kind": "qr_code",
                "decoded_count": sum(bool(value) for value in decoded_values),
                "decoded_sha256": [sha256_bytes(value.encode("utf-8")) for value in decoded_values if value],
            }
        )
    return hits


def opencv_burned_text_scanner(png_bytes: bytes) -> list[dict[str, Any]]:
    """Detect text-like aligned neutral components without decoding their content."""
    try:
        import cv2
    except ImportError as exc:
        raise PrivacyViolation("Required burned-text scanner is unavailable") from exc
    encoded = np.frombuffer(png_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise PrivacyViolation("Burned-text scanner could not decode the re-encoded PNG")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    # Neutral near-black/gray glyphs are the common WSI-label and burned-text case.
    mask = ((saturation <= 55) & (value <= 145)).astype(np.uint8) * 255
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for index in range(1, component_count):
        x, y, width, height, area = (int(item) for item in stats[index])
        if 2 <= width <= 100 and 6 <= height <= 80 and 10 <= area <= 3500 and area / (width * height) >= 0.10:
            components.append((x, y, width, height, area, float(centroids[index][0]), float(centroids[index][1])))
    components.sort(key=lambda item: item[5])
    for start, first in enumerate(components):
        run = [first]
        for candidate in components[start + 1 :]:
            prior = run[-1]
            if abs(candidate[6] - first[6]) <= max(8.0, first[3] * 0.65) and 0 <= candidate[0] - (prior[0] + prior[2]) <= 48:
                run.append(candidate)
            elif candidate[0] - (prior[0] + prior[2]) > 48:
                break
        span = run[-1][0] + run[-1][2] - run[0][0]
        if len(run) >= 3 and span >= 24:
            return [{"kind": "burned_text_geometry", "component_count": len(run), "span_px": span}]
    return []


def default_text_scanner(png_bytes: bytes) -> list[dict[str, Any]]:
    if shutil.which("tesseract"):
        return tesseract_text_scanner(png_bytes)
    return opencv_burned_text_scanner(png_bytes)


def sanitize_image(
    image: Image.Image,
    *,
    source_kind: str,
    text_scanner: TextScanner | None = None,
    code_scanner: CodeScanner | None = None,
) -> SanitizedImage:
    if source_kind not in {"wsi_coordinate_patch", "deterministic_unassessable_control", "nonmedical_synthetic_smoke"}:
        raise PrivacyViolation("Whole-slide, label, thumbnail, and unknown image sources are forbidden")
    if image.size != CANVAS_SIZE:
        raise PrivacyViolation(f"External image must be exactly {CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}")
    metadata_removed = tuple(sorted(str(key) for key in image.info))
    rgb_image = ImageOps.exif_transpose(image).convert("RGB")
    buffer = io.BytesIO()
    rgb_image.save(buffer, format="PNG", optimize=False, compress_level=9)
    encoded = buffer.getvalue()
    with Image.open(io.BytesIO(encoded)) as check:
        check.load()
        if check.mode != "RGB" or check.size != CANVAS_SIZE:
            raise PrivacyViolation("Re-encoded image dimensions or mode changed")
        if check.getexif():
            raise PrivacyViolation("Re-encoded image retained EXIF metadata")
        unsafe_info = set(check.info) - {"srgb"}
        if unsafe_info:
            raise PrivacyViolation(f"Re-encoded PNG retained metadata chunks: {sorted(unsafe_info)}")
        checked_rgb = np.asarray(check.convert("RGB"))
    text_hits = tuple((text_scanner or default_text_scanner)(encoded))
    code_hits = tuple((code_scanner or opencv_code_scanner)(checked_rgb))
    if text_hits or code_hits:
        raise PrivacyViolation("Image contains detected text, QR code, barcode, or burned-in identifier")
    return SanitizedImage(
        png_bytes=encoded,
        sha256=sha256_bytes(encoded),
        width=CANVAS_SIZE[0],
        height=CANVAS_SIZE[1],
        mode="RGB",
        encoder=f"Pillow/{PIL.__version__};zlib={features.version('zlib')}",
        source_kind=source_kind,
        approved=True,
        metadata_removed=metadata_removed,
        text_hits=text_hits,
        code_hits=code_hits,
    )


def render_wsi_coordinate_patch(
    *,
    slide_path: Path,
    x_level0: int,
    y_level0: int,
    physical_fov_um: float,
    mpp_x: float,
    mpp_y: float,
    split: str,
    opener: Callable[[str], Any] | None = None,
) -> Image.Image:
    if split.casefold() in TEST_SPLIT_NAMES:
        raise PrivacyViolation("Test10 image access is forbidden")
    if slide_path.suffix.casefold() not in WSI_SUFFIXES:
        raise PrivacyViolation("Source is not an approved WSI format")
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in (physical_fov_um, mpp_x, mpp_y)):
        raise ContractViolation("Invalid physical field of view or MPP")
    if x_level0 < 0 or y_level0 < 0:
        raise ContractViolation("Level-0 coordinates must be non-negative")
    if opener is None:
        try:
            import openslide
        except ImportError as exc:
            raise ContractViolation("OpenSlide is required for WSI coordinate rendering") from exc
        opener = openslide.OpenSlide
    width = max(1, round(float(physical_fov_um) / float(mpp_x)))
    height = max(1, round(float(physical_fov_um) / float(mpp_y)))
    slide = opener(str(slide_path))
    try:
        region = slide.read_region((int(x_level0), int(y_level0)), 0, (width, height)).convert("RGB")
    finally:
        close = getattr(slide, "close", None)
        if callable(close):
            close()
    return region.resize(CANVAS_SIZE, Image.Resampling.BICUBIC)


def deterministic_unassessable_control(index: int, seed: int = 20260827) -> Image.Image:
    if index < 0:
        raise ValueError("Control index must be non-negative")
    rng = np.random.default_rng(seed + index * 104729)
    canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), 250, dtype=np.int16)
    # Unique, low-amplitude scanner-like background without tissue or text.
    canvas += rng.integers(-2, 3, size=canvas.shape, dtype=np.int16)
    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    offset = 12 + (index * 37) % 120
    draw.line((0, offset, CANVAS_SIZE[0] - 1, offset), fill=(246, 246, 246), width=1)
    return image


def nonmedical_smoke_image() -> Image.Image:
    image = Image.new("RGB", CANVAS_SIZE, (248, 248, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((110, 252, 310, 452), fill=(220, 35, 45))
    draw.ellipse((474, 252, 674, 452), fill=(30, 95, 210))
    return image


def assert_external_payload_private(payload: dict[str, Any], *, allowed_item_id: str) -> None:
    def walk(value: Any, key: str | None = None) -> None:
        if key is not None and key.casefold() in FORBIDDEN_EXTERNAL_KEYS:
            raise PrivacyViolation(f"Forbidden external payload key: {key}")
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                walk(child, key)
        elif isinstance(value, str):
            if PATH_RE.search(value) or IDENTIFIER_RE.search(value):
                raise PrivacyViolation("External payload contains a path or source identifier")
            if any(suffix in value.casefold() for suffix in WSI_SUFFIXES):
                raise PrivacyViolation("External payload contains a WSI filename")
            if value.startswith("item_") and value != allowed_item_id:
                raise PrivacyViolation("External payload contains an unexpected item ID")

    walk(payload)
    image_urls: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "url" and isinstance(child, str) and child.startswith("data:image/"):
                    image_urls.append(child)
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(payload)
    if len(image_urls) != 1 or not image_urls[0].startswith("data:image/png;base64,"):
        raise PrivacyViolation("External request must contain exactly one re-encoded PNG")


def assert_generated_text_private(text: str) -> None:
    """Reject source-like identifiers/paths before generated text enters public artifacts."""
    if not isinstance(text, str):
        raise PrivacyViolation("Generated text is not a string")
    if PATH_RE.search(text) or IDENTIFIER_RE.search(text):
        raise PrivacyViolation("Generated text contains a path or source-like identifier")
    if any(suffix in text.casefold() for suffix in WSI_SUFFIXES):
        raise PrivacyViolation("Generated text contains a WSI filename")


def difference_hash(image: Image.Image, hash_size: int = 8) -> int:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def assert_no_cross_split_near_duplicates(rows: Iterable[dict[str, Any]], max_distance: int = 2) -> None:
    items = list(rows)
    seen_hashes: dict[str, str] = {}
    for row in items:
        image_sha = str(row["image_sha256"])
        split = str(row["split"])
        other = seen_hashes.get(image_sha)
        if other is not None and other != split:
            raise PrivacyViolation("Identical image hash occurs across splits")
        seen_hashes[image_sha] = split
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if left["split"] != right["split"] and hamming_distance(int(left["dhash64"], 16), int(right["dhash64"], 16)) <= max_distance:
                raise PrivacyViolation("Perceptual near-duplicate occurs across splits")
