#!/usr/bin/env python3
"""Run real-weight and offline/online consistency checks for a retriever."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing.wsi_pyramid import (  # noqa: E402
    WSIPyramidReader,
    load_patch_manifest,
    load_retriever_h5,
    load_wsi_manifest,
    resolve_wsi_input_fingerprint,
    retriever_feature_filename,
)
from models.retrievers import create_retriever  # noqa: E402
from models.retrievers.base import sha256_file  # noqa: E402


FIXED_QUERIES = [
    "invasive atypical glands in desmoplastic stroma",
    "solid nested tumor cells with necrosis",
]


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left.astype(np.float64), right.astype(np.float64)))


def _top_ids(features: np.ndarray, query: np.ndarray, patch_ids: list[str], k: int):
    scores = np.asarray(features @ query.T).reshape(-1)
    order = np.argsort(scores, kind="stable")[::-1][:k]
    return [patch_ids[int(index)] for index in order], scores


def _parse_sample(value: str) -> dict[str, Path | str]:
    parts = value.split("::")
    if len(parts) != 4 or any(not part for part in parts):
        raise ValueError(
            "--sample must be WSI_MANIFEST::PATCH_MANIFEST_DIR::FEATURE_DIR::SLIDE_ID"
        )
    return {
        "wsi_manifest": Path(parts[0]),
        "patch_manifest_dir": Path(parts[1]),
        "feature_dir": Path(parts[2]),
        "slide_id": parts[3],
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever-backend", choices=["conch_v1"], required=True)
    parser.add_argument("--retriever-lib-path", type=Path, required=True)
    parser.add_argument("--retriever-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--public-image", type=Path, required=True)
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--comparison-batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    retriever = create_retriever(
        args.retriever_backend,
        args.retriever_lib_path,
        args.retriever_checkpoint,
        device=args.device,
    )
    public_image = Image.open(args.public_image).convert("RGB")
    public_single_1 = retriever.encode_images([public_image], batch_size=1)[0]
    public_single_2 = retriever.encode_images([public_image], batch_size=1)[0]
    public_batch = retriever.encode_images([public_image, public_image], batch_size=2)[0]
    text_1 = retriever.encode_text(FIXED_QUERIES, batch_size=2)
    text_2 = retriever.encode_text(FIXED_QUERIES, batch_size=2)
    checks = {
        "public_image_repeat_max_abs_diff": float(
            np.max(np.abs(public_single_1 - public_single_2))
        ),
        "public_image_single_batch_cosine": _cosine(public_single_1, public_batch),
        "public_image_norm": float(np.linalg.norm(public_single_1)),
        "text_repeat_max_abs_diff": float(np.max(np.abs(text_1 - text_2))),
        "text_worst_norm_error": float(
            np.max(np.abs(np.linalg.norm(text_1, axis=1) - 1.0))
        ),
    }

    sample_results = []
    for raw_sample in args.sample:
        sample = _parse_sample(raw_sample)
        slide_id = str(sample["slide_id"])
        wsi_rows = load_wsi_manifest(sample["wsi_manifest"])
        if slide_id not in wsi_rows:
            raise ValueError(f"Smoke slide {slide_id} is absent from WSI manifest")
        wsi_row = wsi_rows[slide_id]
        manifest_path = sample["patch_manifest_dir"] / f"{slide_id}.jsonl"
        regions = load_patch_manifest(manifest_path, selected_only=True)
        patch_ids = list(regions)
        feature_path = sample["feature_dir"] / retriever_feature_filename(
            slide_id, retriever.spec.backend
        )
        loaded_ids, features, metadata = load_retriever_h5(
            feature_path,
            expected_spec=retriever.spec,
            expected_patch_ids=patch_ids,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_input_fingerprint_sha256=resolve_wsi_input_fingerprint(wsi_row)[0],
        )
        first_regions = [
            regions[patch_id]
            for patch_id in loaded_ids[: args.comparison_batch_size]
        ]
        with WSIPyramidReader(wsi_row) as reader:
            observations = [
                reader.read(region, retriever.spec.input_size) for region in first_regions
            ]
        online_single = retriever.encode_images(
            [observations[0].image], batch_size=1
        )[0]
        online_repeat = retriever.encode_images(
            [observations[0].image], batch_size=1
        )[0]
        online_batch = retriever.encode_images(
            [observation.image for observation in observations],
            batch_size=len(observations),
        )[0]
        same_batch_repeat = retriever.encode_images(
            [observation.image for observation in observations],
            batch_size=args.comparison_batch_size,
        )[0]
        offline = features[0]
        query_1 = retriever.encode_text([FIXED_QUERIES[0]], batch_size=1)
        query_2 = retriever.encode_text([FIXED_QUERIES[0]], batch_size=1)
        top_1, scores_1 = _top_ids(
            features, query_1, loaded_ids, min(args.top_k, len(loaded_ids))
        )
        top_2, scores_2 = _top_ids(
            features, query_2, loaded_ids, min(args.top_k, len(loaded_ids))
        )
        region = first_regions[0]
        read_meta = observations[0].metadata
        sidecar_ok = True
        if str(wsi_row.get("format", "")).lower() == "mrxs":
            sidecar = Path(str(wsi_row.get("mrxs_sidecar_path") or ""))
            sidecar_ok = bool(
                wsi_row.get("mrxs_sidecar_ok")
                and sidecar.is_dir()
                and any(sidecar.iterdir())
            )
        sample_results.append(
            {
                "slide_id": slide_id,
                "format": wsi_row.get("format"),
                "patch_count": len(loaded_ids),
                "feature_h5": str(feature_path.resolve()),
                "feature_h5_sha256": sha256_file(feature_path),
                "offline_online_cosine": _cosine(offline, online_single),
                "offline_online_max_abs_diff": float(
                    np.max(np.abs(offline - online_single))
                ),
                "offline_same_batch_cosine": _cosine(offline, same_batch_repeat),
                "offline_same_batch_max_abs_diff": float(
                    np.max(np.abs(offline - same_batch_repeat))
                ),
                "online_repeat_max_abs_diff": float(
                    np.max(np.abs(online_single - online_repeat))
                ),
                "single_batch_cosine": _cosine(online_single, online_batch),
                "fixed_query_repeat_max_score_diff": float(
                    np.max(np.abs(scores_1 - scores_2))
                ),
                "fixed_query_top_k_equal": top_1 == top_2,
                "fixed_query_top_k": top_1,
                "level0_coordinate_equal": [
                    read_meta["x_level0"],
                    read_meta["y_level0"],
                    read_meta["width_level0"],
                    read_meta["height_level0"],
                ]
                == [
                    region.x_level0,
                    region.y_level0,
                    region.width_level0,
                    region.height_level0,
                ],
                "output_size": read_meta["output_size"],
                "read_level": read_meta["read_level"],
                "read_downsample": read_meta["read_downsample"],
                "mpp_valid": float(wsi_row.get("mpp_x") or 0) > 0
                and float(wsi_row.get("mpp_y") or 0) > 0,
                "mrxs_sidecar_valid": sidecar_ok,
                "input_fingerprint_method": metadata.get(
                    "input_fingerprint_method"
                ),
            }
        )

    failures = []
    if checks["public_image_repeat_max_abs_diff"] > 1e-6:
        failures.append("public image repeat max_abs_diff exceeds 1e-6")
    if checks["public_image_single_batch_cosine"] < 0.999999:
        failures.append("public image single/batch cosine is below 0.999999")
    if checks["text_repeat_max_abs_diff"] > 1e-6:
        failures.append("text repeat max_abs_diff exceeds 1e-6")
    if abs(checks["public_image_norm"] - 1.0) > 1e-5:
        failures.append("public image norm error exceeds 1e-5")
    if checks["text_worst_norm_error"] > 1e-5:
        failures.append("text norm error exceeds 1e-5")
    for row in sample_results:
        prefix = row["slide_id"]
        if row["offline_same_batch_cosine"] < 0.999999:
            failures.append(f"{prefix}: offline/same-batch cosine is below 0.999999")
        if row["offline_same_batch_max_abs_diff"] > 1e-5:
            failures.append(f"{prefix}: offline/same-batch max_abs_diff exceeds 1e-5")
        if row["online_repeat_max_abs_diff"] > 1e-6:
            failures.append(f"{prefix}: repeated encoding max_abs_diff exceeds 1e-6")
        if row["single_batch_cosine"] < 0.99998:
            failures.append(f"{prefix}: single/batch cosine is below 0.99998")
        if row["fixed_query_repeat_max_score_diff"] > 1e-5:
            failures.append(f"{prefix}: repeated query score drift exceeds 1e-5")
        if not row["fixed_query_top_k_equal"]:
            failures.append(f"{prefix}: repeated query Top-K differs")
        if not row["level0_coordinate_equal"]:
            failures.append(f"{prefix}: Level-0 coordinate drift")
        if row["output_size"] != [retriever.spec.input_size] * 2:
            failures.append(f"{prefix}: wrong runtime image size")
        if not row["mpp_valid"]:
            failures.append(f"{prefix}: invalid MPP")
        if not row["mrxs_sidecar_valid"]:
            failures.append(f"{prefix}: invalid MRXS sidecar")

    payload = {
        "schema_version": "pathagent_retriever_smoke_v1",
        "status": "passed" if not failures else "failed",
        "retriever_spec": retriever.spec.to_dict(),
        "checkpoint_key_audit": {
            "missing_count": len(getattr(retriever, "missing_keys", [])),
            "missing_keys_sha256": getattr(retriever, "missing_keys_sha256", None),
            "unexpected_count": len(getattr(retriever, "unexpected_keys", [])),
        },
        "public_image": str(args.public_image.resolve()),
        "public_image_sha256": sha256_file(args.public_image),
        "fixed_queries": FIXED_QUERIES,
        "checks": checks,
        "samples": sample_results,
        "failures": failures,
    }
    _write_json_atomic(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
