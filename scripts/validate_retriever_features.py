#!/usr/bin/env python3
"""Validate a complete model-bound WSI retriever feature directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing.wsi_pyramid import (  # noqa: E402
    load_patch_manifest,
    load_retriever_h5,
    load_wsi_manifest,
    resolve_wsi_input_fingerprint,
    retriever_feature_filename,
)
from models.retrievers import create_retriever  # noqa: E402
from models.retrievers.base import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever-backend", choices=["conch_v1"], required=True)
    parser.add_argument("--retriever-lib-path", type=Path, required=True)
    parser.add_argument("--retriever-checkpoint", type=Path, required=True)
    parser.add_argument("--wsi-manifest", type=Path, required=True)
    parser.add_argument("--patch-manifest-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--slide-id", action="append", default=[])
    parser.add_argument("--slide-list", type=Path)
    args = parser.parse_args()

    retriever = create_retriever(
        args.retriever_backend,
        args.retriever_lib_path,
        args.retriever_checkpoint,
        device=args.device,
    )
    wsi_rows = load_wsi_manifest(args.wsi_manifest)
    requested = list(args.slide_id)
    if args.slide_list:
        payload = json.loads(args.slide_list.read_text(encoding="utf-8"))
        entries = payload if isinstance(payload, list) else payload.get("slides", [])
        requested.extend(
            str(entry.get("slide_id") if isinstance(entry, dict) else entry)
            for entry in entries
        )
    if len(set(requested)) != len(requested):
        raise ValueError("Requested slide IDs contain duplicates")
    missing = sorted(set(requested) - set(wsi_rows))
    if missing:
        raise ValueError(f"Requested slide IDs are absent from WSI manifest: {missing}")
    selected_rows = (
        {slide_id: wsi_rows[slide_id] for slide_id in requested}
        if requested
        else wsi_rows
    )
    results = []
    seen_keys = set()
    total_rows = 0
    worst_norm_error = 0.0
    for slide_id, wsi_row in selected_rows.items():
        manifest_path = args.patch_manifest_dir / f"{slide_id}.jsonl"
        regions = load_patch_manifest(manifest_path, selected_only=True)
        patch_ids = list(regions)
        feature_path = args.feature_dir / retriever_feature_filename(
            slide_id, args.retriever_backend
        )
        loaded_ids, features, metadata = load_retriever_h5(
            feature_path,
            expected_spec=retriever.spec,
            expected_patch_ids=patch_ids,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_input_fingerprint_sha256=resolve_wsi_input_fingerprint(wsi_row)[0],
        )
        for patch_id in loaded_ids:
            key = (slide_id, patch_id)
            if key in seen_keys:
                raise ValueError(f"Duplicate slide/patch key: {key}")
            seen_keys.add(key)
        norms = np.linalg.norm(features, axis=1)
        error = float(np.max(np.abs(norms - 1.0)))
        worst_norm_error = max(worst_norm_error, error)
        total_rows += len(loaded_ids)
        results.append(
            {
                "slide_id": slide_id,
                "patch_count": len(loaded_ids),
                "feature_h5": str(feature_path.resolve()),
                "feature_h5_sha256": sha256_file(feature_path),
                "worst_norm_error": error,
                "format": wsi_row.get("format"),
                "read_schema": metadata.get("schema_version"),
            }
        )
    expected_files = {
        retriever_feature_filename(slide_id, args.retriever_backend)
        for slide_id in selected_rows
    }
    actual_files = {path.name for path in args.feature_dir.glob("*.h5")}
    if not requested and actual_files != expected_files:
        raise RuntimeError(
            f"Feature file set mismatch: missing={sorted(expected_files-actual_files)}, "
            f"extra={sorted(actual_files-expected_files)}"
        )
    payload = {
        "schema_version": "pathagent_retriever_feature_validation_v1",
        "status": "complete",
        "retriever_spec": retriever.spec.to_dict(),
        "wsi_count": len(selected_rows),
        "subset_validation": bool(requested),
        "feature_row_count": total_rows,
        "unique_slide_patch_count": len(seen_keys),
        "worst_norm_error": worst_norm_error,
        "slides": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
