#!/usr/bin/env python3
"""Precompute model-bound WSI patch embeddings with strict resumability."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

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
from models.retrievers.base import canonical_sha256, sha256_file  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_slide_list(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        rows = payload if isinstance(payload, list) else payload.get("slides", [])
    result = []
    for row in rows:
        slide_id = row.get("slide_id") if isinstance(row, dict) else row
        if not slide_id:
            raise ValueError(f"Slide list entry has no slide_id: {row}")
        result.append(str(slide_id))
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate slide IDs in slide list: {path}")
    return result


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(lock_path: Path, payload: dict[str, Any], clear_stale: bool) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists() and clear_stale:
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        same_host = existing.get("hostname") == socket.gethostname()
        pid = int(existing.get("pid") or -1)
        if same_host and pid > 0 and not _process_exists(pid):
            lock_path.unlink()
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Active or unresolved feature lock exists: {lock_path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _hdf5_safe_versions(versions: dict[str, Any]) -> dict[str, str]:
    return {key: str(value) for key, value in versions.items()}


def _runtime_versions() -> dict[str, str]:
    import PIL
    import openslide
    import timm
    import torch
    import torchvision
    import transformers

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "h5py": h5py.__version__,
        "openslide": openslide.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "transformers": transformers.__version__,
    }
    return _hdf5_safe_versions(versions)


def _cache_is_complete(
    path: Path,
    *,
    retriever: Any,
    patch_ids: list[str],
    manifest_sha256: str,
    input_fingerprint_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    load_retriever_h5(
        path,
        expected_spec=retriever.spec,
        expected_patch_ids=patch_ids,
        expected_manifest_sha256=manifest_sha256,
        expected_input_fingerprint_sha256=input_fingerprint_sha256,
    )
    return True


def _write_slide_features(
    *,
    retriever: Any,
    wsi_row: dict[str, Any],
    manifest_path: Path,
    output_path: Path,
    batch_size: int,
    limit_patches: int,
    clear_stale_locks: bool,
) -> dict[str, Any]:
    regions_by_id = load_patch_manifest(manifest_path, selected_only=True)
    regions = list(regions_by_id.values())
    if limit_patches:
        regions = regions[:limit_patches]
    if not regions:
        raise ValueError(f"No selected patches in {manifest_path}")
    patch_ids = [region.patch_id for region in regions]
    manifest_sha256 = sha256_file(manifest_path)
    input_fingerprint, input_fingerprint_method = resolve_wsi_input_fingerprint(wsi_row)
    if _cache_is_complete(
        output_path,
        retriever=retriever,
        patch_ids=patch_ids,
        manifest_sha256=manifest_sha256,
        input_fingerprint_sha256=input_fingerprint,
    ):
        return {
            "slide_id": wsi_row["slide_id"],
            "status": "already_complete",
            "patch_count": len(regions),
            "feature_h5": str(output_path.resolve()),
            "feature_h5_sha256": sha256_file(output_path),
        }

    lock_path = output_path.parent / ".locks" / f"{output_path.name}.lock"
    lock_payload = {
        "schema_version": "pathagent_retriever_feature_lock_v1",
        "slide_id": wsi_row["slide_id"],
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at_utc": utc_now(),
        "retriever_spec": retriever.spec.to_dict(),
        "patch_manifest_sha256": manifest_sha256,
        "input_fingerprint_sha256": input_fingerprint,
        "input_fingerprint_method": input_fingerprint_method,
    }
    acquire_lock(lock_path, lock_payload, clear_stale_locks)
    started = time.monotonic()
    temporary = output_path.with_suffix(
        output_path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        if temporary.exists():
            raise FileExistsError(f"Refusing to overwrite temporary feature file: {temporary}")
        all_features = []
        read_metadata = []
        with WSIPyramidReader(wsi_row) as reader:
            for offset in range(0, len(regions), batch_size):
                batch_regions = regions[offset : offset + batch_size]
                observations = [
                    reader.read(region, retriever.spec.input_size)
                    for region in batch_regions
                ]
                features = retriever.encode_images(
                    [observation.image for observation in observations],
                    batch_size=batch_size,
                )
                all_features.append(features)
                read_metadata.extend(observation.metadata for observation in observations)
                print(
                    f"[{wsi_row['slide_id']}] "
                    f"{min(offset + len(batch_regions), len(regions))}/{len(regions)}",
                    flush=True,
                )
        feature_matrix = np.concatenate(all_features, axis=0).astype(np.float32)
        norms = np.linalg.norm(feature_matrix, axis=1)
        worst_norm_error = float(np.max(np.abs(norms - 1.0)))
        if feature_matrix.shape != (len(regions), retriever.spec.embedding_dim):
            raise RuntimeError(f"Unexpected feature shape: {feature_matrix.shape}")
        if not np.isfinite(feature_matrix).all() or worst_norm_error > 1e-5:
            raise RuntimeError(
                f"Invalid feature matrix for {wsi_row['slide_id']}: "
                f"worst_norm_error={worst_norm_error}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        versions = _runtime_versions()
        with h5py.File(temporary, "w") as handle:
            handle.create_dataset(
                "features", data=feature_matrix, compression="gzip", compression_opts=1
            )
            handle.create_dataset(
                "patch_id",
                data=np.asarray(patch_ids, dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset(
                "coords",
                data=np.asarray(
                    [[region.x_level0, region.y_level0] for region in regions],
                    dtype=np.int64,
                ).reshape((-1, 2)),
            )
            handle.create_dataset(
                "width_level0",
                data=np.asarray([region.width_level0 for region in regions], dtype=np.int32),
            )
            handle.create_dataset(
                "height_level0",
                data=np.asarray([region.height_level0 for region in regions], dtype=np.int32),
            )
            handle.create_dataset(
                "read_level",
                data=np.asarray([row["read_level"] for row in read_metadata], dtype=np.int16),
            )
            handle.create_dataset(
                "read_downsample",
                data=np.asarray(
                    [row["read_downsample"] for row in read_metadata], dtype=np.float32
                ),
            )
            handle.create_dataset(
                "read_size",
                data=np.asarray([row["read_size"] for row in read_metadata], dtype=np.int32),
            )
            attrs = {
                "schema_version": "pathagent_retriever_features_v1",
                "status": "complete",
                "slide_id": str(wsi_row["slide_id"]),
                "coordinate_system": "level0",
                "retriever_backend": retriever.spec.backend,
                "model_id": retriever.spec.model_id,
                "checkpoint_sha256": retriever.spec.checkpoint_sha256,
                "source_revision": retriever.spec.source_revision,
                "source_dirty": retriever.spec.source_dirty,
                "model_config_sha256": retriever.spec.model_config_sha256,
                "preprocess_sha256": retriever.spec.preprocess_sha256,
                "model_input_pixels": retriever.spec.input_size,
                "inference_batch_size": batch_size,
                "embedding_dim": retriever.spec.embedding_dim,
                "feature_dtype": retriever.spec.output_dtype,
                "normalized": retriever.spec.normalized,
                "normalization": "l2",
                "projection": "contrastive",
                "source": "original WSI pyramid",
                "input_fingerprint_sha256": input_fingerprint,
                "input_fingerprint_method": input_fingerprint_method,
                "patch_manifest_sha256": manifest_sha256,
                "selected_patch_order_sha256": canonical_sha256(patch_ids),
                "selected_patch_count": len(patch_ids),
                "created_at_utc": utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "worst_norm_error": worst_norm_error,
            }
            for key, value in {**attrs, **versions}.items():
                handle.attrs[key] = value
        load_retriever_h5(
            temporary,
            expected_spec=retriever.spec,
            expected_patch_ids=patch_ids,
            expected_manifest_sha256=manifest_sha256,
            expected_input_fingerprint_sha256=input_fingerprint,
        )
        temporary.replace(output_path)
        return {
            "slide_id": wsi_row["slide_id"],
            "status": "complete",
            "patch_count": len(regions),
            "worst_norm_error": worst_norm_error,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "feature_h5": str(output_path.resolve()),
            "feature_h5_sha256": sha256_file(output_path),
        }
    finally:
        if temporary.exists():
            temporary.unlink()
        if lock_path.exists():
            try:
                current = json.loads(lock_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = {}
            if current.get("pid") == os.getpid() and current.get("hostname") == socket.gethostname():
                lock_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever-backend", choices=["conch_v1"], required=True)
    parser.add_argument("--retriever-lib-path", type=Path, required=True)
    parser.add_argument("--retriever-checkpoint", type=Path, required=True)
    parser.add_argument("--wsi-manifest", type=Path, required=True)
    parser.add_argument("--patch-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--slide-id", action="append", default=[])
    parser.add_argument("--slide-list", type=Path)
    parser.add_argument("--limit-patches", type=int, default=0)
    parser.add_argument("--clear-stale-locks", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.limit_patches < 0:
        raise ValueError("--limit-patches cannot be negative")

    wsi_rows = load_wsi_manifest(args.wsi_manifest)
    requested = list(args.slide_id)
    if args.slide_list:
        requested.extend(_read_slide_list(args.slide_list))
    if requested:
        if len(set(requested)) != len(requested):
            raise ValueError("Requested slide IDs contain duplicates")
        missing = sorted(set(requested) - set(wsi_rows))
        if missing:
            raise ValueError(f"Requested slide IDs are absent from WSI manifest: {missing}")
        selected_rows = [wsi_rows[slide_id] for slide_id in requested]
    else:
        selected_rows = list(wsi_rows.values())
    if not selected_rows:
        raise ValueError("No WSI selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "batch_runs" / f"{args.run_name}.status.json"
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            raise FileExistsError(f"Run status is already complete: {status_path}")

    retriever = create_retriever(
        args.retriever_backend,
        args.retriever_lib_path,
        args.retriever_checkpoint,
        device=args.device,
    )
    run_manifest = {
        "schema_version": "pathagent_retriever_precompute_run_v1",
        "run_name": args.run_name,
        "status": "running",
        "started_at_utc": utc_now(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "retriever_spec": retriever.spec.to_dict(),
        "wsi_manifest": str(args.wsi_manifest.resolve()),
        "wsi_manifest_sha256": sha256_file(args.wsi_manifest),
        "patch_manifest_dir": str(args.patch_manifest_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": args.device,
        "precision": args.precision,
        "batch_size": args.batch_size,
        "limit_patches": args.limit_patches,
        "requested_slide_ids": [str(row["slide_id"]) for row in selected_rows],
        "runtime_versions": _runtime_versions(),
        "results": [],
        "failures": [],
    }
    write_json_atomic(status_path, run_manifest)

    for index, row in enumerate(selected_rows, 1):
        slide_id = str(row["slide_id"])
        manifest_path = args.patch_manifest_dir / f"{slide_id}.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        output_path = args.output_dir / retriever_feature_filename(
            slide_id, args.retriever_backend
        )
        print(f"[slide {index}/{len(selected_rows)}] {slide_id}", flush=True)
        try:
            result = _write_slide_features(
                retriever=retriever,
                wsi_row=row,
                manifest_path=manifest_path,
                output_path=output_path,
                batch_size=args.batch_size,
                limit_patches=args.limit_patches,
                clear_stale_locks=args.clear_stale_locks,
            )
            run_manifest["results"].append(result)
        except Exception as exc:
            run_manifest["failures"].append(
                {
                    "slide_id": slide_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            run_manifest["status"] = "failed"
            run_manifest["updated_at_utc"] = utc_now()
            write_json_atomic(status_path, run_manifest)
            if not args.continue_on_error:
                raise
        run_manifest["updated_at_utc"] = utc_now()
        write_json_atomic(status_path, run_manifest)

    run_manifest["status"] = "complete" if not run_manifest["failures"] else "completed_with_failures"
    run_manifest["completed_at_utc"] = utc_now()
    run_manifest["slide_count"] = len(selected_rows)
    run_manifest["completed_slide_count"] = len(run_manifest["results"])
    run_manifest["feature_row_count"] = sum(
        int(row["patch_count"]) for row in run_manifest["results"]
    )
    write_json_atomic(status_path, run_manifest)
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
