import sys
import types
from argparse import Namespace
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing.wsi_pyramid import (  # noqa: E402
    load_retriever_h5,
    resolve_wsi_input_fingerprint,
    retriever_feature_filename,
)
from models.retrievers.base import (  # noqa: E402
    RetrieverSpec,
    normalize_and_validate,
)
from models.retrievers.factory import create_retriever  # noqa: E402
from models.retrievers.config import resolve_retriever_args  # noqa: E402
from models.retrievers.plip import PLIPRetriever  # noqa: E402
from scripts.audit_trace_run import audit_trace  # noqa: E402
import scripts.precompute_retriever_features as precompute  # noqa: E402


def _spec(**overrides):
    values = {
        "backend": "conch_v1",
        "model_id": "conch_ViT-B-16",
        "embedding_dim": 2,
        "input_size": 448,
        "checkpoint_sha256": "checkpoint",
        "source_revision": "revision",
        "source_dirty": False,
        "model_config_sha256": "config",
        "preprocess_sha256": "preprocess",
    }
    values.update(overrides)
    return RetrieverSpec(**values)


def _write_feature_h5(path, spec, *, patch_ids=("p1", "p2"), include_read_size=True):
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=features)
        handle.create_dataset("patch_id", data=list(patch_ids), dtype=h5py.string_dtype())
        handle.create_dataset("coords", data=np.asarray([[1, 2], [3, 4]], dtype=np.int64))
        handle.create_dataset("width_level0", data=np.asarray([10, 10], dtype=np.int32))
        handle.create_dataset("height_level0", data=np.asarray([10, 10], dtype=np.int32))
        handle.create_dataset("read_level", data=np.asarray([0, 0], dtype=np.int16))
        handle.create_dataset("read_downsample", data=np.asarray([1, 1], dtype=np.float32))
        if include_read_size:
            handle.create_dataset("read_size", data=np.asarray([[10, 10], [10, 10]], dtype=np.int32))
        attrs = {
            "schema_version": "pathagent_retriever_features_v1",
            "status": "complete",
            "retriever_backend": spec.backend,
            "model_id": spec.model_id,
            "checkpoint_sha256": spec.checkpoint_sha256,
            "source_revision": spec.source_revision,
            "source_dirty": spec.source_dirty,
            "model_config_sha256": spec.model_config_sha256,
            "preprocess_sha256": spec.preprocess_sha256,
            "embedding_dim": spec.embedding_dim,
            "model_input_pixels": spec.input_size,
            "feature_dtype": spec.output_dtype,
            "normalized": spec.normalized,
            "patch_manifest_sha256": "manifest",
            "input_fingerprint_sha256": "wsi",
        }
        for key, value in attrs.items():
            handle.attrs[key] = value


def test_normalize_contract_keeps_batch_dimension_and_float32():
    output = normalize_and_validate(
        [[3.0, 4.0]], expected_count=1, embedding_dim=2, operation="test"
    )
    assert output.shape == (1, 2)
    assert output.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), [1.0])
    with pytest.raises(ValueError, match="zero-norm"):
        normalize_and_validate(
            [[0.0, 0.0]], expected_count=1, embedding_dim=2, operation="test"
        )


def test_adapter_input_contract_rejects_scalar_and_bad_batch(monkeypatch, tmp_path):
    library = tmp_path / "plip-source"
    checkpoint = tmp_path / "plip-checkpoint"
    library.mkdir()
    checkpoint.mkdir()
    (checkpoint / "pytorch_model.bin").write_bytes(b"weight")

    class FakePLIP:
        def __init__(self, _path):
            pass

    monkeypatch.setitem(sys.modules, "plip", types.SimpleNamespace(PLIP=FakePLIP))
    retriever = PLIPRetriever(library, checkpoint)
    with pytest.raises(TypeError, match="sequence"):
        retriever.encode_text("not-a-list")
    with pytest.raises(TypeError, match="sequence"):
        retriever.encode_images(Image.new("RGB", (8, 8)))
    with pytest.raises(ValueError, match="batch_size"):
        retriever.encode_text(["query"], batch_size=0)


def test_retriever_h5_rejects_embedding_space_drift(tmp_path):
    path = tmp_path / "slide.conch_v1.h5"
    spec = _spec()
    _write_feature_h5(path, spec)
    patch_ids, features, attrs = load_retriever_h5(
        path,
        expected_spec=spec,
        expected_patch_ids=["p1", "p2"],
        expected_manifest_sha256="manifest",
        expected_input_fingerprint_sha256="wsi",
    )
    assert patch_ids == ["p1", "p2"]
    assert features.dtype == np.float32
    assert attrs["retriever_backend"] == "conch_v1"
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        load_retriever_h5(path, expected_spec=_spec(checkpoint_sha256="different"))


def test_retriever_h5_rejects_incomplete_schema(tmp_path):
    path = tmp_path / "slide.conch_v1.h5"
    spec = _spec()
    _write_feature_h5(path, spec, include_read_size=False)
    with pytest.raises(RuntimeError, match="missing datasets"):
        load_retriever_h5(path, expected_spec=spec)


def test_wsi_fingerprint_prefers_manifest_and_supports_legacy(tmp_path):
    assert resolve_wsi_input_fingerprint(
        {"input_fingerprint_sha256": "authority"}
    ) == ("authority", "manifest_input_fingerprint_sha256")
    slide = tmp_path / "slide.svs"
    slide.write_bytes(b"legacy-slide-placeholder")
    row = {
        "slide_path": str(slide),
        "format": "svs",
        "width": 100,
        "height": 200,
        "level_count": 1,
        "level_dimensions": [[100, 200]],
        "level_downsamples": [1.0],
        "mpp_x": 0.25,
        "mpp_y": 0.25,
        "objective_power": 40,
    }
    first = resolve_wsi_input_fingerprint(row)
    second = resolve_wsi_input_fingerprint(row)
    assert first == second
    assert first[1] == "legacy_stat_and_pyramid_metadata_v1"
    assert len(first[0]) == 64


def test_plip_adapter_satisfies_generic_contract(monkeypatch, tmp_path):
    library = tmp_path / "plip-source"
    checkpoint = tmp_path / "plip-checkpoint"
    library.mkdir()
    checkpoint.mkdir()
    (checkpoint / "pytorch_model.bin").write_bytes(b"weight")

    class FakePLIP:
        def __init__(self, _path):
            pass

        def encode_text(self, texts, batch_size):
            return np.tile(np.asarray([[3.0] + [4.0] + [0.0] * 510]), (len(texts), 1))

        def encode_images(self, images, batch_size):
            return np.tile(np.asarray([[0.0, 5.0] + [0.0] * 510]), (len(images), 1))

    monkeypatch.setitem(sys.modules, "plip", types.SimpleNamespace(PLIP=FakePLIP))
    retriever = PLIPRetriever(library, checkpoint)
    assert retriever.encode_text(["query"]).shape == (1, 512)
    assert retriever.encode_images([Image.new("RGB", (8, 8))]).shape == (1, 512)
    assert retriever.spec.backend == "plip"


def test_factory_and_feature_filenames_reject_unknown_backend():
    assert retriever_feature_filename("slide", "plip") == "slide.plip.v1.h5"
    assert retriever_feature_filename("slide", "conch_v1") == "slide.conch_v1.h5"
    with pytest.raises(ValueError, match="Unsupported retriever backend"):
        retriever_feature_filename("slide", "unknown")
    with pytest.raises(ValueError, match="Unsupported retriever backend"):
        create_retriever("unknown", "/missing", "/missing")


def test_retriever_config_preserves_plip_aliases_and_requires_explicit_conch(tmp_path):
    library = tmp_path / "library"
    checkpoint = tmp_path / "checkpoint"
    features = tmp_path / "features"
    library.mkdir()
    checkpoint.mkdir()
    features.mkdir()
    legacy = Namespace(
        retriever_backend="plip",
        zoom_backend="wsi",
        retriever_lib_path=None,
        retriever_checkpoint=None,
        retriever_feature_dir=None,
        plip_lib_path=str(library),
        plip_ckpt=str(checkpoint),
        feature_h5_dir=str(features),
    )
    resolved = resolve_retriever_args(legacy)
    assert resolved.retriever_lib_path == str(library)
    assert resolved.retriever_checkpoint == str(checkpoint)
    assert resolved.retriever_feature_dir == str(features)

    conch = Namespace(
        retriever_backend="conch_v1",
        zoom_backend="wsi",
        retriever_lib_path=str(library),
        retriever_checkpoint=str(checkpoint),
        retriever_feature_dir=None,
        plip_lib_path=str(library),
        plip_ckpt=str(checkpoint),
        feature_h5_dir=str(features),
    )
    with pytest.raises(ValueError, match="explicit generic options"):
        resolve_retriever_args(conch)


def test_trace_v3_audit_accepts_stable_retriever_and_rejects_backend_drift():
    trace = {
        "schema_version": "pathagent_trace_v3",
        "runtime": {
            "retriever": {"backend": "conch_v1", "model_id": "conch_ViT-B-16"}
        },
        "task_input": {},
        "events": [
            {
                "phase": "before",
                "call_id": "call",
                "component": "retriever",
                "operation": "initial_morphology_embedding",
                "request": {
                    "backend": "conch_v1",
                    "model_id": "conch_ViT-B-16",
                },
            },
            {
                "phase": "after",
                "call_id": "call",
                "component": "retriever",
                "operation": "initial_morphology_embedding",
                "status": "ok",
                "response": {
                    "backend": "conch_v1",
                    "model_id": "conch_ViT-B-16",
                },
            },
        ],
        "final_output": {"answer": None, "evidence_refs": []},
        "execution": {"status": "completed"},
    }
    assert audit_trace(trace) == []
    trace["events"][1]["response"]["backend"] = "plip"
    assert any("backend drift" in error for error in audit_trace(trace))


def test_precompute_runtime_versions_are_plain_hdf5_safe_strings():
    class StringSubclass(str):
        pass

    versions = precompute._hdf5_safe_versions(
        {"torch": StringSubclass("2.7.1+cu128"), "python": "3.9.25"}
    )
    assert all(type(value) is str for value in versions.values())


def test_precompute_failure_removes_owned_lock_and_temporary(monkeypatch, tmp_path):
    slide = tmp_path / "slide.svs"
    slide.write_bytes(b"placeholder")
    manifest = tmp_path / "slide.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "slide_id": "slide",
                "patch_id": "p1",
                "selected": True,
                "x_level0": 0,
                "y_level0": 0,
                "width_level0": 16,
                "height_level0": 16,
                "mpp_x": 0.25,
                "mpp_y": 0.25,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class BrokenReader:
        def __init__(self, _row):
            pass

        def __enter__(self):
            raise RuntimeError("injected WSI failure")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(precompute, "WSIPyramidReader", BrokenReader)
    retriever = types.SimpleNamespace(spec=_spec())
    output = tmp_path / "features" / "slide.conch_v1.h5"
    with pytest.raises(RuntimeError, match="injected WSI failure"):
        precompute._write_slide_features(
            retriever=retriever,
            wsi_row={
                "slide_id": "slide",
                "slide_path": str(slide),
                "format": "svs",
                "width": 16,
                "height": 16,
                "level_count": 1,
                "level_dimensions": [[16, 16]],
                "level_downsamples": [1.0],
                "mpp_x": 0.25,
                "mpp_y": 0.25,
                "objective_power": 40,
            },
            manifest_path=manifest,
            output_path=output,
            batch_size=1,
            limit_patches=0,
            clear_stale_locks=False,
        )
    assert not output.exists()
    assert not list(tmp_path.rglob("*.lock"))
    assert not list(tmp_path.rglob("*.tmp.*"))
