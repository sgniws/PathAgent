from __future__ import annotations

import os
from argparse import Namespace


def resolve_retriever_args(args: Namespace) -> Namespace:
    """Resolve generic retriever CLI fields while preserving the PLIP aliases."""
    backend = args.retriever_backend
    if backend == "conch_v1" and args.zoom_backend != "wsi":
        raise ValueError("CONCH v1 currently supports only --zoom_backend wsi")
    legacy_values = {
        "library": args.plip_lib_path,
        "checkpoint": args.plip_ckpt,
        "features": args.feature_h5_dir,
    }
    generic_values = {
        "library": args.retriever_lib_path,
        "checkpoint": args.retriever_checkpoint,
        "features": args.retriever_feature_dir,
    }
    if backend == "plip":
        for field in generic_values:
            if (
                generic_values[field]
                and legacy_values[field]
                and os.path.abspath(generic_values[field])
                != os.path.abspath(legacy_values[field])
            ):
                raise ValueError(
                    f"Conflicting PLIP retriever {field} paths were provided via "
                    "new and legacy CLI options"
                )
        args.retriever_lib_path = generic_values["library"] or legacy_values["library"]
        args.retriever_checkpoint = (
            generic_values["checkpoint"] or legacy_values["checkpoint"]
        )
        args.retriever_feature_dir = generic_values["features"] or legacy_values["features"]
    else:
        missing_generic = [
            name
            for name, value in {
                "--retriever_lib_path": args.retriever_lib_path,
                "--retriever_checkpoint": args.retriever_checkpoint,
                "--retriever_feature_dir": args.retriever_feature_dir,
            }.items()
            if not value
        ]
        if missing_generic:
            raise ValueError(
                f"CONCH v1 requires explicit generic options: {', '.join(missing_generic)}"
            )
    missing = [
        name
        for name, value in {
            "retriever library": args.retriever_lib_path,
            "retriever checkpoint": args.retriever_checkpoint,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required retriever configuration: {', '.join(missing)}")
    if not os.path.exists(args.retriever_lib_path):
        raise FileNotFoundError(
            f"Retriever library path not found: {args.retriever_lib_path}"
        )
    if not os.path.exists(args.retriever_checkpoint):
        raise FileNotFoundError(
            f"Retriever checkpoint not found: {args.retriever_checkpoint}"
        )
    return args
