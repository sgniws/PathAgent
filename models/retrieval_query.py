"""Frozen text-query contracts shared by PathAgent and offline evaluation."""

from __future__ import annotations


def initial_retrieval_query(question: str) -> str:
    return (
        f"{question} "
        "Retrieve lesional H&E regions with direct abnormal morphology: architectural distortion, infiltrative atypical "
        "glands, desmoplastic stroma, papillary or pseudopapillary structures, solid tumor cells, mucin, necrosis, "
        "or a tumor-stroma interface."
    )
