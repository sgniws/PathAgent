from __future__ import annotations


def initial_retrieval_count(
    available_count: int, initial_sample_ratio: float, initial_top_k: int | None
) -> int:
    if available_count < 1:
        return 0
    if initial_top_k is not None:
        if initial_top_k < 1:
            raise ValueError("--initial_top_k must be at least 1 when provided")
        return min(available_count, initial_top_k)
    return max(1, int(available_count * initial_sample_ratio))


def replenishment_count(
    available_count: int, replenish_ratio: float, replenish_top_k: int | None
) -> int:
    if available_count < 1:
        return 0
    if replenish_top_k is not None:
        if replenish_top_k < 1:
            raise ValueError("--replenish_top_k must be at least 1 when provided")
        return min(available_count, replenish_top_k)
    return max(1, int(available_count * replenish_ratio))
