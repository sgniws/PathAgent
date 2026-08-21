import pytest

from models.retrieval_policy import initial_retrieval_count, replenishment_count


def test_retrieval_ratios_have_a_minimum_of_one_patch():
    assert initial_retrieval_count(5, 0.10, None) == 1
    assert replenishment_count(5, 0.05, None) == 1


def test_fixed_retrieval_counts_are_capped_by_available_patches():
    assert initial_retrieval_count(3, 0.10, 10) == 3
    assert replenishment_count(3, 0.05, 2) == 2


@pytest.mark.parametrize("counter", [initial_retrieval_count, replenishment_count])
def test_fixed_retrieval_count_must_be_positive(counter):
    with pytest.raises(ValueError, match="must be at least 1"):
        counter(5, 0.10, 0)
