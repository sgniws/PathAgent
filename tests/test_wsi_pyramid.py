from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if "torch" not in sys.modules:
    sys.modules["torch"] = types.SimpleNamespace()
if "qwen_vl_utils" not in sys.modules:
    sys.modules["qwen_vl_utils"] = types.SimpleNamespace(
        process_vision_info=lambda messages: ([], [])
    )
if "datasets" not in sys.modules:
    sys.modules["datasets"] = types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: None
    )

from data_processing.wsi_pyramid import (  # noqa: E402
    WSIRegion,
    choose_read_level,
    classify_focus_candidate,
    partition_region,
    rank_focus_candidates_c,
)
from pathagent_v2 import (  # noqa: E402
    _build_accumulated_executor_evidence,
    _normalize_repeated_focus_action,
    _validate_frozen_patho_canvas,
)


def parent_region(width=4018, height=4018):
    return WSIRegion(
        slide_id="slide",
        patch_id="x100_y200",
        x_level0=100,
        y_level0=200,
        width_level0=width,
        height_level0=height,
        mpp_x=0.2738,
        mpp_y=0.2738,
    )


@pytest.mark.parametrize("magnification,factor", [(10, 2), (20, 4)])
def test_partition_exactly_covers_parent_without_gaps(magnification, factor):
    parent = parent_region(width=4019, height=4021)
    children = partition_region(parent, magnification)
    assert len(children) == factor * factor
    assert sum(child.width_level0 * child.height_level0 for child in children) == (
        parent.width_level0 * parent.height_level0
    )
    assert min(child.x_level0 for child in children) == parent.x_level0
    assert min(child.y_level0 for child in children) == parent.y_level0
    assert max(child.x_level0 + child.width_level0 for child in children) == (
        parent.x_level0 + parent.width_level0
    )
    assert max(child.y_level0 + child.height_level0 for child in children) == (
        parent.y_level0 + parent.height_level0
    )
    assert all(child.parent_patch_id == parent.patch_id for child in children)
    assert all(child.magnification == magnification for child in children)


def test_read_level_never_undersamples_output_canvas():
    levels = [1.0, 2.0, 4.0, 8.0, 16.0]
    assert choose_read_level(levels, 4018, 4018, 784) == 2
    assert choose_read_level(levels, 2009, 2009, 784) == 1
    assert choose_read_level(levels, 1004, 1004, 784) == 0


def test_patho_r1_canvas_is_frozen_at_784():
    _validate_frozen_patho_canvas(SimpleNamespace(patho_observation_size=784))
    with pytest.raises(ValueError, match="frozen at 784x784"):
        _validate_frozen_patho_canvas(SimpleNamespace(patho_observation_size=504))
    with pytest.raises(ValueError, match="Alternative canvas experiments are disabled"):
        _validate_frozen_patho_canvas(SimpleNamespace(patho_observation_size=1008))


def test_focus_strategy_c_hard_filters_blank_and_padding_then_returns_top2():
    candidate_ids = ["blank", "ambiguous", "tissue", "padded"]
    candidate_meta = [
        {
            "candidate_class": "strict_blank",
            "grandqc_tissue_fraction": 0.0,
            "white_fraction": 1.0,
            "padding": {},
        },
        {
            "candidate_class": "ambiguous",
            "grandqc_tissue_fraction": 0.03,
            "white_fraction": 0.94,
            "padding": {},
        },
        {
            "candidate_class": "definite_nonblank",
            "grandqc_tissue_fraction": 0.30,
            "white_fraction": 0.50,
            "padding": {},
        },
        {
            "candidate_class": "definite_nonblank",
            "grandqc_tissue_fraction": 0.50,
            "white_fraction": 0.20,
            "padding": {"right_level0": 10},
        },
    ]
    selected, ranking = rank_focus_candidates_c(
        candidate_ids,
        np.asarray([0.40, 0.35, 0.33, 0.50]),
        candidate_meta,
        top_k=2,
    )
    assert [patch_id for patch_id, _score in selected] == ["tissue", "ambiguous"]
    by_id = {row["patch_id"]: row for row in ranking}
    assert by_id["blank"]["hard_rejection_reasons"] == ["strict_blank"]
    assert by_id["padded"]["hard_rejection_reasons"] == ["padding"]
    assert by_id["ambiguous"]["adjusted_score"] < by_id["ambiguous"]["plip_score"]
    assert classify_focus_candidate(0.0, 1.0) == "strict_blank"
    assert classify_focus_candidate(0.30, 0.50) == "definite_nonblank"


def test_repeated_focus_is_repaired_to_retrieve_or_abstain():
    decision = {
        "missing_evidence": "more atypical glands",
        "next_action": {
            "type": "zoom",
            "query": "",
            "target_patches": ["p1"],
            "magnification": 20,
        },
    }
    repaired, audit = _normalize_repeated_focus_action(deepcopy(decision), 1, 3)
    assert repaired["next_action"]["type"] == "retrieve"
    assert repaired["next_action"]["target_patches"] == []
    assert audit["reason"] == "one_focus_action_limit"

    repaired, _ = _normalize_repeated_focus_action(deepcopy(decision), 1, 0)
    assert repaired["next_action"]["type"] == "abstain"


def test_mixed_magnification_evidence_is_labeled_per_patch():
    text, visible = _build_accumulated_executor_evidence(
        {"base": "broad glands", "focus": "nuclear detail"},
        ["base", "focus"],
        20,
        10_000,
        evidence_metadata={
            "base": {
                "x_level0": 0,
                "y_level0": 0,
                "magnification": 5,
                "physical_width_um": 1100,
                "physical_height_um": 1100,
            },
            "focus": {
                "x_level0": 100,
                "y_level0": 200,
                "magnification": 20,
                "physical_width_um": 275,
                "physical_height_um": 275,
            },
        },
    )
    assert visible == ["base", "focus"]
    assert "Magnification=5x" in text
    assert "Magnification=20x" in text
    assert "FOV=1100.0x1100.0um" in text
    assert "FOV=275.0x275.0um" in text
