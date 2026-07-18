from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluate_amazon_vg_cicpmp_r1_validation_groups import (
    METHOD_LABELS,
    build_model_args,
    load_validation_profile,
)


CODE_ROOT = Path(__file__).resolve().parents[1]


def _profile_rows() -> pd.DataFrame:
    rows = []
    for index, raw_asin in enumerate(("a", "b", "c", "train")):
        row = {
            "raw_asin": raw_asin,
            "split": "train" if raw_asin == "train" else "validate",
            "mp_raw_predicted_increment": 0.01 * (index + 1),
            "mp_category_semantic_increment_prediction": (-1) ** index * 0.02,
            "mp_category_total_increment_prediction": 0.03,
            "mp_category_attribution_positive_share_prediction": 0.6,
            "mp_category_attribution_entropy_prediction": 0.2 + 0.2 * index,
            "mp_fold_prediction_uncertainty": 0.001 * (index + 1),
            "mp_hgb_ridge_disagreement": 0.002 * (index + 1),
        }
        row.update({f"mp_direction16_{dim:02d}": float(index + dim) for dim in range(16)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_load_validation_profile_builds_complete_fixed_groups(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.csv"
    _profile_rows().to_csv(profile_path, index=False)

    profile, audit = load_validation_profile(
        CODE_ROOT,
        profile_path,
        ["c", "a", "b"],
        reliability_scale=50.0,
    )

    assert profile["raw_asin"].tolist() == ["c", "a", "b"]
    assert set(profile["reliability_group"]) == {"low", "mid", "high"}
    assert set(profile["semantic_sign_group"]) == {"positive", "non_positive"}
    assert sum(audit["group_counts"]["reliability_group"].values()) == 3
    assert np.isfinite(profile["mp_direction_norm"]).all()


def test_load_validation_profile_rejects_evaluation_columns(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.csv"
    frame = _profile_rows()
    frame["ndcg@20"] = 0.1
    frame.to_csv(profile_path, index=False)

    with pytest.raises(ValueError, match="forbidden evaluation-result columns"):
        load_validation_profile(
            CODE_ROOT,
            profile_path,
            ["a", "b", "c"],
            reliability_scale=50.0,
        )


def test_build_model_args_carries_frozen_cicpmp_parameters() -> None:
    state_dict = {
        "attr_matrix": np.zeros((4, 5)),
        "user_embedding": np.zeros((3, 6)),
        "gen_layer1.weight": np.zeros((7, 5)),
        "item_embedding": np.zeros((8, 6)),
    }
    config = {
        "seed": 19,
        "method_variant": "cicpmp_r1_reliable_residual",
        "cicpmp_hidden_dim": 32,
        "cicpmp_reliability_scale": 50.0,
    }

    args = build_model_args(state_dict, config)

    assert args.seed == 19
    assert args.cicpmp_hidden_dim == 32
    assert args.cicpmp_reliability_scale == 50.0
    assert args.method_variant == "cicpmp_r1_reliable_residual"


def test_only_one_method_uses_residual_carrier() -> None:
    residual_methods = [name for name in METHOD_LABELS if "residual" in name]
    assert residual_methods == ["cicpmp_r1_reliable_residual"]
    assert len(METHOD_LABELS) == 6
