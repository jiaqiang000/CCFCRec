from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluate_amazon_vg_cicpmp_fr1_validation_groups import (
    METHOD_LABELS,
    build_model_args,
    load_analysis_profile,
)


CODE_ROOT = Path(__file__).resolve().parents[1]


def _raw_mp_rows() -> pd.DataFrame:
    rows = []
    for index, raw_asin in enumerate(("a", "b", "c", "train")):
        row = {
            "raw_asin": raw_asin,
            "split": "train" if raw_asin == "train" else "validate",
            "mp_raw_predicted_increment": 0.01 * (index + 1),
            "mp_category_semantic_increment_prediction": (-1) ** index * 0.02,
            "mp_category_total_increment_prediction": 0.03 * (index + 1),
            "mp_category_attribution_positive_share_prediction": 0.5 + 0.1 * index,
            "mp_category_attribution_entropy_prediction": 0.2 + 0.2 * index,
            "mp_fold_prediction_uncertainty": 0.001 * (index + 1),
            "mp_hgb_ridge_disagreement": 0.002 * (index + 1),
        }
        row.update({f"mp_direction16_{dim:02d}": float(index + dim) for dim in range(16)})
        rows.append(row)
    return pd.DataFrame(rows)


def _scalar_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "raw_asin": ["a", "b", "c", "train"],
            "split": ["validate", "validate", "validate", "train"],
            "cicp_score": [0.1, 0.5, 0.9, 0.3],
        }
    )


def test_load_analysis_profile_covers_all_features_and_groups(tmp_path: Path) -> None:
    mp_path = tmp_path / "mp.csv"
    scalar_path = tmp_path / "scalar.csv"
    _raw_mp_rows().to_csv(mp_path, index=False)
    _scalar_rows().to_csv(scalar_path, index=False)

    profile, audit = load_analysis_profile(
        CODE_ROOT,
        mp_path,
        scalar_path,
        ["c", "a", "b"],
    )

    assert profile["raw_asin"].tolist() == ["c", "a", "b"]
    assert audit["profile_feature_count"] == 23
    assert set(profile["cicp_score_group"]) == {"low", "mid", "high"}
    assert np.isfinite(profile["mp_direction_norm"]).all()
    assert len(audit["group_sources"]) == 9


def test_load_analysis_profile_rejects_evaluation_columns(tmp_path: Path) -> None:
    mp_path = tmp_path / "mp.csv"
    scalar_path = tmp_path / "scalar.csv"
    _raw_mp_rows().to_csv(mp_path, index=False)
    scalar = _scalar_rows()
    scalar["ndcg@20"] = 0.1
    scalar.to_csv(scalar_path, index=False)

    with pytest.raises(ValueError, match="evaluation columns"):
        load_analysis_profile(CODE_ROOT, mp_path, scalar_path, ["a", "b", "c"])


def test_build_model_args_carries_final_repair_parameters() -> None:
    state_dict = {
        "attr_matrix": np.zeros((4, 5)),
        "user_embedding": np.zeros((3, 6)),
        "gen_layer1.weight": np.zeros((7, 5)),
        "item_embedding": np.zeros((8, 6)),
    }
    config = {
        "seed": 19,
        "method_variant": "cicpmp_fr1_cross_modal_attention",
        "cicpmp_fr1_block_dim": 12,
        "cicpmp_fr1_residual_max_ratio": 0.2,
        "optimizer_parameter_groups": {"method_weight_decay": 0.0},
    }

    args = build_model_args(state_dict, config)

    assert args.seed == 19
    assert args.cicpmp_fr1_block_dim == 12
    assert args.cicpmp_fr1_residual_max_ratio == 0.2
    assert args.cicpmp_fr1_method_weight_decay == 0.0


def test_only_scalar_reference_uses_hidden_residual_label() -> None:
    residual_methods = [name for name in METHOD_LABELS if "scalar_residual" in name]
    assert residual_methods == ["cicpmp_fr1_scalar_residual_reference"]
    assert len(METHOD_LABELS) == 5
