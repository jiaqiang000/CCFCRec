import numpy as np
import pandas as pd
import pytest

from analyze_amazon_vg_m11_signal_reaudit import (
    TARGET_COLUMN,
    build_component_ablation,
    build_composition,
    build_group_performance,
    load_validation_benefit,
    route_decision,
)


def _profile() -> pd.DataFrame:
    rows = [
        ("train-a", "train", True, "RSP_low", True, True, 0.8, 0.7, 0.2),
        ("train-b", "train", True, "RSP_high", True, True, 0.6, 0.5, 0.8),
        ("val-a", "validate", True, "RSP_mid", True, True, 0.9, 0.8, 0.3),
        ("val-b", "validate", False, "RSP_low", True, True, 0.4, 0.3, 0.2),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "raw_asin",
            "split",
            "high_acat_flag",
            "RSP_group",
            "category_neighbor_mismatch_proxy_high_flag",
            "support_tail_proxy_high_flag",
            "s_cat_v3",
            "train_safe_hard_proxy_score",
            "RSP_score",
        ],
    )
    frame["category_neighbor_mismatch_proxy_score"] = frame["train_safe_hard_proxy_score"]
    frame["support_tail_proxy_score"] = [0.9, 0.8, 0.95, 0.85]
    frame["category_count"] = [4, 4, 5, 3]
    frame[TARGET_COLUMN] = (
        frame["high_acat_flag"]
        & ~frame["RSP_group"].eq("RSP_high")
        & frame["category_neighbor_mismatch_proxy_high_flag"]
        & frame["support_tail_proxy_high_flag"]
    )
    frame["m11_target_score"] = (
        0.45 * frame["s_cat_v3"]
        + 0.35 * frame["train_safe_hard_proxy_score"]
        + 0.20 * (1 - frame["RSP_score"])
    )
    return frame


def test_composition_and_ablation_expose_not_rsp_high_semantics() -> None:
    profile = _profile()
    composition = build_composition(profile)
    ablation = build_component_ablation(profile)

    assert composition.set_index("split").loc["validate", "target_count"] == 1
    assert composition.set_index("split").loc["train", "target_count"] == 1
    full = ablation[ablation["candidate"].eq("full_target")]
    assert full["jaccard_vs_full_target"].eq(1.0).all()
    assert full["conditions"].str.contains("low_rsp_name_but_not_high").all()


def test_validation_benefit_rejects_test_rows(tmp_path) -> None:
    item_eval = pd.DataFrame(
        [
            {
                "method_variant": "baseline",
                "split": "test",
                "raw_asin": "val-a",
                "target_flag": True,
                "ndcg@20": 0.0,
            }
        ]
    )
    path = tmp_path / "item_eval.csv"
    item_eval.to_csv(path, index=False)

    with pytest.raises(ValueError, match="validation item rows only"):
        load_validation_benefit(path, _profile())


def test_group_performance_keeps_target_and_non_target_complementary() -> None:
    frame = pd.DataFrame(
        {
            "target_flag": [True, False],
            "baseline_ndcg20": [0.1, 0.2],
            "e4_delta": [0.01, -0.02],
            "e1_delta": [0.03, 0.00],
        }
    )
    result = build_group_performance(frame).set_index("group")

    assert result.loc["validation_all", "item_count"] == 2
    assert result.loc["validation_target", "item_count"] == 1
    assert result.loc["validation_non_target", "item_count"] == 1
    assert np.isclose(result.loc["validation_all", "e4_delta_mean"], -0.005)


def test_route_retires_primary_signal_without_opening_training() -> None:
    decision = route_decision()

    assert decision["primary_signal_status"] == "retire"
    assert decision["diagnostic_subgroup_status"] == "retain"
    assert decision["run_training_now"] is False
    assert decision["run_multi_seed_now"] is False
    assert decision["test_items_analyzed"] == 0
    assert decision["test_item_level_metrics_read_or_generated"] is False
    assert decision["historical_test_informed_aggregate_read_for_provenance"] is True
