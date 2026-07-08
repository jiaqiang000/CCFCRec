import math

import pandas as pd

from analyze_amazon_vg_task4_revise import (
    build_candidate_carrier_table,
    build_contribution_summary,
    build_mask_audit,
    build_revise_route_decision,
    build_split_aware_shuffle_flags,
)


def test_split_aware_shuffle_preserves_high_count_per_split_and_changes_mapping() -> None:
    profile = pd.DataFrame(
        [
            {"raw_asin": "b", "split": "train", "high_acat_flag": True},
            {"raw_asin": "a", "split": "train", "high_acat_flag": False},
            {"raw_asin": "d", "split": "test", "high_acat_flag": True},
            {"raw_asin": "c", "split": "test", "high_acat_flag": False},
            {"raw_asin": "f", "split": "validate", "high_acat_flag": True},
            {"raw_asin": "e", "split": "validate", "high_acat_flag": False},
        ]
    )

    shuffled = build_split_aware_shuffle_flags(profile, seed=43)

    sorted_profile = profile.sort_values("raw_asin").reset_index(drop=True)
    result = sorted_profile.assign(shuffled=shuffled)
    for split, sub in result.groupby("split"):
        assert int(sub["shuffled"].sum()) == int(sub["high_acat_flag"].sum())
    assert shuffled.tolist() == build_split_aware_shuffle_flags(profile, seed=43).tolist()


def test_mask_audit_reports_m3_subset_and_shuffle_overlap() -> None:
    profile = pd.DataFrame(
        [
            {"raw_asin": "a", "split": "train", "high_acat_flag": True, "high_acat_train_safe_hard_flag": True},
            {"raw_asin": "b", "split": "train", "high_acat_flag": True, "high_acat_train_safe_hard_flag": False},
            {"raw_asin": "c", "split": "train", "high_acat_flag": False, "high_acat_train_safe_hard_flag": False},
            {"raw_asin": "d", "split": "train", "high_acat_flag": False, "high_acat_train_safe_hard_flag": False},
        ]
    )

    audit = build_mask_audit(profile, seed=43)
    train = audit[audit["split"].eq("train")].iloc[0]

    assert train["m3_weighted_count"] == 1
    assert train["m6_weighted_count"] == 2
    assert math.isclose(train["m3_share"], 0.25)
    assert math.isclose(train["m6_share"], 0.50)


def test_contribution_summary_shows_positive_target_group_can_be_offset() -> None:
    delta_profile = pd.DataFrame(
        [
            {"split": "test", "raw_asin": "a", "high_acat_train_safe_hard_flag": True, "delta_ndcg@20": 0.04, "delta_hr@20": 0.01},
            {"split": "test", "raw_asin": "b", "high_acat_train_safe_hard_flag": False, "delta_ndcg@20": -0.02, "delta_hr@20": -0.01},
            {"split": "test", "raw_asin": "c", "high_acat_train_safe_hard_flag": False, "delta_ndcg@20": -0.02, "delta_hr@20": -0.01},
        ]
    )

    summary = build_contribution_summary(delta_profile, group_col="high_acat_train_safe_hard_flag")
    positive = summary[summary["group_value"].eq("True")].iloc[0]
    outside = summary[summary["group_value"].eq("False")].iloc[0]

    assert math.isclose(positive["mean_delta_ndcg@20"], 0.04)
    assert math.isclose(positive["overall_contribution_ndcg@20"], 0.04 / 3)
    assert math.isclose(outside["overall_contribution_ndcg@20"], -0.04 / 3)
    assert math.isclose(summary["overall_contribution_ndcg@20"].sum(), 0.0)


def test_candidate_table_prioritizes_pairwise_margin_and_keeps_negative_controls() -> None:
    table = build_candidate_carrier_table()

    assert table.iloc[0]["candidate_id"] == "M4a"
    assert "pairwise margin" in table.iloc[0]["carrier_family"]
    assert set(table["required_controls"]).issuperset({"M1_RSP_only", "M6_shuffle"})


def test_route_decision_enters_design_not_code_without_clean_shuffle_win() -> None:
    comparison = pd.DataFrame(
        [
            {
                "comparison": "M3_acat_trainhard_minus_M6_acat_shuffle",
                "delta_ndcg@20": 0.000012,
                "delta_hr@20": -0.000321,
                "meaningful_ndcg_win": False,
                "hr_not_worse": False,
            }
        ]
    )
    contribution = pd.DataFrame(
        [
            {
                "split": "test",
                "group_column": "high_acat_train_safe_hard_flag",
                "group_value": "True",
                "mean_delta_ndcg@20": 0.00077,
                "overall_contribution_ndcg@20": 0.00024,
            },
            {
                "split": "test",
                "group_column": "high_acat_train_safe_hard_flag",
                "group_value": "False",
                "mean_delta_ndcg@20": -0.00033,
                "overall_contribution_ndcg@20": -0.00023,
            },
        ]
    )

    decision = build_revise_route_decision(comparison, contribution)

    assert decision["route"] == "design_new_carrier"
    assert decision["enter_code_implementation"] is True
    assert decision["run_multi_seed_now"] is False
    assert decision["next_screen_seed"] == 43
