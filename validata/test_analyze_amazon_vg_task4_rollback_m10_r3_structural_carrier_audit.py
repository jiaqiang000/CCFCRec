import pandas as pd

from analyze_amazon_vg_task4_rollback_m10_r3_structural_carrier_audit import (
    build_control_correlation,
    build_group_summary,
    build_shuffle_summary,
    build_structural_delta_profile,
    decide_route,
)


def _baseline_items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"asin": "A", "category_group": "cat_weak_1_3", "ndcg@20": 0.00, "hr@20": 0.00},
            {"asin": "B", "category_group": "cat_weak_1_3", "ndcg@20": 0.10, "hr@20": 0.00},
            {"asin": "C", "category_group": "cat_strong_5_plus", "ndcg@20": 0.30, "hr@20": 0.05},
            {"asin": "D", "category_group": "cat_strong_5_plus", "ndcg@20": 0.40, "hr@20": 0.10},
            {"asin": "E", "category_group": "cat_mid_4", "ndcg@20": 0.20, "hr@20": 0.05},
            {"asin": "F", "category_group": "cat_mid_4", "ndcg@20": 0.25, "hr@20": 0.05},
        ]
    )


def _category_conf_items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"asin": "A", "ndcg@20": 0.08, "hr@20": 0.02},
            {"asin": "B", "ndcg@20": 0.14, "hr@20": 0.01},
            {"asin": "C", "ndcg@20": 0.25, "hr@20": 0.04},
            {"asin": "D", "ndcg@20": 0.35, "hr@20": 0.09},
            {"asin": "E", "ndcg@20": 0.21, "hr@20": 0.05},
            {"asin": "F", "ndcg@20": 0.23, "hr@20": 0.04},
        ]
    )


def _task4_profile() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"raw_asin": "A", "category_count": 2, "s_cat_v3": 0.1, "s_cat_v3_group": "s_cat_v3_weak", "RSP_score": 0.3, "RSP_group": "RSP_low", "high_acat_train_safe_hard_flag": False},
            {"raw_asin": "B", "category_count": 3, "s_cat_v3": 0.2, "s_cat_v3_group": "s_cat_v3_weak", "RSP_score": 0.4, "RSP_group": "RSP_low", "high_acat_train_safe_hard_flag": False},
            {"raw_asin": "C", "category_count": 6, "s_cat_v3": 0.9, "s_cat_v3_group": "s_cat_v3_strong", "RSP_score": 0.8, "RSP_group": "RSP_high", "high_acat_train_safe_hard_flag": True},
            {"raw_asin": "D", "category_count": 7, "s_cat_v3": 0.8, "s_cat_v3_group": "s_cat_v3_strong", "RSP_score": 0.7, "RSP_group": "RSP_high", "high_acat_train_safe_hard_flag": True},
            {"raw_asin": "E", "category_count": 4, "s_cat_v3": 0.5, "s_cat_v3_group": "s_cat_v3_mid", "RSP_score": 0.5, "RSP_group": "RSP_mid", "high_acat_train_safe_hard_flag": False},
            {"raw_asin": "F", "category_count": 4, "s_cat_v3": 0.6, "s_cat_v3_group": "s_cat_v3_mid", "RSP_score": 0.6, "RSP_group": "RSP_mid", "high_acat_train_safe_hard_flag": False},
        ]
    )


def test_structural_delta_profile_merges_method_delta_with_acat_and_rsp_controls() -> None:
    profile = build_structural_delta_profile(_baseline_items(), _category_conf_items(), _task4_profile())

    assert len(profile) == 6
    assert {"baseline_ndcg@20", "category_conf_ndcg@20", "delta_ndcg@20", "s_cat_v3_group", "RSP_group"}.issubset(profile.columns)
    row = profile[profile["raw_asin"].eq("A")].iloc[0]
    assert row["delta_ndcg@20"] == 0.08
    assert row["s_cat_v3_group"] == "s_cat_v3_weak"


def test_group_summary_shows_old_category_leverage_is_not_acat_clean() -> None:
    profile = build_structural_delta_profile(_baseline_items(), _category_conf_items(), _task4_profile())
    summary = build_group_summary(profile)

    weak_category = summary[(summary["scope"].eq("category_group")) & (summary["group"].eq("cat_weak_1_3"))].iloc[0]
    acat_strong = summary[(summary["scope"].eq("s_cat_v3_group")) & (summary["group"].eq("s_cat_v3_strong"))].iloc[0]

    assert weak_category["delta_ndcg@20_mean"] > 0.05
    assert acat_strong["delta_ndcg@20_mean"] < 0


def test_shuffle_summary_compares_real_acat_spread_with_placebo() -> None:
    profile = build_structural_delta_profile(_baseline_items(), _category_conf_items(), _task4_profile())
    shuffle = build_shuffle_summary(profile, n_shuffles=16, seed=7)

    assert shuffle["real_acat_high_minus_low_delta_ndcg@20"] < 0
    assert shuffle["shuffle_count"] == 16
    assert "shuffle_p95_acat_high_minus_low_delta_ndcg@20" in shuffle


def test_control_correlation_reports_acat_and_rsp_association() -> None:
    profile = build_structural_delta_profile(_baseline_items(), _category_conf_items(), _task4_profile())
    corr = build_control_correlation(profile)

    rights = set(corr["right"])
    assert {"s_cat_v3", "RSP_score", "category_count", "baseline_ndcg@20"}.issubset(rights)


def test_route_marks_structural_leverage_as_not_acat_clean_when_old_category_helps_but_acat_strong_hurts() -> None:
    profile = build_structural_delta_profile(_baseline_items(), _category_conf_items(), _task4_profile())
    summary = build_group_summary(profile)
    corr = build_control_correlation(profile)
    shuffle = build_shuffle_summary(profile, n_shuffles=16, seed=7)
    validation = pd.DataFrame(
        [
            {"run": "baseline", "best_ndcg@20": 0.123},
            {"run": "category_conf_input", "best_ndcg@20": 0.126},
        ]
    )

    decision = decide_route(validation, summary, corr, shuffle)

    assert decision["route"] == "structural_carrier_leverage_not_acat_clean"
    assert decision["enter_training_now"] is False
