import math

import pandas as pd

from analyze_amazon_vg_task4_post0_detail_acat import (
    attach_detail_groups,
    build_detail_acat_summary,
    build_detail_trend_checks,
    build_m4_gate_summary,
    build_method_layer_delta,
    build_method_layer_summary,
    build_within_detail_acat_contrast,
    build_within_detail_acat_summary,
)


def test_detail_trend_check_separates_original_ndcg_and_acat_trends() -> None:
    data = pd.DataFrame(
        [
            {"split": "test", "raw_asin": "a", "cat_count_bin": "cat_count_1_3", "category_count": 2, "s_cat_v3": 0.60, "s_cat_v3_group": "s_cat_v3_strong", "ndcg@20": 0.10, "hr@20": 0.010},
            {"split": "test", "raw_asin": "b", "cat_count_bin": "cat_count_4", "category_count": 4, "s_cat_v3": 0.40, "s_cat_v3_group": "s_cat_v3_mid", "ndcg@20": 0.20, "hr@20": 0.020},
            {"split": "test", "raw_asin": "c", "cat_count_bin": "cat_count_5_plus", "category_count": 5, "s_cat_v3": 0.50, "s_cat_v3_group": "s_cat_v3_weak", "ndcg@20": 0.30, "hr@20": 0.015},
        ]
    )

    summary = build_detail_acat_summary(data)
    checks = build_detail_trend_checks(summary)
    test_row = checks[checks["split"].eq("test")].iloc[0]

    assert bool(test_row["baseline_ndcg_strict_increasing"]) is True
    assert bool(test_row["baseline_hr_strict_increasing"]) is False
    assert bool(test_row["acat_mean_strict_increasing"]) is False


def test_within_detail_contrast_can_show_high_acat_baseline_failure() -> None:
    data = pd.DataFrame(
        [
            {"split": "test", "raw_asin": "a", "cat_count_bin": "cat_count_5_plus", "category_count": 5, "s_cat_v3": 0.20, "s_cat_v3_group": "s_cat_v3_weak", "ndcg@20": 0.30, "hr@20": 0.030},
            {"split": "test", "raw_asin": "b", "cat_count_bin": "cat_count_5_plus", "category_count": 5, "s_cat_v3": 0.50, "s_cat_v3_group": "s_cat_v3_mid", "ndcg@20": 0.20, "hr@20": 0.020},
            {"split": "test", "raw_asin": "c", "cat_count_bin": "cat_count_5_plus", "category_count": 6, "s_cat_v3": 0.80, "s_cat_v3_group": "s_cat_v3_strong", "ndcg@20": 0.10, "hr@20": 0.010},
        ]
    )

    summary = build_within_detail_acat_summary(data)
    contrast = build_within_detail_acat_contrast(summary)
    row = contrast.iloc[0]

    assert math.isclose(row["strong_minus_weak_ndcg@20_mean"], -0.20)
    assert row["best_acat_group_by_ndcg@20"] == "s_cat_v3_weak"
    assert bool(row["strong_acat_is_best_ndcg@20"]) is False


def test_method_layer_delta_compares_m3_against_shuffle_inside_detail_layer() -> None:
    availability = pd.DataFrame(
        [
            {"split": "test", "raw_asin": "a", "cat_count_bin": "cat_count_4"},
            {"split": "test", "raw_asin": "b", "cat_count_bin": "cat_count_4"},
        ]
    )
    item_eval = pd.DataFrame(
        [
            {"split": "test", "raw_asin": "a", "method_variant": "task4_acat_trainhard_weight", "high_acat_flag": True, "ndcg@20": 0.20, "hr@20": 0.02},
            {"split": "test", "raw_asin": "a", "method_variant": "task4_acat_shuffle_high_weight", "high_acat_flag": True, "ndcg@20": 0.15, "hr@20": 0.01},
            {"split": "test", "raw_asin": "b", "method_variant": "task4_acat_trainhard_weight", "high_acat_flag": False, "ndcg@20": 0.10, "hr@20": 0.01},
            {"split": "test", "raw_asin": "b", "method_variant": "task4_acat_shuffle_high_weight", "high_acat_flag": False, "ndcg@20": 0.12, "hr@20": 0.02},
        ]
    )

    merged = attach_detail_groups(item_eval, availability)
    layer_summary = build_method_layer_summary(merged)
    delta = build_method_layer_delta(layer_summary)
    high_row = delta[delta["high_acat_flag"].eq("True")].iloc[0]
    low_row = delta[delta["high_acat_flag"].eq("False")].iloc[0]

    assert math.isclose(high_row["m3_minus_m6_ndcg@20_mean"], 0.05)
    assert math.isclose(low_row["m3_minus_m6_ndcg@20_mean"], -0.02)


def test_m4_gate_summary_reads_metric_columns_with_at_symbol(tmp_path) -> None:
    best_csv = tmp_path / "best.csv"
    control_csv = tmp_path / "control.csv"
    pd.DataFrame(
        [
            {
                "method_id": "M4b",
                "method_variant": "task4_acat_rsp_residual_pairmargin",
                "best_epoch": 40,
                "best_ndcg@20": 0.123,
                "best_hr@20": 0.021,
            }
        ]
    ).to_csv(best_csv, index=False)
    pd.DataFrame(
        [
            {
                "method_id": "M4b",
                "method_variant": "task4_acat_rsp_residual_pairmargin",
                "control_id": "M6",
                "delta_ndcg@20": -0.001,
                "delta_hr@20": -0.002,
                "pass_seed43_gate": False,
            }
        ]
    ).to_csv(control_csv, index=False)

    summary = build_m4_gate_summary(best_csv, control_csv)
    row = summary.iloc[0]

    assert math.isclose(row["best_ndcg@20"], 0.123)
    assert math.isclose(row["delta_vs_M6_ndcg@20"], -0.001)
    assert bool(row["pass_seed43_gate"]) is False
