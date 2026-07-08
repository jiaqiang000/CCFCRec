import math

import pandas as pd

from analyze_amazon_vg_task4_post import (
    build_group_delta_summary,
    build_method_comparison,
    build_route_decision,
    select_best_checkpoint_from_result,
)


def test_select_best_checkpoint_uses_validate_ndcg_then_hr_then_earlier_epoch() -> None:
    result = pd.DataFrame(
        [
            {"checkpoint_index": 1, "epoch": 1, "ndcg@20": 0.10, "hr@20": 0.020},
            {"checkpoint_index": 2, "epoch": 2, "ndcg@20": 0.12, "hr@20": 0.018},
            {"checkpoint_index": 3, "epoch": 3, "ndcg@20": 0.12, "hr@20": 0.021},
            {"checkpoint_index": 4, "epoch": 4, "ndcg@20": 0.12, "hr@20": 0.021},
        ]
    )

    best = select_best_checkpoint_from_result(result)

    assert int(best["checkpoint_index"]) == 3
    assert int(best["epoch"]) == 3


def test_method_comparison_marks_tiny_shuffle_win_as_not_meaningful() -> None:
    best = pd.DataFrame(
        [
            {"method_variant": "task4_rsp_high_weight", "best_ndcg@20": 0.123012, "best_hr@20": 0.020461},
            {"method_variant": "task4_acat_high_weight", "best_ndcg@20": 0.123566, "best_hr@20": 0.021027},
            {"method_variant": "task4_acat_shuffle_high_weight", "best_ndcg@20": 0.123855, "best_hr@20": 0.021083},
            {"method_variant": "task4_acat_trainhard_weight", "best_ndcg@20": 0.123867, "best_hr@20": 0.020763},
        ]
    )

    comparison = build_method_comparison(best, min_meaningful_ndcg_delta=0.0005)
    row = comparison[
        comparison["comparison"].eq("M3_acat_trainhard_minus_M6_acat_shuffle")
    ].iloc[0]

    assert math.isclose(row["delta_ndcg@20"], 0.000012, abs_tol=1e-6)
    assert row["meaningful_ndcg_win"] is False
    assert row["hr_not_worse"] is False


def test_route_decision_revises_m3_when_shuffle_gap_is_tiny_and_hr_reverses() -> None:
    best = pd.DataFrame(
        [
            {"method_variant": "task4_rsp_high_weight", "best_ndcg@20": 0.123012, "best_hr@20": 0.020461},
            {"method_variant": "task4_acat_shuffle_high_weight", "best_ndcg@20": 0.123855, "best_hr@20": 0.021083},
            {"method_variant": "task4_acat_trainhard_weight", "best_ndcg@20": 0.123867, "best_hr@20": 0.020763},
        ]
    )
    comparison = build_method_comparison(best, min_meaningful_ndcg_delta=0.0005)
    dynamics = pd.DataFrame(
        [
            {"method_variant": "task4_acat_trainhard_weight", "peak_minus_final_ndcg@20": 0.0030},
        ]
    )

    decision = build_route_decision(best, comparison, dynamics, group_delta_summary=pd.DataFrame())

    assert decision["route"] == "revise_m3"
    assert decision["need_seed_repeat"] is False
    assert decision["seed_repeat_gate_status"] == "not_open_current_m3"
    assert decision["go_m4"] is False


def test_group_delta_summary_compares_m3_against_rsp_and_shuffle_within_group() -> None:
    group_summary = pd.DataFrame(
        [
            {"split": "test", "group_column": "high_acat_flag", "group_value": "True", "method_variant": "task4_rsp_high_weight", "ndcg@20_mean": 0.10, "hr@20_mean": 0.020},
            {"split": "test", "group_column": "high_acat_flag", "group_value": "True", "method_variant": "task4_acat_shuffle_high_weight", "ndcg@20_mean": 0.11, "hr@20_mean": 0.021},
            {"split": "test", "group_column": "high_acat_flag", "group_value": "True", "method_variant": "task4_acat_trainhard_weight", "ndcg@20_mean": 0.12, "hr@20_mean": 0.022},
        ]
    )

    delta = build_group_delta_summary(group_summary)
    row = delta.iloc[0]

    assert row["split"] == "test"
    assert row["group_column"] == "high_acat_flag"
    assert row["group_value"] == "True"
    assert math.isclose(row["m3_minus_m1_ndcg@20_mean"], 0.02)
    assert math.isclose(row["m3_minus_m6_ndcg@20_mean"], 0.01)
