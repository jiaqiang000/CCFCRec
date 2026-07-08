import math

import pandas as pd

from analyze_amazon_vg_task4_rollback_m10_r0_m9_post_audit import (
    build_acat_correlation,
    build_item_delta_profile,
    build_layer_summary,
    build_route_decision,
    classify_m9_run,
    enrich_item_delta_profile,
)


def test_classify_m9_run_marks_q_only_real_and_shuffle_by_alpha() -> None:
    assert classify_m9_run("task4_highdetail_trainhard_weight", 0.75, False, True) == "M9a075_q_only_real"
    assert classify_m9_run("task4_highdetail_trainhard_shuffle_weight", 0.75, False, True) == "M9a075_q_only_shuffle"
    assert classify_m9_run("task4_highdetail_trainhard_weight", 1.0, False, True) == "M9a100_q_only_real"
    assert classify_m9_run("task4_highdetail_trainhard_shuffle_weight", 1.0, False, True) == "M9a100_q_only_shuffle"


def test_item_delta_profile_pairs_real_and_shuffle_within_alpha_split_item() -> None:
    item_eval = pd.DataFrame(
        [
            {
                "run_label": "M9a075_q_only_real",
                "alpha": 0.75,
                "is_shuffle": False,
                "split": "test",
                "raw_asin": "A",
                "ndcg@20": 0.20,
                "hr@20": 0.05,
                "margin_to_top20_cutoff": 0.30,
                "best_target_rank": 18,
            },
            {
                "run_label": "M9a075_q_only_shuffle",
                "alpha": 0.75,
                "is_shuffle": True,
                "split": "test",
                "raw_asin": "A",
                "ndcg@20": 0.12,
                "hr@20": 0.00,
                "margin_to_top20_cutoff": -0.10,
                "best_target_rank": 31,
            },
            {
                "run_label": "M9a075_q_only_real",
                "alpha": 0.75,
                "is_shuffle": False,
                "split": "test",
                "raw_asin": "B",
                "ndcg@20": 0.00,
                "hr@20": 0.00,
                "margin_to_top20_cutoff": -0.50,
                "best_target_rank": 80,
            },
            {
                "run_label": "M9a075_q_only_shuffle",
                "alpha": 0.75,
                "is_shuffle": True,
                "split": "test",
                "raw_asin": "B",
                "ndcg@20": 0.04,
                "hr@20": 0.05,
                "margin_to_top20_cutoff": -0.20,
                "best_target_rank": 40,
            },
        ]
    )

    delta = build_item_delta_profile(item_eval)
    row_a = delta[delta["raw_asin"].eq("A")].iloc[0]
    row_b = delta[delta["raw_asin"].eq("B")].iloc[0]

    assert math.isclose(row_a["delta_ndcg@20"], 0.08)
    assert math.isclose(row_a["delta_hr@20"], 0.05)
    assert math.isclose(row_a["delta_margin_to_top20_cutoff"], 0.40)
    assert math.isclose(row_a["delta_best_target_rank"], -13.0)
    assert row_a["m9_helped_flag"] is True
    assert row_a["m9_harmed_flag"] is False

    assert math.isclose(row_b["delta_ndcg@20"], -0.04)
    assert math.isclose(row_b["delta_best_target_rank"], 40.0)
    assert row_b["m9_helped_flag"] is False
    assert row_b["m9_harmed_flag"] is True


def test_enrich_item_delta_profile_joins_task4_and_recoverability_audit_profiles() -> None:
    delta = pd.DataFrame(
        [
            {"alpha": 0.75, "split": "test", "raw_asin": "A", "delta_ndcg@20": 0.08},
            {"alpha": 0.75, "split": "test", "raw_asin": "B", "delta_ndcg@20": -0.02},
        ]
    )
    task4 = pd.DataFrame(
        [
            {
                "raw_asin": "A",
                "split": "test",
                "category_count": 5,
                "cat_count_bin": "cat_count_5_plus",
                "s_cat_v3": 0.81,
                "s_cat_v3_group": "s_cat_v3_strong",
                "RSP_group": "RSP_low",
                "high_detail_flag": True,
                "high_acat_train_safe_hard_flag": True,
                "train_safe_hard_proxy_group": "category_neighbor_mismatch_proxy_high",
            },
            {
                "raw_asin": "B",
                "split": "test",
                "category_count": 3,
                "cat_count_bin": "cat_count_1_3",
                "s_cat_v3": 0.22,
                "s_cat_v3_group": "s_cat_v3_weak",
                "RSP_group": "RSP_high",
                "high_detail_flag": False,
                "high_acat_train_safe_hard_flag": False,
                "train_safe_hard_proxy_group": "category_neighbor_mismatch_proxy_low",
            },
        ]
    )
    near = pd.DataFrame(
        [
            {"asin": "A", "target_rank_near_cutoff": True, "target_history_bucket": "target_activity_high"},
            {"asin": "B", "target_rank_near_cutoff": False, "target_history_bucket": "target_activity_mid"},
        ]
    )
    recoverability = pd.DataFrame(
        [
            {"asin": "A", "proxy_ensemble_group": "ensemble_high"},
            {"asin": "B", "proxy_ensemble_group": "ensemble_low"},
        ]
    )
    failure = pd.DataFrame(
        [
            {"asin": "A", "consensus_group": "consensus_high"},
            {"asin": "B", "consensus_group": "consensus_low"},
        ]
    )
    rank = pd.DataFrame(
        [
            {"asin": "A", "rank_aware_group": "rank_mid"},
            {"asin": "B", "rank_aware_group": "rank_low"},
        ]
    )

    enriched = enrich_item_delta_profile(delta, task4, near, recoverability, failure, rank)
    row_a = enriched[enriched["raw_asin"].eq("A")].iloc[0]

    assert row_a["s_cat_v3_group"] == "s_cat_v3_strong"
    assert row_a["near_cutoff_group"] == "near_cutoff"
    assert row_a["recoverability_proxy_ensemble_group"] == "ensemble_high"
    assert row_a["failure_consensus_group"] == "consensus_high"
    assert row_a["rank_recoverability_group"] == "rank_mid"
    assert row_a["target_history_bucket"] == "target_activity_high"


def test_layer_summary_reports_helped_and_harmed_counts_by_group() -> None:
    enriched = pd.DataFrame(
        [
            {
                "alpha": 0.75,
                "split": "test",
                "raw_asin": "A",
                "delta_ndcg@20": 0.08,
                "delta_hr@20": 0.05,
                "delta_margin_to_top20_cutoff": 0.40,
                "delta_best_target_rank": -13,
                "m9_helped_flag": True,
                "m9_harmed_flag": False,
                "near_cutoff_group": "near_cutoff",
            },
            {
                "alpha": 0.75,
                "split": "test",
                "raw_asin": "B",
                "delta_ndcg@20": -0.02,
                "delta_hr@20": -0.05,
                "delta_margin_to_top20_cutoff": -0.10,
                "delta_best_target_rank": 20,
                "m9_helped_flag": False,
                "m9_harmed_flag": True,
                "near_cutoff_group": "not_near_cutoff",
            },
        ]
    )

    summary = build_layer_summary(enriched, group_columns=["near_cutoff_group"])
    near_row = summary[summary["group_value"].eq("near_cutoff")].iloc[0]
    other_row = summary[summary["group_value"].eq("not_near_cutoff")].iloc[0]

    assert near_row["item_count"] == 1
    assert near_row["helped_item_count"] == 1
    assert near_row["harmed_item_count"] == 0
    assert math.isclose(near_row["delta_ndcg@20_mean"], 0.08)
    assert other_row["harmed_item_count"] == 1


def test_route_decision_uses_tradeoff_before_weak_acat_alignment() -> None:
    delta = pd.DataFrame(
        [
            {"alpha": 0.75, "split": "test", "raw_asin": "A", "delta_ndcg@20": 0.08, "m9_helped_flag": True, "m9_harmed_flag": False},
            {"alpha": 0.75, "split": "test", "raw_asin": "B", "delta_ndcg@20": 0.02, "m9_helped_flag": True, "m9_harmed_flag": False},
            {"alpha": 0.75, "split": "test", "raw_asin": "C", "delta_ndcg@20": -0.04, "m9_helped_flag": False, "m9_harmed_flag": True},
            {"alpha": 0.75, "split": "test", "raw_asin": "D", "delta_ndcg@20": -0.03, "m9_helped_flag": False, "m9_harmed_flag": True},
        ]
    )
    layer = build_layer_summary(delta, group_columns=[])
    acat_corr = build_acat_correlation(pd.DataFrame())

    decision = build_route_decision(delta, layer, acat_corr)

    assert decision["route"] == "m9_tradeoff_requires_rollback"
    assert decision["run_m9_style_m10"] is False


def test_route_decision_marks_weak_acat_alignment_when_no_dominant_layer() -> None:
    delta = pd.DataFrame(
        [
            {"alpha": 0.75, "split": "test", "raw_asin": "A", "delta_ndcg@20": 0.03, "m9_helped_flag": True, "m9_harmed_flag": False, "s_cat_v3": 0.1},
            {"alpha": 0.75, "split": "test", "raw_asin": "B", "delta_ndcg@20": 0.03, "m9_helped_flag": True, "m9_harmed_flag": False, "s_cat_v3": 0.9},
            {"alpha": 0.75, "split": "test", "raw_asin": "C", "delta_ndcg@20": 0.03, "m9_helped_flag": True, "m9_harmed_flag": False, "s_cat_v3": 0.5},
        ]
    )
    layer = build_layer_summary(delta, group_columns=[])
    acat_corr = build_acat_correlation(delta)

    decision = build_route_decision(delta, layer, acat_corr, min_abs_acat_spearman=0.10)

    assert decision["route"] == "m9_signal_not_acat_aligned"
    assert decision["next_action"] == "do_not_continue_acat_only_loss_weight"


def test_route_decision_does_not_sum_dominant_layer_across_splits() -> None:
    delta = pd.DataFrame(
        [
            {
                "alpha": 0.75,
                "split": "validate",
                "raw_asin": "A",
                "delta_ndcg@20": 0.40,
                "m9_helped_flag": True,
                "m9_harmed_flag": False,
                "high_acat_train_safe_hard_flag": True,
                "s_cat_v3": 0.2,
            },
            {
                "alpha": 0.75,
                "split": "validate",
                "raw_asin": "B",
                "delta_ndcg@20": 0.60,
                "m9_helped_flag": True,
                "m9_harmed_flag": False,
                "high_acat_train_safe_hard_flag": False,
                "s_cat_v3": 0.8,
            },
            {
                "alpha": 0.75,
                "split": "test",
                "raw_asin": "C",
                "delta_ndcg@20": 0.40,
                "m9_helped_flag": True,
                "m9_harmed_flag": False,
                "high_acat_train_safe_hard_flag": True,
                "s_cat_v3": 0.3,
            },
            {
                "alpha": 0.75,
                "split": "test",
                "raw_asin": "D",
                "delta_ndcg@20": 0.60,
                "m9_helped_flag": True,
                "m9_harmed_flag": False,
                "high_acat_train_safe_hard_flag": False,
                "s_cat_v3": 0.7,
            },
        ]
    )
    layer = build_layer_summary(delta, group_columns=["high_acat_train_safe_hard_flag"])
    acat_corr = build_acat_correlation(delta)

    decision = build_route_decision(delta, layer, acat_corr, min_dominant_positive_share=0.60)

    assert decision["route"] == "m9_signal_not_acat_aligned"
