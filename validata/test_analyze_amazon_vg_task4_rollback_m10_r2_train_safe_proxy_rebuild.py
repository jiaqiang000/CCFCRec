import math

import pandas as pd

from analyze_amazon_vg_task4_rollback_m10_r2_train_safe_proxy_rebuild import (
    build_candidate_summary,
    build_proxy_profile,
    build_r1_overlap_summary,
    decide_route,
)


def _base_profile() -> pd.DataFrame:
    rows = []
    for i in range(6):
        rows.append(
            {
                "raw_asin": f"T{i}",
                "split": "train",
                "category_count": 3 + i,
                "s_cat_v3": 0.2 + i * 0.1,
                "RSP_score": 0.8 - i * 0.1,
                "RSP_group": "RSP_high" if i < 2 else "RSP_low",
                "P_popularity_score": 0.2 + i * 0.1,
                "S_train_support_score": 0.8 - i * 0.1,
                "S_train_token_user_support_mean": 10 + i,
                "S_train_token_interaction_support_mean": 20 + i,
                "A_collab_train_token_user_support_mean": 30 + i,
                "A_collab_user_set_jaccard_mean": 0.8 - i * 0.1,
                "A_collab_support_entropy_mean": 0.2 + i * 0.1,
                "Acat_v3_disc_residual_pct": 0.2 + i * 0.1,
                "Acat_v3_collab_residual_pct": 0.2 + i * 0.1,
                "eval_baseline_hard_flag": False,
                "eval_metric_available_flag": False,
                "baseline_ndcg@20": None,
                "baseline_margin_proxy": None,
                "baseline_best_target_rank": None,
            }
        )
    eval_rows = [
        ("A", 0.0, -1.0, 100, True, 0.95, 0.10, "RSP_low"),
        ("B", 0.0, -0.8, 80, True, 0.90, 0.20, "RSP_low"),
        ("C", 0.2, -0.3, 30, True, 0.85, 0.30, "RSP_low"),
        ("D", 0.5, 0.2, 10, False, 0.20, 0.80, "RSP_high"),
        ("E", 0.6, 0.5, 5, False, 0.15, 0.90, "RSP_high"),
        ("F", 0.7, 0.7, 3, False, 0.10, 0.95, "RSP_high"),
    ]
    for asin, ndcg, margin, rank, hard, pressure, rsp_score, rsp_group in eval_rows:
        rows.append(
            {
                "raw_asin": asin,
                "split": "test",
                "category_count": 5,
                "s_cat_v3": pressure,
                "RSP_score": rsp_score,
                "RSP_group": rsp_group,
                "P_popularity_score": pressure,
                "S_train_support_score": 1.0 - pressure,
                "S_train_token_user_support_mean": pressure * 100,
                "S_train_token_interaction_support_mean": pressure * 100,
                "A_collab_train_token_user_support_mean": pressure * 100,
                "A_collab_user_set_jaccard_mean": 1.0 - pressure,
                "A_collab_support_entropy_mean": pressure,
                "Acat_v3_disc_residual_pct": pressure,
                "Acat_v3_collab_residual_pct": pressure,
                "eval_baseline_hard_flag": hard,
                "eval_metric_available_flag": True,
                "baseline_ndcg@20": ndcg,
                "baseline_margin_proxy": margin,
                "baseline_best_target_rank": rank,
            }
        )
    return pd.DataFrame(rows)


def test_proxy_profile_adds_diagnostic_and_train_deployable_scores() -> None:
    profile = build_proxy_profile(_base_profile())

    required = {
        "eval_margin_proxy_score",
        "eval_near_cutoff_proxy_score",
        "competitor_pressure_proxy_score",
        "train_graph_near_cutoff_proxy_score",
        "residual_acat_pressure_proxy_score",
        "eval_margin_proxy_high_flag",
        "competitor_pressure_proxy_high_flag",
    }

    assert required.issubset(set(profile.columns))
    hard_row = profile[profile["raw_asin"].eq("A")].iloc[0]
    easy_row = profile[profile["raw_asin"].eq("F")].iloc[0]
    assert hard_row["eval_margin_proxy_score"] > easy_row["eval_margin_proxy_score"]
    assert hard_row["competitor_pressure_proxy_score"] > easy_row["competitor_pressure_proxy_score"]


def test_candidate_summary_reports_hard_rate_lift_and_gate_flags() -> None:
    profile = build_proxy_profile(_base_profile())
    summary = build_candidate_summary(profile)
    row = summary[summary["candidate"].eq("eval_margin_proxy")].iloc[0]

    assert row["candidate_type"] == "diagnostic"
    assert math.isclose(row["base_hard_rate"], 0.5)
    assert row["proxy_high_hard_rate"] > 0.9
    assert row["hard_rate_lift"] >= 0.4
    assert row["spearman_vs_eval_baseline_hard_flag"] > 0.7
    assert row["selected_count"] < 500
    assert row["passes_main_gate"] is False


def test_route_prefers_train_deployable_proxy_when_it_passes() -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate": "competitor_pressure_proxy",
                "candidate_type": "train_deployable",
                "passes_main_gate": True,
                "hard_rate_lift": 0.12,
                "spearman_vs_eval_baseline_hard_flag": 0.31,
            },
            {
                "candidate": "eval_margin_proxy",
                "candidate_type": "diagnostic",
                "passes_main_gate": True,
                "hard_rate_lift": 0.40,
                "spearman_vs_eval_baseline_hard_flag": 0.80,
            },
        ]
    )

    decision = decide_route(summary)

    assert decision["route"] == "train_deployable_proxy_ready"
    assert decision["selected_candidate"] == "competitor_pressure_proxy"
    assert decision["enter_training_now"] is False


def test_route_reports_diagnostic_only_when_no_train_deployable_passes() -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate": "competitor_pressure_proxy",
                "candidate_type": "train_deployable",
                "passes_main_gate": False,
                "hard_rate_lift": 0.02,
                "spearman_vs_eval_baseline_hard_flag": 0.10,
            },
            {
                "candidate": "eval_margin_proxy",
                "candidate_type": "diagnostic",
                "passes_main_gate": True,
                "hard_rate_lift": 0.40,
                "spearman_vs_eval_baseline_hard_flag": 0.80,
            },
        ]
    )

    decision = decide_route(summary)

    assert decision["route"] == "diagnostic_proxy_only_space_exists"
    assert decision["selected_candidate"] == "eval_margin_proxy"
    assert decision["enter_training_now"] is False


def test_r1_overlap_summary_counts_overlap_with_upper_bound_masks() -> None:
    profile = build_proxy_profile(_base_profile())
    r1 = pd.DataFrame(
        [
            {"raw_asin": "A", "recoverability_ensemble_high_hard": True, "near_cutoff_hard": True},
            {"raw_asin": "B", "recoverability_ensemble_high_hard": True, "near_cutoff_hard": False},
            {"raw_asin": "C", "recoverability_ensemble_high_hard": False, "near_cutoff_hard": False},
            {"raw_asin": "D", "recoverability_ensemble_high_hard": False, "near_cutoff_hard": False},
        ]
    )

    overlap = build_r1_overlap_summary(profile, r1)
    row = overlap[
        overlap["candidate"].eq("eval_margin_proxy")
        & overlap["r1_mask"].eq("recoverability_ensemble_high_hard")
    ].iloc[0]

    assert row["r1_mask_count"] == 2
    assert row["selected_r1_overlap_count"] >= 1
