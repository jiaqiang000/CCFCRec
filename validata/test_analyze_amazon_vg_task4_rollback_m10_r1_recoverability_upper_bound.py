import math

import pandas as pd

from analyze_amazon_vg_task4_rollback_m10_r1_recoverability_upper_bound import (
    build_placebo_summary,
    build_route_decision,
    build_upper_bound_profile,
    build_upper_bound_summary,
    single_hit_ndcg_floor,
)


def test_upper_bound_profile_builds_required_hard_candidate_masks() -> None:
    task4 = pd.DataFrame(
        [
            {
                "raw_asin": "A",
                "split": "test",
                "baseline_ndcg@20": 0.0,
                "eval_baseline_hard_flag": True,
                "high_acat_flag": True,
                "RSP_group": "RSP_low",
                "s_cat_v3": 0.8,
            },
            {
                "raw_asin": "B",
                "split": "test",
                "baseline_ndcg@20": 0.4,
                "eval_baseline_hard_flag": False,
                "high_acat_flag": False,
                "RSP_group": "RSP_high",
                "s_cat_v3": 0.2,
            },
        ]
    )
    near = pd.DataFrame([{"asin": "A", "target_rank_near_cutoff": True}, {"asin": "B", "target_rank_near_cutoff": False}])
    recoverability = pd.DataFrame([{"asin": "A", "proxy_ensemble_group": "ensemble_high"}, {"asin": "B", "proxy_ensemble_group": "ensemble_low"}])
    failure = pd.DataFrame([{"asin": "A", "consensus_group": "consensus_high"}, {"asin": "B", "consensus_group": "consensus_low"}])
    rank = pd.DataFrame([{"asin": "A", "rank_aware_group": "rank_high"}, {"asin": "B", "rank_aware_group": "rank_low"}])

    profile = build_upper_bound_profile(task4, near, recoverability, failure, rank)
    row_a = profile[profile["raw_asin"].eq("A")].iloc[0]
    row_b = profile[profile["raw_asin"].eq("B")].iloc[0]

    assert row_a["recoverability_ensemble_high_hard"] is True
    assert row_a["rank_recoverability_high_hard"] is True
    assert row_a["near_cutoff_hard"] is True
    assert row_a["failure_consensus_high"] is True
    assert row_a["acat_v3_high_hard"] is True
    assert row_b["rsp_high_hard"] is False


def test_upper_bound_summary_uses_single_hit_and_perfect_recovery_formulas() -> None:
    floor = single_hit_ndcg_floor()
    profile = pd.DataFrame(
        [
            {"raw_asin": "A", "baseline_ndcg@20": 0.0, "recoverability_ensemble_high_hard": True},
            {"raw_asin": "B", "baseline_ndcg@20": 0.5, "recoverability_ensemble_high_hard": True},
            {"raw_asin": "C", "baseline_ndcg@20": 0.25, "recoverability_ensemble_high_hard": False},
            {"raw_asin": "D", "baseline_ndcg@20": 0.25, "recoverability_ensemble_high_hard": False},
        ]
    )

    summary = build_upper_bound_summary(profile, candidate_columns=["recoverability_ensemble_high_hard"])
    row = summary.iloc[0]

    assert row["candidate"] == "recoverability_ensemble_high_hard"
    assert row["selected_count"] == 2
    assert math.isclose(row["baseline_item_ndcg_mean"], 0.25)
    assert math.isclose(row["three_pct_abs_gate"], 0.0075)
    assert math.isclose(row["single_hit_upper_gain_abs"], floor / 4)
    assert math.isclose(row["perfect_recovery_upper_gain_abs"], (1.0 + 0.5) / 4)
    assert row["passes_single_hit_3pct_gate"] is True
    assert row["passes_perfect_recovery_3pct_gate"] is True


def test_placebo_summary_preserves_selected_count_and_reports_p95() -> None:
    profile = pd.DataFrame(
        [
            {"raw_asin": f"I{i}", "baseline_ndcg@20": 0.0 if i < 4 else 0.5, "recoverability_ensemble_high_hard": i < 2}
            for i in range(8)
        ]
    )

    placebo = build_placebo_summary(
        profile,
        candidate_columns=["recoverability_ensemble_high_hard"],
        shuffle_count=20,
        random_seed=7,
    )
    row = placebo.iloc[0]

    assert row["candidate"] == "recoverability_ensemble_high_hard"
    assert row["selected_count"] == 2
    assert row["shuffle_count"] == 20
    assert row["single_hit_upper_gain_abs_p95"] >= row["single_hit_upper_gain_abs_mean"]


def test_route_decision_opens_r2_when_recoverability_single_hit_beats_gate_and_shuffle() -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate": "recoverability_ensemble_high_hard",
                "candidate_type": "recoverability",
                "single_hit_upper_gain_abs": 0.020,
                "perfect_recovery_upper_gain_abs": 0.20,
                "three_pct_abs_gate": 0.010,
            },
            {
                "candidate": "acat_v3_high_hard",
                "candidate_type": "acat_control",
                "single_hit_upper_gain_abs": 0.008,
                "perfect_recovery_upper_gain_abs": 0.18,
                "three_pct_abs_gate": 0.010,
            },
            {
                "candidate": "rsp_high_hard",
                "candidate_type": "rsp_control",
                "single_hit_upper_gain_abs": 0.007,
                "perfect_recovery_upper_gain_abs": 0.16,
                "three_pct_abs_gate": 0.010,
            },
        ]
    )
    placebo = pd.DataFrame(
        [
            {
                "candidate": "recoverability_ensemble_high_hard",
                "single_hit_upper_gain_abs_p95": 0.012,
                "perfect_recovery_upper_gain_abs_p95": 0.18,
            }
        ]
    )

    decision = build_route_decision(summary, placebo)

    assert decision["route"] == "recoverability_single_hit_space_exists"
    assert decision["continue_overall_3pct_search"] is True


def test_route_decision_keeps_space_open_when_controls_are_larger_but_recoverability_beats_shuffle() -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate": "failure_consensus_high",
                "candidate_type": "recoverability",
                "single_hit_upper_gain_abs": 0.055,
                "perfect_recovery_upper_gain_abs": 0.24,
                "three_pct_abs_gate": 0.010,
            },
            {
                "candidate": "acat_v3_high_hard",
                "candidate_type": "acat_control",
                "single_hit_upper_gain_abs": 0.068,
                "perfect_recovery_upper_gain_abs": 0.30,
                "three_pct_abs_gate": 0.010,
            },
        ]
    )
    placebo = pd.DataFrame(
        [
            {
                "candidate": "failure_consensus_high",
                "single_hit_upper_gain_abs_p95": 0.044,
                "perfect_recovery_upper_gain_abs_p95": 0.22,
            }
        ]
    )

    decision = build_route_decision(summary, placebo)

    assert decision["route"] == "recoverability_single_hit_space_exists"
    assert decision["continue_overall_3pct_search"] is True
    assert decision["recoverability_beats_controls"] is False
    assert decision["control_dominance_warning"] is True


def test_route_decision_blocks_when_only_perfect_recovery_passes() -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate": "rank_recoverability_high_hard",
                "candidate_type": "recoverability",
                "single_hit_upper_gain_abs": 0.004,
                "perfect_recovery_upper_gain_abs": 0.050,
                "three_pct_abs_gate": 0.010,
            }
        ]
    )
    placebo = pd.DataFrame(
        [
            {
                "candidate": "rank_recoverability_high_hard",
                "single_hit_upper_gain_abs_p95": 0.005,
                "perfect_recovery_upper_gain_abs_p95": 0.030,
            }
        ]
    )

    decision = build_route_decision(summary, placebo)

    assert decision["route"] == "only_perfect_recovery_space_exists"
    assert decision["continue_overall_3pct_search"] is False
