import math

import pandas as pd

from analyze_amazon_vg_task4_revise3_m8_surface import (
    build_pair_comparison,
    build_route_decision,
    classify_m8_run,
)


def test_classify_m8_run_uses_disable_flags() -> None:
    assert classify_m8_run("task4_highdetail_trainhard_weight", False, True) == "M8q_q_only_real"
    assert classify_m8_run("task4_highdetail_trainhard_shuffle_weight", False, True) == "M8qs_q_only_shuffle"
    assert classify_m8_run("task4_highdetail_trainhard_weight", True, False) == "M8s_self_only_real"
    assert classify_m8_run("task4_highdetail_trainhard_shuffle_weight", True, False) == "M8ss_self_only_shuffle"


def test_pair_comparison_marks_q_only_positive_but_below_gate() -> None:
    best = pd.DataFrame(
        [
            {"run_label": "M8q_q_only_real", "best_ndcg@20": 0.124049, "best_hr@20": 0.020630},
            {"run_label": "M8qs_q_only_shuffle", "best_ndcg@20": 0.123919, "best_hr@20": 0.020545},
            {"run_label": "M8s_self_only_real", "best_ndcg@20": 0.123679, "best_hr@20": 0.020687},
            {"run_label": "M8ss_self_only_shuffle", "best_ndcg@20": 0.124088, "best_hr@20": 0.020800},
        ]
    )

    comparison = build_pair_comparison(best, min_ndcg_delta=0.0005)
    q_row = comparison[comparison["comparison"].eq("M8q_minus_M8qs")].iloc[0]
    self_row = comparison[comparison["comparison"].eq("M8s_minus_M8ss")].iloc[0]

    assert math.isclose(q_row["delta_ndcg@20"], 0.00013, abs_tol=1e-8)
    assert bool(q_row["pass_hr_gate"]) is True
    assert bool(q_row["pass_seed43_gate"]) is False
    assert self_row["delta_ndcg@20"] < 0


def test_route_decision_keeps_q_only_and_drops_self_only() -> None:
    comparison = pd.DataFrame(
        [
            {"comparison": "M8q_minus_M8qs", "delta_ndcg@20": 0.00013, "delta_hr@20": 0.00008, "pass_seed43_gate": False},
            {"comparison": "M8s_minus_M8ss", "delta_ndcg@20": -0.00041, "delta_hr@20": -0.00011, "pass_seed43_gate": False},
        ]
    )

    decision = build_route_decision(comparison)

    assert decision["route"] == "m8_gate_failed_but_q_only_is_safer"
    assert decision["run_multi_seed_now"] is False
    assert decision["next_action"] == "m9_q_only_alpha_sweep_seed43"
