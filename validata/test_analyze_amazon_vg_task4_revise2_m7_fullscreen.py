import math

import pandas as pd

from analyze_amazon_vg_task4_revise2_m7_fullscreen import (
    build_pair_comparison,
    build_route_decision,
    select_best_checkpoint,
)


def test_select_best_checkpoint_uses_ndcg_then_hr_then_earlier_epoch() -> None:
    result = pd.DataFrame(
        [
            {"checkpoint_index": 1, "epoch": 1, "ndcg@20": 0.10, "hr@20": 0.020},
            {"checkpoint_index": 2, "epoch": 2, "ndcg@20": 0.12, "hr@20": 0.018},
            {"checkpoint_index": 3, "epoch": 3, "ndcg@20": 0.12, "hr@20": 0.021},
            {"checkpoint_index": 4, "epoch": 4, "ndcg@20": 0.12, "hr@20": 0.021},
        ]
    )

    best = select_best_checkpoint(result)

    assert int(best["checkpoint_index"]) == 3


def test_pair_comparison_marks_borderline_ndcg_and_negative_hr_as_fail() -> None:
    best = pd.DataFrame(
        [
            {"method_variant": "task4_highdetail_trainhard_weight", "best_ndcg@20": 0.123997, "best_hr@20": 0.020744},
            {"method_variant": "task4_highdetail_trainhard_shuffle_weight", "best_ndcg@20": 0.123547, "best_hr@20": 0.020791},
            {"method_variant": "task4_highdetail_pairmargin", "best_ndcg@20": 0.123610, "best_hr@20": 0.020781},
            {"method_variant": "task4_highdetail_pairmargin_shuffle", "best_ndcg@20": 0.123244, "best_hr@20": 0.020895},
        ]
    )

    comparison = build_pair_comparison(best, min_ndcg_delta=0.0005)
    m7a = comparison[comparison["comparison"].eq("M7a_minus_M7s")].iloc[0]

    assert math.isclose(m7a["delta_ndcg@20"], 0.00045, abs_tol=1e-8)
    assert bool(m7a["pass_ndcg_gate"]) is False
    assert bool(m7a["pass_hr_gate"]) is False
    assert bool(m7a["pass_seed43_gate"]) is False


def test_route_decision_does_not_open_multiseed_when_all_pairs_fail_gate() -> None:
    comparison = pd.DataFrame(
        [
            {"comparison": "M7a_minus_M7s", "delta_ndcg@20": 0.00045, "delta_hr@20": -0.00004, "pass_seed43_gate": False},
            {"comparison": "M7b_minus_M7ps", "delta_ndcg@20": 0.00036, "delta_hr@20": -0.00011, "pass_seed43_gate": False},
        ]
    )

    decision = build_route_decision(comparison)

    assert decision["route"] == "m7_seed43_gate_failed_but_narrowing_helped"
    assert decision["run_multi_seed_now"] is False
    assert decision["next_action"] == "revise_m7a_weight_carrier_strength_seed43_only"
