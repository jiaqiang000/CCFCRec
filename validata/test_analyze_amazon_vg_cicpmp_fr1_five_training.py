from __future__ import annotations

import pandas as pd

from analyze_amazon_vg_cicpmp_fr1_five_training import (
    HISTORICAL_CICPR1_E1_NDCG20,
    build_decision,
    summarize_mechanisms,
)


def _curve_summary(e3_ndcg: float) -> pd.DataFrame:
    rows = []
    for method, ndcg, hr, late in (
        ("cicpmp_fr1_scalar_residual_reference", 0.1238, 0.0208, 0.1),
        ("cicpmp_fr1_modality_film", 0.1245, 0.0205, 0.1),
        ("cicpmp_fr1_content_expert_routing", e3_ndcg, 0.0210, 2.0),
        ("cicpmp_fr1_cross_modal_attention", 0.12699, 0.0212, 2.7),
        ("cicpmp_fr1_modality_film_shuffle", 0.1208, 0.0201, -2.0),
    ):
        rows.append(
            {
                "method_variant": method,
                "method_label": method,
                "best_epoch": 87,
                "best_ndcg@20": ndcg,
                "relative_pct_ndcg@20_vs_baseline_best": 2.0,
                "relative_pct_hr@20_vs_baseline_best": (hr / 0.0206209890524726 - 1) * 100,
                "late30_matched_positive_epochs": 30 if late > 0 else 0,
                "late30_matched_relative_pct_ndcg@20": late,
                "relative_pct_vs_historical_cicpr1_e1": (ndcg / HISTORICAL_CICPR1_E1_NDCG20 - 1) * 100,
                "passed_historical_anchor": ndcg > HISTORICAL_CICPR1_E1_NDCG20,
                "passed_three_pct": ndcg >= 0.1238145211709585 * 1.03,
            }
        )
    return pd.DataFrame(rows)


def _mechanism_activity() -> pd.DataFrame:
    rows = []
    for method in (
        "cicpmp_fr1_scalar_residual_reference",
        "cicpmp_fr1_modality_film",
        "cicpmp_fr1_content_expert_routing",
        "cicpmp_fr1_cross_modal_attention",
        "cicpmp_fr1_modality_film_shuffle",
    ):
        rows.append(
            {
                "method_variant": method,
                "method_label": method,
                "epoch": 87,
                "is_best_epoch": True,
                "actual_vs_permuted_embedding_relative_l2_mean": (
                    1e-7 if method == "cicpmp_fr1_cross_modal_attention" else 0.1
                ),
                "gate_abs_mean": 1.0 if "cross_modal" in method else 0.5,
            }
        )
    return pd.DataFrame(rows)


def _groups() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method_variant": method,
                "group_dimension": "overall",
                "bootstrap_relative_pct_ci95_low": -0.1,
                "bootstrap_relative_pct_ci95_high": 4.0,
            }
            for method in (
                "cicpmp_fr1_scalar_residual_reference",
                "cicpmp_fr1_modality_film",
                "cicpmp_fr1_content_expert_routing",
                "cicpmp_fr1_cross_modal_attention",
                "cicpmp_fr1_modality_film_shuffle",
            )
        ]
    )


def _control() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "comparison_scope": "independent_best_checkpoints_item_paired",
                "absolute_delta_ndcg@20": 0.003,
                "relative_pct_vs_shuffle": 2.5,
                "bootstrap_relative_pct_ci95_low": 0.5,
                "bootstrap_relative_pct_ci95_high": 4.5,
            }
        ]
    )


def test_summarize_mechanisms_marks_saturated_constant_e4_inactive() -> None:
    summary = summarize_mechanisms(_mechanism_activity()).set_index("method_variant")
    assert not bool(summary.loc["cicpmp_fr1_cross_modal_attention", "functional_signal_active"])
    assert bool(summary.loc["cicpmp_fr1_cross_modal_attention", "gate_saturated"])
    assert bool(summary.loc["cicpmp_fr1_content_expert_routing", "functional_signal_active"])


def test_decision_uses_active_e3_not_inactive_performance_winner() -> None:
    mechanism = summarize_mechanisms(_mechanism_activity())
    decision = build_decision(
        _curve_summary(HISTORICAL_CICPR1_E1_NDCG20 + 0.0001),
        mechanism,
        _groups(),
        _control(),
    )
    assert decision["performance_winner_method"] == "cicpmp_fr1_cross_modal_attention"
    assert not decision["performance_winner_functional_mp_active"]
    assert decision["active_semantic_winner_method"] == "cicpmp_fr1_content_expert_routing"
    assert decision["route"] == "one_bounded_performance_breakthrough_design"
    assert not decision["multi_seed_authorized"]


def test_decision_stops_when_all_active_mp_methods_miss_history() -> None:
    decision = build_decision(
        _curve_summary(HISTORICAL_CICPR1_E1_NDCG20 - 0.0001),
        summarize_mechanisms(_mechanism_activity()),
        _groups(),
        _control(),
    )
    assert decision["route"] == "stop_cicpmp_route"
