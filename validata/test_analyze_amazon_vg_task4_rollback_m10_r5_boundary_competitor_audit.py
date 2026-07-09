import numpy as np
import pandas as pd

from analyze_amazon_vg_task4_rollback_m10_r5_boundary_competitor_audit import (
    add_branch_flags,
    build_item_positive_user_sets,
    build_target_item_table,
    decide_route,
    select_boundary_from_score_row,
    summarize_boundary_audit,
)


def _profile() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "raw_asin": "a",
                "split": "train",
                "cat_count_bin": "cat_count_5_plus",
                "category_count": 5,
                "high_acat_train_safe_hard_flag": True,
                "train_safe_hard_proxy_score": 0.8,
                "s_cat_v3": 0.9,
                "high_acat_flag": True,
                "RSP_group": "RSP_low",
                "RSP_score": 0.1,
                "baseline_ndcg@20": 1.0,
            },
            {
                "raw_asin": "b",
                "split": "train",
                "cat_count_bin": "cat_count_5_plus",
                "category_count": 5,
                "high_acat_train_safe_hard_flag": False,
                "train_safe_hard_proxy_score": 0.2,
                "s_cat_v3": 0.1,
                "high_acat_flag": False,
                "RSP_group": "RSP_high",
                "RSP_score": 0.9,
                "baseline_ndcg@20": 0.0,
            },
            {
                "raw_asin": "c",
                "split": "validate",
                "cat_count_bin": "cat_count_5_plus",
                "category_count": 5,
                "high_acat_train_safe_hard_flag": True,
                "train_safe_hard_proxy_score": 0.7,
                "s_cat_v3": 0.8,
                "high_acat_flag": True,
                "RSP_group": "RSP_high",
                "RSP_score": 0.8,
                "baseline_ndcg@20": 0.0,
            },
            {
                "raw_asin": "d",
                "split": "train",
                "cat_count_bin": "cat_count_4",
                "category_count": 4,
                "high_acat_train_safe_hard_flag": True,
                "train_safe_hard_proxy_score": 0.9,
                "s_cat_v3": 0.9,
                "high_acat_flag": True,
                "RSP_group": "RSP_high",
                "RSP_score": 0.9,
                "baseline_ndcg@20": 0.0,
            },
        ]
    )


def test_add_branch_flags_uses_train_safe_masks_and_ignores_eval_columns() -> None:
    original = add_branch_flags(_profile(), shuffle_seed=43)
    changed_profile = _profile()
    changed_profile["baseline_ndcg@20"] = 999.0
    changed = add_branch_flags(changed_profile, shuffle_seed=43)

    assert original["r5_real_target_flag"].tolist() == [True, False, False, False]
    assert original["r5_rsp_control_target_flag"].tolist() == [False, True, False, False]
    assert original["r5_acat_control_target_flag"].tolist() == [True, False, False, False]
    assert original["r5_union_target_flag"].astype(bool).sum() >= 2
    assert original["r5_real_target_flag"].tolist() == changed["r5_real_target_flag"].tolist()
    assert original["r5_union_target_flag"].tolist() == changed["r5_union_target_flag"].tolist()


def test_target_item_table_keeps_only_train_mapped_union_items() -> None:
    flagged = add_branch_flags(_profile(), shuffle_seed=43)
    target_items = build_target_item_table(flagged, {"a": 0, "b": 1}, max_target_items=0)

    assert set(target_items["raw_asin"]) == {"a", "b"}
    assert target_items["serial_item_id"].tolist() == [0, 1]


def test_positive_user_sets_map_raw_ids_to_serial_ids() -> None:
    train = pd.DataFrame(
        {
            "asin": ["a", "a", "b"],
            "reviewerID": ["u1", "u2", "u3"],
            "rating": [1.0, 1.0, 1.0],
        }
    )

    positives = build_item_positive_user_sets(train, {"u1": 0, "u2": 1, "u3": 2}, {"a": 10, "b": 11})

    assert positives == {10: {0, 1}, 11: {2}}


def test_select_boundary_excludes_positive_users_and_beats_existing_neg() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    selected = select_boundary_from_score_row(
        scores=scores,
        positive_users={0},
        existing_neg_users={3},
        serial_user_to_raw={0: "u0", 1: "u1", 2: "u2", 3: "u3"},
        top_user_ids=np.asarray([0, 1, 2, 3]),
    )

    assert selected["candidate_found_flag"] is True
    assert selected["boundary_competitor_serial_user"] == 1
    assert selected["boundary_competitor_user"] == "u1"
    assert selected["leak_safe_flag"] is True
    assert selected["boundary_minus_best_existing_neg"] > 0


def test_route_reaches_training_node_only_for_full_strong_cache() -> None:
    audit = pd.DataFrame(
        [
            {
                "r5_real_target_flag": True,
                "r5_shuffle_target_flag": True,
                "r5_rsp_control_target_flag": False,
                "r5_acat_control_target_flag": True,
                "candidate_found_flag": True,
                "leak_safe_flag": True,
                "boundary_rank": 5,
                "existing_neg_user_count": 1,
                "boundary_minus_best_existing_neg": 0.4,
                "boundary_minus_mean_positive": 0.1,
            }
            for _ in range(1000)
        ]
    )
    summary = summarize_boundary_audit(audit)
    decision = decide_route(summary, audit, max_target_items=0)

    assert decision["route"] == "r5_boundary_competitor_cache_feasible_training_node"
    assert decision["training_node_reached"] is True
    assert decision["forbidden_train_columns_used"] == []


def test_route_blocks_sampled_audit_even_when_metrics_look_good() -> None:
    audit = pd.DataFrame(
        [
            {
                "r5_real_target_flag": True,
                "r5_shuffle_target_flag": False,
                "r5_rsp_control_target_flag": False,
                "r5_acat_control_target_flag": False,
                "candidate_found_flag": True,
                "leak_safe_flag": True,
                "boundary_rank": 5,
                "existing_neg_user_count": 1,
                "boundary_minus_best_existing_neg": 0.4,
                "boundary_minus_mean_positive": 0.1,
            }
            for _ in range(1000)
        ]
    )
    summary = summarize_boundary_audit(audit)
    decision = decide_route(summary, audit, max_target_items=1000)

    assert decision["training_node_reached"] is False
    assert decision["gates"]["not_sampled_full_cache"] is False
