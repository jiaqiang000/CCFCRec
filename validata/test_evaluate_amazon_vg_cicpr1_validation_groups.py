from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluate_amazon_vg_cicpr1_validation_groups import (
    BASELINE_LABEL,
    _item_metrics,
    build_group_summary,
    load_validation_profile,
)


def test_item_metrics_match_protocol_normalization() -> None:
    hr20, ndcg20 = _item_metrics([2, 9, 3] + list(range(20, 37)), [2, 3])
    assert hr20 == 0.1
    assert 0.0 < ndcg20 <= 1.0


def test_load_validation_profile_builds_fixed_complementary_groups(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.csv"
    pd.DataFrame(
        {
            "raw_asin": ["a", "b", "c", "train"],
            "split": ["validate", "validate", "validate", "train"],
            "cicp_score": [0.1, 0.5, 0.9, 0.2],
        }
    ).to_csv(profile_path, index=False)
    result = load_validation_profile(profile_path, ["a", "b", "c"])
    assert result["cicp_group"].tolist() == ["low", "mid", "high"]
    assert len(result) == 3


def test_load_validation_profile_rejects_evaluation_columns(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.csv"
    pd.DataFrame(
        {
            "raw_asin": ["a"],
            "split": ["validate"],
            "cicp_score": [0.1],
            "ndcg@20": [0.2],
        }
    ).to_csv(profile_path, index=False)
    with pytest.raises(ValueError, match="forbidden evaluation columns"):
        load_validation_profile(profile_path, ["a"])


def test_group_summary_uses_paired_baseline_and_complete_groups() -> None:
    rows = []
    groups = ["low", "mid", "high"]
    for index, group in enumerate(groups):
        rows.append(
            {
                "raw_asin": str(index),
                "method_label": BASELINE_LABEL,
                "method_variant": "baseline",
                "cicp_group": group,
                "ndcg@20": 0.1 + index * 0.1,
                "hr@20": 0.01 + index * 0.01,
            }
        )
        rows.append(
            {
                "raw_asin": str(index),
                "method_label": "method",
                "method_variant": "cicpr1_e4_residual",
                "cicp_group": group,
                "ndcg@20": 0.11 + index * 0.1,
                "hr@20": 0.011 + index * 0.01,
            }
        )
    summary = build_group_summary(pd.DataFrame(rows))
    method = summary[summary["method_label"].eq("method")].set_index("cicp_group")
    assert method.loc["overall", "item_count"] == 3
    assert method.loc[["low", "mid", "high"], "item_count"].sum() == 3
    assert np.isclose(method.loc["overall", "absolute_delta_ndcg@20"], 0.01)
