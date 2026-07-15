from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from diagnose_amazon_vg_cicp_r1_r2_mechanism_sensitivity import (
    build_signal_dose_audit,
    feature_tensor,
    score_modes,
)


def test_score_modes_are_deterministic_and_complementary() -> None:
    scores = torch.tensor([0.1, 0.3, 0.7, 0.9])
    first = score_modes(scores)
    second = score_modes(scores)
    assert set(first) == {"true", "neutral", "shuffle", "invert", "zero", "one"}
    assert torch.equal(first["shuffle"], second["shuffle"])
    assert torch.allclose(first["invert"], 1.0 - scores)
    assert torch.allclose(first["neutral"], torch.full_like(scores, 0.5))


def test_feature_tensor_retains_one_independent_score() -> None:
    scores = torch.tensor([0.0, 0.25, 0.5, 1.0])
    features = feature_tensor(scores)
    assert features.shape == (4, 3)
    assert torch.allclose(features[:, 0], scores)
    assert torch.allclose(features[:, 1], 1.0 - scores)
    assert torch.allclose(features[:, 2], 4.0 * scores * (1.0 - scores))


def test_signal_dose_audit_exposes_uniform_rank_score_doses(tmp_path) -> None:
    profile = tmp_path / "profile.csv"
    pd.DataFrame(
        {
            "split": ["train"] * 4 + ["validate"] * 4,
            "cicp_score": [0.0, 0.25, 0.75, 1.0] * 2,
        }
    ).to_csv(profile, index=False)
    audit = build_signal_dose_audit(profile)
    train = audit[audit["split"].eq("train")].set_index("quantity")
    assert np.isclose(train.loc["CICP-R2-E2-CID category_gate", "mean"], 1.0)
    assert np.isclose(
        train.loc["CICP-R2-E6-RCD whole_category_drop_probability", "mean"], 0.25
    )
