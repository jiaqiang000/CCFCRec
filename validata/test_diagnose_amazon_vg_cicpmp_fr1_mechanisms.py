from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from diagnose_amazon_vg_cicpmp_fr1_mechanisms import (
    embedding_relative_l2,
    selected_epoch_rows,
    tensor_summary,
)


def test_selected_epoch_rows_includes_fixed_and_best() -> None:
    result = pd.DataFrame(
        {
            "epoch": range(1, 101),
            "checkpoint_index": range(1, 101),
            "ndcg@20": [0.1] * 86 + [0.2] + [0.1] * 13,
            "hr@20": [0.02] * 100,
        }
    )
    selected = selected_epoch_rows(result)
    assert selected["epoch"].tolist() == [1, 20, 40, 74, 87, 100]
    assert selected.loc[selected["is_best_epoch"], "epoch"].tolist() == [87]


def test_tensor_summary_detects_nonzero_gate() -> None:
    summary = tensor_summary(torch.tensor([-1.0, 0.0, 1.0]))
    assert summary["mean"] == 0.0
    assert summary["abs_mean"] == pytest.approx(2.0 / 3.0)
    assert summary["nonzero_fraction"] == pytest.approx(2.0 / 3.0)


def test_embedding_relative_l2_is_zero_for_equal_embeddings() -> None:
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(embedding_relative_l2(values, values), 0.0)
