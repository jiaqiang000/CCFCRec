from __future__ import annotations

import numpy as np

from evaluate_amazon_vg_cicpr1_validation_groups import build_model_args
from evaluate_amazon_vg_cicpr2_validation_groups import METHOD_LABELS


def test_cicpr2_labels_keep_formal_round_prefixes() -> None:
    assert len(METHOD_LABELS) == 6
    assert set(METHOD_LABELS.values()) == {
        "CICP-R2-E1-CDR",
        "CICP-R2-E2-CID",
        "CICP-R2-E3-CMA",
        "CICP-R2-E4-SD",
        "CICP-R2-E5-OCS",
        "CICP-R2-E6-RCD",
    }


def test_build_model_args_preserves_cicpr2_checkpoint_configuration() -> None:
    state_dict = {
        "attr_matrix": np.zeros((4, 8)),
        "user_embedding": np.zeros((5, 16)),
        "gen_layer1.weight": np.zeros((32, 64)),
        "item_embedding": np.zeros((6, 16)),
    }
    config = {
        "method_variant": "cicpr2_cross_modal_attention",
        "cicpr2_cross_attention_strength": 0.42,
        "cicpr2_cross_attention_temperature": 0.17,
    }
    args = build_model_args(state_dict, config)
    assert args.method_variant == "cicpr2_cross_modal_attention"
    assert args.cicpr2_cross_attention_strength == 0.42
    assert args.cicpr2_cross_attention_temperature == 0.17
    assert args.cicpr2_residual_max_ratio == 0.15
