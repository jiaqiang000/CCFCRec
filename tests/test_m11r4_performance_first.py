import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

os.environ["CCFCREC_DEVICE"] = "cpu"
REPO_ROOT = Path(__file__).resolve().parents[1]
AMAZON_VG_DIR = REPO_ROOT / "Amazon VG"
sys.path.insert(0, str(AMAZON_VG_DIR))

import m11_features as feature_module
import model as model_module

CCFCRec = model_module.CCFCRec
build_m11r4_continuous_focal_weights = model_module.build_m11r4_continuous_focal_weights
build_m11r4_relational_alignment_loss = model_module.build_m11r4_relational_alignment_loss
uses_m11r2_feature_fusion = model_module.uses_m11r2_feature_fusion
validate_method_args = model_module.validate_method_args
build_m11_feature_tensor = feature_module.build_m11_feature_tensor


M11R4_VARIANTS = [
    "m11r4_protected_experts",
    "m11r4_continuous_fusion",
    "m11r4_relational_alignment",
    "m11r4_continuous_focal",
]


def make_args(method_variant: str) -> SimpleNamespace:
    return SimpleNamespace(
        method_variant=method_variant,
        attr_num=8,
        attr_present_dim=4,
        implicit_dim=4,
        cat_implicit_dim=4,
        user_number=7,
        item_number=5,
        pretrain=False,
        pretrain_update=False,
        category_conf_dim=3,
        category_conf_max_count=5,
        category_gate_scale=0.5,
        weak_cat_threshold=3,
        weak_loss_alpha=0.5,
        adaptive_loss_alpha=1.0,
        adaptive_history_max_count=20,
        task4_profile_path="profile.csv",
        task4_loss_alpha=0.75,
        task4_shuffle_seed=43,
        task4_disable_q_bpr_weight=False,
        task4_disable_self_contrast_weight=False,
        task4_reweight_contrast=False,
        task4_pair_margin=0.2,
        task4_competitor_alpha=0.25,
        task4_competitor_margin=0.1,
        task4_competitor_k=20,
        task4_boundary_competitor_cache_path="",
        m11r2_focal_gamma=2.0,
        m11r2_focal_temperature=1.0,
        m11r2_curriculum_warmup_epochs=20,
        m11r2_feature_dim=3,
        m11r3_residual_max_ratio=0.15,
        m11r3_neighbor_loss_weight=0.1,
        m11r3_neighbor_temperature=0.25,
        m11r3_film_strength=0.1,
        m11r4_expert_film_strength=0.2,
        m11r4_fusion_strength=0.25,
        m11r4_relation_loss_weight=0.05,
        m11r4_focal_alpha=1.5,
        m11r4_focal_gamma=2.0,
        m11r4_focal_temperature=0.5,
        m11r4_focal_floor=0.35,
        reweight_q_bpr=False,
        reweight_self_contrast=False,
        reweight_contrast=False,
    )


def make_batch():
    attributes = torch.tensor(
        [
            [0, 1, -1, -1, -1, -1, -1, -1],
            [2, -1, -1, -1, -1, -1, -1, -1],
            [3, 4, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    images = torch.ones((3, 4096), dtype=torch.float32)
    features = torch.tensor(
        [
            [1.0, 0.80, 0.70, 0.60, 0.80, 0.90],
            [0.0, 0.55, 0.50, 0.40, 0.60, 0.50],
            [0.0, 0.30, 0.20, 0.10, 0.30, 0.20],
        ],
        dtype=torch.float32,
    )
    return attributes, images, features


def test_all_m11r4_variants_require_full_structural_features():
    for variant in M11R4_VARIANTS:
        args = make_args(variant)
        assert uses_m11r2_feature_fusion(args)
        assert model_module.resolve_m11_feature_mode(args) == "full_structural"
        validate_method_args(args)


def test_m11r4_strict_profile_loading_rejects_evaluation_results():
    profile = pd.DataFrame(
        {
            "raw_asin": ["a"],
            "m11r1_full_target_flag": [True],
            "m11_target_score": [0.8],
            "m11r1_full_target_loss_score": [0.8],
            "s_cat_v3": [0.7],
            "RSP_score": [0.2],
            "category_neighbor_mismatch_proxy_score": [0.8],
            "support_tail_proxy_score": [0.9],
            "baseline_ndcg@20": [0.0],
        }
    )
    with pytest.raises(ValueError, match="forbidden evaluation-result columns"):
        build_m11_feature_tensor(
            profile,
            {"a": 0},
            reject_evaluation_columns=True,
        )


def test_protected_experts_hard_isolate_target_and_non_target_paths():
    attributes, images, features = make_batch()
    model = CCFCRec(make_args("m11r4_protected_experts"))
    first = model(attributes, images, 3, m11_features=features).detach()

    with torch.no_grad():
        model.m11r4_non_target_scale.weight.fill_(0.1)
        model.m11r4_non_target_shift.weight.fill_(0.1)
    second = model(attributes, images, 3, m11_features=features).detach()

    torch.testing.assert_close(first[0], second[0])
    assert not torch.allclose(first[1:], second[1:])
    assert model._last_m11_residual[0].abs().sum().item() > 0.0
    assert model._last_m11_residual[1:].abs().sum().item() > 0.0


def test_continuous_fusion_receives_gradients_for_target_and_non_target_items():
    attributes, images, features = make_batch()
    model = CCFCRec(make_args("m11r4_continuous_fusion"))
    output = model(attributes, images, 3, m11_features=features)
    output.square().mean().backward()

    assert model.m11r4_attr_scale.weight.grad.abs().sum().item() > 0.0
    assert model.m11r4_image_scale.weight.grad.abs().sum().item() > 0.0


def test_relational_alignment_is_target_anchored_and_differentiable():
    _, _, features = make_batch()
    q_v_c = torch.randn((3, 4), requires_grad=True)
    loss = build_m11r4_relational_alignment_loss(q_v_c, features)
    assert loss.item() > 0.0
    loss.backward()
    assert q_v_c.grad.abs().sum().item() > 0.0

    no_target = features.clone()
    no_target[:, 0] = 0.0
    zero = build_m11r4_relational_alignment_loss(q_v_c.detach(), no_target)
    assert zero.item() == 0.0


def test_continuous_focal_weights_cover_every_item_and_use_continuous_score():
    _, _, features = make_batch()
    difficulty = torch.tensor([0.9, 0.7, 0.2])
    args = make_args("m11r4_continuous_focal")
    weights = build_m11r4_continuous_focal_weights(difficulty, features, args)

    assert torch.all(weights > 0)
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
    assert weights[0] > weights[1] > weights[2]


@pytest.mark.parametrize(
    ("variant", "field", "value"),
    [
        ("m11r4_protected_experts", "m11r4_expert_film_strength", 0.0),
        ("m11r4_continuous_fusion", "m11r4_fusion_strength", 1.1),
        ("m11r4_relational_alignment", "m11r4_relation_loss_weight", 0.0),
        ("m11r4_continuous_focal", "m11r4_focal_floor", 0.0),
    ],
)
def test_m11r4_invalid_hyperparameters_are_rejected(variant, field, value):
    args = make_args(variant)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_method_args(args)
