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

import cicp_features as feature_module
import model as model_module


CCFCRec = model_module.CCFCRec
build_cicp_feature_tensor = feature_module.build_cicp_feature_tensor
build_cicpr1_alignment_loss = model_module.build_cicpr1_alignment_loss
build_cicpr1_counterfactual_margin_loss = model_module.build_cicpr1_counterfactual_margin_loss
cap_cicp_residual_norm = model_module.cap_cicp_residual_norm
validate_method_args = model_module.validate_method_args

CICPR1_VARIANTS = [
    "cicpr1_e4_residual",
    "cicpr1_modality_routing",
    "cicpr1_category_expert",
    "cicpr1_alignment_curriculum",
    "cicpr1_counterfactual_margin",
    "cicpr1_adaptive_attention",
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
        task4_profile_path="",
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
        cicp_profile_path="profile.csv",
        cicp_feature_dim=3,
        cicp_residual_max_ratio=0.15,
        cicp_modality_strength=0.25,
        cicp_expert_strength=0.20,
        cicp_alignment_weight=0.05,
        cicp_alignment_warmup_epochs=20,
        cicp_counterfactual_weight=0.05,
        cicp_counterfactual_margin=0.05,
        cicp_attention_strength=0.50,
        reweight_q_bpr=False,
        reweight_self_contrast=False,
        reweight_contrast=False,
        tau=0.1,
        lambda1=0.6,
        epoch=1,
        save_batch_time=999,
        pin_memory=False,
        seed=43,
        num_workers=0,
        batch_size=3,
        validate_batch_size=3,
        negative_sampling_mode="fast_uniform",
        negative_sampling_cache_size=32,
    )


def make_batch():
    attributes = torch.tensor(
        [
            [1, 1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, -1, -1, -1, -1],
            [-1, 1, 1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    images = torch.randn((3, 4096), generator=torch.Generator().manual_seed(43))
    score = torch.tensor([0.9, 0.5, 0.1], dtype=torch.float32)
    features = torch.stack((score, 1.0 - score, 4.0 * score * (1.0 - score)), dim=1)
    return attributes, images, features


def test_cicp_feature_profile_builds_frozen_basis_and_rejects_outcomes():
    profile = pd.DataFrame(
        {
            "raw_asin": ["a", "b"],
            "cicp_score": [0.8, 0.25],
        }
    )
    tensor = build_cicp_feature_tensor(profile, {"b": 0, "a": 1}, item_number=2)
    torch.testing.assert_close(tensor[0], torch.tensor([0.25, 0.75, 0.75]))
    torch.testing.assert_close(tensor[1], torch.tensor([0.8, 0.2, 0.64]))

    unsafe = profile.assign(**{"baseline_ndcg@20": 0.0})
    with pytest.raises(ValueError, match="forbidden evaluation-result columns"):
        build_cicp_feature_tensor(unsafe, {"a": 0, "b": 1}, item_number=2)


def test_all_six_variants_require_cicp_features_and_validate():
    attributes, images, features = make_batch()
    for variant in CICPR1_VARIANTS:
        args = make_args(variant)
        validate_method_args(args)
        model = CCFCRec(args)
        with pytest.raises(ValueError, match="cicp_features are required"):
            model(attributes, images, len(attributes))
        output = model(attributes, images, len(attributes), cicp_features=features)
        assert output.shape == (3, 4)
        assert torch.isfinite(output).all()


def test_only_e1_owns_and_uses_e4_style_residual_adapter():
    attributes, images, features = make_batch()
    for variant in CICPR1_VARIANTS:
        model = CCFCRec(make_args(variant))
        model(attributes, images, len(attributes), cicp_features=features)
        owns_adapter = hasattr(model, "cicp_feature_projection") or hasattr(
            model, "cicp_feature_to_hidden"
        )
        assert owns_adapter is (variant == "cicpr1_e4_residual")
        assert (model._last_cicp_residual is not None) is (
            variant == "cicpr1_e4_residual"
        )


def test_e1_residual_is_norm_capped():
    residual = torch.full((3, 4), 10.0)
    hidden = torch.randn((3, 4), generator=torch.Generator().manual_seed(43))
    capped = cap_cicp_residual_norm(residual, hidden, 0.15)
    assert torch.all(capped.norm(dim=1) <= 0.15 * hidden.norm(dim=1) + 1e-6)


def test_modality_router_and_category_expert_receive_gradients_without_residual_adapter():
    attributes, images, features = make_batch()
    for variant, parameter_name in [
        ("cicpr1_modality_routing", "cicp_modality_gate"),
        ("cicpr1_category_expert", "cicp_category_expert"),
    ]:
        model = CCFCRec(make_args(variant))
        output = model(attributes, images, len(attributes), cicp_features=features)
        output.square().mean().backward()
        parameter = getattr(model, parameter_name).weight
        assert parameter.grad is not None
        assert parameter.grad.abs().sum().item() > 0
        assert not hasattr(model, "cicp_feature_projection")


def test_alignment_and_counterfactual_objectives_are_differentiable_and_score_aware():
    _, _, features = make_batch()
    q_v_c = torch.randn((3, 4), requires_grad=True)
    item_embedding = torch.randn((3, 4), requires_grad=True)
    args = make_args("cicpr1_alignment_curriculum")
    early = build_cicpr1_alignment_loss(q_v_c, item_embedding, features, 0, args)
    late = build_cicpr1_alignment_loss(q_v_c, item_embedding, features, 19, args)
    assert late.item() > early.item() > 0
    late.backward(retain_graph=True)
    assert q_v_c.grad.abs().sum().item() > 0
    assert item_embedding.grad is None

    real_margin = torch.tensor([0.4, 0.2, 0.1], requires_grad=True)
    shuffled_margin = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    cf_args = make_args("cicpr1_counterfactual_margin")
    loss = build_cicpr1_counterfactual_margin_loss(
        real_margin,
        shuffled_margin,
        features,
        cf_args,
    )
    assert loss.item() > 0
    loss.backward()
    assert real_margin.grad.abs().sum().item() > 0
    assert shuffled_margin.grad.abs().sum().item() > 0


def test_objective_only_variants_do_not_use_cicp_in_inference_forward():
    attributes, images, features = make_batch()
    reversed_features = torch.flip(features, dims=[0])
    for variant in ["cicpr1_alignment_curriculum", "cicpr1_counterfactual_margin"]:
        model = CCFCRec(make_args(variant))
        first = model(attributes, images, len(attributes), cicp_features=features)
        second = model(attributes, images, len(attributes), cicp_features=reversed_features)
        torch.testing.assert_close(first, second)
        assert not hasattr(model, "cicp_feature_projection")


def test_adaptive_attention_changes_forward_without_residual_module():
    attributes, images, features = make_batch()
    model = CCFCRec(make_args("cicpr1_adaptive_attention"))
    first = model(attributes, images, len(attributes), cicp_features=features)
    second = model(attributes, images, len(attributes), cicp_features=torch.flip(features, dims=[0]))
    assert not torch.allclose(first, second)
    assert not hasattr(model, "cicp_feature_projection")


class _SyntheticTrainingDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.item_serialize_dict = {f"item_{index}": index for index in range(5)}
        self.user_serialize_dict = {f"user_{index}": index for index in range(7)}
        self.item_number = 5
        self.item_category_tensor = torch.tensor(
            [
                [1, 1, -1, -1, -1, -1, -1, -1],
                [1, -1, 1, -1, -1, -1, -1, -1],
                [-1, 1, 1, -1, -1, -1, -1, -1],
                [1, -1, -1, 1, -1, -1, -1, -1],
                [-1, 1, -1, 1, -1, -1, -1, -1],
            ],
            dtype=torch.long,
        )
        self.item_image_feature_tensor = torch.randn(
            (5, 4096),
            generator=torch.Generator().manual_seed(43),
        )

    def __len__(self):
        return 3

    def __getitem__(self, index):
        positive_items = torch.tensor([(index + 1) % 5, (index + 2) % 5])
        negative_items = torch.tensor(
            [
                [(index + 2) % 5, (index + 3) % 5],
                [(index + 3) % 5, (index + 4) % 5],
            ]
        )
        self_neg_items = torch.tensor([(index + 3) % 5, (index + 4) % 5])
        return (
            torch.tensor(index, dtype=torch.long),
            torch.tensor(index, dtype=torch.long),
            torch.tensor(index + 3, dtype=torch.long),
            positive_items,
            negative_items,
            self_neg_items,
            torch.tensor(0.5, dtype=torch.float32),
        )


def test_each_variant_completes_one_integrated_optimizer_batch(tmp_path):
    profile_path = tmp_path / "cicp_profile.csv"
    pd.DataFrame(
        {
            "raw_asin": [f"item_{index}" for index in range(5)],
            "split": ["train"] * 5,
            "cicp_score": [0.9, 0.7, 0.5, 0.3, 0.1],
        }
    ).to_csv(profile_path, index=False)
    loader = torch.utils.data.DataLoader(_SyntheticTrainingDataset(), batch_size=3, shuffle=False)

    for variant in CICPR1_VARIANTS:
        args = make_args(variant)
        args.cicp_profile_path = str(profile_path)
        model = CCFCRec(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
        output_dir = tmp_path / variant
        output_dir.mkdir()
        model_module.train(model, loader, optimizer, object(), args, str(output_dir))
        assert (output_dir / "run_config.json").is_file()
        assert (output_dir / "result.csv").read_text(encoding="utf-8").count("\n") == 1


@pytest.mark.parametrize(
    ("variant", "field", "value"),
    [
        ("cicpr1_e4_residual", "cicp_residual_max_ratio", 0.0),
        ("cicpr1_modality_routing", "cicp_modality_strength", 1.1),
        ("cicpr1_category_expert", "cicp_expert_strength", 0.0),
        ("cicpr1_alignment_curriculum", "cicp_alignment_weight", 0.0),
        ("cicpr1_counterfactual_margin", "cicp_counterfactual_margin", 0.0),
        ("cicpr1_adaptive_attention", "cicp_attention_strength", 0.0),
    ],
)
def test_invalid_cicpr1_hyperparameters_are_rejected(variant, field, value):
    args = make_args(variant)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_method_args(args)
