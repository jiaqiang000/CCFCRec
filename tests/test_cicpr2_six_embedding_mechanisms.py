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

import model as model_module


CCFCRec = model_module.CCFCRec
build_ordinal_loss = model_module.build_cicpr2_ordinal_counterfactual_loss
build_score_loss = model_module.build_cicpr2_score_distillation_loss
validate_method_args = model_module.validate_method_args

CICPR2_VARIANTS = [
    "cicpr2_content_direction_residual",
    "cicpr2_category_increment_gate",
    "cicpr2_cross_modal_attention",
    "cicpr2_score_distillation",
    "cicpr2_ordinal_counterfactual",
    "cicpr2_reliability_dropout",
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
        cicpr2_residual_max_ratio=0.15,
        cicpr2_increment_strength=0.50,
        cicpr2_cross_attention_strength=0.50,
        cicpr2_cross_attention_temperature=0.25,
        cicpr2_distillation_weight=0.05,
        cicpr2_ordinal_weight=0.05,
        cicpr2_ordinal_margin=0.02,
        cicpr2_category_dropout_max=0.50,
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


def make_batch(batch_size=3):
    attributes = torch.tensor(
        [
            [1, 1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, -1, -1, -1, -1],
            [-1, 1, 1, -1, -1, -1, -1, -1],
        ][:batch_size],
        dtype=torch.long,
    )
    images = torch.randn((batch_size, 4096), generator=torch.Generator().manual_seed(43))
    score = torch.tensor([0.9, 0.5, 0.1][:batch_size], dtype=torch.float32)
    features = torch.stack((score, 1.0 - score, 4.0 * score * (1.0 - score)), dim=1)
    return attributes, images, features


def test_all_six_variants_validate_require_score_and_backpropagate():
    attributes, images, features = make_batch()
    for variant in CICPR2_VARIANTS:
        args = make_args(variant)
        validate_method_args(args)
        model = CCFCRec(args)
        with pytest.raises(ValueError, match="cicp_features are required"):
            model(attributes, images, len(attributes))
        output = model(attributes, images, len(attributes), cicp_features=features)
        loss = output.square().mean()
        if variant == "cicpr2_score_distillation":
            prediction = model.predict_cicpr2_score(output)
            loss = loss + build_score_loss(prediction, features)
        loss.backward()
        assert output.shape == (3, 4)
        assert torch.isfinite(output).all()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_only_e1_uses_post_generator_content_directed_residual():
    attributes, images, features = make_batch()
    for variant in CICPR2_VARIANTS:
        model = CCFCRec(make_args(variant))
        model(attributes, images, len(attributes), cicp_features=features)
        owns_direction_adapter = hasattr(model, "cicpr2_residual_category_direction")
        assert owns_direction_adapter is (variant == "cicpr2_content_direction_residual")
        assert (model._last_cicp_residual is not None) is (
            variant == "cicpr2_content_direction_residual"
        )


def test_e1_direction_comes_from_item_content_and_score_only_scales_it():
    attributes, images, _ = make_batch()
    model = CCFCRec(make_args("cicpr2_content_direction_residual"))
    with torch.no_grad():
        model.cicpr2_residual_category_direction.weight.copy_(torch.eye(4))
        model.cicpr2_residual_image_direction.weight.zero_()
    score = torch.full((3,), 0.8)
    features = torch.stack((score, 1.0 - score, 4.0 * score * (1.0 - score)), dim=1)
    model(attributes, images, len(attributes), cicp_features=features)
    residual = model._last_cicp_residual
    assert not torch.allclose(residual[0], residual[1])
    assert torch.all(residual.norm(dim=1) <= 0.15 * model.gen_layer1(
        torch.cat(model.encode_content_components(
            attributes, images, len(attributes), cicp_features=features
        )[1:], dim=1)
    ).norm(dim=1) + 1e-5)


def test_category_increment_and_cross_modal_attention_use_score_without_e4_adapter():
    attributes, images, _ = make_batch(batch_size=2)
    for variant in ["cicpr2_category_increment_gate", "cicpr2_cross_modal_attention"]:
        model = CCFCRec(make_args(variant)).eval()
        low = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        high = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        low_output = model(attributes, images, 2, cicp_features=low)
        high_output = model(attributes, images, 2, cicp_features=high)
        assert not torch.allclose(low_output, high_output)
        assert not hasattr(model, "cicpr2_residual_category_direction")


def test_training_only_mechanisms_leave_inference_embedding_score_invariant():
    attributes, images, features = make_batch()
    for variant in [
        "cicpr2_score_distillation",
        "cicpr2_ordinal_counterfactual",
        "cicpr2_reliability_dropout",
    ]:
        model = CCFCRec(make_args(variant)).eval()
        first = model(attributes, images, 3, cicp_features=features)
        second = model(attributes, images, 3, cicp_features=torch.flip(features, dims=[0]))
        torch.testing.assert_close(first, second)


def test_ordinal_counterfactual_loss_is_finite_and_differentiable():
    _, _, features = make_batch()
    real = torch.randn((3, 4), requires_grad=True)
    shuffled = torch.randn((3, 4), requires_grad=True)
    teacher = torch.randn((3, 4), requires_grad=True)
    loss = build_ordinal_loss(real, shuffled, teacher, features, make_args(
        "cicpr2_ordinal_counterfactual"
    ))
    loss.backward()
    assert torch.isfinite(loss)
    assert real.grad.abs().sum().item() > 0
    assert shuffled.grad.abs().sum().item() > 0
    assert teacher.grad is None


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
            (5, 4096), generator=torch.Generator().manual_seed(43)
        )

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return (
            torch.tensor(index),
            torch.tensor(index),
            torch.tensor(index + 3),
            torch.tensor([(index + 1) % 5, (index + 2) % 5]),
            torch.tensor(
                [
                    [(index + 2) % 5, (index + 3) % 5],
                    [(index + 3) % 5, (index + 4) % 5],
                ]
            ),
            torch.tensor([(index + 3) % 5, (index + 4) % 5]),
            torch.tensor(0.5),
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
    loader = torch.utils.data.DataLoader(
        _SyntheticTrainingDataset(), batch_size=3, shuffle=False
    )

    for variant in CICPR2_VARIANTS:
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
        ("cicpr2_content_direction_residual", "cicpr2_residual_max_ratio", 0.0),
        ("cicpr2_category_increment_gate", "cicpr2_increment_strength", 1.1),
        ("cicpr2_cross_modal_attention", "cicpr2_cross_attention_strength", 0.0),
        ("cicpr2_score_distillation", "cicpr2_distillation_weight", 0.0),
        ("cicpr2_ordinal_counterfactual", "cicpr2_ordinal_margin", 0.0),
        ("cicpr2_reliability_dropout", "cicpr2_category_dropout_max", 1.0),
    ],
)
def test_invalid_cicpr2_hyperparameters_are_rejected(variant, field, value):
    args = make_args(variant)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_method_args(args)
