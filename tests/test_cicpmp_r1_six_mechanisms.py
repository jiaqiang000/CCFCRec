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

import cicp_mp_features as feature_module
import model as model_module


CCFCRec = model_module.CCFCRec
build_feature_tensor = feature_module.build_cicp_mp_feature_tensor
common_hash = model_module.ccfcrec_common_parameter_sha256

CICPMP_VARIANTS = [
    "cicpmp_r1_reliable_residual",
    "cicpmp_r1_direction_alignment",
    "cicpmp_r1_attention_entropy",
    "cicpmp_r1_reliable_expert",
    "cicpmp_r1_counterfactual_calibration",
    "cicpmp_r1_direction_hard_negative",
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
        cicp_profile_path="",
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
        cicp_mp_profile_path="profile.csv",
        cicpmp_hidden_dim=6,
        cicpmp_residual_max_ratio=0.15,
        cicpmp_reliability_scale=50.0,
        cicpmp_direction_weight=0.05,
        cicpmp_entropy_weight=0.02,
        cicpmp_expert_strength=0.20,
        cicpmp_counterfactual_weight=0.05,
        cicpmp_hard_negative_strength=0.50,
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


def make_features(item_count=5):
    generator = torch.Generator().manual_seed(43)
    features = torch.zeros((item_count, feature_module.CICP_MP_FEATURE_WIDTH))
    features[:, 0] = torch.linspace(-0.02, 0.03, item_count)
    features[:, 1] = torch.linspace(-0.01, 0.04, item_count)
    features[:, 2] = torch.linspace(0.00, 0.05, item_count)
    features[:, 3] = torch.linspace(0.45, 0.80, item_count)
    features[:, 4] = torch.linspace(0.35, 0.75, item_count)
    features[:, 5] = torch.linspace(0.001, 0.009, item_count)
    features[:, 6] = torch.linspace(0.003, 0.015, item_count)
    features[:, 7:] = torch.randn((item_count, 16), generator=generator) * 0.02
    return features


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
    return attributes, images, make_features(3)


def profile_from_features(item_count=5):
    features = make_features(item_count)
    data = {
        "raw_asin": [f"item_{index}" for index in range(item_count)],
        "split": ["train"] * item_count,
    }
    for index, name in enumerate(feature_module.CICP_MP_FEATURE_NAMES):
        data[name] = features[:, index].tolist()
    return pd.DataFrame(data)


def test_profile_loader_requires_exact_retained_components_and_rejects_outcomes():
    profile = profile_from_features(2)
    tensor = build_feature_tensor(profile, {"item_1": 0, "item_0": 1}, item_number=2)
    assert tensor.shape == (2, 23)
    torch.testing.assert_close(tensor[0], make_features(2)[1])

    with pytest.raises(ValueError, match="forbidden evaluation-result columns"):
        build_feature_tensor(
            profile.assign(**{"baseline_ndcg@20": 0.0}),
            {"item_0": 0, "item_1": 1},
            item_number=2,
        )
    with pytest.raises(ValueError, match="missing required columns"):
        build_feature_tensor(
            profile.drop(columns=["mp_direction16_15"]),
            {"item_0": 0, "item_1": 1},
            item_number=2,
        )


def test_all_six_variants_validate_and_require_23d_features():
    attributes, images, features = make_batch()
    for variant in CICPMP_VARIANTS:
        args = make_args(variant)
        model_module.validate_method_args(args)
        model = CCFCRec(args)
        with pytest.raises(ValueError, match="cicp_mp_features are required"):
            model(attributes, images, 3)
        output = model(
            attributes,
            images,
            3,
            cicp_mp_features=features,
        )
        assert output.shape == (3, 4)
        assert torch.isfinite(output).all()


def test_common_parameters_match_baseline_and_all_six_variants_elementwise():
    models = []
    post_init_random = []
    for variant in ["baseline", *CICPMP_VARIANTS]:
        torch.manual_seed(43)
        model = CCFCRec(make_args(variant))
        models.append(model)
        post_init_random.append(torch.rand(8))

    hashes = [common_hash(model) for model in models]
    assert len(set(hashes)) == 1
    for name in model_module.CCFCREC_COMMON_PARAMETER_NAMES:
        reference = dict(models[0].named_parameters())[name]
        for model in models[1:]:
            torch.testing.assert_close(reference, dict(model.named_parameters())[name])
    for sample in post_init_random[1:]:
        torch.testing.assert_close(post_init_random[0], sample)


def test_only_e1_owns_e4_style_residual_and_expert_is_separate():
    attributes, images, features = make_batch()
    for variant in CICPMP_VARIANTS:
        model = CCFCRec(make_args(variant))
        model(attributes, images, 3, cicp_mp_features=features)
        owns_residual = hasattr(model, "cicpmp_residual_projection") or hasattr(
            model,
            "cicpmp_residual_to_hidden",
        )
        assert owns_residual is (variant == "cicpmp_r1_reliable_residual")
        assert (model._last_cicpmp_residual is not None) is (
            variant == "cicpmp_r1_reliable_residual"
        )
        assert hasattr(model, "cicpmp_category_expert") is (
            variant == "cicpmp_r1_reliable_expert"
        )


def test_objective_and_sampling_variants_do_not_read_profile_in_inference_forward():
    attributes, images, features = make_batch()
    flipped = torch.flip(features, dims=[0])
    for variant in [
        "cicpmp_r1_direction_alignment",
        "cicpmp_r1_attention_entropy",
        "cicpmp_r1_counterfactual_calibration",
        "cicpmp_r1_direction_hard_negative",
    ]:
        model = CCFCRec(make_args(variant)).eval()
        first = model(attributes, images, 3, cicp_mp_features=features)
        second = model(attributes, images, 3, cicp_mp_features=flipped)
        torch.testing.assert_close(first, second)


def test_direction_entropy_counterfactual_and_hard_negative_mechanisms_backpropagate():
    attributes, images, features = make_batch()

    direction_args = make_args("cicpmp_r1_direction_alignment")
    direction_model = CCFCRec(direction_args)
    output = direction_model(attributes, images, 3, cicp_mp_features=features)
    direction_loss = model_module.build_cicpmp_direction_alignment_loss(
        direction_model.predict_cicpmp_direction(output),
        features,
        direction_args,
    )
    direction_loss.backward()
    assert direction_model.cicpmp_direction_head.weight.grad.abs().sum().item() > 0

    entropy_args = make_args("cicpmp_r1_attention_entropy")
    entropy_model = CCFCRec(entropy_args)
    entropy_model(attributes, images, 3, cicp_mp_features=features)
    entropy_loss = model_module.build_cicpmp_attention_entropy_loss(
        entropy_model._last_attr_attention_weight,
        attributes,
        features,
        entropy_args,
    )
    entropy_loss.backward()
    assert entropy_model.attr_W1.grad.abs().sum().item() > 0

    real = torch.randn((3, 4), requires_grad=True)
    shuffled = torch.randn((3, 4), requires_grad=True)
    teacher = torch.randn((3, 4), requires_grad=True)
    cf_loss = model_module.build_cicpmp_counterfactual_loss(
        real,
        shuffled,
        teacher,
        features,
        make_args("cicpmp_r1_counterfactual_calibration"),
    )
    cf_loss.backward()
    assert real.grad.abs().sum().item() > 0
    assert shuffled.grad.abs().sum().item() > 0
    assert teacher.grad is None

    negative_features = make_features(3)[:, None, None, :].expand(-1, 2, 4, -1)
    weights = model_module.build_cicpmp_hard_negative_weights(
        features,
        negative_features,
        make_args("cicpmp_r1_direction_hard_negative"),
    )
    assert weights.shape == (3, 2, 4)
    torch.testing.assert_close(weights.mean(dim=-1), torch.ones((3, 2)))
    assert torch.all(weights > 0.0)


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
    profile_path = tmp_path / "cicp_mp_profile.csv"
    profile_from_features().to_csv(profile_path, index=False)
    loader = torch.utils.data.DataLoader(
        _SyntheticTrainingDataset(),
        batch_size=3,
        shuffle=False,
    )

    for variant in CICPMP_VARIANTS:
        args = make_args(variant)
        args.cicp_mp_profile_path = str(profile_path)
        model = CCFCRec(args)
        initial_common_hash = common_hash(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
        output_dir = tmp_path / variant
        output_dir.mkdir()
        model_module.train(model, loader, optimizer, object(), args, str(output_dir))
        config = pd.read_json(output_dir / "run_config.json", typ="series")
        assert config["cicp_mp_feature_input_width"] == 23
        assert config["ccfcrec_common_parameter_sha256"] == initial_common_hash
        assert (output_dir / "result.csv").read_text(encoding="utf-8").count("\n") == 1


@pytest.mark.parametrize(
    ("variant", "field", "value"),
    [
        ("cicpmp_r1_reliable_residual", "cicpmp_residual_max_ratio", 0.0),
        ("cicpmp_r1_direction_alignment", "cicpmp_direction_weight", 0.0),
        ("cicpmp_r1_attention_entropy", "cicpmp_entropy_weight", 0.0),
        ("cicpmp_r1_reliable_expert", "cicpmp_expert_strength", 1.1),
        (
            "cicpmp_r1_counterfactual_calibration",
            "cicpmp_counterfactual_weight",
            0.0,
        ),
        (
            "cicpmp_r1_direction_hard_negative",
            "cicpmp_hard_negative_strength",
            1.1,
        ),
    ],
)
def test_invalid_hyperparameters_are_rejected(variant, field, value):
    args = make_args(variant)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        model_module.validate_method_args(args)
