import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


os.environ["CCFCREC_DEVICE"] = "cpu"
REPO_ROOT = Path(__file__).resolve().parents[1]
AMAZON_VG_DIR = REPO_ROOT / "Amazon VG"
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(AMAZON_VG_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import cicp_mp_features as feature_module
import model as model_module
from audit_cicpmp_fr1_protocol import VARIANTS, make_args, run_audit
from prepare_cicpmp_fr1_profiles import prepare
from test_cicpmp_r1_six_mechanisms import (
    _SyntheticTrainingDataset,
    make_args as make_training_args,
    profile_from_features,
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
    scalar = torch.tensor(
        [[0.2, 0.8, 0.64], [0.5, 0.5, 1.0], [0.8, 0.2, 0.64]],
        dtype=torch.float32,
    )
    mp = torch.randn((3, 23), generator=torch.Generator().manual_seed(44))
    return attributes, images, scalar, mp


def make_source_profiles():
    rows = 6
    split = ["train", "train", "train", "validate", "validate", "validate"]
    scalar = pd.DataFrame(
        {
            "raw_asin": [f"item_{index}" for index in range(rows)],
            "split": split,
            "cicp_score": np.linspace(0.1, 0.9, rows),
        }
    )
    mp_data = {
        "raw_asin": scalar["raw_asin"],
        "split": split,
    }
    for index, name in enumerate(feature_module.CICP_MP_FEATURE_NAMES):
        mp_data[name] = np.arange(rows, dtype=float) * (index + 1) + index * 0.1
    return scalar, pd.DataFrame(mp_data)


def test_preflight_audit_proves_initialization_and_residual_ownership():
    audit = run_audit(seed=43)
    assert audit["common_hash_unique_count"] == 1
    assert audit["post_initialization_rng_hash_unique_count"] == 1
    assert audit["e4_style_hidden_residual_branches"] == [
        "cicpmp_fr1_scalar_residual_reference"
    ]
    assert audit["modality_film_control_parameter_state_equal"] is True
    assert audit["per_item_layer_norm_in_fr1_modules"] is False
    assert audit["activation_schedule"] == "none"
    assert set(audit["initial_effect_max_abs_by_variant"].values()) == {0.0}


def test_variants_require_their_declared_profile_type():
    for variant in VARIANTS:
        args = make_args(variant, seed=43)
        model_module.validate_method_args(args)
        if variant == "cicpmp_fr1_scalar_residual_reference":
            args.cicp_profile_path = ""
            with pytest.raises(ValueError, match="cicp_profile_path is required"):
                model_module.validate_method_args(args)
        else:
            args.cicp_mp_profile_path = ""
            with pytest.raises(ValueError, match="cicp_mp_profile_path is required"):
                model_module.validate_method_args(args)


def test_optimizer_keeps_base_decay_and_removes_method_decay():
    for variant in VARIANTS:
        args = make_args(variant, seed=43)
        model = model_module.CCFCRec(args)
        optimizer = model_module.build_optimizer(model, args)
        groups = {group.get("group_name"): group for group in optimizer.param_groups}
        assert groups["ccfcrec_base"]["weight_decay"] == 0.1
        assert groups["cicpmp_fr1_method"]["weight_decay"] == 0.0
        method_parameter_ids = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if name.startswith("cicpmp_fr1_")
        }
        assert {id(parameter) for parameter in groups["cicpmp_fr1_method"]["params"]} == method_parameter_ids


def test_zero_effect_heads_receive_ranking_gradients_and_begin_to_move():
    attributes, images, scalar, mp = make_batch()
    for variant in VARIANTS:
        args = make_args(variant, seed=43)
        torch.manual_seed(43)
        model = model_module.CCFCRec(args)
        optimizer = model_module.build_optimizer(model, args)
        kwargs = (
            {"cicp_features": scalar}
            if variant == "cicpmp_fr1_scalar_residual_reference"
            else {"cicp_mp_features": mp}
        )
        initial = model(attributes, images, 3, **kwargs).detach().clone()
        loss = model(attributes, images, 3, **kwargs).pow(2).sum()
        loss.backward()
        nonzero_method_gradients = [
            parameter.grad.detach().abs().sum().item()
            for name, parameter in model.named_parameters()
            if name.startswith("cicpmp_fr1_") and parameter.grad is not None
        ]
        assert nonzero_method_gradients
        assert max(nonzero_method_gradients) > 0.0
        optimizer.step()
        moved = model(attributes, images, 3, **kwargs).detach()
        assert not torch.equal(initial, moved)


def test_standardized_profile_marker_allows_signed_confidence_components():
    scalar, mp = make_source_profiles()
    standardized = mp.copy()
    standardized.loc[:, feature_module.CICP_MP_FEATURE_NAMES] = -1.0
    standardized["cicpmp_standardization"] = "train_feature_wise_zscore_v1"
    frame = feature_module.build_cicp_mp_feature_frame(standardized)
    assert frame.shape == (6, 23)

    with pytest.raises(ValueError, match="must be non-negative"):
        feature_module.build_cicp_mp_feature_frame(
            standardized.drop(columns=["cicpmp_standardization"])
        )


def test_profile_preparation_fits_train_only_and_shuffles_whole_rows(tmp_path):
    scalar, mp = make_source_profiles()
    scalar_source = tmp_path / "scalar_source.csv"
    mp_source = tmp_path / "mp_source.csv"
    scalar.to_csv(scalar_source, index=False)
    mp.to_csv(mp_source, index=False)
    args = argparse.Namespace(
        scalar_source=scalar_source,
        mp_source=mp_source,
        scalar_output=tmp_path / "scalar.csv",
        mp_output=tmp_path / "mp.csv",
        shuffle_output=tmp_path / "shuffle.csv",
        audit_output=tmp_path / "audit.json",
        seed=43,
        dry_run=True,
    )
    audit = prepare(args)
    assert audit["standardization_fit_split"] == "train"
    assert audit["shuffle_unit"] == "whole_23d_row_within_split"
    assert audit["shuffle_fixed_points"] == {"train": 0, "validate": 0}
    assert audit["test_rows_passed_to_training"] == 0
    assert audit["train_post_standardization_max_abs_mean"] < 1e-6
    assert audit["train_post_standardization_max_abs_std_error"] < 1e-6

    real = pd.read_csv(args.mp_output)
    shuffled = pd.read_csv(args.shuffle_output)
    feature_names = list(feature_module.CICP_MP_FEATURE_NAMES)
    for split in ("train", "validate"):
        real_rows = {
            tuple(row)
            for row in real.loc[real["split"].eq(split), feature_names].to_numpy()
        }
        shuffled_rows = {
            tuple(row)
            for row in shuffled.loc[shuffled["split"].eq(split), feature_names].to_numpy()
        }
        assert shuffled_rows == real_rows
        paired = real.loc[real["split"].eq(split), feature_names].to_numpy()
        paired_shuffle = shuffled.loc[
            shuffled["split"].eq(split), feature_names
        ].to_numpy()
        assert not np.equal(paired, paired_shuffle).all(axis=1).any()

    saved_audit = json.loads(args.audit_output.read_text(encoding="utf-8"))
    assert saved_audit["semantic_blocks"]["direction"] == list(
        feature_module.CICP_MP_FEATURE_NAMES[7:]
    )


def test_all_five_variants_complete_an_integrated_optimizer_batch(tmp_path):
    mp_profile = profile_from_features()
    mp_profile["cicpmp_standardization"] = "train_feature_wise_zscore_v1"
    mp_profile_path = tmp_path / "mp_profile.csv"
    mp_profile.to_csv(mp_profile_path, index=False)
    scalar_profile_path = tmp_path / "scalar_profile.csv"
    pd.DataFrame(
        {
            "raw_asin": [f"item_{index}" for index in range(5)],
            "split": ["train"] * 5,
            "cicp_score": np.linspace(0.1, 0.9, 5),
        }
    ).to_csv(scalar_profile_path, index=False)
    loader = torch.utils.data.DataLoader(
        _SyntheticTrainingDataset(),
        batch_size=3,
        shuffle=False,
    )

    for variant in VARIANTS:
        args = make_training_args("cicpmp_r1_reliable_residual")
        args.method_variant = variant
        args.cicp_profile_path = (
            str(scalar_profile_path)
            if variant == "cicpmp_fr1_scalar_residual_reference"
            else ""
        )
        args.cicp_mp_profile_path = (
            str(mp_profile_path)
            if variant != "cicpmp_fr1_scalar_residual_reference"
            else ""
        )
        args.cicpmp_fr1_block_dim = 6
        args.cicpmp_fr1_residual_max_ratio = 0.15
        args.cicpmp_fr1_method_weight_decay = 0.0
        args.weight_decay = 0.1
        args.learning_rate = 0.0001
        model = model_module.CCFCRec(args)
        optimizer = model_module.build_optimizer(model, args)
        output_dir = tmp_path / variant
        output_dir.mkdir()
        model_module.train(model, loader, optimizer, object(), args, str(output_dir))
        config = json.loads((output_dir / "run_config.json").read_text())
        assert config["cicpmp_fr1_activation_schedule"] == "none"
        assert config["optimizer_parameter_groups"]["method_weight_decay"] == 0.0
        assert config["cicpmp_fr1_e4_style_hidden_residual"] is (
            variant == "cicpmp_fr1_scalar_residual_reference"
        )
        assert (output_dir / "result.csv").read_text(encoding="utf-8").count("\n") == 1
