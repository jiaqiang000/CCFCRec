#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


os.environ["CCFCREC_DEVICE"] = "cpu"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Amazon VG"))

import model as model_module  # noqa: E402


VARIANTS = (
    "cicpmp_fr1_scalar_residual_reference",
    "cicpmp_fr1_modality_film",
    "cicpmp_fr1_content_expert_routing",
    "cicpmp_fr1_cross_modal_attention",
    "cicpmp_fr1_modality_film_shuffle",
)


def make_args(variant: str, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        method_variant=variant,
        attr_num=8,
        attr_present_dim=4,
        implicit_dim=4,
        cat_implicit_dim=4,
        user_number=7,
        item_number=5,
        pretrain=False,
        pretrain_update=False,
        seed=seed,
        cicp_profile_path=("scalar.csv" if "scalar" in variant else ""),
        cicp_mp_profile_path=("mp.csv" if "scalar" not in variant else ""),
        cicpmp_fr1_block_dim=6,
        cicpmp_fr1_residual_max_ratio=0.15,
        cicpmp_fr1_method_weight_decay=0.0,
        weight_decay=0.1,
        learning_rate=0.0001,
    )


def method_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith("cicpmp_fr1_")
    }


def run_audit(seed: int) -> dict:
    attributes = torch.tensor(
        [
            [1, 1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, -1, -1, -1, -1],
            [-1, 1, 1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    images = torch.randn((3, 4096), generator=torch.Generator().manual_seed(seed))
    scalar_features = torch.tensor(
        [[0.2, 0.8, 0.64], [0.5, 0.5, 1.0], [0.8, 0.2, 0.64]],
        dtype=torch.float32,
    )
    mp_features = torch.randn(
        (3, 23),
        generator=torch.Generator().manual_seed(seed + 1),
    )

    models: dict[str, torch.nn.Module] = {}
    common_hashes: dict[str, str] = {}
    post_rng_hashes: dict[str, str] = {}
    initial_effect_max_abs: dict[str, float] = {}
    optimizer_groups: dict[str, dict] = {}

    for variant in ("baseline", *VARIANTS):
        args = make_args(variant, seed)
        torch.manual_seed(seed)
        model = model_module.CCFCRec(args)
        models[variant] = model
        common_hashes[variant] = model_module.ccfcrec_common_parameter_sha256(model)
        post_rng_hashes[variant] = hashlib.sha256(
            torch.get_rng_state().numpy().tobytes()
        ).hexdigest()
        if variant != "baseline":
            model_module.validate_method_args(args)
            optimizer_groups[variant] = model_module.optimizer_parameter_group_audit(
                model,
                args,
            )

    if len(set(common_hashes.values())) != 1:
        raise ValueError(f"common initialization mismatch: {common_hashes}")
    if len(set(post_rng_hashes.values())) != 1:
        raise ValueError(f"post-initialization RNG mismatch: {post_rng_hashes}")

    baseline = models["baseline"](attributes, images, len(attributes))
    residual_branches = []
    for variant in VARIANTS:
        model = models[variant]
        kwargs = (
            {"cicp_features": scalar_features}
            if variant == "cicpmp_fr1_scalar_residual_reference"
            else {"cicp_mp_features": mp_features}
        )
        output = model(attributes, images, len(attributes), **kwargs)
        initial_effect_max_abs[variant] = float((output - baseline).abs().max())
        if initial_effect_max_abs[variant] > 1e-7:
            raise ValueError(
                f"{variant} does not have zero initial effect: "
                f"{initial_effect_max_abs[variant]}"
            )
        if model._last_cicpmp_residual is not None:
            residual_branches.append(variant)
        if any(
            isinstance(module, torch.nn.LayerNorm)
            for name, module in model.named_modules()
            if name.startswith("cicpmp_fr1_")
        ):
            raise ValueError(f"{variant} unexpectedly uses per-item LayerNorm")
        audit = optimizer_groups[variant]
        if audit["base_weight_decay"] != 0.1:
            raise ValueError(f"{variant} changed base weight decay")
        if audit["method_weight_decay"] != 0.0:
            raise ValueError(f"{variant} method weight decay is not zero")

    expected_residual = ["cicpmp_fr1_scalar_residual_reference"]
    if residual_branches != expected_residual:
        raise ValueError(
            f"E4-style residual branch audit failed: {residual_branches}"
        )

    real_state = method_state(models["cicpmp_fr1_modality_film"])
    shuffle_state = method_state(models["cicpmp_fr1_modality_film_shuffle"])
    if real_state.keys() != shuffle_state.keys():
        raise ValueError("real and shuffled modality FiLM state keys differ")
    for name in real_state:
        torch.testing.assert_close(real_state[name], shuffle_state[name])

    return {
        "protocol": "cicpmp_fr1_five_final_repairs_v1",
        "seed": seed,
        "variants": list(VARIANTS),
        "common_parameter_sha256_by_variant": common_hashes,
        "common_hash_unique_count": len(set(common_hashes.values())),
        "post_initialization_rng_sha256_by_variant": post_rng_hashes,
        "post_initialization_rng_hash_unique_count": len(
            set(post_rng_hashes.values())
        ),
        "initial_effect_max_abs_by_variant": initial_effect_max_abs,
        "all_initial_effects_exact_zero_with_tolerance_1e_7": True,
        "e4_style_hidden_residual_branch_count": len(residual_branches),
        "e4_style_hidden_residual_branches": residual_branches,
        "modality_film_control_parameter_state_equal": True,
        "optimizer_parameter_groups": optimizer_groups,
        "per_item_layer_norm_in_fr1_modules": False,
        "activation_schedule": "none",
        "offline_auxiliary_target": False,
        "fixed_reliability_multiplication": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    audit = run_audit(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
