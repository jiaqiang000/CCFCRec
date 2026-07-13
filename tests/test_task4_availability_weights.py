import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

os.environ["CCFCREC_DEVICE"] = "cpu"

REPO_ROOT = Path(__file__).resolve().parents[1]
AMAZON_VG_DIR = REPO_ROOT / "Amazon VG"
sys.path.insert(0, str(AMAZON_VG_DIR))

import pandas as pd
import torch
from m11_features import (
    M11_FEATURE_MODE_FULL_STRUCTURAL,
    M11_FEATURE_MODE_TARGET_MASKED,
    build_m11_feature_tensor,
)

from model import (
    CCFCRec,
    M11R2_FEATURE_METHOD_VARIANTS,
    TASK4_METHOD_VARIANTS,
    build_m11r2_curriculum_weights,
    build_m11r2_focal_qbpr_weights,
    build_m11r2_target_score_tensor_from_profile,
    build_m11r3_neighbor_transfer_loss,
    build_run_config,
    build_task4_competitor_pair_targets_from_profile,
    build_task4_pair_margin_targets_from_profile,
    build_task4_item_weights_from_profile,
    cap_m11_residual_norm,
    load_task4_boundary_competitors,
    resolve_task4_competitor_user_ids,
    resolve_m11_feature_mode,
    task4_competitor_pair_loss,
    task4_pair_margin_loss,
    uses_task4_boundary_competitor_pair,
    uses_task4_competitor_pair,
    uses_task4_item_weights,
    uses_task4_pair_margin,
    uses_m11r2_feature_fusion,
    uses_m11r2_focal_qbpr,
    validate_method_args,
)


def make_args(method_variant):
    return SimpleNamespace(
        attr_num=8,
        attr_present_dim=4,
        implicit_dim=4,
        cat_implicit_dim=4,
        user_number=3,
        item_number=4,
        pretrain=False,
        pretrain_update=False,
        method_variant=method_variant,
        category_conf_dim=3,
        category_conf_max_count=5,
        category_gate_scale=0.5,
        weak_cat_threshold=3,
        weak_loss_alpha=0.5,
        adaptive_loss_alpha=1.0,
        adaptive_history_max_count=20,
        reweight_q_bpr=False,
        reweight_self_contrast=False,
        reweight_contrast=False,
        task4_profile_path="/tmp/task4_profile.csv",
        task4_loss_alpha=0.5,
        task4_shuffle_seed=123,
        task4_disable_q_bpr_weight=False,
        task4_disable_self_contrast_weight=False,
        task4_reweight_contrast=False,
        task4_pair_margin=0.2,
        task4_competitor_alpha=0.25,
        task4_competitor_margin=0.1,
        task4_competitor_k=20,
        task4_boundary_competitor_cache_path="/tmp/task4_boundary_cache.csv",
        m11r2_focal_gamma=2.0,
        m11r2_focal_temperature=1.0,
        m11r2_curriculum_warmup_epochs=20,
        m11r2_feature_dim=3,
        m11r3_residual_max_ratio=0.15,
        m11r3_neighbor_loss_weight=0.1,
        m11r3_neighbor_temperature=0.25,
        m11r3_film_strength=0.1,
        seed=43,
        num_workers=0,
        batch_size=1024,
        save_batch_time=300,
        validate_batch_size=512,
        negative_sampling_mode="fast_uniform",
        negative_sampling_cache_size=512,
    )


def make_profile():
    return pd.DataFrame(
        [
            {
                "raw_asin": "a",
                "split": "train",
                "cat_count_bin": "cat_count_5_plus",
                "category_count": 5,
                "RSP_group": "RSP_high",
                "RSP_score": 0.9,
                "s_cat_v3": 0.20,
                "high_acat_flag": False,
                "train_safe_hard_proxy_score": 0.10,
                "train_safe_hard_proxy_high_flag": False,
                "high_acat_train_safe_hard_flag": False,
                "eval_baseline_hard_flag": True,
                "baseline_ndcg@20": 0.0,
                "m11_high_acat_low_rsp_neighbor_support_flag": False,
                "m11_target_score": 0.20,
                "m11r1_full_target_flag": False,
                "m11r1_full_target_loss_score": 0.0,
                "m11r1_popmatch_control_flag": False,
                "m11r1_popmatch_control_loss_score": 0.0,
                "m11r1_lowacat_control_flag": False,
                "m11r1_lowacat_control_loss_score": 0.0,
                "category_neighbor_mismatch_proxy_high_flag": True,
                "category_neighbor_mismatch_proxy_score": 0.6,
                "support_tail_proxy_high_flag": True,
                "support_tail_proxy_score": 0.5,
            },
            {
                "raw_asin": "b",
                "split": "train",
                "cat_count_bin": "cat_count_4",
                "category_count": 4,
                "RSP_group": "RSP_low",
                "RSP_score": 0.1,
                "s_cat_v3": 0.30,
                "high_acat_flag": True,
                "train_safe_hard_proxy_score": 0.20,
                "train_safe_hard_proxy_high_flag": False,
                "high_acat_train_safe_hard_flag": False,
                "eval_baseline_hard_flag": False,
                "baseline_ndcg@20": 0.5,
                "m11_high_acat_low_rsp_neighbor_support_flag": False,
                "m11_target_score": 0.40,
                "m11r1_full_target_flag": False,
                "m11r1_full_target_loss_score": 0.0,
                "m11r1_popmatch_control_flag": True,
                "m11r1_popmatch_control_loss_score": 0.8,
                "m11r1_lowacat_control_flag": False,
                "m11r1_lowacat_control_loss_score": 0.0,
                "category_neighbor_mismatch_proxy_high_flag": True,
                "category_neighbor_mismatch_proxy_score": 0.7,
                "support_tail_proxy_high_flag": True,
                "support_tail_proxy_score": 0.6,
            },
            {
                "raw_asin": "c",
                "split": "train",
                "cat_count_bin": "cat_count_5_plus",
                "category_count": 5,
                "RSP_group": "RSP_low",
                "RSP_score": 0.2,
                "s_cat_v3": 0.90,
                "high_acat_flag": True,
                "train_safe_hard_proxy_score": 0.80,
                "train_safe_hard_proxy_high_flag": True,
                "high_acat_train_safe_hard_flag": True,
                "eval_baseline_hard_flag": True,
                "baseline_ndcg@20": 0.0,
                "m11_high_acat_low_rsp_neighbor_support_flag": True,
                "m11_target_score": 0.80,
                "m11r1_full_target_flag": True,
                "m11r1_full_target_loss_score": 0.8,
                "m11r1_popmatch_control_flag": False,
                "m11r1_popmatch_control_loss_score": 0.0,
                "m11r1_lowacat_control_flag": False,
                "m11r1_lowacat_control_loss_score": 0.0,
                "category_neighbor_mismatch_proxy_high_flag": True,
                "category_neighbor_mismatch_proxy_score": 0.9,
                "support_tail_proxy_high_flag": True,
                "support_tail_proxy_score": 0.8,
            },
            {
                "raw_asin": "d",
                "split": "train",
                "cat_count_bin": "cat_count_4",
                "category_count": 4,
                "RSP_group": "RSP_mid",
                "RSP_score": 0.5,
                "s_cat_v3": 0.40,
                "high_acat_flag": False,
                "train_safe_hard_proxy_score": 0.30,
                "train_safe_hard_proxy_high_flag": True,
                "high_acat_train_safe_hard_flag": False,
                "eval_baseline_hard_flag": False,
                "baseline_ndcg@20": 0.4,
                "m11_high_acat_low_rsp_neighbor_support_flag": False,
                "m11_target_score": 0.30,
                "m11r1_full_target_flag": False,
                "m11r1_full_target_loss_score": 0.0,
                "m11r1_popmatch_control_flag": False,
                "m11r1_popmatch_control_loss_score": 0.0,
                "m11r1_lowacat_control_flag": True,
                "m11r1_lowacat_control_loss_score": 0.8,
                "category_neighbor_mismatch_proxy_high_flag": True,
                "category_neighbor_mismatch_proxy_score": 0.5,
                "support_tail_proxy_high_flag": True,
                "support_tail_proxy_score": 0.4,
            },
        ]
    )


class Task4AvailabilityWeightsTest(unittest.TestCase):
    def test_task4_variants_keep_baseline_architecture(self):
        for method_variant in TASK4_METHOD_VARIANTS - M11R2_FEATURE_METHOD_VARIANTS:
            model = CCFCRec(make_args(method_variant))
            self.assertEqual(model.gen_layer1.in_features, 8)
            self.assertFalse(hasattr(model, "category_conf_embedding"))

    def test_m11r2_feature_fusion_adds_only_projected_train_safe_features(self):
        args = make_args("m11r2_target_feature_fusion")
        model = CCFCRec(args)

        self.assertTrue(uses_m11r2_feature_fusion(args))
        self.assertEqual(model.gen_layer1.in_features, 8)
        self.assertEqual(model.m11r2_feature_projection.in_features, 6)
        self.assertEqual(model.m11r2_feature_projection.out_features, 3)
        self.assertEqual(model.m11r2_feature_to_hidden.in_features, 3)
        self.assertEqual(model.m11r2_feature_to_hidden.out_features, 4)

    def test_m1_uses_rsp_high_flag(self):
        weights = build_task4_item_weights_from_profile(
            make_profile(),
            {"a": 0, "b": 1, "c": 2, "d": 3},
            make_args("task4_rsp_high_weight"),
        )

        self.assertGreater(weights[0].item(), weights[1].item())
        self.assertGreater(weights[0].item(), weights[2].item())
        self.assertAlmostEqual(weights.mean().item(), 1.0)

    def test_m2_and_m3_use_distinct_train_safe_flags(self):
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        m2 = build_task4_item_weights_from_profile(make_profile(), item_map, make_args("task4_acat_high_weight"))
        m3 = build_task4_item_weights_from_profile(make_profile(), item_map, make_args("task4_acat_trainhard_weight"))

        self.assertGreater(m2[1].item(), m2[0].item())
        self.assertGreater(m2[2].item(), m2[0].item())
        self.assertEqual(m3[1].item(), m3[0].item())
        self.assertGreater(m3[2].item(), m3[0].item())

    def test_m6_shuffle_is_deterministic_and_breaks_direct_acat_mapping(self):
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        args = make_args("task4_acat_shuffle_high_weight")

        first = build_task4_item_weights_from_profile(make_profile(), item_map, args)
        second = build_task4_item_weights_from_profile(make_profile(), item_map, args)
        acat = build_task4_item_weights_from_profile(make_profile(), item_map, make_args("task4_acat_high_weight"))

        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, acat))

    def test_eval_columns_do_not_change_train_safe_weights(self):
        profile = make_profile()
        perturbed = profile.copy()
        perturbed["eval_baseline_hard_flag"] = ~perturbed["eval_baseline_hard_flag"].astype(bool)
        perturbed["baseline_ndcg@20"] = 1.0
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        args = make_args("task4_acat_trainhard_weight")

        original = build_task4_item_weights_from_profile(profile, item_map, args)
        changed = build_task4_item_weights_from_profile(perturbed, item_map, args)

        torch.testing.assert_close(original, changed)

    def test_task4_args_require_profile_path_and_positive_alpha(self):
        args = make_args("task4_acat_trainhard_weight")
        args.task4_profile_path = ""
        with self.assertRaises(ValueError):
            validate_method_args(args)

        args = make_args("task4_acat_trainhard_weight")
        args.task4_loss_alpha = 0.0
        with self.assertRaises(ValueError):
            validate_method_args(args)

    def test_run_config_records_task4_params(self):
        args = make_args("task4_acat_trainhard_weight")
        model = CCFCRec(args)

        config = build_run_config(args, model)

        self.assertEqual(config["method_variant"], "task4_acat_trainhard_weight")
        self.assertEqual(config["task4_profile_path"], "/tmp/task4_profile.csv")
        self.assertEqual(config["task4_loss_alpha"], 0.5)
        self.assertFalse(config["task4_disable_q_bpr_weight"])

    def test_m4_variants_use_pair_margin_not_item_weighting(self):
        for method_variant in {
            "task4_acat_pairmargin_weight",
            "task4_acat_rsp_residual_pairmargin",
            "task4_acat_hardonly_qmargin",
        }:
            args = make_args(method_variant)
            self.assertIn(method_variant, TASK4_METHOD_VARIANTS)
            self.assertFalse(uses_task4_item_weights(args))
            self.assertTrue(uses_task4_pair_margin(args))
            validate_method_args(args)

    def test_m4a_targets_only_high_acat_train_safe_hard_items(self):
        targets = build_task4_pair_margin_targets_from_profile(
            make_profile(),
            {"a": 0, "b": 1, "c": 2, "d": 3},
            make_args("task4_acat_pairmargin_weight"),
        )

        self.assertEqual(targets["loss_weight"][0].item(), 0.0)
        self.assertEqual(targets["loss_weight"][1].item(), 0.0)
        self.assertGreater(targets["loss_weight"][2].item(), 0.0)
        self.assertEqual(targets["loss_weight"][3].item(), 0.0)
        self.assertGreater(targets["margin"][2].item(), 0.2)

    def test_m4b_requires_positive_acat_residual_against_rsp_group(self):
        profile = make_profile()
        profile.loc[profile["raw_asin"] == "c", "s_cat_v3"] = 0.90
        profile.loc[profile["raw_asin"] == "b", "s_cat_v3"] = 0.10

        targets = build_task4_pair_margin_targets_from_profile(
            profile,
            {"a": 0, "b": 1, "c": 2, "d": 3},
            make_args("task4_acat_rsp_residual_pairmargin"),
        )

        self.assertEqual(targets["loss_weight"][1].item(), 0.0)
        self.assertGreater(targets["loss_weight"][2].item(), 0.0)
        self.assertGreater(targets["margin"][2].item(), 0.2)

    def test_m4c_uses_hardonly_qside_margin_mask(self):
        targets = build_task4_pair_margin_targets_from_profile(
            make_profile(),
            {"a": 0, "b": 1, "c": 2, "d": 3},
            make_args("task4_acat_hardonly_qmargin"),
        )

        self.assertEqual(targets["loss_weight"][0].item(), 0.0)
        self.assertEqual(targets["loss_weight"][1].item(), 0.0)
        self.assertGreater(targets["loss_weight"][2].item(), 0.0)
        self.assertEqual(targets["loss_weight"][3].item(), 0.0)

    def test_pair_margin_targets_ignore_eval_only_columns(self):
        profile = make_profile()
        perturbed = profile.copy()
        perturbed["eval_baseline_hard_flag"] = ~perturbed["eval_baseline_hard_flag"].astype(bool)
        perturbed["baseline_ndcg@20"] = 1.0
        perturbed["baseline_margin_proxy"] = -999.0
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        args = make_args("task4_acat_pairmargin_weight")

        original = build_task4_pair_margin_targets_from_profile(profile, item_map, args)
        changed = build_task4_pair_margin_targets_from_profile(perturbed, item_map, args)

        torch.testing.assert_close(original["loss_weight"], changed["loss_weight"])
        torch.testing.assert_close(original["margin"], changed["margin"])

    def test_pair_margin_loss_penalizes_only_weighted_margin_violations(self):
        diff = torch.tensor([0.10, 0.30, -0.10])
        targets = {
            "loss_weight": torch.tensor([0.5, 0.5, 0.0]),
            "margin": torch.tensor([0.20, 0.20, 0.20]),
        }

        loss = task4_pair_margin_loss(diff, targets)

        self.assertAlmostEqual(loss.item(), 0.05)

    def test_highdetail_trainhard_weight_targets_only_high_detail_trainhard_items(self):
        profile = make_profile()
        profile.loc[profile["raw_asin"] == "b", "high_acat_train_safe_hard_flag"] = True
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}

        weights = build_task4_item_weights_from_profile(
            profile,
            item_map,
            make_args("task4_highdetail_trainhard_weight"),
        )

        self.assertEqual(weights[0].item(), weights[1].item())
        self.assertGreater(weights[2].item(), weights[1].item())
        self.assertEqual(weights[3].item(), weights[1].item())

    def test_highdetail_trainhard_shuffle_preserves_high_detail_target_count(self):
        profile = make_profile()
        profile.loc[profile["raw_asin"] == "a", "high_acat_train_safe_hard_flag"] = True
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        args = make_args("task4_highdetail_trainhard_shuffle_weight")

        first = build_task4_item_weights_from_profile(profile, item_map, args)
        second = build_task4_item_weights_from_profile(profile, item_map, args)
        real = build_task4_item_weights_from_profile(profile, item_map, make_args("task4_highdetail_trainhard_weight"))

        torch.testing.assert_close(first, second)
        self.assertEqual(int((first > first.min()).sum().item()), int((real > real.min()).sum().item()))

    def test_highdetail_pairmargin_targets_only_high_detail_trainhard_items(self):
        profile = make_profile()
        profile.loc[profile["raw_asin"] == "b", "high_acat_train_safe_hard_flag"] = True
        targets = build_task4_pair_margin_targets_from_profile(
            profile,
            {"a": 0, "b": 1, "c": 2, "d": 3},
            make_args("task4_highdetail_pairmargin"),
        )

        self.assertEqual(targets["loss_weight"][0].item(), 0.0)
        self.assertEqual(targets["loss_weight"][1].item(), 0.0)
        self.assertGreater(targets["loss_weight"][2].item(), 0.0)
        self.assertEqual(targets["loss_weight"][3].item(), 0.0)
        self.assertGreater(targets["margin"][2].item(), 0.2)

    def test_highdetail_variants_are_registered_in_expected_carrier_sets(self):
        self.assertTrue(uses_task4_item_weights(make_args("task4_highdetail_trainhard_weight")))
        self.assertTrue(uses_task4_item_weights(make_args("task4_highdetail_trainhard_shuffle_weight")))
        self.assertTrue(uses_task4_pair_margin(make_args("task4_highdetail_pairmargin")))
        self.assertTrue(uses_task4_pair_margin(make_args("task4_highdetail_pairmargin_shuffle")))

        for method_variant in {
            "task4_highdetail_trainhard_weight",
            "task4_highdetail_trainhard_shuffle_weight",
            "task4_highdetail_pairmargin",
            "task4_highdetail_pairmargin_shuffle",
        }:
            self.assertIn(method_variant, TASK4_METHOD_VARIANTS)
            validate_method_args(make_args(method_variant))

    def test_competitor_pair_variants_are_registered_separately(self):
        for method_variant in {
            "task4_competitor_pair",
            "task4_competitor_pair_shuffle",
            "task4_competitor_pair_rsp_control",
            "task4_competitor_pair_acat_control",
        }:
            args = make_args(method_variant)
            self.assertIn(method_variant, TASK4_METHOD_VARIANTS)
            self.assertTrue(uses_task4_competitor_pair(args))
            self.assertFalse(uses_task4_item_weights(args))
            self.assertFalse(uses_task4_pair_margin(args))
            validate_method_args(args)

    def test_boundary_competitor_variants_are_registered_and_require_cache(self):
        for method_variant in {
            "task4_boundary_competitor_pair",
            "task4_boundary_competitor_pair_shuffle",
            "task4_boundary_competitor_pair_rsp_control",
            "task4_boundary_competitor_pair_acat_control",
        }:
            args = make_args(method_variant)
            self.assertIn(method_variant, TASK4_METHOD_VARIANTS)
            self.assertTrue(uses_task4_competitor_pair(args))
            self.assertTrue(uses_task4_boundary_competitor_pair(args))
            self.assertFalse(uses_task4_item_weights(args))
            self.assertFalse(uses_task4_pair_margin(args))
            validate_method_args(args)

            args.task4_boundary_competitor_cache_path = ""
            with self.assertRaises(ValueError):
                validate_method_args(args)

    def test_boundary_competitor_targets_reuse_competitor_pair_masks(self):
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        real = build_task4_competitor_pair_targets_from_profile(
            make_profile(),
            item_map,
            make_args("task4_competitor_pair"),
        )
        boundary = build_task4_competitor_pair_targets_from_profile(
            make_profile(),
            item_map,
            make_args("task4_boundary_competitor_pair"),
        )

        torch.testing.assert_close(real["loss_weight"], boundary["loss_weight"])
        torch.testing.assert_close(real["margin"], boundary["margin"])

    def test_competitor_pair_targets_highdetail_trainhard_items(self):
        profile = make_profile()
        profile.loc[profile["raw_asin"] == "b", "high_acat_train_safe_hard_flag"] = True
        targets = build_task4_competitor_pair_targets_from_profile(
            profile,
            {"a": 0, "b": 1, "c": 2, "d": 3},
            make_args("task4_competitor_pair"),
        )

        self.assertEqual(targets["loss_weight"][0].item(), 0.0)
        self.assertEqual(targets["loss_weight"][1].item(), 0.0)
        self.assertGreater(targets["loss_weight"][2].item(), 0.0)
        self.assertEqual(targets["loss_weight"][3].item(), 0.0)
        self.assertGreater(targets["margin"][2].item(), 0.1)

    def test_competitor_pair_shuffle_preserves_target_count(self):
        profile = make_profile()
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        shuffle_args = make_args("task4_competitor_pair_shuffle")
        shuffle_args.task4_shuffle_seed = 2

        shuffled = build_task4_competitor_pair_targets_from_profile(
            profile,
            item_map,
            shuffle_args,
        )
        real = build_task4_competitor_pair_targets_from_profile(
            profile,
            item_map,
            make_args("task4_competitor_pair"),
        )

        self.assertEqual(
            int((shuffled["loss_weight"] > 0).sum().item()),
            int((real["loss_weight"] > 0).sum().item()),
        )
        self.assertFalse(torch.equal(shuffled["loss_weight"], real["loss_weight"]))

    def test_competitor_pair_controls_use_rsp_and_acat_masks(self):
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        rsp = build_task4_competitor_pair_targets_from_profile(
            make_profile(),
            item_map,
            make_args("task4_competitor_pair_rsp_control"),
        )
        acat = build_task4_competitor_pair_targets_from_profile(
            make_profile(),
            item_map,
            make_args("task4_competitor_pair_acat_control"),
        )

        self.assertGreater(rsp["loss_weight"][0].item(), 0.0)
        self.assertEqual(acat["loss_weight"][0].item(), 0.0)
        self.assertGreater(acat["loss_weight"][2].item(), 0.0)
        self.assertFalse(torch.equal(rsp["loss_weight"], acat["loss_weight"]))

    def test_competitor_pair_loss_uses_softplus_only_for_weighted_items(self):
        diff = torch.tensor([0.30, -0.10, 0.00])
        targets = {
            "loss_weight": torch.tensor([0.5, 0.0, 1.0]),
            "margin": torch.tensor([0.10, 0.10, 0.20]),
        }
        expected = torch.nn.functional.softplus(torch.tensor(-0.20)) * 0.5
        expected = expected + torch.nn.functional.softplus(torch.tensor(0.20))

        loss = task4_competitor_pair_loss(diff, targets)

        torch.testing.assert_close(loss, expected)

    def test_boundary_competitor_cache_maps_raw_user_ids(self):
        with TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "boundary_cache.csv"
            pd.DataFrame(
                [
                    {
                        "raw_asin": "a",
                        "boundary_competitor_user": "user_c",
                        "boundary_competitor_serial_user": 999,
                    },
                    {
                        "raw_asin": "missing_item",
                        "boundary_competitor_user": "user_b",
                    },
                ]
            ).to_csv(cache_path, index=False)
            args = make_args("task4_boundary_competitor_pair")
            args.task4_boundary_competitor_cache_path = str(cache_path)

            boundary = load_task4_boundary_competitors(
                {"a": 0, "b": 1},
                {"user_a": 0, "user_b": 1, "user_c": 2},
                args,
                item_number=2,
            )

        self.assertEqual(boundary.tolist(), [2, -1])

    def test_boundary_competitor_user_resolution_falls_back_to_batch_negative(self):
        item = torch.tensor([0, 1, 2])
        neg_user = torch.tensor([10, 11, 12])
        boundary = torch.tensor([2, -1, 5])

        resolved = resolve_task4_competitor_user_ids(item, neg_user, boundary)

        self.assertEqual(resolved.tolist(), [2, 11, 5])

    def test_run_config_records_competitor_pair_params(self):
        args = make_args("task4_competitor_pair")
        model = CCFCRec(args)

        config = build_run_config(args, model)

        self.assertEqual(config["task4_competitor_alpha"], 0.25)
        self.assertEqual(config["task4_competitor_margin"], 0.1)
        self.assertEqual(config["task4_competitor_k"], 20)

    def test_run_config_records_boundary_competitor_cache_path(self):
        args = make_args("task4_boundary_competitor_pair")
        model = CCFCRec(args)

        config = build_run_config(args, model)

        self.assertEqual(config["task4_boundary_competitor_cache_path"], "/tmp/task4_boundary_cache.csv")

    def test_m11_competitor_pair_variants_are_registered_without_boundary_cache(self):
        for method_variant in {
            "m11_target_competitor_pair",
            "m11_target_competitor_pair_shuffle",
            "m11_target_competitor_pair_lowrsp_control",
            "m11_target_competitor_pair_rsp_control",
        }:
            args = make_args(method_variant)
            self.assertIn(method_variant, TASK4_METHOD_VARIANTS)
            self.assertTrue(uses_task4_competitor_pair(args))
            self.assertFalse(uses_task4_boundary_competitor_pair(args))
            self.assertFalse(uses_task4_item_weights(args))
            self.assertFalse(uses_task4_pair_margin(args))
            args.task4_boundary_competitor_cache_path = ""
            validate_method_args(args)

    def test_m11_real_targets_explicit_m11_flag_and_score(self):
        targets = build_task4_competitor_pair_targets_from_profile(
            make_profile(),
            {"a": 0, "b": 1, "c": 2, "d": 3},
            make_args("m11_target_competitor_pair"),
        )

        self.assertEqual(targets["loss_weight"][0].item(), 0.0)
        self.assertEqual(targets["loss_weight"][1].item(), 0.0)
        self.assertAlmostEqual(targets["loss_weight"][2].item(), 0.225, places=6)
        self.assertEqual(targets["loss_weight"][3].item(), 0.0)
        self.assertAlmostEqual(targets["margin"][2].item(), 0.18, places=6)

    def test_m11_targets_ignore_eval_only_columns(self):
        profile = make_profile()
        perturbed = profile.copy()
        perturbed["eval_baseline_hard_flag"] = ~perturbed["eval_baseline_hard_flag"].astype(bool)
        perturbed["baseline_ndcg@20"] = 1.0
        perturbed["baseline_best_target_rank"] = 1
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        args = make_args("m11_target_competitor_pair")

        original = build_task4_competitor_pair_targets_from_profile(profile, item_map, args)
        changed = build_task4_competitor_pair_targets_from_profile(perturbed, item_map, args)

        torch.testing.assert_close(original["loss_weight"], changed["loss_weight"])
        torch.testing.assert_close(original["margin"], changed["margin"])

    def test_m11_shuffle_preserves_real_target_count(self):
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        shuffle_args = make_args("m11_target_competitor_pair_shuffle")
        shuffle_args.task4_shuffle_seed = 2

        shuffled = build_task4_competitor_pair_targets_from_profile(make_profile(), item_map, shuffle_args)
        real = build_task4_competitor_pair_targets_from_profile(
            make_profile(),
            item_map,
            make_args("m11_target_competitor_pair"),
        )

        self.assertEqual(
            int((shuffled["loss_weight"] > 0).sum().item()),
            int((real["loss_weight"] > 0).sum().item()),
        )
        self.assertFalse(torch.equal(shuffled["loss_weight"], real["loss_weight"]))

    def test_m11_controls_separate_lowrsp_matched_and_rsp_high_targets(self):
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        lowrsp = build_task4_competitor_pair_targets_from_profile(
            make_profile(),
            item_map,
            make_args("m11_target_competitor_pair_lowrsp_control"),
        )
        rsp = build_task4_competitor_pair_targets_from_profile(
            make_profile(),
            item_map,
            make_args("m11_target_competitor_pair_rsp_control"),
        )

        self.assertGreater(rsp["loss_weight"][0].item(), 0.0)
        self.assertEqual(rsp["loss_weight"][2].item(), 0.0)
        self.assertGreater(lowrsp["loss_weight"][3].item(), 0.0)
        self.assertEqual(lowrsp["loss_weight"][2].item(), 0.0)
        self.assertFalse(torch.equal(lowrsp["loss_weight"], rsp["loss_weight"]))

    def test_m11r1_explicit_profile_variants_are_registered(self):
        for method_variant in {
            "m11r1_full_target_competitor_pair",
            "m11r1_popmatch_competitor_pair_control",
            "m11r1_lowacat_competitor_pair_control",
        }:
            args = make_args(method_variant)
            self.assertIn(method_variant, TASK4_METHOD_VARIANTS)
            self.assertTrue(uses_task4_competitor_pair(args))
            self.assertFalse(uses_task4_boundary_competitor_pair(args))
            validate_method_args(args)

    def test_m11r1_variants_use_only_their_explicit_flag_and_score_columns(self):
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        real = build_task4_competitor_pair_targets_from_profile(
            make_profile(), item_map, make_args("m11r1_full_target_competitor_pair")
        )
        popmatch = build_task4_competitor_pair_targets_from_profile(
            make_profile(), item_map, make_args("m11r1_popmatch_competitor_pair_control")
        )
        lowacat = build_task4_competitor_pair_targets_from_profile(
            make_profile(), item_map, make_args("m11r1_lowacat_competitor_pair_control")
        )

        self.assertEqual((real["loss_weight"] > 0).nonzero().flatten().tolist(), [2])
        self.assertEqual((popmatch["loss_weight"] > 0).nonzero().flatten().tolist(), [1])
        self.assertEqual((lowacat["loss_weight"] > 0).nonzero().flatten().tolist(), [3])
        self.assertAlmostEqual(real["loss_weight"][2].item(), 0.225, places=6)
        self.assertAlmostEqual(popmatch["loss_weight"][1].item(), 0.225, places=6)
        self.assertAlmostEqual(lowacat["loss_weight"][3].item(), 0.225, places=6)

    def test_m11r2_four_performance_variants_are_registered_as_distinct_mechanisms(self):
        static_args = make_args("m11r2_qbpr_score_weight")
        focal_args = make_args("m11r2_qbpr_focal")
        curriculum_args = make_args("m11r2_qbpr_curriculum")
        feature_args = make_args("m11r2_target_feature_fusion")

        for args in [static_args, focal_args, curriculum_args, feature_args]:
            self.assertIn(args.method_variant, TASK4_METHOD_VARIANTS)
            validate_method_args(args)
        self.assertTrue(uses_task4_item_weights(static_args))
        self.assertTrue(uses_task4_item_weights(curriculum_args))
        self.assertTrue(uses_m11r2_focal_qbpr(focal_args))
        self.assertTrue(uses_m11r2_feature_fusion(feature_args))
        self.assertFalse(uses_task4_item_weights(focal_args))
        self.assertFalse(uses_task4_competitor_pair(feature_args))

    def test_m11r2_static_and_curriculum_use_unchanged_full_target_scores(self):
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        static = build_task4_item_weights_from_profile(
            make_profile(), item_map, make_args("m11r2_qbpr_score_weight")
        )
        curriculum = build_task4_item_weights_from_profile(
            make_profile(), item_map, make_args("m11r2_qbpr_curriculum")
        )

        torch.testing.assert_close(static, curriculum)
        self.assertGreater(static[2].item(), static[0].item())
        self.assertEqual(static[0].item(), static[1].item())
        self.assertEqual(static[1].item(), static[3].item())
        self.assertAlmostEqual(static.mean().item(), 1.0, places=6)

    def test_m11r2_focal_weights_only_emphasize_hard_full_target_interactions(self):
        args = make_args("m11r2_qbpr_focal")
        args.task4_loss_alpha = 0.75
        score_diff = torch.tensor([0.0, 0.0, -2.0, 0.0])
        target_scores = torch.tensor([0.0, 0.0, 0.8, 0.0])

        weights = build_m11r2_focal_qbpr_weights(score_diff, target_scores, args)

        self.assertGreater(weights[2].item(), weights[0].item())
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=6)
        easier = build_m11r2_focal_qbpr_weights(
            torch.tensor([0.0, 0.0, 2.0, 0.0]), target_scores, args
        )
        self.assertGreater(weights[2].item(), easier[2].item())

    def test_m11r2_curriculum_reaches_full_target_weight_after_warmup(self):
        full = torch.tensor([0.8, 1.2])

        first = build_m11r2_curriculum_weights(full, epoch_index=0, warmup_epochs=20)
        middle = build_m11r2_curriculum_weights(full, epoch_index=9, warmup_epochs=20)
        complete = build_m11r2_curriculum_weights(full, epoch_index=19, warmup_epochs=20)

        self.assertLess(torch.abs(first - 1.0).sum(), torch.abs(middle - 1.0).sum())
        torch.testing.assert_close(complete, full)

    def test_m11r2_target_score_tensor_uses_only_explicit_full_target_scope(self):
        tensor = build_m11r2_target_score_tensor_from_profile(
            make_profile(), {"a": 0, "b": 1, "c": 2, "d": 3}
        )

        torch.testing.assert_close(tensor, torch.tensor([0.0, 0.0, 0.8, 0.0]))

    def test_m11r2_feature_tensor_masks_non_target_items_and_ignores_eval_columns(self):
        profile = make_profile()
        perturbed = profile.copy()
        perturbed["baseline_ndcg@20"] = 999.0
        perturbed["eval_baseline_hard_flag"] = ~perturbed["eval_baseline_hard_flag"].astype(bool)
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}

        original = build_m11_feature_tensor(profile, item_map)
        changed = build_m11_feature_tensor(perturbed, item_map)

        torch.testing.assert_close(original, changed)
        torch.testing.assert_close(original[[0, 1, 3]], torch.zeros((3, 6)))
        self.assertEqual(original[2, 0].item(), 1.0)
        self.assertAlmostEqual(original[2, 1].item(), 0.8, places=6)

    def test_m11r2_feature_fusion_forward_requires_and_uses_six_features(self):
        model = CCFCRec(make_args("m11r2_target_feature_fusion"))
        attributes = torch.tensor(
            [
                [0, 1, -1, -1, -1, -1, -1, -1],
                [2, -1, -1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.long,
        )
        images = torch.ones((2, 4096), dtype=torch.float32)

        with self.assertRaises(ValueError):
            model(attributes, images, 2)
        output = model(attributes, images, 2, m11_features=torch.ones((2, 6)))

        self.assertEqual(tuple(output.shape), (2, 4))

    def test_m11r3_full_structural_features_keep_non_target_recommendation_time_signals(self):
        profile = make_profile()
        item_map = {"a": 0, "b": 1, "c": 2, "d": 3}

        masked = build_m11_feature_tensor(
            profile,
            item_map,
            feature_mode=M11_FEATURE_MODE_TARGET_MASKED,
        )
        structural = build_m11_feature_tensor(
            profile,
            item_map,
            feature_mode=M11_FEATURE_MODE_FULL_STRUCTURAL,
        )

        torch.testing.assert_close(masked[0], torch.zeros(6))
        self.assertEqual(structural[0, 0].item(), 0.0)
        self.assertAlmostEqual(structural[0, 1].item(), 0.2, places=6)
        self.assertGreater(structural[0, 1:].abs().sum().item(), 0.0)
        self.assertAlmostEqual(structural[2, 1].item(), 0.8, places=6)

    def test_m11r3_four_variants_use_expected_feature_scope_and_distinct_modules(self):
        dual_args = make_args("m11r3_dual_residual")
        norm_args = make_args("m11r3_norm_capped_residual")
        neighbor_args = make_args("m11r3_neighbor_transfer")
        film_args = make_args("m11r3_target_film")

        for args in [dual_args, norm_args, neighbor_args, film_args]:
            self.assertIn(args.method_variant, TASK4_METHOD_VARIANTS)
            self.assertTrue(uses_m11r2_feature_fusion(args))
            validate_method_args(args)
        self.assertEqual(resolve_m11_feature_mode(norm_args), M11_FEATURE_MODE_TARGET_MASKED)
        for args in [dual_args, neighbor_args, film_args]:
            self.assertEqual(resolve_m11_feature_mode(args), M11_FEATURE_MODE_FULL_STRUCTURAL)

        dual = CCFCRec(dual_args)
        norm = CCFCRec(norm_args)
        neighbor = CCFCRec(neighbor_args)
        film = CCFCRec(film_args)
        self.assertTrue(hasattr(dual, "m11r3_global_to_hidden"))
        self.assertFalse(hasattr(norm, "m11r3_global_to_hidden"))
        self.assertTrue(neighbor.uses_m11r3_neighbor_transfer())
        self.assertTrue(hasattr(film, "m11r3_film_scale"))
        self.assertFalse(hasattr(film, "m11r2_feature_to_hidden"))

    def test_m11r3_norm_cap_enforces_hidden_relative_bound(self):
        hidden = torch.tensor([[3.0, 4.0], [0.0, 10.0]])
        residual = torch.tensor([[6.0, 8.0], [0.0, 1.0]])

        capped = cap_m11_residual_norm(residual, hidden, max_ratio=0.15)

        ratios = capped.norm(dim=1) / hidden.norm(dim=1)
        self.assertLessEqual(ratios.max().item(), 0.150001)
        torch.testing.assert_close(capped[1], residual[1])

    def test_m11r3_neighbor_transfer_is_one_way_and_requires_complementary_groups(self):
        residual = torch.tensor(
            [[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]],
            requires_grad=True,
        )
        features = torch.tensor(
            [
                [1.0, 0.8, 0.8, 0.8, 0.8, 0.8],
                [0.0, 0.79, 0.79, 0.79, 0.79, 0.79],
                [0.0, 0.1, 0.1, 0.1, 0.1, 0.1],
            ]
        )

        loss = build_m11r3_neighbor_transfer_loss(residual, features, temperature=0.25)
        loss.backward()

        self.assertGreater(loss.item(), 0.0)
        torch.testing.assert_close(residual.grad[0], torch.zeros(2))
        self.assertGreater(residual.grad[1].abs().sum().item(), 0.0)
        no_target = features.clone()
        no_target[:, 0] = 0.0
        self.assertEqual(build_m11r3_neighbor_transfer_loss(residual.detach(), no_target).item(), 0.0)

    def test_m11r3_forward_paths_are_finite_and_record_bounded_corrections(self):
        attributes = torch.tensor(
            [[0, 1, -1, -1, -1, -1, -1, -1], [2, -1, -1, -1, -1, -1, -1, -1]],
            dtype=torch.long,
        )
        images = torch.ones((2, 4096), dtype=torch.float32)
        structural = torch.tensor(
            [[1.0, 0.8, 0.7, 0.6, 0.8, 0.9], [0.0, 0.4, 0.3, 0.2, 0.4, 0.5]],
        )

        for variant in [
            "m11r3_dual_residual",
            "m11r3_norm_capped_residual",
            "m11r3_neighbor_transfer",
            "m11r3_target_film",
        ]:
            model = CCFCRec(make_args(variant))
            output = model(attributes, images, 2, m11_features=structural)
            self.assertEqual(tuple(output.shape), (2, 4))
            self.assertTrue(torch.isfinite(output).all())
            self.assertIsNotNone(model._last_m11_residual)
        norm_model = CCFCRec(make_args("m11r3_norm_capped_residual"))
        norm_model(attributes, images, 2, m11_features=structural * structural[:, :1])
        base_hidden = norm_model.gen_layer1(
            torch.cat(norm_model.encode_content_components(attributes, images, 2, m11_features=structural * structural[:, :1])[1:], dim=1)
        )
        ratio = norm_model._last_m11_residual.norm(dim=1) / base_hidden.norm(dim=1).clamp_min(1e-12)
        self.assertLessEqual(ratio.max().item(), 0.150001)

    def test_m11r3_zero_initialized_transfer_and_film_paths_leave_zero_after_update(self):
        attributes = torch.tensor(
            [[0, 1, -1, -1, -1, -1, -1, -1], [2, -1, -1, -1, -1, -1, -1, -1]],
            dtype=torch.long,
        )
        images = torch.ones((2, 4096), dtype=torch.float32)
        structural = torch.tensor(
            [[1.0, 0.8, 0.7, 0.6, 0.8, 0.9], [0.0, 0.4, 0.3, 0.2, 0.4, 0.5]],
        )

        for variant in ["m11r3_neighbor_transfer", "m11r3_target_film"]:
            model = CCFCRec(make_args(variant))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            output = model(attributes, images, 2, m11_features=structural)
            torch.testing.assert_close(model._last_m11_residual, torch.zeros_like(model._last_m11_residual))
            output.square().mean().backward()
            optimizer.step()
            model(attributes, images, 2, m11_features=structural)
            self.assertGreater(model._last_m11_residual.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
