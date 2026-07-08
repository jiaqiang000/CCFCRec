import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ["CCFCREC_DEVICE"] = "cpu"

REPO_ROOT = Path(__file__).resolve().parents[1]
AMAZON_VG_DIR = REPO_ROOT / "Amazon VG"
sys.path.insert(0, str(AMAZON_VG_DIR))

import pandas as pd
import torch

from model import (
    CCFCRec,
    TASK4_METHOD_VARIANTS,
    build_run_config,
    build_task4_pair_margin_targets_from_profile,
    build_task4_item_weights_from_profile,
    task4_pair_margin_loss,
    uses_task4_item_weights,
    uses_task4_pair_margin,
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
                "s_cat_v3": 0.20,
                "high_acat_flag": False,
                "train_safe_hard_proxy_score": 0.10,
                "train_safe_hard_proxy_high_flag": False,
                "high_acat_train_safe_hard_flag": False,
                "eval_baseline_hard_flag": True,
                "baseline_ndcg@20": 0.0,
            },
            {
                "raw_asin": "b",
                "split": "train",
                "cat_count_bin": "cat_count_4",
                "category_count": 4,
                "RSP_group": "RSP_low",
                "s_cat_v3": 0.30,
                "high_acat_flag": True,
                "train_safe_hard_proxy_score": 0.20,
                "train_safe_hard_proxy_high_flag": False,
                "high_acat_train_safe_hard_flag": False,
                "eval_baseline_hard_flag": False,
                "baseline_ndcg@20": 0.5,
            },
            {
                "raw_asin": "c",
                "split": "train",
                "cat_count_bin": "cat_count_5_plus",
                "category_count": 5,
                "RSP_group": "RSP_low",
                "s_cat_v3": 0.90,
                "high_acat_flag": True,
                "train_safe_hard_proxy_score": 0.80,
                "train_safe_hard_proxy_high_flag": True,
                "high_acat_train_safe_hard_flag": True,
                "eval_baseline_hard_flag": True,
                "baseline_ndcg@20": 0.0,
            },
            {
                "raw_asin": "d",
                "split": "train",
                "cat_count_bin": "cat_count_4",
                "category_count": 4,
                "RSP_group": "RSP_mid",
                "s_cat_v3": 0.40,
                "high_acat_flag": False,
                "train_safe_hard_proxy_score": 0.30,
                "train_safe_hard_proxy_high_flag": True,
                "high_acat_train_safe_hard_flag": False,
                "eval_baseline_hard_flag": False,
                "baseline_ndcg@20": 0.4,
            },
        ]
    )


class Task4AvailabilityWeightsTest(unittest.TestCase):
    def test_task4_variants_keep_baseline_architecture(self):
        for method_variant in TASK4_METHOD_VARIANTS:
            model = CCFCRec(make_args(method_variant))
            self.assertEqual(model.gen_layer1.in_features, 8)
            self.assertFalse(hasattr(model, "category_conf_embedding"))

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


if __name__ == "__main__":
    unittest.main()
