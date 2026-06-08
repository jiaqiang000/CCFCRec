import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ["CCFCREC_DEVICE"] = "cpu"

REPO_ROOT = Path(__file__).resolve().parents[1]
AMAZON_VG_DIR = REPO_ROOT / "Amazon VG"
sys.path.insert(0, str(AMAZON_VG_DIR))

import torch

from model import (
    CCFCRec,
    build_adaptive_qbpr_weights,
    build_run_config,
    validate_method_args,
)
from support import compute_history_category_support_confidence


def make_args(method_variant="baseline"):
    return SimpleNamespace(
        attr_num=8,
        attr_present_dim=4,
        implicit_dim=4,
        cat_implicit_dim=4,
        user_number=3,
        item_number=5,
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
        seed=43,
        num_workers=0,
    )


class AdaptiveConfidenceQbprTest(unittest.TestCase):
    def test_history_support_excludes_current_item_and_uses_overlap(self):
        item_category_sets = {
            "target": frozenset({1, 2}),
            "hist_a": frozenset({2, 4}),
            "hist_b": frozenset({5}),
        }

        confidence = compute_history_category_support_confidence(
            current_item="target",
            history_items=["target", "hist_a", "hist_b"],
            item_category_sets=item_category_sets,
            adaptive_history_max_count=20,
        )

        expected_overlap = 1 / 2
        expected_history = math.log1p(2) / math.log1p(20)
        self.assertAlmostEqual(confidence, expected_overlap * expected_history)

    def test_history_support_zero_without_non_current_history(self):
        confidence = compute_history_category_support_confidence(
            current_item="target",
            history_items=["target"],
            item_category_sets={"target": frozenset({1, 2})},
            adaptive_history_max_count=20,
        )

        self.assertEqual(confidence, 0.0)

    def test_adaptive_qbpr_weights_only_raise_supported_weak_items(self):
        args = make_args("adaptive_conf_qbpr")
        item_genres = torch.full((4, 8), -1.0)
        item_genres[0, :2] = 1.0
        item_genres[1, :3] = 1.0
        item_genres[2, :4] = 1.0
        item_genres[3, :6] = 1.0
        support_confidence = torch.tensor([0.5, 0.0, 1.0, 1.0])

        weights = build_adaptive_qbpr_weights(item_genres, support_confidence, args)

        raw = torch.tensor([1.5, 1.0, 1.0, 1.0])
        expected = raw / raw.mean()
        torch.testing.assert_close(weights, expected)

    def test_adaptive_qbpr_keeps_baseline_model_architecture(self):
        model = CCFCRec(make_args("adaptive_conf_qbpr"))

        self.assertEqual(model.gen_layer1.in_features, 8)
        self.assertFalse(hasattr(model, "category_conf_embedding"))

    def test_adaptive_qbpr_rejects_legacy_reweight_flags(self):
        args = make_args("adaptive_conf_qbpr")
        args.reweight_q_bpr = True

        with self.assertRaises(ValueError):
            validate_method_args(args)

    def test_run_config_records_adaptive_params(self):
        args = make_args("adaptive_conf_qbpr")
        model = CCFCRec(args)

        config = build_run_config(args, model)

        self.assertEqual(config["method_variant"], "adaptive_conf_qbpr")
        self.assertEqual(config["weak_cat_threshold"], 3)
        self.assertEqual(config["adaptive_loss_alpha"], 1.0)
        self.assertEqual(config["adaptive_history_max_count"], 20)


if __name__ == "__main__":
    unittest.main()
