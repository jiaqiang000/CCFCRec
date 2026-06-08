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

from model import CCFCRec, build_run_config, validate_method_args


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
        seed=43,
        num_workers=0,
    )


class CategoryConfidenceInputTest(unittest.TestCase):
    def test_baseline_generator_input_dim_stays_unchanged(self):
        model = CCFCRec(make_args("baseline"))

        self.assertEqual(model.gen_layer1.in_features, 8)
        self.assertFalse(hasattr(model, "category_conf_embedding"))

    def test_legacy_args_without_variant_fields_load_as_baseline(self):
        args = make_args("baseline")
        delattr(args, "method_variant")
        delattr(args, "category_conf_dim")
        delattr(args, "category_conf_max_count")

        model = CCFCRec(args)

        self.assertEqual(model.gen_layer1.in_features, 8)
        self.assertFalse(hasattr(model, "category_conf_embedding"))

    def test_category_conf_input_extends_generator_input(self):
        model = CCFCRec(make_args("category_conf_input"))

        self.assertEqual(model.gen_layer1.in_features, 13)
        self.assertEqual(model.category_conf_embedding.num_embeddings, 4)
        self.assertEqual(model.category_conf_embedding.embedding_dim, 3)
        self.assertFalse(hasattr(model, "category_fusion_gate"))

    def test_category_conf_fusion_gate_uses_same_generator_input_with_gate(self):
        model = CCFCRec(make_args("category_conf_fusion_gate"))

        self.assertEqual(model.gen_layer1.in_features, 13)
        self.assertEqual(model.category_conf_embedding.num_embeddings, 4)
        self.assertEqual(model.category_conf_embedding.embedding_dim, 3)
        self.assertEqual(model.category_fusion_gate.in_features, 5)
        self.assertEqual(model.category_fusion_gate.out_features, 1)

    def test_category_conf_bins_and_scalars_use_clamped_count(self):
        model = CCFCRec(make_args("category_conf_input"))
        attributes = torch.full((4, 8), -1.0)
        attributes[1, :2] = 1.0
        attributes[2, :4] = 1.0
        attributes[3, :6] = 1.0

        bins = model.build_category_conf_bins(attributes)
        features = model.build_category_conf_features(attributes)

        self.assertEqual(bins.tolist(), [0, 1, 2, 3])
        self.assertEqual(features.shape, (4, 5))
        expected_log_norm = torch.tensor(
            [
                0.0,
                math.log1p(2) / math.log1p(5),
                math.log1p(4) / math.log1p(5),
                1.0,
            ],
            dtype=features.dtype,
        )
        expected_density = torch.tensor([0.0, 0.4, 0.8, 1.0], dtype=features.dtype)
        torch.testing.assert_close(features[:, -2].detach().cpu(), expected_log_norm)
        torch.testing.assert_close(features[:, -1].detach().cpu(), expected_density)

    def test_category_conf_fusion_gate_starts_neutral(self):
        model = CCFCRec(make_args("category_conf_fusion_gate"))
        attributes = torch.full((2, 8), -1.0)
        attributes[0, :3] = 1.0
        attributes[1, :5] = 1.0
        attr_emb = torch.randn(2, 4)
        image_proj = torch.randn(2, 4)

        features = model.build_category_conf_features(attributes)
        gated_attr, gated_image = model.apply_category_fusion_gate(attr_emb, image_proj, features)

        torch.testing.assert_close(gated_attr, attr_emb)
        torch.testing.assert_close(gated_image, image_proj)

    def test_category_conf_fusion_gate_scales_attr_and_image_oppositely(self):
        model = CCFCRec(make_args("category_conf_fusion_gate"))
        with torch.no_grad():
            model.category_fusion_gate.bias.fill_(math.log(4.0))
        attributes = torch.full((1, 8), -1.0)
        attributes[0, :3] = 1.0
        attr_emb = torch.ones(1, 4)
        image_proj = torch.ones(1, 4)

        features = model.build_category_conf_features(attributes)
        gated_attr, gated_image = model.apply_category_fusion_gate(attr_emb, image_proj, features)

        # sigmoid(log(4)) = 0.8, centered gate = 0.6, gate_scale = 0.5.
        torch.testing.assert_close(gated_attr, torch.full((1, 4), 1.3))
        torch.testing.assert_close(gated_image, torch.full((1, 4), 0.7))

    def test_forward_uses_shared_content_encoder(self):
        model = CCFCRec(make_args("category_conf_input"))
        attributes = torch.full((2, 8), -1.0)
        attributes[0, :3] = 1.0
        attributes[1, :5] = 1.0
        image_features = torch.randn(2, 4096)

        q_v_c, attr_emb, image_proj = model.encode_content_components(attributes, image_features, 2)
        forward_q = model(attributes, image_features, 2)

        self.assertEqual(q_v_c.shape, (2, 4))
        self.assertEqual(attr_emb.shape, (2, 4))
        self.assertEqual(image_proj.shape, (2, 4))
        torch.testing.assert_close(forward_q, q_v_c)

    def test_run_config_records_architecture(self):
        args = make_args("category_conf_input")
        model = CCFCRec(args)

        config = build_run_config(args, model)

        self.assertEqual(config["method_variant"], "category_conf_input")
        self.assertEqual(config["category_conf_dim"], 3)
        self.assertEqual(config["category_conf_max_count"], 5)
        self.assertEqual(config["category_gate_scale"], 0.5)
        self.assertEqual(config["category_bin_count"], 4)
        self.assertEqual(config["category_fusion_gate_output_dim"], 0)
        self.assertEqual(config["gen_layer1_input_dim"], 13)
        self.assertEqual(config["seed"], 43)

    def test_run_config_records_category_conf_fusion_gate(self):
        args = make_args("category_conf_fusion_gate")
        model = CCFCRec(args)

        config = build_run_config(args, model)

        self.assertEqual(config["method_variant"], "category_conf_fusion_gate")
        self.assertEqual(config["category_conf_dim"], 3)
        self.assertEqual(config["category_conf_max_count"], 5)
        self.assertEqual(config["category_gate_scale"], 0.5)
        self.assertEqual(config["category_fusion_gate_output_dim"], 1)
        self.assertEqual(config["gen_layer1_input_dim"], 13)

    def test_category_conf_input_rejects_reweight_flags(self):
        args = make_args("category_conf_input")
        args.reweight_q_bpr = True
        args.reweight_self_contrast = False
        args.reweight_contrast = False

        with self.assertRaises(ValueError):
            validate_method_args(args)

    def test_category_conf_fusion_gate_rejects_reweight_flags(self):
        args = make_args("category_conf_fusion_gate")
        args.reweight_q_bpr = False
        args.reweight_self_contrast = True
        args.reweight_contrast = False

        with self.assertRaises(ValueError):
            validate_method_args(args)

    def test_category_conf_fusion_gate_rejects_invalid_gate_scale(self):
        args = make_args("category_conf_fusion_gate")
        args.category_gate_scale = 1.5

        with self.assertRaises(ValueError):
            validate_method_args(args)


if __name__ == "__main__":
    unittest.main()
