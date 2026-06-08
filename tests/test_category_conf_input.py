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
        self.assertEqual(config["category_bin_count"], 4)
        self.assertEqual(config["gen_layer1_input_dim"], 13)
        self.assertEqual(config["seed"], 43)

    def test_category_conf_input_rejects_reweight_flags(self):
        args = make_args("category_conf_input")
        args.reweight_q_bpr = True
        args.reweight_self_contrast = False
        args.reweight_contrast = False

        with self.assertRaises(ValueError):
            validate_method_args(args)


if __name__ == "__main__":
    unittest.main()
