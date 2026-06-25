import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

os.environ["CCFCREC_DEVICE"] = "cpu"

REPO_ROOT = Path(__file__).resolve().parents[1]
AMAZON_VG_DIR = REPO_ROOT / "Amazon VG"
sys.path.insert(0, str(AMAZON_VG_DIR))

from support import build_item_feature_tensors
from test import Validate


class StaticItemFeatureTensorTest(unittest.TestCase):
    def test_builds_category_and_image_tensors_by_serial_item_id(self):
        category_tensor, image_tensor = build_item_feature_tensors(
            item_serialize_dict={"item_a": 0, "item_b": 1},
            img_features={
                "item_a": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                "item_b": np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
            },
            genres={
                "item_a": [1, 3],
                "item_b": [0],
            },
            category_num=4,
        )

        self.assertEqual(category_tensor.dtype, torch.int8)
        self.assertEqual(image_tensor.dtype, torch.float32)
        self.assertEqual(category_tensor.tolist(), [[-1, 1, -1, 1], [1, -1, -1, -1]])
        self.assertTrue(torch.equal(image_tensor, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])))


class BatchedValidateTest(unittest.TestCase):
    def test_start_validate_batches_items_without_changing_metrics(self):
        class IdentityContentModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.user_embedding = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.2, 0.2, 0.0],
                        ]
                    )
                )
                self.forward_batch_sizes = []

            def forward(self, attribute, image_feature, batch_size):
                self.forward_batch_sizes.append(int(batch_size))
                return image_feature

        with tempfile.TemporaryDirectory() as tmp_dir:
            validate_path = Path(tmp_dir) / "validate_rating.csv"
            validate_path.write_text(
                "reviewerID,asin,rating\n"
                "user_a,item_a,1\n"
                "user_b,item_b,1\n",
                encoding="utf-8",
            )

            validator = Validate(
                validate_csv=str(validate_path),
                user_serialize_dict={"user_a": 0, "user_b": 1},
                img={
                    "item_a": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                    "item_b": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                },
                genres={"item_a": [0], "item_b": [1]},
                category_num=3,
                batch_size=2,
            )

            model = IdentityContentModel()
            metrics = validator.start_validate(model)

        self.assertEqual(metrics, (0.2, 0.1, 0.05, 1.0, 1.0, 1.0))
        self.assertEqual(model.forward_batch_sizes, [2])


if __name__ == "__main__":
    unittest.main()
