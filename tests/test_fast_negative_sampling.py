import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ["CCFCREC_DEVICE"] = "cpu"

REPO_ROOT = Path(__file__).resolve().parents[1]
AMAZON_VG_DIR = REPO_ROOT / "Amazon VG"
sys.path.insert(0, str(AMAZON_VG_DIR))

from support import sample_negative_serial_items


class FastNegativeSamplingTest(unittest.TestCase):
    def test_samples_requested_count_without_positive_items(self):
        samples = sample_negative_serial_items(
            item_number=5,
            excluded_items={1, 3},
            sample_size=200,
            rng=np.random.default_rng(7),
        )

        self.assertEqual(samples.shape, (200,))
        self.assertTrue(np.issubdtype(samples.dtype, np.integer))
        self.assertFalse(set(samples.tolist()).intersection({1, 3}))
        self.assertTrue(set(samples.tolist()).issubset({0, 2, 4}))

    def test_raises_when_no_negative_candidate_exists(self):
        with self.assertRaises(ValueError):
            sample_negative_serial_items(
                item_number=3,
                excluded_items={0, 1, 2},
                sample_size=1,
                rng=np.random.default_rng(7),
            )


if __name__ == "__main__":
    unittest.main()
