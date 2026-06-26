import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ["CCFCREC_DEVICE"] = "cpu"

REPO_ROOT = Path(__file__).resolve().parents[1]
AMAZON_VG_DIR = REPO_ROOT / "Amazon VG"
sys.path.insert(0, str(AMAZON_VG_DIR))

import support
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


class LegacyCachedNegativeSamplingTest(unittest.TestCase):
    def test_samples_from_legacy_user_negative_candidate_set(self):
        self.assertTrue(hasattr(support, "LegacyCachedNegativeSampler"))
        sampler = support.LegacyCachedNegativeSampler(
            item_set={"item_a", "item_b", "item_c", "item_d"},
            item_serialize_dict={
                "item_a": 0,
                "item_b": 1,
                "item_c": 2,
                "item_d": 3,
            },
            user_item_interaction_dict={
                "user_1": ["item_b", "item_d"],
            },
            max_cache_size=8,
        )

        samples = sampler.sample("user_1", sample_size=200, rng=np.random.default_rng(7))

        self.assertEqual(samples.shape, (200,))
        self.assertTrue(np.issubdtype(samples.dtype, np.integer))
        self.assertTrue(set(samples.tolist()).issubset({0, 2}))
        self.assertFalse(set(samples.tolist()).intersection({1, 3}))

    def test_maps_compact_candidate_indices_without_building_full_candidate_array(self):
        class FixedIntegerRng:
            def integers(self, low, high, size):
                self.assertEqual((low, high, size), (0, 3, 4))
                return np.asarray([0, 1, 2, 1], dtype=np.int64)

            def assertEqual(self, left, right):
                assert left == right

        sampler = support.LegacyCachedNegativeSampler(
            item_set={"item_a", "item_b", "item_c", "item_d", "item_e"},
            item_serialize_dict={
                "item_a": 0,
                "item_b": 1,
                "item_c": 2,
                "item_d": 3,
                "item_e": 4,
            },
            user_item_interaction_dict={
                "user_1": ["item_b", "item_d"],
            },
            max_cache_size=1,
        )

        samples = sampler.sample("user_1", sample_size=4, rng=FixedIntegerRng())

        self.assertEqual(samples.tolist(), [0, 2, 4, 2])
        self.assertFalse(hasattr(sampler, "_candidate_cache"))


if __name__ == "__main__":
    unittest.main()
