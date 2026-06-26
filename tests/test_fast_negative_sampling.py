import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

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


class OriginalNpChoiceNegativeSamplingTest(unittest.TestCase):
    def test_matches_pre_optimization_raw_np_choice_path(self):
        class RecordingChoiceRng:
            def __init__(self):
                self.calls = []

            def choice(self, candidates, size, replace):
                candidates = list(candidates)
                self.calls.append((candidates, size, replace))
                return np.asarray(
                    [candidates[index % len(candidates)] for index in range(size)],
                    dtype=object,
                )

        item_set = {"item_a", "item_b", "item_c", "item_d"}
        positive_raw_items = ["item_b", "item_d"]
        item_serialize_dict = {
            "item_a": 0,
            "item_b": 1,
            "item_c": 2,
            "item_d": 3,
        }
        rng = RecordingChoiceRng()

        positive_items, negative_items = support.sample_original_np_choice_items(
            item_set=item_set,
            item_serialize_dict=item_serialize_dict,
            positive_raw_items=positive_raw_items,
            positive_number=3,
            negative_sample_size=5,
            rng=rng,
        )

        expected_negative_candidates = list(item_set - set(positive_raw_items))
        self.assertEqual(rng.calls[0], (list(positive_raw_items), 3, True))
        self.assertEqual(rng.calls[1], (expected_negative_candidates, 5, True))
        self.assertEqual(positive_items.tolist(), [1, 3, 1])
        self.assertEqual(
            negative_items.tolist(),
            [
                item_serialize_dict[expected_negative_candidates[index % len(expected_negative_candidates)]]
                for index in range(5)
            ],
        )

    def test_rating_dataset_can_use_original_np_choice_mode(self):
        original_user_builder = support.build_user_item_interaction_dict
        original_item_builder = support.build_item_user_interaction_dict
        support.build_user_item_interaction_dict = lambda *args, **kwargs: {
            "user_1": ["item_a", "item_b"],
            "user_2": ["item_c"],
        }
        support.build_item_user_interaction_dict = lambda *args, **kwargs: {
            "item_a": ["user_1"],
            "item_b": ["user_1"],
            "item_c": ["user_2"],
        }
        try:
            dataset = support.RatingDataset(
                train_csv=pd.DataFrame(
                    {
                        "reviewerID": ["user_1", "user_1", "user_2"],
                        "asin": ["item_a", "item_b", "item_c"],
                        "rating": [1, 1, 1],
                        "neg_user": ["user_2", "user_2", "user_1"],
                    }
                ),
                img_features={
                    "item_a": np.asarray([1.0, 0.0], dtype=np.float32),
                    "item_b": np.asarray([0.0, 1.0], dtype=np.float32),
                    "item_c": np.asarray([1.0, 1.0], dtype=np.float32),
                },
                genres={
                    "item_a": [0],
                    "item_b": [1],
                    "item_c": [0, 1],
                },
                category_num=2,
                user_serialize_dict={"user_1": 0, "user_2": 1},
                positive_number=2,
                negative_number=3,
                negative_sampling_mode="original_np_choice",
            )
            _, _, _, positive_items, negative_item_list, self_neg_list, _ = dataset[0]
        finally:
            support.build_user_item_interaction_dict = original_user_builder
            support.build_item_user_interaction_dict = original_item_builder

        item_c = dataset.item_serialize_dict["item_c"]
        self.assertEqual(positive_items.shape, (2,))
        self.assertEqual(negative_item_list.shape, (2, 3))
        self.assertEqual(self_neg_list.shape, (3,))
        self.assertTrue(set(positive_items.tolist()).issubset({
            dataset.item_serialize_dict["item_a"],
            dataset.item_serialize_dict["item_b"],
        }))
        self.assertEqual(set(negative_item_list.flatten().tolist()), {item_c})
        self.assertEqual(set(self_neg_list.tolist()), {item_c})


if __name__ == "__main__":
    unittest.main()
