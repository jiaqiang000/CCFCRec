import numpy as np
import pandas as pd

from analyze_amazon_vg_cicp_train_only_signal import (
    build_fallacy_scan,
    empirical_percentile,
    normalize_interactions,
    reliability_weight,
    row_cosine,
    support_bucket,
)


def test_reliability_weight_is_bounded_and_monotonic() -> None:
    values = reliability_weight(np.asarray([0, 1, 2, 5, 20, 100], dtype=float))

    assert values[0] == 0.0
    assert np.all(np.diff(values) >= 0)
    assert values[4] == 1.0
    assert values[5] == 1.0


def test_normalized_interactions_use_two_sided_degree_scaling() -> None:
    from scipy.sparse import csr_matrix

    matrix = csr_matrix(np.asarray([[1.0, 1.0], [0.0, 1.0]]))
    result = normalize_interactions(matrix).toarray()

    assert np.isclose(result[0, 0], 1 / np.sqrt(2))
    assert np.isclose(result[0, 1], 0.5)
    assert np.isclose(result[1, 1], 1 / np.sqrt(2))


def test_row_cosine_and_percentile_are_full_coverage() -> None:
    left = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    right = np.asarray([[1.0, 0.0], [1.0, 0.0]])

    assert np.allclose(row_cosine(left, right), [1.0, 0.0])
    percentiles = empirical_percentile(np.asarray([1.0, 2.0, 3.0]), np.asarray([0.0, 2.0, 4.0]))
    assert np.allclose(percentiles, [0.0, 2 / 3, 1.0])
    assert np.isfinite(percentiles).all()


def test_support_buckets_are_complementary() -> None:
    buckets = support_bucket(pd.Series([1, 2, 3, 4, 10, 11, 100]))

    assert buckets.tolist() == [
        "support_1",
        "support_2_3",
        "support_2_3",
        "support_4_10",
        "support_4_10",
        "support_11_plus",
        "support_11_plus",
    ]


def test_fallacy_scan_covers_all_eleven_types() -> None:
    scan = build_fallacy_scan()

    assert scan["fallacy_id"].tolist() == list(range(1, 12))
    assert set(scan["status"]).issubset({"PASS", "CAUTION", "NOT_APPLICABLE"})
