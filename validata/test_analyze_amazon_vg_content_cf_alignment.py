#!/usr/bin/env python3
"""
CCFCRec Amazon-VG content-CF alignment 诊断脚本的最小测试。

测试只覆盖稳定口径：
1. item 与用户历史中心的 cosine 汇总；
2. 用户历史内容中心是否按 train_rating.csv 中的历史 item 均值构造。
"""

import math

import numpy as np
import pandas as pd

from analyze_amazon_vg_content_cf_alignment import (
    build_user_content_centroids,
    cosine_summary,
    mean_history_interaction_count,
)


def test_cosine_summary_uses_only_users_with_history() -> None:
    item_vector = np.array([1.0, 0.0], dtype=np.float32)
    centroids = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    interaction_counts = np.array([2, 1, 0], dtype=np.int32)

    result = cosine_summary(item_vector, centroids, [0, 1, 2], interaction_counts)

    assert result["known_user_count"] == 2
    assert math.isclose(result["cosine_mean"], 0.5)
    assert math.isclose(result["cosine_max"], 1.0)


def test_build_user_content_centroids_averages_train_history_vectors() -> None:
    train_df = pd.DataFrame(
        {
            "reviewerID": ["u1", "u1", "u2"],
            "asin": ["a", "b", "c"],
        }
    )
    user_ser_dict = {"u1": 0, "u2": 1}
    asin_to_pos = {"a": 0, "b": 1, "c": 2}
    q_vectors = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    attr_vectors = q_vectors + 10.0
    img_vectors = q_vectors + 20.0

    q_centroids, attr_centroids, img_centroids, counts = build_user_content_centroids(
        train_df,
        user_ser_dict,
        asin_to_pos,
        q_vectors,
        attr_vectors,
        img_vectors,
    )

    np.testing.assert_allclose(q_centroids[0], np.array([2.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(q_centroids[1], np.array([0.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(attr_centroids[0], np.array([12.0, 10.0], dtype=np.float32))
    np.testing.assert_allclose(img_centroids[1], np.array([20.0, 22.0], dtype=np.float32))
    assert counts.tolist() == [2, 1]


def test_mean_history_interaction_count_filters_unknown_or_empty_users() -> None:
    interaction_counts = np.array([2, 4, 0], dtype=np.int32)

    result = mean_history_interaction_count([0, 1, 2, 99], interaction_counts)

    assert math.isclose(result, 3.0)
