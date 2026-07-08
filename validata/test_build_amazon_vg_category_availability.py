#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability 变量构造脚本的最小测试。

测试只覆盖后续实验依赖的稳定口径：
1. raw category 字符串解析；
2. asin metadata 去重规则；
3. 只输出 train/validate/test 中出现过的 item；
4. train split 拟合 support / popularity / group threshold 后应用到 valid/test。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from build_amazon_vg_category_availability import (
    build_category_availability,
    deduplicate_metadata,
    split_category_tokens,
)


def test_split_category_tokens() -> None:
    assert split_category_tokens("Video Games,PC,Games") == ["Video Games", "PC", "Games"]
    assert split_category_tokens(" Video Games, , Accessories ") == ["Video Games", "Accessories"]
    assert split_category_tokens("") == []
    assert split_category_tokens(None) == []


def test_deduplicate_metadata_prefers_more_categories_then_longer_title() -> None:
    metadata = pd.DataFrame(
        [
            {"asin": "a", "title": "Short", "category": "Video Games"},
            {"asin": "a", "title": "Longer descriptive title", "category": "Video Games"},
            {"asin": "a", "title": "Tiny", "category": "Video Games,PC,Action"},
        ]
    )

    result = deduplicate_metadata(metadata)

    assert result.shape[0] == 1
    assert result.iloc[0]["asin"] == "a"
    assert result.iloc[0]["category"] == "Video Games,PC,Action"


def test_build_category_availability_uses_union_items_and_train_fit_only() -> None:
    metadata = pd.DataFrame(
        [
            {"asin": "train_a", "title": "Fast action game", "category": "Video Games,PC,Action"},
            {"asin": "train_b", "title": "Controller accessory", "category": "Video Games,Accessories"},
            {"asin": "valid_c", "title": "Indie PC game", "category": "Video Games,PC,Indie"},
            {"asin": "valid_c", "title": "Short duplicate", "category": "Video Games"},
            {"asin": "test_d", "title": "", "category": "Video Games"},
            {"asin": "metadata_only", "title": "Unused", "category": "Video Games,PC"},
        ]
    )
    train = pd.DataFrame(
        [
            {"reviewerID": "u1", "asin": "train_a", "rating": 1},
            {"reviewerID": "u2", "asin": "train_a", "rating": 1},
            {"reviewerID": "u1", "asin": "train_b", "rating": 1},
        ]
    )
    validate = pd.DataFrame([{"reviewerID": "u3", "asin": "valid_c", "rating": 1}])
    test = pd.DataFrame([{"reviewerID": "u4", "asin": "test_d", "rating": 1}])
    image_features = {
        "train_a": np.asarray([3.0, 4.0], dtype=np.float32),
        "valid_c": np.asarray([1.0, 2.0], dtype=np.float32),
    }

    result, meta = build_category_availability(
        metadata_df=metadata,
        train_df=train,
        validate_df=validate,
        test_df=test,
        image_features=image_features,
        fit_scope="train_metadata_fit",
    )

    assert set(result["raw_asin"]) == {"train_a", "train_b", "valid_c", "test_d"}
    assert "metadata_only" not in set(result["raw_asin"])
    assert meta["fit_scope"] == "train_metadata_fit"
    assert meta["stable_key"] == "raw_asin"

    valid_row = result[result["raw_asin"].eq("valid_c")].iloc[0]
    test_row = result[result["raw_asin"].eq("test_d")].iloc[0]
    train_a = result[result["raw_asin"].eq("train_a")].iloc[0]

    assert valid_row["split"] == "validate"
    assert valid_row["category_tokens"] == "Video Games|PC|Indie"
    assert valid_row["A_collab_train_token_item_support_min"] == 0
    assert valid_row["S_train_token_item_support_min"] == 0
    assert test_row["A_disc_generic_token_ratio"] == 1.0
    assert train_a["R_has_image_feature"] == 1
    assert math.isclose(train_a["R_image_norm"], 5.0)

    assert result["s_cat"].between(0.0, 1.0).all()
    assert set(result["s_cat_group"]).issubset({"s_cat_weak", "s_cat_mid", "s_cat_strong"})
    assert set(result[result["split"].eq("train")]["s_cat_group"])
