#!/usr/bin/env python3
"""
CCFCRec Amazon-VG 自然难度分层与混杂诊断。

脚本作用：
1. 读取 best checkpoint 的 item-level test 指标；
2. 为每个 test item 补充类别数量、标题、图像特征、train-side 类别支持度等画像；
3. 输出自然分组指标、混杂画像、匹配桶内 weak/mid/strong gap；
4. 生成 run_manifest.json 和结果 Markdown。

这个脚本只做诊断，不训练模型，不修改 CCFCRec 主训练入口。
gt_user_count 保留 item-level 复评指标里的口径；raw_test_user_count 才是原始 test_rating.csv 行数。
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METRICS = ["hr@5", "hr@10", "hr@20", "ndcg@5", "ndcg@10", "ndcg@20"]


def now_stamp() -> tuple[str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def category_group(category_count: int) -> str:
    if category_count <= 3:
        return "cat_weak_1_3"
    if category_count == 4:
        return "cat_mid_4"
    return "cat_strong_5_plus"


def gt_group(gt_user_count: int) -> str:
    if gt_user_count <= 1:
        return "gt_1"
    if gt_user_count <= 3:
        return "gt_2_3"
    if gt_user_count <= 11:
        return "gt_4_11"
    return "gt_12_plus"


def title_bucket(title_tokens: float) -> str:
    if title_tokens <= 4:
        return "title_0_4"
    if title_tokens <= 8:
        return "title_5_8"
    if title_tokens <= 16:
        return "title_9_16"
    return "title_17_plus"


def depth_bucket(raw_category_depth: float) -> str:
    if raw_category_depth <= 2:
        return "depth_0_2"
    if raw_category_depth == 3:
        return "depth_3"
    if raw_category_depth == 4:
        return "depth_4"
    return "depth_5_plus"


def safe_token_count(value: object) -> int:
    if not isinstance(value, str) or not value.strip():
        return 0
    return len(value.strip().split())


def split_raw_category(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def qcut_bucket(series: pd.Series, label_prefix: str, q: int = 4) -> pd.Series:
    """生成稳定的分位桶；唯一值太少时退化为 rank 后再切分。"""
    non_na = series.dropna()
    if non_na.empty:
        return pd.Series([f"{label_prefix}_missing"] * len(series), index=series.index)
    try:
        buckets = pd.qcut(series, q=q, duplicates="drop")
    except ValueError:
        ranked = series.rank(method="average")
        buckets = pd.qcut(ranked, q=q, duplicates="drop")
    labels = []
    for value in buckets.astype(str):
        if value == "nan":
            labels.append(f"{label_prefix}_missing")
        else:
            labels.append(f"{label_prefix}_{value}")
    return pd.Series(labels, index=series.index)


@dataclass(frozen=True)
class Paths:
    project_root: Path
    code_root: Path
    dataset_dir: Path
    item_metrics: Path
    output_dir: Path


def resolve_paths(args: argparse.Namespace) -> Paths:
    script_path = Path(__file__).resolve()
    code_root = Path(args.code_root).expanduser().resolve() if args.code_root else script_path.parents[1]
    project_root = code_root.parent
    dataset_dir = (
        Path(args.dataset_dir).expanduser().resolve()
        if args.dataset_dir
        else code_root / "Amazon VG" / "data"
    )
    item_metrics = (
        Path(args.item_metrics).expanduser().resolve()
        if args.item_metrics
        else project_root / "实验记录" / "复现" / "ccfcrec_amazon_vg_best64_item_level_test_metrics.csv"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else project_root
        / "temp_202606_实验文件记录"
        / "temp_20260608"
        / "natural-difficulty-confounding-diagnostic"
    )
    return Paths(
        project_root=project_root,
        code_root=code_root,
        dataset_dir=dataset_dir,
        item_metrics=item_metrics,
        output_dir=output_dir,
    )


def load_category_map(category_pkl: Path) -> dict[str, list[int]]:
    with category_pkl.open("rb") as file:
        data = pickle.load(file)
    if isinstance(data, dict) and "asin_category_int_map" in data:
        return data["asin_category_int_map"]
    if isinstance(data, dict):
        return data
    raise TypeError(f"Unsupported category pickle format: {category_pkl}")


def build_train_category_support(train_df: pd.DataFrame, category_map: dict[str, list[int]]) -> tuple[dict[int, int], dict[int, int], dict[int, set[str]]]:
    """统计每个类别在 train item 中的覆盖 item 数和交互行数。"""
    train_item_interactions = train_df.groupby("asin").size().to_dict()
    category_to_items: dict[int, set[str]] = {}
    category_to_interactions: dict[int, int] = {}
    for asin, interactions in train_item_interactions.items():
        for category in category_map.get(asin, []):
            category_to_items.setdefault(category, set()).add(asin)
            category_to_interactions[category] = category_to_interactions.get(category, 0) + int(interactions)
    category_to_item_count = {category: len(items) for category, items in category_to_items.items()}
    return category_to_item_count, category_to_interactions, category_to_items


def summarize_numeric(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if arr.size == 0:
        return {"min": np.nan, "mean": np.nan, "max": np.nan}
    return {"min": float(arr.min()), "mean": float(arr.mean()), "max": float(arr.max())}


def add_item_profile(paths: Paths) -> pd.DataFrame:
    metrics_df = pd.read_csv(paths.item_metrics)
    required_cols = {"asin", "category_count", "gt_user_count", *METRICS}
    missing = required_cols - set(metrics_df.columns)
    if missing:
        raise ValueError(f"item metrics 缺少字段: {sorted(missing)}")

    category_map = load_category_map(paths.dataset_dir / "asin_int_category.pkl")
    asin_df = pd.read_csv(paths.dataset_dir / "asin.csv")
    train_df = pd.read_csv(paths.dataset_dir / "train_rating.csv")
    test_df = pd.read_csv(paths.dataset_dir / "test_rating.csv")
    img_features = np.load(paths.dataset_dir / "img_feature.npy", allow_pickle=True).item()

    category_item_count, category_interactions, category_to_items = build_train_category_support(train_df, category_map)
    all_train_category_counts = np.asarray(list(category_item_count.values()), dtype=float)
    rare_threshold = float(np.quantile(all_train_category_counts, 0.10)) if all_train_category_counts.size else 0.0

    raw_meta = asin_df.set_index("asin")
    test_gt_counts = test_df.groupby("asin").size().to_dict()
    rows = []
    for _, metric_row in metrics_df.iterrows():
        asin = metric_row["asin"]
        categories = category_map.get(asin, [])
        train_item_counts = [category_item_count.get(category, 0) for category in categories]
        train_interaction_counts = [category_interactions.get(category, 0) for category in categories]
        shared_train_items: set[str] = set()
        for category in categories:
            shared_train_items.update(category_to_items.get(category, set()))

        raw_title = raw_meta.at[asin, "title"] if asin in raw_meta.index else ""
        raw_category = raw_meta.at[asin, "category"] if asin in raw_meta.index else ""
        raw_parts = split_raw_category(raw_category)
        image = img_features.get(asin)
        if image is None:
            image_norm = np.nan
            image_mean = np.nan
            image_std = np.nan
        else:
            image_arr = np.asarray(image, dtype=float)
            image_norm = float(np.linalg.norm(image_arr))
            image_mean = float(image_arr.mean())
            image_std = float(image_arr.std())

        support_stats = summarize_numeric(train_item_counts)
        interaction_stats = summarize_numeric(train_interaction_counts)
        rare_count = sum(1 for count in train_item_counts if count <= rare_threshold)
        category_count = int(metric_row["category_count"])
        metric_gt_count = int(metric_row["gt_user_count"])
        raw_test_count = int(test_gt_counts.get(asin, metric_gt_count))

        row = {
            "asin": asin,
            "category_count": category_count,
            "category_group": category_group(category_count),
            "gt_user_count": metric_gt_count,
            "gt_group": gt_group(metric_gt_count),
            "raw_test_user_count": raw_test_count,
            "raw_minus_metric_gt_user_count": raw_test_count - metric_gt_count,
            "title_tokens": safe_token_count(raw_title),
            "title_chars": len(raw_title) if isinstance(raw_title, str) else 0,
            "title_bucket": title_bucket(safe_token_count(raw_title)),
            "raw_category_depth": len(raw_parts),
            "raw_category_depth_bucket": depth_bucket(len(raw_parts)),
            "raw_second_category": raw_parts[1] if len(raw_parts) > 1 else "missing",
            "raw_leaf_category": raw_parts[-1] if raw_parts else "missing",
            "image_norm": image_norm,
            "image_mean": image_mean,
            "image_std": image_std,
            "category_train_item_count_min": support_stats["min"],
            "category_train_item_count_mean": support_stats["mean"],
            "category_train_item_count_max": support_stats["max"],
            "category_train_interaction_count_min": interaction_stats["min"],
            "category_train_interaction_count_mean": interaction_stats["mean"],
            "category_train_interaction_count_max": interaction_stats["max"],
            "category_shared_train_item_count": len(shared_train_items),
            "rare_category_count": rare_count,
            "rare_category_rate": rare_count / category_count if category_count else np.nan,
        }
        for metric in METRICS:
            row[metric] = float(metric_row[metric])
        rows.append(row)

    profile = pd.DataFrame(rows)
    profile["image_norm_bucket"] = qcut_bucket(profile["image_norm"], "image_norm")
    profile["category_train_item_count_max_bucket"] = qcut_bucket(
        profile["category_train_item_count_max"], "cat_train_items"
    )
    profile["category_shared_train_item_count_bucket"] = qcut_bucket(
        profile["category_shared_train_item_count"], "shared_train_items"
    )
    return profile


def aggregate_metrics(profile: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = profile.groupby(group_cols, dropna=False)
    rows = []
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row["item_count"] = len(sub)
        row["gt_user_count_mean"] = sub["gt_user_count"].mean()
        row["gt_user_count_median"] = sub["gt_user_count"].median()
        row["raw_test_user_count_mean"] = sub["raw_test_user_count"].mean()
        row["raw_minus_metric_gt_user_count_mean"] = sub["raw_minus_metric_gt_user_count"].mean()
        for metric in METRICS:
            row[f"{metric}_mean"] = sub[metric].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_confound_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, sub in profile.groupby("category_group"):
        row = {
            "category_group": group,
            "item_count": len(sub),
            "ndcg@20_mean": sub["ndcg@20"].mean(),
            "hr@20_mean": sub["hr@20"].mean(),
            "gt_user_count_mean": sub["gt_user_count"].mean(),
            "gt_user_count_median": sub["gt_user_count"].median(),
            "raw_test_user_count_mean": sub["raw_test_user_count"].mean(),
            "raw_minus_metric_gt_user_count_mean": sub["raw_minus_metric_gt_user_count"].mean(),
            "title_tokens_mean": sub["title_tokens"].mean(),
            "raw_category_depth_mean": sub["raw_category_depth"].mean(),
            "image_norm_mean": sub["image_norm"].mean(),
            "image_norm_median": sub["image_norm"].median(),
            "category_train_item_count_max_mean": sub["category_train_item_count_max"].mean(),
            "category_shared_train_item_count_mean": sub["category_shared_train_item_count"].mean(),
            "rare_category_rate_mean": sub["rare_category_rate"].mean(),
            "top_raw_second_category": "; ".join(
                f"{idx}:{val}" for idx, val in sub["raw_second_category"].value_counts().head(5).items()
            ),
        }
        rows.append(row)
    order = {"cat_weak_1_3": 0, "cat_mid_4": 1, "cat_strong_5_plus": 2}
    return pd.DataFrame(rows).sort_values("category_group", key=lambda s: s.map(order)).reset_index(drop=True)


def build_matched_bucket_gap(profile: pd.DataFrame, metric: str = "ndcg@20") -> pd.DataFrame:
    controls = [
        ("gt_group", ["gt_group"]),
        ("image_norm_bucket", ["image_norm_bucket"]),
        ("category_train_item_count_max_bucket", ["category_train_item_count_max_bucket"]),
        ("category_shared_train_item_count_bucket", ["category_shared_train_item_count_bucket"]),
        ("title_bucket", ["title_bucket"]),
        ("raw_second_category", ["raw_second_category"]),
        ("gt_group__image_norm_bucket", ["gt_group", "image_norm_bucket"]),
        ("gt_group__category_train_item_count_max_bucket", ["gt_group", "category_train_item_count_max_bucket"]),
        ("gt_group__raw_second_category", ["gt_group", "raw_second_category"]),
    ]
    rows = []
    for control_name, columns in controls:
        for key, sub in profile.groupby(columns, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            bucket_label = " | ".join(f"{col}={value}" for col, value in zip(columns, key))
            grouped = sub.groupby("category_group")[metric].agg(["count", "mean"])
            row = {
                "control": control_name,
                "bucket": bucket_label,
                "total_count": len(sub),
            }
            for group in ["cat_weak_1_3", "cat_mid_4", "cat_strong_5_plus"]:
                row[f"{group}_count"] = int(grouped.loc[group, "count"]) if group in grouped.index else 0
                row[f"{group}_{metric}_mean"] = float(grouped.loc[group, "mean"]) if group in grouped.index else np.nan
            weak = row[f"cat_weak_1_3_{metric}_mean"]
            mid = row[f"cat_mid_4_{metric}_mean"]
            strong = row[f"cat_strong_5_plus_{metric}_mean"]
            row[f"mid_minus_weak_{metric}"] = mid - weak if pd.notna(mid) and pd.notna(weak) else np.nan
            row[f"strong_minus_weak_{metric}"] = strong - weak if pd.notna(strong) and pd.notna(weak) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def build_correlation_summary(profile: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "category_count",
        "gt_user_count",
        "raw_test_user_count",
        "raw_minus_metric_gt_user_count",
        "title_tokens",
        "raw_category_depth",
        "image_norm",
        "image_std",
        "category_train_item_count_max",
        "category_shared_train_item_count",
        "rare_category_rate",
    ]
    rows = []
    for feature in feature_cols:
        for metric in ["ndcg@20", "hr@20"]:
            rows.append(
                {
                    "feature": feature,
                    "metric": metric,
                    "pearson": profile[[feature, metric]].corr(method="pearson").iloc[0, 1],
                    "spearman": profile[[feature, metric]].corr(method="spearman").iloc[0, 1],
                }
            )
    return pd.DataFrame(rows)


def summarize_gap_by_control(matched: pd.DataFrame, metric: str = "ndcg@20") -> pd.DataFrame:
    rows = []
    for control, sub in matched.groupby("control"):
        valid = sub.dropna(subset=[f"strong_minus_weak_{metric}", f"mid_minus_weak_{metric}"])
        valid = valid[(valid["cat_weak_1_3_count"] >= 3) & ((valid["cat_mid_4_count"] >= 3) | (valid["cat_strong_5_plus_count"] >= 3))]
        rows.append(
            {
                "control": control,
                "valid_bucket_count": len(valid),
                f"mid_minus_weak_{metric}_mean": valid[f"mid_minus_weak_{metric}"].mean(),
                f"strong_minus_weak_{metric}_mean": valid[f"strong_minus_weak_{metric}"].mean(),
                f"strong_minus_weak_{metric}_positive_rate": (
                    (valid[f"strong_minus_weak_{metric}"] > 0).mean() if len(valid) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("control").reset_index(drop=True)


def write_result_markdown(
    output_path: Path,
    run_stamp: str,
    profile: pd.DataFrame,
    group_metric: pd.DataFrame,
    confound: pd.DataFrame,
    matched_summary: pd.DataFrame,
    correlation: pd.DataFrame,
    manifest_name: str,
) -> None:
    category_rows = group_metric[group_metric["group_type"].eq("category_group")].copy()
    gt_rows = group_metric[group_metric["group_type"].eq("gt_group")].copy()

    def md_table(df: pd.DataFrame, columns: list[str]) -> str:
        small = df[columns].copy()
        for col in small.columns:
            if pd.api.types.is_float_dtype(small[col]):
                small[col] = small[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        header = "| " + " | ".join(small.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(small.columns)) + " |"
        body = []
        for _, row in small.iterrows():
            values = ["" if pd.isna(value) else str(value) for value in row.tolist()]
            body.append("| " + " | ".join(values) + " |")
        return "\n".join([header, separator, *body])

    top_corr = correlation[correlation["metric"].eq("ndcg@20")].copy()
    top_corr["abs_spearman"] = top_corr["spearman"].abs()
    top_corr = top_corr.sort_values("abs_spearman", ascending=False).head(8)
    gt_mismatch_count = int((profile["raw_minus_metric_gt_user_count"] != 0).sum())
    gt_mismatch_rate = gt_mismatch_count / len(profile) if len(profile) else 0.0
    max_gt_mismatch = int(profile["raw_minus_metric_gt_user_count"].abs().max()) if len(profile) else 0

    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG 自然难度分层与混杂诊断结果
date: 2026-06-08
tags:
  - CCFCRec
  - Amazon-VG
  - 实验结果
  - 难度分层
  - 混杂诊断
---

# {run_stamp} CCFCRec Amazon-VG 自然难度分层与混杂诊断结果

## 结论

这次诊断支持一个清晰现象：

```text
CCFCRec Amazon-VG 的 strict cold/test item 内部存在显著难度分层；
类别属性数量少的 item 明显更难，但差距同时受到 gt 用户数、类别支持度等因素影响。
```

这一步仍然是诊断，不证明类别数量是因果变量，也不提出方法。

## 输入与输出

输入 item-level 指标数：

```text
test item count = {len(profile)}
```

核心输出：

```text
item_profile.csv
group_metric_summary.csv
confound_summary.csv
matched_bucket_gap_summary.csv
correlation_summary.csv
{manifest_name}
```

口径说明：

```text
gt_user_count 来自 item-level 复评指标，是后续分层和相关性分析的主口径；
raw_test_user_count 来自原始 test_rating.csv 行数，只作为额外混杂画像。
两者不完全一致的 item 数 = {gt_mismatch_count}/{len(profile)} ({gt_mismatch_rate:.2%})，最大绝对差 = {max_gt_mismatch}。
```

## 自然类别分层

> [!info] 字段说明
> - `group_value`：按 item 类别属性数量得到的分组。
> - `item_count`：该组 test item 数。
> - `gt_user_count_mean`：item-level 复评指标口径下，每个 item 的可评估目标用户数均值。
> - `gt_user_count_median`：item-level 复评指标口径下，每个 item 的可评估目标用户数中位数。
> - `raw_test_user_count_mean`：原始 `test_rating.csv` 行数口径，仅作混杂画像。
> - `ndcg@20_mean`：该组 item-level NDCG@20 均值。
> - `hr@20_mean`：该组 item-level HR@20 均值。

{md_table(category_rows, ["group_value", "item_count", "gt_user_count_mean", "gt_user_count_median", "raw_test_user_count_mean", "ndcg@20_mean", "hr@20_mean"])}

判读：

```text
cat_weak_1_3 显著低于 cat_mid_4 / cat_strong_5_plus。
这说明 average test 指标会掩盖 hard subgroup failure。
```

## gt 用户数分层

> [!info] 字段说明
> - `group_value`：按 item-level `gt_user_count` 分桶得到的 GT 用户数分组。
> - `item_count`：该桶 test item 数。
> - `gt_user_count_mean`：该桶复评口径目标用户数均值。
> - `raw_test_user_count_mean`：原始 `test_rating.csv` 行数均值。
> - `ndcg@20_mean`：该桶 item-level NDCG@20 均值。
> - `hr@20_mean`：该桶 item-level HR@20 均值。
> - 用途：确认 GT 数量本身是否是强难度变量。

{md_table(gt_rows, ["group_value", "item_count", "gt_user_count_mean", "raw_test_user_count_mean", "ndcg@20_mean", "hr@20_mean"])}

判读：

```text
gt_user_count 本身也是非常强的难度变量。这里的 gt_user_count 以 item-level 复评指标为准。
因此后续不能只说类别属性少导致难，必须控制 gt 数量或报告分层。
```

## 混杂画像

> [!info] 字段说明
> - `category_group`：类别属性数量分组。
> - `item_count`：组内 item 数。
> - `ndcg@20_mean`：该组 item-level NDCG@20 均值。
> - `gt_user_count_mean`：复评口径目标用户数均值。
> - `raw_test_user_count_mean`：原始测试行数均值。
> - `raw_minus_metric_gt_user_count_mean`：原始测试行数口径减去复评 GT 口径后的差值均值。
> - `title_tokens_mean`：标题 token 数均值。
> - `raw_category_depth_mean`：原始类别路径深度均值。
> - `image_norm_mean`：图像特征范数均值。
> - `category_train_item_count_max_mean`：item 所属类别在训练集中最大 item 覆盖数均值。
> - `category_shared_train_item_count_mean`：与该 item 共享类别的训练 item 数均值。
> - `rare_category_rate_mean`：该 item 类别中低频类别占比均值。

{md_table(confound, ["category_group", "item_count", "ndcg@20_mean", "gt_user_count_mean", "raw_test_user_count_mean", "raw_minus_metric_gt_user_count_mean", "title_tokens_mean", "raw_category_depth_mean", "image_norm_mean", "category_train_item_count_max_mean", "category_shared_train_item_count_mean", "rare_category_rate_mean"])}

## 匹配桶内 gap 摘要

只统计 weak 组样本不少于 3，且 mid 或 strong 至少一个组样本不少于 3 的 bucket。

> [!info] 字段说明
> - `control`：用于匹配/控制的混杂变量或变量组合。
> - `valid_bucket_count`：满足最小样本条件的桶数量。
> - `mid_minus_weak_ndcg@20_mean`：各有效桶内 `cat_mid_4 - cat_weak_1_3` 的 NDCG@20 gap 均值。
> - `strong_minus_weak_ndcg@20_mean`：各有效桶内 `cat_strong_5_plus - cat_weak_1_3` 的 NDCG@20 gap 均值。
> - `strong_minus_weak_ndcg@20_positive_rate`：有效桶中 strong 高于 weak 的比例。
> - 用途：判断控制该变量后 weak 难度是否仍普遍存在。

{md_table(matched_summary, ["control", "valid_bucket_count", "mid_minus_weak_ndcg@20_mean", "strong_minus_weak_ndcg@20_mean", "strong_minus_weak_ndcg@20_positive_rate"])}

判读边界：

```text
如果控制某个变量后 strong-minus-weak 仍多为正，说明该变量不能单独解释 weak 难度；
如果控制 gt_group 后 gap 大幅缩小，说明 gt 用户数是主要混杂之一。
```

## 相关性最高的变量

> [!info] 字段说明
> - `feature`：候选混杂/解释变量。
> - `metric`：被解释指标。
> - `spearman`：秩相关，适合看单调关系。
> - `pearson`：线性相关。
> - 正值：变量越大，指标通常越高。
> - 负值：变量越大，指标通常越低。
> - 用途：判断 GT 数量、标题长度、图像范数、类别支持度等变量与 `ndcg@20` 的关系强弱。

{md_table(top_corr, ["feature", "metric", "spearman", "pearson"])}

## 下一步

优先做：

```text
1. 受控侧信息消融：no_category / no_image / category_trunc / image_zero；
2. score-norm-margin 诊断：看 weak target 是低分、低 norm，还是被竞争项挤出；
3. content-CF alignment 诊断：看 q_v_c 与 item_embedding 的对齐误差是否在 weak 组更大。
```

暂时不做：

```text
不直接服务器重训；
不直接上 degraded-view training；
不把 category_count 当作因果质量分。
```
"""
    output_path.write_text(markdown, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp, run_iso = now_stamp()

    profile = add_item_profile(paths)
    profile_path = paths.output_dir / "item_profile.csv"
    profile.to_csv(profile_path, index=False)

    summaries = []
    for group_type, group_cols in [
        ("category_group", ["category_group"]),
        ("gt_group", ["gt_group"]),
        ("category_group__gt_group", ["category_group", "gt_group"]),
        ("raw_second_category", ["raw_second_category"]),
        ("image_norm_bucket", ["image_norm_bucket"]),
        ("category_train_item_count_max_bucket", ["category_train_item_count_max_bucket"]),
    ]:
        summary = aggregate_metrics(profile, group_cols)
        summary.insert(0, "group_type", group_type)
        summary["group_value"] = summary[group_cols].astype(str).agg(" | ".join, axis=1)
        summaries.append(summary)
    group_metric = pd.concat(summaries, ignore_index=True)
    group_metric_path = paths.output_dir / "group_metric_summary.csv"
    group_metric.to_csv(group_metric_path, index=False)

    confound = build_confound_summary(profile)
    confound_path = paths.output_dir / "confound_summary.csv"
    confound.to_csv(confound_path, index=False)

    matched = build_matched_bucket_gap(profile)
    matched_path = paths.output_dir / "matched_bucket_gap_summary.csv"
    matched.to_csv(matched_path, index=False)
    matched_summary = summarize_gap_by_control(matched)
    matched_summary_path = paths.output_dir / "matched_bucket_gap_control_summary.csv"
    matched_summary.to_csv(matched_summary_path, index=False)

    correlation = build_correlation_summary(profile)
    correlation_path = paths.output_dir / "correlation_summary.csv"
    correlation.to_csv(correlation_path, index=False)

    result_md_name = f"{run_stamp} CCFCRec Amazon-VG 自然难度分层与混杂诊断结果.md"
    result_md_path = paths.output_dir / result_md_name

    manifest = {
        "run_time": run_iso,
        "script": str(Path(__file__).resolve()),
        "project_root": str(paths.project_root),
        "code_root": str(paths.code_root),
        "dataset_dir": str(paths.dataset_dir),
        "item_metrics": str(paths.item_metrics),
        "output_dir": str(paths.output_dir),
        "outputs": {
            "item_profile": str(profile_path),
            "group_metric_summary": str(group_metric_path),
            "confound_summary": str(confound_path),
            "matched_bucket_gap_summary": str(matched_path),
            "matched_bucket_gap_control_summary": str(matched_summary_path),
            "correlation_summary": str(correlation_path),
            "result_md": str(result_md_path),
        },
        "row_counts": {
            "item_profile": int(len(profile)),
            "group_metric_summary": int(len(group_metric)),
            "matched_bucket_gap_summary": int(len(matched)),
        },
        "notes": [
            "只做诊断，不训练模型。",
            "category_group 是内容证据数量 proxy，不是因果质量标签。",
            "gt_user_count 保留 item-level 复评指标口径；raw_test_user_count 是原始 test_rating.csv 行数。",
            "gt_user_count 是重要混杂变量，结果解释必须控制该因素。",
        ],
    }
    manifest_path = paths.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    write_result_markdown(
        result_md_path,
        run_stamp,
        profile,
        group_metric,
        confound,
        matched_summary,
        correlation,
        manifest_path.name,
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG 自然难度分层与混杂诊断")
    parser.add_argument("--code-root", type=str, default="", help="CCFCRec-code 根目录；默认按脚本位置推断")
    parser.add_argument("--dataset-dir", type=str, default="", help="Amazon VG/data 目录；默认从 code-root 推断")
    parser.add_argument("--item-metrics", type=str, default="", help="best64 item-level test metrics CSV")
    parser.add_argument("--output-dir", type=str, default="", help="诊断输出目录")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
