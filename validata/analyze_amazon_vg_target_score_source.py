#!/usr/bin/env python3
"""
CCFCRec Amazon-VG target-score 来源诊断。

脚本作用：
1. 从 workers45 瘦身 tar 中加载 save_dict.pkl 和 best_epoch_64.pt；
2. 对每个 test item 计算 q_v_c 与目标用户 user_embedding 的 dot/cos/norm 分解；
3. 对比目标用户和 top20 用户的历史类别重合度；
4. 输出 item-level profile、分组汇总、相关性和结果 Markdown。

重要边界：
save_dict.pkl 没有保存训练时的 item_serialize_dict，因此本脚本不把 raw asin 强行映射到 checkpoint item_embedding 行。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyze_amazon_vg_score_norm_margin import (
    METRICS,
    build_attribute_and_image_batch,
    category_order,
    load_model,
    load_pickle_from_tar,
    md_table,
    nan_float,
    now_stamp,
    select_device,
)


ALIGNMENT_FEATURES = [
    "target_score_max",
    "target_score_mean",
    "q_norm",
    "target_cosine_at_score_max",
    "target_cosine_max",
    "target_cosine_mean",
    "target_user_norm_at_score_max",
    "target_user_norm_mean",
    "top20_cosine_mean",
    "top20_user_norm_mean",
    "target_minus_top20_cosine_mean",
    "target_minus_top20_user_norm_mean",
    "target_history_category_overlap_rate_mean",
    "target_history_category_overlap_rate_max",
    "target_history_category_jaccard_mean",
    "target_history_category_jaccard_max",
    "target_history_item_count_mean",
    "top20_history_category_overlap_rate_mean",
    "top20_history_category_jaccard_mean",
    "target_minus_top20_history_overlap_rate_mean",
]


@dataclass(frozen=True)
class Paths:
    project_root: Path
    code_root: Path
    amazon_code_dir: Path
    dataset_dir: Path
    reproduction_tar: Path
    item_profile: Path
    score_margin_profile: Path
    output_dir: Path


def resolve_paths(args: argparse.Namespace) -> Paths:
    script_path = Path(__file__).resolve()
    code_root = Path(args.code_root).expanduser().resolve() if args.code_root else script_path.parents[1]
    project_root = code_root.parent
    amazon_code_dir = code_root / "Amazon VG"
    dataset_dir = (
        Path(args.dataset_dir).expanduser().resolve()
        if args.dataset_dir
        else amazon_code_dir / "data"
    )
    reproduction_tar = (
        Path(args.reproduction_tar).expanduser().resolve()
        if args.reproduction_tar
        else project_root / "实验记录" / "复现" / "amazon_vg_workers45_slim.tar"
    )
    item_profile = (
        Path(args.item_profile).expanduser().resolve()
        if args.item_profile
        else project_root
        / "temp_202606_实验文件记录"
        / "temp_20260608"
        / "natural-difficulty-confounding-diagnostic"
        / "item_profile.csv"
    )
    score_margin_profile = (
        Path(args.score_margin_profile).expanduser().resolve()
        if args.score_margin_profile
        else project_root
        / "temp_202606_实验文件记录"
        / "temp_20260608"
        / "score-norm-margin-diagnostic"
        / "item_score_margin_profile.csv"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else project_root
        / "temp_202606_实验文件记录"
        / "temp_20260608"
        / "target-score-source-diagnostic"
    )
    return Paths(
        project_root=project_root,
        code_root=code_root,
        amazon_code_dir=amazon_code_dir,
        dataset_dir=dataset_dir,
        reproduction_tar=reproduction_tar,
        item_profile=item_profile,
        score_margin_profile=score_margin_profile,
        output_dir=output_dir,
    )


def target_users_by_asin(test_df: pd.DataFrame, user_ser_dict: dict[object, int]) -> dict[str, dict[str, object]]:
    """按 asin 保留可序列化目标用户，同时保留 raw reviewerID 供历史画像使用。"""
    result: dict[str, dict[str, object]] = {}
    for asin, sub in test_df.groupby("asin"):
        raw_users = sub["reviewerID"].tolist()
        mapped_indices: list[int] = []
        mapped_raw_users: list[object] = []
        seen_indices = set()
        unknown_count = 0
        for raw_user in raw_users:
            mapped = user_ser_dict.get(raw_user)
            if mapped is None:
                unknown_count += 1
                continue
            mapped_int = int(mapped)
            if mapped_int in seen_indices:
                continue
            seen_indices.add(mapped_int)
            mapped_indices.append(mapped_int)
            mapped_raw_users.append(raw_user)
        result[asin] = {
            "raw_target_user_count": len(raw_users),
            "mapped_target_user_count": len(mapped_indices),
            "unknown_target_user_count": unknown_count,
            "target_user_indices": mapped_indices,
            "target_raw_users": mapped_raw_users,
        }
    return result


def build_user_history_profiles(
    train_df: pd.DataFrame,
    category_map: dict[str, list[int]],
) -> dict[object, dict[str, object]]:
    """从 train_rating.csv 构造 raw reviewerID 的历史 item 和类别集合。"""
    histories: dict[object, dict[str, object]] = {}
    for raw_user, asin in zip(train_df["reviewerID"], train_df["asin"]):
        history = histories.setdefault(raw_user, {"items": set(), "categories": set(), "interaction_count": 0})
        history["items"].add(asin)
        history["interaction_count"] += 1
        history["categories"].update(int(category) for category in category_map.get(asin, []))
    return histories


def category_overlap_stats(
    raw_users: list[object],
    item_categories: set[int],
    user_histories: dict[object, dict[str, object]],
) -> dict[str, float]:
    """计算一组用户历史类别与目标 item 类别的重合画像。"""
    overlap_rates = []
    jaccards = []
    item_counts = []
    interaction_counts = []
    category_counts = []
    for raw_user in raw_users:
        history = user_histories.get(raw_user)
        if history is None:
            continue
        history_categories = set(history["categories"])
        intersection_count = len(item_categories.intersection(history_categories))
        union_count = len(item_categories.union(history_categories))
        overlap_rates.append(intersection_count / len(item_categories) if item_categories else nan_float())
        jaccards.append(intersection_count / union_count if union_count else nan_float())
        item_counts.append(len(history["items"]))
        interaction_counts.append(int(history["interaction_count"]))
        category_counts.append(len(history_categories))

    def mean(values: list[float]) -> float:
        arr = np.asarray([value for value in values if not pd.isna(value)], dtype=float)
        return float(arr.mean()) if arr.size else nan_float()

    def max_value(values: list[float]) -> float:
        arr = np.asarray([value for value in values if not pd.isna(value)], dtype=float)
        return float(arr.max()) if arr.size else nan_float()

    return {
        "history_category_overlap_rate_mean": mean(overlap_rates),
        "history_category_overlap_rate_max": max_value(overlap_rates),
        "history_category_jaccard_mean": mean(jaccards),
        "history_category_jaccard_max": max_value(jaccards),
        "history_item_count_mean": mean(item_counts),
        "history_interaction_count_mean": mean(interaction_counts),
        "history_category_count_mean": mean(category_counts),
        "history_known_user_count": float(len(overlap_rates)),
    }


def compute_alignment_stats(
    q_vec: torch.Tensor,
    user_embedding: torch.Tensor,
    target_indices: list[int],
    user_norms: torch.Tensor,
    top_k: int = 20,
) -> tuple[dict[str, float], list[int]]:
    """拆解 target_score 的 dot/cos/norm 来源，并返回 top-k 用户索引。"""
    eps = 1e-12
    scores = torch.matmul(user_embedding, q_vec)
    q_norm = torch.norm(q_vec)
    top_values, top_indices = torch.topk(scores, k=min(top_k, int(scores.shape[0])), largest=True)
    top_norms = user_norms[top_indices]
    top_cosines = top_values / (q_norm * top_norms + eps)
    row = {
        "q_norm": float(q_norm.item()),
        "top20_score_mean": float(top_values.mean().item()),
        "top20_cosine_mean": float(top_cosines.mean().item()),
        "top20_cosine_max": float(top_cosines.max().item()),
        "top20_user_norm_mean": float(top_norms.mean().item()),
    }

    if not target_indices:
        row.update(
            {
                "target_score_max": nan_float(),
                "target_score_mean": nan_float(),
                "target_cosine_at_score_max": nan_float(),
                "target_cosine_max": nan_float(),
                "target_cosine_mean": nan_float(),
                "target_user_norm_at_score_max": nan_float(),
                "target_user_norm_mean": nan_float(),
                "target_minus_top20_cosine_mean": nan_float(),
                "target_minus_top20_user_norm_mean": nan_float(),
            }
        )
        return row, [int(value) for value in top_indices.detach().cpu().tolist()]

    target_tensor = torch.tensor(target_indices, dtype=torch.long, device=q_vec.device)
    target_scores = scores[target_tensor]
    target_norms = user_norms[target_tensor]
    target_cosines = target_scores / (q_norm * target_norms + eps)
    best_score_pos = int(torch.argmax(target_scores).item())
    row.update(
        {
            "target_score_max": float(target_scores[best_score_pos].item()),
            "target_score_mean": float(target_scores.mean().item()),
            "target_cosine_at_score_max": float(target_cosines[best_score_pos].item()),
            "target_cosine_max": float(target_cosines.max().item()),
            "target_cosine_mean": float(target_cosines.mean().item()),
            "target_user_norm_at_score_max": float(target_norms[best_score_pos].item()),
            "target_user_norm_mean": float(target_norms.mean().item()),
            "target_minus_top20_cosine_mean": float(target_cosines.mean().item() - top_cosines.mean().item()),
            "target_minus_top20_user_norm_mean": float(target_norms.mean().item() - top_norms.mean().item()),
        }
    )
    return row, [int(value) for value in top_indices.detach().cpu().tolist()]


def index_to_raw_user_map(user_ser_dict: dict[object, int]) -> dict[int, object]:
    return {int(value): key for key, value in user_ser_dict.items()}


def aggregate_summary(profile: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, sub in profile.groupby(group_col, dropna=False):
        row = {
            group_col: group,
            "item_count": len(sub),
            "ndcg@20_mean": sub["ndcg@20"].mean(),
            "hr@20_mean": sub["hr@20"].mean(),
        }
        for feature in ALIGNMENT_FEATURES:
            row[f"{feature}_mean"] = sub[feature].mean()
            row[f"{feature}_median"] = sub[feature].median()
        rows.append(row)
    result = pd.DataFrame(rows)
    for col in list(result.columns):
        if col.endswith("_mean_mean"):
            alias = col[: -len("_mean")]
            if alias not in result.columns:
                result[alias] = result[col]
    if group_col == "category_group":
        return result.sort_values(group_col, key=lambda s: s.map(category_order)).reset_index(drop=True)
    return result.sort_values(group_col).reset_index(drop=True)


def build_correlation_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in ALIGNMENT_FEATURES:
        for metric in ["ndcg@20", "hr@20"]:
            sub = profile[[feature, metric]].dropna()
            if len(sub) < 3:
                pearson = nan_float()
                spearman = nan_float()
            else:
                pearson = float(sub.corr(method="pearson").iloc[0, 1])
                spearman = float(sub.corr(method="spearman").iloc[0, 1])
            rows.append({"feature": feature, "metric": metric, "pearson": pearson, "spearman": spearman})
    return pd.DataFrame(rows)


def field_callout(lines: list[tuple[str, str]]) -> str:
    body = ["+> [!info] 字段说明"]
    body.extend(f"> - `{field}`：{meaning}" for field, meaning in lines)
    return "\n".join(body).replace("+> [!info]", "> [!info]")


def write_result_markdown(
    output_path: Path,
    run_stamp: str,
    item_count: int,
    category_summary: pd.DataFrame,
    gt_summary: pd.DataFrame,
    correlation: pd.DataFrame,
    manifest_name: str,
) -> None:
    top_corr = correlation[correlation["metric"].eq("ndcg@20")].copy()
    top_corr["abs_spearman"] = top_corr["spearman"].abs()
    top_corr = top_corr.sort_values("abs_spearman", ascending=False).head(10)

    category_index = category_summary.set_index("category_group")
    weak = category_index.loc["cat_weak_1_3"] if "cat_weak_1_3" in category_index.index else None
    mid = category_index.loc["cat_mid_4"] if "cat_mid_4" in category_index.index else None
    strong = category_index.loc["cat_strong_5_plus"] if "cat_strong_5_plus" in category_index.index else None

    if weak is not None and mid is not None and strong is not None:
        mechanism_text = f"""核心结论：
weak target score 低主要来自两个方向：目标用户与 q_v_c 的方向对齐更弱，以及目标用户 embedding 范数/历史活跃度更弱；
历史类别 overlap 不是单独主因，因为 weak 不低于 strong。

关键证据：
weak target_cosine_at_score_max_mean = {weak["target_cosine_at_score_max_mean"]:.4f}，mid/strong 分别为 {mid["target_cosine_at_score_max_mean"]:.4f} / {strong["target_cosine_at_score_max_mean"]:.4f}；
weak target_user_norm_at_score_max_mean = {weak["target_user_norm_at_score_max_mean"]:.4f}，mid/strong 分别为 {mid["target_user_norm_at_score_max_mean"]:.4f} / {strong["target_user_norm_at_score_max_mean"]:.4f}；
weak target_history_item_count_mean = {weak["target_history_item_count_mean"]:.4f}，mid/strong 分别为 {mid["target_history_item_count_mean"]:.4f} / {strong["target_history_item_count_mean"]:.4f}；
weak target_history_category_overlap_rate_mean = {weak["target_history_category_overlap_rate_mean"]:.4f}，mid/strong 分别为 {mid["target_history_category_overlap_rate_mean"]:.4f} / {strong["target_history_category_overlap_rate_mean"]:.4f}。"""
    else:
        mechanism_text = "本诊断用于拆解 target_score 的 cosine、user norm 和历史类别重合来源。"

    category_fields = field_callout(
        [
            ("category_group", "按 item 类别属性数量得到的 weak/mid/strong 分组。"),
            ("item_count", "该组 test item 数。"),
            ("ndcg@20_mean", "该组 item-level NDCG@20 均值。"),
            ("target_score_max_mean", "每个 item 的可序列化目标用户最高 dot 分数，再取组均值。"),
            ("target_cosine_at_score_max_mean", "取得最高 target score 的目标用户与 q_v_c 的 cosine 均值。"),
            ("target_cosine_max_mean", "每个 item 所有目标用户中的最高 cosine，再取组均值。"),
            ("target_user_norm_at_score_max_mean", "取得最高 target score 的目标用户 embedding 范数均值。"),
            ("target_history_category_overlap_rate_mean", "目标用户历史类别覆盖当前 item 类别的比例均值。"),
            ("target_history_item_count_mean", "目标用户训练历史 item 数均值，用来近似用户历史丰富度/活跃度。"),
            ("target_minus_top20_cosine_mean", "目标用户 cosine 均值减 top20 用户 cosine 均值，越负说明目标用户方向更不对齐。"),
        ]
    )
    gt_fields = field_callout(
        [
            ("gt_group", "按 item-level 复评指标中的 gt_user_count 分桶。"),
            ("item_count", "该桶 test item 数。"),
            ("ndcg@20_mean", "该桶 item-level NDCG@20 均值。"),
            ("target_cosine_at_score_max_mean", "最高 target score 目标用户与 q_v_c 的 cosine 均值。"),
            ("target_user_norm_at_score_max_mean", "最高 target score 目标用户 embedding 范数均值。"),
            ("target_history_category_overlap_rate_mean", "目标用户历史类别覆盖当前 item 类别的比例均值。"),
            ("target_history_item_count_mean", "目标用户训练历史 item 数均值。"),
            ("target_minus_top20_cosine_mean", "目标用户 cosine 均值减 top20 用户 cosine 均值。"),
        ]
    )
    corr_fields = field_callout(
        [
            ("feature", "候选解释变量。"),
            ("metric", "被解释指标，这里展示 ndcg@20。"),
            ("spearman", "秩相关，适合看单调关系。"),
            ("pearson", "线性相关。"),
            ("正值", "变量越大，指标通常越高。"),
            ("负值", "变量越大，指标通常越低。"),
        ]
    )

    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG target-score 来源诊断结果
date: 2026-06-08
tags:
  - CCFCRec
  - Amazon-VG
  - 实验结果
  - 难度分层
  - 机制诊断
---

# {run_stamp} CCFCRec Amazon-VG target-score 来源诊断结果

## 结论

这次诊断只解释 best64 已复现模型的 target score 来源，不训练新模型。

```text
{mechanism_text}
```

## 输入与输出

输入 item 数：

```text
test item count = {item_count}
```

核心输出：

```text
target_alignment_profile.csv
target_alignment_category_group_summary.csv
target_alignment_gt_group_summary.csv
target_alignment_correlation_summary.csv
{manifest_name}
```

## 按类别证据强弱分组

{category_fields}

{md_table(category_summary, ["category_group", "item_count", "ndcg@20_mean", "target_score_max_mean", "target_cosine_at_score_max_mean", "target_cosine_max_mean", "target_user_norm_at_score_max_mean", "target_history_category_overlap_rate_mean", "target_history_item_count_mean", "target_minus_top20_cosine_mean"])}

## 按 gt 用户数分组

{gt_fields}

{md_table(gt_summary, ["gt_group", "item_count", "ndcg@20_mean", "target_cosine_at_score_max_mean", "target_user_norm_at_score_max_mean", "target_history_category_overlap_rate_mean", "target_history_item_count_mean", "target_minus_top20_cosine_mean"])}

## 与 NDCG@20 相关性最高的变量

{corr_fields}

{md_table(top_corr, ["feature", "metric", "spearman", "pearson"])}

## 下一步

优先根据本诊断选择：

```text
1. 优先进入 content-CF / target-user alignment 诊断：解释为什么 weak 的 q_v_c 与真实目标用户方向不对齐；
2. 同时记录用户侧活跃度因素：weak 的 target user norm 和历史 item 数也偏低；
3. 暂时不把历史类别 overlap 当成主因；
4. 仍不直接服务器重训，先把 alignment 机制拆清楚。
```
"""
    output_path.write_text(markdown, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp, run_iso = now_stamp()
    device_name = select_device(args.device)
    device = torch.device(device_name)

    save_dict = load_pickle_from_tar(paths.reproduction_tar, args.save_dict_member)
    model = load_model(paths, device_name, args.checkpoint_member)
    user_embedding = model.user_embedding.detach()
    user_norms = torch.norm(user_embedding, dim=1)

    item_profile = pd.read_csv(paths.item_profile)
    score_margin_profile = pd.read_csv(paths.score_margin_profile)
    keep_score_cols = [
        "asin",
        "target_score_max",
        "target_score_mean",
        "best_target_rank",
        "target_hr_at20_like",
        "margin_to_top20_cutoff",
    ]
    item_profile = item_profile.merge(
        score_margin_profile[keep_score_cols],
        on="asin",
        how="left",
        suffixes=("", "_score_margin"),
    )
    if args.limit_items > 0:
        item_profile = item_profile.head(args.limit_items).copy()

    train_df = pd.read_csv(paths.dataset_dir / "train_rating.csv")
    test_df = pd.read_csv(paths.dataset_dir / "test_rating.csv")
    targets_by_asin = target_users_by_asin(test_df, save_dict["user_ser_dict"])
    user_histories = build_user_history_profiles(train_df, save_dict["asin_category_int_map"])
    index_to_raw = index_to_raw_user_map(save_dict["user_ser_dict"])
    category_num = int(save_dict["category_ser_map_len"])

    rows = []
    asins = item_profile["asin"].tolist()
    with torch.no_grad():
        for start in range(0, len(asins), args.batch_size):
            batch_asins = asins[start : start + args.batch_size]
            attributes, images = build_attribute_and_image_batch(
                batch_asins,
                category_num,
                save_dict["asin_category_int_map"],
                save_dict["img_feature_dict"],
                device,
            )
            q_v_c = model(attributes, images, len(batch_asins))
            for row_idx, asin in enumerate(batch_asins):
                source_row = item_profile.iloc[start + row_idx].to_dict()
                target_info = targets_by_asin.get(
                    asin,
                    {
                        "raw_target_user_count": 0,
                        "mapped_target_user_count": 0,
                        "unknown_target_user_count": 0,
                        "target_user_indices": [],
                        "target_raw_users": [],
                    },
                )
                alignment, top20_indices = compute_alignment_stats(
                    q_v_c[row_idx],
                    user_embedding,
                    target_info["target_user_indices"],
                    user_norms,
                    top_k=args.top_k,
                )
                item_categories = set(
                    int(category) for category in save_dict["asin_category_int_map"].get(asin, [])
                )
                target_history = category_overlap_stats(
                    target_info["target_raw_users"],
                    item_categories,
                    user_histories,
                )
                top20_raw_users = [index_to_raw[index] for index in top20_indices if index in index_to_raw]
                top20_history = category_overlap_stats(top20_raw_users, item_categories, user_histories)

                output_row = {
                    "asin": asin,
                    "category_group": source_row["category_group"],
                    "category_count": source_row["category_count"],
                    "gt_group": source_row["gt_group"],
                    "gt_user_count": source_row["gt_user_count"],
                    "raw_test_user_count": source_row.get("raw_test_user_count", target_info["raw_target_user_count"]),
                    "raw_target_user_count": target_info["raw_target_user_count"],
                    "mapped_target_user_count": target_info["mapped_target_user_count"],
                    "unknown_target_user_count": target_info["unknown_target_user_count"],
                }
                for metric in METRICS:
                    output_row[metric] = source_row[metric]
                for source_col in ["best_target_rank", "target_hr_at20_like", "margin_to_top20_cutoff"]:
                    output_row[source_col] = source_row.get(source_col, nan_float())
                output_row.update(alignment)
                output_row.update(
                    {
                        "target_history_category_overlap_rate_mean": target_history[
                            "history_category_overlap_rate_mean"
                        ],
                        "target_history_category_overlap_rate_max": target_history[
                            "history_category_overlap_rate_max"
                        ],
                        "target_history_category_jaccard_mean": target_history["history_category_jaccard_mean"],
                        "target_history_category_jaccard_max": target_history["history_category_jaccard_max"],
                        "target_history_item_count_mean": target_history["history_item_count_mean"],
                        "target_history_interaction_count_mean": target_history["history_interaction_count_mean"],
                        "target_history_category_count_mean": target_history["history_category_count_mean"],
                        "target_history_known_user_count": target_history["history_known_user_count"],
                        "top20_history_category_overlap_rate_mean": top20_history[
                            "history_category_overlap_rate_mean"
                        ],
                        "top20_history_category_jaccard_mean": top20_history["history_category_jaccard_mean"],
                        "top20_history_item_count_mean": top20_history["history_item_count_mean"],
                    }
                )
                output_row["target_minus_top20_history_overlap_rate_mean"] = (
                    output_row["target_history_category_overlap_rate_mean"]
                    - output_row["top20_history_category_overlap_rate_mean"]
                    if not pd.isna(output_row["target_history_category_overlap_rate_mean"])
                    and not pd.isna(output_row["top20_history_category_overlap_rate_mean"])
                    else nan_float()
                )
                rows.append(output_row)

    profile = pd.DataFrame(rows)
    profile_path = paths.output_dir / "target_alignment_profile.csv"
    profile.to_csv(profile_path, index=False)

    category_summary = aggregate_summary(profile, "category_group")
    category_summary_path = paths.output_dir / "target_alignment_category_group_summary.csv"
    category_summary.to_csv(category_summary_path, index=False)

    gt_summary = aggregate_summary(profile, "gt_group")
    gt_summary_path = paths.output_dir / "target_alignment_gt_group_summary.csv"
    gt_summary.to_csv(gt_summary_path, index=False)

    correlation = build_correlation_summary(profile)
    correlation_path = paths.output_dir / "target_alignment_correlation_summary.csv"
    correlation.to_csv(correlation_path, index=False)

    result_md_name = f"{run_stamp} CCFCRec Amazon-VG target-score 来源诊断结果.md"
    result_md_path = paths.output_dir / result_md_name
    manifest_path = paths.output_dir / "run_manifest.json"
    manifest = {
        "run_time": run_iso,
        "script": str(Path(__file__).resolve()),
        "project_root": str(paths.project_root),
        "code_root": str(paths.code_root),
        "dataset_dir": str(paths.dataset_dir),
        "reproduction_tar": str(paths.reproduction_tar),
        "save_dict_member": args.save_dict_member,
        "checkpoint_member": args.checkpoint_member,
        "item_profile": str(paths.item_profile),
        "score_margin_profile": str(paths.score_margin_profile),
        "output_dir": str(paths.output_dir),
        "device": device_name,
        "batch_size": args.batch_size,
        "top_k": args.top_k,
        "limit_items": args.limit_items,
        "outputs": {
            "target_alignment_profile": str(profile_path),
            "target_alignment_category_group_summary": str(category_summary_path),
            "target_alignment_gt_group_summary": str(gt_summary_path),
            "target_alignment_correlation_summary": str(correlation_path),
            "result_md": str(result_md_path),
        },
        "row_counts": {
            "target_alignment_profile": int(len(profile)),
            "category_summary": int(len(category_summary)),
            "gt_summary": int(len(gt_summary)),
            "correlation": int(len(correlation)),
        },
        "notes": [
            "只做本地诊断，不训练模型。",
            "不使用 checkpoint item_embedding，因为 save_dict.pkl 未保存训练时 item_serialize_dict。",
            "target_score 拆解为 q_norm、user_norm、cosine 三部分。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_result_markdown(
        result_md_path,
        run_stamp,
        len(profile),
        category_summary,
        gt_summary,
        correlation,
        manifest_path.name,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG target-score 来源诊断")
    parser.add_argument("--code-root", type=str, default="", help="CCFCRec-code 根目录；默认按脚本位置推断")
    parser.add_argument("--dataset-dir", type=str, default="", help="Amazon VG/data 目录；默认从 code-root 推断")
    parser.add_argument("--reproduction-tar", type=str, default="", help="workers45 瘦身 tar 路径")
    parser.add_argument("--item-profile", type=str, default="", help="自然难度诊断 item_profile.csv")
    parser.add_argument("--score-margin-profile", type=str, default="", help="score-norm-margin item profile CSV")
    parser.add_argument("--output-dir", type=str, default="", help="诊断输出目录")
    parser.add_argument("--save-dict-member", type=str, default="amazon_vg_slim/save_dict.pkl", help="tar 内 save_dict.pkl 成员名")
    parser.add_argument("--checkpoint-member", type=str, default="amazon_vg_slim/best_epoch_64.pt", help="tar 内 checkpoint 成员名")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="推理设备")
    parser.add_argument("--batch-size", type=int, default=64, help="item 推理 batch size")
    parser.add_argument("--top-k", type=int, default=20, help="用于对照的 top-k 用户数")
    parser.add_argument("--limit-items", type=int, default=0, help="调试时限制 item 数；0 表示全量")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
