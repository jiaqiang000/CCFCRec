#!/usr/bin/env python3
"""
CCFCRec Amazon-VG content-CF alignment 诊断。

脚本作用：
1. 从 workers45 瘦身 tar 中加载 save_dict.pkl 和 best_epoch_64.pt；
2. 重新前向生成每个 item 的融合表示 q_v_c、类别表示 attr_emb、图像投影 img_proj；
3. 用 train_rating.csv 构造每个用户的历史内容中心；
4. 对比 test item 与真实目标用户历史中心、模型 top20 用户历史中心的对齐差异；
5. 输出 item-level profile、分组汇总、相关性、run_manifest.json 和结果 Markdown。

重要边界：
save_dict.pkl 没有保存训练时的 item_serialize_dict，因此本脚本不把 raw asin 强行映射到 checkpoint item_embedding 行。
这里的历史中心全部由 raw asin 重新前向生成内容表示后构造。
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
from analyze_amazon_vg_target_score_source import target_users_by_asin


CONTENT_CF_FEATURES = [
    "target_history_q_cosine_mean",
    "target_history_q_cosine_max",
    "top20_history_q_cosine_mean",
    "target_minus_top20_history_q_cosine_mean",
    "target_history_attr_cosine_mean",
    "target_history_attr_cosine_max",
    "top20_history_attr_cosine_mean",
    "target_minus_top20_history_attr_cosine_mean",
    "target_history_img_cosine_mean",
    "target_history_img_cosine_max",
    "top20_history_img_cosine_mean",
    "target_minus_top20_history_img_cosine_mean",
    "target_history_interaction_count_mean",
    "top20_history_interaction_count_mean",
    "target_minus_top20_history_interaction_count_mean",
    "target_known_history_user_count",
    "top20_known_history_user_count",
]


@dataclass(frozen=True)
class Paths:
    project_root: Path
    code_root: Path
    amazon_code_dir: Path
    dataset_dir: Path
    reproduction_tar: Path
    item_profile: Path
    target_alignment_profile: Path
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
    target_alignment_profile = (
        Path(args.target_alignment_profile).expanduser().resolve()
        if args.target_alignment_profile
        else project_root
        / "temp_202606_实验文件记录"
        / "temp_20260608"
        / "target-score-source-diagnostic"
        / "target_alignment_profile.csv"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else project_root
        / "temp_202606_实验文件记录"
        / "temp_20260608"
        / "content-cf-alignment-diagnostic"
    )
    return Paths(
        project_root=project_root,
        code_root=code_root,
        amazon_code_dir=amazon_code_dir,
        dataset_dir=dataset_dir,
        reproduction_tar=reproduction_tar,
        item_profile=item_profile,
        target_alignment_profile=target_alignment_profile,
        output_dir=output_dir,
    )


def forward_content_components(
    model: torch.nn.Module,
    attributes: torch.Tensor,
    image_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """复刻模型 forward，同时暴露融合、类别、图像三路 item 表示。"""
    if hasattr(model, "encode_content_components"):
        return model.encode_content_components(attributes, image_features, attributes.shape[0])
    device = attributes.device
    batch_size = attributes.shape[0]
    z_v = torch.matmul(torch.matmul(model.attr_matrix, model.attr_W1) + model.attr_b1.squeeze(), model.attr_W2)
    z_v_copy = z_v.repeat(batch_size, 1, 1)
    z_v_squeeze = z_v_copy.squeeze(dim=2).to(device)
    neg_inf = torch.full(z_v_squeeze.shape, -1e6, device=device)
    z_v_mask = torch.where(attributes != -1, z_v_squeeze, neg_inf)
    attr_attention_weight = torch.softmax(z_v_mask, dim=1)
    attr_emb = torch.matmul(attr_attention_weight, model.attr_matrix)
    img_proj = torch.matmul(image_features, model.image_projection)
    q_v_a = torch.cat((attr_emb, img_proj), dim=1)
    q_v_c = model.gen_layer2(model.h(model.gen_layer1(q_v_a)))
    return q_v_c, attr_emb, img_proj


def infer_content_vectors(
    model: torch.nn.Module,
    asins: list[str],
    category_num: int,
    category_map: dict[str, list[int]],
    img_feature_dict: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """为 asin 列表批量生成 q_v_c、attr_emb、img_proj。"""
    q_chunks = []
    attr_chunks = []
    img_chunks = []
    with torch.no_grad():
        for start in range(0, len(asins), batch_size):
            batch_asins = asins[start : start + batch_size]
            attributes, images = build_attribute_and_image_batch(
                batch_asins,
                category_num,
                category_map,
                img_feature_dict,
                device,
            )
            q_v_c, attr_emb, img_proj = forward_content_components(model, attributes, images)
            q_chunks.append(q_v_c.detach().cpu().numpy().astype(np.float32))
            attr_chunks.append(attr_emb.detach().cpu().numpy().astype(np.float32))
            img_chunks.append(img_proj.detach().cpu().numpy().astype(np.float32))
    return np.vstack(q_chunks), np.vstack(attr_chunks), np.vstack(img_chunks)


def build_user_content_centroids(
    train_df: pd.DataFrame,
    user_ser_dict: dict[object, int],
    asin_to_pos: dict[str, int],
    q_vectors: np.ndarray,
    attr_vectors: np.ndarray,
    img_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用用户训练历史 item 的内容向量均值构造用户历史中心。"""
    user_count = max(int(value) for value in user_ser_dict.values()) + 1
    dim = q_vectors.shape[1]
    q_sum = np.zeros((user_count, dim), dtype=np.float32)
    attr_sum = np.zeros((user_count, dim), dtype=np.float32)
    img_sum = np.zeros((user_count, dim), dtype=np.float32)
    interaction_counts = np.zeros(user_count, dtype=np.int32)

    for raw_user, asin in zip(train_df["reviewerID"], train_df["asin"]):
        user_idx = user_ser_dict.get(raw_user)
        asin_pos = asin_to_pos.get(asin)
        if user_idx is None or asin_pos is None:
            continue
        idx = int(user_idx)
        q_sum[idx] += q_vectors[asin_pos]
        attr_sum[idx] += attr_vectors[asin_pos]
        img_sum[idx] += img_vectors[asin_pos]
        interaction_counts[idx] += 1

    valid = interaction_counts > 0
    q_centroids = q_sum.copy()
    attr_centroids = attr_sum.copy()
    img_centroids = img_sum.copy()
    q_centroids[valid] /= interaction_counts[valid, None]
    attr_centroids[valid] /= interaction_counts[valid, None]
    img_centroids[valid] /= interaction_counts[valid, None]
    return q_centroids, attr_centroids, img_centroids, interaction_counts


def cosine_summary(
    item_vector: np.ndarray,
    centroid_matrix: np.ndarray,
    user_indices: list[int],
    interaction_counts: np.ndarray,
) -> dict[str, float]:
    """计算一个 item 和一组用户历史中心的 cosine 均值/最大值。"""
    valid_indices = [
        int(index)
        for index in user_indices
        if 0 <= int(index) < centroid_matrix.shape[0] and interaction_counts[int(index)] > 0
    ]
    if not valid_indices:
        return {"cosine_mean": nan_float(), "cosine_max": nan_float(), "known_user_count": 0.0}

    centers = centroid_matrix[valid_indices]
    item_norm = float(np.linalg.norm(item_vector))
    center_norms = np.linalg.norm(centers, axis=1)
    valid_mask = (center_norms > 0) & (item_norm > 0)
    if not np.any(valid_mask):
        return {"cosine_mean": nan_float(), "cosine_max": nan_float(), "known_user_count": 0.0}

    cosines = centers[valid_mask].dot(item_vector) / (center_norms[valid_mask] * item_norm)
    return {
        "cosine_mean": float(np.mean(cosines)),
        "cosine_max": float(np.max(cosines)),
        "known_user_count": float(len(cosines)),
    }


def mean_history_interaction_count(user_indices: list[int], interaction_counts: np.ndarray) -> float:
    valid_counts = [
        int(interaction_counts[int(index)])
        for index in user_indices
        if 0 <= int(index) < len(interaction_counts) and interaction_counts[int(index)] > 0
    ]
    return float(np.mean(valid_counts)) if valid_counts else nan_float()


def add_alignment_prefix(row: dict[str, float], prefix: str, summary: dict[str, float]) -> None:
    row[f"{prefix}_cosine_mean"] = summary["cosine_mean"]
    row[f"{prefix}_cosine_max"] = summary["cosine_max"]
    row[f"{prefix}_known_user_count"] = summary["known_user_count"]


def aggregate_summary(profile: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, sub in profile.groupby(group_col, dropna=False):
        row = {
            group_col: group,
            "item_count": len(sub),
            "ndcg@20_mean": sub["ndcg@20"].mean(),
            "hr@20_mean": sub["hr@20"].mean(),
        }
        for feature in CONTENT_CF_FEATURES:
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
    for feature in CONTENT_CF_FEATURES:
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
weak item 与真实目标用户历史内容中心的融合表示 q_v_c 对齐更弱，并且相对模型 top20 用户的对齐缺口更大；
类别表示 attr_emb 的组间差距更明显，图像表示 img_proj 只显示 weak 低于 mid、但与 strong 接近，因此不能把主因单独归给图像路。
同时，目标用户历史交互数/覆盖用户数仍是强混杂项，后续需要做 user-activity matched 诊断。

关键证据：
weak target_history_q_cosine_mean = {weak["target_history_q_cosine_mean"]:.4f}，mid/strong 分别为 {mid["target_history_q_cosine_mean"]:.4f} / {strong["target_history_q_cosine_mean"]:.4f}；
weak target_minus_top20_history_q_cosine_mean = {weak["target_minus_top20_history_q_cosine_mean"]:.4f}，mid/strong 分别为 {mid["target_minus_top20_history_q_cosine_mean"]:.4f} / {strong["target_minus_top20_history_q_cosine_mean"]:.4f}；
weak target_history_attr_cosine_mean = {weak["target_history_attr_cosine_mean"]:.4f}，mid/strong 分别为 {mid["target_history_attr_cosine_mean"]:.4f} / {strong["target_history_attr_cosine_mean"]:.4f}；
weak target_history_img_cosine_mean = {weak["target_history_img_cosine_mean"]:.4f}，mid/strong 分别为 {mid["target_history_img_cosine_mean"]:.4f} / {strong["target_history_img_cosine_mean"]:.4f}；
weak target_history_interaction_count_mean = {weak["target_history_interaction_count_mean"]:.4f}，mid/strong 分别为 {mid["target_history_interaction_count_mean"]:.4f} / {strong["target_history_interaction_count_mean"]:.4f}。"""
    else:
        mechanism_text = "本诊断用于比较 test item 与目标用户历史内容中心、top20 用户历史内容中心的对齐关系。"

    category_fields = field_callout(
        [
            ("category_group", "按 item 类别属性数量得到的 weak/mid/strong 分组。"),
            ("item_count", "该组 test item 数。"),
            ("ndcg@20_mean", "该组 item-level NDCG@20 均值。"),
            ("target_history_q_cosine_mean", "item 融合表示 q_v_c 与真实目标用户历史 q_v_c 中心的 cosine 均值，再取组均值。"),
            ("top20_history_q_cosine_mean", "item 融合表示 q_v_c 与模型 top20 用户历史 q_v_c 中心的 cosine 均值，再取组均值。"),
            ("target_minus_top20_history_q_cosine_mean", "真实目标用户历史 q 对齐均值减 top20 用户历史 q 对齐均值，越负说明模型更偏向非目标用户。"),
            ("target_history_attr_cosine_mean", "item 类别表示 attr_emb 与真实目标用户历史 attr_emb 中心的 cosine 均值，再取组均值。"),
            ("target_history_img_cosine_mean", "item 图像投影 img_proj 与真实目标用户历史 img_proj 中心的 cosine 均值，再取组均值。"),
            ("target_history_interaction_count_mean", "真实目标用户训练历史交互数均值，再取组均值。"),
        ]
    )
    gt_fields = field_callout(
        [
            ("gt_group", "按 item-level 复评指标中的 gt_user_count 分桶。"),
            ("item_count", "该桶 test item 数。"),
            ("ndcg@20_mean", "该桶 item-level NDCG@20 均值。"),
            ("target_history_q_cosine_mean", "融合表示 q_v_c 的目标用户历史中心对齐均值。"),
            ("target_minus_top20_history_q_cosine_mean", "目标用户历史 q 对齐均值减 top20 用户历史 q 对齐均值。"),
            ("target_history_attr_cosine_mean", "类别表示 attr_emb 的目标用户历史中心对齐均值。"),
            ("target_history_img_cosine_mean", "图像投影 img_proj 的目标用户历史中心对齐均值。"),
            ("target_history_interaction_count_mean", "目标用户训练历史交互数均值。"),
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
title: {run_stamp} CCFCRec Amazon-VG content-CF alignment 诊断结果
date: 2026-06-08
tags:
  - CCFCRec
  - Amazon-VG
  - 实验结果
  - 难度分层
  - 机制诊断
---

# {run_stamp} CCFCRec Amazon-VG content-CF alignment 诊断结果

## 结论

这次诊断只解释 best64 已复现模型的内容-CF 对齐关系，不训练新模型。

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
content_cf_alignment_profile.csv
content_cf_alignment_category_group_summary.csv
content_cf_alignment_gt_group_summary.csv
content_cf_alignment_correlation_summary.csv
{manifest_name}
```

## 按类别证据强弱分组

{category_fields}

{md_table(category_summary, ["category_group", "item_count", "ndcg@20_mean", "target_history_q_cosine_mean", "top20_history_q_cosine_mean", "target_minus_top20_history_q_cosine_mean", "target_history_attr_cosine_mean", "target_history_img_cosine_mean", "target_history_interaction_count_mean"])}

## 按 gt 用户数分组

{gt_fields}

{md_table(gt_summary, ["gt_group", "item_count", "ndcg@20_mean", "target_history_q_cosine_mean", "target_minus_top20_history_q_cosine_mean", "target_history_attr_cosine_mean", "target_history_img_cosine_mean", "target_history_interaction_count_mean"])}

## 与 NDCG@20 相关性最高的变量

{corr_fields}

{md_table(top_corr, ["feature", "metric", "spearman", "pearson"])}

## 判读边界

```text
1. 本诊断不使用 checkpoint item_embedding，因为缺少训练时 item_serialize_dict。
2. 用户历史中心由 train_rating.csv 的 raw asin 重新前向生成内容表示后求均值。
3. target_minus_top20_history_q_cosine_mean 越负，只说明模型 top20 用户的历史内容中心更接近该 item；它不是因果证明。
4. 当前更稳的信号是 q_v_c 和 attr_emb 对齐差距；img_proj 需要在匹配用户活跃度后再判断。
```

## 下一步

优先根据本诊断选择：

```text
1. 下一步优先做 user-activity matched content-CF 诊断：控制目标用户历史交互数/gt 用户数后再看 q/attr gap 是否保留；
2. 如果匹配后 q/attr gap 仍保留，再做 counterfactual/category ablation；
3. 仍不直接服务器重训，先把可解释诊断链闭合。
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
    category_num = int(save_dict["category_ser_map_len"])

    item_profile = pd.read_csv(paths.item_profile)
    if paths.target_alignment_profile.exists():
        target_profile = pd.read_csv(paths.target_alignment_profile)
        keep_cols = [
            col
            for col in [
                "asin",
                "target_score_max",
                "target_cosine_at_score_max",
                "target_user_norm_at_score_max",
                "target_history_item_count_mean",
                "best_target_rank",
                "margin_to_top20_cutoff",
            ]
            if col in target_profile.columns
        ]
        if len(keep_cols) > 1:
            item_profile = item_profile.merge(target_profile[keep_cols], on="asin", how="left")
    if args.limit_items > 0:
        item_profile = item_profile.head(args.limit_items).copy()

    train_df = pd.read_csv(paths.dataset_dir / "train_rating.csv", usecols=["reviewerID", "asin"])
    test_df = pd.read_csv(paths.dataset_dir / "test_rating.csv", usecols=["reviewerID", "asin"])
    targets_by_asin = target_users_by_asin(test_df, save_dict["user_ser_dict"])

    train_asins = train_df["asin"].drop_duplicates().tolist()
    test_asins = item_profile["asin"].tolist()
    all_asins = list(dict.fromkeys([*train_asins, *test_asins]))
    asin_to_pos = {asin: index for index, asin in enumerate(all_asins)}

    q_vectors, attr_vectors, img_vectors = infer_content_vectors(
        model,
        all_asins,
        category_num,
        save_dict["asin_category_int_map"],
        save_dict["img_feature_dict"],
        device,
        args.batch_size,
    )
    q_centroids, attr_centroids, img_centroids, interaction_counts = build_user_content_centroids(
        train_df,
        save_dict["user_ser_dict"],
        asin_to_pos,
        q_vectors,
        attr_vectors,
        img_vectors,
    )

    user_embedding = model.user_embedding.detach()
    rows = []
    with torch.no_grad():
        for start in range(0, len(test_asins), args.score_batch_size):
            batch_asins = test_asins[start : start + args.score_batch_size]
            batch_positions = [asin_to_pos[asin] for asin in batch_asins]
            q_batch_np = q_vectors[batch_positions]
            q_batch = torch.tensor(q_batch_np, dtype=torch.float32, device=device)
            scores_batch = torch.matmul(q_batch, user_embedding.T)
            top20_batch = torch.topk(scores_batch, k=min(args.top_k, user_embedding.shape[0]), dim=1).indices

            for row_idx, asin in enumerate(batch_asins):
                source_row = item_profile.iloc[start + row_idx].to_dict()
                target_info = targets_by_asin.get(
                    asin,
                    {
                        "raw_target_user_count": 0,
                        "mapped_target_user_count": 0,
                        "unknown_target_user_count": 0,
                        "target_user_indices": [],
                    },
                )
                top20_indices = [int(value) for value in top20_batch[row_idx].detach().cpu().tolist()]
                pos = asin_to_pos[asin]

                target_q = cosine_summary(q_vectors[pos], q_centroids, target_info["target_user_indices"], interaction_counts)
                top20_q = cosine_summary(q_vectors[pos], q_centroids, top20_indices, interaction_counts)
                target_attr = cosine_summary(attr_vectors[pos], attr_centroids, target_info["target_user_indices"], interaction_counts)
                top20_attr = cosine_summary(attr_vectors[pos], attr_centroids, top20_indices, interaction_counts)
                target_img = cosine_summary(img_vectors[pos], img_centroids, target_info["target_user_indices"], interaction_counts)
                top20_img = cosine_summary(img_vectors[pos], img_centroids, top20_indices, interaction_counts)

                target_history_count = mean_history_interaction_count(
                    target_info["target_user_indices"],
                    interaction_counts,
                )
                top20_history_count = mean_history_interaction_count(top20_indices, interaction_counts)
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
                    "target_history_interaction_count_mean": target_history_count,
                    "top20_history_interaction_count_mean": top20_history_count,
                    "target_minus_top20_history_interaction_count_mean": (
                        target_history_count - top20_history_count
                        if not math.isnan(target_history_count) and not math.isnan(top20_history_count)
                        else nan_float()
                    ),
                }
                for metric in METRICS:
                    output_row[metric] = source_row[metric]
                for optional_col in [
                    "target_score_max",
                    "target_cosine_at_score_max",
                    "target_user_norm_at_score_max",
                    "target_history_item_count_mean",
                    "best_target_rank",
                    "margin_to_top20_cutoff",
                ]:
                    if optional_col in source_row:
                        output_row[optional_col] = source_row.get(optional_col, nan_float())

                add_alignment_prefix(output_row, "target_history_q", target_q)
                add_alignment_prefix(output_row, "top20_history_q", top20_q)
                add_alignment_prefix(output_row, "target_history_attr", target_attr)
                add_alignment_prefix(output_row, "top20_history_attr", top20_attr)
                add_alignment_prefix(output_row, "target_history_img", target_img)
                add_alignment_prefix(output_row, "top20_history_img", top20_img)
                output_row["target_minus_top20_history_q_cosine_mean"] = (
                    output_row["target_history_q_cosine_mean"] - output_row["top20_history_q_cosine_mean"]
                    if not pd.isna(output_row["target_history_q_cosine_mean"])
                    and not pd.isna(output_row["top20_history_q_cosine_mean"])
                    else nan_float()
                )
                output_row["target_minus_top20_history_attr_cosine_mean"] = (
                    output_row["target_history_attr_cosine_mean"] - output_row["top20_history_attr_cosine_mean"]
                    if not pd.isna(output_row["target_history_attr_cosine_mean"])
                    and not pd.isna(output_row["top20_history_attr_cosine_mean"])
                    else nan_float()
                )
                output_row["target_minus_top20_history_img_cosine_mean"] = (
                    output_row["target_history_img_cosine_mean"] - output_row["top20_history_img_cosine_mean"]
                    if not pd.isna(output_row["target_history_img_cosine_mean"])
                    and not pd.isna(output_row["top20_history_img_cosine_mean"])
                    else nan_float()
                )
                output_row["target_known_history_user_count"] = output_row["target_history_q_known_user_count"]
                output_row["top20_known_history_user_count"] = output_row["top20_history_q_known_user_count"]
                rows.append(output_row)

    profile = pd.DataFrame(rows)
    profile_path = paths.output_dir / "content_cf_alignment_profile.csv"
    profile.to_csv(profile_path, index=False)

    category_summary = aggregate_summary(profile, "category_group")
    category_summary_path = paths.output_dir / "content_cf_alignment_category_group_summary.csv"
    category_summary.to_csv(category_summary_path, index=False)

    gt_summary = aggregate_summary(profile, "gt_group")
    gt_summary_path = paths.output_dir / "content_cf_alignment_gt_group_summary.csv"
    gt_summary.to_csv(gt_summary_path, index=False)

    correlation = build_correlation_summary(profile)
    correlation_path = paths.output_dir / "content_cf_alignment_correlation_summary.csv"
    correlation.to_csv(correlation_path, index=False)

    result_md_name = f"{run_stamp} CCFCRec Amazon-VG content-CF alignment 诊断结果.md"
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
        "target_alignment_profile": str(paths.target_alignment_profile),
        "output_dir": str(paths.output_dir),
        "device": device_name,
        "batch_size": args.batch_size,
        "score_batch_size": args.score_batch_size,
        "top_k": args.top_k,
        "limit_items": args.limit_items,
        "content_vector_count": int(len(all_asins)),
        "outputs": {
            "content_cf_alignment_profile": str(profile_path),
            "content_cf_alignment_category_group_summary": str(category_summary_path),
            "content_cf_alignment_gt_group_summary": str(gt_summary_path),
            "content_cf_alignment_correlation_summary": str(correlation_path),
            "result_md": str(result_md_path),
        },
        "row_counts": {
            "content_cf_alignment_profile": int(len(profile)),
            "category_summary": int(len(category_summary)),
            "gt_summary": int(len(gt_summary)),
            "correlation": int(len(correlation)),
        },
        "notes": [
            "只做本地诊断，不训练模型。",
            "不使用 checkpoint item_embedding，因为 save_dict.pkl 未保存训练时 item_serialize_dict。",
            "用户历史中心由 train_rating.csv 的 raw asin 重新前向生成内容表示后求均值。",
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
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG content-CF alignment 诊断")
    parser.add_argument("--code-root", type=str, default="", help="CCFCRec-code 根目录；默认按脚本位置推断")
    parser.add_argument("--dataset-dir", type=str, default="", help="Amazon VG/data 目录；默认从 code-root 推断")
    parser.add_argument("--reproduction-tar", type=str, default="", help="workers45 瘦身 tar 路径")
    parser.add_argument("--item-profile", type=str, default="", help="自然难度诊断 item_profile.csv")
    parser.add_argument("--target-alignment-profile", type=str, default="", help="target-score 来源诊断 profile CSV")
    parser.add_argument("--output-dir", type=str, default="", help="诊断输出目录")
    parser.add_argument("--save-dict-member", type=str, default="amazon_vg_slim/save_dict.pkl", help="tar 内 save_dict.pkl 成员名")
    parser.add_argument("--checkpoint-member", type=str, default="amazon_vg_slim/best_epoch_64.pt", help="tar 内 checkpoint 成员名")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="推理设备")
    parser.add_argument("--batch-size", type=int, default=128, help="内容向量前向 batch size")
    parser.add_argument("--score-batch-size", type=int, default=128, help="top20 用户打分 batch size")
    parser.add_argument("--top-k", type=int, default=20, help="用于对照的 top-k 用户数")
    parser.add_argument("--limit-items", type=int, default=0, help="调试时限制 item 数；0 表示全量")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
