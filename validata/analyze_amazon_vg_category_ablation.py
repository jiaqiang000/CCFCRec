#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category ablation / counterfactual 诊断。

脚本作用：
1. 从 workers45 瘦身 tar 中加载 save_dict.pkl 和 best_epoch_64.pt；
2. 对 test item 构造 original、image_only、category_only、all_category_upper 四种前向 variant；
3. 比较四种 variant 的 target score、top20 margin、best target rank、target-history q 对齐；
4. 输出 item-variant 长表、分组汇总、gap/delta 汇总、相关性和结果 Markdown。

重要边界：
all_category_upper 是不现实上界，只用于判断“类别补全”方向是否值得进一步做方法实验。
category_only 的图像置零是分布外输入，只用于机制诊断。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyze_amazon_vg_content_cf_alignment import (
    build_user_content_centroids,
    cosine_summary,
    field_callout,
    forward_content_components,
    infer_content_vectors,
)
from analyze_amazon_vg_score_norm_margin import (
    METRICS,
    build_attribute_and_image_batch,
    category_order,
    compute_single_score_margin,
    load_model,
    load_pickle_from_tar,
    md_table,
    nan_float,
    now_stamp,
    select_device,
)
from analyze_amazon_vg_target_score_source import target_users_by_asin


VARIANTS = ["original", "image_only", "category_only", "all_category_upper"]
CORE_METRICS = [
    "target_score_max",
    "margin_to_top20_cutoff",
    "best_target_rank",
    "target_history_q_cosine_mean",
    "top20_history_q_cosine_mean",
    "target_minus_top20_history_q_cosine_mean",
    "q_norm",
]


@dataclass(frozen=True)
class Paths:
    project_root: Path
    code_root: Path
    amazon_code_dir: Path
    dataset_dir: Path
    reproduction_tar: Path
    item_profile: Path
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
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else project_root
        / "temp_202606_实验文件记录"
        / "temp_20260608"
        / "category-ablation-diagnostic"
    )
    return Paths(
        project_root=project_root,
        code_root=code_root,
        amazon_code_dir=amazon_code_dir,
        dataset_dir=dataset_dir,
        reproduction_tar=reproduction_tar,
        item_profile=item_profile,
        output_dir=output_dir,
    )


def build_variant_attribute_and_image_batch(
    asins: list[str],
    variant: str,
    category_num: int,
    category_map: dict[str, list[int]],
    img_feature_dict: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 category ablation variant 的 attribute/image 输入。"""
    if variant not in VARIANTS:
        raise ValueError(f"未知 variant: {variant}")
    attributes, images = build_attribute_and_image_batch(
        asins,
        category_num,
        category_map,
        img_feature_dict,
        device,
    )
    if variant == "original":
        return attributes, images
    if variant == "image_only":
        return torch.full_like(attributes, -1.0), images
    if variant == "category_only":
        return attributes, torch.zeros_like(images)
    all_attributes = torch.ones_like(attributes)
    return all_attributes, images


def infer_variant_q(
    model: torch.nn.Module,
    asins: list[str],
    variant: str,
    category_num: int,
    category_map: dict[str, list[int]],
    img_feature_dict: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for start in range(0, len(asins), batch_size):
            batch_asins = asins[start : start + batch_size]
            attributes, images = build_variant_attribute_and_image_batch(
                batch_asins,
                variant,
                category_num,
                category_map,
                img_feature_dict,
                device,
            )
            q_v_c, _, _ = forward_content_components(model, attributes, images)
            chunks.append(q_v_c.detach().cpu().numpy().astype(np.float32))
    return np.vstack(chunks)


def build_variant_group_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, group), sub in profile.groupby(["variant", "category_group"], dropna=False):
        row = {
            "variant": variant,
            "category_group": group,
            "item_count": len(sub),
            "ndcg@20_mean": sub["ndcg@20"].mean(),
        }
        for metric in CORE_METRICS:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_median"] = sub[metric].median()
        rows.append(row)
    result = pd.DataFrame(rows)
    result["variant_order"] = result["variant"].map({variant: idx for idx, variant in enumerate(VARIANTS)})
    return (
        result.sort_values(["variant_order", "category_group"], key=lambda s: s.map(category_order) if s.name == "category_group" else s)
        .drop(columns=["variant_order"])
        .reset_index(drop=True)
    )


def build_variant_gap_summary(group_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, sub in group_summary.groupby("variant"):
        indexed = sub.set_index("category_group")
        weak = indexed.loc["cat_weak_1_3"] if "cat_weak_1_3" in indexed.index else None
        mid = indexed.loc["cat_mid_4"] if "cat_mid_4" in indexed.index else None
        strong = indexed.loc["cat_strong_5_plus"] if "cat_strong_5_plus" in indexed.index else None
        row = {"variant": variant}
        for metric in ["ndcg@20", *CORE_METRICS]:
            weak_value = weak[f"{metric}_mean"] if weak is not None and f"{metric}_mean" in weak else nan_float()
            mid_value = mid[f"{metric}_mean"] if mid is not None and f"{metric}_mean" in mid else nan_float()
            strong_value = strong[f"{metric}_mean"] if strong is not None and f"{metric}_mean" in strong else nan_float()
            row[f"weak_{metric}_mean"] = weak_value
            row[f"mid_minus_weak_{metric}"] = (
                mid_value - weak_value if pd.notna(mid_value) and pd.notna(weak_value) else nan_float()
            )
            row[f"strong_minus_weak_{metric}"] = (
                strong_value - weak_value if pd.notna(strong_value) and pd.notna(weak_value) else nan_float()
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    result["variant_order"] = result["variant"].map({variant: idx for idx, variant in enumerate(VARIANTS)})
    return result.sort_values("variant_order").drop(columns=["variant_order"]).reset_index(drop=True)


def build_delta_summary(profile: pd.DataFrame) -> pd.DataFrame:
    original = profile[profile["variant"].eq("original")][["asin", *CORE_METRICS]].copy()
    original = original.rename(columns={metric: f"original_{metric}" for metric in CORE_METRICS})
    merged = profile.merge(original, on="asin", how="left")
    for metric in CORE_METRICS:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"original_{metric}"]
    rows = []
    for (variant, group), sub in merged.groupby(["variant", "category_group"], dropna=False):
        row = {
            "variant": variant,
            "category_group": group,
            "item_count": len(sub),
        }
        for metric in CORE_METRICS:
            row[f"delta_{metric}_mean"] = sub[f"delta_{metric}"].mean()
            row[f"delta_{metric}_median"] = sub[f"delta_{metric}"].median()
        rows.append(row)
    result = pd.DataFrame(rows)
    result["variant_order"] = result["variant"].map({variant: idx for idx, variant in enumerate(VARIANTS)})
    return (
        result.sort_values(["variant_order", "category_group"], key=lambda s: s.map(category_order) if s.name == "category_group" else s)
        .drop(columns=["variant_order"])
        .reset_index(drop=True)
    )


def build_correlation_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, sub_profile in profile.groupby("variant"):
        for feature in CORE_METRICS:
            sub = sub_profile[[feature, "ndcg@20"]].dropna()
            if len(sub) < 3:
                pearson = nan_float()
                spearman = nan_float()
            else:
                pearson = float(sub.corr(method="pearson").iloc[0, 1])
                spearman = float(sub.corr(method="spearman").iloc[0, 1])
            rows.append(
                {
                    "variant": variant,
                    "feature": feature,
                    "metric": "ndcg@20",
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
    return pd.DataFrame(rows)


def write_result_markdown(
    output_path: Path,
    run_stamp: str,
    profile: pd.DataFrame,
    group_summary: pd.DataFrame,
    gap_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    correlation: pd.DataFrame,
    manifest_name: str,
) -> None:
    original_gap = gap_summary[gap_summary["variant"].eq("original")].iloc[0]
    image_only_gap = gap_summary[gap_summary["variant"].eq("image_only")].iloc[0]
    category_only_gap = gap_summary[gap_summary["variant"].eq("category_only")].iloc[0]
    all_category_gap = gap_summary[gap_summary["variant"].eq("all_category_upper")].iloc[0]

    mechanism_text = f"""核心结论：
移除类别后，weak 与 mid/strong 的 target score gap 明显缩小，说明 mid/strong 从类别路获得了更多收益，而 weak 没能同等利用类别信息；
category_only 下 target_history_q_cosine_mean 提升，但 target_score 下降且 best_target_rank 变差，说明类别路有内容对齐信号，却不足以把 weak 推进目标用户排序；
all_category_upper 与 image_only 几乎一致，说明“盲目打开更多类别”不是有效补救。

关键证据：
original mid_minus_weak_target_score_max = {original_gap["mid_minus_weak_target_score_max"]:.4f}，strong_minus_weak = {original_gap["strong_minus_weak_target_score_max"]:.4f}；
image_only mid_minus_weak_target_score_max = {image_only_gap["mid_minus_weak_target_score_max"]:.4f}，strong_minus_weak = {image_only_gap["strong_minus_weak_target_score_max"]:.4f}；
category_only mid_minus_weak_target_score_max = {category_only_gap["mid_minus_weak_target_score_max"]:.4f}，strong_minus_weak = {category_only_gap["strong_minus_weak_target_score_max"]:.4f}；
all_category_upper mid_minus_weak_target_score_max = {all_category_gap["mid_minus_weak_target_score_max"]:.4f}，strong_minus_weak = {all_category_gap["strong_minus_weak_target_score_max"]:.4f}。"""

    group_fields = field_callout(
        [
            ("variant", "前向输入 variant：original、image_only、category_only、all_category_upper。"),
            ("category_group", "按 item 类别属性数量得到的 weak/mid/strong 分组。"),
            ("item_count", "该组 item-variant 行数。"),
            ("target_score_max_mean", "真实目标用户最高 dot score 的组均值。"),
            ("margin_to_top20_cutoff_mean", "目标用户最高分减 top20 cutoff 的组均值，越大越好。"),
            ("best_target_rank_mean", "真实目标用户最佳排名的组均值，越小越好。"),
            ("target_history_q_cosine_mean_mean", "variant q_v_c 与目标用户历史 q_v_c 中心 cosine 均值的组均值。"),
            ("target_minus_top20_history_q_cosine_mean_mean", "目标用户历史中心对齐减 top20 用户历史中心对齐的组均值。"),
        ]
    )
    gap_fields = field_callout(
        [
            ("variant", "前向输入 variant。"),
            ("weak_target_score_max_mean", "weak 组 target_score_max 均值。"),
            ("mid_minus_weak_target_score_max", "mid 组 target_score_max 均值减 weak 组。"),
            ("strong_minus_weak_target_score_max", "strong 组 target_score_max 均值减 weak 组。"),
            ("mid_minus_weak_margin_to_top20_cutoff", "mid 组 top20 margin 均值减 weak 组。"),
            ("strong_minus_weak_margin_to_top20_cutoff", "strong 组 top20 margin 均值减 weak 组。"),
            ("mid_minus_weak_target_history_q_cosine_mean", "mid 组目标用户历史 q 对齐均值减 weak 组。"),
            ("strong_minus_weak_target_history_q_cosine_mean", "strong 组目标用户历史 q 对齐均值减 weak 组。"),
            ("注意", "best_target_rank 越小越好，因此 rank gap 要单独解释。"),
        ]
    )
    delta_fields = field_callout(
        [
            ("variant", "前向输入 variant；original 的 delta 恒为 0。"),
            ("category_group", "weak/mid/strong 分组。"),
            ("delta_target_score_max_mean", "该 variant 的 target_score_max 减 original target_score_max 后的组均值。"),
            ("delta_margin_to_top20_cutoff_mean", "该 variant 的 top20 margin 减 original top20 margin 后的组均值。"),
            ("delta_best_target_rank_mean", "该 variant 的 best target rank 减 original rank 后的组均值；正值表示排名变差。"),
            ("delta_target_history_q_cosine_mean_mean", "该 variant 的目标用户历史 q 对齐减 original 后的组均值。"),
        ]
    )
    corr_fields = field_callout(
        [
            ("variant", "前向输入 variant。"),
            ("feature", "候选解释变量。"),
            ("metric", "被解释指标，这里为 ndcg@20。"),
            ("spearman", "秩相关，适合看单调关系。"),
            ("pearson", "线性相关。"),
        ]
    )
    corr_top = correlation.copy()
    corr_top["abs_spearman"] = corr_top["spearman"].abs()
    corr_top = corr_top.sort_values(["variant", "abs_spearman"], ascending=[True, False])
    corr_top = corr_top.groupby("variant").head(4).reset_index(drop=True)

    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG category ablation 诊断结果
date: 2026-06-08
tags:
  - CCFCRec
  - Amazon-VG
  - 实验结果
  - 难度分层
  - 机制诊断
---

# {run_stamp} CCFCRec Amazon-VG category ablation 诊断结果

## 结论

这次诊断只解释 best64 checkpoint 的 category ablation 行为，不训练模型。

```text
{mechanism_text}
```

## 输入与输出

输入 item 数：

```text
test item count = {profile["asin"].nunique()}
variant row count = {len(profile)}
```

核心输出：

```text
category_ablation_profile.csv
category_ablation_variant_group_summary.csv
category_ablation_variant_gap_summary.csv
category_ablation_delta_summary.csv
category_ablation_correlation_summary.csv
{manifest_name}
```

## variant 分组表现

{group_fields}

{md_table(group_summary, ["variant", "category_group", "item_count", "target_score_max_mean", "margin_to_top20_cutoff_mean", "best_target_rank_mean", "target_history_q_cosine_mean_mean", "target_minus_top20_history_q_cosine_mean_mean"])}

## weak-mid/strong gap

{gap_fields}

{md_table(gap_summary, ["variant", "weak_target_score_max_mean", "mid_minus_weak_target_score_max", "strong_minus_weak_target_score_max", "mid_minus_weak_margin_to_top20_cutoff", "strong_minus_weak_margin_to_top20_cutoff", "mid_minus_weak_target_history_q_cosine_mean", "strong_minus_weak_target_history_q_cosine_mean"])}

## 相对 original 的 delta

{delta_fields}

{md_table(delta_summary[delta_summary["variant"].ne("original")], ["variant", "category_group", "delta_target_score_max_mean", "delta_margin_to_top20_cutoff_mean", "delta_best_target_rank_mean", "delta_target_history_q_cosine_mean_mean"])}

## 与 NDCG@20 的相关性

{corr_fields}

{md_table(corr_top, ["variant", "feature", "metric", "spearman", "pearson"])}

## 判读边界

```text
1. image_only 是“移除类别”的 counterfactual，不是可部署方法。
2. category_only 的图像置零是分布外输入，只能看机制方向。
3. all_category_upper 把所有类别打开，是不现实上界；如果它不能显著改善 weak，就更不支持简单类别补全。
4. category_only 的 margin 变好不能直接解释为性能变好，因为 target score 下降且 rank 变差，说明 top20 cutoff 也下降了。
5. 本结果仍需结合 user-activity matched 诊断；最终指标差距不能只归因于类别。
```

## 下一步

优先根据本诊断选择：

```text
1. 下一步不建议做盲目类别补全；
2. 更值得做 category-side reweighting / adaptive category confidence，让少类别 item 的可靠类别信号权重更合理；
3. 如果要设计方法实验，必须同时报告 user-activity / gt 分层，避免把用户活跃度混杂当成类别收益。
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
    if args.limit_items > 0:
        item_profile = item_profile.head(args.limit_items).copy()

    train_df = pd.read_csv(paths.dataset_dir / "train_rating.csv", usecols=["reviewerID", "asin"])
    test_df = pd.read_csv(paths.dataset_dir / "test_rating.csv", usecols=["reviewerID", "asin"])
    targets_by_asin = target_users_by_asin(test_df, save_dict["user_ser_dict"])
    test_asins = item_profile["asin"].tolist()

    train_asins = train_df["asin"].drop_duplicates().tolist()
    all_history_asins = list(dict.fromkeys([*train_asins, *test_asins]))
    history_asin_to_pos = {asin: index for index, asin in enumerate(all_history_asins)}
    history_q, history_attr, history_img = infer_content_vectors(
        model,
        all_history_asins,
        category_num,
        save_dict["asin_category_int_map"],
        save_dict["img_feature_dict"],
        device,
        args.batch_size,
    )
    q_centroids, _, _, interaction_counts = build_user_content_centroids(
        train_df,
        save_dict["user_ser_dict"],
        history_asin_to_pos,
        history_q,
        history_attr,
        history_img,
    )

    user_embedding = model.user_embedding.detach()
    user_norms = torch.norm(user_embedding, dim=1)
    rows = []

    for variant in VARIANTS:
        q_variant = infer_variant_q(
            model,
            test_asins,
            variant,
            category_num,
            save_dict["asin_category_int_map"],
            save_dict["img_feature_dict"],
            device,
            args.batch_size,
        )
        with torch.no_grad():
            for start in range(0, len(test_asins), args.score_batch_size):
                batch_asins = test_asins[start : start + args.score_batch_size]
                q_batch_np = q_variant[start : start + len(batch_asins)]
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
                    score_row = compute_single_score_margin(
                        scores_batch[row_idx],
                        target_info["target_user_indices"],
                        user_norms,
                        top_k=args.top_k,
                        analysis_top_k=args.analysis_top_k,
                    )
                    top20_indices = [int(value) for value in top20_batch[row_idx].detach().cpu().tolist()]
                    target_history = cosine_summary(
                        q_batch_np[row_idx],
                        q_centroids,
                        target_info["target_user_indices"],
                        interaction_counts,
                    )
                    top20_history = cosine_summary(
                        q_batch_np[row_idx],
                        q_centroids,
                        top20_indices,
                        interaction_counts,
                    )
                    target_mean = target_history["cosine_mean"]
                    top20_mean = top20_history["cosine_mean"]
                    output_row = {
                        "asin": asin,
                        "variant": variant,
                        "category_group": source_row["category_group"],
                        "category_count": source_row["category_count"],
                        "gt_group": source_row["gt_group"],
                        "gt_user_count": source_row["gt_user_count"],
                        "raw_test_user_count": source_row.get("raw_test_user_count", target_info["raw_target_user_count"]),
                        "raw_target_user_count": target_info["raw_target_user_count"],
                        "mapped_target_user_count": target_info["mapped_target_user_count"],
                        "unknown_target_user_count": target_info["unknown_target_user_count"],
                        "q_norm": float(np.linalg.norm(q_batch_np[row_idx])),
                        "target_history_q_cosine_mean": target_mean,
                        "target_history_q_cosine_max": target_history["cosine_max"],
                        "target_history_q_known_user_count": target_history["known_user_count"],
                        "top20_history_q_cosine_mean": top20_mean,
                        "top20_history_q_cosine_max": top20_history["cosine_max"],
                        "top20_history_q_known_user_count": top20_history["known_user_count"],
                        "target_minus_top20_history_q_cosine_mean": (
                            target_mean - top20_mean
                            if pd.notna(target_mean) and pd.notna(top20_mean)
                            else nan_float()
                        ),
                    }
                    for metric in METRICS:
                        output_row[metric] = source_row[metric]
                    output_row.update(score_row)
                    rows.append(output_row)

    profile = pd.DataFrame(rows)
    profile_path = paths.output_dir / "category_ablation_profile.csv"
    profile.to_csv(profile_path, index=False)

    group_summary = build_variant_group_summary(profile)
    group_summary_path = paths.output_dir / "category_ablation_variant_group_summary.csv"
    group_summary.to_csv(group_summary_path, index=False)

    gap_summary = build_variant_gap_summary(group_summary)
    gap_summary_path = paths.output_dir / "category_ablation_variant_gap_summary.csv"
    gap_summary.to_csv(gap_summary_path, index=False)

    delta_summary = build_delta_summary(profile)
    delta_summary_path = paths.output_dir / "category_ablation_delta_summary.csv"
    delta_summary.to_csv(delta_summary_path, index=False)

    correlation = build_correlation_summary(profile)
    correlation_path = paths.output_dir / "category_ablation_correlation_summary.csv"
    correlation.to_csv(correlation_path, index=False)

    result_md_path = paths.output_dir / f"{run_stamp} CCFCRec Amazon-VG category ablation 诊断结果.md"
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
        "output_dir": str(paths.output_dir),
        "device": device_name,
        "batch_size": args.batch_size,
        "score_batch_size": args.score_batch_size,
        "top_k": args.top_k,
        "analysis_top_k": args.analysis_top_k,
        "limit_items": args.limit_items,
        "variants": VARIANTS,
        "outputs": {
            "category_ablation_profile": str(profile_path),
            "category_ablation_variant_group_summary": str(group_summary_path),
            "category_ablation_variant_gap_summary": str(gap_summary_path),
            "category_ablation_delta_summary": str(delta_summary_path),
            "category_ablation_correlation_summary": str(correlation_path),
            "result_md": str(result_md_path),
        },
        "row_counts": {
            "category_ablation_profile": int(len(profile)),
            "variant_group_summary": int(len(group_summary)),
            "variant_gap_summary": int(len(gap_summary)),
            "delta_summary": int(len(delta_summary)),
            "correlation": int(len(correlation)),
        },
        "notes": [
            "只做本地诊断，不训练模型。",
            "all_category_upper 是不现实上界，不能当方法结果。",
            "category_only 的图像置零是分布外输入，只能解释机制方向。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_result_markdown(
        result_md_path,
        run_stamp,
        profile,
        group_summary,
        gap_summary,
        delta_summary,
        correlation,
        manifest_path.name,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG category ablation 诊断")
    parser.add_argument("--code-root", type=str, default="", help="CCFCRec-code 根目录；默认按脚本位置推断")
    parser.add_argument("--dataset-dir", type=str, default="", help="Amazon VG/data 目录；默认从 code-root 推断")
    parser.add_argument("--reproduction-tar", type=str, default="", help="workers45 瘦身 tar 路径")
    parser.add_argument("--item-profile", type=str, default="", help="自然难度诊断 item_profile.csv")
    parser.add_argument("--output-dir", type=str, default="", help="诊断输出目录")
    parser.add_argument("--save-dict-member", type=str, default="amazon_vg_slim/save_dict.pkl", help="tar 内 save_dict.pkl 成员名")
    parser.add_argument("--checkpoint-member", type=str, default="amazon_vg_slim/best_epoch_64.pt", help="tar 内 checkpoint 成员名")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="推理设备")
    parser.add_argument("--batch-size", type=int, default=128, help="前向 batch size")
    parser.add_argument("--score-batch-size", type=int, default=128, help="打分 batch size")
    parser.add_argument("--top-k", type=int, default=20, help="主诊断 top-k cutoff")
    parser.add_argument("--analysis-top-k", type=int, default=100, help="额外分析 top-k 范围")
    parser.add_argument("--limit-items", type=int, default=0, help="调试时限制 item 数；0 表示全量")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
