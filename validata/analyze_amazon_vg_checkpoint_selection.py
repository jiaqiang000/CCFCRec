#!/usr/bin/env python3
"""
CCFCRec Amazon-VG checkpoint selection 分组诊断。

脚本作用：
1. 从 workers45 瘦身 tar 中加载 save_dict.pkl、best_epoch_64.pt 和 100.pt；
2. 按官方 test.py 的 HR/NDCG 口径重新计算 test item-level 指标；
3. 比较 overall best checkpoint 与 weak/mid/strong 分组 best checkpoint 是否一致；
4. 输出 item-level 长表、overall/category/gt 分组汇总、delta、selection summary 和结果 Markdown。

这个脚本只做本地诊断，不训练模型，不修改 CCFCRec 主训练入口。
"""

from __future__ import annotations

import argparse
import json
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


DEFAULT_CHECKPOINTS = [
    "epoch64_best_ndcg20=amazon_vg_slim/best_epoch_64.pt",
    "epoch100_last=amazon_vg_slim/100.pt",
]


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    member: str
    order: int


@dataclass(frozen=True)
class Paths:
    project_root: Path
    code_root: Path
    amazon_code_dir: Path
    dataset_dir: Path
    reproduction_tar: Path
    item_metrics: Path
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
        / "checkpoint-selection-diagnostic"
    )
    return Paths(
        project_root=project_root,
        code_root=code_root,
        amazon_code_dir=amazon_code_dir,
        dataset_dir=dataset_dir,
        reproduction_tar=reproduction_tar,
        item_metrics=item_metrics,
        output_dir=output_dir,
    )


def parse_checkpoint_specs(values: list[str] | None) -> list[CheckpointSpec]:
    raw_values = values if values else DEFAULT_CHECKPOINTS
    specs = []
    labels = set()
    for order, raw_value in enumerate(raw_values):
        if "=" not in raw_value:
            raise ValueError(f"checkpoint 必须是 label=tar_member 格式: {raw_value}")
        label, member = raw_value.split("=", 1)
        label = label.strip()
        member = member.strip()
        if not label or not member:
            raise ValueError(f"checkpoint label/member 不能为空: {raw_value}")
        if label in labels:
            raise ValueError(f"checkpoint label 重复: {label}")
        labels.add(label)
        specs.append(CheckpointSpec(label=label, member=member, order=order))
    return specs


def dcg_k(labels: list[float] | np.ndarray) -> float:
    values = np.asarray(labels, dtype=np.float64)
    if values.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, values.size + 2))
    return float(np.sum((np.power(2.0, values) - 1.0) / discounts))


def ndcg_from_recommended(recommended_users: list[int], target_users: set[int], k: int) -> float:
    labels = [1.0 if user in target_users else 0.0 for user in recommended_users[:k]]
    ideal = sorted(labels, reverse=True)
    ideal_dcg = dcg_k(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_k(labels) / ideal_dcg


def compute_item_ranking_metrics(
    recommended_users: list[int],
    target_user_indices: list[int],
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    """按官方 test.py 口径计算单个 item 的 HR/NDCG。"""
    target_users = set(int(user) for user in target_user_indices)
    row: dict[str, float] = {}
    for k in ks:
        top_users = recommended_users[:k]
        hit_count = len(target_users.intersection(top_users))
        row[f"hr@{k}"] = hit_count / k
        row[f"ndcg@{k}"] = ndcg_from_recommended(recommended_users, target_users, k)
    return row


def evaluate_checkpoint(
    model: torch.nn.Module,
    checkpoint: CheckpointSpec,
    item_profile: pd.DataFrame,
    targets_by_asin: dict[str, dict[str, object]],
    category_num: int,
    category_map: dict[str, list[int]],
    img_feature_dict: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    top_k: int,
) -> pd.DataFrame:
    rows = []
    asins = item_profile["asin"].tolist()
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
            q_v_c = model(attributes, images, len(batch_asins))
            scores_batch = torch.matmul(q_v_c, model.user_embedding.detach().T)
            _, top_indices = torch.topk(scores_batch, k=top_k, dim=1, largest=True)
            top_indices = top_indices.detach().cpu().numpy()

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
                metric_row = compute_item_ranking_metrics(
                    top_indices[row_idx].astype(int).tolist(),
                    target_info["target_user_indices"],
                    ks=(5, 10, 20),
                )
                rows.append(
                    {
                        "checkpoint_label": checkpoint.label,
                        "checkpoint_member": checkpoint.member,
                        "checkpoint_order": checkpoint.order,
                        "asin": asin,
                        "category_count": source_row["category_count"],
                        "category_group": source_row["category_group"],
                        "gt_user_count": source_row["gt_user_count"],
                        "gt_group": source_row["gt_group"],
                        "raw_target_user_count": target_info["raw_target_user_count"],
                        "mapped_target_user_count": target_info["mapped_target_user_count"],
                        "unknown_target_user_count": target_info["unknown_target_user_count"],
                        **metric_row,
                    }
                )
    return pd.DataFrame(rows)


def build_overall_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (checkpoint_order, checkpoint_label, checkpoint_member), sub in profile.groupby(
        ["checkpoint_order", "checkpoint_label", "checkpoint_member"],
        dropna=False,
    ):
        row = {
            "checkpoint_order": checkpoint_order,
            "checkpoint_label": checkpoint_label,
            "checkpoint_member": checkpoint_member,
            "item_count": len(sub),
            "gt_user_count_mean": sub["gt_user_count"].mean(),
            "mapped_target_user_count_mean": sub["mapped_target_user_count"].mean(),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = sub[metric].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("checkpoint_order").reset_index(drop=True)


def build_group_summary(profile: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for (checkpoint_order, checkpoint_label, group_value), sub in profile.groupby(
        ["checkpoint_order", "checkpoint_label", group_col],
        dropna=False,
    ):
        row = {
            "checkpoint_order": checkpoint_order,
            "checkpoint_label": checkpoint_label,
            group_col: group_value,
            "item_count": len(sub),
            "gt_user_count_mean": sub["gt_user_count"].mean(),
            "gt_user_count_median": sub["gt_user_count"].median(),
            "mapped_target_user_count_mean": sub["mapped_target_user_count"].mean(),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = sub[metric].mean()
        rows.append(row)
    result = pd.DataFrame(rows)
    if group_col == "category_group":
        return result.sort_values(
            ["checkpoint_order", group_col],
            key=lambda s: s.map(category_order) if s.name == group_col else s,
        ).reset_index(drop=True)
    return result.sort_values(["checkpoint_order", group_col]).reset_index(drop=True)


def build_checkpoint_delta_summary(
    profile: pd.DataFrame,
    baseline_label: str,
    compare_label: str,
) -> pd.DataFrame:
    baseline = profile[profile["checkpoint_label"].eq(baseline_label)].copy()
    compare = profile[profile["checkpoint_label"].eq(compare_label)].copy()
    baseline_cols = ["asin", *METRICS]
    merged = compare.merge(
        baseline[baseline_cols].rename(columns={metric: f"baseline_{metric}" for metric in METRICS}),
        on="asin",
        how="inner",
    )
    for metric in METRICS:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"baseline_{metric}"]

    scopes = [
        ("overall", None),
        ("category_group", "category_group"),
        ("gt_group", "gt_group"),
    ]
    rows = []
    for scope, group_col in scopes:
        groups = [("all", merged)] if group_col is None else merged.groupby(group_col, dropna=False)
        for group_value, sub in groups:
            row = {
                "baseline_checkpoint": baseline_label,
                "compare_checkpoint": compare_label,
                "scope": scope,
                "group_value": group_value,
                "item_count": len(sub),
            }
            for metric in METRICS:
                delta_col = f"delta_{metric}"
                row[f"{delta_col}_mean"] = sub[delta_col].mean()
                row[f"{delta_col}_median"] = sub[delta_col].median()
                row[f"{metric}_improved_rate"] = float((sub[delta_col] > 0).mean())
                row[f"{metric}_declined_rate"] = float((sub[delta_col] < 0).mean())
            rows.append(row)
    result = pd.DataFrame(rows)
    result["scope_order"] = result["scope"].map({"overall": 0, "category_group": 1, "gt_group": 2})
    result["group_order"] = result["group_value"].map(category_order)
    result["group_order"] = result["group_order"].fillna(99)
    return result.sort_values(["scope_order", "group_order", "group_value"]).drop(
        columns=["scope_order", "group_order"]
    ).reset_index(drop=True)


def build_checkpoint_selection_summary(
    overall_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    gt_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    scopes = [
        ("overall", "group_value", overall_summary.assign(group_value="all")),
        ("category_group", "category_group", category_summary),
        ("gt_group", "gt_group", gt_summary),
    ]
    for scope, group_col, summary in scopes:
        for group_value, sub in summary.groupby(group_col, dropna=False):
            for metric in METRICS:
                metric_col = f"{metric}_mean"
                ranked = sub.sort_values(metric_col, ascending=False).reset_index(drop=True)
                best = ranked.iloc[0]
                second = ranked.iloc[1] if len(ranked) > 1 else None
                second_value = second[metric_col] if second is not None else nan_float()
                rows.append(
                    {
                        "scope": scope,
                        "group_value": group_value,
                        "metric": metric,
                        "best_checkpoint": best["checkpoint_label"],
                        "best_value": best[metric_col],
                        "second_checkpoint": second["checkpoint_label"] if second is not None else "",
                        "second_value": second_value,
                        "best_minus_second": best[metric_col] - second_value if second is not None else nan_float(),
                    }
                )
    result = pd.DataFrame(rows)
    result["scope_order"] = result["scope"].map({"overall": 0, "category_group": 1, "gt_group": 2})
    result["group_order"] = result["group_value"].map(category_order)
    result["group_order"] = result["group_order"].fillna(99)
    result["metric_order"] = result["metric"].map({metric: idx for idx, metric in enumerate(METRICS)})
    return result.sort_values(["scope_order", "group_order", "group_value", "metric_order"]).drop(
        columns=["scope_order", "group_order", "metric_order"]
    ).reset_index(drop=True)


def build_consistency_summary(profile: pd.DataFrame, source_item_metrics: pd.DataFrame, checkpoint_label: str) -> pd.DataFrame:
    checkpoint_profile = profile[profile["checkpoint_label"].eq(checkpoint_label)]
    merged = checkpoint_profile.merge(
        source_item_metrics[["asin", *METRICS]].rename(columns={metric: f"source_{metric}" for metric in METRICS}),
        on="asin",
        how="inner",
    )
    rows = []
    for metric in METRICS:
        diff = merged[metric] - merged[f"source_{metric}"]
        rows.append(
            {
                "checkpoint_label": checkpoint_label,
                "metric": metric,
                "matched_item_count": len(merged),
                "mean_abs_diff": diff.abs().mean(),
                "max_abs_diff": diff.abs().max(),
            }
        )
    return pd.DataFrame(rows)


def write_result_markdown(
    output_path: Path,
    run_stamp: str,
    checkpoints: list[CheckpointSpec],
    item_count: int,
    overall_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    gt_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
    consistency_summary: pd.DataFrame,
    manifest_name: str,
) -> None:
    overall_ndcg = selection_summary[
        selection_summary["scope"].eq("overall") & selection_summary["metric"].eq("ndcg@20")
    ].iloc[0]
    weak_ndcg_rows = selection_summary[
        selection_summary["scope"].eq("category_group")
        & selection_summary["group_value"].eq("cat_weak_1_3")
        & selection_summary["metric"].eq("ndcg@20")
    ]
    weak_ndcg = weak_ndcg_rows.iloc[0] if len(weak_ndcg_rows) else None
    weak_delta_rows = delta_summary[
        delta_summary["scope"].eq("category_group") & delta_summary["group_value"].eq("cat_weak_1_3")
    ]
    weak_delta = weak_delta_rows.iloc[0] if len(weak_delta_rows) else None
    all_delta = delta_summary[delta_summary["scope"].eq("overall") & delta_summary["group_value"].eq("all")].iloc[0]

    if weak_ndcg is not None and weak_delta is not None:
        if weak_ndcg["best_checkpoint"] == overall_ndcg["best_checkpoint"]:
            conclusion = f"""核心结论：
overall NDCG@20 最优 checkpoint 与 weak group NDCG@20 最优 checkpoint 一致，都是 {overall_ndcg["best_checkpoint"]}；
因此当前 weak group 低表现不是简单由“只选择 overall best checkpoint”造成。
关键差异：
{all_delta["compare_checkpoint"]} 相对 {all_delta["baseline_checkpoint"]} 的 overall NDCG@20 delta = {all_delta["delta_ndcg@20_mean"]:.4f}；
cat_weak_1_3 的 NDCG@20 delta = {weak_delta["delta_ndcg@20_mean"]:.4f}，improved_rate = {weak_delta["ndcg@20_improved_rate"]:.4f}。"""
        else:
            conclusion = f"""核心结论：
overall NDCG@20 最优 checkpoint 是 {overall_ndcg["best_checkpoint"]}，
但 cat_weak_1_3 NDCG@20 最优 checkpoint 是 {weak_ndcg["best_checkpoint"]}；
这说明后续实验需要报告 subgroup-aware checkpoint selection。
关键差异：
{all_delta["compare_checkpoint"]} 相对 {all_delta["baseline_checkpoint"]} 的 overall NDCG@20 delta = {all_delta["delta_ndcg@20_mean"]:.4f}；
cat_weak_1_3 的 NDCG@20 delta = {weak_delta["delta_ndcg@20_mean"]:.4f}，improved_rate = {weak_delta["ndcg@20_improved_rate"]:.4f}。"""
    else:
        conclusion = "核心结论：本轮完成 checkpoint 分组比较；需查看 selection summary 判断 overall 与 weak 最优 checkpoint 是否一致。"

    category_delta = delta_summary[delta_summary["scope"].eq("category_group")]
    gt_delta = delta_summary[delta_summary["scope"].eq("gt_group")]
    selection_ndcg20 = selection_summary[selection_summary["metric"].eq("ndcg@20")]

    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG checkpoint selection 分组诊断结果
date: 2026-06-08
tags:
  - CCFCRec
  - Amazon-VG
  - 实验结果
  - 难度分层
  - checkpoint-selection
---

# {run_stamp} CCFCRec Amazon-VG checkpoint selection 分组诊断结果

## 结论

这次诊断只比较本地 slim 包已有 checkpoint，不训练模型。

```text
{conclusion}
```

## 输入与输出

输入 item 数：

```text
test item count = {item_count}
checkpoints = {", ".join(checkpoint.label for checkpoint in checkpoints)}
```

核心输出：

```text
checkpoint_item_metrics.csv
checkpoint_overall_summary.csv
checkpoint_category_group_summary.csv
checkpoint_gt_group_summary.csv
checkpoint_delta_summary.csv
checkpoint_selection_summary.csv
checkpoint_consistency_summary.csv
{manifest_name}
```

## overall 指标

> [!info] 字段说明
> - `checkpoint_label`：checkpoint 名称。
> - `item_count`：参与 test 复评的 item 数。
> - `gt_user_count_mean`：item-level 复评口径下目标用户数均值。
> - `mapped_target_user_count_mean`：能映射进 user embedding 空间的目标用户数均值。
> - `hr@5_mean`：官方口径 item-level HR@5 均值。
> - `hr@10_mean`：官方口径 item-level HR@10 均值。
> - `hr@20_mean`：官方口径 item-level HR@20 均值。
> - `ndcg@5_mean`：官方口径 item-level NDCG@5 均值。
> - `ndcg@10_mean`：官方口径 item-level NDCG@10 均值。
> - `ndcg@20_mean`：官方口径 item-level NDCG@20 均值。

{md_table(overall_summary, ["checkpoint_label", "item_count", "gt_user_count_mean", "mapped_target_user_count_mean", "hr@5_mean", "hr@10_mean", "hr@20_mean", "ndcg@5_mean", "ndcg@10_mean", "ndcg@20_mean"])}

## 类别证据分组

> [!info] 字段说明
> - `checkpoint_label`：checkpoint 名称。
> - `category_group`：按类别属性数量得到的 weak/mid/strong 分组。
> - `item_count`：该 checkpoint 下该组 item 数。
> - `gt_user_count_mean`：该组目标用户数均值。
> - `mapped_target_user_count_mean`：该组可映射目标用户数均值。
> - `hr@20_mean`：该组官方口径 item-level HR@20 均值。
> - `ndcg@20_mean`：该组官方口径 item-level NDCG@20 均值。

{md_table(category_summary, ["checkpoint_label", "category_group", "item_count", "gt_user_count_mean", "mapped_target_user_count_mean", "hr@20_mean", "ndcg@20_mean"])}

## epoch100 相对 epoch64 的 delta

> [!info] 字段说明
> - `baseline_checkpoint`：基准 checkpoint。
> - `compare_checkpoint`：被比较 checkpoint。
> - `scope`：比较范围，overall、category_group 或 gt_group。
> - `group_value`：该范围内的组名。
> - `item_count`：参与 delta 的 item 数。
> - `delta_hr@20_mean`：compare 的 HR@20 减 baseline 的 HR@20 后的均值。
> - `delta_ndcg@20_mean`：compare 的 NDCG@20 减 baseline 的 NDCG@20 后的均值。
> - `ndcg@20_improved_rate`：NDCG@20 在 item-level 上 compare 大于 baseline 的比例。
> - `ndcg@20_declined_rate`：NDCG@20 在 item-level 上 compare 小于 baseline 的比例。

{md_table(category_delta, ["baseline_checkpoint", "compare_checkpoint", "scope", "group_value", "item_count", "delta_hr@20_mean", "delta_ndcg@20_mean", "ndcg@20_improved_rate", "ndcg@20_declined_rate"])}

## gt 用户数分组 delta

> [!info] 字段说明
> - `baseline_checkpoint`：基准 checkpoint。
> - `compare_checkpoint`：被比较 checkpoint。
> - `scope`：这里为 gt_group。
> - `group_value`：gt 用户数分组。
> - `item_count`：参与 delta 的 item 数。
> - `delta_hr@20_mean`：compare 的 HR@20 减 baseline 的 HR@20 后的均值。
> - `delta_ndcg@20_mean`：compare 的 NDCG@20 减 baseline 的 NDCG@20 后的均值。
> - `ndcg@20_improved_rate`：NDCG@20 在 item-level 上 compare 大于 baseline 的比例。
> - `ndcg@20_declined_rate`：NDCG@20 在 item-level 上 compare 小于 baseline 的比例。

{md_table(gt_delta, ["baseline_checkpoint", "compare_checkpoint", "scope", "group_value", "item_count", "delta_hr@20_mean", "delta_ndcg@20_mean", "ndcg@20_improved_rate", "ndcg@20_declined_rate"])}

## NDCG@20 最优 checkpoint

> [!info] 字段说明
> - `scope`：比较范围，overall、category_group 或 gt_group。
> - `group_value`：该范围内的组名。
> - `metric`：用于选择 checkpoint 的指标。
> - `best_checkpoint`：该 scope/group/metric 下最优 checkpoint。
> - `best_value`：最优 checkpoint 的指标值。
> - `second_checkpoint`：第二名 checkpoint。
> - `second_value`：第二名指标值。
> - `best_minus_second`：最优值减第二名值。

{md_table(selection_ndcg20, ["scope", "group_value", "metric", "best_checkpoint", "best_value", "second_checkpoint", "second_value", "best_minus_second"])}

## 一致性校验

> [!info] 字段说明
> - `checkpoint_label`：被校验 checkpoint。
> - `metric`：校验指标。
> - `matched_item_count`：和既有 best64 item-level CSV 对齐的 item 数。
> - `mean_abs_diff`：本脚本复算值与既有 CSV 的平均绝对差。
> - `max_abs_diff`：本脚本复算值与既有 CSV 的最大绝对差。

{md_table(consistency_summary, ["checkpoint_label", "metric", "matched_item_count", "mean_abs_diff", "max_abs_diff"])}

## 判读边界

```text
1. 本轮只比较 slim 包里的两个 checkpoint，不代表 1-100 全部 checkpoint 的 subgroup best。
2. 如果两个 checkpoint 的 weak 都差，说明下一步应进入 category-side reweighting / adaptive category confidence 方法设计。
3. 如果 weak 最优 checkpoint 与 overall 最优 checkpoint 不一致，后续方法实验必须报告 subgroup-aware checkpoint selection。
4. 如果一致，则当前 weak 低表现不是简单由 checkpoint selection 造成。
```
"""
    output_path.write_text(markdown, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp, run_iso = now_stamp()
    device_name = select_device(args.device)
    device = torch.device(device_name)
    checkpoints = parse_checkpoint_specs(args.checkpoint)

    save_dict = load_pickle_from_tar(paths.reproduction_tar, args.save_dict_member)
    item_profile = pd.read_csv(paths.item_metrics)
    if args.limit_items > 0:
        item_profile = item_profile.head(args.limit_items).copy()
    test_df = pd.read_csv(paths.dataset_dir / "test_rating.csv", usecols=["reviewerID", "asin"])
    targets_by_asin = target_users_by_asin(test_df, save_dict["user_ser_dict"])
    category_num = int(save_dict["category_ser_map_len"])

    frames = []
    for checkpoint in checkpoints:
        model = load_model(paths, device_name, checkpoint.member)
        frames.append(
            evaluate_checkpoint(
                model=model,
                checkpoint=checkpoint,
                item_profile=item_profile,
                targets_by_asin=targets_by_asin,
                category_num=category_num,
                category_map=save_dict["asin_category_int_map"],
                img_feature_dict=save_dict["img_feature_dict"],
                device=device,
                batch_size=args.batch_size,
                top_k=args.top_k,
            )
        )
        del model

    checkpoint_profile = pd.concat(frames, ignore_index=True)
    item_metrics_path = paths.output_dir / "checkpoint_item_metrics.csv"
    checkpoint_profile.to_csv(item_metrics_path, index=False)

    overall_summary = build_overall_summary(checkpoint_profile)
    overall_summary_path = paths.output_dir / "checkpoint_overall_summary.csv"
    overall_summary.to_csv(overall_summary_path, index=False)

    category_summary = build_group_summary(checkpoint_profile, "category_group")
    category_summary_path = paths.output_dir / "checkpoint_category_group_summary.csv"
    category_summary.to_csv(category_summary_path, index=False)

    gt_summary = build_group_summary(checkpoint_profile, "gt_group")
    gt_summary_path = paths.output_dir / "checkpoint_gt_group_summary.csv"
    gt_summary.to_csv(gt_summary_path, index=False)

    baseline_label = checkpoints[0].label
    compare_label = checkpoints[1].label if len(checkpoints) > 1 else checkpoints[0].label
    delta_summary = build_checkpoint_delta_summary(checkpoint_profile, baseline_label, compare_label)
    delta_summary_path = paths.output_dir / "checkpoint_delta_summary.csv"
    delta_summary.to_csv(delta_summary_path, index=False)

    selection_summary = build_checkpoint_selection_summary(overall_summary, category_summary, gt_summary)
    selection_summary_path = paths.output_dir / "checkpoint_selection_summary.csv"
    selection_summary.to_csv(selection_summary_path, index=False)

    consistency_summary = build_consistency_summary(checkpoint_profile, item_profile, baseline_label)
    consistency_summary_path = paths.output_dir / "checkpoint_consistency_summary.csv"
    consistency_summary.to_csv(consistency_summary_path, index=False)

    result_md_name = f"{run_stamp} CCFCRec Amazon-VG checkpoint selection 分组诊断结果.md"
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
        "item_metrics": str(paths.item_metrics),
        "output_dir": str(paths.output_dir),
        "device": device_name,
        "batch_size": args.batch_size,
        "top_k": args.top_k,
        "limit_items": args.limit_items,
        "checkpoints": [
            {"label": checkpoint.label, "member": checkpoint.member, "order": checkpoint.order}
            for checkpoint in checkpoints
        ],
        "outputs": {
            "checkpoint_item_metrics": str(item_metrics_path),
            "checkpoint_overall_summary": str(overall_summary_path),
            "checkpoint_category_group_summary": str(category_summary_path),
            "checkpoint_gt_group_summary": str(gt_summary_path),
            "checkpoint_delta_summary": str(delta_summary_path),
            "checkpoint_selection_summary": str(selection_summary_path),
            "checkpoint_consistency_summary": str(consistency_summary_path),
            "result_md": str(result_md_path),
        },
        "row_counts": {
            "checkpoint_item_metrics": int(len(checkpoint_profile)),
            "checkpoint_overall_summary": int(len(overall_summary)),
            "checkpoint_category_group_summary": int(len(category_summary)),
            "checkpoint_gt_group_summary": int(len(gt_summary)),
            "checkpoint_delta_summary": int(len(delta_summary)),
            "checkpoint_selection_summary": int(len(selection_summary)),
            "checkpoint_consistency_summary": int(len(consistency_summary)),
        },
        "notes": [
            "只做本地诊断，不训练模型。",
            "HR/NDCG 口径保持和官方 test.py 一致。",
            "本轮只比较 slim 包中已有 checkpoint；不遍历 1-100 全部 checkpoint。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    write_result_markdown(
        output_path=result_md_path,
        run_stamp=run_stamp,
        checkpoints=checkpoints,
        item_count=int(item_profile["asin"].nunique()),
        overall_summary=overall_summary,
        category_summary=category_summary,
        gt_summary=gt_summary,
        delta_summary=delta_summary,
        selection_summary=selection_summary,
        consistency_summary=consistency_summary,
        manifest_name=manifest_path.name,
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG checkpoint selection 分组诊断")
    parser.add_argument("--code-root", type=str, default="", help="CCFCRec-code 根目录；默认按脚本位置推断")
    parser.add_argument("--dataset-dir", type=str, default="", help="Amazon VG/data 目录；默认从 code-root 推断")
    parser.add_argument("--reproduction-tar", type=str, default="", help="workers45 瘦身 tar 路径")
    parser.add_argument("--item-metrics", type=str, default="", help="best64 item-level test metrics CSV")
    parser.add_argument("--output-dir", type=str, default="", help="诊断输出目录")
    parser.add_argument("--save-dict-member", type=str, default="amazon_vg_slim/save_dict.pkl", help="tar 内 save_dict.pkl 成员名")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="checkpoint 规格，格式 label=tar_member；可重复指定",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="推理设备")
    parser.add_argument("--batch-size", type=int, default=64, help="item 推理 batch size")
    parser.add_argument("--top-k", type=int, default=20, help="最大推荐用户数；默认 20")
    parser.add_argument("--limit-items", type=int, default=0, help="调试时限制 item 数；0 表示全量")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
