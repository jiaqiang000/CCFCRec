#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3 checkpoint 复评。

脚本作用：
1. 把 v2 category_availability_v2_item.csv 转成旧诊断脚本可读的 item_profile；
2. 用 v2 s_cat_group 复评 baseline 与 category_conf_input 的旧 checkpoint；
3. 输出分组 HR/NDCG、delta、selection summary、variant 专属 item_profile 和结果 Markdown。

这个脚本只做本地诊断，不训练模型，不修改 CCFCRec 主训练入口。
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from analyze_amazon_vg_checkpoint_selection import (
    CheckpointSpec,
    build_checkpoint_delta_summary,
    build_checkpoint_selection_summary,
    build_group_summary,
    build_overall_summary,
    evaluate_checkpoint,
)
from analyze_amazon_vg_natural_difficulty_confounding import gt_group
from analyze_amazon_vg_score_norm_margin import (
    METRICS,
    load_model,
    load_pickle_from_tar,
    md_table,
    now_stamp,
    select_device,
)
from analyze_amazon_vg_target_score_source import target_users_by_asin


TASK3_PROFILE_COLUMNS = [
    "asin",
    "category_count",
    "category_group",
    "gt_user_count",
    "gt_group",
    "raw_test_user_count",
    "s_cat",
    "s_cat_group",
    "s_cat_v1",
    "s_cat_group_v1",
    "s_cat_v2_disc_within_control",
    "s_cat_v2_collab_within_control",
]


@dataclass(frozen=True)
class VariantSpec:
    label: str
    tar_path: Path
    save_dict_member: str
    checkpoint_member: str
    order: int


def slugify_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


def build_task3_item_profile(
    availability: pd.DataFrame,
    test_rating: pd.DataFrame,
    group_col: str = "s_cat_group",
) -> pd.DataFrame:
    required = {
        "raw_asin",
        "split",
        "category_count",
        group_col,
        "s_cat",
        "s_cat_v1",
        "s_cat_group_v1",
        "s_cat_v2_disc_within_control",
        "s_cat_v2_collab_within_control",
    }
    missing = sorted(required.difference(availability.columns))
    if missing:
        raise ValueError(f"availability 缺少必要列: {missing}")
    if not {"reviewerID", "asin"}.issubset(test_rating.columns):
        raise ValueError("test_rating 需要包含 reviewerID 和 asin 列")

    target_counts = (
        test_rating.groupby("asin", dropna=False)["reviewerID"]
        .nunique()
        .rename("gt_user_count")
        .reset_index()
    )

    profile = availability[availability["split"].eq("test")].copy()
    profile = profile.merge(target_counts, left_on="raw_asin", right_on="asin", how="left")
    profile["asin"] = profile["raw_asin"]
    profile["gt_user_count"] = profile["gt_user_count"].fillna(0).astype(int)
    profile["raw_test_user_count"] = profile["gt_user_count"]
    profile["gt_group"] = profile["gt_user_count"].map(gt_group)
    profile["category_group"] = profile[group_col]

    for column in TASK3_PROFILE_COLUMNS:
        if column not in profile.columns:
            profile[column] = pd.NA
    return profile[TASK3_PROFILE_COLUMNS].sort_values("asin").reset_index(drop=True)


def build_task3_comparison_summary(
    baseline_group_summary: pd.DataFrame,
    category_conf_group_summary: pd.DataFrame,
    baseline_label: str = "baseline",
    category_conf_label: str = "category_conf_input",
) -> pd.DataFrame:
    baseline = baseline_group_summary.rename(
        columns={
            "item_count": "baseline_item_count",
            "hr@20_mean": "baseline_hr@20_mean",
            "ndcg@20_mean": "baseline_ndcg@20_mean",
        }
    )[["category_group", "baseline_item_count", "baseline_hr@20_mean", "baseline_ndcg@20_mean"]]
    category_conf = category_conf_group_summary.rename(
        columns={
            "item_count": "category_conf_item_count",
            "hr@20_mean": "category_conf_hr@20_mean",
            "ndcg@20_mean": "category_conf_ndcg@20_mean",
        }
    )[["category_group", "category_conf_item_count", "category_conf_hr@20_mean", "category_conf_ndcg@20_mean"]]
    merged = baseline.merge(category_conf, on="category_group", how="outer")
    merged["baseline_label"] = baseline_label
    merged["category_conf_label"] = category_conf_label
    merged["delta_hr@20_mean"] = merged["category_conf_hr@20_mean"] - merged["baseline_hr@20_mean"]
    merged["delta_ndcg@20_mean"] = merged["category_conf_ndcg@20_mean"] - merged["baseline_ndcg@20_mean"]
    columns = [
        "baseline_label",
        "category_conf_label",
        "category_group",
        "baseline_item_count",
        "category_conf_item_count",
        "baseline_hr@20_mean",
        "category_conf_hr@20_mean",
        "delta_hr@20_mean",
        "baseline_ndcg@20_mean",
        "category_conf_ndcg@20_mean",
        "delta_ndcg@20_mean",
    ]
    return merged[columns].sort_values("category_group").reset_index(drop=True)


def parse_variant(value: str, order: int) -> VariantSpec:
    parts = value.split("|")
    if len(parts) != 4:
        raise ValueError("variant 必须是 label|tar_path|save_dict_member|checkpoint_member")
    label, tar_path, save_dict_member, checkpoint_member = [part.strip() for part in parts]
    if not label or not tar_path or not save_dict_member or not checkpoint_member:
        raise ValueError(f"variant 字段不能为空: {value}")
    return VariantSpec(
        label=label,
        tar_path=Path(tar_path).expanduser().resolve(),
        save_dict_member=save_dict_member,
        checkpoint_member=checkpoint_member,
        order=order,
    )


def evaluate_variant(
    variant: VariantSpec,
    item_profile: pd.DataFrame,
    test_df: pd.DataFrame,
    amazon_code_dir: Path,
    device_name: str,
    batch_size: int,
    top_k: int,
) -> pd.DataFrame:
    save_dict = load_pickle_from_tar(variant.tar_path, variant.save_dict_member)
    targets_by_asin = target_users_by_asin(test_df, save_dict["user_ser_dict"])
    paths = SimpleNamespace(amazon_code_dir=amazon_code_dir, reproduction_tar=variant.tar_path)
    model = load_model(paths, device_name, variant.checkpoint_member)
    try:
        return evaluate_checkpoint(
            model=model,
            checkpoint=CheckpointSpec(
                label=variant.label,
                member=variant.checkpoint_member,
                order=variant.order,
            ),
            item_profile=item_profile,
            targets_by_asin=targets_by_asin,
            category_num=int(save_dict["category_ser_map_len"]),
            category_map=save_dict["asin_category_int_map"],
            img_feature_dict=save_dict["img_feature_dict"],
            device=torch.device(device_name),
            batch_size=batch_size,
            top_k=top_k,
        )
    finally:
        del model


def write_task3_markdown(
    output_path: Path,
    run_stamp: str,
    source_notes: list[str],
    result_dir_display: str,
    item_count: int,
    variants: list[VariantSpec],
    overall_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
    manifest_name: str,
) -> None:
    source_lines = "\n".join(f"> {note}" for note in source_notes)
    if not source_lines:
        source_lines = "> 上游设计：未提供"
    variant_text = "\n".join(
        f"{variant.label}: {variant.tar_path.name} / {variant.checkpoint_member}"
        for variant in variants
    )
    category_selection = selection_summary[
        selection_summary["scope"].eq("category_group") & selection_summary["metric"].eq("ndcg@20")
    ].copy()
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG category availability v2 Task3 checkpoint复评诊断结果
date: {run_stamp[:10]}
time: "{run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
created_at: "{run_stamp[:10]} {run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - Task3
  - checkpoint复评
---

# {run_stamp} CCFCRec Amazon-VG category availability v2 Task3 checkpoint复评诊断结果

## 来源说明

> [!info] 来源说明
{source_lines}
> 本结果目录：`{result_dir_display}`

## 结论

本报告只复评旧 checkpoint，不训练模型，不修改模型结构。结论只描述 v2 `s_cat_group` 分层下 baseline 与 category_conf_input 的局部差异。

## 数据验收

> [!info] 字段说明
> `field`：验收项。
> `value`：验收项取值。

| field | value |
|---|---:|
| item_count | {item_count} |
| variants | `{", ".join(variant.label for variant in variants)}` |
| manifest | `{manifest_name}` |

## Checkpoint

```text
{variant_text}
```

## Overall 结果

> [!info] 字段说明
> `checkpoint_label`：被复评的 checkpoint 标签。
> `item_count`：参与复评的 test item 数。
> `hr@20_mean`：item-level HR@20 均值。
> `ndcg@20_mean`：item-level NDCG@20 均值。

{md_table(overall_summary, ["checkpoint_label", "item_count", "hr@20_mean", "ndcg@20_mean"])}

## v2 s_cat_group 分组结果

> [!info] 字段说明
> `checkpoint_label`：被复评的 checkpoint 标签。
> `category_group`：v2 `s_cat_group`，不是旧 category_count 分组。
> `item_count`：该组 test item 数。
> `hr@20_mean`：该组 item-level HR@20 均值。
> `ndcg@20_mean`：该组 item-level NDCG@20 均值。

{md_table(category_summary, ["checkpoint_label", "category_group", "item_count", "hr@20_mean", "ndcg@20_mean"])}

## baseline vs category_conf_input

> [!info] 字段说明
> `category_group`：v2 `s_cat_group`。
> `baseline_ndcg@20_mean`：baseline 该组 NDCG@20。
> `category_conf_ndcg@20_mean`：category_conf_input 该组 NDCG@20。
> `delta_ndcg@20_mean`：category_conf_input 减 baseline。
> `delta_hr@20_mean`：category_conf_input 减 baseline。

{md_table(comparison_summary, ["category_group", "baseline_ndcg@20_mean", "category_conf_ndcg@20_mean", "delta_ndcg@20_mean", "baseline_hr@20_mean", "category_conf_hr@20_mean", "delta_hr@20_mean"])}

## NDCG@20 分组最优

> [!info] 字段说明
> `scope`：比较范围。
> `group_value`：分组值。
> `best_checkpoint`：该分组 NDCG@20 最优 checkpoint。
> `best_value`：最优值。
> `second_checkpoint`：第二名 checkpoint。
> `best_minus_second`：最优值减第二名。

{md_table(category_selection, ["scope", "group_value", "best_checkpoint", "best_value", "second_checkpoint", "second_value", "best_minus_second"])}

## 产物

```text
task3_item_profile.csv
checkpoint_item_metrics.csv
checkpoint_overall_summary.csv
checkpoint_category_group_summary.csv
checkpoint_gt_group_summary.csv
checkpoint_delta_summary.csv
checkpoint_selection_summary.csv
task3_comparison_summary.csv
<variant>_item_profile.csv
run_manifest.json
```
"""
    output_path.write_text(markdown, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    code_root = Path(args.code_root).expanduser().resolve() if args.code_root else Path(__file__).resolve().parents[1]
    amazon_code_dir = code_root / "Amazon VG"
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else amazon_code_dir / "data"
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp, run_iso = now_stamp()
    device_name = select_device(args.device)
    variants = [parse_variant(raw, order) for order, raw in enumerate(args.variant)]
    if len(variants) < 2:
        raise ValueError("Task3 至少需要两个 variant：baseline 与 category_conf_input")

    availability = pd.read_csv(Path(args.availability).expanduser().resolve())
    test_df = pd.read_csv(dataset_dir / "test_rating.csv", usecols=["reviewerID", "asin"])
    item_profile = build_task3_item_profile(availability, test_df, group_col=args.group_col)
    if args.limit_items > 0:
        item_profile = item_profile.head(args.limit_items).copy()
    item_profile_path = output_dir / "task3_item_profile.csv"
    item_profile.to_csv(item_profile_path, index=False)

    frames = []
    for variant in variants:
        frames.append(
            evaluate_variant(
                variant=variant,
                item_profile=item_profile,
                test_df=test_df,
                amazon_code_dir=amazon_code_dir,
                device_name=device_name,
                batch_size=args.batch_size,
                top_k=args.top_k,
            )
        )
    checkpoint_profile = pd.concat(frames, ignore_index=True)
    checkpoint_profile_path = output_dir / "checkpoint_item_metrics.csv"
    checkpoint_profile.to_csv(checkpoint_profile_path, index=False)

    variant_profile_outputs = {}
    for variant in variants:
        variant_profile = checkpoint_profile[checkpoint_profile["checkpoint_label"].eq(variant.label)].copy()
        variant_profile_path = output_dir / f"{slugify_label(variant.label)}_item_profile.csv"
        variant_profile.to_csv(variant_profile_path, index=False)
        variant_profile_outputs[variant.label] = str(variant_profile_path)

    overall_summary = build_overall_summary(checkpoint_profile)
    overall_summary_path = output_dir / "checkpoint_overall_summary.csv"
    overall_summary.to_csv(overall_summary_path, index=False)

    category_summary = build_group_summary(checkpoint_profile, "category_group")
    category_summary_path = output_dir / "checkpoint_category_group_summary.csv"
    category_summary.to_csv(category_summary_path, index=False)

    gt_summary = build_group_summary(checkpoint_profile, "gt_group")
    gt_summary_path = output_dir / "checkpoint_gt_group_summary.csv"
    gt_summary.to_csv(gt_summary_path, index=False)

    baseline_label = variants[0].label
    category_conf_label = variants[1].label
    delta_summary = build_checkpoint_delta_summary(checkpoint_profile, baseline_label, category_conf_label)
    delta_summary_path = output_dir / "checkpoint_delta_summary.csv"
    delta_summary.to_csv(delta_summary_path, index=False)

    selection_summary = build_checkpoint_selection_summary(overall_summary, category_summary, gt_summary)
    selection_summary_path = output_dir / "checkpoint_selection_summary.csv"
    selection_summary.to_csv(selection_summary_path, index=False)

    baseline_group = category_summary[category_summary["checkpoint_label"].eq(baseline_label)]
    category_conf_group = category_summary[category_summary["checkpoint_label"].eq(category_conf_label)]
    comparison_summary = build_task3_comparison_summary(
        baseline_group,
        category_conf_group,
        baseline_label=baseline_label,
        category_conf_label=category_conf_label,
    )
    comparison_summary_path = output_dir / "task3_comparison_summary.csv"
    comparison_summary.to_csv(comparison_summary_path, index=False)

    result_md_name = f"{run_stamp} CCFCRec Amazon-VG category availability v2 Task3 checkpoint复评诊断结果.md"
    result_md_path = output_dir / result_md_name
    manifest_path = output_dir / "run_manifest.json"
    result_dir_display = args.result_dir_display or str(output_dir)

    manifest = {
        "run_time": run_iso,
        "script": str(Path(__file__).resolve()),
        "code_root": str(code_root),
        "dataset_dir": str(dataset_dir),
        "availability": str(Path(args.availability).expanduser().resolve()),
        "group_col": args.group_col,
        "output_dir": str(output_dir),
        "device": device_name,
        "batch_size": args.batch_size,
        "top_k": args.top_k,
        "limit_items": args.limit_items,
        "variants": [
            {
                "label": variant.label,
                "tar_path": str(variant.tar_path),
                "save_dict_member": variant.save_dict_member,
                "checkpoint_member": variant.checkpoint_member,
                "order": variant.order,
            }
            for variant in variants
        ],
        "outputs": {
            "task3_item_profile": str(item_profile_path),
            "checkpoint_item_metrics": str(checkpoint_profile_path),
            "checkpoint_overall_summary": str(overall_summary_path),
            "checkpoint_category_group_summary": str(category_summary_path),
            "checkpoint_gt_group_summary": str(gt_summary_path),
            "checkpoint_delta_summary": str(delta_summary_path),
            "checkpoint_selection_summary": str(selection_summary_path),
            "task3_comparison_summary": str(comparison_summary_path),
            "variant_item_profiles": variant_profile_outputs,
            "result_md": str(result_md_path),
            "run_manifest": str(manifest_path),
        },
        "row_counts": {
            "task3_item_profile": int(len(item_profile)),
            "checkpoint_item_metrics": int(len(checkpoint_profile)),
            "checkpoint_overall_summary": int(len(overall_summary)),
            "checkpoint_category_group_summary": int(len(category_summary)),
            "checkpoint_gt_group_summary": int(len(gt_summary)),
            "checkpoint_delta_summary": int(len(delta_summary)),
            "checkpoint_selection_summary": int(len(selection_summary)),
            "task3_comparison_summary": int(len(comparison_summary)),
        },
        "notes": [
            "只做旧 checkpoint 复评，不训练模型。",
            "category_group 在本轮代表 v2 s_cat_group，不是旧 category_count 分组。",
            "baseline 与 category_conf_input 来自不同 slim 包。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_task3_markdown(
        output_path=result_md_path,
        run_stamp=run_stamp,
        source_notes=args.source_note,
        result_dir_display=result_dir_display,
        item_count=int(item_profile["asin"].nunique()),
        variants=variants,
        overall_summary=overall_summary,
        category_summary=category_summary,
        comparison_summary=comparison_summary,
        selection_summary=selection_summary,
        manifest_name=manifest_path.name,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG category availability v2 Task3 checkpoint复评")
    parser.add_argument("--code-root", type=str, default="", help="CCFCRec-code 根目录；默认按脚本位置推断")
    parser.add_argument("--dataset-dir", type=str, default="", help="Amazon VG/data 目录；默认从 code-root 推断")
    parser.add_argument("--availability", required=True, type=str, help="category_availability_v2_item.csv 路径")
    parser.add_argument("--group-col", type=str, default="s_cat_group", help="availability 中的主分组列")
    parser.add_argument("--output-dir", required=True, type=str, help="Task3 输出目录")
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="格式 label|tar_path|save_dict_member|checkpoint_member；至少两个，第一项为 baseline",
    )
    parser.add_argument("--source-note", action="append", default=[], help="写入结果 MD 来源说明的 Obsidian wikilink 行")
    parser.add_argument("--result-dir-display", type=str, default="", help="写入结果 MD 的相对结果目录")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="推理设备")
    parser.add_argument("--batch-size", type=int, default=64, help="item 推理 batch size")
    parser.add_argument("--top-k", type=int, default=20, help="最大推荐用户数")
    parser.add_argument("--limit-items", type=int, default=0, help="调试时限制 item 数；0 表示全量")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
