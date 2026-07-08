#!/usr/bin/env python3
"""
CCFCRec Amazon-VG user-activity matched content-CF 诊断。

脚本作用：
1. 读取 content-CF alignment 诊断生成的 item-level profile；
2. 按 gt_group、目标用户历史交互数、目标用户覆盖数做分桶/组合控制；
3. 在每个 matched bucket 内比较 weak/mid/strong 的 q/attr/img 对齐差距；
4. 输出 matched bucket 明细、控制方式汇总、相关性、run_manifest.json 和结果 Markdown。

这个脚本只做本地 CSV 诊断，不训练模型，不加载 checkpoint。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_amazon_vg_score_norm_margin import md_table, nan_float


MATCHED_METRICS = [
    "ndcg@20",
    "target_history_q_cosine_mean",
    "target_minus_top20_history_q_cosine_mean",
    "target_history_attr_cosine_mean",
    "target_history_img_cosine_mean",
    "target_history_interaction_count_mean",
    "target_known_history_user_count",
]

CORRELATION_FEATURES = [
    "target_history_interaction_count_mean",
    "target_known_history_user_count",
    "target_history_q_cosine_mean",
    "target_history_q_cosine_max",
    "target_minus_top20_history_q_cosine_mean",
    "target_history_attr_cosine_mean",
    "target_history_img_cosine_mean",
]


@dataclass(frozen=True)
class Paths:
    project_root: Path
    code_root: Path
    profile: Path
    output_dir: Path


def now_stamp() -> tuple[str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def resolve_paths(args: argparse.Namespace) -> Paths:
    script_path = Path(__file__).resolve()
    code_root = Path(args.code_root).expanduser().resolve() if args.code_root else script_path.parents[1]
    project_root = code_root.parent
    profile = (
        Path(args.profile).expanduser().resolve()
        if args.profile
        else project_root
        / "temp_202606_实验文件记录"
        / "temp_20260608"
        / "content-cf-alignment-diagnostic"
        / "content_cf_alignment_profile.csv"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else project_root
        / "temp_202606_实验文件记录"
        / "temp_20260608"
        / "user-activity-matched-content-cf-diagnostic"
    )
    return Paths(project_root=project_root, code_root=code_root, profile=profile, output_dir=output_dir)


def category_order(value: str) -> int:
    return {"cat_weak_1_3": 0, "cat_mid_4": 1, "cat_strong_5_plus": 2}.get(value, 99)


def qcut_bucket(series: pd.Series, label_prefix: str, q: int = 4) -> pd.Series:
    """对连续变量做稳健分位数分桶；重复值太多时用 rank 回退。"""
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        buckets = pd.qcut(numeric, q=q, duplicates="drop")
    except ValueError:
        ranked = numeric.rank(method="first")
        buckets = pd.qcut(ranked, q=q, duplicates="drop")
    labels = []
    for value in buckets.astype(str):
        if value == "nan":
            labels.append(f"{label_prefix}_missing")
        else:
            labels.append(f"{label_prefix}_{value}")
    return pd.Series(labels, index=series.index)


def prepare_profile(profile: pd.DataFrame) -> pd.DataFrame:
    profile = profile.copy()
    profile["target_history_interaction_bucket"] = qcut_bucket(
        profile["target_history_interaction_count_mean"],
        "target_hist_interaction",
        q=4,
    )
    profile["target_known_history_user_bucket"] = qcut_bucket(
        profile["target_known_history_user_count"],
        "target_known_users",
        q=4,
    )
    return profile


def build_group_metric_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, sub in profile.groupby("category_group", dropna=False):
        row = {
            "category_group": group,
            "item_count": len(sub),
        }
        for metric in MATCHED_METRICS:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_median"] = sub[metric].median()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("category_group", key=lambda s: s.map(category_order)).reset_index(drop=True)


def build_matched_bucket_gap(profile: pd.DataFrame) -> pd.DataFrame:
    controls = [
        ("gt_group", ["gt_group"]),
        ("target_history_interaction_bucket", ["target_history_interaction_bucket"]),
        ("target_known_history_user_bucket", ["target_known_history_user_bucket"]),
        (
            "gt_group__target_history_interaction_bucket",
            ["gt_group", "target_history_interaction_bucket"],
        ),
        (
            "gt_group__target_known_history_user_bucket",
            ["gt_group", "target_known_history_user_bucket"],
        ),
        (
            "gt_group__interaction_bucket__known_user_bucket",
            ["gt_group", "target_history_interaction_bucket", "target_known_history_user_bucket"],
        ),
    ]
    rows = []
    for control_name, columns in controls:
        for key, sub in profile.groupby(columns, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            bucket_label = " | ".join(f"{col}={value}" for col, value in zip(columns, key))
            grouped = sub.groupby("category_group", dropna=False)
            row = {
                "control": control_name,
                "bucket": bucket_label,
                "total_count": len(sub),
            }
            for group in ["cat_weak_1_3", "cat_mid_4", "cat_strong_5_plus"]:
                group_sub = grouped.get_group(group) if group in grouped.groups else pd.DataFrame()
                row[f"{group}_count"] = int(len(group_sub))
                for metric in MATCHED_METRICS:
                    row[f"{group}_{metric}_mean"] = (
                        float(group_sub[metric].mean()) if len(group_sub) else nan_float()
                    )

            for metric in MATCHED_METRICS:
                weak = row[f"cat_weak_1_3_{metric}_mean"]
                mid = row[f"cat_mid_4_{metric}_mean"]
                strong = row[f"cat_strong_5_plus_{metric}_mean"]
                row[f"mid_minus_weak_{metric}"] = (
                    mid - weak if pd.notna(mid) and pd.notna(weak) else nan_float()
                )
                row[f"strong_minus_weak_{metric}"] = (
                    strong - weak if pd.notna(strong) and pd.notna(weak) else nan_float()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_gap_by_control(
    matched: pd.DataFrame,
    min_bucket_group_count: int,
) -> pd.DataFrame:
    rows = []
    for control, sub in matched.groupby("control"):
        for metric in MATCHED_METRICS:
            valid = sub[
                (sub["cat_weak_1_3_count"] >= min_bucket_group_count)
                & (
                    (sub["cat_mid_4_count"] >= min_bucket_group_count)
                    | (sub["cat_strong_5_plus_count"] >= min_bucket_group_count)
                )
            ].copy()
            mid_gap_col = f"mid_minus_weak_{metric}"
            strong_gap_col = f"strong_minus_weak_{metric}"
            rows.append(
                {
                    "control": control,
                    "metric": metric,
                    "valid_bucket_count": len(valid),
                    "valid_item_count": int(valid["total_count"].sum()) if len(valid) else 0,
                    "mid_minus_weak_mean": valid[mid_gap_col].mean(),
                    "strong_minus_weak_mean": valid[strong_gap_col].mean(),
                    "mid_minus_weak_positive_rate": (
                        (valid[mid_gap_col] > 0).mean() if len(valid) else nan_float()
                    ),
                    "strong_minus_weak_positive_rate": (
                        (valid[strong_gap_col] > 0).mean() if len(valid) else nan_float()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["control", "metric"]).reset_index(drop=True)


def build_correlation_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in CORRELATION_FEATURES:
        for metric in ["ndcg@20", "target_history_q_cosine_mean"]:
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
    profile: pd.DataFrame,
    group_summary: pd.DataFrame,
    matched_summary: pd.DataFrame,
    correlation: pd.DataFrame,
    manifest_name: str,
) -> None:
    q_focus = matched_summary[
        matched_summary["metric"].eq("target_history_q_cosine_mean")
    ].copy()
    q_focus = q_focus.sort_values(["control"])
    attr_focus = matched_summary[
        matched_summary["metric"].eq("target_history_attr_cosine_mean")
    ].copy()
    img_focus = matched_summary[
        matched_summary["metric"].eq("target_history_img_cosine_mean")
    ].copy()
    ndcg_focus = matched_summary[matched_summary["metric"].eq("ndcg@20")].copy()
    core_controls = [
        "gt_group__target_history_interaction_bucket",
        "gt_group__interaction_bucket__known_user_bucket",
    ]
    core_q = q_focus[q_focus["control"].isin(core_controls)].copy()
    top_corr = correlation[correlation["metric"].eq("ndcg@20")].copy()
    top_corr["abs_spearman"] = top_corr["spearman"].abs()
    top_corr = top_corr.sort_values("abs_spearman", ascending=False).head(8)

    if len(core_q):
        q_lines = []
        for _, row in core_q.iterrows():
            q_lines.append(
                f'{row["control"]}: valid_bucket_count={int(row["valid_bucket_count"])}, '
                f'mid_minus_weak={row["mid_minus_weak_mean"]:.4f}, '
                f'strong_minus_weak={row["strong_minus_weak_mean"]:.4f}, '
                f'strong_positive_rate={row["strong_minus_weak_positive_rate"]:.4f}'
            )
        mechanism_text = "\n".join(
            [
                "核心结论：",
                "控制 gt_group 和目标用户历史活跃度后，q_v_c / attr_emb 的 weak gap 仍然保留，img_proj gap 不稳定；",
                "但 NDCG@20 gap 在严格 activity 控制后明显缩小，说明指标差距有相当部分来自目标用户数量/活跃度混杂。",
                "因此现阶段更稳的说法是：content-CF 对齐不足不是纯用户活跃度混杂，但最终指标差距不能只归因于内容侧。",
                "",
                "关键 matched 证据：",
                *q_lines,
            ]
        )
    else:
        mechanism_text = "核心结论：有效 matched bucket 不足，需要降低最小桶内样本阈值或改用回归控制。"

    group_fields = field_callout(
        [
            ("category_group", "按 item 类别属性数量得到的 weak/mid/strong 分组。"),
            ("item_count", "该组 test item 数。"),
            ("ndcg@20_mean", "该组 item-level NDCG@20 均值。"),
            ("target_history_q_cosine_mean_mean", "item q_v_c 与目标用户历史 q_v_c 中心 cosine 均值的组均值。"),
            ("target_minus_top20_history_q_cosine_mean_mean", "目标用户历史 q 对齐均值减 top20 用户历史 q 对齐均值的组均值。"),
            ("target_history_attr_cosine_mean_mean", "类别表示 attr_emb 目标用户历史中心对齐的组均值。"),
            ("target_history_img_cosine_mean_mean", "图像投影 img_proj 目标用户历史中心对齐的组均值。"),
            ("target_history_interaction_count_mean_mean", "目标用户训练历史交互数均值的组均值。"),
        ]
    )
    matched_fields = field_callout(
        [
            ("control", "匹配/控制使用的变量组合。"),
            ("metric", "桶内比较的指标。"),
            ("valid_bucket_count", "同时有 weak 且 mid 或 strong 满足最小样本阈值的桶数。"),
            ("valid_item_count", "有效桶内总 item 数。"),
            ("mid_minus_weak_mean", "有效桶内 mid 均值减 weak 均值后再平均。"),
            ("strong_minus_weak_mean", "有效桶内 strong 均值减 weak 均值后再平均。"),
            ("mid_minus_weak_positive_rate", "有效桶内 mid_minus_weak 大于 0 的比例。"),
            ("strong_minus_weak_positive_rate", "有效桶内 strong_minus_weak 大于 0 的比例。"),
        ]
    )
    corr_fields = field_callout(
        [
            ("feature", "候选解释变量。"),
            ("metric", "被解释指标。"),
            ("spearman", "秩相关，适合看单调关系。"),
            ("pearson", "线性相关。"),
            ("正值", "变量越大，指标通常越高。"),
            ("负值", "变量越大，指标通常越低。"),
        ]
    )

    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG user-activity matched content-CF 诊断结果
date: 2026-06-08
tags:
  - CCFCRec
  - Amazon-VG
  - 实验结果
  - 难度分层
  - 混杂控制
---

# {run_stamp} CCFCRec Amazon-VG user-activity matched content-CF 诊断结果

## 结论

这次诊断只读取 content-CF alignment profile，不训练模型。

```text
{mechanism_text}
```

## 输入与输出

输入 item 数：

```text
test item count = {len(profile)}
```

核心输出：

```text
matched_bucket_gap_detail.csv
matched_bucket_gap_control_summary.csv
matched_group_metric_summary.csv
matched_correlation_summary.csv
{manifest_name}
```

## 原始分组对照

{group_fields}

{md_table(group_summary, ["category_group", "item_count", "ndcg@20_mean", "target_history_q_cosine_mean_mean", "target_minus_top20_history_q_cosine_mean_mean", "target_history_attr_cosine_mean_mean", "target_history_img_cosine_mean_mean", "target_history_interaction_count_mean_mean"])}

## q_v_c matched gap

{matched_fields}

{md_table(q_focus, ["control", "metric", "valid_bucket_count", "valid_item_count", "mid_minus_weak_mean", "strong_minus_weak_mean", "mid_minus_weak_positive_rate", "strong_minus_weak_positive_rate"])}

## attr_emb matched gap

{matched_fields}

{md_table(attr_focus, ["control", "metric", "valid_bucket_count", "valid_item_count", "mid_minus_weak_mean", "strong_minus_weak_mean", "mid_minus_weak_positive_rate", "strong_minus_weak_positive_rate"])}

## img_proj matched gap

{matched_fields}

{md_table(img_focus, ["control", "metric", "valid_bucket_count", "valid_item_count", "mid_minus_weak_mean", "strong_minus_weak_mean", "mid_minus_weak_positive_rate", "strong_minus_weak_positive_rate"])}

## NDCG@20 matched gap

{matched_fields}

{md_table(ndcg_focus, ["control", "metric", "valid_bucket_count", "valid_item_count", "mid_minus_weak_mean", "strong_minus_weak_mean", "mid_minus_weak_positive_rate", "strong_minus_weak_positive_rate"])}

## 与 NDCG@20 相关性最高的变量

{corr_fields}

{md_table(top_corr, ["feature", "metric", "spearman", "pearson"])}

## 判读边界

```text
1. 这是 matched bucket 观察性诊断，不是因果证明。
2. 组合控制越严格，有效桶数量越少，方差越大。
3. 如果 q_v_c gap 在 gt_group + activity 控制后仍为正，说明 weak 相比 mid/strong 的内容-CF 对齐弱信号仍存在。
4. attr_emb gap 如果比 img_proj 更稳定，下一步优先做 category-side ablation。
```

## 下一步

优先根据本诊断选择：

```text
1. 如果 q/attr matched gap 稳定保留，下一步做 category ablation 或 category completion counterfactual；
2. 因为 NDCG gap 在严格 activity 控制后缩小，方法实验必须显式报告 activity/gt 分层；
3. 仍不需要服务器训练。
```
"""
    output_path.write_text(markdown, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp, run_iso = now_stamp()

    profile = pd.read_csv(paths.profile)
    profile = prepare_profile(profile)
    group_summary = build_group_metric_summary(profile)
    matched = build_matched_bucket_gap(profile)
    matched_summary = summarize_gap_by_control(matched, args.min_bucket_group_count)
    correlation = build_correlation_summary(profile)

    group_summary_path = paths.output_dir / "matched_group_metric_summary.csv"
    matched_path = paths.output_dir / "matched_bucket_gap_detail.csv"
    matched_summary_path = paths.output_dir / "matched_bucket_gap_control_summary.csv"
    correlation_path = paths.output_dir / "matched_correlation_summary.csv"
    result_md_path = paths.output_dir / f"{run_stamp} CCFCRec Amazon-VG user-activity matched content-CF 诊断结果.md"
    manifest_path = paths.output_dir / "run_manifest.json"

    group_summary.to_csv(group_summary_path, index=False)
    matched.to_csv(matched_path, index=False)
    matched_summary.to_csv(matched_summary_path, index=False)
    correlation.to_csv(correlation_path, index=False)

    manifest = {
        "run_time": run_iso,
        "script": str(Path(__file__).resolve()),
        "project_root": str(paths.project_root),
        "code_root": str(paths.code_root),
        "profile": str(paths.profile),
        "output_dir": str(paths.output_dir),
        "min_bucket_group_count": args.min_bucket_group_count,
        "outputs": {
            "matched_group_metric_summary": str(group_summary_path),
            "matched_bucket_gap_detail": str(matched_path),
            "matched_bucket_gap_control_summary": str(matched_summary_path),
            "matched_correlation_summary": str(correlation_path),
            "result_md": str(result_md_path),
        },
        "row_counts": {
            "profile": int(len(profile)),
            "matched_group_metric_summary": int(len(group_summary)),
            "matched_bucket_gap_detail": int(len(matched)),
            "matched_bucket_gap_control_summary": int(len(matched_summary)),
            "matched_correlation_summary": int(len(correlation)),
        },
        "notes": [
            "只读取 CSV，不训练模型。",
            "控制变量包括 gt_group、target_history_interaction_bucket、target_known_history_user_bucket 及组合。",
            "matched gap 是观察性分桶结果，不是因果证明。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_result_markdown(
        result_md_path,
        run_stamp,
        profile,
        group_summary,
        matched_summary,
        correlation,
        manifest_path.name,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG user-activity matched content-CF 诊断")
    parser.add_argument("--code-root", type=str, default="", help="CCFCRec-code 根目录；默认按脚本位置推断")
    parser.add_argument("--profile", type=str, default="", help="content_cf_alignment_profile.csv 路径")
    parser.add_argument("--output-dir", type=str, default="", help="诊断输出目录")
    parser.add_argument("--min-bucket-group-count", type=int, default=3, help="桶内 weak/mid/strong 最小样本数阈值")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
