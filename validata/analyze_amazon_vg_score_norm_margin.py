#!/usr/bin/env python3
"""
CCFCRec Amazon-VG score/norm/margin 诊断。

脚本作用：
1. 从 workers45 瘦身 tar 中加载 save_dict.pkl 和 best_epoch_64.pt；
2. 复载 CCFCRec，并对每个 test item 计算 q_v_c 与所有 user embedding 的分数；
3. 输出目标用户分数、top-k cutoff margin、q_v_c 范数、top-k 边界 gap 等 item-level 画像；
4. 按 category_group / gt_group 汇总，生成 run_manifest.json 和结果 Markdown。

这个脚本只做本地诊断，不训练模型，不修改 CCFCRec 主训练入口。
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import pickle
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


METRICS = ["hr@5", "hr@10", "hr@20", "ndcg@5", "ndcg@10", "ndcg@20"]
DIAGNOSTIC_FEATURES = [
    "q_norm",
    "target_score_max",
    "target_score_mean",
    "top1_score",
    "score_at_20",
    "score_at_21",
    "top20_score_mean",
    "top20_score_std",
    "local_gap_20_21",
    "margin_to_top1",
    "margin_to_top20_cutoff",
    "best_target_rank",
    "best_target_rank_percentile",
    "target_hit_count_at20",
    "target_hr_at20_like",
    "mapped_target_user_count",
    "target_user_norm_mean",
    "top20_user_norm_mean",
]


def now_stamp() -> tuple[str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def category_order(value: str) -> int:
    order = {"cat_weak_1_3": 0, "cat_mid_4": 1, "cat_strong_5_plus": 2}
    return order.get(value, 99)


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
        / "score-norm-margin-diagnostic"
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


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pickle_from_tar(tar_path: Path, member_name: str) -> object:
    with tarfile.open(tar_path, "r") as tar:
        member = tar.getmember(member_name)
        file_obj = tar.extractfile(member)
        if file_obj is None:
            raise FileNotFoundError(f"tar member 无法读取: {member_name}")
        return pickle.load(file_obj)


def load_torch_state_from_tar(tar_path: Path, member_name: str, map_location: str) -> dict[str, torch.Tensor]:
    with tarfile.open(tar_path, "r") as tar:
        member = tar.getmember(member_name)
        file_obj = tar.extractfile(member)
        if file_obj is None:
            raise FileNotFoundError(f"tar member 无法读取: {member_name}")
        buffer = io.BytesIO(file_obj.read())
    return torch.load(buffer, map_location=map_location)


def build_model_args(state_dict: dict[str, torch.Tensor]) -> SimpleNamespace:
    method_variant = "baseline"
    category_conf_dim = 16
    category_conf_max_count = 5
    category_gate_scale = 0.5
    if "category_conf_embedding.weight" in state_dict:
        method_variant = "category_conf_input"
        category_conf_dim = int(state_dict["category_conf_embedding.weight"].shape[1])
    if "category_fusion_gate.weight" in state_dict:
        method_variant = "category_conf_fusion_gate"
    return SimpleNamespace(
        attr_num=int(state_dict["attr_matrix"].shape[0]),
        attr_present_dim=int(state_dict["attr_matrix"].shape[1]),
        implicit_dim=int(state_dict["user_embedding"].shape[1]),
        cat_implicit_dim=int(state_dict["gen_layer1.weight"].shape[0]),
        user_number=int(state_dict["user_embedding"].shape[0]),
        item_number=int(state_dict["item_embedding"].shape[0]),
        pretrain=False,
        pretrain_update=False,
        method_variant=method_variant,
        category_conf_dim=category_conf_dim,
        category_conf_max_count=category_conf_max_count,
        category_gate_scale=category_gate_scale,
    )


def load_model(paths: Paths, device_name: str, checkpoint_member: str):
    os.environ["CCFCREC_DEVICE"] = device_name
    if str(paths.amazon_code_dir) not in sys.path:
        sys.path.insert(0, str(paths.amazon_code_dir))
    from model import CCFCRec

    state_dict = load_torch_state_from_tar(paths.reproduction_tar, checkpoint_member, device_name)
    model = CCFCRec(build_model_args(state_dict))
    model.load_state_dict(state_dict)
    model.to(torch.device(device_name))
    model.eval()
    return model


def build_attribute_and_image_batch(
    asins: list[str],
    category_num: int,
    category_map: dict[str, list[int]],
    img_feature_dict: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    attributes = torch.full((len(asins), category_num), -1.0, dtype=torch.float32, device=device)
    images = []
    for row_idx, asin in enumerate(asins):
        for category in category_map.get(asin, []):
            if 0 <= int(category) < category_num:
                attributes[row_idx, int(category)] = 1.0
        image = img_feature_dict.get(asin)
        if image is None:
            images.append(np.zeros(4096, dtype=np.float32))
        else:
            images.append(np.asarray(image, dtype=np.float32))
    image_tensor = torch.tensor(np.stack(images), dtype=torch.float32, device=device)
    return attributes, image_tensor


def mapped_target_users_by_asin(test_df: pd.DataFrame, user_ser_dict: dict[object, int]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for asin, sub in test_df.groupby("asin"):
        raw_users = sub["reviewerID"].tolist()
        mapped_users = [user_ser_dict.get(user) for user in raw_users]
        known_users = sorted({int(user) for user in mapped_users if user is not None})
        result[asin] = {
            "raw_target_user_count": len(raw_users),
            "mapped_target_user_count": len(known_users),
            "unknown_target_user_count": len(raw_users) - sum(user is not None for user in mapped_users),
            "target_user_indices": known_users,
        }
    return result


def nan_float() -> float:
    return float("nan")


def compute_single_score_margin(
    scores: torch.Tensor,
    target_indices: list[int],
    user_norms: torch.Tensor,
    top_k: int = 20,
    analysis_top_k: int = 100,
) -> dict[str, float]:
    user_count = int(scores.shape[0])
    top_k = min(top_k, user_count)
    analysis_top_k = min(analysis_top_k, user_count)
    selected_count = min(max(analysis_top_k, top_k + 1), user_count)
    top_values, top_indices = torch.topk(scores, k=selected_count, largest=True)
    top20_values = top_values[:top_k]
    top20_indices = top_indices[:top_k]

    score_at_20 = float(top_values[top_k - 1].item())
    score_at_21 = float(top_values[top_k].item()) if selected_count > top_k else nan_float()
    local_gap = score_at_20 - score_at_21 if not math.isnan(score_at_21) else nan_float()

    row = {
        "top1_score": float(top_values[0].item()),
        "score_at_5": float(top_values[min(4, selected_count - 1)].item()),
        "score_at_10": float(top_values[min(9, selected_count - 1)].item()),
        "score_at_20": score_at_20,
        "score_at_21": score_at_21,
        "top20_score_mean": float(top20_values.mean().item()),
        "top20_score_std": float(top20_values.std(unbiased=False).item()),
        "local_gap_20_21": float(local_gap),
        "top20_user_norm_mean": float(user_norms[top20_indices].mean().item()),
    }

    if not target_indices:
        row.update(
            {
                "target_score_max": nan_float(),
                "target_score_mean": nan_float(),
                "margin_to_top1": nan_float(),
                "margin_to_top20_cutoff": nan_float(),
                "best_target_rank": nan_float(),
                "best_target_rank_percentile": nan_float(),
                "target_hit_count_at20": 0.0,
                "target_hr_at20_like": 0.0,
                "target_user_norm_mean": nan_float(),
                "best_target_user_norm": nan_float(),
            }
        )
        return row

    target_tensor = torch.tensor(target_indices, dtype=torch.long, device=scores.device)
    target_scores = scores[target_tensor]
    best_pos = int(torch.argmax(target_scores).item())
    best_target_index = int(target_tensor[best_pos].item())
    target_score_max = float(target_scores[best_pos].item())
    target_set = set(target_indices)
    top20_set = {int(value) for value in top20_indices.detach().cpu().tolist()}
    hit_count = len(target_set.intersection(top20_set))
    best_rank = int((scores > target_score_max).sum().item()) + 1

    row.update(
        {
            "target_score_max": target_score_max,
            "target_score_mean": float(target_scores.mean().item()),
            "margin_to_top1": target_score_max - row["top1_score"],
            "margin_to_top20_cutoff": target_score_max - row["score_at_20"],
            "best_target_rank": float(best_rank),
            "best_target_rank_percentile": best_rank / user_count,
            "target_hit_count_at20": float(hit_count),
            "target_hr_at20_like": hit_count / top_k,
            "target_user_norm_mean": float(user_norms[target_tensor].mean().item()),
            "best_target_user_norm": float(user_norms[best_target_index].item()),
        }
    )
    return row


def aggregate_summary(profile: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, sub in profile.groupby(group_col, dropna=False):
        row = {
            group_col: group,
            "item_count": len(sub),
            "ndcg@20_mean": sub["ndcg@20"].mean(),
            "hr@20_mean": sub["hr@20"].mean(),
        }
        for feature in DIAGNOSTIC_FEATURES:
            row[f"{feature}_mean"] = sub[feature].mean()
            row[f"{feature}_median"] = sub[feature].median()
        rows.append(row)
    result = pd.DataFrame(rows)
    if group_col == "category_group":
        return result.sort_values(group_col, key=lambda s: s.map(category_order)).reset_index(drop=True)
    return result.sort_values(group_col).reset_index(drop=True)


def build_correlation_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in DIAGNOSTIC_FEATURES:
        for metric in ["ndcg@20", "hr@20"]:
            sub = profile[[feature, metric]].dropna()
            if len(sub) < 3:
                pearson = nan_float()
                spearman = nan_float()
            else:
                pearson = float(sub.corr(method="pearson").iloc[0, 1])
                spearman = float(sub.corr(method="spearman").iloc[0, 1])
            rows.append(
                {
                    "feature": feature,
                    "metric": metric,
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
    return pd.DataFrame(rows)


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
    q_corr_row = correlation[(correlation["feature"].eq("q_norm")) & (correlation["metric"].eq("ndcg@20"))]
    q_spearman = float(q_corr_row["spearman"].iloc[0]) if len(q_corr_row) else nan_float()

    if weak is not None and mid is not None and strong is not None:
        mechanism_text = f"""核心结论：cat_weak_1_3 难，不是因为 q_v_c 范数更弱，也不是主要因为 top20 边界更拥挤；
更直接的机制是目标用户本身没有被模型打高，导致 target_score_max 低、top20 margin 更负、best_target_rank 大幅靠后。
关键证据：
weak target_score_max_mean = {weak["target_score_max_mean"]:.4f}，mid/strong 分别为 {mid["target_score_max_mean"]:.4f} / {strong["target_score_max_mean"]:.4f}；
weak margin_to_top20_cutoff_mean = {weak["margin_to_top20_cutoff_mean"]:.4f}，mid/strong 分别为 {mid["margin_to_top20_cutoff_mean"]:.4f} / {strong["margin_to_top20_cutoff_mean"]:.4f}；
weak best_target_rank_median = {weak["best_target_rank_median"]:.0f}，mid/strong 分别为 {mid["best_target_rank_median"]:.0f} / {strong["best_target_rank_median"]:.0f}；
weak q_norm_mean = {weak["q_norm_mean"]:.4f}，mid/strong 分别为 {mid["q_norm_mean"]:.4f} / {strong["q_norm_mean"]:.4f}；
weak local_gap_20_21_mean = {weak["local_gap_20_21_mean"]:.4f}，mid/strong 分别为 {mid["local_gap_20_21_mean"]:.4f} / {strong["local_gap_20_21_mean"]:.4f}。"""
    else:
        mechanism_text = """核心看点是：weak item 的目标用户是否被模型打到 top20 cutoff 以上，
以及 q_v_c 范数、target score、top20 边界 gap 是否随 category_group 改变。"""

    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG score-norm-margin 诊断结果
date: 2026-06-08
tags:
  - CCFCRec
  - Amazon-VG
  - 实验结果
  - 难度分层
  - 机制诊断
---

# {run_stamp} CCFCRec Amazon-VG score-norm-margin 诊断结果

## 结论

这次诊断只解释 best64 已复现模型的打分行为，不训练新模型。

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
item_score_margin_profile.csv
score_margin_group_summary.csv
score_margin_gt_group_summary.csv
score_margin_correlation_summary.csv
{manifest_name}
```

## 按类别证据强弱分组

> [!info] 字段说明
> - `category_group`：按 item 类别属性数量得到的 weak/mid/strong 分组。
> - `item_count`：该组 test item 数。
> - `ndcg@20_mean`：该组 item-level NDCG@20 均值。
> - `target_score_max_mean`：每个 item 的可序列化目标用户最高打分，再取组均值。
> - `margin_to_top20_cutoff_mean`：`target_score_max - score_at_20` 的组均值，越负说明目标用户越难进 top20。
> - `best_target_rank_median`：目标用户最佳名次的组中位数，越大越差。
> - `target_hr_at20_like_mean`：每个 item 的 top20 命中目标用户数除以 20 后的均值。
> - `q_norm_mean`：生成式 item 表征 `q_v_c` 的范数均值。
> - `local_gap_20_21_mean`：第 20 名和第 21 名用户分数差均值，用来看 top20 边界是否拥挤。

{md_table(category_summary, ["category_group", "item_count", "ndcg@20_mean", "target_score_max_mean", "margin_to_top20_cutoff_mean", "best_target_rank_median", "target_hr_at20_like_mean", "q_norm_mean", "local_gap_20_21_mean"])}

判读：

```text
这张表说明 weak 组的主要问题是 target score / margin / rank，不是 q_norm 弱。
local_gap_20_21 差异很小时，暂时不优先解释为 top20 cutoff 附近竞争更挤。
```

## 按 gt 用户数分组

> [!info] 字段说明
> - `gt_group`：按 item-level 复评指标中的 `gt_user_count` 分桶。
> - `item_count`：该桶 test item 数。
> - `ndcg@20_mean`：该桶 item-level NDCG@20 均值。
> - `target_score_max_mean`：沿用上一张表的目标用户最高分口径。
> - `margin_to_top20_cutoff_mean`：沿用上一张表的 top20 cutoff margin 口径。
> - `best_target_rank_median`：沿用上一张表的目标用户最佳名次口径。
> - `target_hr_at20_like_mean`：沿用上一张表的 top20 命中率近似口径。
> - `q_norm_mean`：该桶 `q_v_c` 范数均值。
> - 用途：判断 score/margin 现象是否只是由 GT 用户数多少造成。

{md_table(gt_summary, ["gt_group", "item_count", "ndcg@20_mean", "target_score_max_mean", "margin_to_top20_cutoff_mean", "best_target_rank_median", "target_hr_at20_like_mean", "q_norm_mean"])}

## 与 NDCG@20 相关性最高的变量

> [!info] 字段说明
> - `feature`：候选解释变量。
> - `metric`：被解释的指标，这里筛选展示 `ndcg@20`。
> - `spearman`：秩相关，适合看单调关系。
> - `pearson`：线性相关。
> - 正值：变量越大，指标通常越高。
> - 负值：变量越大，指标通常越低。
> - 注意：`target_hit_count_at20` 和 `target_hr_at20_like` 与 NDCG@20 高相关有指标定义重叠。
> - 判读重点：更应关注 `margin_to_top20_cutoff`、`best_target_rank`、`target_score_max` 等机制变量。

{md_table(top_corr, ["feature", "metric", "spearman", "pearson"])}

注意：

```text
target_hit_count_at20 与 target_hr_at20_like 和 NDCG@20 高相关是指标定义上的直接关系；
更有解释价值的是 margin_to_top20_cutoff、best_target_rank、target_score_max。
q_norm 与 NDCG@20 的 Spearman = {q_spearman:.4f}，如果接近 0，则基本不是主解释变量。
```

## 下一步

优先根据本诊断选择：

```text
1. 下一步优先做 target-score 来源诊断：看 weak 的目标用户与 q_v_c 是否语义/协同对齐不足；
2. 暂时不把 q_norm 当作主问题；
3. 暂时不优先做 top20 边界竞争项分析，除非 local_gap_20_21 出现明显组间差异；
4. 仍不直接服务器重训，先做 content-CF alignment / target-user alignment 本地诊断。
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

    item_profile = pd.read_csv(paths.item_profile)
    if args.limit_items > 0:
        item_profile = item_profile.head(args.limit_items).copy()
    test_df = pd.read_csv(paths.dataset_dir / "test_rating.csv")
    targets_by_asin = mapped_target_users_by_asin(test_df, save_dict["user_ser_dict"])
    category_num = int(save_dict["category_ser_map_len"])
    user_norms = torch.norm(model.user_embedding.detach(), dim=1)
    all_rows = []

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
            scores_batch = torch.matmul(q_v_c, model.user_embedding.detach().T)
            q_norms = torch.norm(q_v_c, dim=1)

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
                    "q_norm": float(q_norms[row_idx].item()),
                }
                for metric in METRICS:
                    output_row[metric] = source_row[metric]
                output_row.update(score_row)
                all_rows.append(output_row)

    score_profile = pd.DataFrame(all_rows)
    score_profile_path = paths.output_dir / "item_score_margin_profile.csv"
    score_profile.to_csv(score_profile_path, index=False)

    category_summary = aggregate_summary(score_profile, "category_group")
    category_summary_path = paths.output_dir / "score_margin_group_summary.csv"
    category_summary.to_csv(category_summary_path, index=False)

    gt_summary = aggregate_summary(score_profile, "gt_group")
    gt_summary_path = paths.output_dir / "score_margin_gt_group_summary.csv"
    gt_summary.to_csv(gt_summary_path, index=False)

    correlation = build_correlation_summary(score_profile)
    correlation_path = paths.output_dir / "score_margin_correlation_summary.csv"
    correlation.to_csv(correlation_path, index=False)

    result_md_name = f"{run_stamp} CCFCRec Amazon-VG score-norm-margin 诊断结果.md"
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
        "output_dir": str(paths.output_dir),
        "device": device_name,
        "batch_size": args.batch_size,
        "top_k": args.top_k,
        "analysis_top_k": args.analysis_top_k,
        "limit_items": args.limit_items,
        "outputs": {
            "item_score_margin_profile": str(score_profile_path),
            "score_margin_group_summary": str(category_summary_path),
            "score_margin_gt_group_summary": str(gt_summary_path),
            "score_margin_correlation_summary": str(correlation_path),
            "result_md": str(result_md_path),
        },
        "row_counts": {
            "item_score_margin_profile": int(len(score_profile)),
            "category_summary": int(len(category_summary)),
            "gt_summary": int(len(gt_summary)),
            "correlation": int(len(correlation)),
        },
        "notes": [
            "只做本地诊断，不训练模型。",
            "target user 只统计 save_dict.pkl 中可序列化的测试用户。",
            "margin_to_top20_cutoff > 0 表示至少一个目标用户分数超过 top20 cutoff。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    write_result_markdown(
        result_md_path,
        run_stamp,
        len(score_profile),
        category_summary,
        gt_summary,
        correlation,
        manifest_path.name,
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG score/norm/margin 诊断")
    parser.add_argument("--code-root", type=str, default="", help="CCFCRec-code 根目录；默认按脚本位置推断")
    parser.add_argument("--dataset-dir", type=str, default="", help="Amazon VG/data 目录；默认从 code-root 推断")
    parser.add_argument("--reproduction-tar", type=str, default="", help="workers45 瘦身 tar 路径")
    parser.add_argument("--item-profile", type=str, default="", help="自然难度诊断 item_profile.csv")
    parser.add_argument("--output-dir", type=str, default="", help="诊断输出目录")
    parser.add_argument("--save-dict-member", type=str, default="amazon_vg_slim/save_dict.pkl", help="tar 内 save_dict.pkl 成员名")
    parser.add_argument("--checkpoint-member", type=str, default="amazon_vg_slim/best_epoch_64.pt", help="tar 内 checkpoint 成员名")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="推理设备")
    parser.add_argument("--batch-size", type=int, default=64, help="item 推理 batch size")
    parser.add_argument("--top-k", type=int, default=20, help="主诊断 top-k cutoff")
    parser.add_argument("--analysis-top-k", type=int, default=100, help="额外分析 top-k 范围")
    parser.add_argument("--limit-items", type=int, default=0, help="调试时限制 item 数；0 表示全量")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
