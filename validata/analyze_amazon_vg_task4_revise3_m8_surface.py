#!/usr/bin/env python3
"""
Analyze Task4-revise-3 M8 high-detail M7a surface ablation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/Users/luojiaqiang/Documents/Obsidian Vault/科研/CCFCRec对比学习思路")
DEFAULT_RESULT_ROOT = Path(
    "/Volumes/MyPassport/CCFCRec对比学习思路硬盘/实验记录硬盘/ccfcrec_result/"
    "2026-07-08_020655_task4_highdetail_m7a_surface_ablation_seed43_workers8_fast_uniform_mps_100epoch"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "temp_202607_实验文件记录" / "temp_20260708"
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "实验记录"
DESIGN_NOTE_NAME = "2026-07-07 183600 CCFCRec Amazon-VG Task4-revise-3 M7a high-detail carrier surface ablation 设计"
ORDER = {
    "M8q_q_only_real": 1,
    "M8qs_q_only_shuffle": 2,
    "M8s_self_only_real": 3,
    "M8ss_self_only_shuffle": 4,
}
PAIRS = [
    ("M8q_minus_M8qs", "M8q_q_only_real", "M8qs_q_only_shuffle"),
    ("M8s_minus_M8ss", "M8s_self_only_real", "M8ss_self_only_shuffle"),
]


@dataclass(frozen=True)
class Outputs:
    output_dir: Path
    best_summary_csv: Path
    pair_comparison_csv: Path
    training_dynamics_csv: Path
    manifest_json: Path
    result_md: Path
    route_md: Path


def now_stamp() -> tuple[str, str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H%M%S"), now.strftime("%Y-%m-%d"), now.isoformat(timespec="seconds")


def safe_float(value) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def classify_m8_run(method_variant: str, disable_q: bool, disable_self: bool) -> str:
    if method_variant == "task4_highdetail_trainhard_weight" and disable_self and not disable_q:
        return "M8q_q_only_real"
    if method_variant == "task4_highdetail_trainhard_shuffle_weight" and disable_self and not disable_q:
        return "M8qs_q_only_shuffle"
    if method_variant == "task4_highdetail_trainhard_weight" and disable_q and not disable_self:
        return "M8s_self_only_real"
    if method_variant == "task4_highdetail_trainhard_shuffle_weight" and disable_q and not disable_self:
        return "M8ss_self_only_shuffle"
    return method_variant


def select_best_checkpoint(result: pd.DataFrame) -> pd.Series:
    return result.sort_values(
        ["ndcg@20", "hr@20", "epoch", "checkpoint_index"],
        ascending=[False, False, True, True],
    ).iloc[0]


def discover_run_dirs(result_root: Path) -> list[Path]:
    return sorted([p for p in result_root.iterdir() if p.is_dir() and (p / "result.csv").exists() and (p / "run_config.json").exists()])


def build_best_summary(result_root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in discover_run_dirs(result_root):
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        method = str(config.get("method_variant", ""))
        disable_q = bool(config.get("task4_disable_q_bpr_weight", False))
        disable_self = bool(config.get("task4_disable_self_contrast_weight", False))
        label = classify_m8_run(method, disable_q, disable_self)
        result = pd.read_csv(run_dir / "result.csv")
        best = select_best_checkpoint(result)
        last = result.sort_values(["epoch", "checkpoint_index"]).iloc[-1]
        rows.append(
            {
                "run_label": label,
                "method_variant": method,
                "run_dir": run_dir.name,
                "disable_q_bpr_weight": disable_q,
                "disable_self_contrast_weight": disable_self,
                "row_count": len(result),
                "best_epoch": int(best["epoch"]),
                "best_checkpoint_index": int(best["checkpoint_index"]),
                "best_ndcg@20": safe_float(best["ndcg@20"]),
                "best_hr@20": safe_float(best["hr@20"]),
                "last_ndcg@20": safe_float(last["ndcg@20"]),
                "last_hr@20": safe_float(last["hr@20"]),
                "task4_loss_alpha": config.get("task4_loss_alpha", ""),
                "seed": config.get("seed", ""),
                "num_workers": config.get("num_workers", ""),
                "negative_sampling_mode": config.get("negative_sampling_mode", ""),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["_order"] = summary["run_label"].map(ORDER).fillna(99)
        summary = summary.sort_values(["_order", "run_dir"]).drop(columns=["_order"]).reset_index(drop=True)
    return summary


def _metric(best: pd.DataFrame, label: str, metric: str) -> float:
    sub = best[best["run_label"].eq(label)]
    if sub.empty:
        return float("nan")
    return safe_float(sub.iloc[0][metric])


def build_pair_comparison(best: pd.DataFrame, min_ndcg_delta: float = 0.0005) -> pd.DataFrame:
    rows = []
    for comparison, method, control in PAIRS:
        method_ndcg = _metric(best, method, "best_ndcg@20")
        control_ndcg = _metric(best, control, "best_ndcg@20")
        method_hr = _metric(best, method, "best_hr@20")
        control_hr = _metric(best, control, "best_hr@20")
        delta_ndcg = method_ndcg - control_ndcg
        delta_hr = method_hr - control_hr
        pass_ndcg = bool(delta_ndcg >= min_ndcg_delta)
        pass_hr = bool(delta_hr >= 0)
        rows.append(
            {
                "comparison": comparison,
                "method_label": method,
                "control_label": control,
                "method_best_ndcg@20": method_ndcg,
                "control_best_ndcg@20": control_ndcg,
                "delta_ndcg@20": delta_ndcg,
                "method_best_hr@20": method_hr,
                "control_best_hr@20": control_hr,
                "delta_hr@20": delta_hr,
                "min_ndcg_delta": min_ndcg_delta,
                "pass_ndcg_gate": pass_ndcg,
                "pass_hr_gate": pass_hr,
                "pass_seed43_gate": bool(pass_ndcg and pass_hr),
            }
        )
    out = pd.DataFrame(rows)
    for col in ["pass_ndcg_gate", "pass_hr_gate", "pass_seed43_gate"]:
        out[col] = out[col].astype(object)
    return out


def build_training_dynamics(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in best.iterrows():
        rows.append(
            {
                "run_label": row["run_label"],
                "best_epoch": int(row["best_epoch"]),
                "last_epoch": 100,
                "peak_minus_last_ndcg@20": float(row["best_ndcg@20"]) - float(row["last_ndcg@20"]),
                "peak_minus_last_hr@20": float(row["best_hr@20"]) - float(row["last_hr@20"]),
            }
        )
    return pd.DataFrame(rows)


def build_route_decision(pair_comparison: pd.DataFrame) -> dict:
    any_pass = bool(pair_comparison["pass_seed43_gate"].map(bool).any()) if not pair_comparison.empty else False
    q = pair_comparison[pair_comparison["comparison"].eq("M8q_minus_M8qs")]
    s = pair_comparison[pair_comparison["comparison"].eq("M8s_minus_M8ss")]
    q_delta = float(q.iloc[0]["delta_ndcg@20"]) if not q.empty else float("nan")
    q_hr = float(q.iloc[0]["delta_hr@20"]) if not q.empty else float("nan")
    s_delta = float(s.iloc[0]["delta_ndcg@20"]) if not s.empty else float("nan")
    if any_pass:
        route = "m8_seed43_gate_passed"
        next_action = "layer_audit_then_consider_multiseed"
    elif q_delta > 0 and q_hr >= 0 and s_delta < 0:
        route = "m8_gate_failed_but_q_only_is_safer"
        next_action = "m9_q_only_alpha_sweep_seed43"
    else:
        route = "m8_gate_failed"
        next_action = "stop_m7a_surface_family"
    return {
        "route": route,
        "next_action": next_action,
        "run_multi_seed_now": False,
        "seed43_any_pass": any_pass,
        "q_only_delta_ndcg@20": q_delta,
        "q_only_delta_hr@20": q_hr,
        "self_only_delta_ndcg@20": s_delta,
        "plain_explanation": "q-only is safer than self-only, but the gain is too small for multi-seed.",
    }


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_empty_"
    small = df[columns].copy()
    for col in small.columns:
        if pd.api.types.is_float_dtype(small[col]):
            small[col] = small[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    header = "| " + " | ".join(small.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(small.columns)) + " |"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |" for _, row in small.iterrows()]
    return "\n".join([header, sep, *body])


def write_result_md(path: Path, run_stamp: str, best: pd.DataFrame, comparison: pd.DataFrame, dynamics: pd.DataFrame, decision: dict, manifest_name: str) -> None:
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG Task4-revise-3 M8 surface ablation 分析结果
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - 实验结果
---

# {run_stamp} CCFCRec Amazon-VG Task4-revise-3 M8 surface ablation 分析结果

> [!info] 来源说明
> 上游设计：[[{DESIGN_NOTE_NAME}]]
> manifest（运行清单）：`{manifest_name}`

## 结论

```text
route = {decision["route"]}
next_action = {decision["next_action"]}
run_multi_seed_now = {decision["run_multi_seed_now"]}
```

通俗解释：只加 q_bpr（q 侧排序训练）比打乱版好一点，而且 HR@20 没有反向；只加 self_contrast（自对比训练）反而输给打乱版。所以后面只保留 q_bpr 方向，但当前增益太小，不能做多 seed。

## 最佳结果

{md_table(best, ["run_label", "best_epoch", "best_ndcg@20", "best_hr@20", "last_ndcg@20", "last_hr@20"])}

## 真实方法 vs shuffle

{md_table(comparison, ["comparison", "delta_ndcg@20", "delta_hr@20", "pass_seed43_gate"])}

## 后段衰减

{md_table(dynamics, ["run_label", "best_epoch", "last_epoch", "peak_minus_last_ndcg@20", "peak_minus_last_hr@20"])}

## 路线判断

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```
"""
    path.write_text(markdown, encoding="utf-8")


def write_route_md(path: Path, run_stamp: str, result_note_name: str, comparison: pd.DataFrame, decision: dict) -> None:
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG Task4-revise-3 M8 surface ablation 路线判断
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - 路线判断
---

# {run_stamp} CCFCRec Amazon-VG Task4-revise-3 M8 surface ablation 路线判断

## 来源

设计：[[{DESIGN_NOTE_NAME}]]
结果：[[{result_note_name}]]

## 判断

```text
route = {decision["route"]}
next_action = {decision["next_action"]}
run_multi_seed_now = {decision["run_multi_seed_now"]}
```

通俗解释：M8 告诉我们，不要继续加 self_contrast；只保留 q_bpr。但 q_bpr 现在只是小正向，所以下一步只能做 seed43 小范围强度筛选，不能多 seed。

{md_table(comparison, ["comparison", "delta_ndcg@20", "delta_hr@20", "pass_seed43_gate"])}
"""
    path.write_text(markdown, encoding="utf-8")


def build_outputs(output_root: Path, experiment_root: Path, run_stamp: str) -> Outputs:
    output_dir = output_root / f"{run_stamp} task4-revise3-m8-surface-ablation-analysis"
    return Outputs(
        output_dir=output_dir,
        best_summary_csv=output_dir / "task4_revise3_m8_best_summary.csv",
        pair_comparison_csv=output_dir / "task4_revise3_m8_pair_comparison.csv",
        training_dynamics_csv=output_dir / "task4_revise3_m8_training_dynamics.csv",
        manifest_json=output_dir / "run_manifest.json",
        result_md=output_dir / f"{run_stamp} CCFCRec Amazon-VG Task4-revise-3 M8 surface ablation 分析结果.md",
        route_md=experiment_root / f"{run_stamp} CCFCRec Amazon-VG Task4-revise-3 M8 surface ablation 路线判断.md",
    )


def run_analysis(args: argparse.Namespace) -> Outputs:
    if args.run_stamp:
        run_stamp = args.run_stamp
        run_date = run_stamp[:10]
        run_iso = datetime.strptime(run_stamp, "%Y-%m-%d %H%M%S").isoformat(timespec="seconds")
    else:
        run_stamp, run_date, run_iso = now_stamp()
    outputs = build_outputs(args.output_root, args.experiment_root, run_stamp)
    outputs.output_dir.mkdir(parents=True, exist_ok=True)
    best = build_best_summary(args.result_root)
    comparison = build_pair_comparison(best, min_ndcg_delta=args.min_ndcg_delta)
    dynamics = build_training_dynamics(best)
    decision = build_route_decision(comparison)
    best.to_csv(outputs.best_summary_csv, index=False)
    comparison.to_csv(outputs.pair_comparison_csv, index=False)
    dynamics.to_csv(outputs.training_dynamics_csv, index=False)
    result_note_name = outputs.result_md.stem
    write_result_md(outputs.result_md, run_stamp, best, comparison, dynamics, decision, outputs.manifest_json.name)
    write_route_md(outputs.route_md, run_stamp, result_note_name, comparison, decision)
    manifest = {
        "run_stamp": run_stamp,
        "run_date": run_date,
        "run_iso": run_iso,
        "analysis_script": "validata/analyze_amazon_vg_task4_revise3_m8_surface.py",
        "inputs": {"result_root": str(args.result_root)},
        "parameters": {"min_ndcg_delta": args.min_ndcg_delta},
        "outputs": {field: str(getattr(outputs, field)) for field in outputs.__dataclass_fields__},
        "decision": decision,
    }
    outputs.manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Task4-revise-3 M8 surface ablation.")
    parser.add_argument("--result_root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment_root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--run_stamp", default="")
    parser.add_argument("--min_ndcg_delta", type=float, default=0.0005)
    return parser.parse_args()


def main() -> None:
    outputs = run_analysis(parse_args())
    print(f"output_dir={outputs.output_dir}")
    print(f"result_md={outputs.result_md}")
    print(f"route_md={outputs.route_md}")


if __name__ == "__main__":
    main()
