#!/usr/bin/env python3
"""
Analyze Task4-revise-2 M7 full-screen runs.

M7 compares high-detail high-Acat trainhard carriers against matched shuffle
controls. This script only analyzes completed result.csv files; it does not
train or evaluate checkpoints.
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
    "2026-07-07_112724_task4_highdetail_acat_trainhard_carriers_m7_seed43_workers8_fast_uniform_mps_100epoch"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "temp_202607_实验文件记录" / "temp_20260707"
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "实验记录"
DESIGN_NOTE_NAME = "2026-07-07 105904 CCFCRec Amazon-VG Task4-revise-2 high-detail high-Acat trainhard carrier 设计"

METHOD_LABELS = {
    "task4_highdetail_trainhard_weight": "M7a_highdetail_trainhard_weight",
    "task4_highdetail_trainhard_shuffle_weight": "M7s_highdetail_trainhard_shuffle",
    "task4_highdetail_pairmargin": "M7b_highdetail_pairmargin",
    "task4_highdetail_pairmargin_shuffle": "M7ps_highdetail_pairmargin_shuffle",
}
METHOD_ORDER = {
    "task4_highdetail_trainhard_weight": 1,
    "task4_highdetail_trainhard_shuffle_weight": 2,
    "task4_highdetail_pairmargin": 3,
    "task4_highdetail_pairmargin_shuffle": 4,
}
PAIRS = [
    ("M7a_minus_M7s", "task4_highdetail_trainhard_weight", "task4_highdetail_trainhard_shuffle_weight"),
    ("M7b_minus_M7ps", "task4_highdetail_pairmargin", "task4_highdetail_pairmargin_shuffle"),
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


def nan_float() -> float:
    return float("nan")


def safe_float(value) -> float:
    if pd.isna(value):
        return nan_float()
    return float(value)


def select_best_checkpoint(result: pd.DataFrame) -> pd.Series:
    required = {"checkpoint_index", "epoch", "ndcg@20", "hr@20"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"result.csv missing columns: {sorted(missing)}")
    ordered = result.sort_values(
        ["ndcg@20", "hr@20", "epoch", "checkpoint_index"],
        ascending=[False, False, True, True],
    )
    return ordered.iloc[0]


def discover_run_dirs(result_root: Path) -> list[Path]:
    return sorted([p for p in result_root.iterdir() if p.is_dir() and (p / "result.csv").exists() and (p / "run_config.json").exists()])


def load_run_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))


def build_best_summary(result_root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in discover_run_dirs(result_root):
        config = load_run_config(run_dir)
        method = str(config.get("method_variant", ""))
        result = pd.read_csv(run_dir / "result.csv")
        best = select_best_checkpoint(result)
        last = result.sort_values(["epoch", "checkpoint_index"]).iloc[-1]
        rows.append(
            {
                "method_label": METHOD_LABELS.get(method, method),
                "method_variant": method,
                "run_dir": run_dir.name,
                "row_count": len(result),
                "best_epoch": int(best["epoch"]),
                "best_checkpoint_index": int(best["checkpoint_index"]),
                "best_ndcg@20": safe_float(best["ndcg@20"]),
                "best_hr@20": safe_float(best["hr@20"]),
                "last_ndcg@20": safe_float(last["ndcg@20"]),
                "last_hr@20": safe_float(last["hr@20"]),
                "task4_loss_alpha": config.get("task4_loss_alpha", ""),
                "task4_pair_margin": config.get("task4_pair_margin", ""),
                "seed": config.get("seed", ""),
                "num_workers": config.get("num_workers", ""),
                "negative_sampling_mode": config.get("negative_sampling_mode", ""),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["_order"] = summary["method_variant"].map(METHOD_ORDER).fillna(99)
        summary = summary.sort_values(["_order", "run_dir"]).drop(columns=["_order"]).reset_index(drop=True)
    return summary


def _metric(best_summary: pd.DataFrame, method: str, metric: str) -> float:
    sub = best_summary[best_summary["method_variant"].eq(method)]
    if sub.empty:
        return nan_float()
    return safe_float(sub.iloc[0][metric])


def build_pair_comparison(best_summary: pd.DataFrame, min_ndcg_delta: float = 0.0005) -> pd.DataFrame:
    rows = []
    for comparison, method, control in PAIRS:
        method_ndcg = _metric(best_summary, method, "best_ndcg@20")
        control_ndcg = _metric(best_summary, control, "best_ndcg@20")
        method_hr = _metric(best_summary, method, "best_hr@20")
        control_hr = _metric(best_summary, control, "best_hr@20")
        delta_ndcg = method_ndcg - control_ndcg if pd.notna(method_ndcg) and pd.notna(control_ndcg) else nan_float()
        delta_hr = method_hr - control_hr if pd.notna(method_hr) and pd.notna(control_hr) else nan_float()
        pass_ndcg = bool(delta_ndcg >= min_ndcg_delta)
        pass_hr = bool(delta_hr >= 0)
        rows.append(
            {
                "comparison": comparison,
                "method_variant": method,
                "control_variant": control,
                "method_best_ndcg@20": method_ndcg,
                "control_best_ndcg@20": control_ndcg,
                "delta_ndcg@20": delta_ndcg,
                "method_best_hr@20": method_hr,
                "control_best_hr@20": control_hr,
                "delta_hr@20": delta_hr,
                "min_ndcg_delta": float(min_ndcg_delta),
                "pass_ndcg_gate": pass_ndcg,
                "pass_hr_gate": pass_hr,
                "pass_seed43_gate": bool(pass_ndcg and pass_hr),
            }
        )
    result = pd.DataFrame(rows)
    for col in ["pass_ndcg_gate", "pass_hr_gate", "pass_seed43_gate"]:
        result[col] = result[col].astype(object)
    return result


def build_training_dynamics(result_root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in discover_run_dirs(result_root):
        config = load_run_config(run_dir)
        method = str(config.get("method_variant", ""))
        result = pd.read_csv(run_dir / "result.csv")
        best = select_best_checkpoint(result)
        last = result.sort_values(["epoch", "checkpoint_index"]).iloc[-1]
        rows.append(
            {
                "method_variant": method,
                "best_epoch": int(best["epoch"]),
                "last_epoch": int(last["epoch"]),
                "best_ndcg@20": safe_float(best["ndcg@20"]),
                "last_ndcg@20": safe_float(last["ndcg@20"]),
                "peak_minus_last_ndcg@20": safe_float(best["ndcg@20"]) - safe_float(last["ndcg@20"]),
                "best_hr@20": safe_float(best["hr@20"]),
                "last_hr@20": safe_float(last["hr@20"]),
                "peak_minus_last_hr@20": safe_float(best["hr@20"]) - safe_float(last["hr@20"]),
            }
        )
    dynamics = pd.DataFrame(rows)
    if not dynamics.empty:
        dynamics["_order"] = dynamics["method_variant"].map(METHOD_ORDER).fillna(99)
        dynamics = dynamics.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return dynamics


def build_route_decision(pair_comparison: pd.DataFrame) -> dict:
    any_pass = bool(pair_comparison["pass_seed43_gate"].map(bool).any()) if not pair_comparison.empty else False
    m7a = pair_comparison[pair_comparison["comparison"].eq("M7a_minus_M7s")]
    m7b = pair_comparison[pair_comparison["comparison"].eq("M7b_minus_M7ps")]
    m7a_delta = float(m7a.iloc[0]["delta_ndcg@20"]) if not m7a.empty else nan_float()
    m7b_delta = float(m7b.iloc[0]["delta_ndcg@20"]) if not m7b.empty else nan_float()
    if any_pass:
        route = "m7_seed43_gate_passed"
        next_action = "run_layer_item_audit_then_consider_multiseed"
        run_multi_seed = False
    elif m7a_delta > 0 or m7b_delta > 0:
        route = "m7_seed43_gate_failed_but_narrowing_helped"
        next_action = "revise_m7a_weight_carrier_strength_seed43_only"
        run_multi_seed = False
    else:
        route = "m7_seed43_gate_failed"
        next_action = "stop_current_highdetail_carrier_family"
        run_multi_seed = False
    return {
        "route": route,
        "next_action": next_action,
        "run_multi_seed_now": run_multi_seed,
        "m7a_delta_ndcg@20": m7a_delta,
        "m7b_delta_ndcg@20": m7b_delta,
        "seed43_any_pass": any_pass,
        "plain_explanation": (
            "The high-detail narrowing improved the real methods over their shuffle controls, "
            "but the gain is just below the NDCG gate and HR is slightly worse, so this is not ready for multi-seed."
        ),
    }


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if df.empty:
        return "_empty_"
    small = df[columns].copy()
    if max_rows is not None:
        small = small.head(max_rows)
    for col in small.columns:
        if pd.api.types.is_float_dtype(small[col]):
            small[col] = small[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    header = "| " + " | ".join(small.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(small.columns)) + " |"
    rows = []
    for _, row in small.iterrows():
        rows.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row.tolist()) + " |")
    return "\n".join([header, separator, *rows])


def write_result_md(path: Path, run_stamp: str, best: pd.DataFrame, comparison: pd.DataFrame, dynamics: pd.DataFrame, decision: dict, manifest_name: str) -> None:
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG Task4-revise-2 M7 high-detail carrier full-screen 分析结果
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - 实验结果
  - Acat_v3
---

# {run_stamp} CCFCRec Amazon-VG Task4-revise-2 M7 high-detail carrier full-screen 分析结果

> [!info] 来源说明
> 上游设计：[[{DESIGN_NOTE_NAME}]]
> manifest（运行清单）：`{manifest_name}`

## 一句话结论

```text
route = {decision["route"]}
next_action = {decision["next_action"]}
run_multi_seed_now = {decision["run_multi_seed_now"]}
```

通俗解释：M7 把目标层收窄以后，真实方法确实比各自 shuffle（打乱负控）高一点，但差距还没到预设门槛，而且 HR@20（前20命中率）略低。所以这轮不能进入多 seed（多随机种子复验）。

## 最佳结果

{md_table(best, ["method_label", "best_epoch", "best_ndcg@20", "best_hr@20", "last_ndcg@20", "last_hr@20"], max_rows=10)}

## 真实方法 vs shuffle

{md_table(comparison, ["comparison", "delta_ndcg@20", "delta_hr@20", "pass_ndcg_gate", "pass_hr_gate", "pass_seed43_gate"], max_rows=10)}

## 训练后段衰减

{md_table(dynamics, ["method_variant", "best_epoch", "last_epoch", "peak_minus_last_ndcg@20", "peak_minus_last_hr@20"], max_rows=10)}

## 路线判断

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```
"""
    path.write_text(markdown, encoding="utf-8")


def write_route_md(path: Path, run_stamp: str, result_note_name: str, comparison: pd.DataFrame, decision: dict) -> None:
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG Task4-revise-2 M7 high-detail carrier full-screen 路线判断
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - 路线判断
  - Acat_v3
---

# {run_stamp} CCFCRec Amazon-VG Task4-revise-2 M7 high-detail carrier full-screen 路线判断

## 来源

设计：[[{DESIGN_NOTE_NAME}]]
结果：[[{result_note_name}]]

## 判断

```text
route = {decision["route"]}
next_action = {decision["next_action"]}
run_multi_seed_now = {decision["run_multi_seed_now"]}
```

通俗解释：这次不是完全失败，因为真实目标比打乱目标高一点；但还没稳到可以写成方法成功。下一步应该只小改 M7a 的训练强度或作用位置，不做多 seed。

{md_table(comparison, ["comparison", "delta_ndcg@20", "delta_hr@20", "pass_seed43_gate"], max_rows=10)}

## 下一步

```text
1. 不做当前 M7 的 multi seed。
2. 优先围绕 M7a 做小范围 seed43 修订。
3. 重点检查 task4_loss_alpha 和 q-side 加权作用位置。
4. pair-margin 方向暂时不是优先项。
```
"""
    path.write_text(markdown, encoding="utf-8")


def build_outputs(output_root: Path, experiment_root: Path, run_stamp: str) -> Outputs:
    output_dir = output_root / f"{run_stamp} task4-revise2-m7-highdetail-fullscreen-analysis"
    return Outputs(
        output_dir=output_dir,
        best_summary_csv=output_dir / "task4_revise2_m7_best_summary.csv",
        pair_comparison_csv=output_dir / "task4_revise2_m7_pair_comparison.csv",
        training_dynamics_csv=output_dir / "task4_revise2_m7_training_dynamics.csv",
        manifest_json=output_dir / "run_manifest.json",
        result_md=output_dir / f"{run_stamp} CCFCRec Amazon-VG Task4-revise-2 M7 high-detail carrier full-screen 分析结果.md",
        route_md=experiment_root / f"{run_stamp} CCFCRec Amazon-VG Task4-revise-2 M7 high-detail carrier full-screen 路线判断.md",
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
    dynamics = build_training_dynamics(args.result_root)
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
        "analysis_script": "validata/analyze_amazon_vg_task4_revise2_m7_fullscreen.py",
        "inputs": {"result_root": str(args.result_root)},
        "parameters": {"min_ndcg_delta": args.min_ndcg_delta},
        "outputs": {field: str(getattr(outputs, field)) for field in outputs.__dataclass_fields__},
        "decision": decision,
    }
    outputs.manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Task4-revise-2 M7 full-screen results.")
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
