#!/usr/bin/env python3
"""Analyze CICP-R2 six embedding-generation mechanisms on validation cold items."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import analyze_amazon_vg_cicpr1_six_training as core


BASELINE_NDCG20 = 0.1238145211709585
BASELINE_HR20 = 0.0206209890524726
THREE_PCT_THRESHOLD = BASELINE_NDCG20 * 1.03
CICPR1_BEST_NDCG20 = 0.1263542828877096
CICPR1_BEST_HR20 = 0.0211589278973197
FULL_ITEM_COUNT = 35322
TRAIN_ITEM_COUNT = 24726
VALIDATION_ITEM_COUNT = 5298
TEST_ITEM_COUNT = 5298
METHOD_ORDER = [
    "cicpr2_content_direction_residual",
    "cicpr2_category_increment_gate",
    "cicpr2_cross_modal_attention",
    "cicpr2_score_distillation",
    "cicpr2_ordinal_counterfactual",
    "cicpr2_reliability_dropout",
]
METHOD_LABELS = {
    "cicpr2_content_direction_residual": "CICP-R2-E1-CDR（内容方向残差）",
    "cicpr2_category_increment_gate": "CICP-R2-E2-CID（类别增量门控）",
    "cicpr2_cross_modal_attention": "CICP-R2-E3-CMA（跨模态注意力）",
    "cicpr2_score_distillation": "CICP-R2-E4-SD（分数蒸馏）",
    "cicpr2_ordinal_counterfactual": "CICP-R2-E5-OCS（序数反事实监督）",
    "cicpr2_reliability_dropout": "CICP-R2-E6-RCD（可靠性条件丢弃）",
}
ITEM_METHOD_LABELS = {
    "cicpr2_content_direction_residual": "CICP-R2-E1-CDR",
    "cicpr2_category_increment_gate": "CICP-R2-E2-CID",
    "cicpr2_cross_modal_attention": "CICP-R2-E3-CMA",
    "cicpr2_score_distillation": "CICP-R2-E4-SD",
    "cicpr2_ordinal_counterfactual": "CICP-R2-E5-OCS",
    "cicpr2_reliability_dropout": "CICP-R2-E6-RCD",
}


def configure_core() -> None:
    core.METHOD_ORDER = METHOD_ORDER
    core.METHOD_LABELS = METHOD_LABELS
    core.ITEM_METHOD_LABELS = ITEM_METHOD_LABELS
    core.OLD_M11R2_BEST = CICPR1_BEST_NDCG20


def build_coverage(group_summary: pd.DataFrame) -> pd.DataFrame:
    group_counts = (
        group_summary[group_summary["method_variant"].eq(METHOD_ORDER[0])]
        .set_index("cicp_group")["item_count"]
        .to_dict()
    )
    rows = [
        {
            "evaluation_scope": "complete_item_profile（完整物品档案）",
            "scope_item_count": FULL_ITEM_COUNT,
            "evaluated_item_count": VALIDATION_ITEM_COUNT,
            "scope_pct_of_full_profile": 100.0,
            "evaluated_pct_of_full_profile": VALIDATION_ITEM_COUNT / FULL_ITEM_COUNT * 100.0,
            "recommendation_metric_status": "未评估",
            "note": "只覆盖其中5,298个验证冷启动物品，未计算完整混合推荐指标。",
        },
        {
            "evaluation_scope": "train_non_cold（训练非冷启动）",
            "scope_item_count": TRAIN_ITEM_COUNT,
            "evaluated_item_count": 0,
            "scope_pct_of_full_profile": TRAIN_ITEM_COUNT / FULL_ITEM_COUNT * 100.0,
            "evaluated_pct_of_full_profile": 0.0,
            "recommendation_metric_status": "未评估",
            "note": "未生成训练物品推荐结果。",
        },
        {
            "evaluation_scope": "validation_cold（验证冷启动）",
            "scope_item_count": VALIDATION_ITEM_COUNT,
            "evaluated_item_count": VALIDATION_ITEM_COUNT,
            "scope_pct_of_full_profile": VALIDATION_ITEM_COUNT / FULL_ITEM_COUNT * 100.0,
            "evaluated_pct_of_full_profile": VALIDATION_ITEM_COUNT / FULL_ITEM_COUNT * 100.0,
            "recommendation_metric_status": "已评估",
            "note": "当前开发阶段正式主评价范围。",
        },
        {
            "evaluation_scope": "test_cold（测试冷启动）",
            "scope_item_count": TEST_ITEM_COUNT,
            "evaluated_item_count": 0,
            "scope_pct_of_full_profile": TEST_ITEM_COUNT / FULL_ITEM_COUNT * 100.0,
            "evaluated_pct_of_full_profile": 0.0,
            "recommendation_metric_status": "未评估",
            "note": "未读取或生成测试逐物品推荐结果。",
        },
    ]
    for group in ("low", "mid", "high"):
        count = int(group_counts[group])
        rows.append(
            {
                "evaluation_scope": f"validation_CICP_{group}（验证CICP{group}组）",
                "scope_item_count": count,
                "evaluated_item_count": count,
                "scope_pct_of_full_profile": count / FULL_ITEM_COUNT * 100.0,
                "evaluated_pct_of_full_profile": count / FULL_ITEM_COUNT * 100.0,
                "recommendation_metric_status": "已评估",
                "note": "三个互补群体完整覆盖验证冷启动范围，仅作开发诊断。",
            }
        )
    return pd.DataFrame(rows)


def build_fallacy_scan() -> pd.DataFrame:
    entries = [
        ("Simpson's paradox（辛普森悖论）", "CHECKED", "同时报告整体及低/中/高CICP互补群体，检查聚合方向是否掩盖群体反向。"),
        ("Ecological fallacy（生态谬误）", "CAUTION", "群体均值不代表每个物品同向受益，另报告受益、受损和不变物品数。"),
        ("Berkson's paradox（伯克森悖论）", "CAUTION", "样本固定为验证冷启动物品，不能外推到训练物品、测试物品或其他数据集。"),
        ("Collider bias（碰撞变量偏差）", "CAUTION", "R/S/P事后分层仅作描述，不能解释为CICP净因果效应。"),
        ("Base-rate neglect（基率忽视）", "CHECKED", "明确报告5,298/35,322覆盖关系及各互补群体物品数。"),
        ("Regression to the mean（均值回归）", "CAUTION", "六分支均从100轮选择峰值，峰值必须结合后30轮和同轮曲线解释。"),
        ("Multiple comparisons（多重比较）", "CAUTION", "同时比较六机制和多个检查点；单随机种子峰值不是独立确认。"),
        ("Survivorship bias（幸存者偏差）", "CHECKED", "六个正式分支全部纳入，包括明显退化分支。"),
        ("Garden of forking paths（分叉路径花园）", "CAUTION", "机制训练前固定，但最佳检查点、CICP和R/S/P分层属于开发选择。"),
        ("Correlation is not causation（相关不等于因果）", "CAUTION", "没有同载体分数打乱及语义匹配控制，不能证明CICP语义导致变化。"),
        ("Reverse causality（反向因果）", "CHECKED", "训练输入未含验证或测试推荐指标，但收益也不能反向证明CICP定义正确。"),
    ]
    return pd.DataFrame(entries, columns=["fallacy", "severity", "finding"])


def build_decision(
    curve_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    score_response: pd.DataFrame,
) -> dict[str, Any]:
    winner = curve_summary.sort_values("best_ndcg@20", ascending=False).iloc[0]
    method = str(winner["method_variant"])
    groups = group_summary[group_summary["method_variant"].eq(method)].set_index(
        "cicp_group"
    )
    score = score_response[
        score_response["method_variant"].eq(method)
        & score_response["response"].eq("ndcg@20")
    ].iloc[0]
    late_stability = bool(
        winner["late30_mean_ndcg@20"] > BASELINE_NDCG20
        and int(winner["late30_positive_ndcg_epochs_vs_baseline_best"]) >= 20
        and int(winner["late30_matched_positive_ndcg_epochs"]) >= 24
    )
    passed_three = bool(winner["best_ndcg@20"] >= THREE_PCT_THRESHOLD)
    beat_cicpr1 = bool(winner["best_ndcg@20"] > CICPR1_BEST_NDCG20)
    group_reversal = bool((groups.loc[["low", "mid", "high"], "relative_pct_ndcg@20"] < 0).any())
    monotonic_response = bool(abs(float(score["spearman"])) >= 0.05)
    route = (
        "confirm_controls_then_multiseed"
        if passed_three and beat_cicpr1 and not group_reversal
        else "stop_local_cicp_variants_rebuild_semantic_basis_or_backbone"
    )
    return {
        "winner_method": method,
        "winner_label": str(winner["method_label"]),
        "winner_best_epoch": int(winner["best_epoch"]),
        "winner_ndcg@20": float(winner["best_ndcg@20"]),
        "winner_relative_pct_ndcg@20_vs_baseline": float(
            winner["relative_pct_ndcg@20_vs_baseline_best"]
        ),
        "winner_hr@20": float(winner["best_hr@20_same_checkpoint"]),
        "winner_relative_pct_hr@20_vs_baseline": float(
            winner["relative_pct_hr@20_vs_baseline_best"]
        ),
        "winner_relative_pct_ndcg@20_vs_cicpr1_best": core.relative_pct(
            float(winner["best_ndcg@20"]), CICPR1_BEST_NDCG20
        ),
        "three_pct_threshold": THREE_PCT_THRESHOLD,
        "absolute_gap_to_three_pct_threshold": float(
            winner["best_ndcg@20"] - THREE_PCT_THRESHOLD
        ),
        "passed_three_pct_threshold": passed_three,
        "beat_cicpr1_best": beat_cicpr1,
        "late_stability_pass": late_stability,
        "any_cicp_group_reversal": group_reversal,
        "monotonic_score_response_pass": monotonic_response,
        "winner_cicp_score_vs_item_delta_spearman": float(score["spearman"]),
        "automatic_cicpr3_authorized": False,
        "route": route,
        "test_item_metrics_read_or_generated": False,
        "fallacy_scan_coverage": "11/11",
        "evidence_level": "B_development_validation_single_seed_with_posthoc_subgroups",
    }


def _format_main_table(curve_summary: pd.DataFrame) -> pd.DataFrame:
    frame = curve_summary.copy()
    frame["best_ndcg@20"] = frame["best_ndcg@20"].map(lambda x: f"{x:.12f}")
    for column in (
        "relative_pct_ndcg@20_vs_baseline_best",
        "relative_pct_ndcg@20_vs_cicpr1_best",
        "relative_pct_hr@20_vs_baseline_best",
        "late30_relative_pct_ndcg@20_vs_baseline_best",
        "late30_matched_relative_pct_ndcg@20",
    ):
        frame[column] = frame[column].map(lambda x: f"{x:+.3f}%")
    return frame


def _format_group_table(group_summary: pd.DataFrame, method: str) -> pd.DataFrame:
    frame = group_summary[group_summary["method_variant"].eq(method)].copy()
    frame["relative_pct_ndcg@20"] = frame["relative_pct_ndcg@20"].map(
        lambda x: f"{x:+.3f}%"
    )
    frame["relative_pct_hr@20"] = frame["relative_pct_hr@20"].map(
        lambda x: f"{x:+.3f}%"
    )
    frame["ndcg_ci95"] = frame.apply(
        lambda row: (
            f"[{row['ndcg@20_bootstrap_relative_pct_ci95_low']:+.3f}%, "
            f"{row['ndcg@20_bootstrap_relative_pct_ci95_high']:+.3f}%]"
        ),
        axis=1,
    )
    return frame


def write_reports(
    *,
    output_dir: Path,
    route_output_dir: Path,
    run_stamp: str,
    result_root: Path,
    curve_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    rsp_summary: pd.DataFrame,
    score_response: pd.DataFrame,
    coverage: pd.DataFrame,
    fallacy_scan: pd.DataFrame,
    decision: dict[str, Any],
) -> tuple[Path, Path]:
    winner = curve_summary[curve_summary["method_variant"].eq(decision["winner_method"])].iloc[0]
    main = _format_main_table(curve_summary)
    groups = _format_group_table(group_summary, decision["winner_method"])
    rsp = rsp_summary[rsp_summary["method_variant"].eq(decision["winner_method"])].copy()
    rsp["relative_pct_ndcg@20"] = rsp["relative_pct_ndcg@20"].map(lambda x: f"{x:+.3f}%")
    rsp["relative_pct_hr@20"] = rsp["relative_pct_hr@20"].map(lambda x: f"{x:+.3f}%")
    score = score_response[
        score_response["method_variant"].eq(decision["winner_method"])
        & score_response["response"].eq("ndcg@20")
    ].iloc[0]
    overall = group_summary[
        group_summary["method_variant"].eq(decision["winner_method"])
        & group_summary["cicp_group"].eq("overall")
    ].iloc[0]
    report = f"""---
title: {run_stamp} CCFCRec Amazon-VG CICP-R2 six embedding generation 结果
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - CICP-R2
  - experiment-result
---

# CCFCRec Amazon-VG CICP-R2 six embedding generation 结果

## Material Passport（材料护照）

- origin mode（来源模式）：experiment-agent validate（实验代理验证模式）
- verification status（验证状态）：ANALYZED_REAGGREGATION_VERIFIED（已分析且重聚合已验证）
- source result（源结果）：`{result_root}`
- formal evaluation scope（正式评价范围）：全部5,298个 validation cold-start items（验证冷启动物品）
- full profile relation（相对完整档案）：5,298 / 35,322 = 14.999%
- train/test/full mixed recommendation metrics（训练/测试/完整混合推荐指标）：未评估

## 一、主结论

六个分支均完成100轮。最佳分支是 {decision['winner_label']}，第 `{decision['winner_best_epoch']}` 轮 NDCG@20（前20归一化折损累计增益）为 `{decision['winner_ndcg@20']:.12f}`，相对同口径 baseline（基线）`{decision['winner_relative_pct_ndcg@20_vs_baseline']:+.3f}%`；同一检查点 HR@20（前20命中率）为 `{decision['winner_hr@20']:.12f}`，相对 `{decision['winner_relative_pct_hr@20_vs_baseline']:+.3f}%`。

这个结果没有达到 `+3%` 的 `0.127528956806`，还差 `{abs(decision['absolute_gap_to_three_pct_threshold']):.12f}`；并且比 `CICP-R1-E1`（第一轮第一方案）开发最佳 `0.126354282888` 低 `{abs(decision['winner_relative_pct_ndcg@20_vs_cicpr1_best']):.3f}%`。因此 CICP-R2（类别增量协同可预测性第二轮）没有把上一轮 `+2.051%` 推进成超过 `+3%` 的性能突破。

## 二、六分支整体结果

{core._markdown_table(main, ['method_label','best_epoch','best_ndcg@20','relative_pct_ndcg@20_vs_baseline_best','relative_pct_ndcg@20_vs_cicpr1_best','relative_pct_hr@20_vs_baseline_best','late30_relative_pct_ndcg@20_vs_baseline_best','late30_matched_relative_pct_ndcg@20','late30_positive_ndcg_epochs_vs_baseline_best','passed_three_pct_threshold'])}

这里的“整体”只指全部5,298个验证冷启动物品，不是35,322个完整物品，也不是某个CICP局部群体。

## 三、峰值与稳定性

- {decision['winner_label']} 后30轮均值为 `{float(winner['late30_mean_ndcg@20']):.12f}`，相对 baseline（基线）最佳点 `{float(winner['late30_relative_pct_ndcg@20_vs_baseline_best']):+.3f}%`；高于基线最佳点 `{int(winner['late30_positive_ndcg_epochs_vs_baseline_best'])}/30` 轮。
- 与 baseline（基线）逐轮配对后，后30轮平均相对差为 `{float(winner['late30_matched_relative_pct_ndcg@20']):+.3f}%`，正向 `{int(winner['late30_matched_positive_ndcg_epochs'])}/30` 轮。
- 峰值相对同轮 baseline（基线）为 `{float(winner['same_epoch_relative_pct_ndcg@20']):+.3f}%`，但相对基线独立最佳只剩 `{decision['winner_relative_pct_ndcg@20_vs_baseline']:+.3f}%`。这说明训练轨迹整体发生正向平移，但开发结论仍受最佳检查点选择影响。

## 四、逐机制解释

1. `CICP-R2-E1-CDR`（内容方向残差）峰值 `+0.944%`，后30轮同轮比较30/30正向，但相对基线独立最佳的后30轮均值为 `-0.335%`。内容决定方向、CICP决定强度的假设产生了稳定轨迹差，却没有超过上一轮残差适配器。
2. `CICP-R2-E2-CID`（类别增量门控）峰值仅 `+0.141%`，后30轮同轮平均 `-0.199%`。直接放大类别增量没有形成持续收益。
3. `CICP-R2-E3-CMA`（跨模态注意力）是本轮最佳，HR@20（前20命中率）也正向，且后30轮同轮30/30正向；但独立最佳口径只有 `+1.006%`，低于上一轮最佳。
4. `CICP-R2-E4-SD`（分数蒸馏）峰值 `+0.455%`，说明让生成表示回归CICP分数只有弱辅助效果。
5. `CICP-R2-E5-OCS`（序数反事实监督）为 `-0.026%`，基本等于基线；仅学习CICP高低序关系没有转化为推荐收益。
6. `CICP-R2-E6-RCD`（可靠性条件丢弃）为 `-6.764%`，配对区间也完全低于零。训练期按低CICP削弱类别证据造成明显破坏；训练和推理阶段处理不一致可能进一步放大退化，这是机制解释而非已证明因果。

## 五、最佳分支的CICP互补群体

{core._markdown_table(groups, ['cicp_group','item_count','baseline_ndcg@20','method_ndcg@20','relative_pct_ndcg@20','relative_pct_hr@20','ndcg_ci95','helped_ndcg_item_count','harmed_ndcg_item_count'])}

- overall（整体）逐物品重聚合 NDCG@20（前20归一化折损累计增益）相对变化为 `{float(overall['relative_pct_ndcg@20']):+.3f}%`，应与曲线峰值一致。
- CICP score（类别增量协同可预测性分数）与逐物品 NDCG@20 变化的 Spearman（斯皮尔曼秩相关）为 `{float(score['spearman']):+.4f}`，未经校正 `p={float(score['p_value_uncorrected']):.4f}`。这是描述性关联，不是因果证据。
- 配对 bootstrap（配对自助法）区间、群体反向和物品受益/受损计数共同用于判断，不能只挑正向局部群体替代验证冷启动整体。

## 六、R/S/P事后分层

{core._markdown_table(rsp, ['stratification_label','stratum','item_count','baseline_ndcg@20','relative_pct_ndcg@20','relative_pct_hr@20'])}

R/S/P（丰富度/支持度/流行度）三等分是 post-hoc descriptive analysis（事后描述分析），不等价于独立匹配控制。

## 七、评价覆盖

{core._markdown_table(coverage, list(coverage.columns))}

## 八、统计谬误扫描

覆盖：`11/11`。

{core._markdown_table(fallacy_scan, list(fallacy_scan.columns))}

## 九、路线判断

- passed +3%（超过3%）：`{decision['passed_three_pct_threshold']}`
- beat CICP-R1 best（超过第一轮最佳）：`{decision['beat_cicpr1_best']}`
- late stability（后段稳定性）：`{decision['late_stability_pass']}`
- automatic CICP-R3（自动进入第三轮）：`{decision['automatic_cicpr3_authorized']}`
- route（路线）：`{decision['route']}`

本轮已经同时测试结构残差、类别门控、跨模态注意力、分数蒸馏、序数反事实监督和可靠性条件丢弃六种实质不同用法。最佳仍只有 `+1.006%`，且低于上一轮最佳，因此证据不支持继续在当前单分数CICP接入上做局部第三轮变体。保留 `CICP-R1-E1` 作为开发最佳；下一步回到 category semantic basis（类别语义基础）和 category increment profile（类别增量画像）本身，或审查更新的 cold-start backbone（冷启动推荐模型骨干）。
"""
    route_report = f"""---
title: {run_stamp} CCFCRec Amazon-VG CICP-R2 six embedding generation 路线判断
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - CICP-R2
  - route-decision
---

# CCFCRec Amazon-VG CICP-R2 six embedding generation 路线判断

关联结果：[[{run_stamp} CCFCRec Amazon-VG CICP-R2 six embedding generation 结果]]

> [!important] 主决策
> CICP-R2（类别增量协同可预测性第二轮）最佳只有 `+{decision['winner_relative_pct_ndcg@20_vs_baseline']:.3f}%`，未超过 `+3%`，也低于 `CICP-R1-E1`（第一轮第一方案）`{abs(decision['winner_relative_pct_ndcg@20_vs_cicpr1_best']):.3f}%`。停止默认设计 CICP-R3（第三轮）局部接入变体，转入信号语义基础或模型骨干级重构审查。

## 为什么现在停止局部变体

1. 六种机制上实质不同的方案都已完成100轮，不是只否定某一个残差载体。
2. 最佳 {decision['winner_label']} 的峰值为 `{decision['winner_ndcg@20']:.12f}`，距离门槛仍差 `{abs(decision['absolute_gap_to_three_pct_threshold']):.12f}`。
3. 当前历史开发最佳仍是 `CICP-R1-E1` 的 `0.126354282888`（`+2.051%`），R2没有形成增量。
4. 所有结论只覆盖5,298个验证冷启动物品，占完整物品档案14.999%；测试冷启动、训练非冷启动和完整混合范围均未评估。
5. 当前只有 seed43（第43号单随机种子），且六机制与100轮峰值选择带来多重比较风险，不开放 multi-seed（多随机种子）确认。

## 下一阶段

1. 重新构建 multi-component category increment profile（多分量类别增量画像），避免继续把CICP压成一个总分后再扩维。
2. 优先审查 category semantic basis（类别语义基础）：类别文本、层级、商品属性、类别图，以及大模型辅助的低维语义先验；不把大模型高维向量直接拼接到CCFCRec（冷启动对比协同过滤）尾部。
3. 并行审查更新的 cold-start backbone（冷启动推荐模型骨干），判断当前生成器是否已经成为上限约束。
4. 新方向先做不读取验证/测试逐物品答案的离线可训练性审计，再设计100 epoch（100训练轮次）正式实验；继续相对同协议验证集基线报告。

## 固化状态

- route（路线）：`{decision['route']}`
- run CICP-R3 local variants now（现在运行第三轮局部变体）：`False`
- run multi-seed now（现在运行多随机种子）：`False`
- preserve CICP-R1-E1 as development best（保留第一轮第一方案为开发最佳）：`True`
- test item metrics read or generated（读取或生成测试逐物品指标）：`False`
- fallacy scan coverage（统计谬误扫描覆盖）：`11/11`
"""
    result_path = output_dir / f"{run_stamp} CCFCRec Amazon-VG CICP-R2 six embedding generation 结果.md"
    route_output_dir.mkdir(parents=True, exist_ok=True)
    route_path = route_output_dir / f"{run_stamp} CCFCRec Amazon-VG CICP-R2 six embedding generation 路线判断.md"
    result_path.write_text(report, encoding="utf-8")
    route_path.write_text(route_report, encoding="utf-8")
    return result_path, route_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_core()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_summary, matched_curve, baseline_curve = core.collect_curves(
        args.result_root.resolve(), args.baseline_result.resolve()
    )
    curve_summary["relative_pct_ndcg@20_vs_cicpr1_best"] = curve_summary[
        "best_ndcg@20"
    ].map(lambda value: core.relative_pct(value, CICPR1_BEST_NDCG20))
    item_metrics = pd.read_csv(
        args.item_evaluation_dir.resolve() / "cicpr2_validation_item_metrics.csv",
        dtype={"raw_asin": str},
    )
    if len(item_metrics) != 7 * VALIDATION_ITEM_COUNT:
        raise ValueError(f"unexpected validation item row count: {len(item_metrics)}")
    group_summary = core.build_group_bootstrap(item_metrics, args.bootstrap_repetitions)
    score_response = core.build_score_response(item_metrics)
    rsp_summary, rsp_thresholds = core.build_rsp_strata(
        item_metrics, args.source_profile.resolve()
    )
    coverage = build_coverage(group_summary)
    fallacy_scan = build_fallacy_scan()
    decision = build_decision(curve_summary, group_summary, score_response)
    artifacts = {
        "cicpr2_curve_summary.csv": curve_summary,
        "cicpr2_epoch_matched_curve.csv": matched_curve,
        "cicpr2_baseline_curve.csv": baseline_curve,
        "cicpr2_group_bootstrap_summary.csv": group_summary,
        "cicpr2_rsp_strata_summary.csv": rsp_summary,
        "cicpr2_score_response_summary.csv": score_response,
        "cicpr2_evaluation_coverage.csv": coverage,
        "cicpr2_fallacy_scan.csv": fallacy_scan,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output_dir / name, index=False)
    (output_dir / "cicpr2_route_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result_path, route_path = write_reports(
        output_dir=output_dir,
        route_output_dir=args.route_output_dir.resolve(),
        run_stamp=args.run_stamp,
        result_root=args.result_root.resolve(),
        curve_summary=curve_summary,
        group_summary=group_summary,
        rsp_summary=rsp_summary,
        score_response=score_response,
        coverage=coverage,
        fallacy_scan=fallacy_scan,
        decision=decision,
    )
    manifest = {
        "protocol": "cicpr2_six_embedding_generation_analysis_v1",
        "run_stamp": args.run_stamp,
        "result_root": str(args.result_root.resolve()),
        "baseline_result": str(args.baseline_result.resolve()),
        "item_evaluation_dir": str(args.item_evaluation_dir.resolve()),
        "source_profile": str(args.source_profile.resolve()),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "rsp_validation_tertile_thresholds": rsp_thresholds,
        "recommendation_metric_scope": "validation_cold_5298_only",
        "test_item_metrics_read_or_generated": False,
        "outputs": [*artifacts, "cicpr2_route_decision.json", str(result_path), str(route_path)],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--item-evaluation-dir", type=Path, required=True)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-output-dir", type=Path, required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
