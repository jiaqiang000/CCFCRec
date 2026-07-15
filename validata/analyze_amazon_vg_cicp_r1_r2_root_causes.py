#!/usr/bin/env python3
"""Synthesize route-level root causes for CICP-R1-E1 and CICP-R2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_amazon_vg_cicpr1_six_training import _markdown_table


BASELINE_NDCG20 = 0.1238145211709585
FULL_ITEM_COUNT = 35322
VALIDATION_ITEM_COUNT = 5298
DIRECT_METHODS = [
    "CICP-R1-E1",
    "CICP-R2-E1-CDR",
    "CICP-R2-E2-CID",
    "CICP-R2-E3-CMA",
]


def relative_pct(value: float, baseline: float) -> float:
    return (float(value) / float(baseline) - 1.0) * 100.0


def paired_bootstrap(
    values: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        selected = rng.integers(0, len(values), len(values))
        estimates[index] = float(values[selected].mean())
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def build_contrasts(item_metrics: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rows = []
    contrast_specs = {
        "CICP-R1-E1": [("true", "off"), ("true", "shuffle"), ("true", "neutral"), ("true", "invert")],
        "CICP-R2-E1-CDR": [("true", "off"), ("true", "shuffle"), ("true", "neutral"), ("true", "invert")],
        "CICP-R2-E2-CID": [("true", "off"), ("true", "shuffle"), ("true", "neutral"), ("true", "invert"), ("one", "true")],
        "CICP-R2-E3-CMA": [("true", "off"), ("true", "shuffle"), ("true", "neutral"), ("true", "invert"), ("one", "true")],
    }
    for method_index, (method, contrasts) in enumerate(contrast_specs.items()):
        frame = item_metrics[item_metrics["method_label"].eq(method)]
        if frame.empty:
            continue
        pivot = frame.pivot(index="raw_asin", columns="intervention_mode", values="ndcg@20")
        for contrast_index, (left, right) in enumerate(contrasts):
            delta = (pivot[left] - pivot[right]).to_numpy(dtype=float)
            low, high = paired_bootstrap(
                delta,
                seed=43 + method_index * 100 + contrast_index,
                repetitions=repetitions,
            )
            left_mean = float(pivot[left].mean())
            right_mean = float(pivot[right].mean())
            rows.append(
                {
                    "method_label": method,
                    "contrast": f"{left}_minus_{right}",
                    "left_ndcg@20": left_mean,
                    "right_ndcg@20": right_mean,
                    "absolute_delta_ndcg@20": left_mean - right_mean,
                    "relative_pct_ndcg@20_vs_right": relative_pct(left_mean, right_mean),
                    "bootstrap_absolute_ci95_low": low,
                    "bootstrap_absolute_ci95_high": high,
                    "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
                    "helped_item_count": int((delta > 0.0).sum()),
                    "harmed_item_count": int((delta < 0.0).sum()),
                    "equal_item_count": int((delta == 0.0).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_curve_process_summary(
    cicpr1_curve: pd.DataFrame,
    cicpr2_curve: pd.DataFrame,
) -> pd.DataFrame:
    r1 = cicpr1_curve[cicpr1_curve["method_variant"].eq("cicpr1_e4_residual")].copy()
    r1["round"] = "CICP-R1"
    r1["method_label"] = "CICP-R1-E1（残差适配器）"
    r2 = cicpr2_curve.copy()
    r2["round"] = "CICP-R2"
    keep = [
        "round",
        "method_label",
        "best_epoch",
        "best_ndcg@20",
        "relative_pct_ndcg@20_vs_baseline_best",
        "relative_pct_hr@20_vs_baseline_best",
        "late30_relative_pct_ndcg@20_vs_baseline_best",
        "late30_matched_relative_pct_ndcg@20",
        "late30_matched_positive_ndcg_epochs",
    ]
    return pd.concat([r1[keep], r2[keep]], ignore_index=True)


def build_parameter_scan(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = {
        "CICP-R1-E1": {
            "parameter": "residual cap（残差上限）",
            "default": 0.15,
            "modes": {0.05: "dose_cap_0.05", 0.10: "dose_cap_0.1", 0.15: "true", 0.25: "dose_cap_0.25", 0.40: "dose_cap_0.4"},
        },
        "CICP-R2-E1-CDR": {
            "parameter": "residual cap（残差上限）",
            "default": 0.15,
            "modes": {0.05: "dose_cap_0.05", 0.10: "dose_cap_0.1", 0.15: "true", 0.25: "dose_cap_0.25", 0.40: "dose_cap_0.4"},
        },
        "CICP-R2-E2-CID": {
            "parameter": "increment strength（增量强度）",
            "default": 0.50,
            "modes": {0.0: "dose_strength_0", 0.25: "dose_strength_0.25", 0.50: "true", 0.75: "dose_strength_0.75", 1.0: "dose_strength_1"},
        },
        "CICP-R2-E3-CMA-strength": {
            "method": "CICP-R2-E3-CMA",
            "parameter": "attention strength（注意力强度）",
            "default": 0.50,
            "modes": {0.0: "dose_strength_0", 0.25: "dose_strength_0.25", 0.50: "true", 0.75: "dose_strength_0.75", 1.0: "dose_strength_1"},
        },
        "CICP-R2-E3-CMA-temperature": {
            "method": "CICP-R2-E3-CMA",
            "parameter": "attention temperature（注意力温度）",
            "default": 0.25,
            "modes": {0.125: "dose_temperature_0.125", 0.25: "true", 0.50: "dose_temperature_0.5", 1.0: "dose_temperature_1"},
        },
    }
    for label, spec in specs.items():
        method = str(spec.get("method", label))
        frame = summary[summary["method_label"].eq(method)].set_index("intervention_mode")
        for value, mode in spec["modes"].items():
            row = frame.loc[mode]
            rows.append(
                {
                    "method_label": method,
                    "parameter": spec["parameter"],
                    "value": value,
                    "is_training_default": bool(np.isclose(value, spec["default"])),
                    "ndcg@20": float(row["ndcg@20"]),
                    "relative_pct_ndcg@20_vs_official_baseline": float(
                        row["relative_pct_ndcg@20_vs_official_baseline"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_auxiliary_training_scale(
    baseline_curve: pd.DataFrame,
    cicpr2_result_root: Path,
    initialization: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    init_by_variant = initialization.set_index("method_variant")
    for config_path in cicpr2_result_root.rglob("run_config.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        method = str(config.get("method_variant", ""))
        if method not in {"cicpr2_score_distillation", "cicpr2_ordinal_counterfactual"}:
            continue
        curve = pd.read_csv(config_path.parent / "result.csv")
        for period, mask in (
            ("all100", curve["epoch"].between(1, 100)),
            ("late30", curve["epoch"].between(71, 100)),
        ):
            base = baseline_curve.loc[mask.to_numpy()]
            method_loss = curve.loc[mask, "loss"].to_numpy(dtype=float)
            baseline_loss = base["loss"].to_numpy(dtype=float)
            method_contrast = curve.loc[mask, "contrast_sum"].to_numpy(dtype=float)
            baseline_contrast = base["contrast_sum"].to_numpy(dtype=float)
            rows.append(
                {
                    "method_variant": method,
                    "method_label": (
                        "CICP-R2-E4-SD" if method == "cicpr2_score_distillation" else "CICP-R2-E5-OCS"
                    ),
                    "period": period,
                    "common_initialization_equal_to_baseline": bool(
                        init_by_variant.loc[method, "common_initialization_equal_to_baseline"]
                    ),
                    "method_loss_mean": float(method_loss.mean()),
                    "baseline_loss_mean": float(baseline_loss.mean()),
                    "loss_mean_absolute_difference": float((method_loss - baseline_loss).mean()),
                    "loss_mean_relative_pct_difference": relative_pct(
                        float(method_loss.mean()), float(baseline_loss.mean())
                    ),
                    "contrast_sum_mean_absolute_difference": float(
                        (method_contrast - baseline_contrast).mean()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["method_label", "period"]).reset_index(drop=True)


def build_evidence_matrix(
    intervention: pd.DataFrame,
    initialization: pd.DataFrame,
    cicpr1_score_response: pd.DataFrame,
    cicpr2_score_response: pd.DataFrame,
    deployability: pd.DataFrame,
) -> pd.DataFrame:
    r1_response = cicpr1_score_response[
        cicpr1_score_response["method_variant"].eq("cicpr1_e4_residual")
        & cicpr1_score_response["response"].eq("ndcg@20")
    ].iloc[0]
    r2_response = cicpr2_score_response[
        cicpr2_score_response["method_variant"].eq("cicpr2_cross_modal_attention")
        & cicpr2_score_response["response"].eq("ndcg@20")
    ].iloc[0]
    hgb = deployability[deployability["mapping"].eq("hist_gradient_boosting")].iloc[0]
    distill = intervention[
        intervention["method_label"].eq("CICP-R2-E4-SD")
        & intervention["intervention_mode"].eq("true")
    ].iloc[0]
    r1_init = initialization[initialization["method_label"].eq("CICP-R1-E1")].iloc[0]
    rows = [
        (
            "current_scalar_alignment",
            "主因",
            "STRONG（强）",
            f"冷物品映射折外Spearman仅{float(hgb['oof_spearman']):.3f}；R1/R2最佳分数-收益Spearman分别为{float(r1_response['spearman']):+.4f}/{float(r2_response['spearman']):+.4f}。",
            "CICP排序只弱预测训练类别协同增量，且几乎不排序最终逐物品推荐收益。",
        ),
        (
            "scalar_compression",
            "主因",
            "STRONG（强）",
            "训练输入只保留经验百分位s；原始增量幅度、五折预测离散度、类别级归因和映射不确定性均被丢弃。",
            "一个均匀排序分数不足以决定修正方向、可信度和剂量。",
        ),
        (
            "r2_cardinal_semantics",
            "次主因",
            "STRONG（强）",
            "R2-E1/E2/E3/E6把经验百分位线性解释为残差强度、类别倍率、融合权重或丢弃概率。",
            "R2在机制语义上更具体，但对未校准分数施加了没有数据依据的基数解释。",
        ),
        (
            "r1_flexible_adapter",
            "解释R1优势",
            "STRONG（强）",
            "R1真实分数优于打乱分数，配对区间排除零；其3→16→256映射可学习非线性共享修正轴。",
            "R1较少约束方向和剂量，因此更容易适配代理分数，但不等于语义更正确。",
        ),
        (
            "score_is_representable_not_sufficient",
            "主因佐证",
            "STRONG（强）",
            f"R2-E4生成表示可恢复CICP分数，Spearman={float(distill['distilled_score_spearman']):.3f}，推荐仍仅+0.455%。",
            "瓶颈不是网络完全读不到该分数，而是编码该分数不足以优化Top-20推荐。",
        ),
        (
            "common_initialization_stream_drift",
            "协议混杂",
            "STRONG（存在）/MODERATE（解释幅度）",
            f"R1公共初值哈希与基线不同={not bool(r1_init['common_initialization_equal_to_baseline'])}；R2形成三组不同公共初值流。",
            "单随机种子下最终差异混入公共参数初值与训练轨迹，无法把R1高于R2全部归因于载体。",
        ),
        (
            "gross_default_dose_error",
            "非全局主因",
            "WEAK（弱）",
            "R1、R2-E1、R2-E2、R2-E3默认推理剂量均为扫描局部最好或近最好。",
            "不能用统一的过强/过弱解释六支失败；训练时最优仍未由本次事后扫描证明。",
        ),
        (
            "r2_e6_dropout_damage",
            "分支特定原因",
            "STRONG（强）",
            "R2-E6平均约25%概率清零整条类别向量，最终NDCG为-6.764%。",
            "这是激进且结构性破坏的正则，不代表其他R2机制也同样过强。",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["root_cause", "role", "evidence_strength", "evidence", "interpretation"],
    )


def build_fallacy_scan() -> pd.DataFrame:
    entries = [
        ("Simpson's paradox（辛普森悖论）", "CHECKED", "整体、CICP互补群体和逐物品配对干预同时报告。"),
        ("Ecological fallacy（生态谬误）", "CAUTION", "均值干预效应不代表每个物品都依赖CICP。"),
        ("Berkson's paradox（伯克森悖论）", "CAUTION", "仅分析验证冷启动物品，不能外推到训练或测试范围。"),
        ("Collider bias（碰撞变量偏差）", "CAUTION", "训练后最佳检查点和模型内干预可能受共同训练轨迹影响。"),
        ("Base-rate neglect（基率忽视）", "CHECKED", "明确5,298/35,322覆盖关系。"),
        ("Regression to the mean（均值回归）", "CAUTION", "每支从100轮选择峰值，结合后30轮解释。"),
        ("Multiple comparisons（多重比较）", "CAUTION", "六分支、100轮和多剂量均为开发诊断，不产生新正式胜者。"),
        ("Survivorship bias（幸存者偏差）", "CHECKED", "包括退化的R2-E6和近基线的R2-E5。"),
        ("Garden of forking paths（分叉路径花园）", "CAUTION", "剂量扫描为事后审计，不能回写为预注册性能。"),
        ("Correlation is not causation（相关不等于因果）", "CAUTION", "推理时off/shuffle是模型内干预，但没有替代训练时同初始化控制。"),
        ("Reverse causality（反向因果）", "CHECKED", "训练输入不含验证/测试推荐结果；推荐收益不反向证明信号定义。"),
    ]
    return pd.DataFrame(entries, columns=["fallacy", "severity", "finding"])


def _format_percent(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = result[column].map(lambda value: f"{float(value):+.3f}%")
    return result


def write_reports(
    *,
    output_dir: Path,
    route_output_dir: Path,
    run_stamp: str,
    curve: pd.DataFrame,
    intervention: pd.DataFrame,
    contrasts: pd.DataFrame,
    parameter_scan: pd.DataFrame,
    auxiliary_scale: pd.DataFrame,
    initialization: pd.DataFrame,
    dose_audit: pd.DataFrame,
    mechanism: pd.DataFrame,
    evidence: pd.DataFrame,
    fallacy_scan: pd.DataFrame,
    decision: dict[str, Any],
) -> tuple[Path, Path]:
    curve_table = _format_percent(
        curve,
        [
            "relative_pct_ndcg@20_vs_baseline_best",
            "relative_pct_hr@20_vs_baseline_best",
            "late30_relative_pct_ndcg@20_vs_baseline_best",
            "late30_matched_relative_pct_ndcg@20",
        ],
    )
    standard = intervention[
        intervention["intervention_mode"].isin(["off", "true", "neutral", "shuffle", "invert", "zero", "one"])
    ].copy()
    standard = _format_percent(
        standard,
        ["relative_pct_ndcg@20_vs_official_baseline", "relative_pct_hr@20_vs_official_baseline"],
    )
    contrast_table = contrasts.copy()
    contrast_table["absolute_delta_ndcg@20"] = contrast_table["absolute_delta_ndcg@20"].map(
        lambda value: f"{value:+.6f}"
    )
    contrast_table["ci95"] = contrast_table.apply(
        lambda row: f"[{row['bootstrap_absolute_ci95_low']:+.6f}, {row['bootstrap_absolute_ci95_high']:+.6f}]",
        axis=1,
    )
    scan_table = _format_percent(
        parameter_scan,
        ["relative_pct_ndcg@20_vs_official_baseline"],
    )
    auxiliary_table = auxiliary_scale.copy()
    auxiliary_table["loss_mean_relative_pct_difference"] = auxiliary_table[
        "loss_mean_relative_pct_difference"
    ].map(lambda value: f"{float(value):+.3f}%")
    init_table = initialization[
        [
            "method_label",
            "total_parameter_count",
            "extra_parameter_count_vs_baseline",
            "common_initialization_equal_to_baseline",
            "common_initialization_sha256",
        ]
    ].copy()
    dose_validation = dose_audit[dose_audit["split"].eq("validate")].copy()
    r1 = standard[
        standard["method_label"].eq("CICP-R1-E1")
        & standard["intervention_mode"].isin(["off", "true", "shuffle", "invert", "neutral"])
    ]
    r2_direct = standard[
        standard["method_label"].isin(["CICP-R2-E1-CDR", "CICP-R2-E2-CID", "CICP-R2-E3-CMA"])
    ]
    distill = intervention[
        intervention["method_label"].eq("CICP-R2-E4-SD")
        & intervention["intervention_mode"].eq("true")
    ].iloc[0]
    report = f"""---
title: {run_stamp} CCFCRec Amazon-VG CICP-R1-R2路线级根因诊断
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - CICP
  - root-cause-analysis
---

# CCFCRec Amazon-VG CICP-R1-R2路线级根因诊断

## Material Passport（材料护照）

- verification status（验证状态）：ANALYZED_WITH_INFERENCE_INTERVENTIONS（已分析并完成推理干预）
- formal performance scope（正式性能范围）：全部5,298个 validation cold-start items（验证冷启动物品）
- full profile relation（相对完整档案）：5,298 / 35,322 = 14.999%
- intervention scale（干预规模）：8个模型、54个条件、286,092条逐物品记录
- test item metrics（测试逐物品指标）：未读取、未生成
- train recommendation metrics（训练推荐指标）：未评估
- post-hoc dose scans（事后剂量扫描）：只作根因诊断，不产生新正式结果

## 一、总判断

两轮结果更支持：

> **当前CICP单分数与最终Top-20推荐收益不够对齐、并且单分数表达不够，是主因；R2对这个未校准百分位施加了过强的基数语义和方向约束，是次主因；不同分支公共参数初值不一致是重要协议混杂。**

它们不支持“所有载体都没有用对，所以再换一批载体即可”的单一解释，也不支持“六个默认参数普遍设错”这一统一解释。

## 二、真实结果与训练轨迹

{_markdown_table(curve_table, ['round','method_label','best_epoch','best_ndcg@20','relative_pct_ndcg@20_vs_baseline_best','relative_pct_hr@20_vs_baseline_best','late30_relative_pct_ndcg@20_vs_baseline_best','late30_matched_relative_pct_ndcg@20','late30_matched_positive_ndcg_epochs'])}

`CICP-R1-E1` 的独立峰值最高，但两轮均只有 seed43（第43号单随机种子），而不同结构实际落入四套公共参数初值流，因此不能把最终峰值差全部解释成载体优劣。

## 三、为什么简单R1反而更高

### 3.1 R1不是完全靠额外参数

{_markdown_table(r1, ['intervention_mode','ndcg@20','relative_pct_ndcg@20_vs_official_baseline','relative_pct_hr@20_vs_official_baseline','embedding_relative_drift_vs_off_mean','residual_ratio_mean','residual_cap_saturation_share'])}

- 关闭R1残差后仍有 `+0.684%`，说明训练轨迹或公共初值贡献了部分优势。
- 真实分数为 `+2.051%`，打乱后只剩 `+0.367%`，反转后为 `-0.811%`。
- `true - shuffle` 的绝对NDCG@20差为 `+0.002085`，配对95%区间 `[+0.000852, +0.003359]`，排除零。真实分数分配确实被R1利用。

### 3.2 简单不等于低容量或低自由度

R1只增加4,144个参数，但它把 `[s,1-s,4s(1-s)]` 送入 `3→16→256` 非线性映射，能学习一个不要求内容方向、也不要求线性剂量的共享修正轴。CICP是经验百分位，不是校准概率；R1这种“语义约束较弱、优化自由度较高”的载体反而更适合吸收代理信号中的非线性和混杂相关。

R2-E1/E2/E3/E6分别把同一个百分位线性解释为残差强度、类别贡献倍率、融合权重和可靠性丢弃概率。它们在文字语义上更具体，却对分数提出了更强、未被数据证明的基数假设。

## 四、载体问题还是信号问题

### 4.1 R2三个直接载体的干预

{_markdown_table(r2_direct, ['method_label','intervention_mode','ndcg@20','relative_pct_ndcg@20_vs_official_baseline','embedding_relative_drift_vs_off_mean','embedding_cosine_vs_off_mean'])}

{_markdown_table(contrast_table, ['method_label','contrast','absolute_delta_ndcg@20','ci95','ci_excludes_zero','helped_item_count','harmed_item_count'])}

解释：

1. `CICP-R2-E1-CDR` 的 `true - shuffle` 区间跨零，说明内容方向残差的主要收益更多来自“启用该方向支路”，真实分数分配的额外贡献不稳定。
2. `CICP-R2-E2-CID` 的真实分数显著优于打乱和反转，说明该载体确实使用了CICP排序；但把所有物品设为 `s=1` 的事后诊断达到 `+1.250%`，高于真实分数 `+0.141%`，提示“广泛增强类别”比当前逐物品剂量更有利。
3. `CICP-R2-E3-CMA` 的中性常数分数为 `+1.221%`，反而高于真实分数 `+1.006%`；`true - shuffle` 与 `true - neutral` 区间均跨零。跨模态注意力结构有效，但CICP逐物品门控没有提供稳定附加价值。

因此载体确有差异，但共同瓶颈位于载体之前：CICP单分数没有稳定提供“哪个物品应该改多少、向哪里改、该相信多少”。

### 4.2 信号链条存在两次压缩

原始标签是“真实类别相对打乱类别预测训练协同表示的余弦增量”，不是NDCG@20（前20归一化折损累计增益）。随后：

1. 冷物品映射对训练残差标签的折外Spearman（斯皮尔曼秩相关）只有 `0.253`；
2. 映射输出再转成经验百分位，只保留排序；
3. 原始增量幅度、映射不确定性、类别级贡献和绝对可靠性被丢弃；
4. R1与R2最佳分支中，CICP分数和逐物品推荐收益的Spearman分别只有 `-0.0102` 和 `+0.0045`。

`CICP-R2-E4-SD`（分数蒸馏）进一步证明“网络读不到分数”不是主因：训练后生成表示预测CICP分数的Spearman为 `{float(distill['distilled_score_spearman']):.3f}`，但推荐提升仍只有 `+0.455%`。模型能够编码分数，不代表该分数足以指导Top-20排序。

## 五、参数是否设得过强或过弱

{_markdown_table(scan_table, ['method_label','parameter','value','is_training_default','ndcg@20','relative_pct_ndcg@20_vs_official_baseline'])}

结论分三类：

1. R1残差上限 `0.15`、R2-E1残差上限 `0.15`、R2-E2增量强度 `0.50`、R2-E3注意力强度 `0.50` 与温度 `0.25` 均是训练后扫描中的局部最好或近最好。不存在一个明显统一的“全部过强/全部过弱”错误。
2. 这只是固定检查点上的 inference-time sensitivity（推理时敏感性），不能证明训练时超参数已经全局最优；E4蒸馏权重、E5序数权重与间隔仍需要重新训练才能严格判定。
3. R2-E6是明确例外：默认设置使训练物品平均约25%概率整条类别表示被清零，属于激进且结构性破坏的正则，足以解释该分支 `-6.764%`，但不能解释其余五支。

辅助目标的训练尺度：

{_markdown_table(auxiliary_table, list(auxiliary_table.columns))}

`CICP-R2-E5-OCS` 与基线共享公共初值，其后30轮总损失高 `+3.132%`，而 contrast sum（对比损失和）平均只差 `+0.383`、最终NDCG近乎基线。这说明序数辅助目标确实产生了非零优化压力，但压力没有转化到主推荐目标；不能只用“权重太小”解释。`CICP-R2-E4-SD` 公共初值不同，不能根据总损失差直接反推辅助权重，但其分数恢复Spearman `0.583` 已证明蒸馏目标被学到。

验证分数与默认剂量关系：

{_markdown_table(dose_validation, ['quantity','item_count','mean','std','min','p10','p50','p90','max'])}

## 六、初始化与公平比较问题

{_markdown_table(init_table, list(init_table.columns))}

同一个seed43并没有产生同一公共参数初值。原因是变体专用线性层在公共参数手动初始化前已经构造并消耗随机数：

- baseline、R2-E2、R2-E5、R2-E6共享一组公共初值；
- R2-E1与R2-E3共享第二组；
- R1-E1独占第三组；
- R2-E4独占第四组。

这不使已有结果“无效”，但使“R1比R2高多少完全由机制造成”无法成立。R1关闭机制后仍有 `+0.684%`，R2-E1关闭后为 `-1.017%`，R2-E3关闭后为 `-0.283%`，已经说明各自训练底座不同。

## 七、跨模态注意力实际行为

{_markdown_table(mechanism, list(mechanism.columns))}

R2-E3内容注意力最大权重均值为 `0.737`，默认门控均值只有 `0.248`；注意力本身较集中，但只以中等权重混入。强度增至 `0.75/1.0` 后性能下降，说明它不是简单“太保守”。中性常数优于真实分数，进一步把问题指向逐物品门控，而不是注意力容量不足。

## 八、根因证据矩阵

{_markdown_table(evidence, list(evidence.columns))}

## 九、未被原问题显式指出的其他问题

1. percentile-as-probability error（把百分位当概率）：R2的可靠性和剂量语义缺乏校准依据。
2. proxy-target mismatch（代理目标错配）：训练协同表示余弦增量与最终用户Top-20排序不是同一目标。
3. uncertainty deletion（不确定性删除）：验证冷物品原本有五折预测标准差，但正式画像只保留一个分数。
4. category attribution loss（类别归因丢失）：多类别物品被压成一个总分，无法告诉注意力或残差应修改哪个类别。
5. initialization confounding（初始化混杂）：结构分支改变公共随机流，单seed比较不再是严格配对。
6. checkpoint multiplicity（检查点多重选择）：六支各选100轮峰值，R1的 `+2.051%` 仍是开发上界而非确认效应。
7. intervention limitation（干预局限）：off/shuffle是在训练后模型上执行，能证明依赖和敏感性，不能替代从训练开始的同初始化控制。

## 十、路线决策

- primary root cause（主根因）：`{decision['primary_root_cause']}`
- secondary root cause（次根因）：`{decision['secondary_root_cause']}`
- protocol confound（协议混杂）：`{decision['protocol_confound']}`
- gross hyperparameter error（统一超参数错误）：`{decision['gross_hyperparameter_error_supported']}`
- run CICP-R3 local variants（运行第三轮局部变体）：`{decision['run_cicpr3_local_variants']}`
- route（路线）：`{decision['route']}`

下一步不应再做一轮单分数门控/残差参数扫描。应先完成两项基础修复：

1. 将CICP从单一经验百分位改为 multi-component category increment profile（多分量类别增量画像），至少保留原始预测增量、百分位、五折不确定性和类别级/类别组级贡献；训练输入仍不得使用验证或测试推荐答案。
2. 建立 common-initialization harness（公共初始化配对框架）：先冻结一份公共CCFCRec参数，再加载到所有载体，只单独初始化新增模块；下一次机制比较才是严格的同seed配对。

完成这两项离线审计后，再决定是用修复后的多分量信号做一次100 epoch（100训练轮次）正式验证，还是直接升级 category semantic basis（类别语义基础）/cold-start backbone（冷启动模型骨干）。

## 十一、统计谬误扫描

覆盖：`11/11`。

{_markdown_table(fallacy_scan, list(fallacy_scan.columns))}
"""
    route_report = f"""---
title: {run_stamp} CCFCRec Amazon-VG CICP-R1-R2根因诊断路线判断
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - CICP
  - route-decision
---

# CCFCRec Amazon-VG CICP-R1-R2根因诊断路线判断

关联诊断：[[{run_stamp} CCFCRec Amazon-VG CICP-R1-R2路线级根因诊断]]

> [!important] 主决策
> R1高于R2不能解释为“简单方法天然更好”，也不能统一归因于R2参数设置错误。当前更强证据指向：单一经验百分位CICP与最终Top-20推荐收益不够对齐且表达不足；R2又把该百分位当作线性强度/可靠性使用。另有公共初始化随机流不一致的协议混杂，必须在下一次正式训练前修复。

## 决策顺序

1. 不运行CICP-R3（第三轮）单分数局部变体，也不做大规模强度网格训练。
2. 先重建多分量类别增量画像，保留原始幅度、排序、不确定性和类别归因。
3. 同时实现公共初始化配对框架，消除结构分支改变公共参数初值的问题。
4. 两项离线门槛通过后，只允许一次新的100轮正式机制比较；仍无充分收益则升级类别语义基础或冷启动模型骨干。

## 证据边界

- 正式性能范围：5,298个验证冷启动物品，占完整档案14.999%。
- 当前历史开发最佳仍为CICP-R1-E1的 `+2.051%`，但不是确认性论文效应。
- 剂量扫描和off/shuffle干预均为训练后诊断，不回写为新正式实验结果。
- 测试逐物品指标未读取、未生成。
- multi-seed（多随机种子）仍不开放。

## 固化状态

- route（路线）：`{decision['route']}`
- repair scalar profile first（先修复单分数画像）：`True`
- repair common initialization first（先修复公共初始化）：`True`
- run CICP-R3 local variants now（现在运行第三轮局部变体）：`False`
- run multi-seed now（现在运行多随机种子）：`False`
- fallacy scan coverage（统计谬误扫描覆盖）：`11/11`
"""
    result_path = output_dir / f"{run_stamp} CCFCRec Amazon-VG CICP-R1-R2路线级根因诊断.md"
    route_output_dir.mkdir(parents=True, exist_ok=True)
    route_path = route_output_dir / f"{run_stamp} CCFCRec Amazon-VG CICP-R1-R2根因诊断路线判断.md"
    result_path.write_text(report, encoding="utf-8")
    route_path.write_text(route_report, encoding="utf-8")
    return result_path, route_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    diagnostic_dir = args.diagnostic_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    item_metrics = pd.read_csv(
        diagnostic_dir / "cicp_root_intervention_item_metrics.csv", dtype={"raw_asin": str}
    )
    intervention = pd.read_csv(diagnostic_dir / "cicp_root_intervention_summary.csv")
    initialization = pd.read_csv(diagnostic_dir / "cicp_root_initialization_audit.csv")
    dose_audit = pd.read_csv(diagnostic_dir / "cicp_root_signal_dose_audit.csv")
    mechanism = pd.read_csv(diagnostic_dir / "cicp_root_mechanism_statistics.csv")
    diagnostic_audit = json.loads(
        (diagnostic_dir / "cicp_root_diagnostic_audit.json").read_text(encoding="utf-8")
    )
    if diagnostic_audit["test_item_metrics_read_or_generated"]:
        raise ValueError("root-cause diagnostic unexpectedly used test item metrics")
    if len(item_metrics) != int(diagnostic_audit["item_metric_row_count"]):
        raise ValueError("intervention item row count does not match diagnostic audit")
    cicpr1_curve = pd.read_csv(args.cicpr1_analysis_dir / "cicpr1_curve_summary.csv")
    cicpr2_curve = pd.read_csv(args.cicpr2_analysis_dir / "cicpr2_curve_summary.csv")
    cicpr1_response = pd.read_csv(args.cicpr1_analysis_dir / "cicpr1_score_response_summary.csv")
    cicpr2_response = pd.read_csv(args.cicpr2_analysis_dir / "cicpr2_score_response_summary.csv")
    deployability = pd.read_csv(args.cicp_signal_dir / "cicp_deployability_summary.csv")
    curve = build_curve_process_summary(cicpr1_curve, cicpr2_curve)
    contrasts = build_contrasts(item_metrics, args.bootstrap_repetitions)
    parameter_scan = build_parameter_scan(intervention)
    cicpr2_manifest = json.loads(
        (args.cicpr2_analysis_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    baseline_curve = pd.read_csv(args.cicpr2_analysis_dir / "cicpr2_baseline_curve.csv")
    auxiliary_scale = build_auxiliary_training_scale(
        baseline_curve,
        Path(cicpr2_manifest["result_root"]),
        initialization,
    )
    evidence = build_evidence_matrix(
        intervention,
        initialization,
        cicpr1_response,
        cicpr2_response,
        deployability,
    )
    fallacy_scan = build_fallacy_scan()
    decision = {
        "primary_root_cause": "cicp_scalar_expression_and_top20_utility_alignment_insufficient",
        "secondary_root_cause": "r2_imposes_uncalibrated_cardinal_semantics_and_direction_constraints",
        "protocol_confound": "variant_modules_shift_common_parameter_initialization_stream",
        "gross_hyperparameter_error_supported": False,
        "branch_specific_hyperparameter_failure": "CICP-R2-E6-RCD whole-category dropout",
        "unresolved_training_hyperparameters": [
            "CICP-R2-E4-SD distillation weight",
            "CICP-R2-E5-OCS ordinal weight and margin",
        ],
        "run_cicpr3_local_variants": False,
        "run_multi_seed_now": False,
        "route": "repair_multicomponent_signal_and_common_initialization_before_new_training",
        "formal_metric_scope": "validation_cold_5298_only",
        "validation_coverage_pct_of_full_profile": VALIDATION_ITEM_COUNT / FULL_ITEM_COUNT * 100.0,
        "test_item_metrics_read_or_generated": False,
        "evidence_level": "B_development_validation_single_seed_with_posthoc_interventions",
        "fallacy_scan_coverage": "11/11",
    }
    artifacts = {
        "cicp_root_curve_process_summary.csv": curve,
        "cicp_root_intervention_contrast_bootstrap.csv": contrasts,
        "cicp_root_parameter_scan_summary.csv": parameter_scan,
        "cicp_root_auxiliary_training_scale.csv": auxiliary_scale,
        "cicp_root_cause_evidence_matrix.csv": evidence,
        "cicp_root_cause_fallacy_scan.csv": fallacy_scan,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output_dir / name, index=False)
    (output_dir / "cicp_root_cause_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result_path, route_path = write_reports(
        output_dir=output_dir,
        route_output_dir=args.route_output_dir.resolve(),
        run_stamp=args.run_stamp,
        curve=curve,
        intervention=intervention,
        contrasts=contrasts,
        parameter_scan=parameter_scan,
        auxiliary_scale=auxiliary_scale,
        initialization=initialization,
        dose_audit=dose_audit,
        mechanism=mechanism,
        evidence=evidence,
        fallacy_scan=fallacy_scan,
        decision=decision,
    )
    manifest = {
        "protocol": "cicp_r1_r2_route_root_cause_analysis_v1",
        "run_stamp": args.run_stamp,
        "diagnostic_dir": str(diagnostic_dir),
        "cicpr1_analysis_dir": str(args.cicpr1_analysis_dir.resolve()),
        "cicpr2_analysis_dir": str(args.cicpr2_analysis_dir.resolve()),
        "cicp_signal_dir": str(args.cicp_signal_dir.resolve()),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "formal_metric_scope": "validation_cold_5298_only",
        "test_item_metrics_read_or_generated": False,
        "outputs": [*artifacts, "cicp_root_cause_decision.json", str(result_path), str(route_path)],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--cicpr1-analysis-dir", type=Path, required=True)
    parser.add_argument("--cicpr2-analysis-dir", type=Path, required=True)
    parser.add_argument("--cicp-signal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-output-dir", type=Path, required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
