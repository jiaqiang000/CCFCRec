#!/usr/bin/env python3
"""
Task4-pre Acat v3 method-carrier control audit for Amazon-VG.

This is an offline audit. It does not train a new model. It checks whether
Acat_v3 can act as a method carrier better than shuffle and R/S/P-only controls.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DESIGN_NOTE = (
    "上游设计：[[2026-07-05 114102 CCFCRec Amazon-VG Task4-pre Acat v3 方法载体对照审计设计]]"
)

CONTROL_COLUMNS = [
    "category_count",
    "R_metadata_richness_score",
    "S_train_support_score",
    "P_popularity_score",
]

CARRIER_SPECS = {
    "acat_v3": "acat_v3_score",
    "acat_v3_shuffle": "acat_v3_shuffle_score",
    "rsp_only": "rsp_only_score",
    "recoverability_proxy": "recoverability_proxy_score",
    "v2_s_cat": "v2_s_cat_score",
}


@dataclass(frozen=True)
class CarrierAuditOutputs:
    profile_csv: Path
    summary_csv: Path
    correlation_csv: Path
    placebo_csv: Path
    route_json: Path
    result_md: Path
    manifest_json: Path


def now_stamp() -> tuple[str, str]:
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_mean(series: pd.Series) -> float:
    values = _numeric(series).dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _safe_corr(left: pd.Series, right: pd.Series, method: str = "spearman") -> float:
    valid = pd.concat([_numeric(left), _numeric(right)], axis=1).dropna()
    if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} 缺少字段: {missing}")


def percentile_score(series: pd.Series) -> pd.Series:
    values = _numeric(series)
    if values.dropna().nunique() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    return values.rank(method="average", pct=True).fillna(0.5).astype(float)


def bucket_score(score: pd.Series) -> pd.Series:
    values = _numeric(score)
    if values.dropna().nunique() <= 1:
        return pd.Series("mid", index=score.index)
    low = values.quantile(1 / 3)
    high = values.quantile(2 / 3)
    return pd.Series(
        np.select(
            [values <= low, values >= high],
            ["low", "high"],
            default="mid",
        ),
        index=score.index,
    )


def _merge_on_asin(left: pd.DataFrame, right: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = right.copy()
    if "raw_asin" not in frame.columns and "asin" in frame.columns:
        frame = frame.rename(columns={"asin": "raw_asin"})
    if "raw_asin" not in frame.columns:
        raise ValueError(f"{label} 缺少 asin/raw_asin 字段")
    return left.merge(frame.drop_duplicates("raw_asin"), on="raw_asin", how="inner")


def build_carrier_profile(
    v3: pd.DataFrame,
    delta_profile: pd.DataFrame,
    recoverability_profile: pd.DataFrame,
    random_seed: int = 43,
) -> pd.DataFrame:
    _require_columns(v3, ["raw_asin", "s_cat_v3", *CONTROL_COLUMNS], "Acat_v3")
    _require_columns(delta_profile.rename(columns={"asin": "raw_asin"}), ["raw_asin", "delta_ndcg@20"], "delta profile")

    keep_v3 = [
        "raw_asin",
        "s_cat_v3",
        "s_cat_v3_group",
        "s_cat_v2",
        *CONTROL_COLUMNS,
    ]
    profile = v3[[column for column in keep_v3 if column in v3.columns]].copy()
    profile = _merge_on_asin(profile, delta_profile, "delta profile")

    recoverability = recoverability_profile.copy()
    if "raw_asin" not in recoverability.columns and "asin" in recoverability.columns:
        recoverability = recoverability.rename(columns={"asin": "raw_asin"})
    keep_recoverability = [
        "raw_asin",
        "proxy_ensemble_score",
        "margin_proxy",
        "best_target_rank",
        "ndcg@20",
    ]
    profile = profile.merge(
        recoverability[[column for column in keep_recoverability if column in recoverability.columns]].drop_duplicates("raw_asin"),
        on="raw_asin",
        how="left",
        suffixes=("", "_recoverability"),
    )

    profile["acat_v3_score"] = _numeric(profile["s_cat_v3"])
    rng = np.random.default_rng(random_seed)
    profile["acat_v3_shuffle_score"] = rng.permutation(profile["acat_v3_score"].to_numpy(dtype=float))

    rsp_percentiles = [percentile_score(profile[column]) for column in CONTROL_COLUMNS if column in profile.columns]
    profile["rsp_only_score"] = pd.concat(rsp_percentiles, axis=1).mean(axis=1) if rsp_percentiles else 0.5

    if "proxy_ensemble_score" in profile.columns:
        profile["recoverability_proxy_score"] = percentile_score(profile["proxy_ensemble_score"])
    elif "margin_proxy" in profile.columns:
        profile["recoverability_proxy_score"] = percentile_score(-_numeric(profile["margin_proxy"]))
    else:
        profile["recoverability_proxy_score"] = 0.5

    if "s_cat_v2" in profile.columns:
        profile["v2_s_cat_score"] = percentile_score(profile["s_cat_v2"])
    elif "s_cat" in profile.columns:
        profile["v2_s_cat_score"] = percentile_score(profile["s_cat"])
    else:
        profile["v2_s_cat_score"] = 0.5

    for carrier, score_col in CARRIER_SPECS.items():
        profile[f"{carrier}_bucket"] = bucket_score(profile[score_col])

    return profile


def _carrier_row(profile: pd.DataFrame, carrier: str, score_col: str) -> dict[str, Any]:
    bucket_col = f"{carrier}_bucket"
    low = profile[profile[bucket_col].eq("low")]
    high = profile[profile[bucket_col].eq("high")]
    row = {
        "carrier": carrier,
        "score_col": score_col,
        "item_count": int(len(profile)),
        "low_count": int(len(low)),
        "high_count": int(len(high)),
        "low_baseline_ndcg@20_mean": _safe_mean(low["baseline_ndcg@20"]) if "baseline_ndcg@20" in low.columns else float("nan"),
        "high_baseline_ndcg@20_mean": _safe_mean(high["baseline_ndcg@20"]) if "baseline_ndcg@20" in high.columns else float("nan"),
        "low_delta_ndcg@20_mean": _safe_mean(low["delta_ndcg@20"]),
        "high_delta_ndcg@20_mean": _safe_mean(high["delta_ndcg@20"]),
        "low_delta_margin_mean": _safe_mean(low["delta_margin_to_top20_cutoff"]) if "delta_margin_to_top20_cutoff" in low.columns else float("nan"),
        "high_delta_margin_mean": _safe_mean(high["delta_margin_to_top20_cutoff"]) if "delta_margin_to_top20_cutoff" in high.columns else float("nan"),
        "low_delta_q_norm_mean": _safe_mean(low["delta_q_norm"]) if "delta_q_norm" in low.columns else float("nan"),
        "high_delta_q_norm_mean": _safe_mean(high["delta_q_norm"]) if "delta_q_norm" in high.columns else float("nan"),
    }
    row["high_minus_low_baseline_ndcg@20"] = row["high_baseline_ndcg@20_mean"] - row["low_baseline_ndcg@20_mean"]
    row["high_minus_low_delta_ndcg@20"] = row["high_delta_ndcg@20_mean"] - row["low_delta_ndcg@20_mean"]
    row["high_minus_low_delta_margin"] = row["high_delta_margin_mean"] - row["low_delta_margin_mean"]
    row["high_minus_low_delta_q_norm"] = row["high_delta_q_norm_mean"] - row["low_delta_q_norm_mean"]
    return row


def build_carrier_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows = [_carrier_row(profile, carrier, score_col) for carrier, score_col in CARRIER_SPECS.items() if score_col in profile.columns]
    return pd.DataFrame(rows)


def build_carrier_correlation(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for carrier, score_col in CARRIER_SPECS.items():
        if score_col not in profile.columns:
            continue
        rows.append(
            {
                "carrier": carrier,
                "score_col": score_col,
                "spearman_vs_baseline_ndcg@20": _safe_corr(profile[score_col], profile["baseline_ndcg@20"]) if "baseline_ndcg@20" in profile.columns else float("nan"),
                "spearman_vs_delta_ndcg@20": _safe_corr(profile[score_col], profile["delta_ndcg@20"]),
                "spearman_vs_delta_margin": _safe_corr(profile[score_col], profile["delta_margin_to_top20_cutoff"]) if "delta_margin_to_top20_cutoff" in profile.columns else float("nan"),
                "spearman_vs_delta_q_norm": _safe_corr(profile[score_col], profile["delta_q_norm"]) if "delta_q_norm" in profile.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_placebo_summary(profile: pd.DataFrame, shuffle_count: int = 100, random_seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    gaps = []
    values = profile["acat_v3_score"].to_numpy(dtype=float)
    for _ in range(shuffle_count):
        shuffled = pd.Series(rng.permutation(values), index=profile.index)
        buckets = bucket_score(shuffled)
        low = profile[buckets.eq("low")]
        high = profile[buckets.eq("high")]
        gaps.append(_safe_mean(high["delta_ndcg@20"]) - _safe_mean(low["delta_ndcg@20"]))
    arr = np.asarray(gaps, dtype=float)
    return pd.DataFrame(
        [
            {
                "shuffle_count": int(shuffle_count),
                "shuffle_delta_ndcg@20_mean": float(np.nanmean(arr)) if arr.size else float("nan"),
                "shuffle_delta_ndcg@20_p05": float(np.nanpercentile(arr, 5)) if arr.size else float("nan"),
                "shuffle_delta_ndcg@20_p95": float(np.nanpercentile(arr, 95)) if arr.size else float("nan"),
                "shuffle_delta_ndcg@20_max": float(np.nanmax(arr)) if arr.size else float("nan"),
            }
        ]
    )


def _summary_value(summary: pd.DataFrame, carrier: str, column: str) -> float:
    row = summary[summary["carrier"].eq(carrier)]
    if row.empty or column not in row.columns:
        return float("nan")
    return float(row.iloc[0][column])


def decide_carrier_route(summary: pd.DataFrame, placebo: pd.DataFrame) -> dict[str, Any]:
    acat_gap = _summary_value(summary, "acat_v3", "high_minus_low_delta_ndcg@20")
    acat_margin_gap = _summary_value(summary, "acat_v3", "high_minus_low_delta_margin")
    shuffle_gap = _summary_value(summary, "acat_v3_shuffle", "high_minus_low_delta_ndcg@20")
    rsp_gap = _summary_value(summary, "rsp_only", "high_minus_low_delta_ndcg@20")
    recoverability_gap = _summary_value(summary, "recoverability_proxy", "high_minus_low_delta_ndcg@20")
    placebo_p95 = float(placebo.iloc[0]["shuffle_delta_ndcg@20_p95"]) if not placebo.empty else float("nan")

    beats_controls = (
        np.isfinite(acat_gap)
        and acat_gap > 0.01
        and (not np.isfinite(placebo_p95) or acat_gap > placebo_p95)
        and (not np.isfinite(rsp_gap) or acat_gap > rsp_gap + 0.005)
        and (not np.isfinite(shuffle_gap) or acat_gap > shuffle_gap + 0.005)
    )
    recoverability_upper_bound = recoverability_gap - acat_gap if np.isfinite(recoverability_gap) and np.isfinite(acat_gap) else float("nan")

    if beats_controls:
        route = "acat_v3_carrier_ready"
        reason = "Acat_v3 carrier 的旧方法响应优于 shuffle 与 RSP-only，可进入正式方法设计。"
    elif np.isfinite(acat_gap) and acat_gap <= 0 and np.isfinite(recoverability_gap) and recoverability_gap > 0.01:
        route = "acat_v3_needs_new_method_not_category_conf"
        reason = "高 Acat_v3 没被旧 category_conf_input 改善，但 recoverability proxy 显示存在可恢复上界；不能复用旧方法。"
    elif np.isfinite(rsp_gap) and np.isfinite(acat_gap) and rsp_gap >= acat_gap:
        route = "category_conf_response_rsp_only_stronger"
        reason = "旧 category_conf_input 的响应更像 R/S/P-only，而不是 Acat_v3 独立 carrier。"
    elif np.isfinite(recoverability_gap) and recoverability_gap > 0.01:
        route = "recoverability_only_upper_bound"
        reason = "主要只有 recoverability proxy 有响应，availability carrier 暂不成立。"
    else:
        route = "acat_v3_carrier_not_supported"
        reason = "Acat_v3、shuffle、RSP-only 和 recoverability proxy 都没有形成清晰方法载体证据。"

    return {
        "route": route,
        "reason": reason,
        "evidence": {
            "acat_high_minus_low_delta_ndcg@20": _jsonable(acat_gap),
            "acat_high_minus_low_delta_margin": _jsonable(acat_margin_gap),
            "shuffle_high_minus_low_delta_ndcg@20": _jsonable(shuffle_gap),
            "rsp_high_minus_low_delta_ndcg@20": _jsonable(rsp_gap),
            "recoverability_high_minus_low_delta_ndcg@20": _jsonable(recoverability_gap),
            "recoverability_upper_bound_gap": _jsonable(recoverability_upper_bound),
            "placebo_shuffle_p95": _jsonable(placebo_p95),
        },
    }


def _md_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_无数据_"
    selected = frame.head(max_rows)
    headers = list(selected.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in selected.iterrows():
        cells = []
        for header in headers:
            value = row[header]
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_outputs(
    profile: pd.DataFrame,
    summary: pd.DataFrame,
    correlation: pd.DataFrame,
    placebo: pd.DataFrame,
    decision: dict[str, Any],
    output_dir: Path,
    input_files: dict[str, str],
) -> CarrierAuditOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp, run_iso = now_stamp()
    profile_csv = output_dir / "acat_v3_method_carrier_profile.csv"
    summary_csv = output_dir / "acat_v3_method_carrier_summary.csv"
    correlation_csv = output_dir / "acat_v3_method_carrier_correlation.csv"
    placebo_csv = output_dir / "acat_v3_method_carrier_placebo.csv"
    route_json = output_dir / "acat_v3_method_carrier_route_decision.json"
    result_md = output_dir / f"{stamp} CCFCRec Amazon-VG Task4-pre Acat v3 方法载体对照审计结果.md"
    manifest_json = output_dir / "run_manifest.json"

    profile.to_csv(profile_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    correlation.to_csv(correlation_csv, index=False)
    placebo.to_csv(placebo_csv, index=False)
    route_json.write_text(json.dumps(_jsonable(decision), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    markdown = f"""---
title: {stamp} CCFCRec Amazon-VG Task4-pre Acat v3 方法载体对照审计结果
date: {stamp[:10]}
time: "{stamp[11:13]}:{stamp[13:15]}:{stamp[15:17]}"
created_at: "{stamp[:10]} {stamp[11:13]}:{stamp[13:15]}:{stamp[15:17]}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - Acat_v3
  - Task4-pre
---

# {stamp} CCFCRec Amazon-VG Task4-pre Acat v3 方法载体对照审计结果

## 来源说明

> [!info] 来源说明
> {DESIGN_NOTE}
> 本结果目录：`{output_dir}/`

## 结论

route decision 为 `{decision.get("route")}`。

{decision.get("reason", "")}

## 字段解释

- `carrier`：方法载体依据，例如 Acat_v3、shuffle、RSP-only。
- `delta_ndcg@20`：旧 category_conf_input 相对 baseline 的 NDCG@20 变化。
- `delta_margin`：旧 category_conf_input 相对 baseline 的 target-vs-top20 margin 变化。
- `high_minus_low`：carrier 高分组均值减低分组均值。

## Route Evidence

```json
{json.dumps(_jsonable(decision.get("evidence", {})), ensure_ascii=False, indent=2)}
```

## Carrier Summary

{_md_table(summary)}

## Carrier Correlation

{_md_table(correlation)}

## Placebo

{_md_table(placebo)}
"""
    result_md.write_text(markdown, encoding="utf-8")

    manifest = {
        "run_time": run_iso,
        "script": str(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "input_files": input_files,
        "outputs": {
            "profile_csv": str(profile_csv),
            "summary_csv": str(summary_csv),
            "correlation_csv": str(correlation_csv),
            "placebo_csv": str(placebo_csv),
            "route_json": str(route_json),
            "result_md": str(result_md),
        },
        "row_counts": {
            "profile": int(len(profile)),
            "summary": int(len(summary)),
            "correlation": int(len(correlation)),
            "placebo": int(len(placebo)),
        },
        "route_decision": decision,
        "design_note": DESIGN_NOTE,
    }
    manifest_json.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return CarrierAuditOutputs(
        profile_csv=profile_csv,
        summary_csv=summary_csv,
        correlation_csv=correlation_csv,
        placebo_csv=placebo_csv,
        route_json=route_json,
        result_md=result_md,
        manifest_json=manifest_json,
    )


def run_audit(
    acat_v3_path: Path,
    delta_profile_path: Path,
    recoverability_profile_path: Path,
    output_dir: Path,
    shuffle_count: int,
    random_seed: int,
) -> CarrierAuditOutputs:
    v3 = pd.read_csv(acat_v3_path)
    delta = pd.read_csv(delta_profile_path)
    recoverability = pd.read_csv(recoverability_profile_path)
    profile = build_carrier_profile(v3, delta, recoverability, random_seed=random_seed)
    summary = build_carrier_summary(profile)
    correlation = build_carrier_correlation(profile)
    placebo = build_placebo_summary(profile, shuffle_count=shuffle_count, random_seed=random_seed)
    decision = decide_carrier_route(summary, placebo)
    return write_outputs(
        profile=profile,
        summary=summary,
        correlation=correlation,
        placebo=placebo,
        decision=decision,
        output_dir=output_dir,
        input_files={
            "acat_v3": str(acat_v3_path),
            "delta_profile": str(delta_profile_path),
            "recoverability_profile": str(recoverability_profile_path),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acat-v3", type=Path, required=True)
    parser.add_argument("--delta-profile", type=Path, required=True)
    parser.add_argument("--recoverability-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shuffle-count", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=43)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_audit(
        acat_v3_path=args.acat_v3,
        delta_profile_path=args.delta_profile,
        recoverability_profile_path=args.recoverability_profile,
        output_dir=args.output_dir,
        shuffle_count=args.shuffle_count,
        random_seed=args.random_seed,
    )
    print(f"wrote {outputs.profile_csv}")
    print(f"wrote {outputs.summary_csv}")
    print(f"wrote {outputs.correlation_csv}")
    print(f"wrote {outputs.placebo_csv}")
    print(f"wrote {outputs.route_json}")
    print(f"wrote {outputs.result_md}")
    print(f"wrote {outputs.manifest_json}")


if __name__ == "__main__":
    main()
