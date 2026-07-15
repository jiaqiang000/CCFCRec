from __future__ import annotations

import pandas as pd
import numpy as np

from analyze_amazon_vg_cicp_r1_r2_root_causes import (
    build_contrasts,
    build_fallacy_scan,
    build_parameter_scan,
)


def test_contrasts_are_paired_by_item() -> None:
    rows = []
    modes = {
        "off": [0.1, 0.2],
        "true": [0.2, 0.4],
        "shuffle": [0.1, 0.2],
        "neutral": [0.1, 0.2],
        "invert": [0.05, 0.1],
    }
    for mode, values in modes.items():
        for raw_asin, value in zip(["a", "b"], values):
            rows.append(
                {
                    "raw_asin": raw_asin,
                    "method_label": "CICP-R1-E1",
                    "intervention_mode": mode,
                    "ndcg@20": value,
                }
            )
    result = build_contrasts(pd.DataFrame(rows), repetitions=100)
    true_off = result[result["contrast"].eq("true_minus_off")].iloc[0]
    assert np.isclose(true_off["absolute_delta_ndcg@20"], 0.15)
    assert true_off["helped_item_count"] == 2


def test_parameter_scan_marks_training_defaults() -> None:
    rows = []
    specs = {
        "CICP-R1-E1": ["dose_cap_0.05", "dose_cap_0.1", "true", "dose_cap_0.25", "dose_cap_0.4"],
        "CICP-R2-E1-CDR": ["dose_cap_0.05", "dose_cap_0.1", "true", "dose_cap_0.25", "dose_cap_0.4"],
        "CICP-R2-E2-CID": ["dose_strength_0", "dose_strength_0.25", "true", "dose_strength_0.75", "dose_strength_1"],
        "CICP-R2-E3-CMA": [
            "dose_strength_0",
            "dose_strength_0.25",
            "true",
            "dose_strength_0.75",
            "dose_strength_1",
            "dose_temperature_0.125",
            "dose_temperature_0.5",
            "dose_temperature_1",
        ],
    }
    for method, modes in specs.items():
        for index, mode in enumerate(modes):
            rows.append(
                {
                    "method_label": method,
                    "intervention_mode": mode,
                    "ndcg@20": 0.1 + index * 0.001,
                    "relative_pct_ndcg@20_vs_official_baseline": float(index),
                }
            )
    scan = build_parameter_scan(pd.DataFrame(rows))
    defaults = scan[scan["is_training_default"]]
    assert len(defaults) == 5
    assert defaults["value"].tolist() == [0.15, 0.15, 0.5, 0.5, 0.25]


def test_fallacy_scan_has_eleven_entries() -> None:
    assert len(build_fallacy_scan()) == 11
