"""Unit tests for analysis.summary_table.aggregate_seeds (multi-seed averaging)."""
from __future__ import annotations

import math

import pandas as pd

from analysis.summary_table import aggregate_seeds


def _row(eps: str, contains: float, match: float) -> dict:
    return {
        "Method": "dpksa",
        "Dataset": "triviaqa",
        "Variant": "full",
        "Model": "Qwen2.5-3B",
        "ε": eps,
        "ε_kind": "input_total",
        "ContainsAccuracy": contains,
        "MatchAccuracy": match,
        "ROUGE-L": 0.0,
    }


def test_aggregates_three_seeds_to_mean_std_and_count():
    df = pd.DataFrame([
        _row("1.000", 0.2, 0.0),
        _row("1.000", 0.4, 0.1),
        _row("1.000", 0.6, 0.2),
    ])
    out = aggregate_seeds(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert int(row["N_seeds"]) == 3
    assert math.isclose(row["ContainsAccuracy"], 0.4, abs_tol=1e-9)
    # sample std (ddof=1) of [0.2,0.4,0.6] = 0.2
    assert math.isclose(row["ContainsAccuracy_std"], 0.2, abs_tol=1e-9)
    assert math.isclose(row["MatchAccuracy"], 0.1, abs_tol=1e-9)


def test_distinct_epsilons_stay_separate():
    df = pd.DataFrame([
        _row("1.000", 0.2, 0.0),
        _row("1.000", 0.4, 0.0),
        _row("2.000", 0.9, 0.0),
    ])
    out = aggregate_seeds(df).set_index("ε")
    assert int(out.loc["1.000", "N_seeds"]) == 2
    assert int(out.loc["2.000", "N_seeds"]) == 1
    # single-run group -> std is NaN
    assert math.isnan(out.loc["2.000", "ContainsAccuracy_std"])


def test_empty_df_passthrough():
    assert aggregate_seeds(pd.DataFrame()).empty
