"""YAML configuration with defaults. Relative paths resolve against the config file."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "data": {"path": "data/raw/taiwan_default.csv", "id_col": "ID", "target_col": "DEFAULT"},
    "split": {"dev_fraction": 0.5, "mode": "random"},  # or "id_order"
    "scorecard": {
        "features": [
            "LIMIT_BAL",
            "AGE",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
            "PAY_RATIO",
            "UTIL_AVG",
            "PAY6_RATIO",
            "PAY_AMT_LIM",
            "BILL_TREND",
        ],
        "base_score": 600,
        "base_odds": 50,
        "pdo": 20,
        "max_bins": 6,
        "min_bin_share": 0.05,
        "C": None,  # None = tune C and the L1/L2 mix by 5-fold CV log loss
        "binning": "optbinning",  # or "builtin" (no extra dependencies)
        "cv_folds": 5,
        "calibrate_pd": True,  # fit logit PD = a + b * score on development data
    },
    "challenger": {"enabled": True, "n_trials": 40, "n_splits": 5},
    "drift": {
        "enabled": True,
        "high_limit_cut": 200000,
        "high_limit_drop_prob": 0.8,
        "pay_amt_factor": 0.6,
        "label_deterioration_frac": 0.05,
        "missing_pay_amt1_frac": 0.03,
        "age_sentinel_frac": 0.003,
        "duplicate_ids": 20,
    },
    "ratings": {"cut_percentiles": [10, 25, 40, 55, 70, 82, 92]},
    "overrides": {"enabled": True, "cutoff_percentile": 25, "low_side_prob": 0.18, "high_side_prob": 0.02},
    "bootstrap": {"n_boot": 500},
    "data_quality": {
        "ranges": {
            "LIMIT_BAL": [10000, 1000000],
            "AGE": [18, 100],
            "PAY_0": [-2, 9],
            "PAY_2": [-2, 9],
            "PAY_3": [-2, 9],
            "PAY_AMT1": [0, 2000000],
        }
    },
    "thresholds": {
        "psi": {"watch": 0.10, "fail": 0.25},
        "csi": {"watch": 0.10, "fail": 0.25},
        "gini_drop": {"watch": 0.05, "fail": 0.10},
        "ks_drop": {"watch": 0.03, "fail": 0.07},
        "hl_p": {"watch": 0.05, "fail": 0.01},
        "binomial_p": {"watch": 0.05, "fail": 0.01},
        "jeffreys_p": {"watch": 0.05, "fail": 0.01},
        "challenger_gini_gap": {"watch": 0.03, "fail": 0.06},
        "brier_rel_increase": {"watch": 0.05, "fail": 0.15},
        "citl_deviation": {"watch": 0.10, "fail": 0.25},
        "hhi": {"watch": 0.15, "fail": 0.25},
        "inversions": {"watch": 1, "fail": 2},
        "override_rate": {"watch": 0.05, "fail": 0.10},
        "override_p": {"alpha": 0.05},
        "missing_share": {"watch": 0.01, "fail": 0.05},
        "out_of_range_share": {"watch": 0.0, "fail": 0.001},
        "duplicate_share": {"watch": 0.0, "fail": 0.001},
    },
    "output": {"report": "reports/model_audit_report.md", "figures_dir": "docs"},
}


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    base = Path.cwd()
    if path is not None:
        path = Path(path).resolve()
        cfg = _merge(cfg, yaml.safe_load(path.read_text()) or {})
        base = path.parent
    cfg["_base"] = base
    return cfg


def resolve(cfg: dict[str, Any], rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else cfg["_base"] / p
