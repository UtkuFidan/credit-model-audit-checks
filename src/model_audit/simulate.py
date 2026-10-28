"""Development / monitoring split with DOCUMENTED, DELIBERATE drift injection.

The public data has no genuine "later" period, so the monitoring sample is a random half of the
data that is then perturbed. Every perturbation below is a known ground truth the audit checks
should (or should not) detect. Nothing here describes real portfolio behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_dev_monitoring(df: pd.DataFrame, cfg: dict, rng: np.random.Generator):
    """Random split, or (``split.mode: id_order``) the first part of the file by ID as development.

    The ID-ordered split is an ordered holdout. The file does not say that ID is chronological,
    so it is used as the only natural ordering available, not as a time axis.
    """
    n = len(df)
    n_dev = int(n * cfg["split"]["dev_fraction"])
    if cfg["split"].get("mode", "random") == "id_order":
        ordered = df.sort_values(cfg["data"]["id_col"]).reset_index(drop=True)
        return ordered.iloc[:n_dev].reset_index(drop=True), ordered.iloc[n_dev:].reset_index(drop=True)
    perm = rng.permutation(n)
    dev = df.iloc[np.sort(perm[:n_dev])].reset_index(drop=True)
    mon = df.iloc[np.sort(perm[n_dev:])].reset_index(drop=True)
    return dev, mon


def inject_drift(mon: pd.DataFrame, cfg: dict, rng: np.random.Generator):
    """Return (perturbed monitoring sample, list of human-readable injection notes)."""
    d, tgt, idc = cfg["drift"], cfg["data"]["target_col"], cfg["data"]["id_col"]
    mon = mon.copy()
    notes: list[str] = []

    # 1. population mix shift: fewer high-limit customers (covariate drift)
    hi = mon["LIMIT_BAL"] >= d["high_limit_cut"]
    drop = hi & (rng.random(len(mon)) < d["high_limit_drop_prob"])
    mon = mon.loc[~drop].reset_index(drop=True)
    notes.append(
        f"Population mix: {int(drop.sum())} accounts with LIMIT_BAL >= {d['high_limit_cut']:,} "
        f"removed (drop prob {d['high_limit_drop_prob']:.0%})."
    )

    # 2. behavioural drift: customers repay less of their bill (repayment-ratio drift)
    mon["PAY_AMT1"] = mon["PAY_AMT1"] * d["pay_amt_factor"]
    notes.append(f"Repayment drift: PAY_AMT1 multiplied by {d['pay_amt_factor']}.")

    # 3. hidden deterioration the scorecard cannot see: some current payers default
    cand = np.where((mon[tgt] == 0) & (mon["PAY_0"] <= 0))[0]
    flip = rng.choice(cand, size=int(len(cand) * d["label_deterioration_frac"]), replace=False)
    mon.loc[flip, tgt] = 1
    notes.append(
        f"Outcome deterioration: {len(flip)} non-defaulting, currently-paying accounts "
        f"({d['label_deterioration_frac']:.0%} of them) relabelled as defaults."
    )

    # 4. data-quality defects
    miss = rng.random(len(mon)) < d["missing_pay_amt1_frac"]
    mon.loc[miss, "PAY_AMT1"] = np.nan
    notes.append(f"Missing values: PAY_AMT1 blanked for {int(miss.sum())} rows.")
    bad_age = rng.random(len(mon)) < d["age_sentinel_frac"]
    mon.loc[bad_age, "AGE"] = 999
    notes.append(f"Out-of-range values: AGE set to sentinel 999 for {int(bad_age.sum())} rows.")
    dup = mon.sample(n=d["duplicate_ids"], random_state=int(rng.integers(1e9)))
    mon = pd.concat([mon, dup], ignore_index=True)
    notes.append(f"Duplicate IDs: {d['duplicate_ids']} rows appended twice ({idc} repeated).")
    return mon, notes


def simulate_overrides(
    mon: pd.DataFrame, score: np.ndarray, cutoff: float, cfg: dict, rng: np.random.Generator
) -> np.ndarray:
    """Simulated manual-override flag on monitoring decisions.

    Policy decision: approve if score >= cutoff. Low-side override = approve below cutoff;
    high-side override = decline above cutoff. Selection is random (no information advantage),
    which is the null case an auditor wants to be able to detect: overrides that add no value.
    """
    o = cfg["overrides"]
    p = np.where(score < cutoff, o["low_side_prob"], o["high_side_prob"])
    return rng.random(len(mon)) < p
