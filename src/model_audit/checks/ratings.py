"""Rating-grade structure: concentration (HHI) and monotonicity of default rates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..status import CheckResult, classify


def assign_grades(score: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Grade 1 = best ... len(cuts)+1 = worst. ``cuts`` are ascending score cut-offs."""
    return (len(cuts) + 1 - np.searchsorted(cuts, score, side="right")).astype(int)


def hhi(shares: np.ndarray) -> float:
    s = np.asarray(shares, dtype=float)
    s = s / s.sum()
    return float(np.sum(s**2))


def inversions(rates: np.ndarray) -> int:
    """Number of adjacent grade pairs where the better grade has the higher default rate."""
    r = np.asarray(rates, dtype=float)
    return int(np.sum(np.diff(r) < 0))


def run(grade_dev, grade_mon, y_dev, y_mon, n_grades: int, cfg: dict):
    th = cfg["thresholds"]
    grades = np.arange(1, n_grades + 1)
    n_d = np.array([(grade_dev == g).sum() for g in grades])
    n_m = np.array([(grade_mon == g).sum() for g in grades])
    k_d = np.array([y_dev[grade_dev == g].sum() for g in grades])
    k_m = np.array([y_mon[grade_mon == g].sum() for g in grades])
    dr_d, dr_m = k_d / np.maximum(n_d, 1), k_m / np.maximum(n_m, 1)
    tbl = pd.DataFrame(
        {
            "grade": grades,
            "dev_share": n_d / n_d.sum(),
            "mon_share": n_m / n_m.sum(),
            "dev_dr": dr_d,
            "mon_dr": dr_m,
        }
    )

    h_d, h_m = hhi(n_d), hhi(n_m)
    t = th["hhi"]
    st = classify(h_m, t["watch"], t["fail"])
    top = int(np.argmax(n_m)) + 1
    res = [
        CheckResult(
            "RG-01",
            "Rating-grade concentration",
            "HHI of monitoring grade distribution",
            h_m,
            st,
            f"Watch >= {t['watch']}, Fail >= {t['fail']} (minimum possible {1 / n_grades:.3f})",
            "HHI = sum of squared grade shares. > 0.25 means the master scale no longer "
            "differentiates the book; it is also a stability signal.",
            f"HHI {h_m:.3f} vs {h_d:.3f} in development (floor {1 / n_grades:.3f} for "
            f"{n_grades} grades); largest grade is {top} with {n_m.max() / n_m.sum():.1%} of accounts.",
            "Review master-scale cut-offs against the current population; consider re-banding.",
            {},
        )
    ]

    inv_d, inv_m = inversions(dr_d), inversions(dr_m)
    t = th["inversions"]
    st = classify(inv_m, t["watch"], t["fail"])
    pairs = [f"{g}->{g + 1}" for g in range(1, n_grades) if dr_m[g] < dr_m[g - 1]]
    res.append(
        CheckResult(
            "RG-02",
            "Monotonicity of default rates",
            "grade-pair inversions, monitoring",
            float(inv_m),
            st,
            f"Watch >= {t['watch']}, Fail >= {t['fail']} inversions",
            "Observed default rates must rise with worsening grade; an inversion means the "
            "ordering the rating system promises is not delivered.",
            (
                f"Default rates are monotone across all {n_grades} grades (development inversions: {inv_d})."
                if inv_m == 0
                else f"{inv_m} inversion(s) at grade pair(s) {', '.join(pairs)} "
                f"(development inversions: {inv_d})."
            ),
            "Merge adjacent grades or re-estimate PDs; test whether inversions are statistically "
            "significant given grade sizes.",
            {},
        )
    )
    return res, {"grades": tbl}
