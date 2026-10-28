"""Override analysis on a (simulated) override flag."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from ..status import CheckResult, Status, classify


def run(y_mon, pd_mon, score_mon, flag, cutoff: float, cfg: dict):
    th = cfg["thresholds"]
    flag = np.asarray(flag, dtype=bool)
    low = score_mon < cutoff  # policy decline zone
    n = len(flag)
    rate = float(flag.mean())
    low_side = flag & low
    high_side = flag & ~low
    t = th["override_rate"]
    st = classify(rate, t["watch"], t["fail"])
    res = [
        CheckResult(
            "OV-01",
            "Override rate",
            "overrides / decisions, monitoring",
            rate,
            st,
            f"Watch >= {t['watch']:.0%}, Fail >= {t['fail']:.0%}",
            "Common policy tolerance: overrides above ~5-10% suggest the scorecard is not trusted "
            "or the policy is mis-specified.",
            f"{int(flag.sum()):,} overrides on {n:,} decisions ({rate:.1%}): "
            f"{int(low_side.sum()):,} low-side (approved below cut-off {cutoff:.0f}, "
            f"{low_side.sum() / max(low.sum(), 1):.1%} of the decline zone) and "
            f"{int(high_side.sum()):,} high-side.",
            "Obtain override reason codes and approver authority, and report rates by approver "
            "and branch against the policy limit.",
            {},
        )
    ]

    ov = low_side
    ref = low & ~flag  # same score zone, no override
    k_o, n_o = int(y_mon[ov].sum()), int(ov.sum())
    k_r, n_r = int(y_mon[ref].sum()), int(ref.sum())
    dr_o, dr_r = k_o / max(n_o, 1), k_r / max(n_r, 1)
    alpha = th["override_p"]["alpha"]
    p_lower = fisher_exact([[k_o, n_o - k_o], [k_r, n_r - k_r]], alternative="less")[1]
    p_higher = fisher_exact([[k_o, n_o - k_o], [k_r, n_r - k_r]], alternative="greater")[1]
    if p_higher < alpha:
        st = Status.FAIL
        msg = "overridden accounts perform significantly worse than non-overridden accounts"
    elif p_lower < alpha:
        st, msg = Status.PASS, "overridden accounts perform significantly better (override adds value)"
    else:
        st, msg = Status.WATCH, "no evidence that overrides add value over the score"
    res.append(
        CheckResult(
            "OV-02",
            "Override performance",
            "low-side override DR minus same-zone non-override DR",
            dr_o - dr_r,
            st,
            f"Pass: override DR significantly lower (Fisher p < {alpha}); Fail: significantly "
            f"higher; else Watch",
            "A low-side override is justified only if the analyst adds information the score lacks, "
            "i.e. overridden accounts outperform comparable non-overridden ones.",
            f"Low-side overrides: default rate {dr_o:.1%} (n={n_o:,}, mean predicted PD "
            f"{pd_mon[ov].mean():.1%}) vs {dr_r:.1%} for non-overridden accounts in the same zone "
            f"(n={n_r:,}); one-sided p = {p_lower:.3f} (lower) / {p_higher:.3f} (higher): {msg}.",
            "Back-test override rationales, require documented evidence for low-side overrides and "
            "monitor their performance separately.",
            {},
        )
    )
    tbl = pd.DataFrame(
        [
            {
                "group": "Low-side override",
                "n": n_o,
                "defaults": k_o,
                "default_rate": dr_o,
                "mean_pd": float(pd_mon[ov].mean()) if n_o else np.nan,
            },
            {
                "group": "Decline zone, no override",
                "n": n_r,
                "defaults": k_r,
                "default_rate": dr_r,
                "mean_pd": float(pd_mon[ref].mean()),
            },
            {
                "group": "Approve zone, no override",
                "n": int((~low & ~flag).sum()),
                "defaults": int(y_mon[~low & ~flag].sum()),
                "default_rate": float(y_mon[~low & ~flag].mean()),
                "mean_pd": float(pd_mon[~low & ~flag].mean()),
            },
            {
                "group": "High-side override",
                "n": int(high_side.sum()),
                "defaults": int(y_mon[high_side].sum()),
                "default_rate": float(y_mon[high_side].mean()) if high_side.any() else np.nan,
                "mean_pd": float(pd_mon[high_side].mean()) if high_side.any() else np.nan,
            },
        ]
    )
    return res, {"overrides": tbl}
