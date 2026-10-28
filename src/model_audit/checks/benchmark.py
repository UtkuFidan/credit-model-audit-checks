"""Benchmarking against a challenger model: does a flexible model find signal the scorecard misses?"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..challenger import Challenger
from ..metrics import gini, ks_statistic, paired_bootstrap_diff
from ..status import CheckResult, classify


def run(ch: Challenger, y_mon, champ_score_mon, ch_pd_mon, champ_pd_mon, cfg: dict, rng):
    th = cfg["thresholds"]["challenger_gini_gap"]
    nb = cfg["bootstrap"]["n_boot"]
    risk_c = -np.asarray(champ_score_mon)
    gap = gini(y_mon, ch_pd_mon) - gini(y_mon, risk_c)
    lo, hi = paired_bootstrap_diff(y_mon, ch_pd_mon, risk_c, gini, nb, rng)
    st = classify(gap, th["watch"], th["fail"])
    rho = float(spearmanr(risk_c, ch_pd_mon)[0])
    thr = f"Watch >= {th['watch']}, Fail >= {th['fail']} (Gini points, challenger minus champion)"
    res = [
        CheckResult(
            "BM-01",
            "Challenger gap, monitoring",
            "Gini(challenger) - Gini(champion) on monitoring data",
            gap,
            st,
            thr,
            "A challenger trained on the same development rows with the same raw inputs sets a "
            "practical ceiling. A large gap means the scorecard structure (binning, additivity, "
            "variable set) is costing rank-ordering power.",
            f"Gini {gini(y_mon, ch_pd_mon):.3f} (challenger) vs {gini(y_mon, risk_c):.3f} (champion), "
            f"gap {gap:.3f} (paired bootstrap 95% CI {lo:.3f} to {hi:.3f}). Rank correlation of the "
            f"two scores {rho:.2f}.",
            "Review variables the challenger uses and the scorecard omits (feature importance, "
            "interactions) and decide whether to redevelop, add characteristics or accept the gap.",
            {"ci": (lo, hi), "spearman": rho},
        )
    ]
    cv = ch.cv
    d = cv["challenger_gini"] - cv["champion_gini"]
    gap_cv = float(d.mean())
    st = classify(gap_cv, th["watch"], th["fail"])
    n = len(cv)
    res.append(
        CheckResult(
            "BM-02",
            "Challenger gap, development CV",
            f"mean fold Gini gap, {n}-fold CV on development data",
            gap_cv,
            st,
            thr,
            "Out-of-fold comparison on identical splits, with the scorecard re-binned and refitted in "
            "every fold. This is the cleaner comparison: the champion gets no in-sample advantage.",
            f"Gini {cv['challenger_gini'].mean():.3f} +/- {cv['challenger_gini'].std(ddof=1):.3f} "
            f"(challenger) vs {cv['champion_gini'].mean():.3f} +/- {cv['champion_gini'].std(ddof=1):.3f} "
            f"(champion), mean gap {gap_cv:.3f} +/- {d.std(ddof=1):.3f} over {n} folds "
            f"(mean +/- sd; challenger won {int((d > 0).sum())} of {n}).",
            "As BM-01. A gap that is stable across folds is structural; one driven by one fold is noise.",
            {"cv": cv},
        )
    )
    table = pd.DataFrame(
        [
            {
                "model": name,
                "cv_gini": f"{cv[f'{k}_gini'].mean():.3f} +/- {cv[f'{k}_gini'].std(ddof=1):.3f}",
                "mon_gini": gini(y_mon, r),
                "mon_ks": ks_statistic(y_mon, r),
                "mon_brier": float(np.mean((p - y_mon) ** 2)),
                "mean_pd": float(p.mean()),
            }
            for name, k, r, p in (
                ("Champion scorecard", "champion", risk_c, champ_pd_mon),
                ("Challenger LightGBM", "challenger", ch_pd_mon, ch_pd_mon),
            )
        ]
    )
    return res, {"benchmark": table, "challenger_cv": cv}
