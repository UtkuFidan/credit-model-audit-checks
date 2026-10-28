"""Calibration: HL test, binomial test per grade, Brier score, calibration-in-the-large."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ..metrics import holm_adjust, hosmer_lemeshow, jeffreys_pvalue, wilson_ci
from ..status import CheckResult, Status, classify


def band_table(y, pd_hat, score, cuts) -> pd.DataFrame:
    """Predicted vs observed default rate per score band, with Wilson intervals."""
    band = np.searchsorted(cuts, score, side="right")  # 0 = lowest score = riskiest
    rows = []
    for b in range(len(cuts) + 1):
        m = band == b
        n = int(m.sum())
        if n == 0:
            continue
        k = int(y[m].sum())
        lo, hi = wilson_ci(np.array([k]), np.array([n]))
        rows.append(
            {
                "band": b + 1,
                "n": n,
                "mean_pd": float(pd_hat[m].mean()),
                "observed_dr": k / n,
                "ci_low": float(lo[0]),
                "ci_high": float(hi[0]),
            }
        )
    return pd.DataFrame(rows)


def run(y_dev, y_mon, pd_dev, pd_mon, grade_mon, grade_pd: dict, cfg: dict) -> list[CheckResult]:
    th = cfg["thresholds"]
    res = []

    _, p_dev, _ = hosmer_lemeshow(y_dev, pd_dev, df_offset=2)
    stat, p_mon, df = hosmer_lemeshow(y_mon, pd_mon, df_offset=0)
    t = th["hl_p"]
    st = classify(p_mon, t["watch"], t["fail"], higher_is_worse=False)
    res.append(
        CheckResult(
            "CA-01",
            "Hosmer-Lemeshow",
            "HL p-value, monitoring (10 groups)",
            p_mon,
            st,
            f"Watch <= {t['watch']}, Fail <= {t['fail']}",
            "H0: predicted PDs are calibrated. Frozen model tested out of sample, so df = groups; "
            "with large n the test rejects economically small deviations.",
            f"HL chi-square {stat:.1f} on {df} df, p = {p_mon:.3g} (development in-sample p = {p_dev:.3g}).",
            "Read together with the band table: if the deviation is directional and material, "
            "recalibrate the score-to-PD mapping; document the choice.",
            {"stat": stat, "df": df, "p_dev": p_dev},
        )
    )

    t = th["binomial_p"]
    rows, sts = [], []
    g_labels = sorted(grade_pd)
    for g in g_labels:
        m = grade_mon == g
        n, k = int(m.sum()), int(y_mon[m].sum())
        pv = binomtest(k, n, grade_pd[g], alternative="greater").pvalue if n else 1.0
        s = classify(pv, t["watch"], t["fail"], higher_is_worse=False)
        sts.append(s)
        rows.append(
            {
                "grade": g,
                "n": n,
                "defaults": k,
                "grade_pd": grade_pd[g],
                "observed_dr": k / n if n else np.nan,
                "p_value": pv,
                "status": s.label,
            }
        )
    gt = pd.DataFrame(rows)
    gt["p_holm"] = holm_adjust(gt["p_value"])
    n_fail = sum(s.value == 2 for s in sts)
    st = Status.FAIL if n_fail >= 2 else Status.WATCH if any(sts) else Status.PASS
    bad = gt[gt["status"] != "Pass"]
    res.append(
        CheckResult(
            "CA-02",
            "Binomial test per grade",
            "grades failing the one-sided test (p <= 0.01)",
            float(n_fail),
            st,
            f"Pass: no grade p < {t['watch']}; Fail: >= 2 grades p <= {t['fail']}; otherwise Watch",
            "Basel-style test of whether observed defaults exceed the grade's master-scale PD "
            "(under-statement of risk). The status rule uses raw p-values; the table adds Holm-adjusted ones "
            "because the grades are tested as a family.",
            (
                "All grades are consistent with their master-scale PD."
                if st.value == 0
                else "Observed default rate above grade PD in grade(s): "
                + "; ".join(
                    f"{r.grade} ({r.observed_dr:.1%} vs {r.grade_pd:.1%}, n={r.n}, p={r.p_value:.3g})"
                    for r in bad.itertuples()
                )
                + "."
            ),
            "Recalibrate grade PDs or add a margin of conservatism; escalate where the "
            "under-prediction affects provisioning or pricing.",
            {"table": gt},
        )
    )

    t = th["jeffreys_p"]
    rows, sts = [], []
    for g in g_labels:
        m = grade_mon == g
        n, k = int(m.sum()), int(y_mon[m].sum())
        pv = float(jeffreys_pvalue(k, n, grade_pd[g])) if n else 1.0
        s = classify(pv, t["watch"], t["fail"], higher_is_worse=False)
        sts.append(s)
        rows.append(
            {"grade": g, "n": n, "defaults": k, "grade_pd": grade_pd[g], "p_value": pv, "status": s.label}
        )
    jt = pd.DataFrame(rows)
    jt["p_holm"] = holm_adjust(jt["p_value"])
    n_fail = sum(s.value == 2 for s in sts)
    st = Status.FAIL if n_fail >= 2 else Status.WATCH if any(sts) else Status.PASS
    bad = jt[jt["status"] != "Pass"]
    res.append(
        CheckResult(
            "CA-05",
            "Jeffreys test per grade",
            "grades failing the Jeffreys test (p <= 0.01)",
            float(n_fail),
            st,
            f"Pass: no grade p < {t['watch']}; Fail: >= 2 grades p <= {t['fail']}; otherwise Watch",
            "Bayesian companion to CA-02 (Jeffreys prior Beta(1/2, 1/2); Tasche 2008): p is "
            "the posterior probability that the true PD is at or below the grade PD. Better behaved "
            "than the exact binomial test in small or low-default grades.",
            (
                "All grades are consistent with their master-scale PD."
                if st.value == 0
                else "Grade PD likely too low in: "
                + "; ".join(f"{r.grade} (p={r.p_value:.3g}, n={r.n})" for r in bad.itertuples())
                + "."
            ),
            "Use with CA-02. Where the two tests disagree, the grade is small enough that the "
            "conclusion should rest on the effect size, not the p-value.",
            {"table": jt},
        )
    )

    br_d = float(np.mean((pd_dev - y_dev) ** 2))
    br_m = float(np.mean((pd_mon - y_mon) ** 2))
    rel = (br_m - br_d) / br_d
    ref = float(y_mon.mean() * (1 - y_mon.mean()))
    t = th["brier_rel_increase"]
    st = classify(rel, t["watch"], t["fail"])
    res.append(
        CheckResult(
            "CA-03",
            "Brier score",
            "relative Brier increase vs development",
            rel,
            st,
            f"Watch >= {t['watch']:.0%}, Fail >= {t['fail']:.0%}",
            "Brier combines calibration and discrimination; a relative increase signals overall "
            "probabilistic accuracy loss.",
            f"Brier {br_d:.4f} (development) vs {br_m:.4f} (monitoring), {rel:+.1%}; the "
            f"base-rate benchmark is {ref:.4f}.",
            "Decompose into calibration and discrimination loss using the band and Gini results.",
            {"brier_dev": br_d, "brier_mon": br_m, "brier_ref": ref},
        )
    )

    pred, obs = float(pd_mon.mean()), float(y_mon.mean())
    dev_ratio = float(y_dev.mean() / pd_dev.mean())
    deviation = abs(obs / pred - 1)
    t = th["citl_deviation"]
    st = classify(deviation, t["watch"], t["fail"])
    res.append(
        CheckResult(
            "CA-04",
            "Calibration in the large",
            "|observed / predicted default rate - 1|",
            deviation,
            st,
            f"Watch >= {t['watch']:.0%}, Fail >= {t['fail']:.0%}",
            "Portfolio-level bias of the PD. Provisioning and pricing depend on it directly.",
            f"Predicted default rate {pred:.2%} vs observed {obs:.2%} (ratio {obs / pred:.2f}; "
            f"development ratio {dev_ratio:.2f}).",
            "If observed exceeds predicted, quantify the provisioning/pricing impact and apply a "
            "scalar recalibration or an overlay until the model is redeveloped.",
            {"pred": pred, "obs": obs},
        )
    )
    return res
