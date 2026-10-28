"""Discrimination: Gini / AUC and KS with bootstrap CIs, development vs monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import auc, bootstrap_values, gini, ks_statistic, percentile_ci
from ..status import CheckResult, classify


def run(y_dev, y_mon, score_dev, score_mon, cfg: dict, rng: np.random.Generator):
    th, nb = cfg["thresholds"], cfg["bootstrap"]["n_boot"]
    risk_d, risk_m = -np.asarray(score_dev), -np.asarray(score_mon)  # low score = high risk
    rows, drops = {}, {}
    for name, fn in (("AUC", auc), ("Gini", gini), ("KS", ks_statistic)):
        boot = {}
        for tag, y, r in (("dev", y_dev, risk_d), ("mon", y_mon, risk_m)):
            boot[tag] = bootstrap_values(y, r, fn, nb, rng)
            lo, hi = percentile_ci(boot[tag])
            rows[(name, tag)] = (fn(y, r), lo, hi)
        # the two samples are independent, so the drop is bootstrapped as a difference of draws
        drops[name] = percentile_ci(boot["dev"] - boot["mon"])
    tbl = pd.DataFrame(
        [
            {"metric": m, "sample": t, "value": v[0], "ci_low": v[1], "ci_high": v[2]}
            for (m, t), v in rows.items()
        ]
    )

    out = []
    for cid, name, key, tkey, scale in (
        ("DI-01", "Gini drift", "Gini", "gini_drop", "Gini"),
        ("DI-02", "KS drift", "KS", "ks_drop", "KS"),
    ):
        d, m = rows[(key, "dev")], rows[(key, "mon")]
        drop = d[0] - m[0]
        dlo, dhi = drops[key]
        t = th[tkey]
        st = classify(drop, t["watch"], t["fail"])
        out.append(
            CheckResult(
                cid,
                name,
                f"{scale} drop, development minus monitoring",
                drop,
                st,
                f"Watch >= {t['watch']}, Fail >= {t['fail']} (absolute points)",
                "Rank-ordering power should be stable out of time; thresholds are an internal "
                "convention to be aligned with the bank's model-monitoring policy.",
                f"{scale} {d[0]:.3f} (95% CI {d[1]:.3f}-{d[2]:.3f}) in development vs {m[0]:.3f} "
                f"(95% CI {m[1]:.3f}-{m[2]:.3f}) in monitoring, a drop of {drop:.3f} "
                f"(95% CI {dlo:.3f} to {dhi:.3f}).",
                "Check whether the loss comes from specific characteristics or segments, and "
                "assess whether recalibration is enough or redevelopment is required.",
                {"drop_ci": (dlo, dhi)},
            )
        )
    return out, {"discrimination": tbl}
