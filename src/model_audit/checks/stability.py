"""Population stability: PSI on the score, CSI per scorecard characteristic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import psi_from_counts
from ..scorecard import Scorecard
from ..status import CheckResult, classify


def score_bands(dev_score: np.ndarray, n: int = 10) -> np.ndarray:
    """Interior cut-offs of ``n`` equal-count bands of the development score."""
    return np.unique(np.quantile(dev_score, np.linspace(0, 1, n + 1)[1:-1]))


def run(sc: Scorecard, dev: pd.DataFrame, mon: pd.DataFrame, dev_score, mon_score, cfg: dict):
    th = cfg["thresholds"]
    res: list[CheckResult] = []

    cuts = score_bands(dev_score)
    d_cnt = np.bincount(np.searchsorted(cuts, dev_score, side="right"), minlength=len(cuts) + 1)
    m_cnt = np.bincount(np.searchsorted(cuts, mon_score, side="right"), minlength=len(cuts) + 1)
    psi = psi_from_counts(d_cnt, m_cnt)
    p = th["psi"]
    st = classify(psi, p["watch"], p["fail"])
    res.append(
        CheckResult(
            "ST-01",
            "Score PSI",
            "PSI of final score (10 development deciles)",
            psi,
            st,
            f"Watch >= {p['watch']}, Fail >= {p['fail']}",
            "Rule of thumb: < 0.10 no shift, 0.10-0.25 moderate shift, "
            ">= 0.25 significant shift (Siddiqi 2006).",
            f"Score PSI is {psi:.3f}; mean score moved from {dev_score.mean():.1f} to "
            f"{mon_score.mean():.1f}.",
            "Investigate which characteristics drive the shift (see CSI), decide whether the "
            "change is a population move or a data issue, and document the decision to keep, "
            "recalibrate or redevelop.",
            {"dev_counts": d_cnt, "mon_counts": m_cnt},
        )
    )

    d_idx, m_idx = sc.bin_index(dev), sc.bin_index(mon)
    rows = []
    c = th["csi"]
    for f in sc.features:
        b = sc.bins[f]
        dc = np.bincount(d_idx[f], minlength=b.n_bins)
        mc = np.bincount(m_idx[f], minlength=b.n_bins)
        csi = psi_from_counts(dc, mc)
        pts = np.asarray(b.points, dtype=float)
        shift = float(((mc / mc.sum()) - (dc / dc.sum())) @ pts)
        st = classify(csi, c["watch"], c["fail"])
        rows.append({"feature": f, "csi": csi, "points_shift": shift, "status": st.label})
        top = int(np.argmax(np.abs(mc / mc.sum() - dc / dc.sum())))
        res.append(
            CheckResult(
                f"ST-02:{f}",
                f"CSI {f}",
                f"CSI of {f} on scorecard bins",
                csi,
                st,
                f"Watch >= {c['watch']}, Fail >= {c['fail']}",
                "Same index as PSI applied per characteristic; points shift is the average score "
                "effect of the movement.",
                f"CSI {csi:.3f}; average score effect {shift:+.1f} points; largest move in bin "
                f"{b.labels()[top]} ({dc[top] / dc.sum():.1%} to {mc[top] / mc.sum():.1%}).",
                f"Confirm whether the {f} shift is genuine population change or an upstream "
                "data/definition change; reassess bin cut-offs and points if it persists.",
                {},
            )
        )
    tbl = pd.DataFrame(rows)
    return res, {"csi": tbl}
