"""Data-quality checks: missing values, out-of-range values, duplicate IDs."""

from __future__ import annotations

import pandas as pd

from ..status import CheckResult, classify, worst


def run(dev: pd.DataFrame, mon: pd.DataFrame, cfg: dict):
    th, dq = cfg["thresholds"], cfg["data_quality"]
    idc = cfg["data"]["id_col"]
    cols = [
        c for c in cfg["scorecard"]["features"] + ["BILL_AMT1", "PAY_AMT1", "LIMIT_BAL"] if c in mon.columns
    ]
    cols = list(dict.fromkeys(cols))
    rows = []
    for c in cols:
        rows.append({"column": c, "dev_missing": dev[c].isna().mean(), "mon_missing": mon[c].isna().mean()})
    miss = pd.DataFrame(rows)
    m_th = th["missing_share"]
    mx = miss.loc[miss["mon_missing"].idxmax()]
    st = worst([classify(v, m_th["watch"], m_th["fail"]) for v in miss["mon_missing"]])
    flagged = miss[miss["mon_missing"] >= m_th["watch"]]
    res = [
        CheckResult(
            "DQ-01",
            "Missing values",
            "max missing share (monitoring)",
            float(mx["mon_missing"]),
            st,
            f"Watch >= {m_th['watch']:.0%}, Fail >= {m_th['fail']:.0%} per column",
            "Scorecards assign missing values a fixed (often neutral) bin; unexpected missingness "
            "silently moves scores.",
            (
                "No column exceeds the missing-value tolerance."
                if st.value == 0
                else "Missing values above tolerance in: "
                + ", ".join(
                    f"{r.column} ({r.mon_missing:.1%} vs {r.dev_missing:.1%} in development)"
                    for r in flagged.itertuples()
                )
                + "."
            ),
            "Trace the source of the gaps, confirm the scorecard's missing-value treatment is "
            "intended and add a completeness control on the input feed.",
            {"table": miss},
        )
    ]

    rng_rows, total_bad = [], 0
    for c, (lo, hi) in dq["ranges"].items():
        if c not in mon.columns:
            continue
        bad_n = int(((mon[c] < lo) | (mon[c] > hi)).sum())
        total_bad += bad_n
        rng_rows.append(
            {"column": c, "allowed": f"[{lo}, {hi}]", "violations": bad_n, "share": bad_n / len(mon)}
        )
    rng_df = pd.DataFrame(rng_rows)
    share = float(rng_df["share"].max()) if len(rng_df) else 0.0
    r_th = th["out_of_range_share"]
    st = classify(share, r_th["watch"], r_th["fail"], inclusive=False)
    v = rng_df[rng_df["violations"] > 0]
    res.append(
        CheckResult(
            "DQ-02",
            "Out-of-range values",
            "max out-of-range share (monitoring)",
            share,
            st,
            f"Watch > {r_th['watch']:.1%}, Fail > {r_th['fail']:.1%} per column",
            "Values outside documented domains indicate sentinel codes or feed errors that are "
            "binned as if they were genuine.",
            (
                "No values outside the configured ranges."
                if st.value == 0
                else "Out-of-range values: "
                + ", ".join(
                    f"{r.column} {r.violations} rows ({r.share:.2%}, allowed {r.allowed})"
                    for r in v.itertuples()
                )
                + "."
            ),
            "Add range validation at ingestion, quarantine violating records and confirm how the "
            "scorecard bins them (they currently fall in the extreme bin).",
            {"table": rng_df},
        )
    )

    dups = int(mon[idc].duplicated().sum())
    d_th = th["duplicate_share"]
    share = dups / len(mon)
    st = classify(share, d_th["watch"], d_th["fail"], inclusive=False)
    res.append(
        CheckResult(
            "DQ-03",
            "Duplicate IDs",
            "duplicate share of IDs (monitoring)",
            share,
            st,
            f"Watch > {d_th['watch']:.1%}, Fail > {d_th['fail']:.1%}",
            "Repeated account IDs double-count exposures and defaults and distort every rate "
            "in the validation.",
            (
                "No duplicate IDs."
                if st.value == 0
                else f"{dups} duplicate {idc} values ({share:.2%} of {len(mon):,} monitoring rows)."
            ),
            "Deduplicate on the account key, find the process that creates repeats and reconcile "
            "record counts to the source system.",
            {"duplicates": dups},
        )
    )
    return res, {"missing": miss, "ranges": rng_df}
