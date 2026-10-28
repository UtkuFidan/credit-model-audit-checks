"""Traffic-light roll-up of check results by risk area."""

from __future__ import annotations

import pandas as pd

from .status import CheckResult, Status

AREAS = {
    "ST": "Stability",
    "DI": "Discrimination",
    "CA": "Calibration",
    "RG": "Rating structure",
    "OV": "Overrides",
    "DQ": "Data quality",
    "BM": "Benchmark",
}
RAG = {Status.PASS: "Green", Status.WATCH: "Amber", Status.FAIL: "Red"}


def traffic_light(results: list[CheckResult]) -> tuple[pd.DataFrame, Status]:
    """One row per area: the worst check status and the Pass/Watch/Fail counts.

    The overall rating is the worst area. An area with no checks (for example overrides when
    switched off) is omitted rather than shown as green.
    """
    rows = []
    for prefix, name in AREAS.items():
        rs = [r for r in results if r.check_id.split("-")[0] == prefix]
        if not rs:
            continue
        worst = max(r.status for r in rs)
        rows.append(
            {
                "area": name,
                "rating": RAG[worst],
                "fail": sum(r.status == Status.FAIL for r in rs),
                "watch": sum(r.status == Status.WATCH for r in rs),
                "pass": sum(r.status == Status.PASS for r in rs),
            }
        )
    tl = pd.DataFrame(rows)
    overall = max(Status.PASS, *(r.status for r in results)) if results else Status.PASS
    return tl, overall
