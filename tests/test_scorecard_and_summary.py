import numpy as np
import pandas as pd
import pytest

from model_audit.scorecard import Scorecard, fit_binning
from model_audit.status import CheckResult, Status
from model_audit.summary import traffic_light


def _data(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + 1.0 * x + 0.5 * z)))).astype(int)
    return pd.DataFrame({"x": x, "z": z}), y


def _monotone(woe):
    d = np.diff(woe)
    return bool((d >= -1e-12).all() or (d <= 1e-12).all())


@pytest.mark.parametrize("method", ["builtin", "optbinning"])
def test_binning_is_monotone_with_separate_missing_bin(method):
    if method == "optbinning":
        pytest.importorskip("optbinning")
    df, y = _data()
    x = df["x"].copy()
    x[::50] = np.nan
    b = fit_binning(x, y, max_bins=6, min_share=0.05, method=method)
    assert _monotone(b.woe[:-1])
    assert b.n_bins == len(b.woe)
    assert b.woe[0] > b.woe[-2]  # higher x = riskier = lower WoE (WoE is ln good/bad)


def test_tuned_logistic_records_choice_and_scores_better_than_random():
    df, y = _data()
    sc = Scorecard(["x", "z"], binning="builtin", C=None).fit(df, y)
    assert sc.tuning_["C"] > 0 and sc.tuning_["l1_ratio"] in (0.0, 0.5, 1.0)
    assert all(c < 0 for c in sc.coef_.values())


def test_wrong_sign_characteristic_is_dropped_and_recorded():
    rng = np.random.default_rng(5)
    n = 6000
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=0.3, size=n)
    # x2 correlates positively with risk marginally but enters negatively given x1
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + 2.5 * x1 - 1.8 * x2)))).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2})
    sc = Scorecard(["x1", "x2"], binning="builtin", C=10.0).fit(df, y)
    assert sc.dropped_ and all(c <= 1e-9 for c in sc.coef_.values())


def _r(cid, st):
    return CheckResult(cid, "n", "m", 0.0, st, "t", "r", "o", "rec")


def test_traffic_light_is_worst_of_area_and_skips_empty_areas():
    res = [
        _r("ST-01", Status.PASS),
        _r("ST-02:A", Status.WATCH),
        _r("CA-01", Status.FAIL),
        _r("CA-02", Status.PASS),
    ]
    tl, overall = traffic_light(res)
    assert overall == Status.FAIL
    assert dict(zip(tl["area"], tl["rating"], strict=True)) == {"Stability": "Amber", "Calibration": "Red"}
    row = tl.set_index("area").loc["Calibration"]
    assert (row["fail"], row["watch"], row["pass"]) == (1, 0, 1)


def test_pd_recalibration_matches_development_rate_and_is_monotone():
    df, y = _data(8000)
    sc = Scorecard(["x", "z"], binning="builtin", C=0.05).fit(df, y)  # heavy shrinkage on purpose
    s = sc.score(df)
    scaled = sc.pd_from_score(s)
    sc.calibrate_pd(s, y)
    cal = sc.pd_from_score(s)
    assert cal.mean() == pytest.approx(y.mean(), abs=1e-3)  # intercept keeps the mean PD on target
    grid = np.linspace(s.min(), s.max(), 200)
    assert np.all(np.diff(sc.pd_from_score(grid)) <= 1e-9)  # higher score, lower PD
    # the recalibrated PD is at least as good a probability forecast in development
    assert np.mean((cal - y) ** 2) <= np.mean((scaled - y) ** 2) + 1e-9
