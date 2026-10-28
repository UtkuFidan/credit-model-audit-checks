import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve

from model_audit.metrics import auc, gini, hosmer_lemeshow, ks_statistic, psi_from_counts


def test_psi_hand_computed():
    # expected 50/50, actual 40/60:
    # (0.4-0.5)ln(0.4/0.5) + (0.6-0.5)ln(0.6/0.5) = 0.0223144 + 0.0182322 = 0.0405465
    assert psi_from_counts([500, 500], [400, 600]) == pytest.approx(0.0405465, abs=1e-6)


def test_psi_identical_is_zero_and_empty_bin_is_finite():
    assert psi_from_counts([10, 20, 30], [1, 2, 3]) == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(psi_from_counts([10, 20, 30], [0, 30, 30]))


@pytest.fixture()
def scored():
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.2).astype(int)
    risk = rng.normal(size=3000) + 1.0 * y
    risk = np.round(risk, 1)  # introduce ties
    return y, risk


def test_auc_gini_match_sklearn(scored):
    y, r = scored
    assert auc(y, r) == pytest.approx(roc_auc_score(y, r), abs=1e-12)
    assert gini(y, r) == pytest.approx(2 * roc_auc_score(y, r) - 1, abs=1e-12)


def test_ks_matches_sklearn_and_scipy(scored):
    y, r = scored
    fpr, tpr, _ = roc_curve(y, r)
    assert ks_statistic(y, r) == pytest.approx(np.max(tpr - fpr), abs=1e-12)
    assert ks_statistic(y, r) == pytest.approx(ks_2samp(r[y == 1], r[y == 0]).statistic, abs=1e-12)


def test_hl_calibrated_vs_miscalibrated():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.02, 0.6, 20000)
    y_ok = (rng.random(20000) < p).astype(int)
    y_bad = (rng.random(20000) < np.clip(p * 1.5, 0, 1)).astype(int)
    stat_ok, pv_ok, df = hosmer_lemeshow(y_ok, p)
    stat_bad, pv_bad, _ = hosmer_lemeshow(y_bad, p)
    assert df == 8 and stat_ok >= 0
    assert pv_ok > 0.01
    assert pv_bad < 1e-10 and stat_bad > stat_ok


def test_hl_exact_small_example():
    # one group, p=0.5 everywhere, 10 obs, 7 defaults: (7-5)^2 / (10*.5*.5) = 1.6
    y = np.array([1] * 7 + [0] * 3)
    stat, _, _ = hosmer_lemeshow(y, np.full(10, 0.5), groups=1, df_offset=0)
    assert stat == pytest.approx(1.6)


def test_jeffreys_matches_beta_posterior_and_flags_underestimated_pd():
    from scipy.stats import beta

    from model_audit.metrics import jeffreys_interval, jeffreys_pvalue

    # 30 defaults in 200 (15%) against an assigned PD of 10%: posterior Beta(30.5, 170.5)
    assert jeffreys_pvalue(30, 200, 0.10) == pytest.approx(beta.cdf(0.10, 30.5, 170.5))
    assert jeffreys_pvalue(30, 200, 0.10) < 0.05
    # observed rate equal to the assigned PD: about even odds either way
    assert 0.4 < jeffreys_pvalue(20, 200, 0.10) < 0.6
    # zero defaults is handled (the exact binomial interval would be degenerate at 0)
    lo, hi = jeffreys_interval(0, 100)
    assert lo == 0.0 and 0 < hi < 0.05


def test_paired_bootstrap_diff_is_zero_for_identical_models_and_positive_for_better_one():
    from model_audit.metrics import paired_bootstrap_diff

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    good = y + rng.normal(0, 0.8, 2000)
    weak = rng.normal(size=2000)
    assert paired_bootstrap_diff(y, good, good, gini, 50, rng) == (0.0, 0.0)
    lo, hi = paired_bootstrap_diff(y, good, weak, gini, 100, rng)
    assert 0 < lo < hi


def test_holm_adjust_matches_hand_values():
    from model_audit.metrics import holm_adjust

    # sorted p = .01, .02, .04 -> .03, .04, .04 (monotone step-down); original order is kept
    adj = holm_adjust([0.04, 0.01, 0.02])
    assert adj == pytest.approx([0.04, 0.03, 0.04])
    assert holm_adjust([0.5, 0.9]).max() <= 1.0
