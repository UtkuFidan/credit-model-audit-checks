"""Statistical building blocks, implemented from definitions so they can be tested independently."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta, chi2, rankdata

EPS = 1e-4


def psi_from_shares(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI = sum (a - e) * ln(a / e); empty bins are floored at 1e-4 to keep the log finite."""
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_from_counts(expected_counts: np.ndarray, actual_counts: np.ndarray) -> float:
    e = np.asarray(expected_counts, dtype=float)
    a = np.asarray(actual_counts, dtype=float)
    return psi_from_shares(e / e.sum(), a / a.sum())


def auc(y: np.ndarray, risk: np.ndarray) -> float:
    """Mann-Whitney AUC of a risk measure (higher = more likely bad); ties get average rank."""
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(risk)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gini(y: np.ndarray, risk: np.ndarray) -> float:
    return 2 * auc(y, risk) - 1


def ks_statistic(y: np.ndarray, risk: np.ndarray) -> float:
    """Maximum distance between the cumulative risk distributions of bads and goods."""
    y = np.asarray(y)
    risk = np.asarray(risk)
    order = np.argsort(risk, kind="stable")
    r, yy = risk[order], y[order]
    last = np.r_[r[1:] != r[:-1], True]  # evaluate only at the end of tied blocks
    cb = np.cumsum(yy) / yy.sum()
    cg = np.cumsum(1 - yy) / (1 - yy).sum()
    return float(np.max(np.abs(cb - cg)[last]))


def bootstrap_values(y, risk, fn, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Statistic ``fn(y, risk)`` on ``n_boot`` resamples of the accounts (with replacement)."""
    y, risk = np.asarray(y), np.asarray(risk)
    n = len(y)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = fn(y[idx], risk[idx])
    return vals


def percentile_ci(vals: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def bootstrap_ci(y, risk, fn, n_boot: int, rng: np.random.Generator, level: float = 0.95):
    return percentile_ci(bootstrap_values(y, risk, fn, n_boot, rng), level)


def paired_bootstrap_diff(y, risk_a, risk_b, fn, n_boot: int, rng: np.random.Generator, level: float = 0.95):
    """fn(a) - fn(b) on the SAME resamples, so the CI reflects the paired difference of two models."""
    y, a, b = np.asarray(y), np.asarray(risk_a), np.asarray(risk_b)
    n = len(y)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = fn(y[idx], a[idx]) - fn(y[idx], b[idx])
    return percentile_ci(vals, level)


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, groups: int = 10, df_offset: int = 2):
    """Hosmer-Lemeshow chi-square over equal-count groups of predicted PD.

    ``df_offset=2`` is the classic value for a model fitted on the same data; use 0 for an
    out-of-sample (monitoring) test of a frozen model.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    stat, used = 0.0, 0
    for idx in np.array_split(order, groups):
        n = len(idx)
        if n == 0:
            continue
        e = p[idx].sum()
        o = y[idx].sum()
        pbar = e / n
        var = n * pbar * (1 - pbar)
        if var > 0:
            stat += (o - e) ** 2 / var
        used += 1
    df = max(used - df_offset, 1)
    return float(stat), float(chi2.sf(stat, df)), df


def wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    ph = k / n
    den = 1 + z**2 / n
    c = (ph + z**2 / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


def jeffreys_pvalue(k, n, pd_assigned):
    """Jeffreys test for PD under-estimation (Brown, Cai and DasGupta 2001; Tasche 2008).

    With the Jeffreys prior Beta(1/2, 1/2) the posterior of the true PD is Beta(k + 1/2, n - k + 1/2).
    The statistic is the posterior probability that the true PD is at or below the assigned PD;
    a small value says the assigned PD is too low.
    """
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    return beta.cdf(pd_assigned, k + 0.5, n - k + 0.5)


def jeffreys_interval(k, n, level: float = 0.95):
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    a = (1 - level) / 2
    lo, hi = beta.ppf(a, k + 0.5, n - k + 0.5), beta.ppf(1 - a, k + 0.5, n - k + 0.5)
    # Brown, Cai and DasGupta (2001): pin the limit at the boundary when k = 0 or k = n
    return np.where(k == 0, 0.0, lo), np.where(k == n, 1.0, hi)


def holm_adjust(p) -> np.ndarray:
    """Holm step-down adjustment of a family of p-values (controls the family-wise error rate)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj
