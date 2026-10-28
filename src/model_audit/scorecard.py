"""Reference scorecard: monotone WoE binning + logistic regression -> points scale (PDO).

Convention: WoE = ln(%good / %bad); score = offset + factor * ln(odds good:bad);
factor = PDO / ln 2. Higher score = lower risk.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass
class Binning:
    feature: str
    edges: list[float]  # interior upper edges, bin k = (edge[k-1], edge[k]]
    woe: list[float]  # len(edges)+1 regular bins, then one trailing missing bin
    points: list[int] | None = None

    @property
    def n_bins(self) -> int:
        return len(self.edges) + 2  # regular bins + missing bin

    def index(self, x: pd.Series) -> np.ndarray:
        idx = np.searchsorted(self.edges, x.to_numpy(dtype=float), side="left")
        return np.where(x.isna().to_numpy(), len(self.edges) + 1, idx)

    def labels(self) -> list[str]:
        e = [-np.inf, *self.edges, np.inf]
        lab = [f"({e[i]:g}, {e[i + 1]:g}]" for i in range(len(e) - 1)]
        return [*lab, "Missing"]


def _woe(good: np.ndarray, bad: np.ndarray) -> np.ndarray:
    g = (good + 0.5) / (good.sum() + 0.5 * len(good))
    b = (bad + 0.5) / (bad.sum() + 0.5 * len(bad))
    return np.log(g / b)


def _bin_stats(xv: np.ndarray, yv: np.ndarray, edges) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(edges, xv, side="left")
    n = np.bincount(idx, minlength=len(edges) + 1)
    bad = np.bincount(idx, weights=yv, minlength=len(edges) + 1)
    return n, bad


def _optbinning_edges(xv: np.ndarray, yv: np.ndarray, name: str, max_bins: int, min_share: float):
    """Interior cut points from an exact optimal-binning solve with a monotone event rate."""
    try:
        from optbinning import OptimalBinning
    except ImportError as exc:  # pragma: no cover
        raise ImportError("binning='optbinning' needs the extra: pip install '.[model]'") from exc
    ob = OptimalBinning(
        name=name,
        dtype="numerical",
        solver="cp",
        monotonic_trend="auto_asc_desc",
        min_bin_size=min_share,
        max_n_bins=max_bins,
    )
    ob.fit(xv, yv)
    if ob.status not in ("OPTIMAL", "FEASIBLE"):
        raise ValueError(f"optimal binning of {name} returned status {ob.status}")
    # optbinning cuts at midpoints between observed values, so side="left" vs "right" is moot
    return [float(e) for e in ob.splits]


def fit_binning(
    x: pd.Series, y: np.ndarray, max_bins: int, min_share: float, method: str = "builtin"
) -> Binning:
    """Monotone WoE binning with a separate missing bin.

    ``builtin``: quantile pre-bins merged until every bin is large enough and the bad rate is
    monotone. ``optbinning``: exact optimal binning (Navas-Suarez and Brunel), monotone trend
    chosen from the data. WoE itself is always computed here, so both methods share one convention.
    """
    mask = x.notna().to_numpy()
    xv, yv = x.to_numpy(dtype=float)[mask], y[mask]
    if method == "optbinning":
        edges = _optbinning_edges(xv, yv, str(x.name), max_bins, min_share)
        return _finish_binning(x, y, mask, edges)
    if method != "builtin":
        raise ValueError(f"unknown binning method {method!r}")
    qs = np.unique(np.quantile(xv, np.linspace(0, 1, 21)[1:-1]))
    edges = [float(e) for e in qs if e < xv.max()]

    def stats(ed):
        return _bin_stats(xv, yv, ed)

    while len(edges) > 0:
        n, bad = stats(edges)
        rate = bad / np.maximum(n, 1)
        small = np.where(n < min_share * len(xv))[0]
        if len(small):
            k = int(small[np.argmin(n[small])])
        elif len(edges) + 1 > max_bins:
            k = int(np.argmin(np.abs(np.diff(rate)))) if len(rate) > 1 else 0
        else:
            rho = spearmanr(np.arange(len(rate)), rate)[0] if len(rate) > 2 else 1.0
            sign = 1 if np.nan_to_num(rho) >= 0 else -1
            viol = np.where(sign * np.diff(rate) < 0)[0]
            if not len(viol):
                break
            k = int(viol[0])
        # merge bin k with its closest neighbour by bad rate (drop the shared edge)
        if k == 0:
            j = 0
        elif k == len(edges):
            j = k - 1
        else:
            j = k - 1 if abs(rate[k] - rate[k - 1]) <= abs(rate[k] - rate[k + 1]) else k
        edges.pop(j)
    return _finish_binning(x, y, mask, edges)


def _finish_binning(x: pd.Series, y: np.ndarray, mask: np.ndarray, edges: list[float]) -> Binning:
    n, bad = _bin_stats(x.to_numpy(dtype=float)[mask], y[mask], edges)
    woe = list(_woe(n - bad, bad))
    n_miss = int((~mask).sum())
    # a missing bin that is empty in development gets neutral WoE (no evidence either way)
    if n_miss:
        g = (n_miss - y[~mask].sum() + 0.5) / (len(y) - y.sum() + 0.5 * (len(edges) + 2))
        b = (y[~mask].sum() + 0.5) / (y.sum() + 0.5 * (len(edges) + 2))
        woe.append(float(np.log(g / b)))
    else:
        woe.append(0.0)
    return Binning(x.name, edges, [float(w) for w in woe])


class Scorecard:
    """WoE scorecard. ``C=None`` tunes the logistic regularisation by cross-validated log loss."""

    def __init__(
        self,
        features,
        base_score=600,
        base_odds=50,
        pdo=20,
        max_bins=6,
        min_bin_share=0.05,
        C: float | None = 10.0,
        binning: str = "builtin",
        cv_folds: int = 5,
        seed: int = 42,
        l1_ratio: float = 0.0,
    ):
        self.features = list(features)
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
        self.max_bins, self.min_bin_share, self.C = max_bins, min_bin_share, C
        self.binning, self.cv_folds, self.seed, self.l1_ratio = binning, cv_folds, seed, l1_ratio
        self.bins: dict[str, Binning] = {}
        self.coef_: dict[str, float] = {}
        self.intercept_ = 0.0
        self.tuning_: dict = {}
        self.dropped_: list[str] = []
        self.zeroed_: list[str] = []
        self.pd_map_: tuple[float, ...] | None = None  # intercept and cubic coefficients
        self.pd_scale_: tuple[float, float] = (0.0, 1.0)

    def _woe_matrix(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for f in self.features:
            b = self.bins[f]
            cols.append(np.asarray(b.woe)[b.index(df[f])])
        return np.column_stack(cols)

    def _tune(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        """Pick C and the L1/L2 mix by 5-fold CV log loss; the model is then refit on all rows."""
        grid = [
            (c, r)
            for r in (0.0, 0.5, 1.0)  # ridge, elastic net, lasso
            for c in np.logspace(-3, 2, 11)
        ]
        cv = StratifiedKFold(self.cv_folds, shuffle=True, random_state=self.seed)
        best = None
        for c, r in grid:
            lr = LogisticRegression(C=c, l1_ratio=r, solver="saga", max_iter=5000, tol=1e-6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # saga convergence chatter on near-collinear WoE columns
                ll = -cross_val_score(lr, X, y, cv=cv, scoring="neg_log_loss").mean()
            if best is None or ll < best[0] - 1e-9:
                best = (ll, c, r)
        ll, c, r = best
        self.tuning_ = {"C": float(c), "l1_ratio": float(r), "cv_log_loss": float(ll)}
        return LogisticRegression(C=c, l1_ratio=r, solver="saga", max_iter=20000, tol=1e-8)

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> Scorecard:
        for f in self.features:
            self.bins[f] = fit_binning(df[f], y, self.max_bins, self.min_bin_share, self.binning)
        X = self._woe_matrix(df)
        if self.C is None:
            lr = self._tune(X, y)
        elif self.l1_ratio == 0.0:
            lr = LogisticRegression(C=self.C, max_iter=1000)
        else:
            lr = LogisticRegression(C=self.C, l1_ratio=self.l1_ratio, solver="saga", max_iter=20000, tol=1e-8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(X, y)
        self.intercept_ = float(lr.intercept_[0])
        # an L1 penalty can zero a characteristic out; it then carries no points and leaves the card
        coefs = dict(zip(self.features, lr.coef_[0], strict=True))
        self.zeroed_ += [f for f, c in coefs.items() if abs(c) < 1e-9]
        self.features = [f for f in self.features if abs(coefs[f]) >= 1e-9]
        n = len(self.features)
        for f in self.features:
            c = coefs[f]
            self.coef_[f] = float(c)
            b = self.bins[f]
            b.points = [
                int(round(-(c * w + self.intercept_ / n) * self.factor + self.offset / n)) for w in b.woe
            ]
        # WoE is oriented so that a sound characteristic has a negative coefficient. A positive one
        # means the variable works against its own binning once the others are in the model:
        # drop the worst offender, record it, and refit.
        wrong = {f: c for f, c in self.coef_.items() if c > 1e-9}
        if wrong:
            worst = max(wrong, key=wrong.get)
            if len(self.features) == 1:
                raise ValueError(f"wrong-sign coefficient for {worst} and no features left")
            self.dropped_.append(worst)
            self.features.remove(worst)
            self.coef_ = {}
            return self.fit(df, y)
        return self

    def bin_index(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f: self.bins[f].index(df[f]) for f in self.features}, index=df.index)

    def points(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = self.bin_index(df)
        return pd.DataFrame(
            {f: np.asarray(self.bins[f].points)[idx[f]] for f in self.features}, index=df.index
        )

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.points(df).sum(axis=1).to_numpy(dtype=float)

    def _z(self, score) -> np.ndarray:
        m, sd = self.pd_scale_
        z = (np.asarray(score, dtype=float) - m) / sd
        return np.column_stack([z, z**2, z**3])

    def calibrate_pd(self, score: np.ndarray, y: np.ndarray) -> Scorecard:
        """Fit logit(PD) = a + b1 z + b2 z^2 + b3 z^3, z = standardised score, on development data.

        The points scale fixes the odds at the base score, but a regularised model shrinks the
        slope and an additive WoE model is not calibrated across the whole range, so the
        scale-implied PD misses in the tails (it fails Hosmer-Lemeshow in development). Flexible
        recalibration of the score-to-PD map is the usual last step of a scorecard build; the
        calibration checks then test the frozen map out of time. The cubic is unpenalised and
        keeps the mean PD equal to the development default rate.
        """
        score = np.asarray(score, dtype=float)
        self.pd_scale_ = (float(score.mean()), float(score.std()))
        lr = LogisticRegression(C=1e12, max_iter=5000).fit(self._z(score), y)
        self.pd_map_ = (float(lr.intercept_[0]), *map(float, lr.coef_[0]))
        return self

    def pd_from_score(self, score: np.ndarray) -> np.ndarray:
        """PD for a score: the calibrated map if ``calibrate_pd`` was called, otherwise the
        scale-implied PD, logit(PD) = (offset - score) / factor."""
        score = np.asarray(score, dtype=float)
        if self.pd_map_ is None:
            return 1.0 / (1.0 + np.exp(-(self.offset - score) / self.factor))
        a, *b = self.pd_map_
        return 1.0 / (1.0 + np.exp(-(a + self._z(score) @ np.array(b))))

    def table(self) -> pd.DataFrame:
        rows = []
        for f in self.features:
            b = self.bins[f]
            for lab, w, p in zip(b.labels(), b.woe, b.points, strict=True):
                rows.append({"feature": f, "bin": lab, "woe": round(w, 3), "points": p})
        return pd.DataFrame(rows)
