"""Challenger model for the champion-challenger comparison: Optuna-tuned LightGBM, 5-fold CV.

The challenger sees every raw predictor except SEX and MARRIAGE (protected attributes, not used
by the champion either) plus the derived repayment ratio. It is a validation yardstick: if a
flexible model finds much more signal in the same data, the scorecard is leaving power on the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .metrics import auc, gini, ks_statistic
from .scorecard import Scorecard

EXCLUDE = {"ID", "SEX", "MARRIAGE", "DEFAULT"}


def challenger_features(df: pd.DataFrame, target: str, id_col: str) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE | {target, id_col}]


def _lgbm(params: dict, seed: int):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="binary",
        subsample_freq=1,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbose=-1,
        **params,
    )


def _suggest(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 48),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


@dataclass
class Challenger:
    features: list[str]
    params: dict
    model: object
    cv: pd.DataFrame  # one row per fold: champion and challenger discrimination out of fold
    study_trials: int
    best_logloss: float
    extras: dict = field(default_factory=dict)

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.features])[:, 1]


def fit_challenger(
    dev: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    champion_factory,
    n_trials: int = 40,
    n_splits: int = 5,
    seed: int = 42,
) -> Challenger:
    """Tune on 5-fold CV log loss, refit on all development rows, then compare with the champion.

    ``champion_factory()`` returns an unfitted Scorecard with the tuned hyper-parameters; the
    champion is re-binned and refitted inside each fold, so both models are scored out of fold on
    identical splits. The folds used for that comparison differ from the tuning folds (seed + 1);
    this reduces, but does not remove, the optimism from tuning on the same rows.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X = dev[features]
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y))

    def objective(trial):
        p = _suggest(trial)
        losses = []
        for tr, va in folds:
            m = _lgbm(p, seed).fit(X.iloc[tr], y[tr])
            q = np.clip(m.predict_proba(X.iloc[va])[:, 1], 1e-6, 1 - 1e-6)
            losses.append(-np.mean(y[va] * np.log(q) + (1 - y[va]) * np.log(1 - q)))
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    params = study.best_params
    model = _lgbm(params, seed).fit(X, y)

    rows = []
    for i, (tr, va) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed + 1).split(X, y)):
        ch = _lgbm(params, seed).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
        sc: Scorecard = champion_factory().fit(dev.iloc[tr].reset_index(drop=True), y[tr])
        cs = -sc.score(dev.iloc[va])
        rows.append(
            {
                "fold": i + 1,
                "champion_gini": gini(y[va], cs),
                "challenger_gini": gini(y[va], ch),
                "champion_auc": auc(y[va], cs),
                "challenger_auc": auc(y[va], ch),
                "champion_ks": ks_statistic(y[va], cs),
                "challenger_ks": ks_statistic(y[va], ch),
            }
        )
    return Challenger(features, params, model, pd.DataFrame(rows), n_trials, float(study.best_value))
