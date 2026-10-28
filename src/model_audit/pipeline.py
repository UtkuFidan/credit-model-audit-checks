"""End-to-end audit: data -> reference scorecard -> simulated monitoring sample -> checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data as data_mod
from . import simulate
from .challenger import challenger_features, fit_challenger
from .checks import benchmark, calibration, discrimination, overrides, quality, ratings, stability
from .config import resolve
from .scorecard import Scorecard
from .status import CheckResult


@dataclass
class AuditOutput:
    results: list[CheckResult]
    tables: dict[str, pd.DataFrame]
    notes: list[str]
    scorecard: Scorecard
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def run_audit(cfg: dict) -> AuditOutput:
    rng = np.random.default_rng(cfg["seed"])
    tgt = cfg["data"]["target_col"]
    raw = data_mod.load_raw(resolve(cfg, cfg["data"]["path"]))

    dev, mon = simulate.split_dev_monitoring(raw, cfg, rng)
    notes: list[str] = []
    if cfg["drift"]["enabled"]:
        mon, notes = simulate.inject_drift(mon, cfg, rng)
    dev, mon = data_mod.add_features(dev), data_mod.add_features(mon)

    sc_cfg = cfg["scorecard"]

    def make_scorecard(**over) -> Scorecard:
        kw = {
            "features": sc_cfg["features"],
            "base_score": sc_cfg["base_score"],
            "base_odds": sc_cfg["base_odds"],
            "pdo": sc_cfg["pdo"],
            "max_bins": sc_cfg["max_bins"],
            "min_bin_share": sc_cfg["min_bin_share"],
            "C": sc_cfg["C"],
            "binning": sc_cfg["binning"],
            "cv_folds": sc_cfg["cv_folds"],
            "seed": cfg["seed"],
        }
        return Scorecard(**{**kw, **over})

    sc = make_scorecard()
    y_dev = dev[tgt].to_numpy(dtype=int)
    sc.fit(dev, y_dev)
    s_dev, s_mon = sc.score(dev), sc.score(mon)
    if sc_cfg["calibrate_pd"]:
        sc.calibrate_pd(s_dev, y_dev)
    pd_dev, pd_mon = sc.pd_from_score(s_dev), sc.pd_from_score(s_mon)
    y_mon = mon[tgt].to_numpy(dtype=int)

    cuts = np.unique(np.percentile(s_dev, cfg["ratings"]["cut_percentiles"]))
    g_dev, g_mon = ratings.assign_grades(s_dev, cuts), ratings.assign_grades(s_mon, cuts)
    n_grades = len(cuts) + 1
    grade_pd = {g: float(pd_dev[g_dev == g].mean()) for g in range(1, n_grades + 1) if (g_dev == g).any()}

    results: list[CheckResult] = []
    tables: dict[str, pd.DataFrame] = {"scorecard": sc.table()}

    def add(out):
        r, t = out
        results.extend(r)
        tables.update(t)

    add(quality.run(dev, mon, cfg))
    add(stability.run(sc, dev, mon, s_dev, s_mon, cfg))
    add(discrimination.run(y_dev, y_mon, s_dev, s_mon, cfg, rng))
    results.extend(calibration.run(y_dev, y_mon, pd_dev, pd_mon, g_mon, grade_pd, cfg))
    cuts_b = stability.score_bands(s_dev)
    tables["calibration_bands"] = calibration.band_table(y_mon, pd_mon, s_mon, cuts_b)
    add(ratings.run(g_dev, g_mon, y_dev, y_mon, n_grades, cfg))
    if cfg["overrides"]["enabled"]:
        cutoff = float(np.percentile(s_dev, cfg["overrides"]["cutoff_percentile"]))
        flag = simulate.simulate_overrides(mon, s_mon, cutoff, cfg, rng)
        add(overrides.run(y_mon, pd_mon, s_mon, flag, cutoff, cfg))

    ch_cfg = cfg["challenger"]
    if ch_cfg["enabled"]:
        feats = challenger_features(dev, tgt, cfg["data"]["id_col"])
        tuned = {"C": sc.tuning_.get("C", sc_cfg["C"]), "l1_ratio": sc.tuning_.get("l1_ratio", 0.0)}
        ch = fit_challenger(
            dev,
            y_dev,
            feats,
            lambda: make_scorecard(features=sc_cfg["features"], **tuned),
            ch_cfg["n_trials"],
            ch_cfg["n_splits"],
            cfg["seed"],
        )
        add(benchmark.run(ch, y_mon, s_mon, ch.predict_pd(mon), pd_mon, cfg, rng))
        tables["challenger_params"] = pd.DataFrame(
            [{"parameter": k, "value": v} for k, v in {**ch.params, "features": len(feats)}.items()]
        )

    meta = {
        "n_dev": len(dev),
        "n_mon": len(mon),
        "dr_dev": float(y_dev.mean()),
        "dr_mon": float(y_mon.mean()),
        "pd_dev": float(pd_dev.mean()),
        "pd_mon": float(pd_mon.mean()),
        "base_score": sc_cfg["base_score"],
        "base_odds": sc_cfg["base_odds"],
        "pdo": sc_cfg["pdo"],
        "seed": cfg["seed"],
        "coef": sc.coef_,
        "dropped": sc.dropped_,
        "zeroed": sc.zeroed_,
        "tuning": sc.tuning_,
        "pd_map": sc.pd_map_,
        "binning": sc_cfg["binning"],
        "challenger": ch_cfg["enabled"],
    }
    arrays = {"y_dev": y_dev, "y_mon": y_mon, "s_dev": s_dev, "s_mon": s_mon}
    return AuditOutput(results, tables, notes, sc, arrays, meta)
