import numpy as np

from model_audit.checks import ratings
from model_audit.cli import main
from model_audit.config import load_config
from model_audit.metrics import auc
from model_audit.pipeline import run_audit
from model_audit.scorecard import Scorecard
from model_audit.status import Status


def _cfg(csv, tmp_path):
    cfg = load_config(None)
    cfg["_base"] = tmp_path
    cfg["data"]["path"] = str(csv)
    cfg["scorecard"]["features"] = ["LIMIT_BAL", "PAY_0", "PAY_RATIO"]
    cfg["bootstrap"]["n_boot"] = 30
    cfg["scorecard"].update(binning="builtin", C=10.0)
    cfg["challenger"]["enabled"] = False
    cfg["output"] = {"report": str(tmp_path / "r" / "report.md"), "figures_dir": str(tmp_path / "figs")}
    return cfg


def test_scorecard_scale_and_pd_mapping(synthetic_csv):
    from model_audit.data import add_features, load_raw

    df = add_features(load_raw(synthetic_csv))
    sc = Scorecard(["LIMIT_BAL", "PAY_0", "PAY_RATIO"]).fit(df, df["DEFAULT"].to_numpy())
    s = sc.score(df)
    # base 600 @ 50:1 -> PD = 1/51; PDO 20 -> 20 points lower doubles the bad odds
    assert abs(sc.pd_from_score(np.array([600.0]))[0] - 1 / 51) < 1e-12

    def odds(x):
        p = sc.pd_from_score(np.array([x]))[0]
        return p / (1 - p)

    assert abs(odds(580) / odds(600) - 2) < 1e-9
    assert auc(df["DEFAULT"].to_numpy(), -s) > 0.6


def test_drift_injection_is_detected(synthetic_csv, tmp_path):
    out = run_audit(_cfg(synthetic_csv, tmp_path))
    byid = {r.check_id: r for r in out.results}
    assert byid["DQ-03"].status == Status.FAIL  # injected duplicate IDs
    assert byid["DQ-02"].status == Status.FAIL  # injected AGE sentinel
    assert byid["ST-02:PAY_RATIO"].status != Status.PASS  # injected repayment drift
    assert all(r.rationale and r.threshold for r in out.results)
    assert {"CA-02", "CA-05"} <= set(byid)  # binomial and Jeffreys tests both reported


def test_no_drift_is_quiet(synthetic_csv, tmp_path):
    cfg = _cfg(synthetic_csv, tmp_path)
    cfg["drift"]["enabled"] = False
    out = run_audit(cfg)
    byid = {r.check_id: r for r in out.results}
    assert byid["DQ-03"].status == Status.PASS
    assert byid["ST-01"].status == Status.PASS


def test_grades_hhi_and_inversions():
    cuts = np.array([10.0, 20.0, 30.0])
    g = ratings.assign_grades(np.array([5.0, 15.0, 25.0, 35.0]), cuts)
    assert g.tolist() == [4, 3, 2, 1]  # lowest score = worst grade
    assert ratings.hhi(np.array([25, 25, 25, 25])) == 0.25
    assert ratings.inversions(np.array([0.05, 0.04, 0.10, 0.20])) == 1


def test_cli_run_writes_report_and_figures(synthetic_csv, tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"data:\n  path: {synthetic_csv}\n"
        "scorecard:\n  features: [LIMIT_BAL, PAY_0, PAY_RATIO]\n"
        "  binning: builtin\n  C: 10\n"
        "challenger:\n  enabled: false\n"
        "bootstrap:\n  n_boot: 20\n"
        "output:\n  report: out/report.md\n  figures_dir: out/figs\n"
    )
    assert main(["run", "--config", str(cfg_path)]) == 0
    text = (tmp_path / "out" / "report.md").read_text()
    assert "## 3. Findings" in text
    assert "Observation" in text and "Recommendation" in text
    assert "Overall rating" in text and "Jeffreys" in text
    for name in ("psi_bars", "roc_ks", "calibration", "traffic_light"):
        assert (tmp_path / "out" / "figs" / f"{name}.png").stat().st_size > 1000


def test_id_order_split_is_an_ordered_holdout(synthetic_csv, tmp_path):
    from model_audit.data import load_raw
    from model_audit.simulate import split_dev_monitoring

    cfg = _cfg(synthetic_csv, tmp_path)
    cfg["split"]["mode"] = "id_order"
    dev, mon = split_dev_monitoring(load_raw(synthetic_csv), cfg, np.random.default_rng(0))
    assert dev["ID"].max() < mon["ID"].min()
    assert len(dev) == len(mon) == 3000
