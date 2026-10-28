import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("optuna")

from model_audit.config import load_config  # noqa: E402
from model_audit.pipeline import run_audit  # noqa: E402
from model_audit.status import Status  # noqa: E402


def test_challenger_benchmark_runs_and_reports_cv_folds(synthetic_csv, tmp_path):
    cfg = load_config(None)
    cfg["_base"] = tmp_path
    cfg["data"]["path"] = str(synthetic_csv)
    cfg["scorecard"].update(features=["LIMIT_BAL", "PAY_0", "PAY_RATIO"], binning="builtin", C=10.0)
    cfg["bootstrap"]["n_boot"] = 20
    cfg["challenger"].update(enabled=True, n_trials=2, n_splits=3)
    out = run_audit(cfg)
    byid = {r.check_id: r for r in out.results}
    assert {"BM-01", "BM-02"} <= set(byid)
    assert all(isinstance(byid[k].status, Status) for k in ("BM-01", "BM-02"))
    assert len(out.tables["challenger_cv"]) == 3
    lo, hi = byid["BM-01"].details["ci"]
    assert lo <= byid["BM-01"].value <= hi or abs(byid["BM-01"].value - (lo + hi) / 2) < 0.2
    assert set(out.tables["benchmark"]["model"]) == {"Champion scorecard", "Challenger LightGBM"}
