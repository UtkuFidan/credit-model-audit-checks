"""Download and cache the public dataset; load it and derive the repayment ratio."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
TARGET_SRC = "default payment next month"


def download_taiwan(dest: str | Path) -> Path:
    """Download UCI 'Default of Credit Card Clients' (id 350, CC BY 4.0) and cache it as CSV."""
    try:
        import xlrd  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Reading the UCI .xls needs xlrd: pip install '.[data]'") from exc
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UCI_URL, timeout=120) as r:  # noqa: S310 (fixed https URL)
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".xls"))
        df = pd.read_excel(io.BytesIO(zf.read(name)), header=1, engine="xlrd")
    df = df.rename(columns={TARGET_SRC: "DEFAULT"})
    df.to_csv(dest, index=False)
    return dest


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `model-audit download` (public UCI data, not shipped in the repo)."
        )
    return pd.read_csv(path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived characteristics; NaN inputs stay NaN.

    PAY_RATIO: latest payment / latest bill. UTIL_AVG: mean six-month bill / limit. PAY6_RATIO:
    six-month payments / six-month positive bills. PAY_AMT_LIM: mean payment / limit.
    BILL_TREND: (latest minus oldest bill) / limit.
    """
    out = df.copy()
    bill = out["BILL_AMT1"]
    ratio = (out["PAY_AMT1"] / bill.where(bill > 0)).clip(0, 1.0)
    # nothing outstanding counts as fully repaid; NaN stays NaN where the payment is missing
    out["PAY_RATIO"] = ratio.where(bill > 0, np.where(out["PAY_AMT1"].isna(), np.nan, 1.0))
    bills, pays = [f"BILL_AMT{i}" for i in range(1, 7)], [f"PAY_AMT{i}" for i in range(1, 7)]
    if set(bills + pays) <= set(out.columns):  # full six-month history (absent in the test fixture)
        limit = out["LIMIT_BAL"]
        out["UTIL_AVG"] = (out[bills].mean(axis=1) / limit).clip(0, 1.5)
        paid = out[pays].sum(axis=1)
        billed = out[bills].clip(lower=0).sum(axis=1)
        ratio6 = (paid / billed.where(billed > 0)).clip(0, 1)
        out["PAY6_RATIO"] = ratio6.where(billed > 0, 1.0)  # nothing billed counts as fully repaid
        out["PAY_AMT_LIM"] = (out[pays].mean(axis=1) / limit).clip(0, 1)
        out["BILL_TREND"] = ((out["BILL_AMT1"] - out["BILL_AMT6"]) / limit).clip(-1, 1)
    return out
