"""Offline fixture: a synthetic credit dataset with the same schema as the UCI Taiwan data."""

import numpy as np
import pandas as pd
import pytest


def make_synthetic(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    limit = rng.choice([20000, 50000, 80000, 120000, 200000, 300000, 500000], n)
    age = rng.integers(21, 70, n)
    pay0 = rng.choice([-2, -1, 0, 1, 2, 3], n, p=[0.1, 0.2, 0.5, 0.1, 0.07, 0.03])
    pay2 = np.clip(pay0 + rng.integers(-1, 2, n), -2, 8)
    pay3 = np.clip(pay0 + rng.integers(-1, 2, n), -2, 8)
    bill = (limit * rng.beta(2, 3, n)).round()
    pay_amt = (bill * rng.beta(1, 4, n)).round()
    logit = -2.0 + 0.8 * np.clip(pay0, 0, 3) + 2.0 * (bill / limit) - 0.000002 * limit
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "ID": np.arange(1, n + 1),
            "LIMIT_BAL": limit,
            "AGE": age,
            "PAY_0": pay0,
            "PAY_2": pay2,
            "PAY_3": pay3,
            "BILL_AMT1": bill,
            "PAY_AMT1": pay_amt,
            "DEFAULT": y,
        }
    )


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "synthetic.csv"
    make_synthetic().to_csv(p, index=False)
    return p
