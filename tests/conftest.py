"""Synthetic Home Credit-shaped data so the suite runs in CI without the real dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic(n: int = 4000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ext2 = rng.uniform(0, 1, n)
    ext3 = np.where(rng.uniform(size=n) < 0.2, np.nan, rng.uniform(0, 1, n))
    income = rng.lognormal(12, 0.5, n)
    credit = income * rng.uniform(0.5, 8, n)
    employed = np.where(rng.uniform(size=n) < 0.15, 365243, -rng.integers(30, 12000, n))
    # moderate signal: keeps every IV below the 0.5 "suspicious" cap, as in the real data
    logit = -1.3 - 1.6 * ext2 - 1.4 * np.nan_to_num(ext3, nan=0.5) + 0.12 * (credit / income)
    target = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    app = pd.DataFrame({
        "SK_ID_CURR": np.arange(100000, 100000 + n),
        "TARGET": target,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
        "CODE_GENDER": rng.choice(["M", "F"], n),
        "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "State servant", "Commercial associate"], n),
        "NAME_EDUCATION_TYPE": rng.choice(["Higher education", "Secondary / secondary special"], n),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married"], n),
        "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "With parents"], n),
        "OCCUPATION_TYPE": rng.choice(["Laborers", "Core staff", None], n),
        "ORGANIZATION_TYPE": rng.choice(["Business Entity Type 3", "Self-employed", "XNA"], n),
        "FLAG_OWN_CAR": rng.choice(["Y", "N"], n),
        "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n),
        "CNT_CHILDREN": rng.integers(0, 3, n),
        "CNT_FAM_MEMBERS": rng.integers(1, 5, n).astype(float),
        "REGION_RATING_CLIENT_W_CITY": rng.integers(1, 4, n),
        "REGION_POPULATION_RELATIVE": rng.uniform(0.001, 0.07, n),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": credit * rng.uniform(0.03, 0.08, n),
        "AMT_GOODS_PRICE": credit * rng.uniform(0.8, 1.0, n),
        "DAYS_BIRTH": -rng.integers(21 * 365, 69 * 365, n),
        "DAYS_EMPLOYED": employed,
        "DAYS_ID_PUBLISH": -rng.integers(0, 7000, n),
        "DAYS_REGISTRATION": -rng.integers(0, 20000, n).astype(float),
        "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 4000, n).astype(float),
        "OWN_CAR_AGE": np.where(rng.uniform(size=n) < 0.6, np.nan, rng.integers(0, 30, n)),
        "EXT_SOURCE_1": np.where(rng.uniform(size=n) < 0.5, np.nan, rng.uniform(0, 1, n)),
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "FLAG_WORK_PHONE": rng.integers(0, 2, n),
        "FLAG_PHONE": rng.integers(0, 2, n),
        "DEF_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 3, n).astype(float),
        "AMT_REQ_CREDIT_BUREAU_YEAR": rng.integers(0, 5, n).astype(float),
    })
    rows = []
    for cid in app["SK_ID_CURR"].to_numpy()[: int(n * 0.85)]:  # 15% thin-file applicants
        for _ in range(int(rng.integers(1, 5))):
            active = rng.uniform() < 0.4
            rows.append({
                "SK_ID_CURR": cid, "SK_ID_BUREAU": len(rows),
                "CREDIT_ACTIVE": "Active" if active else "Closed", "CREDIT_CURRENCY": "currency 1",
                "DAYS_CREDIT": -int(rng.integers(1, 2900)), "CREDIT_DAY_OVERDUE": int(rng.uniform() < 0.05) * 30,
                "DAYS_CREDIT_ENDDATE": float(rng.integers(-2000, 2000)), "DAYS_ENDDATE_FACT": np.nan,
                "AMT_CREDIT_MAX_OVERDUE": np.nan, "CNT_CREDIT_PROLONG": 0,
                "AMT_CREDIT_SUM": float(rng.uniform(1e4, 1e6)), "AMT_CREDIT_SUM_DEBT": float(rng.uniform(0, 5e5)) if active else 0.0,
                "AMT_CREDIT_SUM_LIMIT": np.nan, "AMT_CREDIT_SUM_OVERDUE": 0.0,
                "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Microloan"]),
                "DAYS_CREDIT_UPDATE": -int(rng.integers(0, 100)), "AMT_ANNUITY": np.nan,
            })
    return app, pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    app, bur = make_synthetic()
    app.to_csv(d / "application_train.csv", index=False)
    bur.to_csv(d / "bureau.csv", index=False)
    return d
