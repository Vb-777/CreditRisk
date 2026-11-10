import json
import sqlite3

import pandas as pd
from openpyxl import load_workbook

from creditrisk import db, pipeline


def test_sql_features_on_synthetic(synthetic_dir, tmp_path):
    con = db.connect(tmp_path / "t.db")
    db.load_raw(con, synthetic_dir)
    f = db.build_features(con)
    app = pd.read_csv(synthetic_dir / "application_train.csv")
    bur = pd.read_csv(synthetic_dir / "bureau.csv")

    assert len(f) == len(app) and f["SK_ID_CURR"].is_unique  # LEFT JOIN keeps one row per application
    assert "CODE_GENDER" not in f.columns
    # thin-file applicants have no bureau rows
    assert set(f.loc[f.has_bureau_history == 0, "SK_ID_CURR"]).isdisjoint(set(bur["SK_ID_CURR"]))
    # bureau aggregates agree with pandas
    exp = bur.groupby("SK_ID_CURR").size()
    got = f.set_index("SK_ID_CURR")["bur_n_loans"].dropna()
    assert (got == exp.loc[got.index]).all()
    # DAYS_EMPLOYED placeholder becomes NULL + flag
    anomaly = app.set_index("SK_ID_CURR")["DAYS_EMPLOYED"] == 365243
    ff = f.set_index("SK_ID_CURR")
    assert ff.loc[anomaly[anomaly].index, "employed_years"].isna().all()
    assert (ff.loc[anomaly[anomaly].index, "flag_not_employed"] == 1).all()
    # ratios are true divisions, not integer division
    assert (f["credit_to_income"] - app["AMT_CREDIT"] / app["AMT_INCOME_TOTAL"]).abs().max() < 1e-9


def test_named_queries_parse():
    q = db.named_queries()
    assert {"by_bureau_history", "by_income_type", "by_education"} <= set(q)
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE model_features (has_bureau_history INT, TARGET INT, NAME_INCOME_TYPE TEXT,"
                " NAME_EDUCATION_TYPE TEXT, NAME_CONTRACT_TYPE TEXT, credit_to_income REAL)")
    for sql in q.values():
        con.execute(sql).fetchall()


def test_end_to_end_pipeline(synthetic_dir, tmp_path):
    out = tmp_path / "reports"
    s = pipeline.run(synthetic_dir, out, db_path=tmp_path / "e2e.db", n_boot=20, log=lambda *_: None)
    assert s["models"]["scorecard"]["auc"] > 0.65  # synthetic data carries real signal
    assert s["models"]["gbm_calibrated"]["auc"] > 0.65
    best = s["strategy"]["scorecard_best"]
    assert best["profit"] >= s["strategy"]["approve_all"]["profit"]
    for f in ("summary.json", "iv_table.csv", "scorecard_points.csv", "cutoff_strategy_model.xlsx",
              "figures/roc.png", "figures/cutoff_tradeoff.png", "powerbi/cutoff_strategy.csv", "roc_points.csv"):
        assert (out / f).exists(), f
    json.loads((out / "summary.json").read_text())
    wb = load_workbook(out / "cutoff_strategy_model.xlsx")
    assert {"Inputs", "Scorecard cut-offs", "GBM cut-offs"} <= set(wb.sheetnames)
    assert str(wb["Scorecard cut-offs"]["J5"].value).startswith("=")  # profit is a live formula
