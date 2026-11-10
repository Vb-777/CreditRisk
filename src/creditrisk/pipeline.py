"""End-to-end pipeline: SQL features -> scorecard + GBM -> tests -> strategy -> reports.

    python -m creditrisk.pipeline --data-dir data --out-dir reports
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import db, metrics, stats_tests, strategy, woe
from .gbm import GBMModel
from .scorecard import Scorecard, select_features

ID, TARGET = "SK_ID_CURR", "TARGET"


def split(df: pd.DataFrame, seed: int = 42):
    """Stratified 60 / 20 / 20 train / validation / test split."""
    tr, rest = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=rest[TARGET], random_state=seed)
    return tr, va, te


def run(data_dir: str | Path, out_dir: str | Path, db_path: str | Path | None = None,
        reload: bool = False, n_boot: int = 300, seed: int = 42, make_reports: bool = True,
        log=print) -> dict:
    t0 = time.time()
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(db_path) if db_path else data_dir / "homecredit.db"

    # 1. SQL feature engineering -------------------------------------------------
    con = db.connect(db_path)
    have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if reload or not {"application", "bureau"} <= have:
        log(f"loading raw CSVs: {db.load_raw(con, data_dir)}")
    df = db.build_features(con)
    views = db.portfolio_views(con)
    con.close()
    log(f"features: {df.shape[0]:,} applications x {df.shape[1] - 2} features")

    features = [c for c in df.columns if c not in (ID, TARGET)]
    categorical = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    tr, va, te = split(df, seed)
    X_tr, y_tr = tr[features], tr[TARGET]
    X_va, y_va = va[features], va[TARGET]
    X_te, y_te = te[features], te[TARGET]

    # 2. WoE / IV + scorecard (fit on train only) --------------------------------
    binnings = woe.fit_all(X_tr, y_tr, categorical)
    iv = woe.iv_table(binnings)
    W_tr = woe.woe_frame(X_tr, binnings, iv["feature"].tolist())
    selected = select_features(iv, W_tr)
    card = Scorecard.fit(X_tr, y_tr, binnings, selected)
    pd_card = card.predict_pd(X_te)
    score_te = card.score(X_te)
    log(f"scorecard: {len(card.features)} characteristics (from {len(selected)} IV-selected)")

    # 3. Gradient-boosting challenger --------------------------------------------
    gbm = GBMModel.fit(X_tr, y_tr, X_va, y_va, categorical, seed=seed)
    pd_gbm_raw = gbm.predict_raw(X_te)
    pd_gbm = gbm.predict_pd(X_te)
    log(f"gbm: {gbm.model.n_iter_} boosting iterations")

    # 4. Evaluation on the untouched test split ----------------------------------
    results = {
        "scorecard": metrics.summary(y_te, pd_card),
        "gbm_raw": metrics.summary(y_te, pd_gbm_raw),
        "gbm_calibrated": metrics.summary(y_te, pd_gbm),
    }
    results["train_auc"] = {
        "scorecard": float(metrics.roc_auc_score(y_tr, card.predict_pd(X_tr))),
        "gbm": float(metrics.roc_auc_score(y_tr, gbm.predict_raw(X_tr))),
    }
    results["gbm_vs_scorecard"] = metrics.bootstrap_auc_diff(y_te, pd_card, pd_gbm, n_boot=n_boot, seed=seed)
    importance = gbm.importance(X_te, y_te, seed=seed)

    # 5. Hypothesis tests (whole portfolio) --------------------------------------
    tests = stats_tests.run_all(df)

    # 6. Cut-off strategy on the test split --------------------------------------
    assumptions = strategy.Assumptions()
    exposure = te["AMT_CREDIT"].to_numpy()
    cut_card = strategy.add_economics(strategy.cutoff_table(score_te, y_te, exposure), assumptions)
    cut_gbm = strategy.add_economics(strategy.cutoff_table(-pd_gbm, y_te, exposure), assumptions)
    cut_gbm["score_cutoff"] = -cut_gbm["score_cutoff"]  # ranked on -PD; report the max PD approved
    best_card, best_gbm = strategy.best_cutoff(cut_card), strategy.best_cutoff(cut_gbm)
    full = cut_card[cut_card["approval_rate"] >= 0.999].iloc[0]

    summary = {
        "data": {"applications": int(len(df)), "default_rate": float(df[TARGET].mean()),
                 "features_engineered": len(features), "train": len(tr), "validation": len(va), "test": len(te)},
        "models": results,
        "scorecard_characteristics": card.features,
        "hypothesis_tests": tests,
        "strategy": {
            "assumptions": vars(assumptions),
            "approve_all": full.to_dict(),
            "scorecard_best": best_card.to_dict(),
            "gbm_best": best_gbm.to_dict(),
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }

    # 7. Artefacts -------------------------------------------------------------------
    iv.to_csv(out_dir / "iv_table.csv", index=False)
    card.points_table().to_csv(out_dir / "scorecard_points.csv", index=False)
    importance.to_csv(out_dir / "gbm_permutation_importance.csv", index=False)
    metrics.decile_table(y_te, pd_card).to_csv(out_dir / "deciles_scorecard.csv", index=False)
    metrics.decile_table(y_te, pd_gbm).to_csv(out_dir / "deciles_gbm.csv", index=False)
    cut_card.to_csv(out_dir / "cutoff_scorecard.csv", index=False)
    cut_gbm.to_csv(out_dir / "cutoff_gbm.csv", index=False)
    sample = X_te.head(5)
    reasons = card.reason_codes(sample)
    reasons.insert(0, "score", np.round(card.score(sample), 0))
    reasons.insert(0, ID, te[ID].head(5).to_numpy())
    reasons.to_csv(out_dir / "reason_codes_sample.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    if make_reports:
        from . import reports
        scored = pd.DataFrame({ID: te[ID].to_numpy(), TARGET: y_te.to_numpy(), "score": np.round(score_te, 1),
                               "pd_scorecard": pd_card, "pd_gbm": pd_gbm, "exposure": exposure})
        scored = pd.concat([scored, te[["NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
                                        "has_bureau_history"]].reset_index(drop=True)], axis=1)
        reports.build_all(out_dir, summary=summary, iv=iv, card=card, importance=importance,
                          cut_card=cut_card, cut_gbm=cut_gbm, views=views, scored=scored,
                          y_te=y_te.to_numpy(), pd_card=pd_card, pd_gbm=pd_gbm, pd_gbm_raw=pd_gbm_raw)
    log(f"done in {time.time() - t0:.0f}s -> {out_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--db-path", default=None)
    ap.add_argument("--reload", action="store_true", help="reload raw CSVs into SQLite")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--no-reports", action="store_true")
    a = ap.parse_args()
    run(a.data_dir, a.out_dir, a.db_path, a.reload, a.n_boot, make_reports=not a.no_reports)


if __name__ == "__main__":
    main()
