"""Gradient-boosting challenger model with isotonic calibration.

Uses scikit-learn's HistGradientBoostingClassifier: histogram-based boosting
of the same family as LightGBM, with native handling of missing values and
categorical columns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression


def to_model_frame(X: pd.DataFrame, categorical: list[str], categories: dict[str, list] | None = None):
    """Cast categoricals to a fixed category set so train/valid/test encode identically."""
    X = X.copy()
    cats = categories or {c: sorted(X[c].dropna().astype(str).unique()) for c in categorical}
    for c in categorical:
        X[c] = pd.Categorical(X[c].astype(object).where(X[c].notna(), None), categories=cats[c])
    return X, cats


@dataclass
class GBMModel:
    model: HistGradientBoostingClassifier
    calibrator: IsotonicRegression
    categorical: list[str]
    categories: dict[str, list]

    @classmethod
    def fit(cls, X_tr, y_tr, X_val, y_val, categorical: list[str], seed: int = 42) -> "GBMModel":
        Xt, cats = to_model_frame(X_tr, categorical)
        Xv, _ = to_model_frame(X_val, categorical, cats)
        m = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=2000, max_leaf_nodes=31, min_samples_leaf=200,
            l2_regularization=1.0, categorical_features="from_dtype",
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
            scoring="roc_auc", random_state=seed,
        ).fit(Xt, y_tr)
        # Calibrate on the held-out validation split, never on training rows.
        raw_val = m.predict_proba(Xv)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(raw_val, y_val)
        return cls(m, iso, categorical, cats)

    def frame(self, X):
        return to_model_frame(X, self.categorical, self.categories)[0]

    def predict_raw(self, X) -> np.ndarray:
        return self.model.predict_proba(self.frame(X))[:, 1]

    def predict_pd(self, X) -> np.ndarray:
        return self.calibrator.predict(self.predict_raw(X))

    def importance(self, X, y, n_rows: int = 30_000, n_repeats: int = 3, seed: int = 0) -> pd.DataFrame:
        """Permutation importance: drop in test AUC when one feature is shuffled."""
        Xf = self.frame(X)
        if len(Xf) > n_rows:
            idx = np.random.default_rng(seed).choice(len(Xf), n_rows, replace=False)
            Xf, y = Xf.iloc[idx], np.asarray(y)[idx]
        r = permutation_importance(self.model, Xf, y, scoring="roc_auc",
                                   n_repeats=n_repeats, random_state=seed, n_jobs=1)
        return (pd.DataFrame({"feature": Xf.columns, "auc_drop": r.importances_mean, "std": r.importances_std})
                .sort_values("auc_drop", ascending=False).reset_index(drop=True))
