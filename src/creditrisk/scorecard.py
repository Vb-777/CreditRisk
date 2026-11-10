"""Logistic-regression scorecard on WoE features, scaled to points.

Scaling: a score of ``base_score`` means good:bad odds of ``base_odds``,
and every ``pdo`` points doubles the odds. Each characteristic contributes
points per bin, so a decision can be explained line by line, and the
characteristics that cost an applicant the most points become reason codes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .woe import Binning, woe_frame


@dataclass
class ScoreScale:
    base_score: float = 600.0
    base_odds: float = 50.0  # good:bad odds at the base score
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def score_from_pd(self, pd_: np.ndarray) -> np.ndarray:
        pd_ = np.clip(pd_, 1e-9, 1 - 1e-9)
        return self.offset + self.factor * np.log((1 - pd_) / pd_)

    def pd_from_score(self, score: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp((np.asarray(score) - self.offset) / self.factor))


def select_features(iv: pd.DataFrame, W: pd.DataFrame, min_iv: float = 0.02,
                    max_iv: float = 0.5, max_corr: float = 0.7) -> list[str]:
    """Keep IV in [min_iv, max_iv); drop the weaker of any pair with |r| > max_corr."""
    cands = iv[(iv["iv"] >= min_iv) & (iv["iv"] < max_iv)]["feature"].tolist()
    corr = W[cands].corr().abs()
    kept: list[str] = []
    for f in cands:  # cands are already sorted by IV, strongest first
        if all(corr.loc[f, k] <= max_corr for k in kept):
            kept.append(f)
    return kept


@dataclass
class Scorecard:
    binnings: dict[str, Binning]
    features: list[str]
    model: LogisticRegression
    scale: ScoreScale

    @classmethod
    def fit(cls, X: pd.DataFrame, y: pd.Series, binnings: dict[str, Binning], features: list[str],
            scale: ScoreScale | None = None, C: float = 1.0) -> "Scorecard":
        feats = list(features)
        while True:
            W = woe_frame(X, binnings, feats)
            m = LogisticRegression(C=C, max_iter=2000).fit(W, y)
            # With WoE = ln(good/bad), every coefficient must be negative (higher WoE -> lower PD).
            wrong = [f for f, c in zip(feats, m.coef_[0], strict=True) if c >= 0]
            if not wrong:
                break
            feats.remove(wrong[0])
        return cls(binnings, feats, m, scale or ScoreScale())

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(woe_frame(X, self.binnings, self.features))[:, 1]

    def points_table(self) -> pd.DataFrame:
        """Points for every bin of every characteristic (sums to the score)."""
        n = len(self.features)
        b0 = self.model.intercept_[0]
        rows = []
        for f, b in zip(self.features, self.model.coef_[0], strict=True):
            for _, r in self.binnings[f].table.iterrows():
                pts = self.scale.offset / n - self.scale.factor * (b * r["woe"] + b0 / n)
                rows.append({"characteristic": f, "bin": r["bin"], "share": r["share"],
                             "bad_rate": r["bad_rate"], "woe": r["woe"], "points": pts})
        t = pd.DataFrame(rows)
        t["points"] = t["points"].round(1)
        return t

    def score_components(self, X: pd.DataFrame) -> pd.DataFrame:
        pts = self.points_table()
        out = {}
        for f in self.features:
            m = pts[pts["characteristic"] == f].set_index("bin")["points"]
            neutral = self.scale.offset / len(self.features) - self.scale.factor * self.model.intercept_[0] / len(self.features)
            out[f] = self.binnings[f].labels(X[f]).map(m).fillna(round(neutral, 1)).astype(float)
        return pd.DataFrame(out, index=X.index)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.score_components(X).sum(axis=1).to_numpy()

    def reason_codes(self, X: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        """Top-k characteristics where the applicant lost the most points vs the best bin."""
        comp = self.score_components(X)
        best = self.points_table().groupby("characteristic")["points"].max()
        shortfall = best[comp.columns] - comp
        order = np.argsort(-shortfall.to_numpy(), axis=1)[:, :k]
        cols = np.asarray(comp.columns)
        return pd.DataFrame({f"reason_{i + 1}": cols[order[:, i]] for i in range(k)}, index=X.index)
