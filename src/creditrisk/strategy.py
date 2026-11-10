"""Cut-off strategy: approval rate vs bad rate, expected loss and profit.

Applicants are ranked by score (best first) and approved down to a cut-off.
Loss and revenue use simple, editable assumptions (the Excel model exposes
them as input cells):

    loss    = LGD x exposure of approved loans that default
    revenue = net margin x exposure of approved loans that repay
    profit  = revenue - loss
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Assumptions:
    lgd: float = 0.60         # loss given default, unsecured consumer credit
    net_margin: float = 0.08  # lifetime net interest + fees as share of exposure, after funding cost


def cutoff_table(score: np.ndarray, y: np.ndarray, exposure: np.ndarray,
                 steps: np.ndarray | None = None) -> pd.DataFrame:
    """Aggregates for approving the top x% of applicants by score."""
    steps = np.arange(0.05, 1.0001, 0.05) if steps is None else steps
    order = np.argsort(-np.asarray(score), kind="stable")
    s, yy, ex = np.asarray(score)[order], np.asarray(y, float)[order], np.asarray(exposure, float)[order]
    n = len(s)
    rows = []
    for a in steps:
        k = int(round(a * n))
        rows.append({
            "approval_rate": float(a),
            "score_cutoff": float(s[k - 1]),
            "approved": k,
            "approved_bads": float(yy[:k].sum()),
            "exposure_approved": float(ex[:k].sum()),
            "exposure_bad": float((ex[:k] * yy[:k]).sum()),
        })
    t = pd.DataFrame(rows)
    t["bad_rate"] = t["approved_bads"] / t["approved"]
    t["bads_rejected_share"] = 1 - t["approved_bads"] / yy.sum()
    return t


def add_economics(t: pd.DataFrame, a: Assumptions) -> pd.DataFrame:
    t = t.copy()
    t["expected_loss"] = a.lgd * t["exposure_bad"]
    t["revenue"] = a.net_margin * (t["exposure_approved"] - t["exposure_bad"])
    t["profit"] = t["revenue"] - t["expected_loss"]
    t["loss_rate"] = t["expected_loss"] / t["exposure_approved"]
    return t


def best_cutoff(t: pd.DataFrame) -> pd.Series:
    return t.loc[t["profit"].idxmax()]
