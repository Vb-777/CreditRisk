"""Discrimination and calibration metrics used in credit risk."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def gini(y, p) -> float:
    return 2.0 * roc_auc_score(y, p) - 1.0


def ks(y, p) -> float:
    """Kolmogorov-Smirnov: max gap between cumulative bad and good distributions."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Population-weighted |mean predicted - observed| over predicted-PD deciles."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return float((g["n"] * (g["p"] - g["y"]).abs()).sum() / g["n"].sum())


def summary(y, p) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, p)),
        "gini": float(gini(y, p)),
        "ks": float(ks(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "mean_pd": float(np.mean(p)),
        "observed_default_rate": float(np.mean(y)),
    }


def decile_table(y, p) -> pd.DataFrame:
    """Risk deciles (1 = riskiest) with bad rate, lift and cumulative capture."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["decile"] = 10 - pd.qcut(df["p"].rank(method="first"), 10, labels=False)
    g = df.groupby("decile").agg(n=("y", "size"), bads=("y", "sum"), mean_pd=("p", "mean")).reset_index()
    g["bad_rate"] = g["bads"] / g["n"]
    g["lift"] = g["bad_rate"] / df["y"].mean()
    g["cum_bad_capture"] = g["bads"].cumsum() / g["bads"].sum()
    return g


def bootstrap_auc_diff(y, p_a, p_b, n_boot: int = 300, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap of AUC(b) - AUC(a) on the same test rows."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
    d = np.asarray(diffs)
    return {
        "auc_diff": float(roc_auc_score(y, p_b) - roc_auc_score(y, p_a)),
        "ci_low": float(np.quantile(d, 0.025)),
        "ci_high": float(np.quantile(d, 0.975)),
        "share_boot_b_better": float((d > 0).mean()),
        "n_boot": int(len(d)),
    }
