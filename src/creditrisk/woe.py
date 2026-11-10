"""Weight-of-Evidence binning and Information Value.

Convention: WoE = ln(%good / %bad), so a positive WoE means lower risk than
average. Missing values always get their own bin. Numeric bins are merged
until the bad rate is monotonic (pool-adjacent-violators), the standard
requirement for an explainable scorecard.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MISSING = "MISSING"
OTHER = "OTHER"
SMOOTH = 0.5  # Laplace smoothing so empty cells never give log(0)


@dataclass
class Binning:
    feature: str
    kind: str  # "numeric" or "categorical"
    edges: list[float] = field(default_factory=list)  # numeric: inner cut points
    groups: dict[str, str] = field(default_factory=dict)  # categorical: raw -> group label
    table: pd.DataFrame | None = None  # per-bin counts, bad rate, WoE
    iv: float = 0.0

    # -- labelling -------------------------------------------------------
    def labels(self, x: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
            idx = np.searchsorted(np.asarray(self.edges, dtype=float), v, side="right")
            names = np.array([self._num_label(i) for i in range(len(self.edges) + 1)], dtype=object)
            out = names[idx]
            out[np.isnan(v)] = MISSING
            return pd.Series(out, index=x.index)
        s = x.astype(object).where(x.notna(), MISSING)
        return s.map(lambda c: self.groups.get(str(c), OTHER if c != MISSING else MISSING))

    def _num_label(self, i: int) -> str:
        e = self.edges
        lo = "-inf" if i == 0 else f"{e[i - 1]:.4g}"
        hi = "inf" if i == len(e) else f"{e[i]:.4g}"
        return f"[{lo}, {hi})"

    def transform(self, x: pd.Series) -> pd.Series:
        woe = self.table.set_index("bin")["woe"]
        return self.labels(x).map(woe).fillna(0.0).astype(float)


def _woe_table(labels: pd.Series, y: pd.Series, order: list[str] | None = None) -> tuple[pd.DataFrame, float]:
    df = pd.DataFrame({"bin": labels.to_numpy(), "y": y.to_numpy()})
    g = df.groupby("bin", sort=False)["y"].agg(n="size", bad="sum").reset_index()
    g["good"] = g["n"] - g["bad"]
    tot_bad, tot_good = g["bad"].sum(), g["good"].sum()
    k = len(g)
    g["dist_bad"] = (g["bad"] + SMOOTH) / (tot_bad + SMOOTH * k)
    g["dist_good"] = (g["good"] + SMOOTH) / (tot_good + SMOOTH * k)
    g["woe"] = np.log(g["dist_good"] / g["dist_bad"])
    g["iv_part"] = (g["dist_good"] - g["dist_bad"]) * g["woe"]
    g["bad_rate"] = g["bad"] / g["n"]
    g["share"] = g["n"] / g["n"].sum()
    if order is not None:
        rank = {b: i for i, b in enumerate(order)}
        g = g.sort_values("bin", key=lambda s: s.map(lambda b: rank.get(b, len(rank)))).reset_index(drop=True)
    cols = ["bin", "n", "share", "bad", "good", "bad_rate", "woe", "iv_part"]
    return g[cols], float(g["iv_part"].sum())


def _monotone_edges(v: np.ndarray, y: np.ndarray, edges: list[float]) -> list[float]:
    """Merge adjacent numeric bins until bad rates are monotonic."""
    edges = list(edges)
    while edges:
        idx = np.searchsorted(np.asarray(edges), v, side="right")
        n = np.bincount(idx, minlength=len(edges) + 1).astype(float)
        b = np.bincount(idx, weights=y, minlength=len(edges) + 1)
        rate = np.divide(b, n, out=np.zeros_like(b), where=n > 0)
        diffs = np.diff(rate)
        direction = np.sign(np.corrcoef(v, y)[0, 1]) if np.std(v) > 0 else 1.0
        bad = np.where(diffs * direction < 0)[0]
        if len(bad) == 0:
            break
        # drop the edge between the first violating pair -> merges those two bins
        edges.pop(int(bad[0]))
    return edges


def fit_numeric(x: pd.Series, y: pd.Series, max_bins: int = 10, min_share: float = 0.05) -> Binning:
    v_all = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    mask = ~np.isnan(v_all)
    v, yy = v_all[mask], y.to_numpy()[mask].astype(float)
    edges: list[float] = []
    if len(v) > 0 and np.unique(v).size > 1:
        qs = np.unique(np.quantile(v, np.linspace(0, 1, max_bins + 1)[1:-1]))
        edges = [float(q) for q in qs if v.min() < q <= v.max()]
        # remove edges that would leave a bin smaller than min_share of the data
        changed = True
        while changed and edges:
            changed = False
            idx = np.searchsorted(np.asarray(edges), v, side="right")
            share = np.bincount(idx, minlength=len(edges) + 1) / len(v_all)
            small = np.where(share < min_share)[0]
            if len(small):
                i = int(small[0])
                edges.pop(min(i, len(edges) - 1))
                changed = True
        edges = _monotone_edges(v, yy, edges)
    b = Binning(feature=x.name, kind="numeric", edges=edges)
    order = [b._num_label(i) for i in range(len(edges) + 1)] + [MISSING]
    b.table, b.iv = _woe_table(b.labels(x), y, order=order)
    return b


def fit_categorical(x: pd.Series, y: pd.Series, min_share: float = 0.01) -> Binning:
    s = x.astype(object).where(x.notna(), MISSING).astype(str)
    share = s.value_counts(normalize=True)
    groups = {c: (c if (share[c] >= min_share or c == MISSING) else OTHER) for c in share.index}
    b = Binning(feature=x.name, kind="categorical", groups=groups)
    b.table, b.iv = _woe_table(b.labels(x), y)
    b.table = b.table.sort_values("woe").reset_index(drop=True)
    return b


def fit_all(X: pd.DataFrame, y: pd.Series, categorical: list[str]) -> dict[str, Binning]:
    out = {}
    for col in X.columns:
        out[col] = fit_categorical(X[col], y) if col in categorical else fit_numeric(X[col], y)
    return out


def iv_table(binnings: dict[str, Binning]) -> pd.DataFrame:
    rows = [{"feature": f, "iv": b.iv, "n_bins": len(b.table)} for f, b in binnings.items()]
    df = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    df["strength"] = pd.cut(
        df["iv"], [-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
        labels=["unpredictive", "weak", "medium", "strong", "suspicious"],
    ).astype(str)
    return df


def woe_frame(X: pd.DataFrame, binnings: dict[str, Binning], features: list[str]) -> pd.DataFrame:
    return pd.DataFrame({f: binnings[f].transform(X[f]) for f in features}, index=X.index)
