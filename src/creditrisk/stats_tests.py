"""Hypothesis tests on default rates across customer segments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def two_proportion_ztest(bad_a: int, n_a: int, bad_b: int, n_b: int) -> dict[str, float]:
    """H0: default rate of A equals that of B (two-sided, pooled variance)."""
    p_a, p_b = bad_a / n_a, bad_b / n_b
    pooled = (bad_a + bad_b) / (n_a + n_b)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_a - p_b) / se
    p_value = 2 * stats.norm.sf(abs(z))
    # 95% CI for the difference uses the unpooled standard error
    se_u = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    return {"rate_a": p_a, "rate_b": p_b, "diff": p_a - p_b, "z": float(z), "p_value": float(p_value),
            "ci_low": p_a - p_b - 1.96 * se_u, "ci_high": p_a - p_b + 1.96 * se_u,
            "relative_risk": p_a / p_b}


def chi_square(df: pd.DataFrame, segment: str, target: str = "TARGET") -> dict[str, float]:
    """H0: default is independent of the segment. Reports Cramer's V as effect size."""
    tab = pd.crosstab(df[segment].fillna("MISSING"), df[target])
    chi2, p, dof, _ = stats.chi2_contingency(tab, correction=False)
    n = tab.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
    return {"chi2": float(chi2), "dof": int(dof), "p_value": float(p), "cramers_v": float(v),
            "n_segments": int(tab.shape[0])}


def run_all(df: pd.DataFrame) -> dict[str, dict]:
    thin = df[df["has_bureau_history"] == 0]["TARGET"]
    hist = df[df["has_bureau_history"] == 1]["TARGET"]
    rev = df[df["NAME_CONTRACT_TYPE"] == "Revolving loans"]["TARGET"]
    cash = df[df["NAME_CONTRACT_TYPE"] == "Cash loans"]["TARGET"]
    return {
        "thin_file_vs_bureau_history": {
            "question": "Do thin-file applicants (no bureau record) default more often?",
            **two_proportion_ztest(int(thin.sum()), len(thin), int(hist.sum()), len(hist))},
        "cash_vs_revolving": {
            "question": "Do cash loans default more often than revolving loans?",
            **two_proportion_ztest(int(cash.sum()), len(cash), int(rev.sum()), len(rev))},
        "education": {"question": "Is default independent of education level?",
                      **chi_square(df, "NAME_EDUCATION_TYPE")},
        "income_type": {"question": "Is default independent of income type?",
                        **chi_square(df, "NAME_INCOME_TYPE")},
    }
