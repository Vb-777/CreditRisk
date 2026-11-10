import numpy as np
import pandas as pd
import pytest
from scipy import stats

from creditrisk import metrics, strategy, woe
from creditrisk.scorecard import Scorecard, ScoreScale, select_features
from creditrisk.stats_tests import chi_square, two_proportion_ztest


# ---------- WoE / IV -----------------------------------------------------------------
def test_woe_matches_hand_calculation():
    x = pd.Series(["a"] * 100 + ["b"] * 100, name="x")
    y = pd.Series([1] * 10 + [0] * 90 + [1] * 30 + [0] * 70)
    b = woe.fit_categorical(x, y)
    t = b.table.set_index("bin")
    # smoothed distributions: (count + 0.5) / (total + 0.5 * k)
    dg_a, db_a = (90 + 0.5) / (160 + 1), (10 + 0.5) / (40 + 1)
    assert t.loc["a", "woe"] == pytest.approx(np.log(dg_a / db_a))
    assert t.loc["a", "woe"] > 0 > t.loc["b", "woe"]  # fewer bads -> positive WoE
    assert b.iv > 0


def test_numeric_bins_are_monotonic_and_missing_is_separate():
    rng = np.random.default_rng(1)
    v = rng.uniform(0, 1, 5000)
    v[:300] = np.nan
    y = pd.Series((rng.uniform(size=5000) < 0.3 - 0.25 * np.nan_to_num(v, nan=0.5)).astype(int))
    b = woe.fit_numeric(pd.Series(v, name="v"), y)
    t = b.table[b.table["bin"] != woe.MISSING]
    assert (np.diff(t["bad_rate"].to_numpy()) <= 1e-12).all()  # risk falls as v rises
    assert woe.MISSING in set(b.table["bin"])
    assert (t["share"] >= 0.05 - 1e-9).all()


def test_unseen_category_maps_to_neutral_woe():
    x = pd.Series(["a"] * 50 + ["b"] * 50, name="x")
    y = pd.Series([0, 1] * 50)
    b = woe.fit_categorical(x, y)
    assert b.transform(pd.Series(["zzz"])).iloc[0] == 0.0


# ---------- scorecard ----------------------------------------------------------------
def test_score_scaling_pdo():
    s = ScoreScale(base_score=600, base_odds=50, pdo=20)
    pd_at_base = 1 / 51
    assert s.score_from_pd(np.array([pd_at_base]))[0] == pytest.approx(600)
    pd_double = 1 / 101  # odds 100:1
    assert s.score_from_pd(np.array([pd_double]))[0] == pytest.approx(620)
    assert s.pd_from_score(np.array([600]))[0] == pytest.approx(pd_at_base)


def _toy_card():
    rng = np.random.default_rng(3)
    n = 6000
    X = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n), "c": rng.choice(["x", "y", "z"], n)})
    logit = -2 - 2.5 * X["a"] + 1.5 * X["b"] + X["c"].map({"x": 0.0, "y": 0.5, "z": 1.0})
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int))
    bins = woe.fit_all(X, y, categorical=["c"])
    return Scorecard.fit(X, y, bins, ["a", "b", "c"]), X, y


def test_points_sum_to_score_and_match_model_pd():
    card, X, _ = _toy_card()
    score = card.score(X)
    implied = card.scale.score_from_pd(card.predict_pd(X))
    assert np.abs(score - implied).max() < 0.05 * len(card.features) + 1e-6  # only bin-point rounding
    assert all(c < 0 for c in card.model.coef_[0])


def test_reason_codes_point_to_lost_points():
    card, X, _ = _toy_card()
    rc = card.reason_codes(X.head(50), k=2)
    comp = card.score_components(X.head(50))
    best = card.points_table().groupby("characteristic")["points"].max()
    worst = (best[comp.columns] - comp).idxmax(axis=1)
    assert (rc["reason_1"] == worst).all()


def test_select_features_drops_correlated_duplicate():
    W = pd.DataFrame({"a": np.arange(100.0), "a_copy": np.arange(100.0) * 2, "b": np.random.default_rng(0).normal(size=100)})
    iv = pd.DataFrame({"feature": ["a", "a_copy", "b"], "iv": [0.3, 0.2, 0.05]})
    assert select_features(iv, W) == ["a", "b"]


# ---------- metrics -------------------------------------------------------------------
def test_metrics_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert metrics.ks(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert metrics.gini(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    yr = rng.integers(0, 2, 20000)
    assert abs(metrics.gini(yr, rng.uniform(size=20000))) < 0.05


def test_decile_table_captures_all_bads():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=1000)
    y = (rng.uniform(size=1000) < p).astype(int)
    t = metrics.decile_table(y, p)
    assert t["cum_bad_capture"].iloc[-1] == pytest.approx(1.0)
    assert t["bad_rate"].iloc[0] > t["bad_rate"].iloc[-1]  # decile 1 is riskiest


# ---------- hypothesis tests --------------------------------------------------------
def test_ztest_squared_equals_uncorrected_chi_square():
    r = two_proportion_ztest(120, 1000, 80, 1000)
    chi2, p, _, _ = stats.chi2_contingency([[120, 880], [80, 920]], correction=False)
    assert r["z"] ** 2 == pytest.approx(chi2)
    assert r["p_value"] == pytest.approx(p)
    assert r["ci_low"] < r["diff"] < r["ci_high"]


def test_chi_square_independent_data_not_significant():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"seg": rng.choice(list("abc"), 30000), "TARGET": rng.integers(0, 2, 30000)})
    assert chi_square(df, "seg")["p_value"] > 0.01


# ---------- strategy ----------------------------------------------------------------
def test_cutoff_table_and_economics():
    score = np.array([900, 800, 700, 600, 500, 400, 300, 200, 100, 0.0])
    y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1])
    ex = np.full(10, 100.0)
    t = strategy.cutoff_table(score, y, ex, steps=np.array([0.4, 1.0]))
    assert t["approved_bads"].tolist() == [0, 4]
    e = strategy.add_economics(t, strategy.Assumptions(lgd=0.5, net_margin=0.1))
    # 40% approved: revenue 0.1*400 = 40, loss 0  | 100%: revenue 0.1*600 = 60, loss 0.5*400 = 200
    assert e["profit"].tolist() == pytest.approx([40.0, -140.0])
    assert strategy.best_cutoff(e)["approval_rate"] == pytest.approx(0.4)
