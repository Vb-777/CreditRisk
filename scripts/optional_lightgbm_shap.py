"""Optional: LightGBM challenger + SHAP explanations.

Not part of the tested pipeline (LightGBM and SHAP are not in requirements.txt).
Run after the main pipeline has built data/homecredit.db:

    pip install lightgbm shap
    PYTHONPATH=src python scripts/optional_lightgbm_shap.py
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from creditrisk import db, metrics  # noqa: E402
from creditrisk.pipeline import ID, TARGET, split  # noqa: E402

OUT = Path("reports/optional")
OUT.mkdir(parents=True, exist_ok=True)

con = db.connect("data/homecredit.db")
df = db.build_features(con)
features = [c for c in df.columns if c not in (ID, TARGET)]
cat = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
for c in cat:
    df[c] = df[c].astype("category")
tr, va, te = split(df)

model = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                           random_state=42, verbose=-1)
model.fit(tr[features], tr[TARGET], eval_set=[(va[features], va[TARGET])], eval_metric="auc",
          callbacks=[lgb.early_stopping(100, verbose=False)])
p = model.predict_proba(te[features])[:, 1]
print("LightGBM test:", {k: round(v, 4) for k, v in metrics.summary(te[TARGET], p).items()})

sample = te[features].sample(5000, random_state=0)
sv = shap.TreeExplainer(model).shap_values(sample)
sv = sv[1] if isinstance(sv, list) else sv
shap.summary_plot(sv, sample, show=False, max_display=15)
plt.tight_layout()
plt.savefig(OUT / "shap_summary.png", dpi=200)
print("wrote", OUT / "shap_summary.png")
