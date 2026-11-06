"""Final model v3 for X (E141): add log(age) to the final features and recompute the point MAE and the CQR bands.
Does the E139 gain survive the interval calibration?

    .venv/bin/python src/final_v3_x.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import extra_models as em  # noqa: E402
import reaction_tiers as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def qm(a):
    return LGBMRegressor(objective="quantile", alpha=a, n_estimators=300, learning_rate=.05, num_leaves=31,
                         subsample=.9, colsample_bytree=.9, random_state=SEED, verbose=-1)


def main():
    raw = eu.load(ROOT)
    f, meta = em.load(ROOT)
    age = ((raw.metrics_updated_at - raw.created_at).dt.total_seconds() / 86400).loc[f.index].reset_index(drop=True).clip(0.1, 3650)
    la = np.log10(age)
    m = (meta.platform == "twitter").values
    X = f[m][em.ENG].reset_index(drop=True).assign(log_age=la[m].reset_index(drop=True))
    y, g = meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
    out = {"features": list(X.columns), "coverage": {}, "width": {}}
    for lvl in (50, 90):
        cov, wid = [], []
        for tr, te in GroupKFold(5).split(X, y, g):
            inner = next(iter(GroupShuffleSplit(1, test_size=.25, random_state=SEED).split(X.iloc[tr], y.iloc[tr], g.iloc[tr])))
            fit_idx, cal_idx = np.array(tr)[inner[0]], np.array(tr)[inner[1]]
            lo = qm(.05).fit(X.iloc[fit_idx], y.iloc[fit_idx]); hi = qm(.95).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            E = np.maximum(lo.predict(X.iloc[cal_idx]) - y.iloc[cal_idx].values, y.iloc[cal_idx].values - hi.predict(X.iloc[cal_idx]))
            q = float(np.quantile(E, lvl / 100))
            L, H = lo.predict(X.iloc[te]) - q, hi.predict(X.iloc[te]) + q; yt = y.iloc[te].values
            cov.append(((yt >= L) & (yt <= H)).mean()); wid.append(np.median(H - L))
        out["coverage"][f"{lvl}%"] = round(float(np.mean(cov)), 3); out["width"][f"{lvl}%"] = round(float(np.mean(wid)), 2)
    # point MAE
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        pred[te] = rt.FINAL().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    out["MAE"] = round(float(np.abs(y.values - pred).mean()), 3)
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "final_v3_x_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
