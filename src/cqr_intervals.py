"""Conformalized quantile intervals (E118): the existing 10-90% quantile intervals under-cover (E57). CQR widens the
quantile band by a conformal term fitted on held-out brands; does the 90% interval reach 90% coverage, and at what width?

    .venv/bin/python src/cqr_intervals.py

Writes data/processed/cqr_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAT = ["tiktok", "youtube", "twitter"]
SEED = 42


def qmodel(a):
    return LGBMRegressor(objective="quantile", alpha=a, n_estimators=300, learning_rate=.05, num_leaves=31,
                         subsample=.9, colsample_bytree=.9, random_state=SEED, verbose=-1)


def main():
    f, meta = em.load(ROOT)
    out = {}
    for p in PLAT:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        cov = {"raw": [], "cqr": [], "split": []}
        widths = {"raw": [], "cqr": [], "split": []}
        for outer_tr, outer_te in GroupKFold(5).split(X, y, g):
            gtr = g.iloc[outer_tr]
            inner = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED).split(X.iloc[outer_tr], y.iloc[outer_tr], gtr)
            a, b = next(iter(inner))
            fit_idx, cal_idx = np.array(outer_tr)[a], np.array(outer_tr)[b]
            lo_m = qmodel(.05).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            hi_m = qmodel(.95).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            lo, hi = lo_m.predict(X.iloc[outer_te]), hi_m.predict(X.iloc[outer_te])
            lo_c, hi_c = lo_m.predict(X.iloc[cal_idx]), hi_m.predict(X.iloc[cal_idx])
            E = np.maximum(lo_c - y.iloc[cal_idx].values, y.iloc[cal_idx].values - hi_c)
            q = float(np.quantile(E, .9))
            yt = y.iloc[outer_te].values
            for name, l, h in [("raw", lo, hi), ("cqr", lo - q, hi + q)]:
                cov[name].append(((yt >= l) & (yt <= h)).mean()); widths[name].append(np.median(h - l))
            # split conformal around the median model
            med = qmodel(.5).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            r = np.abs(y.iloc[cal_idx].values - med.predict(X.iloc[cal_idx]))
            qs = float(np.quantile(r, .9)); mp = med.predict(X.iloc[outer_te])
            cov["split"].append(((yt >= mp - qs) & (yt <= mp + qs)).mean()); widths["split"].append(float(np.median(2 * qs)))
        out[p] = {k: {"coverage": round(float(np.mean(cov[k])), 3), "median_width": round(float(np.mean(widths[k])), 2)}
                  for k in cov}
        print(p, out[p])
    (ROOT / "data" / "processed" / "cqr_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
