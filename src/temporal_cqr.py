"""Temporal CQR (E142): do the 90% bands stay calibrated when the test month is in the future? Train on everything
before the month, calibrate on 25% of the training brands, test on the month.

    .venv/bin/python src/temporal_cqr.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import extra_models as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def qm(a):
    return LGBMRegressor(objective="quantile", alpha=a, n_estimators=300, learning_rate=.05, num_leaves=31,
                         subsample=.9, colsample_bytree=.9, random_state=SEED, verbose=-1)


def main():
    raw = eu.load(ROOT)
    f, meta = em.load(ROOT)
    dates = raw.created_at.loc[f.index].reset_index(drop=True)
    month = dates.dt.to_period("M")
    age = ((raw.metrics_updated_at - raw.created_at).dt.total_seconds() / 86400).loc[f.index].reset_index(drop=True).clip(0.1, 3650)
    la = np.log10(age)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X = f[m][em.ENG].reset_index(drop=True); y = meta.y[m].reset_index(drop=True); g = meta.brand[m].reset_index(drop=True)
        mo = month[m].reset_index(drop=True); lama = la[m].reset_index(drop=True)
        if p == "twitter":
            X = X.assign(log_age=lama)
        counts = mo.value_counts().sort_index()
        months = [k for k in counts.index if counts[k] >= 150][-3:]
        rows = []
        for k in months:
            tr = (mo < k).values; te = (mo == k).values
            if tr.sum() < 800: continue
            inner = next(iter(GroupShuffleSplit(1, test_size=.25, random_state=SEED).split(X[tr], y[tr], g[tr])))
            fit_idx, cal_idx = np.where(tr)[0][inner[0]], np.where(tr)[0][inner[1]]
            lo = qm(.05).fit(X.iloc[fit_idx], y.iloc[fit_idx]); hi = qm(.95).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            E = np.maximum(lo.predict(X.iloc[cal_idx]) - y.iloc[cal_idx].values, y.iloc[cal_idx].values - hi.predict(X.iloc[cal_idx]))
            q = float(np.quantile(E, .9))
            L, H = lo.predict(X[te]) - q, hi.predict(X[te]) + q; yt = y[te].values
            rows.append({"month": str(k), "n": int(te.sum()), "coverage": round(float(((yt >= L) & (yt <= H)).mean()), 3),
                         "width": round(float(np.median(H - L)), 2)})
        out[p] = {"windows": rows, "coverage_mean": round(float(np.mean([r["coverage"] for r in rows])), 3) if rows else None}
        print(p, out[p])
    (ROOT / "data" / "processed" / "temporal_cqr_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
