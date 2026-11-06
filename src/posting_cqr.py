"""Intervals at posting time (E130): before any reactions exist, the model sees followers, text and labels only. What
is the MAE and the CQR 90% band then?

    .venv/bin/python src/posting_cqr.py
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
SEED = 42
FEAT = ["log_followers", "na_followers", "english", "text_len", "type_code", "sentiment_code"]


def qm(a):
    return LGBMRegressor(objective="quantile", alpha=a, n_estimators=300, learning_rate=.05, num_leaves=31,
                         subsample=.9, colsample_bytree=.9, random_state=SEED, verbose=-1)


def main():
    f, meta = em.load(ROOT)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][FEAT].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        cov, wid, mae = [], [], []
        for tr, te in GroupKFold(5).split(X, y, g):
            inner = next(iter(GroupShuffleSplit(1, test_size=.25, random_state=SEED).split(X.iloc[tr], y.iloc[tr], g.iloc[tr])))
            fit_idx, cal_idx = np.array(tr)[inner[0]], np.array(tr)[inner[1]]
            lo = qm(.05).fit(X.iloc[fit_idx], y.iloc[fit_idx]); hi = qm(.95).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            mv = qm(.5).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            E = np.maximum(lo.predict(X.iloc[cal_idx]) - y.iloc[cal_idx].values, y.iloc[cal_idx].values - hi.predict(X.iloc[cal_idx]))
            q = float(np.quantile(E, .9))
            L, H = lo.predict(X.iloc[te]) - q, hi.predict(X.iloc[te]) + q
            yt = y.iloc[te].values
            cov.append(((yt >= L) & (yt <= H)).mean()); wid.append(np.median(H - L)); mae.append(np.abs(yt - mv.predict(X.iloc[te])).mean())
        out[p] = {"posts": int(len(y)), "MAE_posting": round(float(np.mean(mae)), 3),
                  "coverage90": round(float(np.mean(cov)), 3), "median_width": round(float(np.mean(wid)), 2)}
        print(p, out[p])
    (ROOT / "data" / "processed" / "posting_cqr_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
