"""A 50% CQR band (E137): the 90% band is very wide; a "likely range" at 50% may be more useful for a client. Coverage
and width of the 50% and 90% CQR bands on the same folds.

    .venv/bin/python src/cqr_50.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def qm(a):
    return LGBMRegressor(objective="quantile", alpha=a, n_estimators=300, learning_rate=.05, num_leaves=31,
                         subsample=.9, colsample_bytree=.9, random_state=SEED, verbose=-1)


def main():
    f, meta = em.load(ROOT)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        res = {50: {"cov": [], "w": []}, 90: {"cov": [], "w": []}}
        for tr, te in GroupKFold(5).split(X, y, g):
            inner = next(iter(GroupShuffleSplit(1, test_size=.25, random_state=SEED).split(X.iloc[tr], y.iloc[tr], g.iloc[tr])))
            fit_idx, cal_idx = np.array(tr)[inner[0]], np.array(tr)[inner[1]]
            lo = qm(.05).fit(X.iloc[fit_idx], y.iloc[fit_idx]); hi = qm(.95).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            E = np.maximum(lo.predict(X.iloc[cal_idx]) - y.iloc[cal_idx].values, y.iloc[cal_idx].values - hi.predict(X.iloc[cal_idx]))
            L, H = lo.predict(X.iloc[te]), hi.predict(X.iloc[te]); yt = y.iloc[te].values
            for lvl in (50, 90):
                q = float(np.quantile(E, lvl / 100))
                res[lvl]["cov"].append(((yt >= L - q) & (yt <= H + q)).mean()); res[lvl]["w"].append(np.median(H - L + 2 * q))
        out[p] = {f"{lvl}%": {"coverage": round(float(np.mean(res[lvl]["cov"])), 3), "median_width": round(float(np.mean(res[lvl]["w"])), 2)} for lvl in (50, 90)}
        print(p, out[p])
    (ROOT / "data" / "processed" / "cqr_50_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
