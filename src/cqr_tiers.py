"""CQR by reaction tier (E125): a single conformal width under-covers zero-reaction posts (E34). Calibrate the CQR
conformity score per reaction tier and compare coverage and width with one global width.

    .venv/bin/python src/cqr_tiers.py
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
import reaction_tiers as rt  # noqa: E402

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
        tier = pd.cut(rt.reactions(f)[m].reset_index(drop=True), rt.EDGES, labels=rt.TIERS)
        cov_g, cov_t, wid = [], [], []
        per_tier = {t: {"n": 0, "cov_g": [], "cov_t": [], "w": []} for t in rt.TIERS}
        for tr, te in GroupKFold(5).split(X, y, g):
            inner = next(iter(GroupShuffleSplit(1, test_size=.25, random_state=SEED).split(X.iloc[tr], y.iloc[tr], g.iloc[tr])))
            fit_idx, cal_idx = np.array(tr)[inner[0]], np.array(tr)[inner[1]]
            lo = qm(.05).fit(X.iloc[fit_idx], y.iloc[fit_idx]); hi = qm(.95).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            Ec = np.maximum(lo.predict(X.iloc[cal_idx]) - y.iloc[cal_idx].values, y.iloc[cal_idx].values - hi.predict(X.iloc[cal_idx]))
            qg = float(np.quantile(Ec, .9))
            qt = {t: float(np.quantile(Ec[(tier.iloc[cal_idx] == t).values], .9)) if (tier.iloc[cal_idx] == t).sum() >= 30 else qg for t in rt.TIERS}
            L, H = lo.predict(X.iloc[te]), hi.predict(X.iloc[te]); yt = y.iloc[te].values
            tt = tier.iloc[te].values
            cov_g.append(((yt >= L - qg) & (yt <= H + qg)).mean()); cov_t.append(((yt >= L - np.array([qt[x] for x in tt])) & (yt <= H + np.array([qt[x] for x in tt]))).mean())
            wid.append(np.median((H + qg) - (L - qg)))
            for t in rt.TIERS:
                k = tt == t
                if k.sum() == 0: continue
                per_tier[t]["n"] += int(k.sum()); per_tier[t]["cov_g"].append(((yt[k] >= L[k] - qg) & (yt[k] <= H[k] + qg)).mean())
                per_tier[t]["cov_t"].append(((yt[k] >= L[k] - qt[t]) & (yt[k] <= H[k] + qt[t])).mean())
        out[p] = {"coverage_global": round(float(np.mean(cov_g)), 3), "coverage_by_tier": round(float(np.mean(cov_t)), 3),
                  "median_width": round(float(np.mean(wid)), 2),
                  "tiers": {t: {"n": v["n"], "cov_global": round(float(np.mean(v["cov_g"])), 3) if v["cov_g"] else None,
                                "cov_tier": round(float(np.mean(v["cov_t"])), 3) if v["cov_t"] else None} for t, v in per_tier.items()}}
        print(p, json.dumps(out[p], indent=1))
    (ROOT / "data" / "processed" / "cqr_tiers_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
