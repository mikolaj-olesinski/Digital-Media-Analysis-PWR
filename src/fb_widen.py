"""Widening the FB band for the largest pages (E146): the >1M tier is under-covered (0.83-0.86). Test a fixed widening
factor for that tier: coverage and width cost.

    .venv/bin/python src/fb_widen.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import extra_models as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2 = ["log_likes", "log_comments"]
SEED = 42
REPS = 20
FACTORS = [1.0, 1.3, 1.5, 2.0]


def qm(a):
    return LGBMRegressor(objective="quantile", alpha=a, n_estimators=300, learning_rate=.05, num_leaves=31,
                         subsample=.9, colsample_bytree=.9, random_state=SEED, verbose=-1)


def main():
    f, meta = em.load(ROOT)
    src = meta.platform.isin(["twitter", "youtube"]).values
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    fb["views"] = pd.to_numeric(fb.views, errors="coerce")
    fb = fb[(fb.status == "ok") & (fb.views > 0)].reset_index(drop=True)
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    fb["brand"] = raw.loc[fb.mention_id, "brand"].values
    fb["followers"] = pd.to_numeric(raw.loc[fb.mention_id, "author_followers"].values, errors="coerce")
    fb["log_likes"] = np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0))
    fb["log_comments"] = np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))
    Xs, ys = f[src][C2].reset_index(drop=True), meta.y[src].reset_index(drop=True)
    Xf, yf, gf = fb[C2], np.log10(fb.views.values), fb.brand
    big = (fb.followers >= 1e6).values
    lo, hi = qm(.05).fit(Xs, ys), qm(.95).fit(Xs, ys)
    L, H = lo.predict(Xf), hi.predict(Xf)
    rng = np.random.default_rng(SEED)
    out = {f: {"big_cov": [], "big_w": [], "all_cov": []} for f in FACTORS}
    for _ in range(REPS):
        brands = np.array(sorted(set(gf)))
        cal_b = set(rng.choice(brands, max(1, int(.25 * len(brands))), replace=False))
        cal = gf.isin(cal_b).values; te = ~cal
        E = np.maximum(L[cal] - yf[cal], yf[cal] - H[cal]); qg = float(np.quantile(E, .9))
        for fac in FACTORS:
            q = np.where(big, qg * fac, qg)
            cov = (yf[te] >= L[te] - q[te]) & (yf[te] <= H[te] + q[te])
            out[fac]["all_cov"].append(cov.mean())
            k = big[te]
            out[fac]["big_cov"].append(cov[k].mean()); out[fac]["big_w"].append(np.median((H[te] - L[te] + 2 * q[te])[big[te]]))
    res = {f: {"big_coverage": round(float(np.mean(v["big_cov"])), 3), "big_width": round(float(np.mean(v["big_w"])), 2),
               "all_coverage": round(float(np.mean(v["all_cov"])), 3)} for f, v in out.items()}
    print(json.dumps(res, indent=1))
    (ROOT / "data" / "processed" / "fb_widen_numbers.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
