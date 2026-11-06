"""FB CQR by tier (E132): does one 90% band cover small and large FB posts? Calibrate CQR on 25% of FB brands and
report coverage by follower tier and by quartile of measured views.

    .venv/bin/python src/fb_cqr_tiers.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import extra_models as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2 = ["log_likes", "log_comments"]
SEED = 42


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
    cov = np.zeros(len(fb)); wid = np.zeros(len(fb))
    for tr, te in GroupKFold(5).split(Xf, yf, gf):
        rng = np.random.default_rng(SEED)
        brands = np.array(sorted(set(gf.iloc[tr])))
        cal_b = set(rng.choice(brands, max(1, int(.25 * len(brands))), replace=False))
        cal = tr[gf.iloc[tr].isin(cal_b).values]
        lo, hi = qm(.05).fit(Xs, ys), qm(.95).fit(Xs, ys)
        E = np.maximum(lo.predict(Xf.iloc[cal]) - yf[cal], yf[cal] - hi.predict(Xf.iloc[cal]))
        q = float(np.quantile(E, .9))
        L, H = lo.predict(Xf.iloc[te]) - q, hi.predict(Xf.iloc[te]) + q
        cov[te] = (yf[te] >= L) & (yf[te] <= H); wid[te] = H - L
    fb["covered"] = cov; fb["wid"] = wid
    fb["fb_bin"] = pd.cut(fb.followers, [0, 1e3, 1e4, 1e5, 1e6, 1e9], labels=["<1k", "1k-10k", "10k-100k", "100k-1M", ">1M"], right=False)
    fb["v_bin"] = pd.qcut(fb.views, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    out = {"posts": int(len(fb)), "coverage90": round(float(cov.mean()), 3), "median_width": round(float(np.median(wid)), 2),
           "by_followers": {str(k): {"n": int(len(g)), "coverage": round(float(g.covered.mean()), 3), "width": round(float(g.wid.median()), 2)} for k, g in fb.groupby("fb_bin", observed=True)},
           "by_views": {str(k): {"n": int(len(g)), "coverage": round(float(g.covered.mean()), 3), "width": round(float(g.wid.median()), 2)} for k, g in fb.groupby("v_bin", observed=True)}}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_cqr_tiers_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
