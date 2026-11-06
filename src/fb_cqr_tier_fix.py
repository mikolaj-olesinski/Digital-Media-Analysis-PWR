"""FB CQR with follower-tier widths (E144): the global band under-covers large pages (E132). Calibrate the CQR
conformity score per follower tier (pooled into three bins) and compare coverage per tier.

    .venv/bin/python src/fb_cqr_tier_fix.py
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
    fb["tier"] = pd.cut(fb.followers, [0, 1e4, 1e6, 1e9], labels=["<10k", "10k-1M", ">1M"], right=False)
    Xs, ys = f[src][C2].reset_index(drop=True), meta.y[src].reset_index(drop=True)
    Xf, yf, gf, tf = fb[C2], np.log10(fb.views.values), fb.brand, fb.tier
    cov_g = np.zeros(len(fb)); cov_t = np.zeros(len(fb)); wid = np.zeros(len(fb))
    for tr, te in GroupKFold(5).split(Xf, yf, gf):
        rng = np.random.default_rng(SEED)
        brands = np.array(sorted(set(gf.iloc[tr])))
        cal_b = set(rng.choice(brands, max(1, int(.25 * len(brands))), replace=False))
        cal = tr[gf.iloc[tr].isin(cal_b).values]
        lo, hi = qm(.05).fit(Xs, ys), qm(.95).fit(Xs, ys)
        E = np.maximum(lo.predict(Xf.iloc[cal]) - yf[cal], yf[cal] - hi.predict(Xf.iloc[cal]))
        qg = float(np.quantile(E, .9))
        qt = {t: float(np.quantile(E[(tf.iloc[cal] == t).values], .9)) if (tf.iloc[cal] == t).sum() >= 20 else qg for t in ["<10k", "10k-1M", ">1M"]}
        L, H = lo.predict(Xf.iloc[te]), hi.predict(Xf.iloc[te]); yt = yf[te]
        cov_g[te] = (yt >= L - qg) & (yt <= H + qg)
        qv = np.array([qt.get(x, qg) for x in tf.iloc[te].values], dtype=float)
        cov_t[te] = (yt >= L - qv) & (yt <= H + qv); wid[te] = (H - L) + 2 * qv
    out = {"overall": {"global": round(float(cov_g.mean()), 3), "tier": round(float(cov_t.mean()), 3)},
           "by_tier": {str(t): {"n": int(len(g)), "global": round(float(cov_g[g.index].mean()), 3), "tier": round(float(cov_t[g.index].mean()), 3)} for t, g in fb.groupby("tier", observed=True)}}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_cqr_tier_fix_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
