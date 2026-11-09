"""FB tier calibration with pooled splits (E145): E144 could not calibrate the >1M tier (fewer than 20 calibration posts
per fold). With 20 random 75/25 brand splits the calibration set is larger; do per-tier widths now fix the coverage?

    .venv/bin/python src/fb_cqr_pooled.py
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
    lo, hi = qm(.05).fit(Xs, ys), qm(.95).fit(Xs, ys)
    L, H = lo.predict(Xf), hi.predict(Xf)
    rng = np.random.default_rng(SEED)
    cov_g = {t: [] for t in ["<10k", "10k-1M", ">1M"]}; cov_t = {t: [] for t in cov_g}
    for _ in range(REPS):
        brands = np.array(sorted(set(gf)))
        cal_b = set(rng.choice(brands, max(1, int(.25 * len(brands))), replace=False))
        cal = gf.isin(cal_b).values; te = ~cal
        E = np.maximum(L[cal] - yf[cal], yf[cal] - H[cal])
        qg = float(np.quantile(E, .9))
        qt = {t: float(np.quantile(E[(tf[cal] == t).values], .9)) if (tf[cal] == t).sum() >= 20 else qg for t in cov_g}
        for t in cov_g:
            k = te & (tf == t).values
            if k.sum() == 0: continue
            cov_g[t].append(((yf[k] >= L[k] - qg) & (yf[k] <= H[k] + qg)).mean())
            cov_t[t].append(((yf[k] >= L[k] - qt[t]) & (yf[k] <= H[k] + qt[t])).mean())
    out = {t: {"n": int((tf == t).sum()), "global": round(float(np.mean(cov_g[t])), 3), "tier": round(float(np.mean(cov_t[t])), 3)} for t in cov_g}
    out["reps"] = REPS
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_cqr_pooled_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
