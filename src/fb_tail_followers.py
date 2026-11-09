"""FB tail, two checks (E148/E149): (a) excluding pages above 1M followers, is the band calibrated? (b) does adding
followers to the source ridge help the >1M pages?

    .venv/bin/python src/fb_tail_followers.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    fb["log_followers"] = np.log10(fb.followers.fillna(1).clip(lower=1))
    fb["log_likes"] = np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0))
    fb["log_comments"] = np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))
    big = (fb.followers >= 1e6).values
    Xs = pd.DataFrame({"log_likes": f[src].log_likes.fillna(0).values, "log_comments": f[src].log_comments.fillna(0).values,
                       "log_followers": np.log10(meta.author_followers[src].fillna(1).clip(lower=1).values)})
    ys = meta.y[src].reset_index(drop=True)
    Xf, yf, gf = fb[C2 + ["log_followers"]], np.log10(fb.views.values), fb.brand
    # point models
    out = {}
    for name, cols in [("likes+comments", C2), ("+followers", C2 + ["log_followers"])]:
        # the target is FB, the model never sees FB: fit on all source rows, evaluate on all FB posts
        mod = make_pipeline(StandardScaler(), Ridge()).fit(Xs[cols], ys)
        pred = mod.predict(Xf[cols])
        out[name] = {"MAE_big": round(float(np.abs(yf[big] - pred[big]).mean()), 3),
                     "MAE_small": round(float(np.abs(yf[~big] - pred[~big]).mean()), 3),
                     "bias_big": round(float((yf[big] - pred[big]).mean()), 3)}
    # coverage excluding >1M (global CQR, 5-fold)
    lo, hi = qm(.05), qm(.95)
    lo.fit(Xs[C2], ys); hi.fit(Xs[C2], ys)
    L, H = lo.predict(Xf[C2]), hi.predict(Xf[C2])
    cov_small, wid_small = [], []
    for tr, te in GroupKFold(5).split(Xf, yf, gf):
        rng = np.random.default_rng(SEED)
        brands = np.array(sorted(set(gf.iloc[tr])))
        cal_b = set(rng.choice(brands, max(1, int(.25 * len(brands))), replace=False))
        cal = tr[gf.iloc[tr].isin(cal_b).values]
        E = np.maximum(L[cal] - yf[cal], yf[cal] - H[cal]); q = float(np.quantile(E, .9))
        k = te[~big[te]]
        if k.sum():
            cov_small.append(((yf[k] >= L[k] - q) & (yf[k] <= H[k] + q)).mean()); wid_small.append(np.median((H[k] - L[k]) + 2 * q))
    out["coverage_excl_1M"] = round(float(np.mean(cov_small)), 3)
    out["width_excl_1M"] = round(float(np.mean(wid_small)), 2)
    out["n_big"] = int(big.sum())
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_tail_followers_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
