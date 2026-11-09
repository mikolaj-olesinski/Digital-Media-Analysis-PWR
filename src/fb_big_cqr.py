"""CQR for the >1M FB pages with a large-page model (E151): quantile models trained on source pages above 1M followers
(with followers), calibrated on FB large pages. Does the coverage move above 0.84?

    .venv/bin/python src/fb_big_cqr.py
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
SEED = 42
REPS = 20
COLS = ["log_likes", "log_comments", "log_followers"]


def qm(a):
    return LGBMRegressor(objective="quantile", alpha=a, n_estimators=300, learning_rate=.05, num_leaves=15,
                         subsample=.9, colsample_bytree=.9, random_state=SEED, verbose=-1)


def main():
    f, meta = em.load(ROOT)
    src = meta.platform.isin(["twitter", "youtube"]).values
    big_src = src & (meta.author_followers.fillna(0) >= 1e6).values
    Xs = pd.DataFrame({"log_likes": f[big_src].log_likes.fillna(0).values, "log_comments": f[big_src].log_comments.fillna(0).values,
                       "log_followers": np.log10(meta.author_followers[big_src].fillna(1).clip(lower=1).values)})
    ys = meta.y[big_src].reset_index(drop=True)
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    fb["views"] = pd.to_numeric(fb.views, errors="coerce")
    fb = fb[(fb.status == "ok") & (fb.views > 0)].reset_index(drop=True)
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    fb["brand"] = raw.loc[fb.mention_id, "brand"].values
    fb["followers"] = pd.to_numeric(raw.loc[fb.mention_id, "author_followers"].values, errors="coerce")
    fb["log_likes"] = np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0))
    fb["log_comments"] = np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))
    fb["log_followers"] = np.log10(fb.followers.fillna(1).clip(lower=1))
    big = (fb.followers >= 1e6).values
    B = fb[big].reset_index(drop=True); yb = np.log10(B.views.values); gb = B.brand
    lo, hi, med = qm(.05).fit(Xs, ys), qm(.95).fit(Xs, ys), qm(.5).fit(Xs, ys)
    L, H, M = lo.predict(B[COLS]), hi.predict(B[COLS]), med.predict(B[COLS])
    rng = np.random.default_rng(SEED)
    cov, wid, mae = [], [], []
    for _ in range(REPS):
        brands = np.array(sorted(set(gb)))
        cal_b = set(rng.choice(brands, max(1, int(.25 * len(brands))), replace=False))
        cal = gb.isin(cal_b).values; te = ~cal
        if cal.sum() < 5 or te.sum() < 5: continue
        E = np.maximum(L[cal] - yb[cal], yb[cal] - H[cal]); q = float(np.quantile(E, .9))
        cov.append(((yb[te] >= L[te] - q) & (yb[te] <= H[te] + q)).mean()); wid.append(np.median((H[te] - L[te]) + 2 * q))
        mae.append(np.abs(yb[te] - M[te]).mean())
    out = {"source_big_posts": int(big_src.sum()), "target_big_posts": int(big.sum()),
           "coverage": round(float(np.mean(cov)), 3), "width": round(float(np.mean(wid)), 2), "MAE": round(float(np.mean(mae)), 3)}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_big_cqr_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
