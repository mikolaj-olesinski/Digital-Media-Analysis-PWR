"""Reverse-engineering the visible distribution of views (E167): fit a production function per platform,
log10(views) = a * log10(followers+1) + b * log10(1 + reactions) + c, and measure the amplification beyond the
follower base (median views per follower, share of posts with more views than followers). Brand-grouped OOF R2 and a
brand bootstrap for a and b.

    .venv/bin/python src/platform_law.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ["tiktok", "youtube", "twitter"]
SEED = 42


def reactions(f):
    likes = np.expm1(f.log_likes.fillna(0).values)
    comm = np.expm1(f.log_comments.fillna(0).values)
    shar = np.expm1(f.log_shares.fillna(0).values)
    return np.log10(1 + likes + comm + shar)


def main():
    f, meta = em.load(ROOT)
    X = np.c_[f.log_followers.fillna(0).values, reactions(f)]
    out = {}
    for p in P:
        m = (meta.platform == p).values
        Xp, yp, gp = X[m], meta.y[m].values, meta.brand[m].values
        oof = np.zeros_like(yp)
        for tr, te in GroupKFold(5).split(Xp, yp, gp):
            oof[te] = LinearRegression().fit(Xp[tr], yp[tr]).predict(Xp[te])
        r2 = 1 - ((yp - oof) ** 2).sum() / ((yp - yp.mean()) ** 2).sum()
        lr = LinearRegression().fit(Xp, yp)
        rng = np.random.default_rng(SEED)
        brands = np.unique(gp)
        ab = []
        for _ in range(200):
            bs = rng.choice(brands, len(brands), replace=True)
            idx = np.concatenate([np.where(gp == b)[0] for b in bs])
            ab.append(LinearRegression().fit(Xp[idx], yp[idx]).coef_)
        ab = np.array(ab)
        vpf = (10 ** meta.y[m].values) / meta.author_followers[m].values
        vpf = vpf[np.isfinite(vpf) & (meta.author_followers[m].values > 0)]
        out[p] = {
            "n": int(m.sum()),
            "follower_elasticity": round(float(lr.coef_[0]), 3),
            "follower_ci": [round(float(np.quantile(ab[:, 0], .025)), 3), round(float(np.quantile(ab[:, 0], .975)), 3)],
            "reaction_elasticity": round(float(lr.coef_[1]), 3),
            "reaction_ci": [round(float(np.quantile(ab[:, 1], .025)), 3), round(float(np.quantile(ab[:, 1], .975)), 3)],
            "oof_r2": round(float(r2), 3),
            "amplification_median": round(float(np.median(vpf)), 2),
            "amplification_p90": round(float(np.quantile(vpf, .9)), 2),
            "share_above_followers": round(float((vpf > 1).mean()), 3),
        }
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "platform_law_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
