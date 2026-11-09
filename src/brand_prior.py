"""If the author's history is unknown, can the brand's history substitute? (E169)

Temporal split (train on earlier posts, test on later ones), per platform: the base model (followers, reactions, flags,
age) against the same model plus a smoothed brand prior (the brand's mean log10 views in the training period) and an
author prior where the author is seen. Reports MAE, the paired gain with a brand bootstrap CI, and how many test posts
actually have a known brand / author.

    .venv/bin/python src/brand_prior.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ["tiktok", "youtube", "twitter"]
CUT = "2026-06-01"
SEED = 42
FEATS = ["log_followers", "log_likes", "log_comments", "log_shares", "na_likes", "na_comments", "na_shares", "log_age"]


def frame():
    d = eu.load(ROOT)
    d["created_at"] = pd.to_datetime(d.created_at, errors="coerce", utc=True).dt.tz_convert(None)
    for c in ["author_followers", "likes", "comments", "shares", "view_count"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d.view_count > 0) & d.created_at.notna()].copy()
    d["y"] = np.log10(d.view_count)
    d["log_followers"] = np.log10(d.author_followers.fillna(0) + 1)
    d["log_likes"] = np.log10(d.likes.fillna(0) + 1)
    d["log_comments"] = np.log10(d.comments.fillna(0) + 1)
    d["log_shares"] = np.log10(d.shares.fillna(0) + 1)
    d["na_likes"] = d.likes.isna().astype(int); d["na_comments"] = d.comments.isna().astype(int); d["na_shares"] = d.shares.isna().astype(int)
    d["log_age"] = np.log10((d.metrics_updated_at - d.created_at).dt.total_seconds().clip(lower=1) / 86400 + 1) if "metrics_updated_at" in d else 0
    return d


def prior(d, col, train, k=5.0):
    g = d.loc[train].groupby(col).y.agg(["sum", "count"])
    glob = d.loc[train].y.mean()
    val = (g["sum"] + k * glob) / (g["count"] + k)
    return d[col].map(val).fillna(glob).values, d[col].isin(val.index).values


def main():
    d = frame()
    out = {}
    for p in P:
        q = d[d.platform == p].copy()
        tr = (q.created_at < CUT).values; te = ~tr
        if tr.sum() < 200 or te.sum() < 100:
            continue
        base = CatBoostRegressor(loss_function="MAE", iterations=300, random_state=SEED, verbose=0, allow_writing_files=False)
        Xb = q[FEATS].values
        base.fit(Xb[tr], q.y[tr]); pb = base.predict(Xb[te])

        brand_vals, brand_known = prior(q, "brand", tr, k=5)
        Xp = np.c_[Xb, brand_vals]
        plus = CatBoostRegressor(loss_function="MAE", iterations=300, random_state=SEED, verbose=0, allow_writing_files=False)
        plus.fit(Xp[tr], q.y[tr]); pp = plus.predict(Xp[te])

        yte, gte = q.y[te].values, q.brand[te].values
        rng = np.random.default_rng(SEED); brands = np.unique(gte); gains = []
        for _ in range(200):
            bs = rng.choice(brands, len(brands), replace=True)
            idx = np.concatenate([np.where(gte == b)[0] for b in bs])
            gains.append(np.abs(yte[idx] - pb[idx]).mean() - np.abs(yte[idx] - pp[idx]).mean())
        out[p] = {
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "base_MAE": round(float(np.abs(yte - pb).mean()), 3),
            "plus_MAE": round(float(np.abs(yte - pp).mean()), 3),
            "gain": round(float(np.abs(yte - pb).mean() - np.abs(yte - pp).mean()), 3),
            "gain_ci": [round(float(np.quantile(gains, .025)), 3), round(float(np.quantile(gains, .975)), 3)],
            "brand_known_share": round(float(brand_known[te].mean()), 3),
        }
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "brand_prior_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
