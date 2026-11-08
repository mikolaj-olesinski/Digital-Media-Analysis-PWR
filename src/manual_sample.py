"""Score a manual Instagram/TikTok sample (tool for the manual protocol).

Reads a CSV with the contract from docs/en/manual_sample.md. For platforms with an in-platform model (TikTok, YouTube, X,
Facebook) it trains on the Brand24 rows and reports MAE/bias; for Instagram, where no in-platform views exist, it reports
descriptives and the views-per-follower / views-per-reaction ratios instead of pretending to have a model.

    .venv/bin/python src/manual_sample.py data/processed/manual_tiktok.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["log_followers", "log_likes", "log_comments", "na_likes", "na_comments", "log_age"]
TRAIN = {"tiktok": "tiktok", "youtube": "youtube", "twitter": "twitter", "facebook": "facebook"}


def internal(platform):
    d = eu.load(ROOT)
    d = d[d.platform == platform].copy()
    for c in ["author_followers", "likes", "comments", "view_count"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d.view_count > 0].copy()
    d["y"] = np.log10(d.view_count)
    d["log_followers"] = np.log10(d.author_followers.fillna(0) + 1)
    d["log_likes"] = np.log10(d.likes.fillna(0) + 1)
    d["log_comments"] = np.log10(d.comments.fillna(0) + 1)
    d["na_likes"] = d.likes.isna().astype(int)
    d["na_comments"] = d.comments.isna().astype(int)
    age = (pd.to_datetime(d.metrics_updated_at, errors="coerce", utc=True).dt.tz_convert(None) - pd.to_datetime(d.created_at, errors="coerce", utc=True).dt.tz_convert(None)).dt.total_seconds() / 86400
    d["log_age"] = np.log10(age.clip(lower=1) + 1)
    return d, d.y


def external(path):
    e = pd.read_csv(path)
    for c in ["view_count", "likes", "comments", "author_followers"]:
        e[c] = pd.to_numeric(e.get(c), errors="coerce")
    e = e[(e.view_count > 0) & (e.author_followers > 0)].copy()
    e["y"] = np.log10(e.view_count)
    e["log_followers"] = np.log10(e.author_followers + 1)
    e["log_likes"] = np.log10(e.likes.fillna(0) + 1)
    e["log_comments"] = np.log10(e.comments.fillna(0) + 1)
    e["na_likes"] = e.likes.isna().astype(int)
    e["na_comments"] = e.comments.isna().astype(int)
    posted = pd.to_datetime(e.posted_at, errors="coerce")
    checked = pd.to_datetime(e.checked_at, errors="coerce")
    e["log_age"] = np.log10((checked - posted).dt.days.clip(lower=1) + 1).fillna(0)
    return e


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python src/manual_sample.py <csv>")
    e = external(sys.argv[1])
    out = {}
    for plat, g in e.groupby("platform"):
        base = {"n": int(len(g)), "median_views": float(np.median(g.view_count)) if len(g) else None}
        if len(g) >= 3 and g.likes.notna().sum() >= 3:
            base["spearman_views_likes"] = round(float(spearmanr(np.log10(g.view_count), np.log10(g.likes.fillna(0) + 1)).statistic), 3)
        if plat in TRAIN:
            tr, y = internal(TRAIN[plat])
            m = CatBoostRegressor(loss_function="MAE", iterations=600, random_state=42, verbose=0, allow_writing_files=False)
            m.fit(tr[FEATS].values, y.values)
            g = g.dropna(subset=["log_followers"])
            if len(g):
                p = m.predict(g[FEATS].values)
                base.update({"MAE": round(float(np.abs(g.y.values - p).mean()), 3), "bias": round(float((g.y.values - p).mean()), 3)})
        elif (g.author_followers > 0).any():
            base["median_views_per_follower"] = float(np.median(g.view_count / g.author_followers.replace(0, np.nan)))
        out[plat] = base
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
