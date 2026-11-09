"""Score the X model on the external X sample (E198).

Trains CatBoost on the Brand24 X rows (followers, likes, replies, reposts, flags, log age) and scores it on the 98 posts
fetched with the API: MAE in log10 views, bias, median baseline.

    .venv/bin/python src/external_x_validate.py
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
FEATS = ["log_followers", "log_likes", "log_comments", "log_shares", "na_likes", "na_comments", "log_age"]
NOW = pd.Timestamp("2026-11-08")


def internal():
    d = eu.load(ROOT)
    d = d[d.platform == "twitter"].copy()
    for c in ["author_followers", "likes", "comments", "shares", "view_count"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d.view_count > 0].copy()
    d["y"] = np.log10(d.view_count)
    d["log_followers"] = np.log10(d.author_followers.fillna(0) + 1)
    d["log_likes"] = np.log10(d.likes.fillna(0) + 1)
    d["log_comments"] = np.log10(d.comments.fillna(0) + 1)
    d["log_shares"] = np.log10(d.shares.fillna(0) + 1)
    d["na_likes"] = d.likes.isna().astype(int)
    d["na_comments"] = d.comments.isna().astype(int)
    age = (pd.to_datetime(d.metrics_updated_at, errors="coerce", utc=True).dt.tz_convert(None) - pd.to_datetime(d.created_at, errors="coerce", utc=True).dt.tz_convert(None)).dt.total_seconds() / 86400
    d["log_age"] = np.log10(age.clip(lower=1) + 1)
    return d, d.y


def external():
    e = pd.read_csv(ROOT / "data" / "processed" / "external_x_fetched.csv")
    e = e[e.view_count.notna() & (e.view_count > 0) & (e.author_followers > 0)].copy()
    e["y"] = np.log10(e.view_count)
    e["log_followers"] = np.log10(e.author_followers + 1)
    e["log_likes"] = np.log10(e.like_count.fillna(0) + 1)
    e["log_comments"] = np.log10(e.reply_count.fillna(0) + 1)
    e["log_shares"] = np.log10(e.retweet_count.fillna(0) + 1)
    e["na_likes"] = 0; e["na_comments"] = 0
    created = pd.to_datetime(e.created_at, errors="coerce", utc=True).dt.tz_convert(None)
    e["log_age"] = np.log10((NOW - created).dt.days.clip(lower=1) + 1)
    return e


def main():
    tr, y = internal()
    e = external()
    m = CatBoostRegressor(loss_function="MAE", iterations=600, random_state=42, verbose=0, allow_writing_files=False)
    m.fit(tr[FEATS].values, y.values)
    p = m.predict(e[FEATS].values)
    base = np.median(y.values)
    out = {"n_external": int(len(e)), "median_external_views": float(np.median(e.view_count)),
           "median_internal_views": float(np.median(10 ** y.values)),
           "MAE": round(float(np.abs(e.y.values - p).mean()), 3),
           "bias": round(float((e.y.values - p).mean()), 3),
           "median_baseline_MAE": round(float(np.abs(e.y.values - base).mean()), 3),
           "share_with_views": 1.0}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "external_x_validation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
