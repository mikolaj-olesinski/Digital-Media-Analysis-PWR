"""Model check on the external YouTube sample (E170).

Trains the YouTube model on the Brand24 rows (no author history, no text) and scores it on the independently fetched
YouTube sample: MAE in log10 views, bias, and the same for a median baseline. The point is not to beat the internal CV but
to see whether the relation survives outside the export.

    .venv/bin/python src/external_validate.py
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
FEATS = ["log_followers", "log_likes", "log_comments", "na_likes", "na_comments", "log_age"]
NOW = pd.Timestamp("2026-11-08")


def internal():
    d = eu.load(ROOT)
    d = d[d.platform == "youtube"].copy()
    for c in ["author_followers", "likes", "comments", "view_count"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d.view_count > 0)].copy()
    d["y"] = np.log10(d.view_count)
    d["log_followers"] = np.log10(d.author_followers.fillna(0) + 1)
    d["log_likes"] = np.log10(d.likes.fillna(0) + 1)
    d["log_comments"] = np.log10(d.comments.fillna(0) + 1)
    d["na_likes"] = d.likes.isna().astype(int)
    d["na_comments"] = d.comments.isna().astype(int)
    age = (pd.to_datetime(d.metrics_updated_at, errors="coerce", utc=True).dt.tz_convert(None) - pd.to_datetime(d.created_at, errors="coerce", utc=True).dt.tz_convert(None)).dt.total_seconds() / 86400
    d["log_age"] = np.log10(age.clip(lower=1) + 1)
    return d, d.y


def external():
    p = ROOT / "data" / "processed" / "external_youtube_fetched.csv"
    e = pd.read_csv(p)
    for c in ["view_count", "like_count", "comment_count", "channel_follower_count"]:
        e[c] = pd.to_numeric(e[c], errors="coerce")
    e = e[(e.view_count > 0) & (e.channel_follower_count > 0)].copy()
    e["y"] = np.log10(e.view_count)
    e["log_followers"] = np.log10(e.channel_follower_count + 1)
    e["log_likes"] = np.log10(e.like_count.fillna(0) + 1)
    e["log_comments"] = np.log10(e.comment_count.fillna(0) + 1)
    e["na_likes"] = e.like_count.isna().astype(int)
    e["na_comments"] = e.comment_count.isna().astype(int)
    up = pd.to_datetime(e.upload_date, format="%Y%m%d", errors="coerce")
    e["log_age"] = np.log10((NOW - up).dt.days.clip(lower=1) + 1)
    return e


def main():
    tr, y = internal()
    e = external()
    m = CatBoostRegressor(loss_function="MAE", iterations=600, random_state=42, verbose=0, allow_writing_files=False)
    m.fit(tr[FEATS].values, y.values)
    p = m.predict(e[FEATS].values)
    base = np.median(y.values)
    res = e.y.values - p
    corrected = np.abs(res - res.mean()).mean()
    slope, intercept = np.polyfit(p, e.y.values, 1)
    out = {
        "n_external": int(len(e)), "brands": int(e.query_brand.nunique()),
        "bias_corrected_MAE": round(float(corrected), 3),
        "calibration_slope": round(float(slope), 3),
        "calibration_intercept": round(float(intercept), 3),
        "MAE": round(float(np.abs(e.y.values - p).mean()), 3),
        "bias": round(float((e.y.values - p).mean()), 3),
        "median_baseline_MAE": round(float(np.abs(e.y.values - base).mean()), 3),
        "median_baseline_bias": round(float((e.y.values - base).mean()), 3),
        "internal_cv_reference": 0.456,
        "median_external_views": float(np.median(10 ** e.y.values)),
        "median_internal_views": float(np.median(10 ** y.values)),
    }
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "external_youtube_validation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
