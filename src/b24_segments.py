"""Where Brand24's reach rule fails (E99): the rule's error by follower tier, reaction tier, age of the check, language
and platform. The rule is `b24_estimated_reach`; the truth is `view_count` where the platform exposes it.

    .venv/bin/python src/b24_segments.py

Writes data/processed/b24_segments_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def bin_followers(x):
    b = [0, 1e3, 1e4, 1e5, 1e6, 1e9]
    lab = ["<1k", "1k-10k", "10k-100k", "100k-1M", ">1M"]
    return pd.cut(x, b, labels=lab, right=False)


def bin_reactions(x):
    b = [-1, 0, 10, 100, 1000, 1e9]
    lab = ["0", "1-10", "11-100", "101-1k", ">1k"]
    return pd.cut(x, b, labels=lab)


def summarise(d, by):
    rows = []
    for k, q in d.groupby(by, observed=True):
        if len(q) < 30:
            continue
        e = q.log_ratio
        rows.append({by: str(k), "posts": int(len(q)), "median_ratio": round(float(e.median()), 2),
                     "MAE_rule": round(float(e.abs().mean()), 2), "share_off_2x": round(float((e.abs() > np.log10(2)).mean()), 2)})
    return rows


def main():
    d = eu.load(ROOT)
    d = d[(d.platform.isin(["tiktok", "youtube", "twitter"])) & (pd.to_numeric(d.view_count, errors="coerce") > 0)
          & (pd.to_numeric(d.b24_estimated_reach, errors="coerce") > 0)].copy()
    d["log_ratio"] = np.log10(d.b24_estimated_reach / d.view_count)
    d["reactions"] = d.likes.fillna(0) + d.comments.fillna(0) + d.shares.fillna(0)
    d["age_days"] = (pd.to_datetime(d.metrics_updated_at, errors="coerce") - pd.to_datetime(d.created_at, errors="coerce")).dt.total_seconds() / 86400
    d["followers_bin"] = bin_followers(d.author_followers)
    d["reactions_bin"] = bin_reactions(d.reactions)
    d["age_bin"] = pd.cut(d.age_days, [-1, 1, 7, 30, 1e9], labels=["<1d", "1-7d", "7-30d", ">30d"])
    d["language_bin"] = np.where(d.language == "en", "en", "other")
    out = {"posts": int(len(d)), "overall": {"median_ratio": round(float(d.log_ratio.median()), 2),
                                            "MAE_rule": round(float(d.log_ratio.abs().mean()), 2)},
           "by_platform": summarise(d, "platform"),
           # the rule is exact on TikTok and YouTube (reach = views), so the interesting error is on X only
           "x_by_followers": summarise(d[d.platform == "twitter"], "followers_bin"),
           "x_by_reactions": summarise(d[d.platform == "twitter"], "reactions_bin"),
           "x_by_age": summarise(d[d.platform == "twitter"], "age_bin"),
           "x_by_language": summarise(d[d.platform == "twitter"], "language_bin")}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "b24_segments_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
