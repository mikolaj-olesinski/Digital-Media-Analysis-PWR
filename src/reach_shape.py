"""Reach concentration and the tail (E96): how much of a platform's or a brand's reach sits in a few posts.

Descriptive but product-relevant for Brand24: if most reach comes from 1-5% of mentions, a monitoring product has to
catch the tail, and a brand total needs an interval, not a point. Also fits a Hill tail index to the top 1% of views.

    .venv/bin/python src/reach_shape.py

Writes data/processed/reach_shape_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAT = ["tiktok", "youtube", "twitter"]


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if n and x.sum() > 0 else np.nan


def shape(v):
    v = np.sort(np.asarray(v, float))[::-1]
    v = v[v > 0]
    n, tot = len(v), v.sum()
    out = {"posts": int(n), "share_top_1pct": float(v[:max(1, int(np.ceil(.01 * n)))].sum() / tot),
           "share_top_5pct": float(v[:max(1, int(np.ceil(.05 * n)))].sum() / tot),
           "share_top_10pct": float(v[:max(1, int(np.ceil(.10 * n)))].sum() / tot),
           "gini": gini(v), "median": float(np.median(v))}
    c = np.cumsum(v) / tot
    out["share_posts_for_80pct"] = float((np.searchsorted(c, .8) + 1) / n)
    k = max(5, int(np.ceil(.01 * n)))
    if n > k:
        out["hill_alpha_top1pct"] = float(1 / np.mean(np.log(v[:k] / v[k])))
    return out


def main():
    d = eu.load(ROOT)
    out = {"platforms": {}, "brands": {}}
    for p in PLAT:
        v = pd.to_numeric(d[d.platform == p].view_count, errors="coerce")
        out["platforms"][p] = shape(v.dropna().values)
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    out["platforms"]["facebook"] = shape(pd.to_numeric(fb[fb.status == "ok"].views, errors="coerce").dropna().values)

    # brands with >= 5 measured posts: how concentrated is a brand's reach?
    raw = d[d.platform.isin(PLAT)].copy()
    raw["v"] = pd.to_numeric(raw.view_count, errors="coerce")
    raw = raw[raw.v > 0]
    rows = []
    for b, q in raw.groupby("brand"):
        if len(q) < 5:
            continue
        s = shape(q.v.values)
        s["brand"] = "x"  # no brand names in the output
        s["platform"] = q.platform.mode().iat[0]
        rows.append(s)
    B = pd.DataFrame(rows)
    out["brands"] = {"n_brands": int(len(B)),
                     "share_top_1pct_median": float(B.share_top_1pct.median()),
                     "share_top_1pct_p90": float(B.share_top_1pct.quantile(.9)),
                     "share_posts_for_80pct_median": float(B.share_posts_for_80pct.median()),
                     "one_post_over_half_of_reach": float((B.share_top_1pct > .5).mean())}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "reach_shape_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
