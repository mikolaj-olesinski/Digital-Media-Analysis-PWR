"""Brand24's Facebook reach against measured views (E111): the FB rule was only scored on the 130 hand-check posts
(E93 baseline). Here it is scored on all 402 public FB Reels/videos, by follower tier, by view tier and by kind.

    .venv/bin/python src/b24_fb.py

Writes data/processed/b24_fb_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2 = ["log_likes", "log_comments"]


def summarise(q, by):
    rows = []
    for k, g in q.groupby(by, observed=True):
        if len(g) < 20:
            continue
        e = g.log_ratio
        rows.append({by: str(k), "posts": int(len(g)), "median_ratio": round(float(e.median()), 2),
                     "MAE": round(float(e.abs().mean()), 2), "share_off_2x": round(float((e.abs() > np.log10(2)).mean()), 2)})
    return rows


def main():
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    fb["views"] = pd.to_numeric(fb.views, errors="coerce")
    fb = fb[(fb.status == "ok") & (fb.views > 0)].copy()
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    r = raw.loc[fb.mention_id]
    fb["b24"] = pd.to_numeric(r.b24_estimated_reach.values, errors="coerce")
    fb["followers"] = pd.to_numeric(r.author_followers.values, errors="coerce")
    fb["kind"] = r.url.str.extract(r"/(reels?|videos?)/", expand=False).values
    fb = fb[fb.b24 > 0].copy()
    fb["log_ratio"] = np.log10(fb.b24 / fb.views)
    fb["followers_bin"] = pd.cut(fb.followers, [0, 1e3, 1e4, 1e5, 1e6, 1e9], labels=["<1k", "1k-10k", "10k-100k", "100k-1M", ">1M"], right=False)
    fb["views_bin"] = pd.qcut(fb.views, 4, labels=["Q1 small", "Q2", "Q3", "Q4 large"])
    # model for context: X + YouTube ridge
    d = eu.load(ROOT)
    tr = d[d.platform.isin(["twitter", "youtube"]) & (pd.to_numeric(d.view_count, errors="coerce") > 0)].copy()
    ytr = np.log10(tr.view_count)
    Xtr = pd.DataFrame({"log_likes": np.log1p(tr.likes.fillna(0)), "log_comments": np.log1p(tr.comments.fillna(0))})
    mod = make_pipeline(StandardScaler(), Ridge()).fit(Xtr, ytr)
    Xfb = pd.DataFrame({"log_likes": np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0)),
                        "log_comments": np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))})
    e_model = np.log10(fb.views.values) - mod.predict(Xfb)
    out = {"posts": int(len(fb)),
           "rule": {"median_ratio": round(float(fb.log_ratio.median()), 2), "MAE": round(float(fb.log_ratio.abs().mean()), 2),
                    "share_off_2x": round(float((fb.log_ratio.abs() > np.log10(2)).mean()), 2)},
           "model": {"MAE": round(float(np.abs(e_model).mean()), 2), "bias": round(float(e_model.mean()), 2)},
           "by_followers": summarise(fb, "followers_bin"), "by_views": summarise(fb, "views_bin"),
           "by_kind": summarise(fb, "kind")}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "b24_fb_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
