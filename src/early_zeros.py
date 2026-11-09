"""What a post with 0 reactions at the first scan is worth (E117): at the start a post often has 0 views and 0 likes;
how many views does it get, and how much does the first check tell us?

    .venv/bin/python src/early_zeros.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def dist(v):
    v = pd.to_numeric(v, errors="coerce").dropna()
    return {"n": int(len(v)), "share_gt0": round(float((v > 0).mean()), 3),
            "share_gt100": round(float((v > 100).mean()), 3), "share_gt1k": round(float((v > 1000).mean()), 3),
            "share_gt10k": round(float((v > 10000).mean()), 3), "median": round(float(v.median()), 1),
            "p90": round(float(v.quantile(.9)), 1)}


def main():
    d = eu.load(ROOT)
    out = {"main_check": {}}
    for p in ["tiktok", "youtube", "twitter"]:
        q = d[d.platform == p]
        z = q[(q.likes.fillna(0) == 0) & (q.comments.fillna(0) == 0)]
        out["main_check"][p] = {"zero_share": round(float(len(z) / len(q)), 3), **dist(z.view_count)}
    p = ROOT / "data" / "raw" / "reach_samples_students_with_history.csv"
    if not p.exists():
        raise SystemExit(f"Missing data file: {p}\n"
                         "See docs/en/reproduce.md, section 'If a data file is missing'.")
    h = pd.read_csv(p)
    h = h[h.history_1_at.notna()].copy()
    h["created_at"] = pd.to_datetime(h.created_at, errors="coerce")
    h["age_h"] = (pd.to_datetime(h.history_1_at, errors="coerce") - h.created_at).dt.total_seconds() / 3600
    hz = h[(h.history_1_likes.fillna(0) == 0) & (h.history_1_comments.fillna(0) == 0)]
    out["first_check"] = {"rows": int(len(h)), "zero_share": round(float(len(hz) / len(h)), 3),
                          "median_age_h": round(float(h.age_h.median()), 1),
                          "zero_views_at_first_check": dist(hz.history_1_views),
                          "zero_current_views": dist(hz.current_views)}
    out["first_check"]["by_platform"] = {}
    for p in ["tiktok", "twitter"]:
        q = hz[hz.platform == p]
        if len(q):
            out["first_check"]["by_platform"][p] = {**dist(q.history_1_views), "current_median": round(float(pd.to_numeric(q.current_views, errors="coerce").median()), 1)}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "early_zeros_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
