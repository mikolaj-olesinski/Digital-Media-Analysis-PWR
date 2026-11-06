"""What to report for a 0/0 post at the first scan (E126): empirical 10-50-90% quantiles of views for posts with zero
reactions at the first check, by platform, plus the same at the last check and at collection.

    .venv/bin/python src/early_intervals.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def q(v):
    v = pd.to_numeric(v, errors="coerce").dropna()
    return {f"p{int(x*100)}": round(float(v.quantile(x)), 1) for x in (.1, .5, .9)} | {"n": int(len(v))}


def main():
    p = ROOT / "data" / "raw" / "reach_samples_students_with_history.csv"
    if not p.exists():
        raise SystemExit(f"Missing data file: {p}\n"
                         "See docs/en/reproduce.md, section 'If a data file is missing'.")
    h = pd.read_csv(p)
    h = h[h.history_1_at.notna()].copy()
    z = h[(h.history_1_likes.fillna(0) == 0) & (h.history_1_comments.fillna(0) == 0)]
    out = {"first_check": {}, "last_check": {}}
    for p in ["tiktok", "twitter"]:
        qq = z[z.platform == p]
        if len(qq):
            out["first_check"][p] = q(qq.history_1_views); out["last_check"][p] = q(qq.current_views)
    out["first_check"]["all"] = q(z.history_1_views); out["last_check"]["all"] = q(z.current_views)
    d = eu.load(ROOT); out["main_check"] = {}
    for p in ["tiktok", "youtube", "twitter"]:
        qq = d[(d.platform == p) & (d.likes.fillna(0) == 0) & (d.comments.fillna(0) == 0)]
        out["main_check"][p] = q(qq.view_count)
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "early_intervals_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
