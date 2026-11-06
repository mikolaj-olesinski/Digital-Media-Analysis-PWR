"""Posting cadence and reach (E110): do brands that post more often get less per post? Two views: across brands
(cadence vs median views, controlling followers and platform) and within a brand (a post soon after another vs a long
gap), controlling reactions.

    .venv/bin/python src/cadence.py

Writes data/processed/cadence_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = eu.load(ROOT)
    q = d[d.platform.isin(["tiktok", "youtube", "twitter"]) & (pd.to_numeric(d.view_count, errors="coerce") > 0)].copy()
    q["created_at"] = pd.to_datetime(q.created_at, errors="coerce")
    q["log_views"] = np.log10(q.view_count)
    q["log_likes"] = np.log1p(q.likes.fillna(0)); q["log_comments"] = np.log1p(q.comments.fillna(0))
    q["log_followers"] = np.log10(q.author_followers.where(q.author_followers > 0))
    out = {}
    # across brands
    g = q.groupby(["brand", "platform"]).agg(n=("log_views", "size"), med_views=("log_views", "median"),
                                             med_followers=("log_followers", "median"),
                                             first=("created_at", "min"), last=("created_at", "max"))
    g = g[g.n >= 5].copy()
    g["cadence"] = g.n / np.maximum((g.last - g.first).dt.days / 30, 1)
    g = g.dropna(subset=["med_followers"])
    rho = spearmanr(g.cadence, g.med_views)
    # residualise median views on followers + platform dummies
    g2 = g.reset_index()
    X = pd.concat([g2.med_followers.rename("lf").reset_index(drop=True),
                   pd.get_dummies(g2.platform, drop_first=True, dtype=float)], axis=1).assign(const=1.0)
    beta = np.linalg.lstsq(X.values, g.med_views.values, rcond=None)[0]
    resid = g.med_views.values - X.values @ beta
    rho_res = spearmanr(g.cadence, resid)
    out["across_brands"] = {"groups": int(len(g)), "spearman": round(float(rho[0]), 3), "p": round(float(rho[1]), 4),
                            "spearman_after_followers_platform": round(float(rho_res[0]), 3), "p_res": round(float(rho_res[1]), 4),
                            "cadence_median_per_month": round(float(g.cadence.median()), 1)}
    # within brand: gap since previous post
    q = q.sort_values(["brand", "platform", "created_at"])
    q["gap_days"] = q.groupby(["brand", "platform"]).created_at.diff().dt.total_seconds() / 86400
    q = q.dropna(subset=["gap_days"])
    q["short"] = q.gap_days < 1
    # residualise views on reactions and followers
    Xw = pd.concat([q[["log_likes", "log_comments", "log_followers"]].fillna(0),
                    pd.get_dummies(q.platform, drop_first=True, dtype=float)], axis=1).assign(const=1.0)
    bw = np.linalg.lstsq(Xw.values, q.log_views.values, rcond=None)[0]
    q["resid"] = q.log_views.values - Xw.values @ bw
    out["within_brand"] = {"posts": int(len(q)), "short_gap_share": round(float(q.short.mean()), 3),
                           "median_resid_short": round(float(q.loc[q.short, "resid"].median()), 3),
                           "median_resid_long": round(float(q.loc[q.gap_days > 7, "resid"].median()), 3),
                           "difference": round(float(q.loc[q.short, "resid"].median() - q.loc[q.gap_days > 7, "resid"].median()), 3)}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "cadence_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
