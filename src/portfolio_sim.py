"""Brand portfolio simulation (E107): on X, where Brand24's reach is a rule that is x9 too high (E99), what would
change for a client's brand totals if the reach were replaced by a views model? Brand totals for brands with >= 5 posts,
the rule, the model (ridge on likes + comments, out-of-fold by brand) and the measured views.

    .venv/bin/python src/portfolio_sim.py

Writes data/processed/portfolio_sim_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2 = ["log_likes", "log_comments"]


def ridge():
    return make_pipeline(StandardScaler(), Ridge())


def table(q, label):
    g = q.groupby("brand").agg(n=("y", "size"), true=("view_count", "sum"),
                               b24=("b24_estimated_reach", "sum"), model=("pred", lambda s: float(np.sum(10 ** s))))
    g = g[g.n >= 5]
    if len(g) < 5:
        return {"brands": int(len(g)), "note": "too few brands"}
    r_b24 = np.log10(g.b24 / g.true)
    r_m = np.log10(g.model / g.true)
    return {"brands": int(len(g)),
            "median_ratio_b24": round(float(r_b24.median()), 2), "median_ratio_model": round(float(r_m.median()), 2),
            "share_off_2x_b24": round(float((r_b24.abs() > np.log10(2)).mean()), 2),
            "share_off_2x_model": round(float((r_m.abs() > np.log10(2)).mean()), 2),
            "median_abs_log_err_b24": round(float(r_b24.abs().median()), 2),
            "median_abs_log_err_model": round(float(r_m.abs().median()), 2),
            "kendall_b24": round(float(kendalltau(g.true, g.b24)[0]), 3),
            "kendall_model": round(float(kendalltau(g.true, g.model)[0]), 3),
            "top5_overlap_b24": int(len(set(g.true.nlargest(5).index) & set(g.b24.nlargest(5).index))),
            "top5_overlap_model": int(len(set(g.true.nlargest(5).index) & set(g.model.nlargest(5).index)))}


def main():
    d = eu.load(ROOT)
    out = {}
    for p in ["twitter", "tiktok", "youtube"]:
        q = d[(d.platform == p) & (pd.to_numeric(d.view_count, errors="coerce") > 0)].copy()
        q["y"] = np.log10(q.view_count)
        q["log_likes"] = np.log1p(q.likes.fillna(0))
        q["log_comments"] = np.log1p(q.comments.fillna(0))
        q["pred"] = va.grouped_cv_predict(ridge, q[C2], q.y, q.brand)
        out[p] = table(q, p)
        print(p, out[p])
    # Facebook from the fetched posts (few brands, but the only measured FB totals)
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    fb = fb[(fb.status == "ok") & (pd.to_numeric(fb.views, errors="coerce") > 0)].copy()
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    r = raw.loc[fb.mention_id]
    fb["brand"] = r.brand.values
    fb["y"] = np.log10(fb.views)
    fb["log_likes"] = np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0))
    fb["log_comments"] = np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))
    # model trained on X + YouTube, applied to FB
    tr = d[d.platform.isin(["twitter", "youtube"]) & (pd.to_numeric(d.view_count, errors="coerce") > 0)].copy()
    tr["y"] = np.log10(tr.view_count)
    mod = ridge().fit(pd.DataFrame({"log_likes": np.log1p(tr.likes.fillna(0)), "log_comments": np.log1p(tr.comments.fillna(0))}), tr.y)
    fb["pred"] = mod.predict(fb[C2].fillna(0))
    fb["b24_estimated_reach"] = np.nan  # not available for the fetched posts
    out["facebook"] = {"brands": int(fb.groupby("brand").size().ge(5).sum()),
                       "note": "no Brand24 reach in the fetched file; model vs views only"}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "portfolio_sim_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
